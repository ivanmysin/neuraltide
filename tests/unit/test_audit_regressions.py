"""
Регрессионные тесты по итогам аудита ветки dev (b3c90f6).

Положить в tests/unit/ и запустить:
    pytest tests/unit/test_audit_regressions.py -v
    pytest tests/unit/test_audit_regressions.py -v -m slow   # тест покрытия UQ

Каждый тест сверяет код с НЕЗАВИСИМЫМ эталоном (аналитика, конечные
разности по другому пути, свободный прогон, статья). На текущем dev
тесты с пометкой «ловит …» должны ПАДАТЬ: это и есть проверка, что тест
видит ошибку. После исправления они должны проходить.

Файл не запускался: в среде аудита нет TensorFlow. Импорты и сигнатуры
взяты из существующих тестов ветки dev; мелкие расхождения API возможны.
"""

import numpy as np
import pytest
import tensorflow as tf

import neuraltide.config
from neuraltide.core.network import NetworkGraph, NetworkOutput, NetworkRNN
from neuraltide.integrators import EulerIntegrator, RK4Integrator
from neuraltide.optimization import (
    CalibrationProblem,
    Experiment,
    FrozenProblem,
    MultiExperimentProblem,
    ObservationSpec,
    ParameterEntry,
    ParameterSpace,
    ResidualLoss,
)
from neuraltide.optimization.identifiability import compute_fim
from neuraltide.optimization.local import least_squares
from neuraltide.optimization.problem import _search_box_phi_from_space
from neuraltide.optimization.robust import cagrad_project
from neuraltide.optimization.sensitivities import (
    residual_jacobian,
    value_and_gradient_checkpointed,
)
from neuraltide.optimization.shooting import MultipleShootingProblem, ShootingConfig
from neuraltide.optimization.transforms import make_transform
from neuraltide.optimization.uq import laplace_approximation
from neuraltide.populations import WilsonCowan
from neuraltide.synapses import TsodyksMarkramSynapse
from neuraltide.training.losses import MSELoss
from neuraltide.training.readouts import HemodynamicReadout


# ---------------------------------------------------------------------------
# Вспомогательное
# ---------------------------------------------------------------------------


def _f64_l0(l0_factory):
    neuraltide.config.set_dtype(tf.float64)
    network, space, t_seq, inputs, loss = l0_factory()
    problem = CalibrationProblem(network, space, t_seq, inputs=inputs, objective=loss)
    return problem, (network, space, t_seq, inputs, loss)


def _perturbed(problem, seed=0, rel=0.05):
    phi0 = np.asarray(problem.pack(), dtype=np.float64)
    rng = np.random.default_rng(seed)
    return phi0 + rng.standard_normal(phi0.size) * (rel * np.abs(phi0) + 1e-3)


def _ee_tm_network(dt, integrator, e_r=1.0, gsyn=0.5, mode="split"):
    """
    E–E сеть Уилсона–Коуэна с рекуррентным синапсом Цодыкса–Маркрама.

    В отличие от L0-фикстуры, e_r != 0: у WilsonCowan нет v_mean, поэтому
    ток синапса равен g_eff·(e_r − 0). При e_r = 0 (как в L0 и в
    test_shooting.py::TestIncludeSynStates) синапс не влияет на частоты.
    """
    pop = WilsonCowan(
        dt=dt,
        params={
            "tau": {"value": [8.0], "trainable": True, "min": 1.0, "max": 50.0},
            "a": 1.0,
            "theta": 0.0,
            "w": 0.2,
            "I_ext": {"value": [0.8], "trainable": True},
            "max_rate": 100.0,
        },
        name="exc",
    )
    syn = TsodyksMarkramSynapse(
        n_pre=1,
        n_post=1,
        dt=dt,
        params={
            "gsyn_max": {
                "value": [[gsyn]],
                "trainable": True,
                "min": 0.001,
                "max": 5.0,
            },
            "tau_f": {"value": [[500.0]], "trainable": False},
            "tau_d": {"value": [[200.0]], "trainable": False},
            "tau_r": {"value": [[50.0]], "trainable": False},
            "Uinc": {"value": [[0.2]], "trainable": False},
            "pconn": [[1.0]],
            "e_r": e_r,
        },
        name="exc->exc",
    )
    graph = NetworkGraph(dt=dt)
    graph.add_population("exc", pop)
    graph.add_synapse("exc->exc", syn, src="exc", tgt="exc")
    return NetworkRNN(graph, integrator, integration_mode=mode)


def _t_seq(n, dt):
    return tf.constant((np.arange(n, dtype=np.float64) * dt)[None, :, None])


def _output(rates):
    return NetworkOutput(
        firing_rates={"pop": tf.constant(rates)},
        stability_loss=tf.constant(0.0, dtype=neuraltide.config.get_dtype()),
        final_state=([], []),
    )


# ---------------------------------------------------------------------------
# К1. Нормировка для статистики: FIM = информация Фишера гауссова шума
# ---------------------------------------------------------------------------


def test_fim_equals_gaussian_fisher_information(l0_factory):
    """
    Ловит К1. Для гауссова шума с σ информация Фишера по всем N точкам —
    F = SᵀS / σ², где S = ∂(частоты)/∂φ. S считается конечными разностями
    через simulate(), то есть мимо ResidualLoss и его нормировки.
    На dev: compute_fim давала JᵀJ = F / N (N = 60 точек L0).

    Невязки ResidualLoss для оптимизации остаются нормированными на N
    (0.5·||r||² = loss); FIM в единицах полного NLL строит compute_fim по
    JacobianResult.likelihood_scale, поэтому сравнивается именно она.
    """
    problem, (_, _, _, _, loss) = _f64_l0(l0_factory)
    sigma = float(loss.observations[0].sigma)
    phi = _perturbed(problem, seed=1)

    jac = residual_jacobian(problem, phi)  # до simulate: см. ограничение forward-AD
    F_code = compute_fim(jac).fim

    h = 1e-6
    cols = []
    for i in range(phi.size):
        e = np.zeros_like(phi)
        e[i] = h
        y_p = np.asarray(problem.simulate(phi + e).firing_rates["exc"]).ravel()
        y_m = np.asarray(problem.simulate(phi - e).firing_rates["exc"]).ravel()
        cols.append((y_p - y_m) / (2.0 * h))
    S = np.stack(cols, axis=1)
    F_true = S.T @ S / sigma**2

    np.testing.assert_allclose(F_code, F_true, rtol=1e-4, atol=1e-10)


@pytest.mark.slow
def test_laplace_coverage_on_noisy_l0(l0_factory):
    """
    Ловит К1 (требование дорожной карты «UQ: coverage на синтетических
    ансамблях»). 100 наборов данных с шумом σ, подгонка TRF, 95 %-интервал
    Лапласа. Доля интервалов, накрывших истинное θ, должна быть около 0.95.
    На dev интервалы шире в √N ≈ 7.7 раза, покрытие ≈ 1.00.
    """
    neuraltide.config.set_dtype(tf.float64)
    network, space, t_seq, inputs, loss = l0_factory()
    clean = np.asarray(loss.observations[0].target, dtype=np.float64)
    sigma = 0.5
    theta_true = {"exc/tau": 8.0, "exc/I_ext": 0.8}

    def make_problem(target):
        obj = ResidualLoss(
            [
                ObservationSpec(
                    name="obs",
                    population="exc",
                    target=tf.constant(target),
                    sigma=sigma,
                )
            ]
        )
        return CalibrationProblem(network, space, t_seq, inputs=inputs, objective=obj)

    # φ истинных параметров — до первой подгонки (подгонка меняет модель).
    phi_true = np.asarray(make_problem(clean).pack(), dtype=np.float64)

    rng = np.random.default_rng(2024)
    hits = {name: 0 for name in theta_true}
    n_sets = 100
    for _ in range(n_sets):
        noisy = clean + sigma * rng.standard_normal(clean.shape)
        fit = least_squares(make_problem(noisy), phi_true.copy())
        phi_hat = np.asarray(fit.base.phi_best, dtype=np.float64)
        lap = laplace_approximation(make_problem(noisy), phi_hat)  # свежий экземпляр
        for name, star in theta_true.items():
            lo, hi = lap.intervals[name]
            hits[name] += int(lo <= star <= hi)

    for name, k in hits.items():
        coverage = k / n_sets
        # биномиальный 99 %-коридор вокруг 0.95 при n = 100: примерно [0.88, 0.99]
        assert 0.88 <= coverage <= 0.99, (name, coverage)


# ---------------------------------------------------------------------------
# К2. Multiple shooting: синапсы со STP по умолчанию
# ---------------------------------------------------------------------------


def test_multiple_shooting_default_matches_free_run_with_tm_synapse():
    """
    Ловит К2. Сеть с синапсом Цодыкса–Маркрама и e_r != 0. После
    initialize_from_single_shooting склеенная траектория обязана совпасть
    со свободным прогоном: loss_ms == loss_single. На dev состояния
    синапсов (R, U, A) сбрасываются на каждом узле, потери расходятся.
    """
    neuraltide.config.set_dtype(tf.float64)
    dt, n = 0.5, 200
    network = _ee_tm_network(dt, EulerIntegrator())
    t_seq = _t_seq(n, dt)
    target = network(t_seq).firing_rates["exc"]

    # Предусловие: синапс действительно влияет на частоты.
    silent = _ee_tm_network(dt, EulerIntegrator(), e_r=0.0)
    diff = np.max(
        np.abs(np.asarray(target) - np.asarray(silent(t_seq).firing_rates["exc"]))
    )
    assert diff > 1e-3, "синапс не влияет на динамику — тест бессмыслен"

    loss = ResidualLoss(
        [ObservationSpec(name="obs", population="exc", target=target, sigma=2.0)]
    )
    space = ParameterSpace.from_network(network)
    problem = CalibrationProblem(network, space, t_seq, objective=loss)
    ms = MultipleShootingProblem(problem, ShootingConfig(4))  # значения по умолчанию
    phi = np.asarray(ms.initialize_from_single_shooting(), dtype=np.float64)
    check = ms.finalize(phi, gap_tol=1e-8)

    assert check.consistent
    assert check.loss_ms == pytest.approx(check.loss_single, rel=1e-6, abs=1e-12)


# ---------------------------------------------------------------------------
# К3. bounded_log — логарифмическая шкала
# ---------------------------------------------------------------------------


def test_bounded_log_maps_to_log_scale():
    """Ловит К3. θ(0) = √(lo·hi); равные шаги sigmoid(φ) → равные шаги log θ."""
    neuraltide.config.set_dtype(tf.float64)
    lo, hi = 0.1, 1000.0
    tr = make_transform("bounded_log", min=lo, max=hi)
    assert float(tr.forward(tf.constant(0.0, tf.float64))) == pytest.approx(
        10.0, rel=1e-8
    )

    phis = np.linspace(-4.0, 4.0, 9)
    u = 1.0 / (1.0 + np.exp(-phis))
    expected = np.exp(np.log(lo) + (np.log(hi) - np.log(lo)) * u)
    np.testing.assert_allclose(
        tr.forward(tf.constant(phis)).numpy(), expected, rtol=1e-8
    )


# ---------------------------------------------------------------------------
# К4. Поисковый бокс φ для DE/CMA не зависит от машинного эпсилона
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
@pytest.mark.parametrize("kind", ["logit"])
def test_uniform_phi_box_samples_interior_of_physical_range(dtype, kind):
    """
    Ловит К4. DE (latinhypercube) и CMA (σ₀ = ¼ ширины) брали бокс из
    problem.bounds_phi(). Равномерная выборка в боксе поиска должна в
    основном попадать внутрь физического диапазона, а не к его краям.
    На dev: 41 % (float32) и 15 % (float64) для logit.

    После исправления глобальные методы берут отдельный бокс
    search_box_phi (_search_box_phi_from_space); bounds_phi для локальных
    оптимизаторов не меняется.
    """
    neuraltide.config.set_dtype(dtype)
    lo_t, hi_t = 1.0, 50.0
    tr = make_transform(kind, min=lo_t, max=hi_t)
    space = ParameterSpace()
    space.add_entry(
        ParameterEntry(
            full_name="p", shape=tf.TensorShape([1]), transform=tr, min=lo_t, max=hi_t
        )
    )
    lo, hi = _search_box_phi_from_space(space)
    assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))

    phi = np.random.default_rng(0).uniform(lo, hi, size=(20000, 1))
    theta = tr.forward(tf.constant(phi, dtype=dtype)).numpy().ravel()
    span = hi_t - lo_t
    inner = np.mean((theta > lo_t + 0.01 * span) & (theta < hi_t - 0.01 * span))
    assert inner >= 0.9, inner


# ---------------------------------------------------------------------------
# В1. Якобиан в точке точного совпадения
# ---------------------------------------------------------------------------


def test_residual_jacobian_finite_at_exact_fit(l0_factory):
    """
    Ловит В1 (в docs/optimization.md:493 это записано как ограничение
    «Exact-fit вырожден для UQ: NaN в J»). У L0 цель — прогон при тех же
    параметрах, невязки в φ* равны нулю. Якобиан Гаусса–Ньютона там
    определён и должен совпасть с конечными разностями.
    """
    problem, _ = _f64_l0(l0_factory)
    phi_star = np.asarray(problem.pack(), dtype=np.float64)
    jac = residual_jacobian(problem, phi_star)
    assert np.all(np.isfinite(jac.J)), "NaN/Inf в якобиане при нулевых невязках"

    h = 1e-6
    J_fd = np.stack(
        [
            (problem.residual(phi_star + h * e) - problem.residual(phi_star - h * e))
            / (2 * h)
            for e in np.eye(phi_star.size)
        ],
        axis=1,
    )
    np.testing.assert_allclose(jac.J, J_fd, rtol=1e-4, atol=1e-9)


# ---------------------------------------------------------------------------
# В2. Градиент по x0 в MultiExperimentProblem
# ---------------------------------------------------------------------------


def test_two_identical_experiments_double_x0_gradient(l0_factory):
    """
    Ловит В2. Две одинаковые копии эксперимента: и потеря, и весь
    градиент (включая координату x0/exc/0) ровно вдвое больше, чем у
    одного эксперимента. На dev вклад в x0 даёт только последний:
    по x0 отношение 1.0, по остальным параметрам 2.0.

    φ задаётся явными значениями: pack(None) для пространства с
    x0-записью невозможен (у неё нет переменной модели), см.
    test_sensitivities._x0_base_phi. Первая версия теста падала именно
    на pack(None) и ошибку кода не показывала.
    """
    neuraltide.config.set_dtype(tf.float64)
    network, space, t_seq, inputs, loss = l0_factory()
    space.add_initial_state_entries(network, include=[("exc", 0)])
    exps = [
        Experiment(name=name, t_sequence=t_seq, inputs=inputs, objective=loss)
        for name in ("a", "b")
    ]
    multi = MultiExperimentProblem(network, space, exps)
    single = CalibrationProblem(network, space, t_seq, inputs=inputs, objective=loss)

    values = {
        "exc/tau": [8.0],
        "exc/I_ext": [0.8],
        "in->exc/gsyn_max": [[0.05]],
        "x0/exc/0": [0.4],  # x0 вдали от значения по умолчанию 0.0
    }
    phi = np.array(space.pack(values), dtype=np.float64)
    rng = np.random.default_rng(3)
    phi[:-1] += rng.standard_normal(phi.size - 1) * (0.05 * np.abs(phi[:-1]) + 1e-3)

    ev_m, g_m = multi.value_and_gradient(phi)
    ev_s, g_s = single.value_and_gradient(phi)
    assert ev_m.loss == pytest.approx(2.0 * ev_s.loss, rel=1e-10)
    np.testing.assert_allclose(
        np.asarray(g_m), 2.0 * np.asarray(g_s), rtol=1e-8, atol=1e-12
    )


# ---------------------------------------------------------------------------
# В3. Маска из единиц любой broadcast-формы не меняет потерю
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mask_shape", [(1, 7, 3), (1, 7, 1)])
def test_ones_mask_is_noop(mask_shape):
    """
    Ловит В3. Маска из единиц эквивалентна отсутствию маски. На dev при
    форме [1, T, 1] и n_units = 3 потеря втрое больше.
    Существующие test_masked_mean_semantics (MSE и MSLE) закрепляют
    ошибку: ожидание там посчитано как sum(m·e)/sum(m) без broadcast.
    """
    neuraltide.config.set_dtype(tf.float64)
    rng = np.random.default_rng(0)
    rates = 5.0 * rng.standard_normal((1, 7, 3))
    target = 5.0 + 2.0 * rng.standard_normal((1, 7, 3))
    ones = np.ones(mask_shape)

    plain = ResidualLoss([ObservationSpec(name="o", population="pop", target=target)])
    masked = ResidualLoss(
        [ObservationSpec(name="o", population="pop", target=target, mask=ones)]
    )
    assert float(masked(_output(rates), None)) == pytest.approx(
        float(plain(_output(rates), None)), rel=1e-12
    )

    mse_plain = MSELoss({"pop": tf.constant(target)})
    mse_masked = MSELoss({"pop": tf.constant(target)}, mask={"pop": tf.constant(ones)})
    assert float(mse_masked(_output(rates), None)) == pytest.approx(
        float(mse_plain(_output(rates), None)), rel=1e-12
    )


# ---------------------------------------------------------------------------
# В4. Checkpointing с MSELoss
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("segment_length", [1, 17, 60])
def test_checkpointed_mse_matches_plain_bptt(l0_factory, segment_length):
    """
    Ловит В4. Существующий test_grad_matches_plain_bptt использует только
    ResidualLoss. С MSELoss без маски сумма сегментов на dev в L раз меньше.
    segment_length = 1 — контроль: при L = 1 ошибка не проявляется, этот
    вариант проходит и на dev.
    """
    problem, (network, space, t_seq, inputs, loss) = _f64_l0(l0_factory)
    target = tf.constant(np.asarray(loss.observations[0].target, dtype=np.float64))
    mse_problem = CalibrationProblem(
        network, space, t_seq, inputs=inputs, objective=MSELoss({"exc": target})
    )
    phi = _perturbed(mse_problem, seed=5)
    ev, grad = mse_problem.value_and_gradient(phi)
    res = value_and_gradient_checkpointed(mse_problem, phi, segment_length)
    assert res.loss == pytest.approx(ev.loss, rel=1e-10)
    np.testing.assert_allclose(res.grad, np.asarray(grad), rtol=1e-6, atol=1e-10)


# ---------------------------------------------------------------------------
# В5. CAGrad: max_d min_i <g_i, d>
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("c", [0.3, 0.6])
def test_cagrad_improves_worst_task_over_mean(c):
    """
    Ловит В5. По Liu et al. (2021) направление d максимизирует худшее
    убывание min_i <g_i, d> в шаре ||d − g0|| ≤ c·||g0||. Для
    g1 = (1, 0.5), g2 = (−1, 0.5) решение d* = (0, 0.5·(1 + c)).
    Существующий test_exact_solution_two_gradients ждёт (0, 0.5·(1 − c)),
    то есть закрепляет ошибку.
    """
    g1, g2 = np.array([1.0, 0.5]), np.array([-1.0, 0.5])
    g0 = 0.5 * (g1 + g2)
    d = cagrad_project([g1, g2], c=c)
    np.testing.assert_allclose(d, [0.0, 0.5 * (1.0 + c)], atol=1e-6)
    assert min(g1 @ d, g2 @ d) >= min(g1 @ g0, g2 @ g0) - 1e-9

    rng = np.random.default_rng(0)
    for _ in range(50):
        gs = [rng.standard_normal(4) for _ in range(3)]
        g0 = np.mean(gs, axis=0)
        d = cagrad_project(gs, c=c)
        assert min(g @ d for g in gs) >= min(g @ g0 for g in gs) - 1e-7


# ---------------------------------------------------------------------------
# В6. Порядок сходимости сети по dt
# ---------------------------------------------------------------------------


def test_rk4_network_order_with_tm_coupling():
    """
    Регрессионный тест исправления В6: сходимость по dt на связанной сети
    со STP в network-wide режиме (integration_mode='network'). Для RK4
    ожидается порядок около 4 (> 3.5). Раньше шаг сети был расщеплён
    (синапсы | популяции): частота пресинаптического юнита бралась в
    начале шага, ток синапса — в конце, и порядок связанной сети был 2.
    Split-понижение (~2) остаётся контролем в
    test_audit_controls.py::test_network_order_diagnostic, а согласованный
    network-wide шаг (step_network) закреплён здесь.
    """
    neuraltide.config.set_dtype(tf.float64)
    t_end = 20.0

    def final_E(dt):
        net = _ee_tm_network(dt, RK4Integrator(), mode="network")
        out = net(_t_seq(int(round(t_end / dt)), dt))
        return float(np.asarray(out.final_state[0][0]).ravel()[0])  # E в момент t_end

    ref = final_E(0.5 / 64)
    errs = [abs(final_E(dt) - ref) for dt in (0.5, 0.25, 0.125)]
    assert errs[0] > errs[1] > errs[2] > 0.0, errs
    order = np.log2(errs[1] / errs[2])
    assert order > 3.5, (errs, order)


# ---------------------------------------------------------------------------
# С1. FrozenProblem.evaluate() без φ
# ---------------------------------------------------------------------------


def test_frozen_evaluate_without_phi_applies_fixed(l0_factory):
    """Ловит С1. evaluate() обязан считать потерю в точке с fixed-значениями."""
    problem, _ = _f64_l0(l0_factory)
    frozen = FrozenProblem(problem, fixed={"exc/tau": 12.0})
    loss_none = frozen.evaluate().loss  # первым вызовом, до любых evaluate(phi)
    loss_phi = frozen.evaluate(np.asarray(frozen.pack(), dtype=np.float64)).loss
    assert loss_none == pytest.approx(loss_phi, rel=1e-12)
    assert loss_none > 0.0  # τ = 12 вместо 8 — не точка генерации


# ---------------------------------------------------------------------------
# С6. HRF
# ---------------------------------------------------------------------------


def test_hrf_kernel_has_canonical_time_course():
    """
    Ловит С6. Каноническая HRF (SPM, Glover 1999): пик через 4–7 с,
    минимум постстимульного провала через 12–20 с, длина ядра ≥ 25 с.
    На dev ядро длиной 20 мс, пик положительной компоненты при t = 15
    (единиц шкалы), провала в окне нет.
    """
    dt = 10.0  # мс
    readout = HemodynamicReadout(dt=dt)
    readout(tf.zeros((1, 8, 1), dtype=neuraltide.config.get_dtype()))
    k = readout._kernel.numpy()[:, 0, 0]
    t_s = np.arange(k.size) * dt / 1000.0
    assert t_s[-1] >= 25.0, t_s[-1]
    assert 4.0 <= t_s[np.argmax(k)] <= 7.0, t_s[np.argmax(k)]
    i_min = np.argmin(k)
    assert k[i_min] < 0.0 and 12.0 <= t_s[i_min] <= 20.0, (k[i_min], t_s[i_min])
