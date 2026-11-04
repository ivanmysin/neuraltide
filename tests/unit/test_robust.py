"""
Тесты P2 robust/adaptive losses (этап P2 дорожной карты, раздел 7.3).

Покрываются:
- kind 'cauchy' / 'tukey' против эталонных psi (numpy), residual
  reconstruction 0.5*||r||^2 == loss, per-step/full эквивалентность с масками
  (включая float64);
- ObservationSpec.weight как 0-d Tensor/Variable (и валидация);
- PCGrad: антагонистические градиенты дают известную проекцию;
- CAGrad: точное решение на двух градиентах, контроль прогресса через c;
- стратегии весов: UncertaintyWeighting реагирует на шум потерь, GradNorm
  выравнивает взвешенные нормы, FixedWeights/SelfAdaptiveWeights, JSON
  state_dict/load_state_dict;
- balanced_gradient: losses == evaluate().components, взвешенная сумма
  градиентов против value_and_gradient, guard adjoint/Trainer, ошибки
  некомпонентного objective;
- adaptive_window_sampler: выбор окна с большим остатком, изменение маски
  focused view, неприкосновенность исходных спецификаций;
- balanced_train на малой L0-задаче: сходимость, непустая weight_history,
  baseline в notes/result, history_path, сериализация BalancedResult.

Сетевые тесты используют L0-фикстуру conftest (WilsonCowan 1 юнит +
StaticSynapse от ConstantRate) в float64; синтетика (проекторы, стратегии)
без сети.
"""

import json
import os

import numpy as np
import pytest
import tensorflow as tf

import neuraltide.config
from neuraltide.core.network import NetworkOutput
from neuraltide.optimization import (
    CalibrationProblem,
    ObservationSpec,
    ResidualLoss,
)
from neuraltide.optimization.robust import (
    AdaptiveWindowFocus,
    BalancedResult,
    FixedWeights,
    GradNormWeights,
    SelfAdaptiveWeights,
    UncertaintyWeighting,
    adaptive_window_sampler,
    balanced_gradient,
    balanced_train,
    cagrad_project,
    pcgrad_project,
)

T = 6
N = 3


@pytest.fixture(autouse=True)
def f64_dtype(reset_dtype):
    """Синтетика файла считает в float64 (после сброса conftest)."""
    neuraltide.config.set_dtype(tf.float64)
    yield


def _rates(rng, scale=5.0, offset=5.0):
    return (offset + scale * rng.standard_normal((1, T, N))).astype(np.float64)


def _predictions(rates) -> NetworkOutput:
    return NetworkOutput(
        firing_rates={"pop": tf.constant(rates)},
        stability_loss=tf.constant(0.0, dtype=neuraltide.config.get_dtype()),
        final_state=([], []),
    )


def _ref_psi(e: np.ndarray, kind: str, scale: float) -> np.ndarray:
    """Эталонные psi для cauchy/tukey на numpy."""
    if kind == "cauchy":
        return 0.5 * scale**2 * np.log1p((e / scale) ** 2)
    return np.where(
        np.abs(e) <= scale,
        scale**2 / 6.0 * (1.0 - (1.0 - (e / scale) ** 2) ** 3),
        scale**2 / 6.0,
    )


def _sum_per_step(loss, rates, target, mask=None):
    total = tf.constant(0.0, dtype=neuraltide.config.get_dtype())
    for t in range(T):
        rates_t = {"pop": rates[:, t, :]}
        target_t = {name: value[:, t, :] for name, value in target.items()}
        mask_t = None
        if mask is not None:
            mask_t = {name: value[:, t, :] for name, value in mask.items()}
        total += loss.per_step_loss(rates_t, target_t, mask=mask_t)
    return float(total)


def _build_f64(l0_factory) -> CalibrationProblem:
    """L0-задача в float64."""
    neuraltide.config.set_dtype(tf.float64)
    network, space, t_seq, inputs, loss = l0_factory()
    return CalibrationProblem(network, space, t_seq, inputs=inputs, objective=loss)


def _two_component_problem(problem: CalibrationProblem) -> CalibrationProblem:
    """
    Двухкомпонентный objective поверх L0: «early» — первая половина горизонта,
    «late» — вторая (комплементарные маски, одна целевая траектория).
    """
    target = problem.objective.observations[0].target
    T_full = int(target.shape[1])
    half = T_full // 2
    dtype = neuraltide.config.get_dtype()
    mask_early = np.zeros((1, T_full, 1), dtype=np.float64)
    mask_early[0, :half, 0] = 1.0
    mask_late = np.zeros((1, T_full, 1), dtype=np.float64)
    mask_late[0, half:, 0] = 1.0
    loss = ResidualLoss(
        [
            ObservationSpec(
                name="early",
                population="exc",
                target=target,
                mask=tf.constant(mask_early, dtype=dtype),
            ),
            ObservationSpec(
                name="late",
                population="exc",
                target=target,
                mask=tf.constant(mask_late, dtype=dtype),
            ),
        ]
    )
    return problem.with_objective(loss)


# ---------------------------------------------------------------------------
# Cauchy / Tukey
# ---------------------------------------------------------------------------


class TestRobustKinds:
    @pytest.mark.parametrize("kind", ["cauchy", "tukey"])
    @pytest.mark.parametrize("scale", [0.5, 1.3, 3.0])
    def test_matches_reference_psi(self, kind, scale):
        rng = np.random.default_rng(11)
        rates, target = _rates(rng), _rates(rng, offset=6.0)
        sigma, weight = 1.4, 0.6
        mask = (rng.uniform(size=(1, T, N)) > 0.4).astype(np.float64)
        spec = ObservationSpec(
            name="o",
            population="pop",
            target=target,
            kind=kind,
            scale=scale,
            sigma=sigma,
            weight=weight,
            mask=mask,
        )
        loss = ResidualLoss([spec])
        value = float(loss(_predictions(rates), None))
        e = rates - target
        psi = _ref_psi(e, kind, scale)
        ref = weight * float((mask * psi).sum()) / (float(mask.sum()) * sigma**2)
        assert value == pytest.approx(ref, rel=1e-10)

    def test_tukey_saturation(self):
        # при |e| > c вклад константен: loss совпадает для 2c и 10c
        c = 1.0
        rates = np.zeros((1, T, N), dtype=np.float64)
        target_far = np.full((1, T, N), 10.0 * c, dtype=np.float64)
        target_thr = np.full((1, T, N), 2.0 * c, dtype=np.float64)
        loss_a = ResidualLoss(
            [
                ObservationSpec(
                    name="o", population="pop", target=target_far, kind="tukey", scale=c
                )
            ]
        )
        loss_b = ResidualLoss(
            [
                ObservationSpec(
                    name="o", population="pop", target=target_thr, kind="tukey", scale=c
                )
            ]
        )
        assert float(loss_a(_predictions(rates), None)) == pytest.approx(
            float(loss_b(_predictions(rates), None)), rel=1e-10
        )
        # без маски norm = tf.size(target) = T*N, sum(psi) = T*N*(c^2/6)
        ref = c**2 / 6.0
        assert float(loss_a(_predictions(rates), None)) == pytest.approx(ref, rel=1e-10)

    @pytest.mark.parametrize("kind", ["cauchy", "tukey"])
    def test_residual_reconstruction(self, kind):
        rng = np.random.default_rng(12)
        rates, target = _rates(rng), _rates(rng, offset=6.0)
        mask = (rng.uniform(size=(1, T, N)) > 0.4).astype(np.float64)
        spec = ObservationSpec(
            name="o",
            population="pop",
            target=target,
            kind=kind,
            scale=1.1,
            sigma=1.3,
            weight=0.8,
            mask=mask,
        )
        loss = ResidualLoss([spec])
        pred = _predictions(rates)
        r = np.asarray(loss.residual(pred))
        assert 0.5 * float(r @ r) == pytest.approx(float(loss(pred, None)), rel=1e-10)

    @pytest.mark.parametrize("kind", ["cauchy", "tukey"])
    def test_full_equals_sum_per_step_with_mask_slices(self, kind):
        rng = np.random.default_rng(13)
        rates, target = _rates(rng), _rates(rng, offset=6.0)
        mask = (rng.uniform(size=(1, T, N)) > 0.4).astype(np.float64)
        spec = ObservationSpec(
            name="o", population="pop", target=target, kind=kind, scale=0.9, mask=mask
        )
        loss = ResidualLoss([spec])
        full = float(loss(_predictions(rates), None))
        stepped = _sum_per_step(loss, rates, loss.target, mask={"o": mask})
        assert full == pytest.approx(stepped, rel=1e-10)

    @pytest.mark.parametrize("kind", ["cauchy", "tukey"])
    def test_float64_dtype(self, kind):
        neuraltide.config.set_dtype(tf.float64)
        rng = np.random.default_rng(14)
        rates, target = _rates(rng), _rates(rng, offset=6.0)
        spec = ObservationSpec(
            name="o", population="pop", target=target, kind=kind, scale=2.0
        )
        loss = ResidualLoss([spec])
        value = loss(_predictions(rates), None)
        assert value.dtype == tf.float64

    def test_bad_scale(self):
        with pytest.raises(ValueError, match="scale"):
            ObservationSpec(
                name="o",
                population="pop",
                target=np.zeros((1, T, N)),
                kind="cauchy",
                scale=0.0,
            )


class TestTensorWeight:
    def test_weight_as_zero_dim_tensor_matches_float(self):
        rng = np.random.default_rng(15)
        rates, target = _rates(rng), _rates(rng, offset=6.0)
        loss_float = ResidualLoss(
            [ObservationSpec(name="o", population="pop", target=target, weight=2.5)]
        )
        loss_tensor = ResidualLoss(
            [
                ObservationSpec(
                    name="o",
                    population="pop",
                    target=target,
                    weight=tf.constant(2.5, dtype=neuraltide.config.get_dtype()),
                )
            ]
        )
        assert float(loss_tensor(_predictions(rates), None)) == pytest.approx(
            float(loss_float(_predictions(rates), None)), rel=1e-10
        )

    def test_weight_as_variable(self):
        rng = np.random.default_rng(16)
        rates, target = _rates(rng), _rates(rng, offset=6.0)
        w = tf.Variable(1.0, dtype=neuraltide.config.get_dtype())
        spec = ObservationSpec(name="o", population="pop", target=target, weight=w)
        loss = ResidualLoss([spec])
        before = float(loss(_predictions(rates), None))
        w.assign(3.0)
        after = float(loss(_predictions(rates), None))
        assert after == pytest.approx(3.0 * before, rel=1e-10)

    def test_tensor_weight_validation(self):
        target = np.zeros((1, T, N))
        with pytest.raises(ValueError, match="0-d"):
            ObservationSpec(
                name="o", population="pop", target=target, weight=tf.ones(2)
            )
        with pytest.raises(ValueError, match="non-negative"):
            ObservationSpec(
                name="o",
                population="pop",
                target=target,
                weight=tf.constant(-1.0, dtype=neuraltide.config.get_dtype()),
            )


# ---------------------------------------------------------------------------
# PCGrad / CAGrad
# ---------------------------------------------------------------------------


class TestPCGrad:
    def test_antagonistic_pair_gives_zero(self):
        # g1=(1,0), g2=(-1,0): каждый проецируется в ноль
        out = pcgrad_project({"a": np.array([1.0, 0.0]), "b": np.array([-1.0, 0.0])})
        assert np.allclose(out, [0.0, 0.0], atol=1e-12)

    def test_known_projection_mixed(self):
        # g1=(2,0), g2=(-1,0.5) — проекции (0.4,0.8) и (0,0.5), mean (0.2,0.65)
        out = pcgrad_project([np.array([2.0, 0.0]), np.array([-1.0, 0.5])])
        assert np.allclose(out, [0.2, 0.65], atol=1e-12)

    def test_no_conflict_unchanged(self):
        out = pcgrad_project({"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])})
        assert np.allclose(out, [0.5, 0.5], atol=1e-12)

    def test_single_gradient_unchanged(self):
        g = np.array([1.0, -2.0, 3.0])
        assert np.allclose(pcgrad_project({"only": g}), g, atol=1e-12)

    def test_validation(self):
        with pytest.raises(ValueError, match="shapes"):
            pcgrad_project([np.zeros(2), np.zeros(3)])
        with pytest.raises(ValueError, match="at least one"):
            pcgrad_project([])


def _cagrad_dual_reference(gs, c):
    """
    Эталон CAGrad по двойственной задаче статьи (Liu et al., 2021, ур. 4–5):
    w* = argmin_{w ∈ симплекс} <g_w, g0> + sqrt(φ)·||g_w||, φ = c²||g0||²,
    d = g0 + sqrt(φ)·g_w*/||g_w*||. Другой алгоритм, чем в коде (там —
    прямая задача max t при <g_i, d> >= t).
    """
    from scipy import optimize

    G = np.stack(gs, axis=0)
    g0 = G.mean(axis=0)
    root_phi = c * np.linalg.norm(g0)

    def f(w):
        gw = w @ G
        return float(gw @ g0 + root_phi * np.linalg.norm(gw))

    n = G.shape[0]
    res = optimize.minimize(
        f,
        np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
        options={"ftol": 1e-14, "maxiter": 1000},
    )
    gw = res.x @ G
    return g0 + root_phi * gw / np.linalg.norm(gw)


class TestCAGrad:
    def test_exact_solution_two_gradients(self):
        # g1=(1,0.5), g2=(-1,0.5): g0=(0,0.5). По Liu et al. (max_d min_i)
        # d* = (0, 0.5(1+c)): худшая задача убывает быстрее, чем при g0.
        # Прежний эталон (0, 0.5(1-c)) был решением min-max и закреплял
        # инвертированную постановку (В5 ревью).
        g1 = np.array([1.0, 0.5])
        g2 = np.array([-1.0, 0.5])
        for c in (0.0, 0.3, 0.6):
            d = cagrad_project({"a": g1, "b": g2}, c=c)
            assert np.allclose(d, [0.0, 0.5 * (1.0 + c)], atol=1e-6)

    def test_c_controls_minimal_progress(self):
        rng = np.random.default_rng(11)
        for c in (0.1, 0.5, 0.9):
            for _ in range(20):
                gs = [rng.standard_normal(4) for _ in range(3)]
                g0 = np.mean(gs, axis=0)
                d = cagrad_project(gs, c=c)
                # допустимость: шар вокруг g0
                assert np.linalg.norm(d - g0) <= c * np.linalg.norm(g0) + 1e-6
                # главное свойство CAGrad: худшая задача убывает не медленнее,
                # чем при простом усреднении
                worst_d = min(float(g @ d) for g in gs)
                worst_g0 = min(float(g @ g0) for g in gs)
                assert worst_d >= worst_g0 - 1e-7

    def test_matches_dual_formulation(self):
        rng = np.random.default_rng(12)
        for c in (0.2, 0.5, 0.8):
            for _ in range(10):
                gs = [rng.standard_normal(5) for _ in range(3)]
                d = cagrad_project(gs, c=c)
                ref = _cagrad_dual_reference(gs, c)
                worst = min(float(g @ d) for g in gs)
                worst_ref = min(float(g @ ref) for g in gs)
                # одинаковое значение задачи; при единственном решении — и d
                assert worst == pytest.approx(worst_ref, rel=1e-5, abs=1e-7)
                np.testing.assert_allclose(d, ref, rtol=1e-4, atol=1e-5)

    def test_zero_mean_returns_zero(self):
        out = cagrad_project([np.array([1.0, 0.0]), np.array([-1.0, 0.0])], c=0.5)
        assert np.allclose(out, [0.0, 0.0], atol=1e-12)

    def test_c_validation(self):
        with pytest.raises(ValueError, match="c"):
            cagrad_project([np.ones(2)], c=1.5)


# ---------------------------------------------------------------------------
# Стратегии весов
# ---------------------------------------------------------------------------


class TestWeightStrategies:
    def test_fixed_weights_constant(self):
        s = FixedWeights({"a": 1.0, "b": 2.0})
        w0 = s.initial_weights(["a", "b"])
        w1 = s.update({"a": 5.0, "b": 0.1}, None, 0)
        assert np.allclose(w0, [1.0, 2.0])
        assert np.allclose(w1, [1.0, 2.0])

    def test_fixed_weights_scalar_and_vector(self):
        assert np.allclose(FixedWeights(3.0).initial_weights(["a", "b"]), [3.0, 3.0])
        assert np.allclose(
            FixedWeights([1.0, 4.0]).initial_weights(["a", "b"]), [1.0, 4.0]
        )

    def test_uncertainty_reacts_to_loss_noise(self):
        s = UncertaintyWeighting()
        s.initial_weights(["noisy", "clean"])
        w = s.update({"noisy": 100.0, "clean": 1.0}, None, 0)
        assert w[0] < w[1]  # шумной компоненте — меньший вес
        assert np.mean(w) == pytest.approx(1.0)  # mean-нормировка
        # реакция на смену шума: веса меняются местами
        w2 = s.update({"noisy": 1.0, "clean": 100.0}, None, 1)
        assert w2[0] > w2[1]

    def test_self_adaptive_exponent_one(self):
        s = SelfAdaptiveWeights()
        s.initial_weights(["hi", "lo"])
        w = s.update({"hi": 100.0, "lo": 1.0}, None, 0)
        # w ∝ 1/loss: отношение ~ 100 (клип не срабатывает при eps)
        assert w[1] / w[0] == pytest.approx(100.0, rel=1e-3)

    def test_gradnorm_equalizes_weighted_norms(self):
        s = GradNormWeights()
        s.initial_weights(["big", "small"])
        grads = {"big": np.array([10.0, 0.0]), "small": np.array([1.0, 0.0])}
        w = s.update({"big": 1.0, "small": 1.0}, grads, 0)
        weighted = np.array([10.0 * w[0], 1.0 * w[1]])
        assert weighted[0] == pytest.approx(weighted[1], rel=1e-6)

    def test_gradnorm_keeps_weights_without_grads(self):
        s = GradNormWeights()
        w0 = s.initial_weights(["a", "b"])
        w1 = s.update({"a": 1.0, "b": 2.0}, None, 0)
        assert np.allclose(w0, w1)

    def test_gradnorm_requires_losses_keys(self):
        s = GradNormWeights()
        s.initial_weights(["a", "b"])
        with pytest.raises(KeyError, match="missing"):
            s.update({"a": 1.0}, {"a": np.ones(2)}, 0)

    @pytest.mark.parametrize(
        "strategy",
        [
            FixedWeights(1.0),
            UncertaintyWeighting(),
            SelfAdaptiveWeights(),
            GradNormWeights(),
        ],
    )
    def test_state_dict_roundtrip(self, strategy):
        names = ["a", "b"]
        strategy.initial_weights(names)
        grads = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 2.0])}
        strategy.update({"a": 3.0, "b": 0.7}, grads, 0)
        state = strategy.state_dict()
        state_json = json.dumps(state)  # JSON-сериализуемо
        clone = (
            FixedWeights(1.0)
            if isinstance(strategy, FixedWeights)
            else type(strategy)()
        )
        clone.load_state_dict(json.loads(state_json))
        assert np.allclose(clone._current(), strategy._current())
        assert clone.state_dict()["step"] == strategy.state_dict()["step"]

    def test_state_dict_type_mismatch(self):
        s = UncertaintyWeighting()
        with pytest.raises(ValueError, match="type"):
            s.load_state_dict({"type": "GradNormWeights"})

    def test_update_before_initial(self):
        s = FixedWeights(1.0)
        with pytest.raises(RuntimeError, match="initial_weights"):
            s.update({"a": 1.0}, None, 0)


# ---------------------------------------------------------------------------
# balanced_gradient
# ---------------------------------------------------------------------------


class TestBalancedGradient:
    def test_losses_match_evaluate_components(self, l0_factory):
        problem = _build_f64(l0_factory)
        phi = np.asarray(problem.pack(), dtype=np.float64) + 0.05
        losses, grad = balanced_gradient(problem, phi, {"obs": 1.0})
        eval_result = problem.evaluate(phi)
        assert losses.keys() == eval_result.components.keys()
        for name, value in losses.items():
            assert value == pytest.approx(eval_result.components[name], rel=1e-10)
        assert grad.shape == (problem.size,)

    def test_weighted_sum_against_value_and_gradient(self, l0_factory):
        problem = _two_component_problem(_build_f64(l0_factory))
        phi = np.asarray(problem.pack(), dtype=np.float64) + 0.05
        losses, grad = balanced_gradient(problem, phi, {"early": 2.0, "late": 1.0})
        # эталон: 2*g_early + g_late через отдельные value_and_gradient view
        pieces = []
        for name, w in (("early", 2.0), ("late", 1.0)):
            single = problem.objective.observations[0 if name == "early" else 1]
            view = problem.with_objective(ResidualLoss([single]))
            _, g = view.value_and_gradient(phi)
            pieces.append(w * np.asarray(g, dtype=np.float64))
        expected = pieces[0] + pieces[1]
        assert np.allclose(grad, expected, atol=1e-8)
        assert losses["early"] > 0.0 and losses["late"] > 0.0

    def test_projector_single_component_identity(self, l0_factory):
        problem = _build_f64(l0_factory)
        phi = np.asarray(problem.pack(), dtype=np.float64) + 0.05
        _, grad_plain = balanced_gradient(problem, phi, {"obs": 1.0})
        _, grad_pc = balanced_gradient(
            problem, phi, {"obs": 1.0}, projector=pcgrad_project
        )
        assert np.allclose(grad_plain, grad_pc, atol=1e-12)

    def test_counters_incremented(self, l0_factory):
        problem = _build_f64(l0_factory)
        phi = np.asarray(problem.pack(), dtype=np.float64) + 0.05
        nf0, ng0 = problem.n_forward, problem.n_gradient
        balanced_gradient(problem, phi, {"obs": 1.0})
        assert problem.n_forward == nf0 + 1
        assert problem.n_gradient == ng0 + 1

    def test_adjoint_problem_rejected(self):
        class FakeAdjoint:
            grad_method = "adjoint"

        with pytest.raises(NotImplementedError, match="adjoint"):
            balanced_gradient(FakeAdjoint(), np.zeros(1), {"a": 1.0})

    def test_adjoint_projector_rejected(self):
        class FakeAdjoint:
            grad_method = "adjoint"

        with pytest.raises(NotImplementedError, match="PCGrad"):
            balanced_gradient(
                FakeAdjoint(), np.zeros(1), {"a": 1.0}, projector=pcgrad_project
            )

    def test_trainer_rejected(self):
        class FakeTrainer:
            __module__ = "neuraltide.training.trainer"
            grad_method = "bptt"

        with pytest.raises(NotImplementedError, match="CalibrationProblem"):
            balanced_gradient(FakeTrainer(), np.zeros(1), {"a": 1.0})

    def test_non_component_objective_rejected(self, l0_factory):
        problem = _build_f64(l0_factory)
        problem = problem.with_objective(lambda predictions, network: tf.constant(0.0))
        phi = np.asarray(problem.pack(), dtype=np.float64)
        with pytest.raises(TypeError, match="components"):
            balanced_gradient(problem, phi, {"a": 1.0})


# ---------------------------------------------------------------------------
# Adaptive window sampler
# ---------------------------------------------------------------------------


class TestAdaptiveWindow:
    def _spiked_problem(self, problem: CalibrationProblem, s: int, e: int):
        """Target = rates(φ*) с трёхкратным выбросом в окне [s, e)."""
        phi = np.asarray(problem.pack(), dtype=np.float64)
        rates = np.asarray(problem.simulate(phi).firing_rates["exc"], dtype=np.float64)
        target = rates.copy()
        target[0, s:e, :] = 3.0 * rates[0, s:e, :]
        loss = ResidualLoss(
            [ObservationSpec(name="obs", population="exc", target=target)]
        )
        return problem.with_objective(loss), phi

    def test_selects_window_with_large_residual(self, l0_factory):
        problem = _build_f64(l0_factory)
        s, e = 30, 45
        p2, phi = self._spiked_problem(problem, s, e)
        sampler = adaptive_window_sampler(p2, window_size=15)
        start, stop = sampler.select(phi)
        assert (start, stop) == (s, e)
        energies = sampler.last_energies
        assert np.argmax(energies) == 2  # окна [0,15), [15,30), [30,45), ...
        assert energies[2] > 0.0
        assert np.sum(energies[:2]) == pytest.approx(0.0, abs=1e-12)

    def test_focused_view_boosts_selected_window_only(self, l0_factory):
        problem = _build_f64(l0_factory)
        s, e = 20, 35
        p2, phi = self._spiked_problem(problem, s, e)
        sampler = adaptive_window_sampler(p2, window_size=15, boost=4.0)
        focused = sampler(phi)
        new_spec = focused.objective.observations[0]
        m = np.asarray(new_spec.mask)
        assert float(m[0, s, 0]) == pytest.approx(4.0)
        assert float(m[0, 0, 0]) == pytest.approx(1.0)
        # исходная спецификация не мутирована
        assert p2.objective.observations[0].mask is None
        # focused loss больше plain (окно с ошибкой перевешено)
        output = p2.simulate(phi)
        assert float(focused.objective(output, None)) > float(
            p2.objective(output, None)
        )

    def test_sampler_does_not_mutate_original_specs(self, l0_factory):
        problem = _build_f64(l0_factory)
        p2, phi = self._spiked_problem(problem, 15, 30)
        sampler = adaptive_window_sampler(p2, window_size=15, boost=2.0)
        sampler(phi)
        assert p2.objective.observations[0].mask is None
        assert sampler.last_window == (15, 30)

    def test_multiplier_shape_and_bounds(self, l0_factory):
        problem = _build_f64(l0_factory)
        sampler = AdaptiveWindowFocus(problem, window_size=10, boost=2.5)
        mult = sampler.window_multiplier(10, 20)
        assert mult.shape == (1, int(problem.t_sequence.shape[1]), 1)
        assert float(mult[0, 10, 0]) == pytest.approx(2.5)
        assert float(mult[0, 19, 0]) == pytest.approx(2.5)
        assert float(mult[0, 20, 0]) == pytest.approx(1.0)

    def test_validation(self, l0_factory):
        problem = _build_f64(l0_factory)
        with pytest.raises(ValueError, match="focus"):
            adaptive_window_sampler(problem, window_size=10, focus="nope")
        with pytest.raises(ValueError, match="window_size"):
            adaptive_window_sampler(problem, window_size=0)
        with pytest.raises(ValueError, match="boost"):
            adaptive_window_sampler(problem, window_size=10, boost=0.5)
        plain = problem.with_objective(lambda predictions, network: tf.constant(0.0))
        with pytest.raises(TypeError, match="observations"):
            adaptive_window_sampler(plain, window_size=10)


# ---------------------------------------------------------------------------
# balanced_train
# ---------------------------------------------------------------------------


class TestBalancedTrain:
    def test_converges_and_logs_weights(self, l0_factory):
        problem = _two_component_problem(_build_f64(l0_factory))
        phi0 = np.asarray(problem.pack(), dtype=np.float64) + 0.1
        eval0 = problem.evaluate(phi0)
        result = balanced_train(
            problem,
            phi0,
            UncertaintyWeighting(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            max_steps=30,
            seed=123,
        )
        assert result.base.loss_best < 0.1 * (
            eval0.components["early"] + eval0.components["late"]
        )
        assert len(result.weight_history) == 30
        entry = result.weight_history[-1]
        assert {
            "step",
            "loss",
            "grad_norm",
            "w:early",
            "w:late",
            "l:early",
            "l:late",
        } <= set(entry)
        # веса неотрицательны и логируются на каждом шаге
        for h in result.weight_history:
            assert h["w:early"] > 0.0 and h["w:late"] > 0.0

    def test_baseline_recorded_in_notes_and_dict(self, l0_factory):
        problem = _two_component_problem(_build_f64(l0_factory))
        phi0 = np.asarray(problem.pack(), dtype=np.float64) + 0.1
        result = balanced_train(
            problem, phi0, FixedWeights(1.0), optimizer="sgd", max_steps=5
        )
        assert result.baseline["method"] == "FixedWeights(1.0)"
        assert "baseline" in result.notes.lower()
        assert "FixedWeights(1.0)" in result.notes
        assert result.baseline["history_len"] == 5
        assert result.baseline["best_loss"] > 0.0

    def test_with_pcgrad_projector(self, l0_factory):
        problem = _two_component_problem(_build_f64(l0_factory))
        phi0 = np.asarray(problem.pack(), dtype=np.float64) + 0.1
        result = balanced_train(
            problem,
            phi0,
            FixedWeights(1.0),
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            max_steps=10,
            projector=pcgrad_project,
        )
        assert "PCGrad" in result.notes
        assert len(result.weight_history) == 10

    def test_history_path_written(self, l0_factory, tmp_path):
        problem = _two_component_problem(_build_f64(l0_factory))
        phi0 = np.asarray(problem.pack(), dtype=np.float64) + 0.1
        path = os.path.join(tmp_path, "history.json")
        balanced_train(
            problem,
            phi0,
            FixedWeights(1.0),
            optimizer="sgd",
            max_steps=4,
            history_path=path,
        )
        with open(path) as f:
            hist = json.load(f)
        assert len(hist) == 4
        assert hist[0]["step"] == 0

    def test_serialization_roundtrip(self, l0_factory, tmp_path):
        problem = _two_component_problem(_build_f64(l0_factory))
        phi0 = np.asarray(problem.pack(), dtype=np.float64) + 0.1
        result = balanced_train(
            problem, phi0, UncertaintyWeighting(), optimizer="sgd", max_steps=4
        )
        path = os.path.join(tmp_path, "result.json")
        result.save(path)
        loaded = BalancedResult.load(path)
        assert loaded.base.loss_best == pytest.approx(result.base.loss_best)
        assert loaded.weight_history == result.weight_history
        assert loaded.notes == result.notes
        assert loaded.baseline == result.baseline
        assert np.allclose(loaded.base.phi_best, result.base.phi_best)
        # to_dict JSON-сериализуем
        json.dumps(result.to_dict())

    def test_validations(self, l0_factory):
        problem = _build_f64(l0_factory)
        phi0 = np.asarray(problem.pack(), dtype=np.float64)
        with pytest.raises(ValueError, match="max_steps"):
            balanced_train(problem, phi0, FixedWeights(1.0), max_steps=0)
        with pytest.raises(ValueError, match="optimizer"):
            balanced_train(
                problem, phi0, FixedWeights(1.0), max_steps=2, optimizer="nope"
            )
        with pytest.raises(TypeError, match="WeightStrategy"):
            balanced_train(problem, phi0, object(), max_steps=2)
        plain = problem.with_objective(lambda predictions, network: tf.constant(0.0))
        with pytest.raises(TypeError, match="components"):
            balanced_train(plain, phi0, FixedWeights(1.0), max_steps=2)

    def test_adjoint_rejected(self, l0_factory):
        class FakeAdjoint:
            grad_method = "adjoint"

        with pytest.raises(NotImplementedError, match="adjoint"):
            balanced_train(FakeAdjoint(), np.zeros(1), FixedWeights(1.0), max_steps=2)
