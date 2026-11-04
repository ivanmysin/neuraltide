"""
Тесты P1: локальные оптимизаторы (minimize_lbfgsb, least_squares) поверх
CalibrationProblem.

Используются L0-фикстуры conftest: e_r=0 → внешний вход не влияет на
динамику, поэтому восстанавливаются только реально влияющие параметры
(exc/tau и exc/I_ext); 'in->exc/gsyn_max' со старта не сдвигается.
Целевая траектория — прогон при параметрах конструктора, так что θ*
известен априори; старт — сдвинутый phi0.

Точность задач повышена до float64 через l0_factory (dtype задаётся до
построения сети); малая сеть и короткий горизонт держат тесты быстрыми.
"""

import json

import numpy as np
import pytest
import scipy
import tensorflow as tf

import neuraltide.config
from neuraltide.optimization import (
    CalibrationProblem,
    Experiment,
    FrozenProblem,
    ObservationSpec,
    ResidualLoss,
)
from neuraltide.optimization.local import least_squares, minimize_lbfgsb
from neuraltide.optimization.results import LocalOptimizerResult
from neuraltide.optimization.sensitivities import residual_jacobian

# Robust tolerances: float64 + гладкая L0-задача, восстановление θ*.
RTOL_THETA = 2e-2
ATOL_THETA = 2e-2


def _scipy_at_least(major: int, minor: int) -> bool:
    parts = scipy.__version__.split(".")
    try:
        return (int(parts[0]), int(parts[1])) >= (major, minor)
    except (IndexError, ValueError):
        return False


def _problem64(l0_factory) -> CalibrationProblem:
    neuraltide.config.set_dtype(tf.float64)
    network, space, t_seq, inputs, loss = l0_factory()
    return CalibrationProblem(network, space, t_seq, inputs=inputs, objective=loss)


def _offsets(problem) -> dict:
    offsets = {}
    start = 0
    for name, entry in problem.parameter_space.entries.items():
        offsets[name] = (start, entry.numel)
        start += entry.numel
    return offsets


def _shifted_start(problem) -> np.ndarray:
    """Сдвинутый от θ* старт (внутри bounds): tau=20, I_ext=0.5, gsyn=0.2."""
    phi = np.asarray(problem.pack(), dtype=np.float64).reshape(-1).copy()
    off = _offsets(problem)
    phi[off["exc/tau"][0]] = 20.0
    phi[off["exc/I_ext"][0]] = 0.5
    phi[off["in->exc/gsyn_max"][0]] = 0.2
    return phi


class _EvalStub:
    def __init__(self, loss):
        self.loss = loss


class _NonFiniteProblem:
    """Duck-typed problem с нефинитной потерей/невязкой (без TensorFlow)."""

    size = 1
    held_out = []
    n_forward = 0
    n_gradient = 0

    def pack(self, phi=None):
        return np.zeros(1)

    def bounds_phi(self):
        return np.array([-np.inf]), np.array([np.inf])

    def unpack(self, phi):
        return {"x": np.asarray(phi, dtype=np.float64)}

    def value_and_gradient(self, phi):
        return _EvalStub(np.nan), np.array([np.nan])

    def residual(self, phi):
        return np.array([np.nan])


class TestLbfgsb:
    def test_recovers_known_theta(self, l0_factory):
        problem = _problem64(l0_factory)
        res = minimize_lbfgsb(problem, _shifted_start(problem))

        assert res.success
        assert res.method == "L-BFGS-B"
        theta = res.base.theta_best
        np.testing.assert_allclose(theta["exc/tau"], [8.0], rtol=RTOL_THETA)
        np.testing.assert_allclose(theta["exc/I_ext"], [0.8], atol=ATOL_THETA)
        # e_r=0: вход не влияет → gsyn_max не идентифицируем и не сдвигается.
        np.testing.assert_allclose(
            np.asarray(theta["in->exc/gsyn_max"]).reshape(-1), [0.2], rtol=1e-6
        )

        assert res.base.loss_best < 1e-8
        assert res.base.seed is None
        assert res.base.dtype == "float64"
        assert len(res.base.history) > 1
        assert res.base.history[0]["nit"] == 0
        assert res.n_forward > 0
        assert res.n_gradient > 0
        assert res.nit > 0 and res.nfev > 0 and res.njev > 0
        assert res.grad_norm < 1e-3
        assert res.bounds_active == []
        assert res.wall_time > 0.0

    def test_phi0_none_uses_current_model(self, l0_factory):
        problem = _problem64(l0_factory)
        res = minimize_lbfgsb(problem)  # старт = θ* (параметры конструктора)
        assert res.success
        assert res.base.loss_best < 1e-10

    def test_options_recorded_in_config(self, l0_factory):
        problem = _problem64(l0_factory)
        res = minimize_lbfgsb(
            problem, _shifted_start(problem), maxiter=50, options={"maxcor": 5}
        )
        opts = res.base.config["options"]
        assert opts["maxiter"] == 50
        assert opts["maxcor"] == 5
        assert opts["gtol"] == 1e-6

    def test_deterministic_same_phi0(self, l0_factory):
        problem = _problem64(l0_factory)
        phi0 = _shifted_start(problem)
        r1 = minimize_lbfgsb(problem, phi0, maxiter=200)
        r2 = minimize_lbfgsb(problem, phi0.copy(), maxiter=200)
        np.testing.assert_array_equal(r1.base.phi_best, r2.base.phi_best)
        assert r1.base.loss_best == r2.base.loss_best
        assert r1.n_forward == r2.n_forward

    def test_nan_phi0_rejected(self, l0_factory):
        problem = _problem64(l0_factory)
        bad = _shifted_start(problem)
        bad[0] = np.nan
        with pytest.raises(ValueError, match="phi0"):
            minimize_lbfgsb(problem, bad)

    def test_nonfinite_start_loss_rejected(self):
        with pytest.raises(ValueError, match="loss"):
            minimize_lbfgsb(_NonFiniteProblem())

    def test_counters_reported_as_delta(self, l0_problem):
        res = minimize_lbfgsb(l0_problem, _shifted_start(l0_problem))
        assert res.n_forward > 0
        assert res.n_gradient > 0
        assert res.n_forward <= l0_problem.n_forward
        assert res.n_gradient <= l0_problem.n_gradient
        # held-out нет → validation пустой, счётчики задачи выросли.
        assert res.validation == {}


class TestLeastSquares:
    def test_trf_recovers_known_theta(self, l0_factory):
        problem = _problem64(l0_factory)
        res = least_squares(problem, _shifted_start(problem))

        assert res.method == "TRF"
        assert res.success
        theta = res.base.theta_best
        np.testing.assert_allclose(theta["exc/tau"], [8.0], rtol=RTOL_THETA)
        np.testing.assert_allclose(theta["exc/I_ext"], [0.8], atol=ATOL_THETA)
        np.testing.assert_allclose(
            np.asarray(theta["in->exc/gsyn_max"]).reshape(-1), [0.2], rtol=1e-6
        )

    def test_half_norm_sq_equals_loss_at_solution(self, l0_factory):
        problem = _problem64(l0_factory)
        res = least_squares(problem, _shifted_start(problem))
        r_star = problem.residual(res.base.phi_best)
        half_norm = 0.5 * float(np.dot(r_star, r_star))
        assert half_norm == pytest.approx(res.base.loss_best, rel=1e-8, abs=1e-12)
        eval_best = problem.evaluate(res.base.phi_best)
        assert eval_best.loss == pytest.approx(res.base.loss_best, rel=1e-8)

    def test_trf_three_point_jacobian(self, l0_factory):
        problem = _problem64(l0_factory)
        res = least_squares(problem, _shifted_start(problem), jacobian="3-point")
        assert res.success
        np.testing.assert_allclose(
            res.base.theta_best["exc/tau"], [8.0], rtol=RTOL_THETA
        )

    def test_lm_rejects_bounds_with_trf_hint(self, l0_factory):
        problem = _problem64(l0_factory)
        with pytest.raises(ValueError, match="trf"):
            least_squares(problem, _shifted_start(problem), method="lm")

    def test_lm_works_without_bounds(self, l0_factory):
        problem = _problem64(l0_factory)
        frozen = FrozenProblem(problem, free_names=["exc/I_ext"])
        res = least_squares(frozen, np.array([0.5]), method="lm")
        assert res.method == "LM"
        assert res.success
        # FrozenProblem без контракта sensitivities → честный FD-fallback.
        assert res.base.config["jacobian"] == "2-point"
        np.testing.assert_allclose(
            res.base.theta_best["exc/I_ext"], [0.8], atol=ATOL_THETA
        )

    def test_objective_without_residual_raises_typeerror(self, l0_problem):
        view = l0_problem.with_objective(lambda preds, net: 0.0)
        with pytest.raises(TypeError, match="minimize_lbfgsb"):
            least_squares(view, np.asarray(view.pack(), dtype=np.float64))

    def test_nan_phi0_rejected(self, l0_factory):
        problem = _problem64(l0_factory)
        bad = _shifted_start(problem)
        bad[0] = np.nan
        with pytest.raises(ValueError, match="phi0"):
            least_squares(problem, bad)

    def test_nonfinite_start_residual_rejected(self):
        with pytest.raises(ValueError, match="residual"):
            least_squares(_NonFiniteProblem())

    def test_deterministic_same_phi0(self, l0_factory):
        problem = _problem64(l0_factory)
        phi0 = _shifted_start(problem)
        s1 = least_squares(problem, phi0, max_nfev=80)
        s2 = least_squares(problem, phi0.copy(), max_nfev=80)
        np.testing.assert_array_equal(s1.base.phi_best, s2.base.phi_best)
        assert s1.base.loss_best == s2.base.loss_best

    def test_fd_jacobian_does_not_use_analytic_gradient(self, l0_factory):
        problem = _problem64(l0_factory)
        res = least_squares(problem, _shifted_start(problem), jacobian="2-point")
        assert res.n_forward > 0
        # Конечно-разностный якобиан не обращается к analytic gradient.
        assert res.n_gradient == 0

    def test_callable_jacobian_via_sensitivities(self, l0_factory):
        """Реальная интеграция: callable = sensitivities.residual_jacobian.

        Адаптер обязан взять поле .J из JacobianResult (а не заворачивать
        DTO в np.asarray) — иначе scipy получает не матрицу и падает.
        Проверяются успех, форма якобиана [n_residuals, size] и качество
        (восстановление θ*; 0.5·||r*||² == loss).
        """
        problem = _problem64(l0_factory)
        phi0 = _shifted_start(problem)
        r0 = np.asarray(problem.residual(phi0), dtype=np.float64)

        res = least_squares(problem, phi0, jacobian=residual_jacobian)

        assert res.success
        assert res.njev > 0
        # Форма якобиана реального модуля: [n_residuals, size].
        jac0 = residual_jacobian(problem, phi0)
        assert jac0.J.shape == (r0.size, problem.size)
        # Качество: θ* восстановлены.
        np.testing.assert_allclose(
            res.base.theta_best["exc/tau"], [8.0], rtol=RTOL_THETA
        )
        np.testing.assert_allclose(
            res.base.theta_best["exc/I_ext"], [0.8], atol=ATOL_THETA
        )
        # 0.5·||r*||² == loss_best (контракт residual-интерфейса).
        r_star = np.asarray(problem.residual(res.base.phi_best), dtype=np.float64)
        assert 0.5 * float(r_star @ r_star) == pytest.approx(
            res.base.loss_best, rel=1e-8, abs=1e-12
        )

    def test_callable_jacobian_without_sensitivities(self, l0_factory):
        problem = _problem64(l0_factory)

        def fd_jac(x, eps=1e-7):
            x = np.asarray(x, dtype=np.float64)
            r0 = problem.residual(x)
            jac = np.empty((r0.size, x.size))
            for j in range(x.size):
                step = np.zeros_like(x)
                step[j] = eps
                rp = problem.residual(x + step)
                rm = problem.residual(x - step)
                jac[:, j] = (rp - rm) / (2.0 * eps)
            return jac

        res = least_squares(problem, _shifted_start(problem), jacobian=fd_jac)
        assert res.success
        np.testing.assert_allclose(
            res.base.theta_best["exc/tau"], [8.0], rtol=RTOL_THETA
        )

    def test_unknown_method_and_jacobian_rejected(self, l0_factory):
        problem = _problem64(l0_factory)
        phi0 = _shifted_start(problem)
        with pytest.raises(ValueError, match="method"):
            least_squares(problem, phi0, method="dogbox")
        with pytest.raises(ValueError, match="jacobian"):
            least_squares(problem, phi0, jacobian="cs")

    def test_default_small_problem_uses_forward_jacobian(self, l0_factory):
        """Дефолт (jacobian=None, size <= 200): forward через sensitivities."""
        problem = _problem64(l0_factory)
        res = least_squares(problem, _shifted_start(problem))

        assert res.base.config["jacobian"] == "forward"
        assert res.base.config["jacobian_max_params"] == 200
        # Forward-якобиан — JVP-прогоны без analytic gradient.
        assert res.n_gradient == 0
        assert res.njev > 0
        assert res.success
        np.testing.assert_allclose(
            res.base.theta_best["exc/tau"], [8.0], rtol=RTOL_THETA
        )
        np.testing.assert_allclose(
            res.base.theta_best["exc/I_ext"], [0.8], atol=ATOL_THETA
        )

    def test_default_jacobian_falls_back_to_2point_over_cap(self, l0_factory):
        """size > jacobian_max_params: default честно переключается на FD."""
        problem = _problem64(l0_factory)
        res = least_squares(problem, _shifted_start(problem), jacobian_max_params=1)

        assert res.base.config["jacobian"] == "2-point"
        assert res.n_gradient == 0
        assert res.success
        np.testing.assert_allclose(
            res.base.theta_best["exc/tau"], [8.0], rtol=RTOL_THETA
        )

    def test_user_jacobian_modes_recorded_in_config(self, l0_factory):
        """Пользовательские строки/callable сохраняют приоритет; режим в config."""
        problem = _problem64(l0_factory)
        phi0 = _shifted_start(problem)

        r_str = least_squares(problem, phi0.copy(), jacobian="3-point")
        assert r_str.base.config["jacobian"] == "3-point"
        assert r_str.success

        r_call = least_squares(problem, phi0.copy(), jacobian=residual_jacobian)
        assert r_call.base.config["jacobian"] == "callable"
        assert r_call.success

    @pytest.mark.skipif(
        not _scipy_at_least(1, 16),
        reason="callback у least_squares поддержан только в scipy >= 1.16",
    )
    def test_callback_and_per_iteration_history_on_modern_scipy(self, l0_factory):
        problem = _problem64(l0_factory)
        calls = []

        def cb(intermediate_result):
            calls.append(intermediate_result)

        res = least_squares(problem, _shifted_start(problem), callback=cb)

        assert len(calls) > 0
        assert all(hasattr(ir, "cost") for ir in calls)
        assert len(res.base.history) > 1
        # История дополняется по итерациям; финальная запись — потеря решения.
        assert res.base.history[-1]["loss"] == pytest.approx(
            res.base.loss_best, rel=1e-12
        )

    def test_callback_safe_path_on_legacy_scipy(self, l0_factory, monkeypatch):
        """Симуляция scipy 1.15: callback не передаётся, оптимизация не падает."""
        monkeypatch.setattr(scipy, "__version__", "1.15.0")
        problem = _problem64(l0_factory)
        calls = []

        with pytest.warns(UserWarning, match="callback"):
            res = least_squares(
                problem,
                _shifted_start(problem),
                callback=lambda intermediate_result: calls.append(intermediate_result),
            )

        assert res.success
        assert calls == []  # callback не вызван
        # История без per-iteration callback: стартовая + финальная записи.
        assert len(res.base.history) == 2
        assert res.base.history[-1]["loss"] == pytest.approx(
            res.base.loss_best, rel=1e-12
        )

    @pytest.mark.skipif(
        not _scipy_at_least(1, 16),
        reason="callback у least_squares поддержан только в scipy >= 1.16",
    )
    def test_callback_with_lm_warns_and_is_ignored(self, l0_factory):
        """C3 (мелко): на scipy >= 1.16 callback с method='lm' молча не
        передавался бы в scipy — адаптер обязан предупредить UserWarning."""
        problem = _problem64(l0_factory)
        frozen = FrozenProblem(problem, free_names=["exc/I_ext"])
        calls = []

        with pytest.warns(UserWarning, match="callback"):
            res = least_squares(
                frozen,
                np.array([0.5]),
                method="lm",
                callback=lambda intermediate_result: calls.append(intermediate_result),
            )

        assert res.method == "LM"
        assert res.success
        assert calls == []  # callback не вызван (lm его не поддерживает)
        # История без per-iteration callback: стартовая + финальная записи.
        assert len(res.base.history) == 2
        assert res.base.history[-1]["loss"] == pytest.approx(
            res.base.loss_best, rel=1e-12
        )

    # -- maxiter: унифицированный бюджет итераций (C2) --------------------

    def test_maxiter_maps_to_max_nfev_budget(self, l0_factory):
        """maxiter без max_nfev: бюджет маппируется max_nfev=maxiter*(n+2)."""
        problem = _problem64(l0_factory)
        res = least_squares(problem, _shifted_start(problem), maxiter=10)

        size = int(problem.size)
        assert res.base.config["maxiter"] == 10
        assert res.base.config["max_nfev"] == 10 * (size + 2)
        # Записанный бюджет — эффективный; scipy получил ровно его.
        assert res.nfev <= 10 * (size + 2)
        assert res.success
        np.testing.assert_allclose(
            res.base.theta_best["exc/tau"], [8.0], rtol=RTOL_THETA
        )

    def test_maxiter_and_max_nfev_mutually_exclusive(self, l0_factory):
        problem = _problem64(l0_factory)
        with pytest.raises(ValueError, match="maxiter"):
            least_squares(problem, _shifted_start(problem), maxiter=5, max_nfev=100)

    def test_maxiter_must_be_positive(self, l0_factory):
        problem = _problem64(l0_factory)
        with pytest.raises(ValueError, match="maxiter"):
            least_squares(problem, _shifted_start(problem), maxiter=0)

    def test_maxiter_none_keeps_legacy_unlimited_budget(self, l0_factory):
        # Совместимость: без maxiter бюджет по-прежнему без ограничения.
        problem = _problem64(l0_factory)
        res = least_squares(problem, _shifted_start(problem))
        assert res.base.config["maxiter"] is None
        assert res.base.config["max_nfev"] is None
        assert res.success

    # -- fallback forward -> '2-point' после gradient-прогонов (C3) -------

    def test_default_jacobian_falls_back_after_gradient_runs(self, l0_factory):
        """value_and_gradient -> least_squares(jacobian=None) на том же
        NetworkRNN: forward-AD несовместим — честный fallback на '2-point'
        с UserWarning, фактический режим зафиксирован в config."""
        problem = _problem64(l0_factory)
        phi0 = _shifted_start(problem)
        problem.value_and_gradient(phi0)  # BPTT-прогон на этом NetworkRNN
        assert problem.n_gradient > 0

        with pytest.warns(UserWarning, match="forward"):
            res = least_squares(problem, phi0)

        assert res.base.config["jacobian"] == "2-point"
        assert res.success
        np.testing.assert_allclose(
            res.base.theta_best["exc/tau"], [8.0], rtol=RTOL_THETA
        )
        np.testing.assert_allclose(
            res.base.theta_best["exc/I_ext"], [0.8], atol=ATOL_THETA
        )


class TestBoundsActive:
    @staticmethod
    def _clamped_problem(l0_factory) -> CalibrationProblem:
        """Цель при tau=200 (вне bounds [1, 50]) → оптимум зажат в tau=50."""
        neuraltide.config.set_dtype(tf.float64)
        network, space, t_seq, inputs, _ = l0_factory()
        tau_var = space.entries["exc/tau"].variable
        tau_var.assign([200.0])
        target_slow = network(t_seq, inputs=inputs).firing_rates["exc"]
        tau_var.assign([20.0])
        loss = ResidualLoss(
            [
                ObservationSpec(
                    name="obs", population="exc", target=target_slow, sigma=2.0
                )
            ]
        )
        return CalibrationProblem(network, space, t_seq, inputs=inputs, objective=loss)

    def test_clamped_solution_reports_bounds(self, l0_factory):
        problem = self._clamped_problem(l0_factory)
        phi0 = np.asarray(problem.pack(), dtype=np.float64).copy()
        res = minimize_lbfgsb(problem, phi0)
        assert np.isclose(res.base.theta_best["exc/tau"][0], 50.0, rtol=1e-4)
        assert res.bounds_active == ["exc/tau"]

    def test_trf_clamped_solution_reports_bounds(self, l0_factory):
        problem = self._clamped_problem(l0_factory)
        phi0 = np.asarray(problem.pack(), dtype=np.float64).copy()
        res = least_squares(problem, phi0)
        assert np.isclose(res.base.theta_best["exc/tau"][0], 50.0, rtol=1e-4)
        assert res.bounds_active == ["exc/tau"]


class TestValidationAndSerialization:
    @staticmethod
    def _held_out_problem(l0_factory) -> CalibrationProblem:
        problem = _problem64(l0_factory)
        target = problem.objective.observations[0].target
        held = Experiment(
            name="held",
            t_sequence=problem.t_sequence[:, :30],
            inputs=problem.inputs[:, :30],
            objective=ResidualLoss(
                [
                    ObservationSpec(
                        name="obs",
                        population="exc",
                        target=target[:, :30],
                        sigma=2.0,
                    )
                ]
            ),
        )
        problem.held_out.append(held)
        return problem

    def test_validate_true_populates_metrics(self, l0_factory):
        problem = self._held_out_problem(l0_factory)
        res = minimize_lbfgsb(problem, _shifted_start(problem), validate=True)
        assert set(res.validation) == {"held"}
        assert res.validation["held"] >= 0.0

    def test_validate_false_keeps_empty(self, l0_factory):
        problem = self._held_out_problem(l0_factory)
        res = minimize_lbfgsb(
            problem, _shifted_start(problem), validate=False, maxiter=5
        )
        assert res.validation == {}

    def test_serialization_roundtrip(self, l0_factory, tmp_path):
        problem = self._held_out_problem(l0_factory)
        res = least_squares(problem, _shifted_start(problem), validate=True)

        data = res.to_dict()
        json.dumps(data)
        restored = LocalOptimizerResult.from_dict(data)

        assert restored.method == res.method
        assert restored.success == res.success
        assert restored.status == res.status
        assert restored.nit == res.nit and restored.nfev == res.nfev
        assert restored.njev == res.njev
        assert restored.grad_norm == pytest.approx(res.grad_norm)
        assert restored.n_forward == res.n_forward
        assert restored.n_gradient == res.n_gradient
        assert restored.bounds_active == res.bounds_active
        assert restored.validation == pytest.approx(res.validation)
        np.testing.assert_array_equal(restored.base.phi_best, res.base.phi_best)
        for name in res.base.theta_best:
            np.testing.assert_array_equal(
                restored.base.theta_best[name], res.base.theta_best[name]
            )
        assert restored.base.loss_best == pytest.approx(res.base.loss_best)
        assert restored.base.seed is None
        assert restored.base.dtype == res.base.dtype
        assert restored.base.config == res.base.config

        path = tmp_path / "local_result.json"
        res.save(path)
        loaded = LocalOptimizerResult.load(path)
        assert loaded.method == res.method
        np.testing.assert_array_equal(loaded.base.phi_best, res.base.phi_best)
        assert loaded.validation == pytest.approx(res.validation)
