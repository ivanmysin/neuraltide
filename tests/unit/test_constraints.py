import json

import numpy as np
import pytest
import tensorflow as tf

import neuraltide.config
from neuraltide.constraints.param_constraints import (
    MinMaxConstraint,
)
from neuraltide.optimization import (
    CalibrationProblem,
    ObservationSpec,
    ResidualLoss,
)
from neuraltide.optimization.constraints import (
    ALResult,
    AugmentedLagrangianState,
    ConstrainedProblem,
    ConstraintSpec,
    augmented_lagrangian,
    epsilon_constraint,
    fd_jacobian,
)
from neuraltide.optimization.diagnostics import finite_difference_scan
from neuraltide.optimization.local import minimize_lbfgsb
from neuraltide.optimization.results import Evaluation

# AL-тесты: допуски float64-задач (L0) и duck-квадратичной задачи.
RTOL_GRAD = 1e-3
ATOL_GRAD = 1e-3


class TestMinMaxConstraint:
    def test_clip_full_bounds(self):
        constraint = MinMaxConstraint(0, 1)
        w = tf.constant([-1.0, 0.5, 2.0])
        result = constraint(w)
        expected = [0.0, 0.5, 1.0]
        assert tf.reduce_all(tf.abs(result - expected) < 1e-6)

    def test_clip_only_min(self):
        constraint = MinMaxConstraint(0.0, None)
        w = tf.constant([-1.0, 0.5, 2.0])
        result = constraint(w)
        expected = [0.0, 0.5, 2.0]
        assert tf.reduce_all(tf.abs(result - expected) < 1e-6)

    def test_clip_only_max(self):
        constraint = MinMaxConstraint(None, 1.0)
        w = tf.constant([-1.0, 0.5, 2.0])
        result = constraint(w)
        expected = [-1.0, 0.5, 1.0]
        assert tf.reduce_all(tf.abs(result - expected) < 1e-6)

    def test_get_config(self):
        constraint = MinMaxConstraint(0.1, 0.9)
        config = constraint.get_config()
        assert config["min_val"] == 0.1
        assert config["max_val"] == 0.9

    def test_from_config(self):
        config = {"min_val": 0.0, "max_val": 1.0}
        constraint = MinMaxConstraint.from_config(config)
        assert constraint.min_val == 0.0
        assert constraint.max_val == 1.0


# ---------------------------------------------------------------------------
# Duck-typed квадратичная задача: f(x) = 0.5·||x − target||², predictions = x
# ---------------------------------------------------------------------------


class _QuadraticDuck:
    """Duck-typed CalibrationProblem для AL-тестов без TensorFlow-прогонов."""

    network = None
    held_out = []

    def __init__(self, target):
        self._target = np.asarray(target, dtype=np.float64).reshape(-1)
        self._phi = self._target.copy()
        self.n_forward = 0
        self.n_gradient = 0
        self.wall_time = 0.0
        self.simulate_calls = 0

    @property
    def size(self):
        return int(self._target.shape[0])

    def pack(self, phi=None):
        if phi is None:
            return self._phi.copy()
        phi = np.asarray(phi, dtype=np.float64).reshape(-1)
        assert phi.shape[0] == self.size
        return phi

    def unpack(self, phi):
        return {"x": np.asarray(phi, dtype=np.float64).reshape(-1)}

    def bounds_phi(self):
        return (
            np.full(self.size, -np.inf, dtype=np.float64),
            np.full(self.size, np.inf, dtype=np.float64),
        )

    def simulate(self, phi):
        self.simulate_calls += 1
        return np.asarray(phi, dtype=np.float64).reshape(-1)

    def _x(self, phi):
        return self.pack(phi) - self._target

    def evaluate(self, phi=None):
        self.n_forward += 1
        x = self._x(phi)
        loss = float(0.5 * (x @ x))
        return Evaluation(
            loss=loss,
            components={"quad": loss},
            n_forward=self.n_forward,
            n_gradient=self.n_gradient,
            dtype="float64",
        )

    def value_and_gradient(self, phi=None):
        self.n_forward += 1
        self.n_gradient += 1
        x = self._x(phi)
        loss = float(0.5 * (x @ x))
        ev = Evaluation(
            loss=loss,
            components={"quad": loss},
            n_forward=self.n_forward,
            n_gradient=self.n_gradient,
            dtype="float64",
        )
        return ev, x.copy()


class _ClippedDuck(_QuadraticDuck):
    """Duck-задача с границами [0, 1] и недопустимой стартовой точкой."""

    def __init__(self, target, start):
        super().__init__(target)
        self._phi = np.asarray(start, dtype=np.float64).reshape(-1)

    def bounds_phi(self):
        return (
            np.zeros(self.size, dtype=np.float64),
            np.ones(self.size, dtype=np.float64),
        )


def _linear_eq_spec(a, b):
    a = np.asarray(a, dtype=np.float64)
    return ConstraintSpec(
        name="lin",
        kind="eq",
        fn=lambda preds, net: float(a @ preds - b),
        jacobian=lambda phi: a[None, :],
    )


_QUAD_TARGET = np.array([1.0, -2.0, 0.5])
_QUAD_A = np.array([1.0, 1.0, 1.0])
_QUAD_B = 1.0


class TestConstraintSpec:
    def test_kind_validation(self):
        with pytest.raises(ValueError, match="kind"):
            ConstraintSpec(name="c", kind="le", fn=lambda p, n: 0.0)

    def test_fn_must_be_callable(self):
        with pytest.raises(TypeError, match="callable"):
            ConstraintSpec(name="c", kind="eq", fn=3.0)

    def test_jacobian_validation(self):
        with pytest.raises(ValueError, match="fd"):
            ConstraintSpec(name="c", kind="eq", fn=lambda p, n: 0.0, jacobian="2-point")
        with pytest.raises(TypeError, match="jacobian"):
            ConstraintSpec(name="c", kind="eq", fn=lambda p, n: 0.0, jacobian=1.5)

    def test_constrained_problem_requires_specs(self):
        problem = _QuadraticDuck([1.0, 2.0])
        with pytest.raises(ValueError, match="ConstraintSpec"):
            ConstrainedProblem(problem, [])
        spec = ConstraintSpec(name="c", kind="eq", fn=lambda p, n: 0.0)
        with pytest.raises(TypeError, match="ConstraintSpec"):
            ConstrainedProblem(problem, [spec, "not-a-spec"])
        with pytest.raises(ValueError, match="unique"):
            ConstrainedProblem(problem, [spec, spec])


class TestAugmentedLagrangianState:
    def test_eq_update(self):
        state = AugmentedLagrangianState(lambdas={"c": np.array([0.0])}, mu=10.0)
        v = state.update({"c": np.array([0.5])}, {"c": "eq"})
        assert v == pytest.approx(0.5)
        assert state.lambdas["c"][0] == pytest.approx(-5.0)
        # первый update: prev_violation None → mu без изменений
        assert state.mu == pytest.approx(10.0)

    def test_ineq_update_clips_at_zero(self):
        state = AugmentedLagrangianState(lambdas={"g": np.array([1.0])}, mu=10.0)
        v = state.update({"g": np.array([-0.5])}, {"g": "ineq"})
        assert v == pytest.approx(0.0)  # g <= 0 выполнено
        assert state.lambdas["g"][0] == pytest.approx(0.0)  # max(0, 1 − 5)

    def test_mu_increases_when_violation_not_reduced(self):
        state = AugmentedLagrangianState(lambdas={"c": np.array([0.0])}, mu=10.0)
        state.update({"c": np.array([1.0])}, {"c": "eq"}, eta=0.25)
        v = state.update({"c": np.array([0.9])}, {"c": "eq"}, eta=0.25)
        assert v == pytest.approx(0.9)
        assert state.mu == pytest.approx(100.0)  # 0.9 > 0.25·1.0

    def test_serialization(self):
        state = AugmentedLagrangianState(
            lambdas={"c": np.array([0.5, -0.25])},
            mu=20.0,
            mu_factor=5.0,
            prev_violation=0.1,
        )
        restored = AugmentedLagrangianState.from_dict(state.to_dict())
        assert restored.mu == pytest.approx(state.mu)
        assert restored.mu_factor == pytest.approx(state.mu_factor)
        assert restored.prev_violation == pytest.approx(0.1)
        for name in state.lambdas:
            np.testing.assert_array_equal(restored.lambdas[name], state.lambdas[name])


class TestFdJacobian:
    def test_central_differences_match_analytic(self):
        def value(x):
            return np.array([x[0] ** 2 + x[1], np.sin(x[1])])

        phi = np.array([0.7, -0.3])
        jac = fd_jacobian(value, phi, h=1e-5)
        expected = np.array([[2 * 0.7, 1.0], [0.0, np.cos(-0.3)]])
        np.testing.assert_allclose(jac, expected, atol=1e-6)

    def test_scan_selects_plateau(self):
        def value(x):
            return np.array([np.exp(x[0])])

        phi = np.array([0.4])
        jac = fd_jacobian(value, phi, h_values=np.logspace(-2, -8, 7))
        assert jac[0, 0] == pytest.approx(np.exp(0.4), rel=1e-6)

    def test_budget_documented_in_contract(self):
        # 2·n + 1 оценок value_fn при фиксированном h.
        calls = {"n": 0}

        def value(x):
            calls["n"] += 1
            return np.array([x[0] + x[1] ** 2])

        phi = np.array([0.3, -0.7])
        jac = fd_jacobian(value, phi, h=1e-4)
        assert jac.shape == (1, 2)
        assert calls["n"] == 2 * 2 + 1


class TestConstrainedProblemDuck:
    def test_duck_api_surface(self):
        problem = _QuadraticDuck([1.0, 2.0, 3.0])
        spec = ConstraintSpec(name="c", kind="eq", fn=lambda p, n: 0.0)
        cp = ConstrainedProblem(problem, [spec])
        assert cp.size == 3
        for method in (
            "pack",
            "unpack",
            "bounds_phi",
            "evaluate",
            "value_and_gradient",
            "residual",
            "validate",
            "augmented_terms",
        ):
            assert callable(getattr(cp, method)), method

    def test_fd_jacobian_gradient_matches_fd_of_loss(self):
        problem = _QuadraticDuck([1.0, 1.0, 0.5])
        spec = ConstraintSpec(
            name="ball",
            kind="ineq",
            fn=lambda preds, net: float(preds @ preds - 2.0),
            jacobian="fd",
        )
        cp = ConstrainedProblem(problem, [spec])
        phi0 = np.array([1.2, 0.9, 0.6])  # g(phi0) = 0.61 > 0 — активно
        eval_res, grad = cp.value_and_gradient(phi0)
        assert eval_res.loss == pytest.approx(
            eval_res.components["quad"] + eval_res.components["al/penalty"]
        )
        scan = finite_difference_scan(lambda p: cp.evaluate(p).loss, phi0)
        np.testing.assert_allclose(
            np.asarray(grad, dtype=np.float64),
            scan["best"],
            rtol=RTOL_GRAD,
            atol=ATOL_GRAD,
        )

    def test_fd_budget_one_forward_per_constraint_value(self):
        problem = _QuadraticDuck([1.0, 1.0, 0.5])
        spec = ConstraintSpec(
            name="ball",
            kind="ineq",
            fn=lambda preds, net: float(preds @ preds - 2.0),
            jacobian="fd",
        )
        cp = ConstrainedProblem(problem, [spec])
        phi0 = np.array([1.2, 0.9, 0.6])
        n_sim = problem.simulate_calls
        n_fwd = problem.n_forward
        cp.value_and_gradient(phi0)
        # 1 прогон на базовое значение + 2·n для центральных разностей,
        # при этом data-loss прогон ровно один.
        assert problem.simulate_calls - n_sim == 1 + 2 * problem.size
        assert problem.n_forward - n_fwd == 1

    def test_missing_jacobian_is_explicit_error(self):
        problem = _QuadraticDuck([1.0, 1.0, 0.5])
        spec = ConstraintSpec(
            name="ball", kind="ineq", fn=lambda preds, net: float(preds @ preds)
        )
        cp = ConstrainedProblem(problem, [spec])
        with pytest.raises(TypeError, match="jacobian"):
            cp.value_and_gradient(np.array([1.2, 0.9, 0.6]))

    def test_augmented_terms_values_and_violations(self):
        problem = _QuadraticDuck([1.0, 1.0, 0.5])
        spec = ConstraintSpec(
            name="ball",
            kind="ineq",
            fn=lambda preds, net: preds @ preds - 2.0,
            jacobian="fd",
        )
        cp = ConstrainedProblem(problem, [spec])
        phi0 = np.array([1.2, 0.9, 0.6])
        terms = cp.augmented_terms(phi0)
        value, violation = terms["ball"]
        assert value.shape == (1,)
        np.testing.assert_allclose(violation, np.maximum(value, 0.0))


class TestAugmentedLagrangianQuadratic:
    def test_eq_constraint_converges(self):
        problem = _QuadraticDuck(_QUAD_TARGET)
        result = augmented_lagrangian(
            problem, [_linear_eq_spec(_QUAD_A, _QUAD_B)], max_outer=8
        )
        assert result.success
        # точное решение с конвенцией AL −λc + μc²/2:
        # x* = t + λ·a, λ* = (b − aᵀt)/||a||².
        lam_exact = (_QUAD_B - _QUAD_A @ _QUAD_TARGET) / (_QUAD_A @ _QUAD_A)
        assert result.state.lambdas["lin"][0] == pytest.approx(lam_exact, abs=1e-4)
        x_exact = _QUAD_TARGET + lam_exact * _QUAD_A
        np.testing.assert_allclose(result.base.phi_best, x_exact, atol=1e-4)
        np.testing.assert_allclose(result.base.theta_best["x"], x_exact, atol=1e-4)

        violations = [rec["violation"] for rec in result.violation_history]
        assert len(violations) == 8
        # Строгое убывание на шумо-свободном префиксе (1/31 за итерацию);
        # дальше последовательность упирается в точность внутреннего
        # решателя (gtol), поэтому проверяем итоговое сокращение в eta раз.
        assert all(
            v_next < v_prev for v_prev, v_next in zip(violations[:4], violations[1:4])
        )
        assert violations[-1] < 0.25 * violations[0]
        assert violations[-1] < 1e-6
        assert result.n_outer == 8
        # loss_best — data loss исходной задачи: 0.5·λ²·||a||² в x*.
        assert result.base.loss_best == pytest.approx(
            0.5 * lam_exact**2 * (_QUAD_A @ _QUAD_A), abs=1e-6
        )

    def test_ineq_constraint_hits_boundary(self):
        problem = _QuadraticDuck(_QUAD_TARGET)
        spec = ConstraintSpec(
            name="cap",
            kind="ineq",
            fn=lambda preds, net: float(preds[0] - 0.2),
            jacobian=lambda phi: np.array([[1.0, 0.0, 0.0]]),
        )
        result = augmented_lagrangian(problem, [spec], max_outer=8)
        assert result.success
        assert result.base.phi_best[0] == pytest.approx(0.2, abs=1e-4)
        np.testing.assert_allclose(
            result.base.phi_best[1:], _QUAD_TARGET[1:], atol=1e-4
        )
        assert result.state.lambdas["cap"][0] == pytest.approx(0.8, abs=1e-3)
        violations = [rec["violation"] for rec in result.violation_history]
        assert violations[-1] < 1e-6

    def test_config_and_history(self):
        problem = _QuadraticDuck(_QUAD_TARGET)
        result = augmented_lagrangian(
            problem, [_linear_eq_spec(_QUAD_A, _QUAD_B)], max_outer=3, seed=11
        )
        assert result.base.seed == 11
        assert result.base.config["local"] == "lbfgsb"
        assert result.base.config["max_outer"] == 3
        assert result.base.config["mu0"] == pytest.approx(10.0)
        assert result.base.config["eta"] == pytest.approx(0.25)
        assert result.base.config["constraints"] == [{"name": "lin", "kind": "eq"}]
        assert result.base.config["projected_start"] is False
        assert all(entry["outer"] in (1, 2, 3) for entry in result.base.history)
        assert all("constraint/lin" in rec for rec in result.violation_history)

    def test_callable_local(self):
        problem = _QuadraticDuck(_QUAD_TARGET)
        result = augmented_lagrangian(
            problem,
            [_linear_eq_spec(_QUAD_A, _QUAD_B)],
            local=lambda cp, phi: minimize_lbfgsb(cp, phi, maxiter=100),
            max_outer=3,
        )
        assert result.n_outer == 3
        assert np.all(np.isfinite(result.base.phi_best))


class TestInvalidStart:
    def test_start_outside_bounds_projected_with_warning(self):
        problem = _ClippedDuck([5.0, 5.0, 5.0], start=[5.0, -1.0, 0.5])
        spec = ConstraintSpec(
            name="cap",
            kind="ineq",
            fn=lambda preds, net: float(preds[0] - 0.2),
            jacobian=lambda phi: np.array([[1.0, 0.0, 0.0]]),
        )
        with pytest.warns(UserWarning, match="bounds"):
            result = augmented_lagrangian(problem, [spec], max_outer=2)
        assert result.base.config["projected_start"] is True
        assert np.all(result.base.phi_best >= 0.0)
        assert np.all(result.base.phi_best <= 1.0)

    def test_nonfinite_start_rejected(self):
        problem = _ClippedDuck([5.0, 5.0, 5.0], start=[np.nan, 0.0, 0.0])
        spec = ConstraintSpec(name="cap", kind="ineq", fn=lambda preds, net: 0.0)
        with pytest.raises(ValueError, match="phi"):
            augmented_lagrangian(problem, [spec], max_outer=1)

    def test_bad_args_rejected(self):
        problem = _QuadraticDuck(_QUAD_TARGET)
        spec = _linear_eq_spec(_QUAD_A, _QUAD_B)
        with pytest.raises(ValueError, match="max_outer"):
            augmented_lagrangian(problem, [spec], max_outer=0)
        with pytest.raises(ValueError, match="mu0"):
            augmented_lagrangian(problem, [spec], mu0=0.0)
        with pytest.raises(ValueError, match="eta"):
            augmented_lagrangian(problem, [spec], eta=1.5)
        with pytest.raises(ValueError, match="ConstraintSpec"):
            augmented_lagrangian(problem, [], max_outer=1)


class TestCheckpointResume:
    def test_resume_matches_continuous_run(self, tmp_path):
        path = tmp_path / "al_checkpoint.json"
        spec = _linear_eq_spec(_QUAD_A, _QUAD_B)

        partial = augmented_lagrangian(
            _QuadraticDuck(_QUAD_TARGET),
            [spec],
            max_outer=1,
            checkpoint_path=path,
            seed=7,
        )
        assert path.exists()
        assert partial.n_outer == 1

        resumed = augmented_lagrangian(
            _QuadraticDuck(_QUAD_TARGET),
            [spec],
            max_outer=3,
            checkpoint_path=path,
            seed=7,
            resume=True,
        )
        full = augmented_lagrangian(
            _QuadraticDuck(_QUAD_TARGET), [spec], max_outer=3, seed=7
        )
        assert resumed.n_outer == 3
        np.testing.assert_allclose(
            resumed.base.phi_best, full.base.phi_best, atol=1e-10
        )
        assert [r["violation"] for r in resumed.violation_history] == (
            pytest.approx([r["violation"] for r in full.violation_history])
        )
        np.testing.assert_allclose(
            resumed.state.lambdas["lin"],
            full.state.lambdas["lin"],
            atol=1e-12,
        )

    def test_resume_finished_run_success_from_violation_history(self, tmp_path):
        """Resume уже завершённого прогона (start_outer == max_outer):
        внутренних запусков нет, success оценивается по итоговой
        violation_history, а не принудительно даётся False из-за пустого
        списка inners."""
        path = tmp_path / "al_finished_checkpoint.json"
        spec = _linear_eq_spec(_QUAD_A, _QUAD_B)
        full = augmented_lagrangian(
            _QuadraticDuck(_QUAD_TARGET), [spec], max_outer=3, seed=7
        )
        assert full.success
        augmented_lagrangian(
            _QuadraticDuck(_QUAD_TARGET),
            [spec],
            max_outer=3,
            seed=7,
            checkpoint_path=path,
        )

        resumed = augmented_lagrangian(
            _QuadraticDuck(_QUAD_TARGET),
            [spec],
            max_outer=3,
            seed=7,
            checkpoint_path=path,
            resume=True,
        )
        # Внешние итерации не повторялись; результат — итог чекпоинта.
        assert resumed.n_outer == 3
        assert len(resumed.violation_history) == 3
        assert resumed.success
        np.testing.assert_allclose(
            resumed.base.phi_best, full.base.phi_best, atol=1e-10
        )
        assert [r["violation"] for r in resumed.violation_history] == (
            pytest.approx([r["violation"] for r in full.violation_history])
        )

    def test_resume_seed_mismatch_rejected(self, tmp_path):
        path = tmp_path / "al_checkpoint.json"
        spec = _linear_eq_spec(_QUAD_A, _QUAD_B)
        augmented_lagrangian(
            _QuadraticDuck(_QUAD_TARGET),
            [spec],
            max_outer=1,
            checkpoint_path=path,
            seed=7,
        )
        with pytest.raises(ValueError, match="seed"):
            augmented_lagrangian(
                _QuadraticDuck(_QUAD_TARGET),
                [spec],
                max_outer=2,
                checkpoint_path=path,
                seed=8,
                resume=True,
            )


class TestALResultSerialization:
    def test_roundtrip(self, tmp_path):
        problem = _QuadraticDuck(_QUAD_TARGET)
        result = augmented_lagrangian(
            problem, [_linear_eq_spec(_QUAD_A, _QUAD_B)], max_outer=2, seed=3
        )
        data = result.to_dict()
        json.dumps(data)
        restored = ALResult.from_dict(data)

        assert restored.n_outer == result.n_outer
        assert restored.success == result.success
        assert restored.validation == result.validation
        np.testing.assert_array_equal(restored.base.phi_best, result.base.phi_best)
        for name in result.base.theta_best:
            np.testing.assert_array_equal(
                restored.base.theta_best[name], result.base.theta_best[name]
            )
        assert restored.base.loss_best == pytest.approx(result.base.loss_best)
        assert restored.base.seed == 3
        assert restored.base.config == result.base.config
        assert restored.state.mu == pytest.approx(result.state.mu)
        for name in result.state.lambdas:
            np.testing.assert_allclose(
                restored.state.lambdas[name], result.state.lambdas[name]
            )
        assert [r["violation"] for r in restored.violation_history] == (
            pytest.approx([r["violation"] for r in result.violation_history])
        )

        path = tmp_path / "al_result.json"
        result.save(path)
        loaded = ALResult.load(path)
        np.testing.assert_array_equal(loaded.base.phi_best, result.base.phi_best)
        assert loaded.n_outer == result.n_outer


# ---------------------------------------------------------------------------
# L0-фикстуры: fast-путь ConstrainedProblem поверх CalibrationProblem
# ---------------------------------------------------------------------------


def _l0_problem64(l0_factory) -> CalibrationProblem:
    neuraltide.config.set_dtype(tf.float64)
    network, space, t_seq, inputs, loss = l0_factory()
    return CalibrationProblem(network, space, t_seq, inputs=inputs, objective=loss)


def _l0_offsets(problem) -> dict:
    offsets = {}
    start = 0
    for name, entry in problem.parameter_space.entries.items():
        offsets[name] = (start, entry.numel)
        start += entry.numel
    return offsets


def _l0_shifted_start(problem) -> np.ndarray:
    phi = np.asarray(problem.pack(), dtype=np.float64).reshape(-1).copy()
    off = _l0_offsets(problem)
    phi[off["exc/tau"][0]] = 20.0
    phi[off["exc/I_ext"][0]] = 0.5
    return phi


def _mean_rate_spec(problem, kind="eq", shift=0.0) -> ConstraintSpec:
    target = problem.objective.observations[0].target
    c0 = float(np.mean(np.asarray(target))) + shift

    def fn(predictions, network):
        return tf.reduce_mean(predictions.firing_rates["exc"]) - c0

    return ConstraintSpec(name="mean_rate", kind=kind, fn=fn)


class TestConstrainedProblemL0:
    def test_augmented_terms_single_forward(self, l0_factory):
        problem = _l0_problem64(l0_factory)
        cp = ConstrainedProblem(problem, [_mean_rate_spec(problem)])
        phi0 = _l0_shifted_start(problem)
        n_fwd = problem.n_forward
        terms = cp.augmented_terms(phi0)
        assert set(terms) == {"mean_rate"}
        value, violation = terms["mean_rate"]
        assert value.shape == (1,)
        np.testing.assert_allclose(violation, np.abs(value))
        assert problem.n_forward - n_fwd == 1  # один прогон на значения

    def test_single_tape_loss_and_gradient(self, l0_factory):
        problem = _l0_problem64(l0_factory)
        spec = _mean_rate_spec(problem, shift=1.0)  # ненулевое нарушение
        cp = ConstrainedProblem(problem, [spec])
        phi0 = _l0_shifted_start(problem)
        n_fwd, n_grad = problem.n_forward, problem.n_gradient
        eval_res, grad = cp.value_and_gradient(phi0)
        # data loss и AL-член в одном прогоне/ленте
        assert problem.n_forward - n_fwd == 1
        assert problem.n_gradient - n_grad == 1
        assert eval_res.grad is not None
        assert eval_res.components["al/penalty"] > 0.0
        assert eval_res.components["constraint/mean_rate"] > 0.0
        # потеря = data loss + штраф (data loss тот же, что у чистой задачи)
        plain = problem.evaluate(phi0)
        assert eval_res.loss == pytest.approx(
            plain.loss + eval_res.components["al/penalty"], rel=1e-10
        )

    def test_gradient_matches_fd(self, l0_factory):
        problem = _l0_problem64(l0_factory)
        spec = _mean_rate_spec(problem, shift=1.0)
        cp = ConstrainedProblem(problem, [spec])
        phi0 = _l0_shifted_start(problem)
        _, grad = cp.value_and_gradient(phi0)
        scan = finite_difference_scan(lambda p: cp.evaluate(p).loss, phi0)
        np.testing.assert_allclose(
            np.asarray(grad, dtype=np.float64),
            scan["best"],
            rtol=RTOL_GRAD,
            atol=ATOL_GRAD,
        )

    def test_evaluate_components_residual_validate(self, l0_factory):
        problem = _l0_problem64(l0_factory)
        cp = ConstrainedProblem(problem, [_mean_rate_spec(problem)])
        phi0 = _l0_shifted_start(problem)
        eval_res = cp.evaluate(phi0)
        assert "al/penalty" in eval_res.components
        assert "constraint/mean_rate" in eval_res.components
        assert "obs" in eval_res.components  # data-компоненты сохранены
        r = cp.residual(phi0)
        np.testing.assert_array_equal(r, problem.residual(phi0))

        assert cp.size == problem.size
        np.testing.assert_array_equal(np.asarray(cp.pack(phi0)), phi0)
        assert "exc/tau" in cp.unpack(phi0)
        lb, ub = cp.bounds_phi()
        assert lb.shape == (cp.size,) and ub.shape == (cp.size,)
        assert cp.validate(phi0) == {}

    def test_augmented_residual_for_least_squares(self, l0_factory):
        problem = _l0_problem64(l0_factory)
        cp = ConstrainedProblem(
            problem,
            [_mean_rate_spec(problem, shift=1.0)],
            augmented_residual=True,
        )
        phi0 = _l0_shifted_start(problem)
        r = cp.residual(phi0)
        r_data = problem.residual(phi0)
        assert r.shape == (r_data.size + 1,)
        # 0.5·||r_aug||² = data loss + max(ψ, 0) (λ = 0 → μc²/2)
        terms = cp.augmented_terms(phi0)
        c = terms["mean_rate"][0][0]
        assert 0.5 * float(r @ r) == pytest.approx(
            problem.evaluate(phi0).loss + 0.5 * cp.state.mu * c * c,
            rel=1e-10,
        )


class TestAugmentedLagrangianL0:
    def test_least_squares_local_mode(self, l0_factory):
        problem = _l0_problem64(l0_factory)
        spec = _mean_rate_spec(problem, shift=1.0)
        result = augmented_lagrangian(
            problem, [spec], local="least_squares", max_outer=2
        )
        assert result.n_outer == 2
        assert np.all(np.isfinite(result.base.phi_best))
        assert len(result.violation_history) == 2


# ---------------------------------------------------------------------------
# epsilon_constraint
# ---------------------------------------------------------------------------


def _two_obs_problem(l0_factory) -> CalibrationProblem:
    neuraltide.config.set_dtype(tf.float64)
    network, space, t_seq, inputs, loss = l0_factory()
    target = loss.observations[0].target
    loss2 = ResidualLoss(
        [
            ObservationSpec(name="obs_a", population="exc", target=target, sigma=2.0),
            ObservationSpec(
                name="obs_b", population="exc", target=0.8 * target, sigma=2.0
            ),
        ]
    )
    return CalibrationProblem(network, space, t_seq, inputs=inputs, objective=loss2)


class TestEpsilonConstraint:
    def test_moves_component_to_constraint(self, l0_factory):
        problem = _two_obs_problem(l0_factory)
        cp = epsilon_constraint(problem, ["obs_a", "obs_b"], 0, eps=0.05)
        phi0 = np.asarray(problem.pack(), dtype=np.float64)
        plain = problem.evaluate(phi0)

        eval_res = cp.evaluate(phi0)
        # целевая функция — сумма оставшихся компонентов (obs_b)
        assert eval_res.loss == pytest.approx(plain.components["obs_b"], rel=1e-8)
        assert {"obs_b", "al/penalty", "constraint/eps/obs_a"} <= set(
            eval_res.components
        )
        # выбранный компонент стал ограничением g = value − eps <= 0
        value, violation = cp.augmented_terms(phi0)["eps/obs_a"]
        assert value[0] == pytest.approx(plain.components["obs_a"] - 0.05, rel=1e-10)
        np.testing.assert_allclose(violation, np.maximum(value, 0.0))
        assert cp.constraints[0].kind == "ineq"
        # λ = 0, g <= 0 → неактивно, штраф нулевой
        assert eval_res.components["al/penalty"] == pytest.approx(0.0)

    def test_weights_and_explicit_objective(self, l0_factory):
        problem = _two_obs_problem(l0_factory)
        phi0 = np.asarray(problem.pack(), dtype=np.float64)
        plain = problem.evaluate(phi0)

        cp = epsilon_constraint(
            problem, ["obs_a", "obs_b"], 1, eps=0.05, weights={"obs_a": 2.0}
        )
        # data loss = 2·obs_a; штраф отдельно (g = obs_b − eps > 0 активно)
        eval_res = cp.evaluate(phi0)
        assert eval_res.loss - eval_res.components["al/penalty"] == (
            pytest.approx(2.0 * plain.components["obs_a"], rel=1e-8)
        )
        value, _ = cp.augmented_terms(phi0)["eps/obs_b"]
        assert eval_res.components["al/penalty"] == pytest.approx(
            0.5 * cp.state.mu * max(value[0], 0.0) ** 2
        )

        cp2 = epsilon_constraint(
            problem,
            ["obs_a", "obs_b"],
            1,
            eps=0.05,
            objective=lambda preds, net: (
                3.0 * problem.objective.components(preds, net)["obs_a"]
            ),
        )
        eval_res2 = cp2.evaluate(phi0)
        assert eval_res2.loss - eval_res2.components["al/penalty"] == (
            pytest.approx(3.0 * plain.components["obs_a"], rel=1e-8)
        )
        # ограничение у cp2 — на obs_b
        value, _ = cp2.augmented_terms(phi0)["eps/obs_b"]
        assert value[0] == pytest.approx(plain.components["obs_b"] - 0.05, rel=1e-10)

    def test_negative_index(self, l0_factory):
        problem = _two_obs_problem(l0_factory)
        cp = epsilon_constraint(problem, ["obs_a", "obs_b"], -1, eps=0.05)
        assert cp.constraints[0].name == "eps/obs_b"

    def test_errors(self, l0_factory):
        problem = _two_obs_problem(l0_factory)
        with pytest.raises(IndexError, match="index"):
            epsilon_constraint(problem, ["obs_a", "obs_b"], 5, 0.1)
        with pytest.raises(ValueError, match="weights"):
            epsilon_constraint(
                problem, ["obs_a", "obs_b"], 0, 0.1, weights={"nope": 1.0}
            )
        with pytest.raises(ValueError, match="objective"):
            epsilon_constraint(problem, ["obs_a"], 0, 0.1)
        view = problem.with_objective(lambda preds, net: 0.0)
        with pytest.raises(TypeError, match="components"):
            epsilon_constraint(view, ["obs_a"], 0, 0.1)
