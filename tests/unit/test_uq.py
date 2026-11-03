"""
Тесты UQ (этап P2): диагностика невязок, Laplace approximation,
bootstrap, parameter ensemble, predictive intervals, noise sensitivity.

Сеть — L0-фикстура conftest (WilsonCowan 1 юнит + StaticSynapse от
ConstantRate; 3 trainable-параметра). Аналитические проверки Laplace —
на синтетических линейно-гауссовых duck-задачах с известной ковариацией
(identity и log-трансформы ParameterSpace), Гессиан квадратичной
duck-задачи известен аналитически.

Покрываются:
- residual diagnostics на L0 с известным residual (нулевой в точке
  генерации target'а; против независимо посчитанного residual в
  возмущённой точке), fallback без observations, JSON-сериализация DTO;
- laplace: аналитическая GN-ковариация (JᵀJ)⁻¹ и с prior_cov, PD/jitter
  на вырожденном J и в не-минимуме (pd=False), fd-hessian против
  аналитического Гессиана и жёсткий кап max_fd_params, внутренний
  residual_jacobian на L0, счётчики, сериализация;
- bootstrap: детерминизм по seed, явный phi у evaluator (serial
  заглушка), схемы experiments / random-weights, запись неудачных draw,
  residual bootstrap (нулевой остаток → возврат в phi0; детерминизм от
  восстановленной референсной точки);
- parameter_ensemble: mean/std/cov, NaN-подушка theta, сериализация;
- predictive intervals: монотонные квантили, медианная центральная
  траектория, пропуск нефинитных строк, сериализация;
- noise sensitivity: точное масштабирование ~1/σ без приора, сублинейное
  с приором, сериализация.
"""

import json
from types import SimpleNamespace
from typing import List

import numpy as np
import pytest
import tensorflow as tf
from scipy import stats as sp_stats

import neuraltide.config
import neuraltide.optimization.uq as uq
from neuraltide.optimization import (
    CalibrationProblem,
    Experiment,
    JacobianResult,
    MultiExperimentProblem,
    ObservationSpec,
    ParameterEntry,
    ParameterSpace,
    ResidualLoss,
)
from neuraltide.optimization.transforms import IdentityTransform, LogTransform

DT = 0.5


# ---------------------------------------------------------------------------
# Фикстуры и duck-typed задачи
# ---------------------------------------------------------------------------


def _build_f64(l0_factory) -> CalibrationProblem:
    """L0-задача в float64."""
    neuraltide.config.set_dtype(tf.float64)
    network, space, t_seq, inputs, loss = l0_factory()
    return CalibrationProblem(network, space, t_seq, inputs=inputs, objective=loss)


def _perturbed_phi(problem: CalibrationProblem, seed: int = 0) -> np.ndarray:
    """Точка вдали от истинных параметров (невязки ненулевые)."""
    phi0 = np.asarray(problem.pack(), dtype=np.float64)
    rng = np.random.default_rng(seed)
    scale = 0.05 * np.abs(phi0) + 1e-3
    return phi0 + rng.standard_normal(problem.size) * scale


@pytest.fixture
def problem_f64(l0_factory) -> CalibrationProblem:
    return _build_f64(l0_factory)


def _identity_space(names) -> ParameterSpace:
    space = ParameterSpace()
    for name in names:
        space.add_entry(
            ParameterEntry(
                full_name=name,
                shape=tf.TensorShape([1]),
                transform=IdentityTransform(),
            )
        )
    return space


@pytest.fixture
def linear_space() -> ParameterSpace:
    """2 параметра: identity + log — для проверки интервалов по θ."""
    neuraltide.config.set_dtype(tf.float64)
    space = ParameterSpace()
    space.add_entry(
        ParameterEntry(
            full_name="a", shape=tf.TensorShape([1]), transform=IdentityTransform()
        )
    )
    space.add_entry(
        ParameterEntry(
            full_name="b", shape=tf.TensorShape([1]), transform=LogTransform()
        )
    )
    return space


def _linear_jacobian(phi_map: np.ndarray, seed: int = 123, m: int = 7):
    """Полного ранга J [m, n] и согласованный r (линейная модель)."""
    rng = np.random.default_rng(seed)
    j_mat = rng.standard_normal((m, phi_map.size))
    r = j_mat @ phi_map
    return JacobianResult(
        J=j_mat,
        r=r,
        loss=0.5 * float(r @ r),
        phi=np.asarray(phi_map, dtype=np.float64),
        n_forward=0,
        wall_time=0.0,
    )


class _StaticProblem:
    """Duck-typed проблема без прогонов сети (pack/size/parameter_space)."""

    def __init__(self, space: ParameterSpace):
        self.parameter_space = space
        self.size = int(space.size)

    @property
    def n_forward(self) -> int:
        return 0

    @property
    def n_gradient(self) -> int:
        return 0

    def pack(self, phi=None) -> np.ndarray:
        if phi is None:
            raise ValueError("pack(None) недоступен в тестовой задаче")
        return np.asarray(phi, dtype=np.float64).reshape(-1)


class _QuadraticProblem(_StaticProblem):
    """loss = 0.5·φᵀAφ — Гессиан известен аналитически (для fd-hessian)."""

    def __init__(self, space: ParameterSpace, A):
        super().__init__(space)
        self._A = np.asarray(A, dtype=np.float64)
        self._n_forward = 0

    @property
    def n_forward(self) -> int:
        return self._n_forward

    def value_and_gradient(self, phi):
        p = np.asarray(self.pack(phi), dtype=np.float64)
        self._n_forward += 1
        return SimpleNamespace(loss=float(0.5 * p @ self._A @ p)), self._A @ p


# ---------------------------------------------------------------------------
# residual_diagnostics
# ---------------------------------------------------------------------------


class TestResidualDiagnostics:
    def test_known_zero_residual_at_map(self, l0_problem):
        """В точке генерации target'а (pack() модели) residual == 0."""
        res = uq.residual_diagnostics(l0_problem, None)
        assert isinstance(res, uq.ResidualDiagnostics)
        assert res.has_observation_metadata
        assert len(res.specs) == 1
        spec = res.specs[0]
        assert spec.name == "obs"
        assert spec.population == "exc"
        assert spec.n_elements == 60
        assert spec.n_masked == 0
        assert spec.mean == 0.0
        assert spec.std == 0.0
        assert spec.max_abs == 0.0
        assert all(v == 0.0 for v in spec.time_mean)
        # Вырожденная дисперсия: автокорреляция не определена (None, не NaN).
        assert len(spec.autocorr) == 10
        assert all(v is None for v in spec.autocorr)
        assert len(spec.bin_stds) == 10
        assert all(v == 0.0 for v in spec.bin_stds)
        assert len(spec.bin_edges) == 11
        assert spec.levene_stat is None and spec.levene_pvalue is None
        assert len(res.windows) == 4
        assert res.windows[0].start == 0
        assert res.windows[-1].stop == 60
        assert res.windows[0].mean["obs"] == 0.0
        assert res.max_abs == 0.0
        assert res.loss == 0.0
        assert res.n_forward == 1
        assert res.n_residuals == 60
        assert res.notes
        json.dumps(res.to_dict())
        restored = uq.ResidualDiagnostics.from_dict(res.to_dict())
        assert restored.specs[0].name == "obs"
        assert restored.max_abs == res.max_abs

    def test_matches_independently_computed_residual(self, l0_problem):
        """mean/std/max/time_mean совпадают с ручным разложением residual."""
        problem = l0_problem
        phi = _perturbed_phi(problem, seed=1)
        res = uq.residual_diagnostics(problem, phi, max_lag=5, n_bins=5)
        r_expected = problem.residual(phi)
        spec = res.specs[0]
        assert spec.mean == pytest.approx(float(np.mean(r_expected)), rel=1e-5)
        assert spec.std == pytest.approx(float(np.std(r_expected)), rel=1e-5)
        assert spec.max_abs == pytest.approx(
            float(np.max(np.abs(r_expected))), rel=1e-5
        )
        assert len(spec.autocorr) == 5
        assert all(v is None or -1.0 <= v <= 1.0 for v in spec.autocorr)
        assert len(spec.bin_stds) == 5
        assert len(spec.bin_edges) == 6
        # n_units=1, batch=1: временной ряд — это сама невязка.
        ts = r_expected.reshape(-1)
        np.testing.assert_allclose(
            np.array([v for v in spec.time_mean], dtype=np.float64),
            ts,
            rtol=1e-5,
        )
        assert res.loss == pytest.approx(0.5 * float(r_expected @ r_expected))
        assert res.n_forward == 1
        json.dumps(res.to_dict())

    def test_fallback_without_observations(self, l0_problem):
        """Objective с residual, но без observations → плоский псевдо-spec."""
        problem = l0_problem
        target = problem.objective.observations[0].target
        n = int(target.shape.num_elements())
        sigma = 2.0

        class _Flat:
            def residual(self, predictions):
                e = predictions.firing_rates["exc"] - target
                return tf.reshape(e, [-1]) / (sigma * np.sqrt(n))

            def __call__(self, predictions, network):
                r = self.residual(predictions)
                return 0.5 * tf.reduce_sum(r * r)

        flat_problem = problem.with_objective(_Flat())
        res = uq.residual_diagnostics(flat_problem, None)
        assert not res.has_observation_metadata
        assert [s.name for s in res.specs] == ["residual"]
        assert res.specs[0].n_elements == n
        assert res.windows == []
        assert len(res.specs[0].autocorr) == 10
        json.dumps(res.to_dict())

    def test_objective_without_residual_raises(self, l0_problem):
        scalar_problem = l0_problem.with_objective(
            lambda predictions, network: tf.reduce_sum(predictions.firing_rates["exc"])
        )
        with pytest.raises(TypeError, match="residual"):
            uq.residual_diagnostics(scalar_problem, None)

    def test_lag_autocorr_helper(self):
        alternating = np.array([1.0, -1.0] * 20)
        ac = uq._lag_autocorr(alternating, 3)
        assert ac[0] == pytest.approx(-1.0)
        assert ac[1] == pytest.approx(1.0)
        constant = np.ones(10)
        assert all(v is None for v in uq._lag_autocorr(constant, 2))
        short = np.array([1.0, 2.0])
        assert all(v is None for v in uq._lag_autocorr(short, 2))

    def test_validation(self, l0_problem):
        with pytest.raises(ValueError, match="max_lag"):
            uq.residual_diagnostics(l0_problem, None, max_lag=0)
        with pytest.raises(ValueError, match="n_bins"):
            uq.residual_diagnostics(l0_problem, None, n_bins=1)
        with pytest.raises(ValueError, match="окно"):
            uq.residual_diagnostics(l0_problem, None, windows=[(10, 5)])
        with pytest.raises(ValueError, match="окно"):
            uq.residual_diagnostics(l0_problem, None, windows=[(0, 100)])


# ---------------------------------------------------------------------------
# laplace_approximation
# ---------------------------------------------------------------------------


class TestLaplaceApproximation:
    def test_gn_covariance_analytic(self, linear_space):
        """Линейно-гауссовая задача: ковариация == (JᵀJ)⁻¹ аналитически."""
        problem = _StaticProblem(linear_space)
        phi_map = np.array([0.3, np.log(2.0)])
        jac = _linear_jacobian(phi_map)
        res = uq.laplace_approximation(problem, phi_map, jac=jac)

        cov_expected = np.linalg.inv(jac.J.T @ jac.J)
        assert np.allclose(res.covariance, cov_expected, rtol=1e-8)
        assert res.pd
        assert res.jitter_used == 0.0
        assert np.allclose(res.std, np.sqrt(np.diag(cov_expected)), rtol=1e-8)
        # Спектр по убыванию.
        assert np.all(np.diff(res.eigvals) <= 0.0)
        np.testing.assert_allclose(
            np.linalg.eigvalsh(res.covariance)[::-1], res.eigvals, rtol=1e-8
        )
        assert res.method == "gn"
        assert res.n_forward == 0
        assert res.loss == pytest.approx(jac.loss)
        assert res.param_names == ["a", "b"]

        z = sp_stats.norm.ppf(0.975)
        # identity-трансформ: интервал симметричен вокруг φ.
        lo_a, hi_a = res.intervals["a"]
        assert lo_a == pytest.approx(phi_map[0] - z * res.std[0], rel=1e-8)
        assert hi_a == pytest.approx(phi_map[0] + z * res.std[0], rel=1e-8)
        assert res.phi_intervals["a"] == pytest.approx((lo_a, hi_a))
        # log-трансформ: интервал в θ = exp(φ ± z·std).
        lo_b, hi_b = res.intervals["b"]
        np.testing.assert_allclose(
            [lo_b, hi_b],
            np.exp([phi_map[1] - z * res.std[1], phi_map[1] + z * res.std[1]]),
            rtol=1e-8,
        )
        assert lo_b > 0.0
        json.dumps(res.to_dict())
        restored = uq.LaplaceResult.from_dict(res.to_dict())
        np.testing.assert_allclose(restored.covariance, res.covariance, atol=1e-15)
        assert restored.intervals == res.intervals
        assert restored.prior_precision is None

    def test_prior_cov_analytic(self, linear_space):
        problem = _StaticProblem(linear_space)
        phi_map = np.array([0.3, np.log(2.0)])
        jac = _linear_jacobian(phi_map, seed=7)
        prior_cov = np.diag([0.5, 2.0])
        res = uq.laplace_approximation(problem, phi_map, jac=jac, prior_cov=prior_cov)
        expected = np.linalg.inv(jac.J.T @ jac.J + np.linalg.inv(prior_cov))
        assert np.allclose(res.covariance, expected, rtol=1e-8)
        np.testing.assert_allclose(res.prior_precision, np.linalg.inv(prior_cov))

        # Скалярный prior_cov — изотропный приор.
        res2 = uq.laplace_approximation(problem, phi_map, jac=jac, prior_cov=0.5)
        expected2 = np.linalg.inv(jac.J.T @ jac.J + 2.0 * np.eye(2))
        assert np.allclose(res2.covariance, expected2, rtol=1e-8)

    def test_jitter_rescues_singular_gn(self, linear_space):
        """Ранг-дефектный J: JᵀJ вырождена, jitter спасает (pd=True)."""
        problem = _StaticProblem(linear_space)
        phi_map = np.array([0.3, np.log(2.0)])
        j_bad = np.array([[1.0, 1.0], [-1.0, -1.0], [2.0, 2.0]])
        jac = JacobianResult(
            J=j_bad,
            r=j_bad @ phi_map,
            loss=0.5,
            phi=phi_map,
            n_forward=0,
            wall_time=0.0,
        )
        res = uq.laplace_approximation(problem, phi_map, jac=jac)
        assert res.pd
        assert res.jitter_used is not None and res.jitter_used > 0.0
        # Ковариация всё равно SPD и конечна.
        assert np.all(np.linalg.eigvalsh(res.covariance) > 0.0)
        assert np.all(np.isfinite(res.covariance))

    def test_pd_false_in_non_minimum(self):
        """Вогнутая точка (H = −I): Cholesky не спасает, pd=False."""

        class _Concave(_QuadraticProblem):
            pass

        space = _identity_space(["x", "y"])
        concave = _Concave(space, -np.eye(2))
        res = uq.laplace_approximation(
            concave, np.array([0.1, -0.2]), method="fd-hessian"
        )
        assert not res.pd
        assert res.jitter_used is None
        assert np.all(np.isfinite(res.covariance))
        assert res.notes

    def test_fd_hessian_matches_quadratic(self):
        """H известен аналитически: fd-hessian восстанавливает A⁻¹."""
        space = _identity_space(["x", "y"])
        A = np.array([[2.0, 0.3], [0.3, 1.0]])
        quad = _QuadraticProblem(space, A)
        phi_map = np.array([0.2, -0.1])
        res = uq.laplace_approximation(quad, phi_map, method="fd-hessian")
        assert np.allclose(res.covariance, np.linalg.inv(A), rtol=1e-5)
        assert res.pd and res.jitter_used == 0.0
        assert res.loss == pytest.approx(0.5 * phi_map @ A @ phi_map)
        # 2n + 1 вычислений value_and_gradient.
        assert res.n_forward == 2 * quad.size + 1
        assert res.method == "fd-hessian"

    def test_fd_hessian_hard_cap(self):
        """Жёсткий кап: n > max_fd_params → ValueError БЕЗ прогонов."""
        space = _identity_space([f"p{i}" for i in range(51)])

        class _Exploding(_StaticProblem):
            def value_and_gradient(self, phi):  # pragma: no cover - не должен
                raise AssertionError("value_and_gradient не должен вызываться")

        duck = _Exploding(space)
        with pytest.raises(ValueError, match="max_fd_params"):
            uq.laplace_approximation(duck, np.zeros(51), method="fd-hessian")

    def test_on_l0_with_internal_jacobian(self, problem_f64):
        """jac=None → внутренний residual_jacobian; согласованность DTO."""
        problem = problem_f64
        phi = _perturbed_phi(problem, seed=3)
        res_internal = uq.laplace_approximation(problem, phi)
        from neuraltide.optimization import sensitivities

        jac = sensitivities.residual_jacobian(problem, phi)
        res_external = uq.laplace_approximation(problem, phi, jac=jac)

        assert np.allclose(res_internal.covariance, res_external.covariance, atol=1e-12)
        assert res_internal.n_forward == jac.n_forward
        names = {"exc/tau", "exc/I_ext", "in->exc/gsyn_max"}
        assert set(res_external.param_names) == names
        assert set(res_external.intervals) == names
        # Интервалы содержат θ точки разложения (identity-трансформы L0).
        theta = problem.parameter_space.unpack(phi)
        for name in names:
            lo, hi = res_external.intervals[name]
            theta_v = float(np.asarray(theta[name], dtype=np.float64).reshape(-1)[0])
            assert lo <= theta_v <= hi
        json.dumps(res_external.to_dict())

    def test_validation(self, linear_space):
        problem = _StaticProblem(linear_space)
        phi_map = np.array([0.3, np.log(2.0)])
        jac = _linear_jacobian(phi_map)
        with pytest.raises(ValueError, match="method"):
            uq.laplace_approximation(problem, phi_map, jac=jac, method="cmaes")
        with pytest.raises(ValueError, match="jac"):
            uq.laplace_approximation(problem, phi_map, jac=jac, method="fd-hessian")
        with pytest.raises(ValueError, match="level"):
            uq.laplace_approximation(problem, phi_map, jac=jac, level=1.5)
        with pytest.raises(ValueError, match="симметричной"):
            uq.laplace_approximation(
                problem,
                phi_map,
                jac=jac,
                prior_cov=np.array([[1.0, 0.5], [0.1, 1.0]]),
            )
        with pytest.raises(ValueError, match="форму"):
            uq.laplace_approximation(problem, phi_map, jac=jac, prior_cov=np.eye(3))
        bad_jac = JacobianResult(
            J=np.zeros((4, 3)),
            r=np.zeros(4),
            loss=0.0,
            phi=phi_map,
        )
        with pytest.raises(ValueError, match="jac.J"):
            uq.laplace_approximation(problem, phi_map, jac=bad_jac)
        # phi_map несовместимого размера.
        with pytest.raises(ValueError, match="элементов"):
            uq.laplace_approximation(problem, np.zeros(3), jac=jac)


# ---------------------------------------------------------------------------
# bootstrap_experiments / residual_bootstrap
# ---------------------------------------------------------------------------


def _serial_stub():
    """Serial evaluator-заглушка: фиксированное смещение phi0, запись вызовов."""
    calls: List[np.ndarray] = []

    def stub(problem, phi0):
        phi = np.asarray(phi0, dtype=np.float64).reshape(-1)
        calls.append(phi.copy())
        return SimpleNamespace(
            base=SimpleNamespace(phi_best=phi * 1.01 + 0.001, loss_best=0.5),
            message="stub",
        )

    stub.calls = calls
    return stub


@pytest.fixture
def multi_problem_f64(l0_factory) -> MultiExperimentProblem:
    """Два эксперимента на L0-сети (полный и укороченный протокол)."""
    neuraltide.config.set_dtype(tf.float64)
    network, space, t_seq, inputs, loss = l0_factory()
    target = loss.observations[0].target
    loss_half = ResidualLoss(
        [
            ObservationSpec(
                name="obs", population="exc", target=target[:, :30], sigma=2.0
            )
        ]
    )
    e1 = Experiment(name="e1", t_sequence=t_seq, inputs=inputs, objective=loss)
    e2 = Experiment(
        name="e2",
        t_sequence=t_seq[:, :30],
        inputs=inputs[:, :30],
        objective=loss_half,
    )
    return MultiExperimentProblem(network, space, [e1, e2])


class TestBootstrapExperiments:
    def test_deterministic_by_seed_and_explicit_phi(self, l0_problem):
        stub = _serial_stub()
        res1 = uq.bootstrap_experiments(l0_problem, n_draws=4, seed=7, evaluator=stub)
        phi_ref = np.asarray(l0_problem.pack(), dtype=np.float64)
        res2 = uq.bootstrap_experiments(l0_problem, n_draws=4, seed=7, evaluator=stub)

        assert res1.scheme == "random-weights"
        assert res1.evaluator == "stub"
        assert res1.n_draws == 4
        # Детерминизм по seed: выборки и веса идентичны.
        np.testing.assert_array_equal(res1.phi_samples, res2.phi_samples)
        assert res1.weights == res2.weights
        assert np.all(np.isfinite(res1.phi_samples))
        assert res1.phi_samples.shape == (4, l0_problem.size)
        # Заглушка применила своё смещение к явно переданному phi0.
        np.testing.assert_allclose(
            res1.phi_samples[0], phi_ref * 1.01 + 0.001, rtol=1e-12
        )
        # Каждая draw получила явный phi0 (все вызовы — одна точка, мутация
        # модели между draw не используется).
        assert len(stub.calls) == 8
        for call in stub.calls:
            np.testing.assert_allclose(call, phi_ref, rtol=1e-6, atol=1e-12)
        # theta samples распакованы ParameterSpace.
        assert set(res1.theta_samples) == {
            "exc/tau",
            "exc/I_ext",
            "in->exc/gsyn_max",
        }
        assert res1.theta_samples["exc/tau"].shape == (4, 1)
        assert res1.statuses == ["ok"] * 4
        assert res1.losses == [0.5] * 4
        assert res1.n_forward == 1  # оценка residual для длины весов
        assert "w ~ Exp(1)" in res1.notes
        json.dumps(res1.to_dict())
        restored = uq.BootstrapResult.from_dict(res1.to_dict())
        np.testing.assert_array_equal(restored.phi_samples, res1.phi_samples)
        assert restored.weights == res1.weights

    def test_experiments_scheme_counts(self, multi_problem_f64):
        stub = _serial_stub()
        res = uq.bootstrap_experiments(
            multi_problem_f64, n_draws=5, seed=11, evaluator=stub
        )
        assert res.scheme == "experiments"
        assert res.n_forward == 0
        assert res.statuses == ["ok"] * 5
        for w in res.weights:
            assert len(w) == 2
            assert sum(w) == 2  # подвыборка с повторением той же длины
        assert np.all(np.isfinite(res.phi_samples))
        assert "iid" in res.notes
        json.dumps(res.to_dict())

    def test_default_evaluator_random_weights(self, problem_f64):
        """evaluator=None → lbfgsb (ленивая резолюция, без импорта в uq)."""
        res = uq.bootstrap_experiments(problem_f64, n_draws=2, seed=3)
        assert res.evaluator == "lbfgsb"
        assert res.scheme == "random-weights"
        assert res.n_forward > 0
        assert np.all(np.isfinite(res.phi_samples))
        assert all(loss is not None and np.isfinite(loss) for loss in res.losses)
        json.dumps(res.to_dict())

    def test_failed_draw_recorded_with_nan(self, l0_problem):
        state = {"n": 0}

        def flaky(problem, phi0):
            state["n"] += 1
            if state["n"] % 2 == 0:
                raise RuntimeError("draw failed")
            phi = np.asarray(phi0, dtype=np.float64).reshape(-1)
            return SimpleNamespace(
                base=SimpleNamespace(phi_best=phi, loss_best=1.0), message="ok"
            )

        res = uq.bootstrap_experiments(l0_problem, n_draws=3, seed=1, evaluator=flaky)
        assert res.statuses == ["ok", "failed", "ok"]
        assert np.all(np.isfinite(res.phi_samples[0]))
        assert np.all(np.isnan(res.phi_samples[1]))
        assert np.all(np.isfinite(res.phi_samples[2]))
        assert res.losses[1] is None
        assert "RuntimeError" in res.messages[1]
        # NaN в DTO кодируется null — JSON валиден, round-trip восстанавливает.
        json.dumps(res.to_dict())
        restored = uq.BootstrapResult.from_dict(res.to_dict())
        assert np.all(np.isnan(restored.phi_samples[1]))
        assert restored.losses[1] is None

    def test_validation(self, l0_problem):
        with pytest.raises(ValueError, match="n_draws"):
            uq.bootstrap_experiments(l0_problem, n_draws=0, seed=1)
        with pytest.raises(ValueError, match="evaluator"):
            uq.bootstrap_experiments(l0_problem, n_draws=1, seed=1, evaluator="cmaes")


class TestResidualBootstrap:
    def test_zero_residual_returns_to_map(self, l0_factory):
        """Остаток ≡ 0 → target* == target → рефит возвращает phi0."""
        neuraltide.config.set_dtype(tf.float64)
        network, space, t_seq, inputs, loss = l0_factory()
        problem = CalibrationProblem(
            network, space, t_seq, inputs=inputs, objective=loss
        )
        phi0 = np.asarray(problem.pack(), dtype=np.float64)
        res = uq.residual_bootstrap(problem, n_draws=3, seed=5)

        assert res.scheme == "residual"
        assert res.evaluator == "lbfgsb"
        assert res.statuses == ["ok"] * 3
        np.testing.assert_allclose(res.phi_samples, np.tile(phi0, (3, 1)), atol=1e-6)
        assert all(loss is not None and loss < 1e-12 for loss in res.losses)
        assert res.weights == [None] * 3
        assert res.n_forward > 0
        # Честные оговорки присутствуют.
        assert "iid" in res.notes
        assert "автокорреляцию" in res.notes
        json.dumps(res.to_dict())
        restored = uq.BootstrapResult.from_dict(res.to_dict())
        np.testing.assert_allclose(restored.phi_samples, res.phi_samples)

    def test_perturbed_reference_deterministic(self, l0_factory):
        """Ненулевой остаток: выборки варьируются, seed детерминирован."""
        neuraltide.config.set_dtype(tf.float64)
        network, space, t_seq, inputs, loss = l0_factory()
        problem = CalibrationProblem(
            network, space, t_seq, inputs=inputs, objective=loss
        )
        phi_pert = _perturbed_phi(problem, seed=2)
        problem.evaluate(phi_pert)  # мутация модели в референсную точку

        res1 = uq.residual_bootstrap(problem, n_draws=4, seed=9)
        assert res1.statuses == ["ok"] * 4
        assert np.all(np.isfinite(res1.phi_samples))
        assert res1.n_forward > 0
        assert not np.allclose(
            res1.phi_samples[0], res1.phi_samples[1], rtol=1e-9, atol=1e-12
        )
        np.testing.assert_allclose(res1.phi0, phi_pert, rtol=1e-12)

        # Детерминизм: та же референсная точка + тот же seed → тот же результат.
        problem.evaluate(phi_pert)
        res2 = uq.residual_bootstrap(problem, n_draws=4, seed=9)
        np.testing.assert_allclose(
            res1.phi_samples, res2.phi_samples, rtol=1e-6, atol=1e-10
        )
        np.testing.assert_allclose(res1.losses, res2.losses, rtol=1e-6)

    def test_requires_observations(self, l0_problem):
        flat_problem = l0_problem.with_objective(
            lambda predictions, network: tf.reduce_sum(predictions.firing_rates["exc"])
        )
        with pytest.raises(ValueError, match="observations"):
            uq.residual_bootstrap(flat_problem, n_draws=2, seed=1)

    def test_validation(self, l0_problem):
        with pytest.raises(ValueError, match="n_draws"):
            uq.residual_bootstrap(l0_problem, n_draws=0, seed=1)


# ---------------------------------------------------------------------------
# parameter_ensemble
# ---------------------------------------------------------------------------


class TestParameterEnsemble:
    @staticmethod
    def _result(phi, loss, theta=None):
        return SimpleNamespace(
            phi_best=np.asarray(phi, dtype=np.float64),
            loss_best=loss,
            theta_best=theta,
        )

    def test_stats_cov_theta_and_json(self):
        phis = np.array([[0.1, -0.2], [0.2, -0.1], [0.0, -0.3]])
        results = [
            self._result(phis[i], 0.5 + i, {"a": np.array([10.0 + i])})
            for i in range(3)
        ]
        res = uq.parameter_ensemble(results)

        assert res.n_samples == 3 and res.size == 2
        np.testing.assert_allclose(res.phi_samples, phis)
        np.testing.assert_allclose(res.mean, phis.mean(axis=0))
        np.testing.assert_allclose(res.std, phis.std(axis=0))
        np.testing.assert_allclose(res.cov, np.cov(phis, rowvar=False), rtol=1e-12)
        assert res.theta_samples["a"].shape == (3, 1)
        np.testing.assert_allclose(
            res.theta_samples["a"].reshape(-1), [10.0, 11.0, 12.0]
        )
        assert res.losses == [0.5, 1.5, 2.5]
        json.dumps(res.to_dict())
        restored = uq.EnsembleResult.from_dict(res.to_dict())
        np.testing.assert_allclose(restored.phi_samples, res.phi_samples)
        np.testing.assert_allclose(restored.cov, res.cov)
        assert restored.theta_samples["a"].shape == (3, 1)

    def test_single_sample_no_cov(self):
        res = uq.parameter_ensemble(
            [self._result([1.0, 2.0], 0.1, {"a": np.array([1.0])})]
        )
        assert res.cov is None
        assert res.n_samples == 1
        np.testing.assert_allclose(res.mean, [1.0, 2.0])
        assert np.all(res.std == 0.0)
        json.dumps(res.to_dict())

    def test_missing_theta_best(self):
        r1 = self._result([1.0, 2.0], 0.1, {"a": np.array([10.0])})
        r2 = SimpleNamespace(
            base=SimpleNamespace(phi_best=np.array([1.1, 2.1]), loss_best=0.2)
        )
        # Часть без theta_best — NaN-подушка в соответствующей строке.
        res = uq.parameter_ensemble([r1, r2])
        assert res.theta_samples is not None
        np.testing.assert_allclose(res.theta_samples["a"][0], [10.0])
        assert np.isnan(res.theta_samples["a"][1])
        assert "theta_best" in res.notes
        # Все без theta_best — None.
        r3 = SimpleNamespace(
            base=SimpleNamespace(phi_best=np.array([1.3, 2.3]), loss_best=0.4)
        )
        res2 = uq.parameter_ensemble([r2, r3])
        assert res2.theta_samples is None
        json.dumps(res2.to_dict())

    def test_base_indirection(self):
        """Объекты с .base (как LocalOptimizerResult) тоже принимаются."""
        inner = self._result([0.5, 0.5], 0.7, {"a": np.array([3.0])})
        outer = SimpleNamespace(base=inner)
        res = uq.parameter_ensemble([outer])
        np.testing.assert_allclose(res.phi_samples[0], [0.5, 0.5])

    def test_validation(self):
        with pytest.raises(ValueError, match="непустая"):
            uq.parameter_ensemble([])
        with pytest.raises(ValueError, match="размерность"):
            uq.parameter_ensemble(
                [self._result([1.0, 2.0], 0.1), self._result([1.0, 2.0, 3.0], 0.2)]
            )


# ---------------------------------------------------------------------------
# predictive_intervals
# ---------------------------------------------------------------------------


class TestPredictiveIntervals:
    def test_intervals_contain_central_and_monotone(self, problem_f64):
        problem = problem_f64
        phi_map = np.asarray(problem.pack(), dtype=np.float64)
        rng = np.random.default_rng(0)
        samples = phi_map + rng.standard_normal((9, problem.size)) * 0.02

        res = uq.predictive_intervals(problem, samples, level=0.95)
        assert res.populations == ["exc"]
        assert res.level == 0.95
        T = int(problem.t_sequence.shape[1])
        lo, up, ce = res.lower["exc"], res.upper["exc"], res.central["exc"]
        assert lo.shape == (T, 1) and up.shape == (T, 1) and ce.shape == (T, 1)
        # Монотонность квантилей и включение центральной траектории.
        assert np.all(lo <= ce)
        assert np.all(ce <= up)
        assert np.all(lo <= up)
        assert res.n_samples == 9
        assert res.n_forward == 9
        assert res.n_skipped_nonfinite == 0
        json.dumps(res.to_dict())
        # Центральная траектория — медиана независимых прогонов.
        stack = []
        for row in samples:
            out = problem.simulate(row)
            stack.append(np.asarray(out.firing_rates["exc"], dtype=np.float64)[0])
        np.testing.assert_allclose(ce, np.median(np.stack(stack), axis=0), rtol=1e-6)
        restored = uq.PredictiveResult.from_dict(res.to_dict())
        np.testing.assert_allclose(restored.upper["exc"], up)

    def test_skips_nonfinite_rows(self, problem_f64):
        problem = problem_f64
        phi_map = np.asarray(problem.pack(), dtype=np.float64)
        samples = np.tile(phi_map, (4, 1))
        samples[2] = np.nan
        res = uq.predictive_intervals(problem, samples)
        assert res.n_skipped_nonfinite == 1
        assert res.n_samples == 3
        assert res.n_forward == 3

    def test_single_point_collapses(self, problem_f64):
        problem = problem_f64
        phi_map = np.asarray(problem.pack(), dtype=np.float64)
        res = uq.predictive_intervals(problem, phi_map, level=0.8)
        assert res.n_samples == 1
        np.testing.assert_allclose(res.lower["exc"], res.upper["exc"])
        np.testing.assert_allclose(res.central["exc"], res.mean["exc"])

    def test_validation(self, problem_f64):
        phi_map = np.asarray(problem_f64.pack(), dtype=np.float64)
        with pytest.raises(ValueError, match="level"):
            uq.predictive_intervals(problem_f64, np.tile(phi_map, (2, 1)), level=1.5)
        with pytest.raises(ValueError, match="координат"):
            uq.predictive_intervals(problem_f64, np.zeros((2, phi_map.size + 1)))
        bad = np.full((2, phi_map.size), np.nan)
        with pytest.raises(ValueError, match="нефинитной"):
            uq.predictive_intervals(problem_f64, bad)


# ---------------------------------------------------------------------------
# noise_sensitivity
# ---------------------------------------------------------------------------


class TestNoiseSensitivity:
    def test_flat_prior_scales_exactly(self, linear_space):
        """Без приора: cov(f) = f²·cov(1), std(f) = f·std(1) (∝ 1/σ)."""
        problem = _StaticProblem(linear_space)
        phi_map = np.array([0.3, np.log(2.0)])
        jac = _linear_jacobian(phi_map)
        lap = uq.laplace_approximation(problem, phi_map, jac=jac)

        res = uq.noise_sensitivity(lap)
        assert [e.sigma_factor for e in res.entries] == [0.5, 1.0, 2.0]
        base = res.entries[1]
        np.testing.assert_allclose(base.covariance, lap.covariance, rtol=1e-8)
        np.testing.assert_allclose(
            res.entries[0].covariance, 0.25 * lap.covariance, rtol=1e-8
        )
        np.testing.assert_allclose(
            res.entries[2].covariance, 4.0 * lap.covariance, rtol=1e-8
        )
        np.testing.assert_allclose(res.entries[2].std, 2.0 * lap.std, rtol=1e-8)
        np.testing.assert_allclose(res.entries[0].std, 0.5 * lap.std, rtol=1e-8)
        # Интервалы масштабируются как 1/σ: ширина при f=2 вдвое больше.
        w_base = lap.phi_intervals["a"][1] - lap.phi_intervals["a"][0]
        w_hi = res.entries[2].phi_intervals["a"]
        assert (w_hi[1] - w_hi[0]) == pytest.approx(2.0 * w_base, rel=1e-8)
        assert all(e.pd for e in res.entries)
        json.dumps(res.to_dict())
        restored = uq.SensitivityResult.from_dict(res.to_dict())
        np.testing.assert_allclose(
            restored.entries[2].covariance, res.entries[2].covariance, atol=1e-15
        )
        assert restored.param_names == ["a", "b"]

    def test_with_prior_scaling_sublinear(self, linear_space):
        """Сильный приор: масштабирование сублинейно, но монотонно."""
        problem = _StaticProblem(linear_space)
        phi_map = np.array([0.3, np.log(2.0)])
        jac = _linear_jacobian(phi_map, seed=7)
        lap = uq.laplace_approximation(
            problem, phi_map, jac=jac, prior_cov=0.01 * np.eye(2)
        )
        res = uq.noise_sensitivity(lap, sigma_factors=(1.0, 2.0))
        std1, std2 = res.entries[0].std, res.entries[1].std
        assert np.all(std2 > std1)
        assert np.all(std2 < 2.0 * std1)
        assert all(e.pd for e in res.entries)
        json.dumps(res.to_dict())

    def test_validation(self, linear_space):
        problem = _StaticProblem(linear_space)
        phi_map = np.array([0.3, np.log(2.0)])
        jac = _linear_jacobian(phi_map)
        lap = uq.laplace_approximation(problem, phi_map, jac=jac)
        with pytest.raises(ValueError, match="sigma_factors"):
            uq.noise_sensitivity(lap, sigma_factors=(0.0, 1.0))
        with pytest.raises(ValueError, match="sigma_factors"):
            uq.noise_sensitivity(lap, sigma_factors=())
        with pytest.raises(TypeError, match="LaplaceResult"):
            uq.noise_sensitivity(object())
