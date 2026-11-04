"""
Тесты identifiability (этап P1): FIM/SVD, профили правдоподобия, выбор
подмножества параметров.

Покрываются:
- compute_fim: линейно-гауссова задача с известной J — F == JᵀJ, eigenpairs,
  effective rank (выбор: энергия собственных значений, а не сингулярных),
  коллинеарность столбцов (порог настраиваемый), param_sensitivity;
- likelihood_profile: квадратичная least-squares задача с известными
  интервалами (порог χ²_0.95(1) = 3.8414588 на 2·ΔNLL, т.е. 1.9207294 на
  Δloss); warm start не хуже cold по суммарному nfev; неуспешный узел
  (NaN-невязка) не ломает профиль; немонотонный/несвязный профиль
  фиксируется в warnings; профиль по реальной L0-задаче (float64);
- select_subset: реальная L0-задача — kept-only refit сдвигает kept θ к θ*
  и снижает loss; theta_delta_fixed структурно нулевой (FrozenProblem);
  prediction_delta заполнен; validation_delta честно None с пояснением
  (held-out отсутствуют); duck-задача без simulate — prediction_delta=None;
- сериализация всех DTO round-trip (JSON);
- float64: спектр FIM и phi_path профиля — float64.

Аналитическая duck-задача — линейная least-squares 0.5·||Aφ − b||² с
реальным ParameterSpace (скалярные model_var-записи с tf.Variable) — для
скорости; L0-фикстуры conftest — где нужен честный residual problem.
"""

import json

import numpy as np
import pytest
import tensorflow as tf
from scipy import stats as sp_stats

import neuraltide.config
from neuraltide.optimization import CalibrationProblem
from neuraltide.optimization.identifiability import (
    CollinearityPair,
    FIMDirection,
    FIMResult,
    ProfileResult,
    SubsetResult,
    compute_fim,
    likelihood_profile,
    select_subset,
)
from neuraltide.optimization.local import minimize_lbfgsb
from neuraltide.optimization.parameters import ParameterEntry, ParameterSpace
from neuraltide.optimization.results import (
    JacobianResult,
    LocalOptimizerResult,
)
from neuraltide.optimization.sensitivities import residual_jacobian
from neuraltide.optimization.uq import laplace_approximation

CHI2_95_DF1 = float(sp_stats.chi2.ppf(0.95, df=1))  # 3.841458820694124
REL_LOSS_THRESHOLD = CHI2_95_DF1 / 2.0  # 1.920729410347062


# ---------------------------------------------------------------------------
# Helpers: аналитический JacobianResult и квадратичная duck-задача
# ---------------------------------------------------------------------------


def _jac_result(J, phi=None) -> JacobianResult:
    """JacobianResult с известной J и ненулевой невязкой в точке."""
    J = np.asarray(J, dtype=np.float64)
    if phi is None:
        phi = np.arange(J.shape[1], dtype=np.float64)
    r = J @ np.asarray(phi, dtype=np.float64) + 0.1
    return JacobianResult(J=J, r=r, loss=float(0.5 * r @ r), phi=phi)


class _EvalStub:
    def __init__(self, loss):
        self.loss = loss


class _QuadraticProblem:
    """
    Линейная least-squares задача 0.5·||Aφ − b||² с реальным ParameterSpace.

    Параметры — скалярные model_var-записи с tf.Variable (identity
    transform, без границ). fail_coord/fail_value включают узел с NaN-
    невязкой; bump добавляет гауссову выпуклую добавку к скалярной потере
    (только value_and_gradient, для профиля через lbfgsb).
    """

    def __init__(self, A, b, names=("p0", "p1", "p2")):
        self._A = np.asarray(A, dtype=np.float64)
        self._b = np.asarray(b, dtype=np.float64)
        dtype = neuraltide.config.get_dtype()
        self.parameter_space = ParameterSpace()
        for full_name in names:
            var = tf.Variable(0.0, dtype=dtype, trainable=True)
            self.parameter_space.add_entry(
                ParameterEntry(full_name=full_name, shape=[], variable=var)
            )
        self._names = list(names)
        self.held_out = []
        self.fail_coord: int | None = None
        self.fail_value: float = 0.0
        self.bump = None  # (coord, center, width, height)
        self._counters = {"n_forward": 0, "n_gradient": 0, "wall_time": 0.0}

    @property
    def size(self) -> int:
        return len(self._names)

    @property
    def n_forward(self) -> int:
        return self._counters["n_forward"]

    @property
    def n_gradient(self) -> int:
        return self._counters["n_gradient"]

    @property
    def wall_time(self) -> float:
        return self._counters["wall_time"]

    def pack(self, phi=None) -> np.ndarray:
        if phi is None:
            return np.concatenate(
                [
                    np.asarray(entry.variable, dtype=np.float64).reshape(-1)
                    for entry in self.parameter_space.entries.values()
                ]
            )
        return np.asarray(phi, dtype=np.float64).reshape(-1)

    def unpack(self, phi):
        x = self.pack(phi)
        return {name: x[i] for i, name in enumerate(self._names)}

    def bounds_phi(self):
        n = self.size
        return np.full(n, -np.inf), np.full(n, np.inf)

    def _x(self, phi):
        x = self.pack(phi)
        if (
            self.fail_coord is not None
            # Допуск: FrozenProblem пропускает fixed-θ через
            # tf.convert_to_tensor с дефолтным float32 перед cast в
            # глобальный dtype (квантование ~1e-7·|θ| на float64).
            and abs(x[self.fail_coord] - self.fail_value) < 1e-6
        ):
            return None
        return x

    def residual(self, phi) -> np.ndarray:
        self._counters["n_forward"] += 1
        x = self._x(phi)
        if x is None:
            return np.array([np.nan])
        return self._A @ x - self._b

    def value_and_gradient(self, phi):
        r = self.residual(phi)
        self._counters["n_gradient"] += 1
        loss = float(0.5 * r @ r)
        grad = self._A.T @ r
        if self.bump is not None:
            x = self._x(phi)
            if x is not None:
                coord, center, width, height = self.bump
                d = (x[coord] - center) / width
                bump_val = height * float(np.exp(-0.5 * d * d))
                loss += bump_val
                grad = grad.copy()
                grad[coord] -= bump_val * d / width
        return _EvalStub(loss), grad


@pytest.fixture
def quadratic():
    """Квадратичная задача 0.5·||Aφ − b||² с известным θ*."""
    neuraltide.config.set_dtype(tf.float64)
    rng = np.random.default_rng(0)
    A = rng.standard_normal((6, 3))
    phi_true = np.array([1.0, -2.0, 0.5])
    return _QuadraticProblem(A, A @ phi_true), A, phi_true


def _profile_theory_hw(A: np.ndarray, coord: int) -> float:
    """Теоретический полуширина 95%-интервала профиля для линейной LS."""
    F = A.T @ A
    schur = 1.0 / np.linalg.inv(F)[coord, coord]
    return float(np.sqrt(CHI2_95_DF1 / schur))


def _entry_offsets(problem) -> dict:
    offsets = {}
    start = 0
    for name, entry in problem.parameter_space.entries.items():
        offsets[name] = start
        start += entry.numel
    return offsets


# ---------------------------------------------------------------------------
# compute_fim
# ---------------------------------------------------------------------------


class TestComputeFIM:
    def test_fim_matches_analytic_quadratic(self):
        J = np.array(
            [
                [3.0, 0.0],
                [0.0, 2.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        jac = _jac_result(J)
        res = compute_fim(jac, energy=0.9, param_names=["a", "b"])

        F = J.T @ J  # diag(9, 4)
        np.testing.assert_allclose(res.fim, F, rtol=1e-12)
        assert res.fim.dtype == np.float64
        np.testing.assert_allclose(res.eigenvalues, [9.0, 4.0], rtol=1e-12)
        np.testing.assert_allclose(res.singular_values, [3.0, 2.0], rtol=1e-12)
        # Eigenpairs: F v = λ v, ||v|| = 1, спектр по убыванию.
        for lam, v in zip(res.eigenvalues, res.eigenvectors.T):
            np.testing.assert_allclose(F @ v, lam * v, rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(np.linalg.norm(v), 1.0, rtol=1e-12)
        # Энергия λ: cumulative после 1-го = 9/13 ≈ 0.692 < 0.9 → rank 2.
        assert res.effective_rank == 2
        assert res.condition_number == pytest.approx(3.0 / 2.0)
        assert res.condition_number_fim == pytest.approx(9.0 / 4.0)
        assert [d.index for d in res.stiff_dirs] == [0, 1]
        assert res.soft_dirs == []
        assert res.collinearity == []
        # Чувствительность: нормы столбцов, по убыванию.
        assert res.param_sensitivity == [("a", 3.0), ("b", 2.0)]
        assert res.param_names == ["a", "b"]
        assert res.J_shape == (4, 2)
        np.testing.assert_allclose(res.phi, jac.phi)
        assert res.loss == pytest.approx(jac.loss)
        assert res.energy == pytest.approx(0.9)

    def test_effective_rank_uses_eigenvalue_energy(self):
        # Задокументированный выбор: накопленная энергия собственных
        # значений (λ), а не сингулярных. diag(100, 1): λ-энергия первого
        # направления 10000/10001 ≥ 0.995, энергия s — 100/101 < 0.995.
        J = np.array([[100.0, 0.0], [0.0, 1.0]])
        res = compute_fim(_jac_result(J), energy=0.995)
        assert res.effective_rank == 1
        assert len(res.stiff_dirs) == 1
        assert len(res.soft_dirs) == 1

    def test_soft_direction_reports_sloppy_params(self):
        c1 = np.array([1.0, 0.0, 0.0])
        c2 = np.array([0.0, 2.0, 0.0])
        c3 = 1e-2 * c1  # почти коллинеарен c1 → мягкое направление
        J = np.column_stack([c1, c2, c3])
        res = compute_fim(_jac_result(J), energy=0.99, param_names=["p0", "p1", "p2"])
        assert res.effective_rank == 2
        assert len(res.stiff_dirs) == 2
        assert len(res.soft_dirs) == 1
        soft_names = {name for name, _ in res.soft_dirs[0].top_params}
        assert {"p0", "p2"} <= soft_names
        # singular value мягкого направления ≈ 1e-2·sqrt(1 + o(1))
        assert res.soft_dirs[0].singular_value < 0.05

    def test_collinearity_flags_and_threshold(self):
        J = np.array(
            [
                [1.0, 1.0, 0.0],
                [0.0, 0.02, 2.0],
                [0.0, 0.0, 0.0],
            ]
        )
        res = compute_fim(_jac_result(J), param_names=["x", "y", "z"])
        pairs = {(p.first, p.second): p.cosine for p in res.collinearity}
        assert ("x", "y") in pairs
        cos_xy = 1.0 / np.sqrt(1.0 + 0.02**2)
        assert pairs[("x", "y")] == pytest.approx(cos_xy, rel=1e-9)
        assert ("x", "z") not in pairs
        assert ("y", "z") not in pairs
        # Порог настраиваемый: выше cos_xy — пусто.
        res2 = compute_fim(_jac_result(J), collinearity_threshold=0.999999)
        assert res2.collinearity == []

    def test_zero_column_skipped_in_collinearity(self):
        J = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        res = compute_fim(_jac_result(J), param_names=["on", "off"])
        assert res.collinearity == []
        assert ("off", 0.0) in res.param_sensitivity

    def test_param_names_length_mismatch(self):
        with pytest.raises(ValueError, match="param_names"):
            compute_fim(_jac_result(np.ones((3, 2))), param_names=["only_one"])

    def test_bad_energy_and_threshold(self):
        with pytest.raises(ValueError, match="energy"):
            compute_fim(_jac_result(np.eye(2)), energy=1.5)
        with pytest.raises(ValueError, match="collinearity_threshold"):
            compute_fim(_jac_result(np.eye(2)), collinearity_threshold=1.5)


# ---------------------------------------------------------------------------
# likelihood_profile: квадратичная задача с известными интервалами
# ---------------------------------------------------------------------------


class TestLikelihoodProfileQuadratic:
    def test_intervals_match_theory(self, quadratic):
        problem, A, phi_true = quadratic
        hw = _profile_theory_hw(A, 1)
        grid = phi_true[1] + np.linspace(-2.0 * hw, 2.0 * hw, 81)
        prof = likelihood_profile(problem, "p1", grid=grid)

        assert prof.name == "p1"
        assert all(prof.successes)
        step = float(grid[1] - grid[0])
        # Интервал по χ²(1): |θ − θ*| <= sqrt(3.8414588 / s).
        lo, hi = prof.intervals["0.95"]
        assert lo == pytest.approx(phi_true[1] - hw, abs=step + 1e-9)
        assert hi == pytest.approx(phi_true[1] + hw, abs=step + 1e-9)
        # Минимум профиля в узле ≈ θ*, rel_loss минимален ≈ 0.
        i_min = int(np.nanargmin(prof.losses))
        assert prof.theta_values[i_min] == pytest.approx(phi_true[1], abs=step)
        assert min(prof.rel_loss) == pytest.approx(0.0, abs=1e-9)
        # Пороги задокументированы в полях.
        assert prof.chi2_threshold == pytest.approx(CHI2_95_DF1)
        assert prof.rel_loss_threshold == pytest.approx(REL_LOSS_THRESHOLD)
        assert prof.warnings == []
        # rel_loss — разница от минимума по успешным узлам.
        assert len(prof.rel_loss) == len(prof.theta_values)
        assert len(prof.phi_path) == len(prof.theta_values)
        assert all(p is not None for p in prof.phi_path)
        # phi_path — полный вектор φ: свободные координаты совпадают с θ*
        # (identity transform), фиксированная отклоняется не более чем на шаг.
        phi_min = prof.phi_path[i_min]
        assert phi_min[0] == pytest.approx(phi_true[0], abs=1e-6)
        assert phi_min[2] == pytest.approx(phi_true[2], abs=1e-6)
        assert phi_min[1] == pytest.approx(phi_true[1], abs=step)
        assert prof.n_forward > 0 and prof.n_gradient > 0

    def test_warm_start_not_worse_than_cold(self, quadratic):
        problem, A, phi_true = quadratic
        grid = phi_true[1] + np.linspace(-1.5, 1.5, 21)
        warm = likelihood_profile(problem, "p1", grid=grid, warm_start=True)
        cold = likelihood_profile(problem, "p1", grid=grid, warm_start=False)

        assert all(warm.successes) and all(cold.successes)
        assert sum(warm.nfev) <= sum(cold.nfev)
        # Одинаковый (в пределах точности) профиль.
        np.testing.assert_allclose(warm.losses, cold.losses, rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(warm.rel_loss, cold.rel_loss, rtol=1e-6, atol=1e-7)
        for pw, pc in zip(warm.phi_path, cold.phi_path):
            np.testing.assert_allclose(pw, pc, rtol=1e-6, atol=1e-5)

    def test_least_squares_optimizer_string(self, quadratic):
        problem, A, phi_true = quadratic
        grid = phi_true[1] + np.linspace(-1.0, 1.0, 11)
        prof = likelihood_profile(problem, "p1", grid=grid, optimizer="least_squares")
        assert prof.optimizer == "least_squares"
        assert all(prof.successes)
        assert min(prof.losses) == pytest.approx(0.0, abs=1e-9)

    def test_callable_optimizer(self, quadratic):
        problem, A, phi_true = quadratic
        starts = []

        def _custom(frozen, phi0):
            starts.append(np.asarray(phi0, dtype=np.float64).copy())
            return minimize_lbfgsb(frozen, phi0)

        grid = phi_true[1] + np.linspace(-1.0, 1.0, 7)
        prof = likelihood_profile(problem, "p1", grid=grid, optimizer=_custom)
        assert prof.optimizer == "callable"
        assert len(starts) == len(grid)
        assert all(prof.successes)
        assert min(prof.losses) == pytest.approx(0.0, abs=1e-9)

    def test_failed_node_does_not_break_profile(self, quadratic):
        problem, A, phi_true = quadratic
        hw = _profile_theory_hw(A, 1)
        grid = phi_true[1] + np.linspace(-2.0 * hw, 3.0 * hw, 41)
        # Последний узел (3σ-зона) отдаёт NaN-невязку.
        problem.fail_coord = 1
        problem.fail_value = float(grid[-1])
        prof = likelihood_profile(problem, "p1", grid=grid)

        assert not prof.successes[-1]
        assert np.isnan(prof.losses[-1])
        assert np.isnan(prof.rel_loss[-1])
        assert prof.phi_path[-1] is None
        assert prof.nfev[-1] == 0
        assert "ValueError" in prof.messages[-1]
        assert all(prof.successes[:-1])
        # Интервалы по успешным узлам совпадают с теорией.
        step = float(grid[1] - grid[0])
        lo, hi = prof.intervals["0.95"]
        assert lo == pytest.approx(phi_true[1] - hw, abs=step + 1e-9)
        assert hi == pytest.approx(phi_true[1] + hw, abs=step + 1e-9)

    def test_nonmonotone_profile_flagged(self, quadratic):
        problem, A, phi_true = quadratic
        hw = _profile_theory_hw(A, 1)
        # Сетка внутри допустимой области; добавка-подъём выше порога
        # χ²/2 разрывает допустимую область на две компоненты.
        problem.bump = (1, phi_true[1] + 0.5 * hw, 0.1 * hw, 5.0)
        grid = phi_true[1] + np.linspace(-hw, hw, 21)
        prof = likelihood_profile(problem, "p1", grid=grid, warm_start=False)

        assert all(prof.successes)
        assert any("несвязна" in w for w in prof.warnings)
        lo, hi = prof.intervals["0.95"]
        # Интервал — связная компонента глобального минимума (левая).
        assert lo < phi_true[1]
        assert hi < phi_true[1] + 0.5 * hw

    def test_bad_inputs(self, quadratic):
        problem, A, phi_true = quadratic
        with pytest.raises(ValueError, match="неизвестная запись"):
            likelihood_profile(problem, "nope", grid=[0.0, 1.0])
        with pytest.raises(ValueError, match="values и grid"):
            likelihood_profile(problem, "p1", values=[0.0, 1.0], grid=[0.0, 1.0])
        with pytest.raises(ValueError, match=">= 2 узлов"):
            likelihood_profile(problem, "p1", grid=[0.0])
        with pytest.raises(ValueError, match="нефинитные"):
            likelihood_profile(problem, "p1", grid=[0.0, np.nan])
        with pytest.raises(ValueError, match="optimizer"):
            likelihood_profile(problem, "p1", grid=[0.0, 1.0], optimizer="cma")
        with pytest.raises(ValueError, match="numel"):
            problem_vec = _QuadraticProblem(A, A @ phi_true, names=("p0",))
            problem_vec.parameter_space.add_entry(
                ParameterEntry(full_name="vec", shape=[2], variable=None)
            )
            likelihood_profile(problem_vec, "vec", grid=[0.0, 1.0])


class _TensorBackedProblem(_QuadraticProblem):
    """Эмуляция задачи с tensor-backed записями: pack(None) невозможен."""

    def pack(self, phi=None):
        if phi is None:
            raise ValueError(
                "ParameterSpace: pack(None) requires model variables; entry "
                "'x0/initial_state' has none. Pass 'values' with explicit "
                "numbers instead."
            )
        return super().pack(phi)


class TestLikelihoodProfilePhiStart:
    def test_pack_none_unavailable_requires_explicit_phi(self, quadratic):
        """pack() без phi невозможен (tensor-backed записи) — честный
        ValueError с подсказкой передать phi=; с phi= профиль строится."""
        problem, A, phi_true = quadratic
        tb = _TensorBackedProblem(problem._A, problem._b, names=problem._names)

        with pytest.raises(ValueError, match="phi"):
            likelihood_profile(tb, "p1", grid=[-2.0, -1.5, -1.0])

        phi0 = phi_true + np.array([0.3, -0.4, 0.2])
        prof = likelihood_profile(
            tb,
            "p1",
            grid=phi_true[1] + np.linspace(-1.0, 1.0, 5),
            phi=phi0,
        )
        assert all(prof.successes)
        assert min(prof.losses) == pytest.approx(0.0, abs=1e-9)

    def test_explicit_phi_sets_cold_start(self, quadratic):
        """phi= задаёт cold start свободных координат и непрофилируемые
        координаты phi_path (а не pack() текущего состояния модели)."""
        problem, A, phi_true = quadratic
        phi0 = phi_true + np.array([0.3, -0.4, 0.2])
        starts = []

        def _custom(frozen, start):
            starts.append(np.asarray(start, dtype=np.float64).copy())
            return minimize_lbfgsb(frozen, start)

        grid = phi_true[1] + np.linspace(-1.0, 1.0, 5)
        prof = likelihood_profile(
            problem, "p1", grid=grid, optimizer=_custom, phi=phi0, warm_start=False
        )

        assert all(prof.successes)
        # cold start — свободные координаты переданного phi (p0, p2),
        # а не текущих переменных модели (нули).
        np.testing.assert_allclose(starts[0], [phi0[0], phi0[2]], atol=1e-12)
        # Решение согласовано с квадратичной задачей: минимум в θ*.
        i_min = int(np.nanargmin(prof.losses))
        phi_min = prof.phi_path[i_min]
        assert phi_min[0] == pytest.approx(phi_true[0], abs=1e-6)
        assert phi_min[2] == pytest.approx(phi_true[2], abs=1e-6)


# ---------------------------------------------------------------------------
# likelihood_profile: реальная L0-задача (float64)
# ---------------------------------------------------------------------------


class TestLikelihoodProfileL0:
    def test_tau_profile_on_l0(self, l0_factory):
        neuraltide.config.set_dtype(tf.float64)
        network, space, t_seq, inputs, loss = l0_factory()
        problem = CalibrationProblem(
            network, space, t_seq, inputs=inputs, objective=loss
        )
        # Шаг сетки 0.1 меньше полуширины 95 %-интервала (~0.44 при N = 60,
        # sigma = 2): иначе интервал схлопывается в один узел.
        # TRF: L-BFGS-B по умолчанию в единицах полного NLL изредка
        # завершает узел с ABNORMAL_TERMINATION_IN_LNSRCH уже в оптимуме
        # (замечание С9 ревью), узел тогда помечается неуспешным.
        grid = np.linspace(7.0, 9.0, 21)
        prof = likelihood_profile(
            problem, "exc/tau", grid=grid, optimizer="least_squares"
        )

        assert all(prof.successes)
        assert prof.phi_path[0].dtype == np.float64
        i_min = int(np.nanargmin(prof.losses))
        assert prof.theta_values[i_min] == pytest.approx(8.0, abs=0.1)
        lo, hi = prof.intervals["0.95"]
        assert lo is not None and hi is not None
        assert lo < 8.0 < hi
        assert min(prof.rel_loss) == pytest.approx(0.0, abs=1e-9)

        # Независимая сверка: у почти линейной в окрестности задачи профильный
        # интервал совпадает с маргинальным интервалом Лапласа (оба — в
        # единицах полного NLL). Лаплас — на свежем экземпляре (ограничение
        # forward-AD после GradientTape, docs/optimization.md).
        network2, space2, t_seq2, inputs2, loss2 = l0_factory()
        fresh = CalibrationProblem(
            network2, space2, t_seq2, inputs=inputs2, objective=loss2
        )
        lap = laplace_approximation(fresh, np.asarray(fresh.pack(), dtype=np.float64))
        lap_lo, lap_hi = lap.intervals["exc/tau"]
        assert (hi - lo) == pytest.approx(lap_hi - lap_lo, rel=0.1)


# ---------------------------------------------------------------------------
# select_subset
# ---------------------------------------------------------------------------


class TestSelectSubset:
    def _problem_with_jac(self, l0_factory, shift=True):
        neuraltide.config.set_dtype(tf.float64)
        network, space, t_seq, inputs, loss = l0_factory()
        problem = CalibrationProblem(
            network, space, t_seq, inputs=inputs, objective=loss
        )
        phi0 = np.asarray(problem.pack(), dtype=np.float64).reshape(-1).copy()
        if shift:
            offsets = _entry_offsets(problem)
            phi0[offsets["exc/tau"]] = 20.0
            phi0[offsets["exc/I_ext"]] = 0.5
            phi0[offsets["in->exc/gsyn_max"]] = 0.2
        jac = residual_jacobian(problem, phi0)
        return problem, jac

    def test_refit_bias_fields_filled(self, l0_factory):
        problem, jac = self._problem_with_jac(l0_factory)
        res = select_subset(problem, jac, ["exc/tau", "exc/I_ext"])

        assert res.kept_names == ["exc/tau", "exc/I_ext"]
        assert res.fixed_names == ["in->exc/gsyn_max"]
        assert isinstance(res.refit_result, LocalOptimizerResult)
        assert res.refit_result.success
        assert res.loss_before == pytest.approx(jac.loss)
        assert res.loss_after < res.loss_before

        # Kept-only refit двигает kept θ к истинным значениям θ*.
        d_tau = float(res.theta_delta_kept["exc/tau"].reshape(-1)[0])
        d_ext = float(res.theta_delta_kept["exc/I_ext"].reshape(-1)[0])
        assert d_tau == pytest.approx(8.0 - 20.0, abs=0.1)
        assert d_ext == pytest.approx(0.8 - 0.5, abs=0.05)
        # Фиксированные заморожены на значениях jac.phi → дельта ровно 0
        # (честно вычислена из phi_full, не hardcoded).
        d_gsyn = float(res.theta_delta_fixed["in->exc/gsyn_max"].reshape(-1)[0])
        assert d_gsyn == 0.0

        # Прогнозный сдвиг заполнен (обучающий протокол).
        assert res.prediction_delta is not None
        assert set(res.prediction_delta) == {"exc"}
        assert 0.0 < res.prediction_delta["exc"] < 1.0
        # Validation-протоколы отсутствуют — честно None с пояснением.
        assert res.validation_delta is None
        assert "held-out" in res.notes
        assert "структурно нулевой" in res.notes
        assert res.n_forward > 0 and res.n_gradient > 0
        assert res.wall_time > 0.0

    def test_refit_at_optimum_zero_bias(self, l0_factory):
        """Если jac взят в оптимуме, refit ничего не сдвигает."""
        problem, jac = self._problem_with_jac(l0_factory, shift=False)
        res = select_subset(problem, jac, ["exc/tau", "exc/I_ext"])
        assert res.refit_result.success
        assert res.loss_after == pytest.approx(res.loss_before, abs=1e-10)
        for name, delta in res.theta_delta_kept.items():
            assert float(np.max(np.abs(delta))) < 1e-8
        assert res.prediction_delta is not None
        assert res.prediction_delta["exc"] < 1e-8

    def test_refit_false(self, l0_factory):
        problem, jac = self._problem_with_jac(l0_factory)
        res = select_subset(problem, jac, ["exc/tau"], refit=False)
        assert res.kept_names == ["exc/tau"]
        assert res.fixed_names == ["exc/I_ext", "in->exc/gsyn_max"]
        assert res.refit_result is None
        assert res.loss_before == pytest.approx(jac.loss)
        assert res.loss_after is None
        assert res.theta_delta_kept is None
        assert res.theta_delta_fixed is None
        assert res.prediction_delta is None
        assert res.validation_delta is None
        assert "refit=False" in res.notes
        assert res.n_forward == 0 and res.n_gradient == 0

    def test_bad_keep(self, l0_factory):
        problem, jac = self._problem_with_jac(l0_factory)
        with pytest.raises(ValueError, match="unknown|неизвестные"):
            select_subset(problem, jac, ["exc/tau", "nope"])
        with pytest.raises(ValueError, match="keep"):
            select_subset(problem, jac, [])

    def test_prediction_unavailable_on_duck(self, quadratic):
        problem, A, phi_true = quadratic
        phi0 = phi_true + np.array([0.3, -0.4, 0.2])
        r = problem.residual(phi0)
        jac = JacobianResult(J=A, r=r, loss=float(0.5 * r @ r), phi=phi0)
        res = select_subset(problem, jac, ["p0", "p2"])
        assert res.refit_result is not None and res.refit_result.success
        # p1 зафиксирован на phi0[1] → ожидаемый constrained optimum —
        # линейная LS-задача по свободным координатам (не ноль).
        free_cols = [0, 2]
        a_free = A[:, free_cols]
        rhs = problem._b - A[:, 1] * phi0[1]
        x_free, *_ = np.linalg.lstsq(a_free, rhs, rcond=None)
        expected_loss = float(0.5 * np.sum((a_free @ x_free - rhs) ** 2))
        assert expected_loss > 0.0
        assert res.loss_after == pytest.approx(expected_loss, abs=1e-10)
        assert res.loss_before > res.loss_after
        # Duck-задача без simulate → prediction_delta честно None + причина.
        assert res.prediction_delta is None
        assert "simulate" in res.notes
        assert res.validation_delta is None
        d_p0 = float(res.theta_delta_kept["p0"].reshape(-1)[0])
        assert d_p0 == pytest.approx(x_free[0] - phi0[0], abs=1e-8)


# ---------------------------------------------------------------------------
# Сериализация DTO
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_fim_result_roundtrip(self):
        J = np.array(
            [
                [1.0, 1.0, 0.0],
                [0.0, 0.02, 2.0],
                [0.0, 0.0, 0.5],
            ]
        )
        jac = _jac_result(J, phi=np.array([0.3, -0.7, 1.1]))
        res = compute_fim(jac, energy=0.95, param_names=["x", "y", "z"])
        blob = json.dumps(res.to_dict())
        res2 = FIMResult.from_dict(json.loads(blob))

        np.testing.assert_allclose(res2.fim, res.fim, rtol=1e-12)
        np.testing.assert_allclose(res2.eigenvalues, res.eigenvalues, rtol=1e-12)
        np.testing.assert_allclose(res2.eigenvectors, res.eigenvectors, rtol=1e-12)
        np.testing.assert_allclose(res2.singular_values, res.singular_values)
        assert res2.effective_rank == res.effective_rank
        assert res2.condition_number == pytest.approx(res.condition_number)
        assert res2.condition_number_fim == pytest.approx(res.condition_number_fim)
        assert res2.energy == pytest.approx(res.energy)
        assert res2.param_names == res.param_names
        assert res2.J_shape == res.J_shape
        assert [d.to_dict() for d in res2.stiff_dirs] == [
            d.to_dict() for d in res.stiff_dirs
        ]
        assert [d.to_dict() for d in res2.soft_dirs] == [
            d.to_dict() for d in res.soft_dirs
        ]
        assert [p.to_dict() for p in res2.collinearity] == [
            p.to_dict() for p in res.collinearity
        ]
        # JSON round-trip сохраняет float repr точно.
        assert res2.param_sensitivity == res.param_sensitivity
        np.testing.assert_allclose(res2.phi, res.phi)
        assert res2.loss == pytest.approx(res.loss)

    def test_profile_result_roundtrip(self, quadratic):
        problem, A, phi_true = quadratic
        hw = _profile_theory_hw(A, 1)
        grid = phi_true[1] + np.linspace(-2.0 * hw, 3.0 * hw, 21)
        problem.fail_coord = 1
        problem.fail_value = float(grid[-1])
        prof = likelihood_profile(problem, "p1", grid=grid)
        blob = json.dumps(prof.to_dict())
        prof2 = ProfileResult.from_dict(json.loads(blob))

        assert prof2.name == prof.name
        assert prof2.theta_values == prof.theta_values
        assert prof2.successes == prof.successes
        assert prof2.nfev == prof.nfev
        for a, b in zip(prof2.losses, prof.losses):
            assert (np.isnan(a) and np.isnan(b)) or a == pytest.approx(b)
        assert prof2.phi_path[-1] is None
        assert prof2.intervals.keys() == prof.intervals.keys()
        for key in prof.intervals:
            expected = prof.intervals[key]
            got = prof2.intervals[key]
            if expected is None:
                assert got is None
            else:
                assert got == pytest.approx(expected)
        assert prof2.chi2_threshold == pytest.approx(prof.chi2_threshold)
        assert prof2.rel_loss_threshold == pytest.approx(prof.rel_loss_threshold)
        assert prof2.warnings == prof.warnings
        assert prof2.optimizer == prof.optimizer
        assert prof2.warm_start == prof.warm_start
        for p2, p1 in zip(prof2.phi_path[:-1], prof.phi_path[:-1]):
            np.testing.assert_allclose(p2, p1, rtol=1e-12)

    def test_subset_result_roundtrip(self, l0_factory):
        problem, jac = TestSelectSubset()._problem_with_jac(l0_factory)
        res = select_subset(problem, jac, ["exc/tau", "exc/I_ext"])
        blob = json.dumps(res.to_dict())
        res2 = SubsetResult.from_dict(json.loads(blob))

        assert res2.kept_names == res.kept_names
        assert res2.fixed_names == res.fixed_names
        assert res2.refit_result is not None
        assert res2.refit_result.base.loss_best == pytest.approx(
            res.refit_result.base.loss_best
        )
        assert res2.loss_before == pytest.approx(res.loss_before)
        assert res2.loss_after == pytest.approx(res.loss_after)
        for name in res.theta_delta_kept:
            np.testing.assert_allclose(
                res2.theta_delta_kept[name], res.theta_delta_kept[name]
            )
        for name in res.theta_delta_fixed:
            np.testing.assert_allclose(
                res2.theta_delta_fixed[name], res.theta_delta_fixed[name]
            )
        assert res2.prediction_delta == pytest.approx(res.prediction_delta)
        assert res2.validation_delta == res.validation_delta
        assert res2.n_forward == res.n_forward

    def test_subset_result_roundtrip_refit_false(self, l0_factory):
        problem, jac = TestSelectSubset()._problem_with_jac(l0_factory)
        res = select_subset(problem, jac, ["exc/tau"], refit=False)
        res2 = SubsetResult.from_dict(json.loads(json.dumps(res.to_dict())))
        assert res2.refit_result is None
        assert res2.loss_after is None
        assert res2.theta_delta_kept is None
        assert res2.prediction_delta is None
        assert res2.notes == res.notes


# ---------------------------------------------------------------------------
# DTO-типы и float64
# ---------------------------------------------------------------------------


class TestDtypeAndTypes:
    def test_fim_arrays_are_float64(self):
        J = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        res = compute_fim(_jac_result(J))
        for arr in (res.fim, res.eigenvalues, res.eigenvectors, res.singular_values):
            assert arr.dtype == np.float64

    def test_direction_and_pair_types(self):
        J = np.array([[1.0, 1.0], [0.0, 0.02]])
        res = compute_fim(_jac_result(J), param_names=["x", "y"])
        assert isinstance(res.stiff_dirs[0], FIMDirection)
        assert isinstance(res.collinearity[0], CollinearityPair)
        assert all(isinstance(name, str) for name, _ in res.param_sensitivity)

    def test_profile_phi_path_float64(self, quadratic):
        problem, A, phi_true = quadratic
        prof = likelihood_profile(
            problem, "p1", grid=phi_true[1] + np.linspace(-1.0, 1.0, 5)
        )
        assert all(p.dtype == np.float64 for p in prof.phi_path if p is not None)
