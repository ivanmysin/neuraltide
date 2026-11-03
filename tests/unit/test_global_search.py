"""
Тесты P2: глобальный безградиентный поиск (global_search).

Основная логика проверяется на duck-typed 2D Rosenbrock без TensorFlow:
f(x, y) = (1-x)^2 + 100*(y - x^2)^2, минимум 0 в (1, 1), границы
[-5, 5]^2. Покрывается: DE (решение, бюджет, детерминизм, доводка
refine_top_k через настоящий minimize_lbfgsb), SPSA (checkpoint/resume
эквивалентен непрерывному прогону; evaluator-заглушка внешнего
параллелизма), NaN -> +inf со счётчиком отказов, отсутствие границ ->
ValueError с советом, сериализация DTO.

CMA-ES и NSGA-II покрываются двояко: при отсутствии пакетов
(optional extras 'global'/'pareto') — ImportError/OptionalDependencyError
с сообщением об установке; при наличии (pytest.importorskip) — короткие
прогоны и resume. На реальной L0 residual-задаче (фикстура l0_problem)
прогоняются spsa/basinhopping малым бюджетом: конечная потеря, счётчики
n_forward == n_evals + 1 (финальная явная оценка), DE на L0 честно
отказывает из-за неограниченной записи (I_ext без min/max).
"""

import json

import numpy as np
import pytest

from neuraltide.optimization.global_search import (
    GlobalOptimizerResult,
    OptionalDependencyError,
    ParetoResult,
    _Oracle,
    global_search,
)
from neuraltide.optimization.results import OptimizationResult, RunCheckpoint

# ---------------------------------------------------------------------------
# Duck-typed задачи (без TensorFlow)
# ---------------------------------------------------------------------------


class _Eval:
    """Минимальная Evaluation-заглушка (loss + components)."""

    def __init__(self, loss, components=None):
        self.loss = float(loss)
        self.components = dict(components or {})


def _rosenbrock(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return float((1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2)


def _rosenbrock_grad(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    g0 = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] ** 2)
    g1 = 200.0 * (x[1] - x[0] ** 2)
    return np.array([g0, g1], dtype=np.float64)


class RosenbrockProblem:
    """
    2D Rosenbrock на [-5, 5]^2; duck-typed контракт problem.

    fail_neg_x0: оценки с x[0] < 0 возвращают NaN (проверка учёта
    отказов). interrupt_at: evaluate бросает KeyboardInterrupt, когда
    n_forward достигает лимита (симуляция прерывания прогона; BaseException
    пробрасывается сквозь oracle).
    """

    size = 2
    held_out = []

    def __init__(self, with_bounds=True, fail_neg_x0=False, interrupt_at=None):
        self.n_forward = 0
        self.n_gradient = 0
        self.with_bounds = with_bounds
        self.fail_neg_x0 = fail_neg_x0
        self.interrupt_at = interrupt_at

    def pack(self, phi=None):
        if phi is None:
            return np.zeros(2, dtype=np.float64)
        return np.asarray(phi, dtype=np.float64).reshape(-1)

    def bounds_phi(self):
        if not self.with_bounds:
            return (
                np.full(2, -np.inf),
                np.full(2, np.inf),
            )
        return np.array([-5.0, -5.0]), np.array([5.0, 5.0])

    def unpack(self, phi):
        phi = np.asarray(phi, dtype=np.float64).reshape(-1)
        return {"x0": phi[0], "x1": phi[1]}

    def evaluate(self, phi):
        if self.interrupt_at is not None and self.n_forward >= self.interrupt_at:
            raise KeyboardInterrupt("simulated interruption")
        self.n_forward += 1
        x = np.asarray(phi, dtype=np.float64).reshape(-1)
        if self.fail_neg_x0 and x[0] < 0.0:
            return _Eval(float("nan"), {"resid": float("nan")})
        loss = _rosenbrock(x)
        return _Eval(loss, {"resid": loss})

    def value_and_gradient(self, phi):
        self.n_forward += 1
        self.n_gradient += 1
        x = np.asarray(phi, dtype=np.float64).reshape(-1)
        return _Eval(_rosenbrock(x)), _rosenbrock_grad(x)


class TwoObjectiveProblem(RosenbrockProblem):
    """Две цели из components для NSGA-II: f1=(x0-1)^2, f2=(x1+1)^2."""

    def evaluate(self, phi):
        if self.interrupt_at is not None and self.n_forward >= self.interrupt_at:
            raise KeyboardInterrupt("simulated interruption")
        self.n_forward += 1
        x = np.asarray(phi, dtype=np.float64).reshape(-1)
        f1 = float((x[0] - 1.0) ** 2)
        f2 = float((x[1] + 1.0) ** 2)
        return _Eval(f1 + f2, {"f1": f1, "f2": f2})


# ---------------------------------------------------------------------------
# DE
# ---------------------------------------------------------------------------


class TestDifferentialEvolution:
    def test_solves_rosenbrock(self):
        problem = RosenbrockProblem()
        result = global_search(
            problem,
            method="de",
            max_evals=4000,
            seed=7,
            options={"popsize": 20, "tol": 0.01},
        )
        assert isinstance(result, GlobalOptimizerResult)
        assert result.method == "de"
        assert result.success
        assert result.base.loss_best < 1e-3
        assert np.allclose(result.base.phi_best, [1.0, 1.0], atol=1e-2)
        assert result.n_evals <= 4000
        # Каждая оценка кандидата — один problem.evaluate; финальная явная
        # оценка phi_best добавляет ровно один прогон.
        assert problem.n_forward == result.n_evals + 1
        assert result.n_forward == problem.n_forward

    def test_budget_stop(self):
        problem = RosenbrockProblem()
        result = global_search(
            problem,
            method="de",
            max_evals=60,
            seed=7,
            options={"popsize": 8},
        )
        assert result.n_evals <= 60
        assert result.n_evals > 0
        assert problem.n_forward <= 61
        assert "max_evals" in result.message
        assert result.base.loss_best < 100.0  # бюджет мал, но прогон полезен

    def test_deterministic_same_seed(self):
        runs = []
        for _ in range(2):
            problem = RosenbrockProblem()
            runs.append(
                global_search(
                    problem,
                    method="de",
                    max_evals=600,
                    seed=11,
                    options={"popsize": 10},
                )
            )
        assert runs[0].n_evals == runs[1].n_evals
        assert runs[0].base.loss_best == runs[1].base.loss_best
        np.testing.assert_array_equal(runs[0].base.phi_best, runs[1].base.phi_best)

    def test_refine_top_k_improves(self):
        unrefined = global_search(
            RosenbrockProblem(),
            method="de",
            max_evals=400,
            seed=3,
            options={"popsize": 10},
            refine_top_k=0,
        )
        problem = RosenbrockProblem()
        refined = global_search(
            problem,
            method="de",
            max_evals=400,
            seed=3,
            options={"popsize": 10},
            refine_top_k=2,
            local="lbfgsb",
        )
        assert unrefined.base.loss_best > 1e-8
        assert refined.base.loss_best < 1e-8
        assert refined.base.loss_best < unrefined.base.loss_best
        assert len(refined.candidates) == 2
        assert all(c.status == "ok" and c.refined for c in refined.candidates)
        assert all(c.local is not None for c in refined.candidates)
        # Доводка и финальная оценка добавляют прогоны поверх бюджета DE.
        assert problem.n_forward > refined.n_evals

    def test_requires_finite_bounds(self):
        problem = RosenbrockProblem(with_bounds=False)
        with pytest.raises(ValueError, match="sample_phi0s"):
            global_search(problem, method="de", max_evals=100, seed=1)
        with pytest.raises(ValueError, match="transforms"):
            global_search(problem, method="de", max_evals=100, seed=1)


# ---------------------------------------------------------------------------
# SPSA: evaluator-заглушка, NaN, checkpoint/resume
# ---------------------------------------------------------------------------


class TestSPSA:
    def test_runs_rosenbrock_small_budget(self):
        problem = RosenbrockProblem()
        result = global_search(problem, method="spsa", max_evals=400, seed=5)
        assert result.method == "spsa"
        assert result.success
        assert np.all(np.isfinite(result.base.phi_best))
        assert result.base.loss_best < 5.0  # приближается к минимуму
        assert result.n_evals <= 400
        assert problem.n_forward == result.n_evals + 1

    def test_evaluator_stub_batches_candidates(self):
        """Внешний evaluator: пакеты кандидатов, main-problem не считает."""
        worker = RosenbrockProblem()
        calls = []

        def evaluator(phis):
            calls.append(len(phis))
            return [worker.evaluate(np.asarray(p)).loss for p in phis]

        problem = RosenbrockProblem()
        result = global_search(
            problem,
            method="spsa",
            max_evals=200,
            seed=9,
            evaluator=evaluator,
        )
        assert calls, "evaluator не использовался"
        assert all(n in (1, 2) for n in calls)  # пары SPSA (хвост — по одному)
        assert sum(calls) == result.n_evals
        # Вычисления ушли в «воркер»: у main-проблемы только финальная оценка.
        assert worker.n_forward == result.n_evals
        assert problem.n_forward == 1

        reference = global_search(
            RosenbrockProblem(), method="spsa", max_evals=200, seed=9
        )
        assert result.base.loss_best == pytest.approx(
            reference.base.loss_best, rel=1e-12, abs=1e-12
        )
        np.testing.assert_allclose(
            result.base.phi_best, reference.base.phi_best, atol=1e-10
        )

    def test_nan_counts_failures(self):
        problem = RosenbrockProblem(fail_neg_x0=True)
        result = global_search(
            problem,
            method="spsa",
            max_evals=60,
            seed=2,
            phi0=[-1.0, -1.0],
        )
        failed = [h for h in result.history if h["failed"]]
        assert len(failed) > 0
        assert all(h["loss"] == float("inf") for h in failed)
        assert result.n_evals == len(result.history)
        # Все оценки отказные: лучшей точки нет, success == False.
        assert not result.success
        assert result.base.loss_best == float("inf")
        assert result.n_evals <= 60

    def test_checkpoint_resume_matches_continuous(self, tmp_path):
        path = tmp_path / "spsa_checkpoint.json"
        with pytest.raises(KeyboardInterrupt, match="simulated"):
            global_search(
                RosenbrockProblem(interrupt_at=41),
                method="spsa",
                max_evals=200,
                seed=17,
                checkpoint_path=path,
            )
        assert path.exists()
        ckpt = RunCheckpoint.load(path)
        assert ckpt.seed == 17
        assert ckpt.config["method"] == "spsa"
        assert ckpt.config["max_evals"] == 200
        assert len(ckpt.completed) == 21  # k=0 + 20 завершённых итераций
        assert ckpt.pending == []  # открытый ask/tell-прогон (документировано)
        assert ckpt.best_index >= 0
        assert ckpt.completed[-1]["k"] == 20

        resumed = global_search(
            RosenbrockProblem(),
            method="spsa",
            max_evals=200,
            seed=17,
            checkpoint_path=path,
            resume=True,
        )
        reference = global_search(
            RosenbrockProblem(), method="spsa", max_evals=200, seed=17
        )
        assert resumed.n_evals == reference.n_evals == 199
        assert resumed.base.loss_best == pytest.approx(
            reference.base.loss_best, rel=1e-12, abs=1e-12
        )
        np.testing.assert_allclose(
            resumed.base.phi_best, reference.base.phi_best, atol=1e-10
        )
        assert resumed.message == reference.message
        assert [h["loss"] for h in resumed.history] == [
            h["loss"] for h in reference.history
        ]

    def test_resume_seed_mismatch_rejected(self, tmp_path):
        path = tmp_path / "spsa_checkpoint.json"
        with pytest.raises(KeyboardInterrupt, match="simulated"):
            global_search(
                RosenbrockProblem(interrupt_at=11),
                method="spsa",
                max_evals=100,
                seed=17,
                checkpoint_path=path,
            )
        with pytest.raises(ValueError, match="seed"):
            global_search(
                RosenbrockProblem(),
                method="spsa",
                max_evals=100,
                seed=18,
                checkpoint_path=path,
                resume=True,
            )


# ---------------------------------------------------------------------------
# CMA-ES (optional extra 'global')
# ---------------------------------------------------------------------------


class TestCMAES:
    def test_missing_dependency_message(self):
        try:
            import cma  # noqa: F401
        except ImportError:
            with pytest.raises(
                (OptionalDependencyError, ImportError), match=r"neuraltide\[global\]"
            ):
                global_search(RosenbrockProblem(), method="cma", max_evals=50, seed=1)
        else:
            pytest.skip("пакет cma установлен — отрицательная ветка недостижима")

    def test_runs_if_available(self):
        pytest.importorskip("cma")
        problem = RosenbrockProblem()
        result = global_search(
            problem,
            method="cma",
            max_evals=120,
            seed=4,
            options={"sigma0": 1.5, "popsize": 8},
        )
        assert result.method == "cma"
        assert result.success
        assert result.base.loss_best < 50.0
        assert result.n_evals <= 120
        assert problem.n_forward == result.n_evals + 1

    def test_checkpoint_resume_if_available(self, tmp_path):
        pytest.importorskip("cma")
        path = tmp_path / "cma_checkpoint.json"
        with pytest.raises(KeyboardInterrupt, match="simulated"):
            global_search(
                RosenbrockProblem(interrupt_at=20),
                method="cma",
                max_evals=100,
                seed=6,
                options={"sigma0": 1.5, "popsize": 8},
                checkpoint_path=path,
            )
        assert path.exists()
        ckpt = RunCheckpoint.load(path)
        assert ckpt.config["method"] == "cma"
        assert len(ckpt.completed) >= 1

        resumed = global_search(
            RosenbrockProblem(),
            method="cma",
            max_evals=100,
            seed=6,
            options={"sigma0": 1.5, "popsize": 8},
            checkpoint_path=path,
            resume=True,
        )
        reference = global_search(
            RosenbrockProblem(),
            method="cma",
            max_evals=100,
            seed=6,
            options={"sigma0": 1.5, "popsize": 8},
        )
        assert resumed.n_evals == reference.n_evals
        assert resumed.base.loss_best == pytest.approx(
            reference.base.loss_best, rel=1e-12, abs=1e-12
        )
        np.testing.assert_allclose(
            resumed.base.phi_best, reference.base.phi_best, atol=1e-10
        )


# ---------------------------------------------------------------------------
# NSGA-II (optional extra 'pareto')
# ---------------------------------------------------------------------------


class TestNSGA2:
    def test_missing_dependency_message(self):
        try:
            import pymoo  # noqa: F401
        except ImportError:
            with pytest.raises(
                (OptionalDependencyError, ImportError), match=r"neuraltide\[pareto\]"
            ):
                global_search(
                    TwoObjectiveProblem(),
                    method="nsga2",
                    max_evals=40,
                    seed=1,
                )
        else:
            pytest.skip("пакет pymoo установлен — отрицательная ветка недостижима")

    def test_needs_two_objectives(self):
        problem = RosenbrockProblem()  # одна компонента 'resid'
        with pytest.raises(ValueError, match=">= 2"):
            global_search(problem, method="nsga2", max_evals=20, seed=1)

    def test_runs_if_available(self, tmp_path):
        pytest.importorskip("pymoo")
        problem = TwoObjectiveProblem()
        result = global_search(
            problem,
            method="nsga2",
            max_evals=48,
            seed=2,
            options={"pop_size": 12},
        )
        assert isinstance(result, ParetoResult)
        assert result.method == "nsga2"
        assert result.success
        assert sorted(result.objective_names) == ["f1", "f2"]
        n_front = len(result.frontier["f1"])
        assert n_front == len(result.frontier["f2"]) >= 1
        assert result.n_evals <= 48
        # Сериализация ParetoResult.
        out = tmp_path / "pareto.json"
        result.save(out)
        loaded = ParetoResult.load(out)
        assert loaded.objective_names == result.objective_names
        np.testing.assert_array_equal(loaded.frontier["f1"], result.frontier["f1"])
        assert loaded.n_evals == result.n_evals


# ---------------------------------------------------------------------------
# _Oracle: учёт кандидатов evaluate_components (без pymoo)
# ---------------------------------------------------------------------------


class TestOracleEvaluateComponents:
    def test_successful_candidate_scored_once(self):
        """Успешный кандидат — ровно одна запись в _scored (через _commit);
        F_history ведётся отдельно для Pareto-фронта."""
        problem = TwoObjectiveProblem()
        oracle = _Oracle(problem, max_evals=10, evaluator=None)
        phis = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]
        F = oracle.evaluate_components(phis, ["f1", "f2"])

        assert F.shape == (2, 2)
        assert np.all(np.isfinite(F))
        assert oracle.n_evals == 2
        assert len(oracle._scored) == 2  # без дублирования
        assert len(oracle.F_history) == 2
        np.testing.assert_array_equal(oracle._scored[0][1], phis[0])

    def test_failed_candidate_not_scored(self):
        """Отказ (NaN-компонента) — строка +inf, в _scored не попадает."""
        problem = RosenbrockProblem(fail_neg_x0=True)
        oracle = _Oracle(problem, max_evals=10, evaluator=None)
        F = oracle.evaluate_components([np.array([-1.0, 0.5])], ["resid"])

        assert F.shape == (1, 1)
        assert not np.all(np.isfinite(F[0]))  # NaN-компонента -> не конечная цель
        assert oracle._scored == []
        assert oracle.F_history == []
        assert oracle.n_evals == 1
        assert oracle.n_failed == 1
        assert oracle.history[0]["failed"] is True


# ---------------------------------------------------------------------------
# Общие проверки: постановка, сериализация, L0-задача
# ---------------------------------------------------------------------------


class TestContract:
    def test_unknown_method_rejected(self):
        with pytest.raises(ValueError, match="unknown-method"):
            global_search(RosenbrockProblem(), method="unknown-method")

    def test_unknown_options_rejected(self):
        with pytest.raises(ValueError, match="неизвестные options"):
            global_search(
                RosenbrockProblem(),
                method="spsa",
                options={"no_such_option": 1},
            )

    def test_nsga2_rejects_refine_and_evaluator(self):
        with pytest.raises(ValueError, match="refine_top_k"):
            global_search(
                TwoObjectiveProblem(),
                method="nsga2",
                refine_top_k=1,
            )
        with pytest.raises(ValueError, match="evaluator"):
            global_search(
                TwoObjectiveProblem(),
                method="nsga2",
                evaluator=lambda phis: [0.0 for _ in phis],
            )

    def test_checkpoint_rejected_for_de(self):
        with pytest.raises(ValueError, match="resume"):
            global_search(
                RosenbrockProblem(),
                method="de",
                checkpoint_path="x.json",
            )

    def test_result_serialization_roundtrip(self, tmp_path):
        result = global_search(RosenbrockProblem(), method="spsa", max_evals=40, seed=8)
        out = tmp_path / "global_result.json"
        result.save(out)
        loaded = GlobalOptimizerResult.load(out)
        assert loaded.method == result.method
        assert loaded.n_evals == result.n_evals
        assert loaded.base.loss_best == pytest.approx(
            result.base.loss_best, rel=1e-12, abs=1e-12
        )
        np.testing.assert_array_equal(loaded.base.phi_best, result.base.phi_best)
        # to_dict JSON-сериализуем.
        json.dumps(result.to_dict())

    def test_pareto_result_direct_serialization(self, tmp_path):
        base = OptimizationResult(
            phi_best=np.array([0.5, 0.5]),
            theta_best={"x": np.array([0.5])},
            loss_best=1.5,
            seed=1,
        )
        result = ParetoResult(
            base=base,
            frontier={"f1": np.array([0.1, 0.5]), "f2": np.array([0.9, 0.2])},
            objective_names=["f1", "f2"],
            n_evals=10,
            wall_time=0.25,
        )
        out = tmp_path / "pareto_direct.json"
        result.save(out)
        loaded = ParetoResult.load(out)
        assert loaded.objective_names == ["f1", "f2"]
        np.testing.assert_array_equal(loaded.frontier["f1"], [0.1, 0.5])
        assert loaded.n_evals == 10
        json.dumps(result.to_dict())


class TestL0Problem:
    """Малый бюджет на реальной L0 residual-задаче (фикстура l0_problem)."""

    @pytest.mark.parametrize("method", ["spsa", "basinhopping"])
    def test_global_methods_on_l0(self, l0_problem, method):
        result = global_search(
            l0_problem,
            method=method,
            max_evals=60,
            seed=3,
            options={"inner_maxiter": 20} if method == "basinhopping" else None,
        )
        assert result.success
        assert np.isfinite(result.base.loss_best)
        assert result.n_evals <= 60
        # Все оценки кандидатов через problem.evaluate + финальная явная.
        assert l0_problem.n_forward == result.n_evals + 1
        assert result.n_forward == l0_problem.n_forward
        assert result.n_gradient == l0_problem.n_gradient
        assert len(result.history) == result.n_evals
        assert result.base.dtype == "float32"
        expected = set(l0_problem.parameter_space.entries)
        assert set(result.base.theta_best) == expected

    def test_de_rejected_on_l0_unbounded_entry(self, l0_problem):
        # exc/I_ext обучаем без min/max -> границы не конечны -> DE отказ.
        with pytest.raises(ValueError, match="sample_phi0s"):
            global_search(l0_problem, method="de", max_evals=50, seed=1)
