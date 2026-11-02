"""
Тесты P1: выборка начальных точек (sample_phi0s) и мультистарт.

Логика мультистарта проверяется на duck-typed 1D двуминимальной задаче
без TensorFlow: f(x) = 0.05*(x-3)^2 + sin(pi x)^2 на [-4, 4] — локальные
минимумы в целых x=k со значением 0.05*(k-3)^2, глобальный — x=3;
и f(x) = (x^2-1)^2 на [-3, 3] — два равных минимума x=±1 для проверки
кластеризации. Для доводки используются настоящие адаптеры local.py
(minimize_lbfgsb поверх duck value_and_gradient). sample_phi0s проверяется
на реальном ParameterSpace L0-сети (фикстура l0_problem).

Resume: прерывание симулируется KeyboardInterrupt из callable-локального
оптимизатора (мультистарт ловит только Exception, BaseException
пролетает); дозапуск с resume=True обязан воспроизводить непрерывный
прогон для тех же (seed, n_starts).
"""

import json

import numpy as np
import pytest

from neuraltide.optimization import ParameterEntry, ParameterSpace
from neuraltide.optimization.local import minimize_lbfgsb
from neuraltide.optimization.results import (
    MultistartResult,
    RunCheckpoint,
    StartResult,
)
from neuraltide.optimization.strategies import multistart, sample_phi0s


class _Eval:
    """Минимальный Evaluation-заглушка (только loss)."""

    def __init__(self, loss):
        self.loss = float(loss)


class TwoMinimaProblem:
    """
    1D двуминимальная задача: f(x) = 0.05*(x-3)^2 + sin(pi x)^2, x в [-4, 4].

    Локальные минимумы в целых x=k (значение 0.05*(k-3)^2); глобальный
    минимум — x=3 (loss=0). bounds_phi конечны, поэтому дизайн стартов —
    равномерный в границах (duck-путь без ParameterSpace).
    """

    size = 1
    held_out = []

    def __init__(self):
        self.n_forward = 0
        self.n_gradient = 0

    @staticmethod
    def _f(x):
        return 0.05 * (x - 3.0) ** 2 + np.sin(np.pi * x) ** 2

    @staticmethod
    def _grad(x):
        return 0.1 * (x - 3.0) + 2.0 * np.pi * np.sin(np.pi * x) * np.cos(np.pi * x)

    def pack(self, phi=None):
        if phi is None:
            return np.zeros(1, dtype=np.float64)
        return np.asarray(phi, dtype=np.float64).reshape(-1)

    def bounds_phi(self):
        return np.array([-4.0]), np.array([4.0])

    def unpack(self, phi):
        return {"x": np.asarray(phi, dtype=np.float64).reshape(())}

    def evaluate(self, phi):
        self.n_forward += 1
        x = float(np.asarray(phi, dtype=np.float64).reshape(-1)[0])
        return _Eval(self._f(x))

    def value_and_gradient(self, phi):
        self.n_forward += 1
        self.n_gradient += 1
        x = float(np.asarray(phi, dtype=np.float64).reshape(-1)[0])
        return _Eval(self._f(x)), np.array([self._grad(x)])


class TwinWellProblem(TwoMinimaProblem):
    """f(x) = (x^2-1)^2 на [-3, 3]: два равных глобальных минимума x=±1."""

    @staticmethod
    def _f(x):
        return (x**2 - 1.0) ** 2

    @staticmethod
    def _grad(x):
        return 4.0 * x * (x**2 - 1.0)

    def bounds_phi(self):
        return np.array([-3.0]), np.array([3.0])


def _interrupting_local(limit):
    """Callable-оптимизатор: первые limit-1 вызовов делегируют L-BFGS-B,
    далее KeyboardInterrupt (симуляция прерывания прогона). __name__
    совпадает с 'lbfgsb', чтобы конфиг чекпоинта соответствовал
    строковому адаптеру при resume."""
    calls = {"n": 0}

    def local(problem, phi0):
        calls["n"] += 1
        if calls["n"] >= limit:
            raise KeyboardInterrupt("simulated interruption")
        return minimize_lbfgsb(problem, phi0)

    local.__name__ = "lbfgsb"
    return local


# ---------------------------------------------------------------------------
# sample_phi0s
# ---------------------------------------------------------------------------


class TestSamplePhi0s:
    def test_deterministic_same_seed(self, l0_problem):
        space = l0_problem.parameter_space
        for method in ("sobol", "lhs", "random"):
            a = sample_phi0s(space, 8, method=method, seed=3)
            b = sample_phi0s(space, 8, method=method, seed=3)
            np.testing.assert_array_equal(a, b)
            assert a.shape == (8, space.size)
            assert a.dtype == np.float64

    def test_methods_give_different_designs(self, l0_problem):
        space = l0_problem.parameter_space
        with pytest.warns(UserWarning):
            sobol = sample_phi0s(space, 8, method="sobol", seed=3)
            lhs = sample_phi0s(space, 8, method="lhs", seed=3)
            random = sample_phi0s(space, 8, method="random", seed=3)
        assert not np.allclose(sobol, lhs)
        assert not np.allclose(sobol, random)

    def test_unknown_method_rejected(self, l0_problem):
        with pytest.raises(ValueError, match="method"):
            sample_phi0s(l0_problem.parameter_space, 4, method="halton")

    def test_bounds_respected_for_all_methods(self, l0_problem):
        space = l0_problem.parameter_space
        for method in ("sobol", "lhs", "random"):
            with pytest.warns(UserWarning):
                phi = sample_phi0s(space, 16, method=method, seed=1)
            assert np.all(np.isfinite(phi))
            for row in phi:
                theta = space.unpack(row)
                tau = float(np.asarray(theta["exc/tau"]).reshape(-1)[0])
                gsyn = float(np.asarray(theta["in->exc/gsyn_max"]).reshape(-1)[0])
                assert 1.0 <= tau <= 50.0
                assert 0.001 <= gsyn <= 1.0

    def test_unbounded_entries_warn(self, l0_problem):
        space = l0_problem.parameter_space
        with pytest.warns(UserWarning, match="exc/I_ext"):
            sample_phi0s(space, 4, seed=0)

    def test_clip_to_bounds(self, l0_problem):
        space = l0_problem.parameter_space
        with pytest.warns(UserWarning):
            clipped = sample_phi0s(space, 8, seed=5, clip_to_bounds=True)
            raw = sample_phi0s(space, 8, seed=5, clip_to_bounds=False)
        assert clipped.shape == raw.shape == (8, space.size)
        assert np.all(np.isfinite(clipped))
        assert np.all(np.isfinite(raw))
        for row in clipped:
            theta = space.unpack(row)
            assert 1.0 <= float(np.asarray(theta["exc/tau"]).reshape(-1)[0]) <= 50.0

    def test_log_uniform_sampling(self, l0_problem):
        space = l0_problem.parameter_space
        phi = sample_phi0s(space, 32, seed=2, log_uniform=["exc/tau"])
        values = []
        for row in phi:
            theta = space.unpack(row)
            values.append(float(np.asarray(theta["exc/tau"]).reshape(-1)[0]))
        values = np.array(values)
        assert np.all(values >= 1.0) and np.all(values <= 50.0)
        # Лог-равномерный дизайн покрывает диапазон шире равномерного.
        assert values.max() / values.min() > 5.0

    def test_log_uniform_unknown_name(self, l0_problem):
        with pytest.raises(ValueError, match="log_uniform"):
            sample_phi0s(l0_problem.parameter_space, 4, log_uniform=["no/such_param"])

    def test_log_uniform_unbounded_rejected(self, l0_problem):
        # exc/I_ext без границ — лог-равномер невозможен.
        with pytest.raises(ValueError, match="конечных"):
            sample_phi0s(l0_problem.parameter_space, 4, log_uniform=["exc/I_ext"])

    def test_log_uniform_nonpositive_min_rejected(self):
        space = ParameterSpace()
        space.add_entry(
            ParameterEntry(
                full_name="noise/sigma",
                shape=[1],
                min=0.0,
                max=1.0,
                group="noise",
            )
        )
        with pytest.raises(ValueError, match="min > 0"):
            sample_phi0s(space, 2, log_uniform=["noise/sigma"])


# ---------------------------------------------------------------------------
# multistart
# ---------------------------------------------------------------------------


class TestMultistart:
    def test_best_beats_median_of_single_starts(self):
        problem = TwoMinimaProblem()
        res = multistart(problem, n_starts=24, refine_top_k=6, seed=0)

        assert res.n_starts == 24
        assert len(res.solutions) == 24  # ВСЕ старты сохранены
        statuses = {s.status for s in res.solutions}
        assert statuses <= {"ok", "screen_rejected", "failed"}
        assert any(s.status == "ok" for s in res.solutions)
        assert any(s.status == "screen_rejected" for s in res.solutions)
        assert res.n_failed == 0
        assert res.n_refined == 6
        assert res.best_index >= 0

        best = res.solutions[res.best_index]
        assert best.refined and best.local is not None
        assert best.local.method == "L-BFGS-B"
        assert best.loss <= best.screened_loss + 1e-12
        # Доводка нашла глобальный минимум (x=3, loss=0).
        assert res.base.loss_best < 1e-8
        assert abs(float(res.base.theta_best["x"]) - 3.0) < 1e-3

        # Медиана одиночных доводок со всех тех же стартов.
        single = [
            minimize_lbfgsb(problem, s.phi0).base.loss_best for s in res.solutions
        ]
        assert res.base.loss_best <= float(np.median(single))

    def test_top_k_only_refined(self):
        problem = TwoMinimaProblem()
        res = multistart(problem, n_starts=16, refine_top_k=3, seed=1)
        refined = [s for s in res.solutions if s.refined]
        assert len(refined) == 3
        assert all(s.status == "ok" and s.local is not None for s in refined)
        assert all(
            s.local is None and s.screened_loss is not None
            for s in res.solutions
            if s.status == "screen_rejected"
        )
        # Эконом прогонов: ровно n_starts screening-оценок.
        assert problem.n_forward >= 16

    def test_failed_local_does_not_abort_run(self):
        problem = TwoMinimaProblem()

        def flaky(problem, phi0):
            if abs(float(np.asarray(phi0).reshape(-1)[0]) - 3.0) < 0.6:
                raise RuntimeError("boom near global minimum")
            return minimize_lbfgsb(problem, phi0)

        res = multistart(problem, n_starts=24, refine_top_k=6, seed=0, local=flaky)
        failed = [s for s in res.solutions if s.status == "failed"]
        assert failed, "ожидался как минимум один упавший старт"
        assert all("boom" in s.message for s in failed)
        assert res.n_failed == len(failed)
        # Остальные старты доведены, лучший найден. Старт в бассейне x=3
        # отвергнут, поэтому лучший доступный минимум — x=4 (loss=0.05).
        assert res.n_refined == 6 - len(failed)
        assert res.best_index >= 0
        assert res.solutions[res.best_index].status == "ok"
        assert res.base.loss_best <= 0.05 + 1e-9
        assert abs(float(res.base.theta_best["x"]) - 4.0) < 0.05

    def test_screening_failure_marks_start_failed(self):
        class _Exploding(TwoMinimaProblem):
            def evaluate(self, phi):
                x = float(np.asarray(phi, dtype=np.float64).reshape(-1)[0])
                if abs(x - 3.0) < 0.3:
                    raise RuntimeError("simulated simulation failure")
                return super().evaluate(phi)

        problem = _Exploding()
        res = multistart(problem, n_starts=16, refine_top_k=4, seed=0)
        failed_screen = [
            s for s in res.solutions if s.status == "failed" and s.screened_loss is None
        ]
        assert failed_screen, "ожидался отказ screening ближайшего к x=3 старта"
        assert all("screening" in s.message for s in failed_screen)
        # Провал screening исключает старт из доводки, но не роняет прогон:
        # доводятся top-k среди успешно отскринированных.
        n_screened_ok = sum(1 for s in res.solutions if s.screened_loss is not None)
        assert res.n_refined == min(4, n_screened_ok)
        assert res.best_index >= 0
        assert res.solutions[res.best_index].status == "ok"

    def test_counters_accumulate(self):
        problem = TwoMinimaProblem()
        res = multistart(problem, n_starts=12, refine_top_k=4, seed=1)
        assert problem.n_forward >= 12  # screening: по evaluate на старт
        assert problem.n_gradient >= 4  # доводка: градиенты L-BFGS-B
        assert res.wall_time > 0.0
        assert len(res.base.history) == 12
        entry = res.base.history[0]
        assert {"index", "status", "screened_loss", "loss", "refined"} <= set(entry)

    def test_clustering_two_equal_wells(self):
        problem = TwinWellProblem()
        res = multistart(problem, n_starts=16, refine_top_k=8, seed=2)
        clustering = res.clustering
        labels = clustering["labels"]
        assert len(labels) == 16
        ok = [i for i, s in enumerate(res.solutions) if s.status == "ok"]
        assert labels.count(-1) == 16 - len(ok)
        # Два равных минимума x=±1 -> два кластера.
        assert clustering["n_clusters"] == 2
        ok_labels = {labels[i] for i in ok}
        assert ok_labels == {0, 1}
        reps = clustering["representatives"]
        assert len(reps) == 2
        assert res.solutions[reps[0]].loss <= res.solutions[reps[1]].loss
        phi_a = res.solutions[reps[0]].phi
        phi_b = res.solutions[reps[1]].phi
        assert abs(float(phi_a[0]) - float(phi_b[0])) > 1.0

    def test_dedup_rtol_merges_refined_copies(self):
        # Один и тот же минимум из разных стартов -> один кластер.
        problem = TwoMinimaProblem()
        res = multistart(problem, n_starts=16, refine_top_k=8, seed=4)
        assert res.clustering["n_clusters"] < sum(
            1 for s in res.solutions if s.status == "ok"
        )

    def test_validation_errors(self):
        problem = TwoMinimaProblem()
        with pytest.raises(ValueError, match="n_starts"):
            multistart(problem, n_starts=0)
        with pytest.raises(ValueError, match="screen_evals"):
            multistart(problem, screen_evals=0)
        with pytest.raises(ValueError, match="refine_top_k"):
            multistart(problem, refine_top_k=-1)
        with pytest.raises(ValueError, match="dedup_rtol"):
            multistart(problem, dedup_rtol=0.0)
        with pytest.raises(ValueError, match="оптимизатор"):
            multistart(problem, local="cmaes")

    def test_zero_refine_top_k(self):
        problem = TwoMinimaProblem()
        res = multistart(problem, n_starts=8, refine_top_k=0, seed=0)
        assert res.n_refined == 0
        assert res.best_index == -1  # некому доводить
        assert all(s.status == "screen_rejected" for s in res.solutions)
        assert res.base.loss_best == min(s.loss for s in res.solutions)


# ---------------------------------------------------------------------------
# checkpoint / resume
# ---------------------------------------------------------------------------


class TestResume:
    ARGS = dict(n_starts=12, refine_top_k=3, seed=5)

    def test_resume_matches_continuous(self, tmp_path):
        path = tmp_path / "checkpoint.json"
        with pytest.raises(KeyboardInterrupt, match="simulated"):
            multistart(
                TwoMinimaProblem(),
                local=_interrupting_local(limit=2),
                checkpoint_path=path,
                **self.ARGS,
            )
        assert path.exists()
        ckpt = RunCheckpoint.load(path)
        # Все 12 стартов в чекпоинте; ровно одна запись доведена (limit=2:
        # первый вызов local отработал, второй прервал прогон).
        assert len(ckpt.completed) == 12
        refined = [c for c in ckpt.completed if c.get("refined")]
        assert len(refined) == 1
        assert len(ckpt.pending) == 0
        assert ckpt.seed == 5

        resumed = multistart(
            TwoMinimaProblem(), checkpoint_path=path, resume=True, **self.ARGS
        )
        reference = multistart(TwoMinimaProblem(), **self.ARGS)

        assert resumed.best_index == reference.best_index
        assert resumed.base.loss_best == pytest.approx(
            reference.base.loss_best, rel=1e-9, abs=1e-12
        )
        np.testing.assert_allclose(
            resumed.base.phi_best, reference.base.phi_best, atol=1e-8
        )
        assert [s.status for s in resumed.solutions] == [
            s.status for s in reference.solutions
        ]
        assert resumed.n_refined == reference.n_refined == 3

    def test_resume_with_empty_completed(self, tmp_path):
        # Чекпоинт без завершённых стартов эквивалентен чистому прогону.
        probe = tmp_path / "probe.json"
        multistart(TwoMinimaProblem(), checkpoint_path=probe, **self.ARGS)
        ckpt = RunCheckpoint.load(probe)
        manual_path = tmp_path / "manual.json"
        RunCheckpoint(
            seed=5,
            config=ckpt.config,
            completed=[],
            pending=list(range(12)),
            best_index=-1,
        ).save(manual_path)

        resumed = multistart(
            TwoMinimaProblem(), checkpoint_path=manual_path, resume=True, **self.ARGS
        )
        reference = multistart(TwoMinimaProblem(), **self.ARGS)
        assert resumed.best_index == reference.best_index
        assert resumed.base.loss_best == pytest.approx(
            reference.base.loss_best, rel=1e-9, abs=1e-12
        )

    def test_resume_seed_mismatch_rejected(self, tmp_path):
        path = tmp_path / "checkpoint.json"
        with pytest.raises(KeyboardInterrupt):
            multistart(
                TwoMinimaProblem(),
                local=_interrupting_local(limit=2),
                checkpoint_path=path,
                **self.ARGS,
            )
        with pytest.raises(ValueError, match="seed"):
            multistart(
                TwoMinimaProblem(),
                checkpoint_path=path,
                resume=True,
                n_starts=12,
                refine_top_k=3,
                seed=6,
            )

    def test_resume_config_mismatch_rejected(self, tmp_path):
        path = tmp_path / "checkpoint.json"
        with pytest.raises(KeyboardInterrupt):
            multistart(
                TwoMinimaProblem(),
                local=_interrupting_local(limit=2),
                checkpoint_path=path,
                **self.ARGS,
            )
        with pytest.raises(ValueError, match="конфигурация"):
            multistart(
                TwoMinimaProblem(),
                checkpoint_path=path,
                resume=True,
                n_starts=10,  # другой конфиг
                refine_top_k=3,
                seed=5,
            )

    def test_fresh_run_overwrites_checkpoint(self, tmp_path):
        path = tmp_path / "checkpoint.json"
        res = multistart(TwoMinimaProblem(), checkpoint_path=path, **self.ARGS)
        assert path.exists()
        ckpt = RunCheckpoint.load(path)
        assert len(ckpt.completed) == 12
        assert ckpt.best_index == res.best_index


# ---------------------------------------------------------------------------
# Сериализация MultistartResult
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_roundtrip(self, tmp_path):
        res = multistart(TwoMinimaProblem(), n_starts=8, refine_top_k=2, seed=4)
        data = res.to_dict()
        json.dumps(data)
        restored = MultistartResult.from_dict(data)

        assert restored.n_starts == res.n_starts
        assert restored.best_index == res.best_index
        assert restored.n_refined == res.n_refined
        assert restored.n_failed == res.n_failed
        assert restored.seed == res.seed
        assert restored.clustering == res.clustering
        assert restored.wall_time == pytest.approx(res.wall_time)
        np.testing.assert_array_equal(restored.base.phi_best, res.base.phi_best)
        assert restored.base.loss_best == pytest.approx(res.base.loss_best)
        assert restored.base.seed == res.base.seed
        assert restored.base.config == res.base.config
        assert len(restored.solutions) == len(res.solutions)
        for got, ref in zip(restored.solutions, res.solutions):
            assert got.status == ref.status
            assert got.refined == ref.refined
            np.testing.assert_array_equal(got.phi0, ref.phi0)
            np.testing.assert_array_equal(got.phi, ref.phi)
            if ref.local is not None:
                assert got.local.method == ref.local.method
                assert got.local.nit == ref.local.nit

        path = tmp_path / "multistart.json"
        res.save(path)
        loaded = MultistartResult.load(path)
        assert loaded.best_index == res.best_index
        assert loaded.base.loss_best == pytest.approx(res.base.loss_best)

    def test_start_result_checkpoint_roundtrip(self, tmp_path):
        start = StartResult(
            index=2,
            x0_origin="sobol",
            phi0=np.array([0.1]),
            phi=np.array([0.2]),
            loss=0.5,
            status="ok",
            screened_loss=0.7,
            refined=True,
            wall_time=0.1,
        )
        ckpt = RunCheckpoint(
            seed=1,
            config={"n_starts": 3},
            completed=[start.to_dict()],
            pending=[0, 1],
            best_index=2,
        )
        data = ckpt.to_dict()
        json.dumps(data)
        restored = RunCheckpoint.from_dict(data)
        assert restored.completed == ckpt.completed
        assert restored.pending == [0, 1]
        restored_start = StartResult.from_dict(restored.completed[0])
        np.testing.assert_array_equal(restored_start.phi, [0.2])
