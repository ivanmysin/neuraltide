"""
Тесты P1: DTO результатов (LocalOptimizerResult, StartResult,
MultistartResult, RunCheckpoint, JacobianResult) — round-trip
to_dict/from_dict/save/load, JSON-сериализуемость, numpy -> list,
вложенность; PEP-562 __getattr__ пакета neuraltide.optimization.
"""

import json

import numpy as np
import pytest

import neuraltide.optimization as opt
from neuraltide.optimization import (
    JacobianResult,
    LocalOptimizerResult,
    MultistartResult,
    OptimizationResult,
    RunCheckpoint,
    StartResult,
)


def _base(seed=0):
    return OptimizationResult(
        phi_best=np.array([0.1, -0.2, 0.3]),
        theta_best={"exc/tau": np.array([8.0]), "x0/exc/0": np.array([[0.0]])},
        loss_best=0.5,
        history=[{"loss": 1.0, "epoch": 0}],
        seed=seed,
        config={"dt": 0.5, "grad_mode": "assign"},
        dtype="float64",
    )


def _assert_base_equal(restored, ref):
    np.testing.assert_allclose(restored.phi_best, ref.phi_best)
    np.testing.assert_allclose(restored.theta_best["exc/tau"], [8.0])
    assert restored.loss_best == pytest.approx(ref.loss_best)
    assert restored.seed == ref.seed
    assert restored.config == ref.config
    assert restored.dtype == ref.dtype


class TestLocalOptimizerResult:
    def test_roundtrip(self, tmp_path):
        ref = LocalOptimizerResult(
            base=_base(seed=1),
            method="L-BFGS-B",
            success=True,
            status=0,
            message="CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH",
            nit=12,
            nfev=30,
            njev=30,
            grad_norm=1e-4,
            bounds_active=["exc/tau"],
            n_forward=31,
            n_gradient=30,
            wall_time=1.5,
            validation={"held": 0.3, "held2": 0.7},
        )
        data = ref.to_dict()
        json.dumps(data)
        restored = LocalOptimizerResult.from_dict(data)
        _assert_base_equal(restored.base, ref.base)
        assert restored.method == "L-BFGS-B"
        assert restored.success is True
        assert restored.status == 0
        assert restored.nit == 12 and restored.nfev == 30 and restored.njev == 30
        assert restored.grad_norm == pytest.approx(1e-4)
        assert restored.bounds_active == ["exc/tau"]
        assert restored.n_forward == 31 and restored.n_gradient == 30
        assert restored.wall_time == pytest.approx(1.5)
        assert restored.validation == {"held": 0.3, "held2": 0.7}

        path = tmp_path / "local.json"
        ref.save(path)
        restored2 = LocalOptimizerResult.load(path)
        assert restored2.nit == 12
        _assert_base_equal(restored2.base, ref.base)


class TestStartResult:
    def test_roundtrip_without_local(self):
        ref = StartResult(
            index=3,
            x0_origin="lhs",
            phi0=np.array([0.1, 0.2]),
            phi=np.array([0.15, 0.25]),
            loss=0.7,
            status="screened",
            message="loss above threshold",
            screened_loss=0.7,
            refined=False,
            local=None,
            wall_time=0.25,
        )
        data = ref.to_dict()
        json.dumps(data)
        restored = StartResult.from_dict(data)
        assert restored.index == 3
        assert restored.x0_origin == "lhs"
        np.testing.assert_allclose(restored.phi0, [0.1, 0.2])
        np.testing.assert_allclose(restored.phi, [0.15, 0.25])
        assert restored.loss == pytest.approx(0.7)
        assert restored.status == "screened"
        assert restored.screened_loss == pytest.approx(0.7)
        assert restored.refined is False
        assert restored.local is None
        assert restored.wall_time == pytest.approx(0.25)

    def test_roundtrip_with_nested_local(self, tmp_path):
        local = LocalOptimizerResult(
            base=_base(seed=2),
            method="TRF",
            success=True,
            status=2,
            nit=8,
            validation={"held": 0.4},
        )
        ref = StartResult(
            index=0,
            x0_origin="sobol",
            phi0=np.zeros(3),
            phi=np.array([0.3, 0.1, -0.2]),
            loss=0.25,
            status="ok",
            refined=True,
            local=local,
            wall_time=4.0,
        )
        data = ref.to_dict()
        json.dumps(data)
        restored = StartResult.from_dict(data)
        assert restored.refined is True
        assert restored.local is not None
        assert restored.local.method == "TRF"
        assert restored.local.nit == 8
        _assert_base_equal(restored.local.base, local.base)

        path = tmp_path / "start.json"
        ref.save(path)
        restored2 = StartResult.load(path)
        assert restored2.local.method == "TRF"
        np.testing.assert_allclose(restored2.phi, ref.phi)


class TestMultistartResult:
    def test_roundtrip(self, tmp_path):
        solutions = [
            StartResult(
                index=0,
                x0_origin="lhs",
                phi0=np.zeros(2),
                phi=np.array([0.1, 0.1]),
                loss=0.5,
                status="ok",
                refined=True,
                local=LocalOptimizerResult(base=_base(seed=10), method="LM", nit=5),
                wall_time=2.0,
            ),
            StartResult(
                index=1,
                x0_origin="lhs",
                phi0=np.ones(2),
                phi=np.ones(2),
                loss=3.0,
                status="failed",
                message="NaN in simulation",
                refined=False,
                wall_time=0.5,
            ),
        ]
        ref = MultistartResult(
            base=_base(seed=42),
            solutions=solutions,
            best_index=0,
            n_starts=2,
            n_refined=1,
            n_failed=1,
            seed=42,
            clustering={"n_clusters": 2, "labels": [0, 1]},
            wall_time=2.5,
        )
        data = ref.to_dict()
        json.dumps(data)
        restored = MultistartResult.from_dict(data)
        _assert_base_equal(restored.base, ref.base)
        assert len(restored.solutions) == 2
        assert restored.solutions[0].local.method == "LM"
        assert restored.solutions[1].status == "failed"
        assert restored.solutions[1].local is None
        assert restored.best_index == 0
        assert restored.n_starts == 2
        assert restored.n_refined == 1
        assert restored.n_failed == 1
        assert restored.seed == 42
        assert restored.clustering == {"n_clusters": 2, "labels": [0, 1]}
        assert restored.wall_time == pytest.approx(2.5)

        path = tmp_path / "multistart.json"
        ref.save(path)
        restored2 = MultistartResult.load(path)
        assert restored2.seed == 42
        assert len(restored2.solutions) == 2


class TestRunCheckpoint:
    def test_roundtrip(self, tmp_path):
        ref = RunCheckpoint(
            seed=7,
            config={"method": "lhs", "n_starts": 8, "budget": {"fev": 100}},
            completed=[0, 1, {"index": 2, "loss": 0.4}],
            pending=[3, 4],
            best_index=2,
        )
        data = ref.to_dict()
        json.dumps(data)
        restored = RunCheckpoint.from_dict(data)
        assert restored.seed == 7
        assert restored.config == ref.config
        assert restored.completed == ref.completed
        assert restored.pending == [3, 4]
        assert restored.best_index == 2

        path = tmp_path / "checkpoint.json"
        ref.save(path)
        restored2 = RunCheckpoint.load(path)
        assert restored2.completed == ref.completed


class TestJacobianResult:
    def test_roundtrip(self, tmp_path):
        ref = JacobianResult(
            J=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            r=np.array([0.1, -0.2, 0.3]),
            loss=0.5,
            phi=np.array([0.7, 0.8]),
            n_forward=3,
            wall_time=0.75,
            method="forward",
        )
        data = ref.to_dict()
        json.dumps(data)
        restored = JacobianResult.from_dict(data)
        np.testing.assert_allclose(restored.J, ref.J)
        np.testing.assert_allclose(restored.r, ref.r)
        np.testing.assert_allclose(restored.phi, ref.phi)
        assert restored.loss == pytest.approx(0.5)
        assert restored.n_forward == 3
        assert restored.wall_time == pytest.approx(0.75)
        assert restored.method == "forward"

        path = tmp_path / "jacobian.json"
        ref.save(path)
        restored2 = JacobianResult.load(path)
        np.testing.assert_allclose(restored2.J, ref.J)


class TestPublicApi:
    def test_new_names_available(self):
        assert opt.StateLocator is not None
        assert opt.Experiment is not None
        assert opt.MultiExperimentProblem is not None
        assert opt.FrozenProblem is not None
        assert opt.LocalOptimizerResult is not None
        assert opt.MultistartResult is not None
        for name in (
            "StateLocator",
            "Experiment",
            "MultiExperimentProblem",
            "FrozenProblem",
            "LocalOptimizerResult",
            "StartResult",
            "MultistartResult",
            "RunCheckpoint",
            "JacobianResult",
        ):
            assert name in opt.__all__

    def test_unknown_name_raises_attribute_error(self):
        with pytest.raises(AttributeError):
            opt.not_a_real_name
        # имя вне публичного API не импортируется и не даёт ImportError;
        # гарантированно несуществующее — чтобы не зависеть от того,
        # привязан ли реальный подмодуль к пакету чужим импортом
        with pytest.raises(AttributeError):
            opt.definitely_not_a_p1_module_xyz
