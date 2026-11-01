"""
Тесты CalibrationProblem / Evaluation / OptimizationResult (этап P0).

Сеть: одна популяция WilsonCowan (1 юнит) + постоянный вход через
StaticSynapse. Обучаемые параметры: exc/tau (log), exc/I_ext (identity),
in->exc/gsyn_max (log). Целевая траектория — прогон при «истинных»
параметрах.

Покрываются: pack/unpack roundtrip и валидация размера; evaluate(None)
против явного network+loss; taylor_test для обоих grad_mode; равенство
градиентов режимов; residual reconstruction; мутация модели по контракту;
счётчики и wall_time; float32/float64; bounds_phi; сериализация
OptimizationResult и Evaluation.to_dict.
"""

import json

import numpy as np
import pytest
import tensorflow as tf

import neuraltide.config
from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.integrators import EulerIntegrator
from neuraltide.inputs import ConstantRateGenerator
from neuraltide.optimization import (
    CalibrationProblem,
    LogTransform,
    ObservationSpec,
    OptimizationResult,
    ParameterSpace,
    ResidualLoss,
    taylor_test,
)
from neuraltide.populations import WilsonCowan
from neuraltide.synapses import StaticSynapse

DT = 0.5
T = 60


def _build():
    """Сеть, входы и ResidualLoss против «истинного» прогона."""
    pop = WilsonCowan(
        dt=DT,
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
    syn = StaticSynapse(
        n_pre=1,
        n_post=1,
        dt=DT,
        params={
            "gsyn_max": {
                "value": [[0.05]],
                "trainable": True,
                "min": 0.001,
                "max": 1.0,
            },
            "pconn": [[1.0]],
            "e_r": 0.0,
        },
        name="in->exc",
    )
    gen = ConstantRateGenerator(dt=DT, params={"rate": 10.0})

    graph = NetworkGraph(dt=DT)
    graph.declare_input("in", n_units=1)
    graph.add_population("exc", pop)
    graph.add_synapse("in->exc", syn, src="in", tgt="exc")
    network = NetworkRNN(graph, EulerIntegrator())

    t_seq = tf.constant((np.arange(T, dtype=np.float32) * DT)[None, :, None])
    inputs = graph.pack_inputs({"in": gen(t_seq)})

    target = network(t_seq, inputs=inputs).firing_rates["exc"]
    loss = ResidualLoss(
        [ObservationSpec(name="obs", population="exc", target=target, sigma=2.0)]
    )

    space = ParameterSpace.from_network(
        network,
        transforms={
            "exc/tau": LogTransform(),
            "in->exc/gsyn_max": LogTransform(),
        },
    )
    return network, space, t_seq, inputs, loss


@pytest.fixture
def problem() -> CalibrationProblem:
    network, space, t_seq, inputs, loss = _build()
    return CalibrationProblem(network, space, t_seq, inputs=inputs, objective=loss)


def _vg(problem):
    def vg(phi_np):
        ev, grad = problem.value_and_gradient(phi_np)
        return ev.loss, np.asarray(grad, dtype=np.float64)

    return vg


class TestPackUnpack:
    def test_roundtrip(self, problem):
        phi0 = problem.pack()
        assert phi0.shape == (problem.size,)
        theta = problem.parameter_space.unpack(phi0)
        phi_back = problem.parameter_space.pack(theta)
        np.testing.assert_allclose(np.asarray(phi_back), np.asarray(phi0), rtol=1e-6)

    def test_pack_size_validation(self, problem):
        with pytest.raises(ValueError, match="elements, expected"):
            problem.pack(np.zeros(problem.size + 1, dtype=np.float32))

    def test_grad_mode_validation(self, problem):
        with pytest.raises(ValueError, match="grad_mode"):
            CalibrationProblem(
                problem.network,
                problem.parameter_space,
                problem.t_sequence,
                inputs=problem.inputs,
                objective=problem.objective,
                grad_mode="implicit",
            )


class TestEvaluate:
    def test_evaluate_none_matches_explicit_forward(self, problem):
        ev = problem.evaluate()
        output = problem.network(problem.t_sequence, inputs=problem.inputs)
        ref = float(problem.objective(output, problem.network))
        assert ev.loss == pytest.approx(ref, rel=1e-6)
        assert ev.status == "ok"
        assert set(ev.components) == {"obs"}
        assert ev.components["obs"] == pytest.approx(ev.loss, rel=1e-6)

    def test_evaluate_with_phi_matches_explicit_at_theta(self, problem):
        phi0 = np.asarray(problem.pack())
        phi_new = phi0 + np.array([0.1, 0.2, -0.1], dtype=np.float32)
        ev = problem.evaluate(phi_new)
        output = problem.network(problem.t_sequence, inputs=problem.inputs)
        ref = float(problem.objective(output, problem.network))
        assert ev.loss == pytest.approx(ref, rel=1e-6)

    def test_missing_objective_raises(self, problem):
        problem.objective = None
        with pytest.raises(ValueError, match="objective is None"):
            problem.evaluate()
        with pytest.raises(ValueError, match="objective is None"):
            problem.value_and_gradient()

    def test_residual_requires_residual_interface(self, problem):
        problem.objective = lambda out, net: tf.reduce_sum(out.firing_rates["exc"])
        with pytest.raises(TypeError, match="residual"):
            problem.residual()

    def test_evaluation_to_dict_json_serializable(self, problem):
        ev, _ = problem.value_and_gradient()
        json.dumps(ev.to_dict())


class TestGradients:
    @pytest.mark.parametrize("grad_mode", ["assign", "chain"])
    def test_taylor_test(self, problem, grad_mode):
        problem.grad_mode = grad_mode
        # точка phi0 — минимум (target снят с неё); сдвигаемся, чтобы
        # направленная производная была ненулевой
        phi0 = np.asarray(problem.pack(), dtype=np.float64) + 0.25
        # float32: ниже h ~ 5e-4 разности тонут в округлении — сетка
        # ограничена чисто квадратичным окном (см. диагностику taylor_test)
        report = taylor_test(_vg(problem), phi0, h_values=np.logspace(-1, -3, 7))
        assert 0.8 < report["slope_R0_median"] < 1.3
        assert 1.6 < report["slope_R1_median"] < 2.4

    def test_both_modes_agree(self, problem):
        phi0 = np.asarray(problem.pack(), dtype=np.float64)
        _, g_assign = problem.value_and_gradient(phi0)
        problem.grad_mode = "chain"
        _, g_chain = problem.value_and_gradient(phi0)
        np.testing.assert_allclose(
            np.asarray(g_assign), np.asarray(g_chain), rtol=1e-5, atol=1e-8
        )

    def test_gradient_finite(self, problem):
        _, grad = problem.value_and_gradient()
        assert np.all(np.isfinite(np.asarray(grad)))


class TestResidual:
    def test_reconstructs_loss(self, problem):
        phi0 = np.asarray(problem.pack())
        r = problem.residual(phi0)
        ev = problem.evaluate(phi0)
        assert 0.5 * float(r @ r) == pytest.approx(ev.loss, rel=1e-6)
        assert r.shape == (T * 1,)

    def test_residual_matches_evaluation_residuals(self, problem):
        r = problem.residual()
        ev = problem.evaluate()
        np.testing.assert_allclose(r, ev.residuals, rtol=1e-6)


class TestModelMutation:
    def test_evaluate_phi_applies_to_model(self, problem):
        space = problem.parameter_space
        i_ext_var = space.entries["exc/I_ext"].variable
        old_value = float(i_ext_var.numpy().ravel()[0])

        phi0 = np.asarray(problem.pack())
        delta = np.zeros_like(phi0)
        delta[list(space.entries).index("exc/I_ext")] = 0.3
        problem.evaluate(phi0 + delta)

        new_value = float(i_ext_var.numpy().ravel()[0])
        assert new_value == pytest.approx(old_value + 0.3, rel=1e-6)
        # модель остаётся в θ(phi) — evaluate(None) видит новое состояние
        ev_none = problem.evaluate()
        output = problem.network(problem.t_sequence, inputs=problem.inputs)
        ref = float(problem.objective(output, problem.network))
        assert ev_none.loss == pytest.approx(ref, rel=1e-6)

    def test_evaluate_none_does_not_touch_model(self, problem):
        space = problem.parameter_space
        var = space.entries["exc/tau"].variable
        before = np.asarray(var.numpy()).copy()
        problem.evaluate()
        np.testing.assert_array_equal(np.asarray(var.numpy()), before)


class TestCounters:
    def test_counters_and_wall_time(self, problem):
        ev1 = problem.evaluate()
        assert problem.n_forward == 1
        assert problem.n_gradient == 0

        ev2, _ = problem.value_and_gradient()
        assert problem.n_forward == 2
        assert problem.n_gradient == 1

        problem.residual()
        assert problem.n_forward == 3
        assert problem.n_gradient == 1

        assert ev2.n_forward == 2
        assert ev2.n_gradient == 1
        assert ev1.wall_time > 0.0
        assert problem.wall_time > 0.0
        assert ev1.dtype == "float32"
        assert ev1.dt == pytest.approx(DT)


class TestFloat64:
    def test_value_and_gradient_float64(self):
        neuraltide.config.set_dtype(tf.float64)
        network, space, t_seq, inputs, loss = _build()
        problem = CalibrationProblem(
            network, space, t_seq, inputs=inputs, objective=loss
        )
        phi0 = np.asarray(problem.pack(), dtype=np.float64) + 0.25
        ev, grad = problem.value_and_gradient(phi0)
        assert ev.dtype == "float64"
        assert np.asarray(grad).dtype == np.float64

        report = taylor_test(_vg(problem), phi0)
        assert 0.9 < report["slope_R0_median"] < 1.1
        assert 1.8 < report["slope_R1_median"] < 2.2

        r = problem.residual(phi0)
        assert 0.5 * float(r @ r) == pytest.approx(ev.loss, rel=1e-8)


class TestBoundsPhi:
    def test_bounds_from_constraints(self, problem):
        lower, upper = problem.bounds_phi()
        assert lower.shape == upper.shape == (problem.size,)
        names = list(problem.parameter_space.entries)
        i_tau = names.index("exc/tau")
        i_iext = names.index("exc/I_ext")
        # tau и gsyn_max ограничены снизу: phi-границы конечны
        assert np.isfinite(lower[i_tau])
        assert lower[i_tau] == pytest.approx(np.log(1.0), rel=1e-6)
        # I_ext без ограничений
        assert lower[i_iext] == -np.inf and upper[i_iext] == np.inf


class TestOptimizationResult:
    def test_to_from_dict_roundtrip(self, problem):
        phi0 = np.asarray(problem.pack())
        theta0 = problem.parameter_space.unpack(phi0)
        ev = problem.evaluate()
        result = OptimizationResult(
            phi_best=phi0,
            theta_best=theta0,
            loss_best=ev.loss,
            history=[{"loss": ev.loss, "epoch": 0}],
            seed=42,
            config={"dt": DT, "grad_mode": "assign"},
            dtype="float32",
        )
        data = result.to_dict()
        json.dumps(data)

        restored = OptimizationResult.from_dict(data)
        np.testing.assert_allclose(restored.phi_best, result.phi_best)
        np.testing.assert_allclose(
            restored.theta_best["exc/I_ext"], result.theta_best["exc/I_ext"]
        )
        assert restored.loss_best == pytest.approx(result.loss_best)
        assert restored.seed == 42
        assert restored.config["dt"] == DT
        assert "neuraltide" in restored.versions
        assert restored.history == [{"loss": ev.loss, "epoch": 0}]

    def test_save_load(self, problem, tmp_path):
        phi0 = np.asarray(problem.pack())
        result = OptimizationResult(
            phi_best=phi0,
            theta_best=problem.parameter_space.unpack(phi0),
            loss_best=problem.evaluate().loss,
        )
        path = tmp_path / "result.json"
        result.save(path)
        restored = OptimizationResult.load(path)
        np.testing.assert_allclose(restored.phi_best, result.phi_best)
        assert restored.loss_best == pytest.approx(result.loss_best)
