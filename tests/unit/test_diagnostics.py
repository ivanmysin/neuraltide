"""
Тесты пользовательских утилит верификации neuraltide.optimization.diagnostics:
FD-сканирование, Taylor test, dot-product test, сходимость по dt,
согласованность dtype, сравнение BPTT ↔ adjoint.
"""

import numpy as np
import pytest
import tensorflow as tf

import neuraltide.config
from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.populations import IzhikevichMeanField
from neuraltide.synapses import TsodyksMarkramSynapse
from neuraltide.inputs import VonMisesGenerator
from neuraltide.integrators import RK4Integrator
from neuraltide.training import MSELoss, CompositeLoss
from neuraltide.optimization.diagnostics import (
    compare_bptt_adjoint,
    dot_product_test,
    dt_convergence,
    dtype_consistency_check,
    finite_difference_scan,
    taylor_test,
)


# ── Чисто-numpy проверки ─────────────────────────────────────────────────────


class TestFiniteDifferenceScan:
    def test_quadratic_gradient(self):
        phi0 = np.array([0.7, -1.3, 2.2])
        func = lambda phi: float(np.sum(phi**2))  # noqa: E731
        report = finite_difference_scan(func, phi0)
        assert report["f0"] == pytest.approx(float(np.sum(phi0**2)))
        # центральные разности квадратичной функции точны до пола округления
        assert np.allclose(report["best"], 2.0 * phi0, atol=1e-6)
        assert len(report["gradients"]) == len(report["h_values"])

    def test_single_h(self):
        phi0 = np.array([1.0, 2.0])
        func = lambda phi: float(np.sum(phi**3))  # noqa: E731
        report = finite_difference_scan(func, phi0, h_values=[1e-4])
        assert report["best_h"] == pytest.approx(1e-4)
        assert np.allclose(report["best"], 3.0 * phi0**2, rtol=1e-4)


class TestTaylorTest:
    def test_quadratic_slopes(self):
        rng = np.random.default_rng(0)
        phi0 = rng.standard_normal(5)
        A = rng.standard_normal((5, 5))
        A = A.T @ A + np.eye(5)  # SDP: f гладкая квадратичная

        def value_and_grad(phi):
            f = 0.5 * float(phi @ A @ phi)
            return f, A @ phi

        report = taylor_test(value_and_grad, phi0)
        assert report["slope_R0_median"] == pytest.approx(1.0, abs=0.2)
        assert report["slope_R1_median"] == pytest.approx(2.0, abs=0.3)
        assert len(report["rows"]) > 3
        assert report["grad_dot_v"] == pytest.approx(
            float(report["v"] @ A @ phi0), rel=1e-12
        )

    def test_direction_override(self):
        phi0 = np.array([1.0, -2.0])
        v = np.array([0.6, 0.8])
        report = taylor_test(lambda phi: (float(np.sum(phi**2)), 2.0 * phi), phi0, v=v)
        assert np.allclose(report["v"], v)


class TestDotProductTest:
    def test_linear_operator(self):
        rng = np.random.default_rng(1)
        J = rng.standard_normal((4, 3))
        x = rng.standard_normal(3)
        y = rng.standard_normal(4)
        report = dot_product_test(lambda v: J @ v, lambda u: J.T @ u, x, y, tol=1e-8)
        assert report["passed"]
        assert report["rel_error"] < 1e-8

    def test_mismatched_operators_fail(self):
        rng = np.random.default_rng(2)
        J = rng.standard_normal((3, 3))
        K = rng.standard_normal((3, 3))
        report = dot_product_test(
            lambda v: J @ v,
            lambda u: K.T @ u,
            rng.standard_normal(3),
            rng.standard_normal(3),
        )
        assert not report["passed"]


# ── Проверки на малой сети ───────────────────────────────────────────────────


def _make_network(dt=0.1, dtype=None):
    if dtype is not None:
        neuraltide.config.set_dtype(dtype)
    pop = IzhikevichMeanField(
        dt=dt,
        params={
            "tau_pop": {"value": 1.0, "trainable": False},
            "alpha": {"value": 0.5, "trainable": False},
            "a": {"value": 0.02, "trainable": False},
            "b": {"value": 0.2, "trainable": False},
            "w_jump": {"value": 0.1, "trainable": False},
            "Delta_I": {"value": 0.5, "trainable": True},
            "I_ext": {"value": 0.1, "trainable": True},
        },
    )
    gen = VonMisesGenerator(
        dt=dt,
        params={"mean_rate": 20.0, "R": 0.5, "freq": 8.0, "phase": 0.0},
        name="theta_gen",
    )
    syn = TsodyksMarkramSynapse(
        n_pre=1,
        n_post=1,
        dt=dt,
        params={
            "gsyn_max": {"value": [[0.1]], "trainable": True},
            "tau_f": {"value": 20.0, "trainable": False},
            "tau_d": {"value": 5.0, "trainable": False},
            "tau_r": {"value": 200.0, "trainable": False},
            "Uinc": {"value": 0.2, "trainable": False},
            "pconn": {"value": [[1.0]], "trainable": False},
            "e_r": {"value": 0.0, "trainable": False},
        },
    )
    graph = NetworkGraph(dt=dt)
    graph.declare_input("theta", n_units=gen.n_units)
    graph.add_population("exc", pop)
    graph.add_synapse("theta->exc", syn, src="theta", tgt="exc")
    return NetworkRNN(graph, integrator=RK4Integrator()), gen


class TestDtConvergence:
    def test_loss_and_grads_by_dt(self):
        def build_and_run(dt):
            network, gen = _make_network(dt=dt)
            t_values = np.arange(30, dtype=np.float32) * dt
            t_seq = tf.constant(t_values[None, :, None])
            inputs = network._graph.pack_inputs({"theta": gen(t_seq)})
            output = network(t_seq, inputs=inputs, training=False)
            target = {k: v + 0.5 for k, v in output.firing_rates.items()}
            loss_fn = CompositeLoss([(1.0, MSELoss(target))])
            with tf.GradientTape() as tape:
                out = network(t_seq, inputs=inputs, training=True)
                loss = loss_fn(out, network)
            grads = tape.gradient(loss, network.trainable_variables)
            return {
                "loss": float(loss),
                "grads": np.concatenate(
                    [g.numpy().ravel() for g in grads if g is not None]
                ),
                "outputs": output.firing_rates["exc"].numpy(),
            }

        rows = dt_convergence(build_and_run, dts=[0.2, 0.1, 0.05])
        assert [r["dt"] for r in rows] == [0.2, 0.1, 0.05]
        for row in rows:
            assert np.isfinite(row["loss"])
            assert row["grad_norm"] > 0
            assert "output_summary" in row
        # различия между соседними dt посчитаны
        assert "loss_diff" in rows[1] and "grad_diff" in rows[1]
        assert rows[1]["loss_diff"] >= 0.0


class TestDtypeConsistency:
    def test_float32_vs_float64_and_restore(self):
        prev_dtype = neuraltide.config.get_dtype()

        def build_and_run(dtype):
            network, gen = _make_network(dt=0.1, dtype=dtype)
            t_values = np.arange(30, dtype=dtype.as_numpy_dtype()) * 0.1
            t_seq = tf.constant(t_values[None, :, None])
            inputs = network._graph.pack_inputs({"theta": gen(t_seq)})
            output = network(t_seq, inputs=inputs, training=False)
            target = {k: v + 0.5 for k, v in output.firing_rates.items()}
            loss_fn = CompositeLoss([(1.0, MSELoss(target))])
            with tf.GradientTape() as tape:
                out = network(t_seq, inputs=inputs, training=True)
                loss = loss_fn(out, network)
            grads = tape.gradient(loss, network.trainable_variables)
            return {
                "loss": float(loss),
                "grads": np.concatenate(
                    [g.numpy().ravel() for g in grads if g is not None]
                ),
            }

        rows = dtype_consistency_check(build_and_run)
        assert [r["dtype"] for r in rows] == ["float32", "float64"]
        for row in rows:
            assert np.isfinite(row["loss"])
        # различие dtype умеренное (float32-шум на длинной траектории)
        assert rows[-1]["loss_rel_diff_vs_first"] < 0.2
        # прежний dtype восстановлен
        assert neuraltide.config.get_dtype() == prev_dtype


class TestCompareBpttAdjoint:
    def test_gradients_match_and_target_extracted(self):
        network, gen = _make_network(dt=0.1)
        t_values = np.arange(30, dtype=np.float32) * 0.1
        t_seq = tf.constant(t_values[None, :, None])
        inputs = network._graph.pack_inputs({"theta": gen(t_seq)})
        output = network(t_seq, inputs=inputs, training=False)
        target = {k: v + 0.5 for k, v in output.firing_rates.items()}
        loss_fn = CompositeLoss([(1.0, MSELoss(target))])

        # target не передаём — должен быть извлечён из loss_fn
        report = compare_bptt_adjoint(network, loss_fn, t_seq, inputs=inputs)

        assert report["loss_abs_diff"] < 1e-4
        assert report["cosine_similarity"] > 0.999
        assert report["max_grad_abs_diff"] < 1e-2
        assert len(report["per_variable"]) == len(network.trainable_variables)

    def test_with_initial_state(self):
        network, gen = _make_network(dt=0.1)
        t_values = np.arange(30, dtype=np.float32) * 0.1
        t_seq = tf.constant(t_values[None, :, None])
        inputs = network._graph.pack_inputs({"theta": gen(t_seq)})
        output = network(t_seq, inputs=inputs, training=False)
        target = {k: v + 0.5 for k, v in output.firing_rates.items()}
        loss_fn = MSELoss(target)

        init_pop, init_syn = network.get_initial_state(1)
        initial_state = (
            [s * 2.5 + 0.05 for s in init_pop],
            [s * 1.5 + 0.01 for s in init_syn],
        )
        report = compare_bptt_adjoint(
            network,
            loss_fn,
            t_seq,
            inputs=inputs,
            initial_state=initial_state,
        )
        assert report["cosine_similarity"] > 0.999
        assert report["max_grad_abs_diff"] < 1e-2


class TestDtConvergenceExcludesBrunel:
    def test_docstring_mentions_brunel(self):
        assert "BrunelFokkerPlanck" in dt_convergence.__doc__
