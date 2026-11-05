"""AdjointSolver × network-wide интегрирование (integration_mode='network').

Покрытие:
- discrete adjoint: градиенты совпадают с BPTT в float64 (Euler и RK4);
- analytical adjoint + network -> NotImplementedError (до прогона);
- split analytical adjoint: поведение прежнее (solver конструируется и
  считает градиенты);
- Trainer(grad_method="adjoint") в network-режиме: шаг обучения работает;
- stability gradients: в network-режиме честные нули + UserWarning
  (скрытого падения нет, фиктивный градиент не выдумывается).
"""

import numpy as np
import pytest
import tensorflow as tf

import neuraltide.config
from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.integrators import EulerIntegrator, RK4Integrator
from neuraltide.inputs import ConstantRateGenerator
from neuraltide.populations import IzhikevichMeanField
from neuraltide.synapses import TsodyksMarkramSynapse
from neuraltide.training import (
    CompositeLoss,
    MSELoss,
    StabilityPenalty,
    Trainer,
)
from neuraltide.training.adjoint import AdjointSolver

DT = 0.1


def _make_pop(name, i_ext):
    return IzhikevichMeanField(
        dt=DT,
        params={
            "tau_pop": {"value": 1.0, "trainable": False},
            "alpha": {"value": 0.5, "trainable": False},
            "a": {"value": 0.02, "trainable": False},
            "b": {"value": 0.2, "trainable": False},
            "w_jump": {"value": 0.1, "trainable": False},
            "Delta_I": {"value": 0.5, "trainable": True},
            "I_ext": {"value": i_ext, "trainable": True},
        },
        name=name,
    )


def _make_tm_synapse():
    return TsodyksMarkramSynapse(
        n_pre=1,
        n_post=1,
        dt=DT,
        params={
            "gsyn_max": {"value": [[0.1]], "trainable": True},
            "tau_f": {"value": 20.0, "trainable": True},
            "tau_d": {"value": 5.0, "trainable": True},
            "tau_r": {"value": 200.0, "trainable": False},
            "Uinc": {"value": 0.2, "trainable": False},
            "pconn": {"value": [[1.0]], "trainable": False},
            "e_r": {"value": 0.0, "trainable": False},
        },
    )


def _make_coupled_tm_network(integrator, integration_mode="network"):
    """in -> p1 -(TM)-> p2: малая связанная сеть с TM-синапсами."""
    gen = ConstantRateGenerator(dt=DT, params={"rate": 15.0})
    graph = NetworkGraph(dt=DT)
    graph.declare_input("in", n_units=1)
    graph.add_population("p1", _make_pop("p1", 0.1))
    graph.add_population("p2", _make_pop("p2", 0.0))
    graph.add_synapse("in->p1", _make_tm_synapse(), src="in", tgt="p1")
    graph.add_synapse("p1->p2", _make_tm_synapse(), src="p1", tgt="p2")
    network = NetworkRNN(
        graph,
        integrator=integrator,
        integration_mode=integration_mode,
    )
    return network, gen


def _make_recurrent_tm_network(integrator, integration_mode="split"):
    """p1 <-> p2 (TM): связанная сеть без внешних входов.

    Используется для split analytical adjoint: пре-существующий
    analytical-путь не умеет синапсы от внешних входов (KeyError в
    _compute_synaptic_context), поэтому здесь только популяции.
    """
    graph = NetworkGraph(dt=DT)
    graph.add_population("p1", _make_pop("p1", 0.2))
    graph.add_population("p2", _make_pop("p2", 0.1))
    graph.add_synapse("p1->p2", _make_tm_synapse(), src="p1", tgt="p2")
    graph.add_synapse("p2->p1", _make_tm_synapse(), src="p2", tgt="p1")
    network = NetworkRNN(
        graph,
        integrator=integrator,
        integration_mode=integration_mode,
    )
    return network


def _zero_inputs(t_steps):
    return tf.zeros([1, t_steps, 0], dtype=tf.float64)


def _make_data(network, gen, t_steps):
    t_values = np.arange(t_steps, dtype=np.float64) * DT
    t_seq = tf.constant(t_values[None, :, None], dtype=tf.float64)
    inputs = network._graph.pack_inputs({"in": gen(t_seq)})
    output = network(t_seq, inputs=inputs, training=False)
    target = {k: v + 0.5 for k, v in output.firing_rates.items()}
    return t_seq, inputs, target


class TestDiscreteAdjointNetworkMode:
    """Discrete adjoint в network-режиме дифференцирует тот же шаг, что BPTT."""

    @pytest.mark.parametrize("integrator_cls", [EulerIntegrator, RK4Integrator])
    def test_gradients_match_bptt_float64(self, integrator_cls):
        neuraltide.config.set_dtype(tf.float64)
        network, gen = _make_coupled_tm_network(integrator_cls(), "network")
        t_seq, inputs, target = _make_data(network, gen, t_steps=20)
        loss_fn = CompositeLoss([(1.0, MSELoss(target))])

        with tf.GradientTape() as tape:
            output = network(t_seq, inputs=inputs, training=False)
            loss = loss_fn(output, network)
        bptt_grads = tape.gradient(loss, network.trainable_variables)

        solver = AdjointSolver(network, network._integrator)
        assert solver._integration_mode == "network"
        adj_grads, variables, _ = solver.compute_gradients(
            t_seq, inputs, target, loss_fn
        )

        assert len(adj_grads) == len(bptt_grads)
        for var, g_b, g_a in zip(network.trainable_variables, bptt_grads, adj_grads):
            assert g_b is not None, f"{var.name}: BPTT gradient is None (sanity)"
            assert g_a is not None, f"{var.name}: adjoint gradient is None"
            max_diff = float(tf.reduce_max(tf.abs(g_b - g_a)))
            scale = max(1.0, float(tf.reduce_max(tf.abs(g_b))))
            assert max_diff <= 1e-6 * scale, (
                f"{var.name}: adjoint vs BPTT max diff {max_diff:.3e} "
                f"(scale {scale:.3e})"
            )

    def test_forward_pass_replays_network_mode(self):
        """forward_pass solver'а совпадает с прямым прогоном сети."""
        neuraltide.config.set_dtype(tf.float64)
        network, gen = _make_coupled_tm_network(RK4Integrator(), "network")
        t_seq, inputs, target = _make_data(network, gen, t_steps=10)

        solver = AdjointSolver(network, network._integrator)
        adj_output, _, _ = solver.forward_pass(t_seq, inputs)
        ref_output = network(t_seq, inputs=inputs, training=False)

        for name in ref_output.firing_rates:
            max_diff = float(
                tf.reduce_max(
                    tf.abs(
                        adj_output.firing_rates[name] - ref_output.firing_rates[name]
                    )
                )
            )
            assert max_diff < 1e-12, f"{name}: forward mismatch {max_diff:.3e}"


class TestAnalyticalAdjointNetworkMode:
    """Analytical adjoint зашит под split: в network-режиме честный отказ."""

    def test_network_mode_raises_not_implemented(self):
        neuraltide.config.set_dtype(tf.float64)
        network, _ = _make_coupled_tm_network(RK4Integrator(), "network")
        with pytest.raises(NotImplementedError, match="integration_mode='network'"):
            AdjointSolver(
                network,
                network._integrator,
                use_analytical_adjoint=True,
            )

    def test_split_analytical_adjoint_still_works(self):
        neuraltide.config.set_dtype(tf.float64)
        network = _make_recurrent_tm_network(RK4Integrator(), "split")
        solver = AdjointSolver(
            network,
            network._integrator,
            use_analytical_adjoint=True,
        )
        assert solver._integration_mode == "split"

        t_seq = tf.constant((np.arange(10, dtype=np.float64) * DT)[None, :, None])
        inputs = _zero_inputs(10)
        output = network(t_seq, inputs=inputs, training=False)
        target = {k: v + 0.5 for k, v in output.firing_rates.items()}
        grads, variables, _ = solver.compute_gradients(
            t_seq,
            inputs,
            target,
            CompositeLoss([(1.0, MSELoss(target))]),
        )
        assert len(grads) == len(network.trainable_variables)
        for var, g in zip(variables, grads):
            assert g is not None, f"{var.name}: analytical gradient is None"
            assert np.all(np.isfinite(g.numpy())), f"{var.name}: non-finite grad"


class TestTrainerAdjointNetworkMode:
    """Trainer с grad_method='adjoint' работает в network-режиме (discrete)."""

    def test_trainer_adjoint_fit_network_mode(self):
        neuraltide.config.set_dtype(tf.float64)
        network, gen = _make_coupled_tm_network(EulerIntegrator(), "network")
        t_seq, inputs, target = _make_data(network, gen, t_steps=15)
        loss_fn = CompositeLoss([(1.0, MSELoss(target))])

        trainer = Trainer(
            network,
            loss_fn,
            tf.keras.optimizers.Adam(1e-3),
            grad_method="adjoint",
        )
        initial_loss = float(loss_fn(network(t_seq, inputs=inputs), network))
        history = trainer.fit(t_seq, inputs, epochs=5, verbose=0)

        assert len(history.loss_history) == 5
        assert all(np.isfinite(v) for v in history.loss_history)
        final_loss = float(loss_fn(network(t_seq, inputs=inputs), network))
        assert final_loss < initial_loss, (
            f"loss should decrease: {initial_loss:.6f} -> {final_loss:.6f}"
        )


class TestStabilityGradientsNetworkMode:
    """Stability-градиенты в network-режиме: честные нули + предупреждение."""

    def test_stability_grads_zero_with_warning(self):
        neuraltide.config.set_dtype(tf.float64)
        network, gen = _make_coupled_tm_network(EulerIntegrator(), "network")
        t_seq, inputs, target = _make_data(network, gen, t_steps=10)

        loss_fn = CompositeLoss(
            [
                (1.0, MSELoss(target)),
                (1e-3, StabilityPenalty()),
            ]
        )
        solver = AdjointSolver(network, network._integrator)
        with pytest.warns(UserWarning, match="identically zero"):
            grads, variables, _ = solver.compute_gradients(
                t_seq, inputs, target, loss_fn
            )
        for var, g in zip(variables, grads):
            assert np.all(np.isfinite(g.numpy())), f"{var.name}: non-finite grad"

        # Тот же прогон без stability-терма даёт те же градиенты: вклад
        # штрафа ровно ноль — скрытого падения нет.
        solver_main = AdjointSolver(network, network._integrator)
        grads_main, _, _ = solver_main.compute_gradients(
            t_seq,
            inputs,
            target,
            CompositeLoss([(1.0, MSELoss(target))]),
        )
        for var, g_full, g_main in zip(variables, grads, grads_main):
            diff = float(tf.reduce_max(tf.abs(g_full - g_main)))
            assert diff == 0.0, f"{var.name}: stability term changed grads by {diff}"
