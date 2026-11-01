"""
AdjointSolver.compute_gradients принимает initial_state: прямой прогон
и backward согласованы (state_info строится от него), градиенты совпадают
с BPTT с тем же начальным состоянием; Trainer-проброс работает.
"""

import numpy as np
import pytest
import tensorflow as tf

from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.populations import IzhikevichMeanField
from neuraltide.synapses import TsodyksMarkramSynapse
from neuraltide.inputs import VonMisesGenerator
from neuraltide.integrators import RK4Integrator
from neuraltide.training import MSELoss, CompositeLoss, Trainer
from neuraltide.training.adjoint import AdjointSolver


def _make_network(dt=0.1):
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


def _make_data(network, gen, T=30):
    dt = network._graph.dt
    t_values = np.arange(T, dtype=np.float32) * dt
    t_seq = tf.constant(t_values[None, :, None])
    inputs = network._graph.pack_inputs({"theta": gen(t_seq)})
    output = network(t_seq, inputs=inputs, training=False)
    target = {k: v + 0.5 for k, v in output.firing_rates.items()}
    return t_seq, inputs, target


def _perturbed_initial_state(network):
    init_pop, init_syn = network.get_initial_state(1)
    perturbed_pop = [s * 3.0 + 0.05 for s in init_pop]
    perturbed_syn = [s * 1.5 + 0.01 for s in init_syn]
    return perturbed_pop, perturbed_syn


class TestAdjointInitialState:
    def test_forward_pass_uses_initial_state(self):
        network, gen = _make_network()
        t_seq, inputs, target = _make_data(network, gen)
        initial_state = _perturbed_initial_state(network)

        solver = AdjointSolver(network, network._integrator)
        _, _, output = solver.compute_gradients(
            t_seq,
            inputs,
            target,
            CompositeLoss([(1.0, MSELoss(target))]),
            initial_state=initial_state,
        )

        ref = network(t_seq, inputs=inputs, initial_state=initial_state)
        for name in ref.firing_rates:
            max_diff = float(
                tf.reduce_max(
                    tf.abs(output.firing_rates[name] - ref.firing_rates[name])
                )
            )
            assert max_diff < 1e-6

        # траектория отлична от прогона с состоянием по умолчанию
        default_out = network(t_seq, inputs=inputs)
        name = next(iter(ref.firing_rates))
        assert (
            float(
                tf.reduce_max(
                    tf.abs(output.firing_rates[name] - default_out.firing_rates[name])
                )
            )
            > 1e-3
        )

    def test_gradients_match_bptt_with_initial_state(self):
        network, gen = _make_network()
        t_seq, inputs, target = _make_data(network, gen)
        loss_fn = CompositeLoss([(1.0, MSELoss(target))])
        initial_state = _perturbed_initial_state(network)

        with tf.GradientTape() as tape:
            output = network(t_seq, inputs=inputs, initial_state=initial_state)
            loss = loss_fn(output, network)
        bptt_grads = tape.gradient(loss, network.trainable_variables)

        solver = AdjointSolver(network, network._integrator)
        adj_grads, _, _ = solver.compute_gradients(
            t_seq, inputs, target, loss_fn, initial_state=initial_state
        )

        for var, g_b, g_a in zip(network.trainable_variables, bptt_grads, adj_grads):
            if g_b is None:
                continue
            assert g_a is not None, f"{var.name}: adjoint gradient is None"
            diff = float(tf.reduce_max(tf.abs(g_b - g_a)))
            assert diff < 1e-2, (
                f"{var.name}: adjoint vs BPTT diff {diff:.6f} (initial_state)"
            )

    def test_gradients_differ_from_default_state(self):
        network, gen = _make_network()
        t_seq, inputs, target = _make_data(network, gen)
        loss_fn = CompositeLoss([(1.0, MSELoss(target))])
        initial_state = _perturbed_initial_state(network)

        solver = AdjointSolver(network, network._integrator)
        grads_perturbed, _, _ = solver.compute_gradients(
            t_seq, inputs, target, loss_fn, initial_state=initial_state
        )
        grads_default, _, _ = solver.compute_gradients(t_seq, inputs, target, loss_fn)

        max_diff = 0.0
        for g_a, g_b in zip(grads_perturbed, grads_default):
            if g_a is None or g_b is None:
                continue
            max_diff = max(max_diff, float(tf.reduce_max(tf.abs(g_a - g_b))))
        assert max_diff > 1e-3, (
            "gradients with perturbed initial_state should differ from default"
        )

    def test_backward_pass_state_info_consistent(self):
        # state_info от forward_pass(initial_state) корректно склеивается
        # в backward: градиенты совпадают с compute_gradients.
        # Порог 1e-2 — как в остальных adjoint-тестах: повторные вызовы
        # backward_pass на float32 дают ulp-джиттер ~1e-4 (пре-существующее
        # поведение GradientTape в tf.while_loop, проверено на HEAD).
        network, gen = _make_network()
        t_seq, inputs, target = _make_data(network, gen)
        loss_fn = MSELoss(target)
        initial_state = _perturbed_initial_state(network)

        solver = AdjointSolver(network, network._integrator)
        _, _, state_info = solver.forward_pass(t_seq, inputs, initial_state)
        grads_bp = solver.backward_pass(t_seq, inputs, state_info, target, loss_fn)
        grads_cg, _, _ = solver.compute_gradients(
            t_seq, inputs, target, loss_fn, initial_state=initial_state
        )

        for var, g_a, g_b in zip(network.trainable_variables, grads_bp, grads_cg):
            if g_a is None and g_b is None:
                continue
            diff = float(tf.reduce_max(tf.abs(g_a - g_b)))
            assert diff < 1e-2, f"{var.name}: backward_pass diff {diff:.6f}"

    def test_trainer_adjoint_step_forwards_initial_state(self):
        network, gen = _make_network()
        t_seq, inputs, target = _make_data(network, gen)
        loss_fn = CompositeLoss([(1.0, MSELoss(target))])
        trainer = Trainer(
            network,
            loss_fn,
            tf.keras.optimizers.Adam(1e-3),
            grad_method="adjoint",
        )
        initial_state = _perturbed_initial_state(network)

        # loss шага — значение до обновления весов; эталон считаем до шага
        ref_output = network(t_seq, inputs=inputs, initial_state=initial_state)
        ref_loss = float(loss_fn(ref_output, network))

        step_result = trainer.train_step(t_seq, inputs, initial_state=initial_state)

        assert float(step_result["loss"]) == pytest.approx(ref_loss, rel=1e-5)
        assert step_result["grads"] is not None
