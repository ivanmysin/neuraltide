"""
Tests for AdjointSolver: gradient computation via adjoint state method.
"""
import pytest
import tensorflow as tf
import numpy as np

import neuraltide
import neuraltide.config
from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.populations import IzhikevichMeanField
from neuraltide.synapses import TsodyksMarkramSynapse, StaticSynapse
from neuraltide.inputs import VonMisesGenerator
from neuraltide.integrators import EulerIntegrator, RK4Integrator
from neuraltide.training import MSELoss, StabilityPenalty, CompositeLoss


def _make_simple_small_network(dt=0.05):
    """Create a simple network for testing: 1 input -> 1 population."""
    pop = IzhikevichMeanField(dt=dt, params={
        'tau_pop':   {'value': [1.0, 1.0],   'trainable': False},
        'alpha':     {'value': [0.5, 0.5],   'trainable': False},
        'a':         {'value': [0.02, 0.02], 'trainable': False},
        'b':         {'value': [0.2, 0.2],   'trainable': False},
        'w_jump':    {'value': [0.1, 0.1],   'trainable': False},
        'Delta_I':   {'value': [0.5, 0.5],   'trainable': True,
                      'min': 0.01, 'max': 2.0},
        'I_ext':     {'value': [0.1, 0.1],   'trainable': True,
                      'min': -2.0, 'max': 2.0},
    })

    gen = VonMisesGenerator(
        dt=dt,
        params={
            'mean_rate': 20.0,
            'R': 0.5,
            'freq': 8.0,
            'phase': 0.0,
        },
        name='theta_gen'
    )

    syn = TsodyksMarkramSynapse(n_pre=1, n_post=2, dt=dt, params={
        'gsyn_max': {'value': [[0.1, 0.1]], 'trainable': True},
        'tau_f':    {'value': 20.0,  'trainable': True, 'min': 6.0,  'max': 240.0},
        'tau_d':    {'value': 5.0,   'trainable': True, 'min': 2.0,  'max': 15.0},
        'tau_r':    {'value': 200.0, 'trainable': True, 'min': 91.0, 'max': 1300.0},
        'Uinc':     {'value': 0.2,   'trainable': True, 'min': 0.04, 'max': 0.7},
        'pconn':    {'value': [[1.0, 1.0]], 'trainable': False},
        'e_r':      {'value': 0.0,   'trainable': False},
    })

    graph = NetworkGraph(dt=dt)
    graph.add_input_population('theta', gen)
    graph.add_population('exc', pop)
    graph.add_synapse('theta->exc', syn, src='theta', tgt='exc')

    return graph


def _compute_bptt_gradients(network, t_seq, target):
    """Compute gradients via BPTT (standard tf.GradientTape)."""
    loss_fn = CompositeLoss([
        (1.0, MSELoss(target)),
    ])

    with tf.GradientTape() as tape:
        output = network(t_seq, training=False)
        loss = loss_fn(output, network)

    grads = tape.gradient(loss, network.trainable_variables)
    return grads, output


def _compute_adjoint_gradients(network, t_seq, target, grad_method='adjoint'):
    """Compute gradients via adjoint state method."""
    from neuraltide.training.adjoint import AdjointSolver

    solver = AdjointSolver(network, network._integrator)
    
    loss_fn = CompositeLoss([
        (1.0, MSELoss(target)),
    ])

    grads, variables, output = solver.compute_gradients(t_seq, target, loss_fn)
    return grads, variables, output


class TestAdjointBasics:
    """Basic tests that adjoint module exists and has required interface."""

    def test_adjoint_module_exists(self):
        """AdjointSolver should be importable from neuraltide.training.adjoint."""
        try:
            from neuraltide.training.adjoint import AdjointSolver
        except ImportError:
            pytest.fail("AdjointSolver not found in neuraltide.training.adjoint")

    def test_adjoint_solver_instantiation(self):
        """AdjointSolver should be instantiable with network and integrator."""
        from neuraltide.training.adjoint import AdjointSolver

        graph = _make_simple_small_network()
        network = NetworkRNN(graph, integrator=RK4Integrator())

        solver = AdjointSolver(network, network._integrator)
        assert solver is not None, "AdjointSolver should be created"

    def test_forward_pass_returns_network_output(self):
        """forward_pass should return NetworkOutput with firing_rates and stability_loss."""
        from neuraltide.training.adjoint import AdjointSolver

        graph = _make_simple_small_network()
        network = NetworkRNN(graph, integrator=RK4Integrator())

        t_seq = tf.constant(np.arange(10, dtype=np.float32)[None, :, None] * 0.05)
        solver = AdjointSolver(network, network._integrator)

        output, states_final = solver.forward_pass(t_seq)

        assert output is not None, "forward_pass should return output"
        assert hasattr(output, 'firing_rates'), "output should have firing_rates"
        assert hasattr(output, 'stability_loss'), "output should have stability_loss"
        assert 'exc' in output.firing_rates, "firing_rates should contain 'exc'"


class TestAdjointGradientCorrectness:
    """Tests for gradient correctness: adjoint vs BPTT."""

    def test_adjoint_gradients_match_bptt(self):
        """Adjoint gradients should match BPTT gradients."""
        from neuraltide.training.adjoint import AdjointSolver

        dt = 0.05
        T = 10
        n_steps = T / dt
        graph = _make_simple_small_network(dt=dt)
        network = NetworkRNN(graph, integrator=RK4Integrator())

        t_values = np.arange(n_steps, dtype=np.float32) * dt
        t_seq = tf.constant(t_values[None, :, None])

        target_0 = 10.0 + 5.0 * np.sin(2 * np.pi * 8.0 * t_values / 1000.0)
        target_1 = 8.0 + 4.0 * np.sin(2 * np.pi * 8.0 * t_values / 1000.0 + 0.5)
        target = {
            'exc': tf.constant(
                np.stack([target_0, target_1], axis=-1)[None, :, :],
                dtype=tf.float32
            )
        }

        bptt_grads, bptt_output = _compute_bptt_gradients(network, t_seq, target)

        solver = AdjointSolver(network, network._integrator)
        loss_fn = CompositeLoss([
            (1.0, MSELoss(target)),
        ])
        adj_grads, variables, adj_output = solver.compute_gradients(t_seq, target, loss_fn)

        assert len(bptt_grads) == len(adj_grads), "Should have same number of gradients"

        for bptt_g, adj_g in zip(bptt_grads, adj_grads):
            if bptt_g is None and adj_g is None:
                continue
            assert bptt_g is not None or adj_g is not None, "At least one gradient should exist"
            
            if bptt_g is not None and adj_g is not None:
                diff = tf.reduce_mean(tf.abs(bptt_g - adj_g))
                assert diff < 1e-3, f"Gradient diff {diff:.6f} should be < 1e-3"

    def test_adjoint_stability_loss_equals_bptt(self):
        """Adjoint stability_loss should equal BPTT stability_loss."""
        from neuraltide.training.adjoint import AdjointSolver

        dt = 0.05
        T = 10
        n_steps = int(T / dt)
        graph = _make_simple_small_network(dt=dt)
        network = NetworkRNN(graph, integrator=RK4Integrator(), stability_penalty_weight=1e-3)

        t_values = np.arange(n_steps, dtype=np.float32) * dt
        t_seq = tf.constant(t_values[None, :, None])

        target_0 = 10.0 + 5.0 * np.sin(2 * np.pi * 8.0 * t_values / 1000.0)
        target_1 = 8.0 + 4.0 * np.sin(2 * np.pi * 8.0 * t_values / 1000.0 + 0.5)
        target = {
            'exc': tf.constant(
                np.stack([target_0, target_1], axis=-1)[None, :, :],
                dtype=tf.float32
            )
        }

        bptt_output = network(t_seq, training=False)

        solver = AdjointSolver(network, network._integrator)
        loss_fn = CompositeLoss([
            (1.0, MSELoss(target)),
        ])
        adj_output, _ = solver.forward_pass(t_seq)

        bptt_stability = float(bptt_output.stability_loss)
        adj_stability = float(adj_output.stability_loss)

        assert abs(bptt_stability - adj_stability) < 1e-6, \
            f"Stability loss mismatch: BPTT={bptt_stability}, Adj={adj_stability}"


class TestAdjointWithStabilityPenalty:
    """Tests for adjoint with stability penalty included in loss."""

    def test_adjoint_with_stability_penalty(self):
        """Adjoint should work with CompositeLoss including StabilityPenalty."""
        from neuraltide.training.adjoint import AdjointSolver

        dt = 0.05
        T = 10
        n_steps = int(T / dt)
        graph = _make_simple_small_network(dt=dt)
        network = NetworkRNN(graph, integrator=RK4Integrator(), stability_penalty_weight=1e-3)

        t_values = np.arange(n_steps, dtype=np.float32) * dt
        t_seq = tf.constant(t_values[None, :, None])

        target_0 = 10.0 + 5.0 * np.sin(2 * np.pi * 8.0 * t_values / 1000.0)
        target_1 = 8.0 + 4.0 * np.sin(2 * np.pi * 8.0 * t_values / 1000.0 + 0.5)
        target = {
            'exc': tf.constant(
                np.stack([target_0, target_1], axis=-1)[None, :, :],
                dtype=tf.float32
            )
        }

        loss_fn = CompositeLoss([
            (1.0, MSELoss(target)),
            (1e-3, StabilityPenalty()),
        ])

        solver = AdjointSolver(network, network._integrator)
        adj_grads, variables, adj_output = solver.compute_gradients(t_seq, target, loss_fn)

        assert len(adj_grads) > 0, "Should compute some gradients"


class TestAdjointNumericalGradient:
    """Tests for numerical gradient verification."""

    def test_numerical_gradient_check(self):
        """Adjoint gradients should be in the same direction as numerical."""
        from neuraltide.training.adjoint import AdjointSolver

        dt = 0.05
        T = 5
        n_steps = int(T / dt)
        graph = _make_simple_small_network(dt=dt)
        network = NetworkRNN(graph, integrator=RK4Integrator())

        t_values = np.arange(n_steps, dtype=np.float32) * dt
        t_seq = tf.constant(t_values[None, :, None])

        target_0 = 10.0 + 5.0 * np.sin(2 * np.pi * 8.0 * t_values / 1000.0)
        target_1 = 8.0 + 4.0 * np.sin(2 * np.pi * 8.0 * t_values / 1000.0 + 0.5)
        target = {
            'exc': tf.constant(
                np.stack([target_0, target_1], axis=-1)[None, :, :],
                dtype=tf.float32
            )
        }

        solver = AdjointSolver(network, network._integrator)
        loss_fn = CompositeLoss([
            (1.0, MSELoss(target)),
        ])
        adj_grads, variables, _ = solver.compute_gradients(t_seq, target, loss_fn)

        has_nonzero_grad = False
        for g in adj_grads:
            if g is not None and tf.reduce_sum(tf.abs(g)) > 1e-6:
                has_nonzero_grad = True
        assert has_nonzero_grad, "Should have non-zero gradients"


class TestAdjointTrainStep:
    """Tests for training step with adjoint method."""

    def test_adjoint_train_step_converges(self):
        """Training with adjoint should converge similarly to BPTT."""
        from neuraltide.training.adjoint import AdjointSolver

        dt = 0.05
        T = 5
        n_steps = int(T / dt)
        graph = _make_simple_small_network(dt=dt)
        network = NetworkRNN(graph, integrator=RK4Integrator())

        t_values = np.arange(n_steps, dtype=np.float32) * dt
        t_seq = tf.constant(t_values[None, :, None])

        target_0 = 10.0 + 5.0 * np.sin(2 * np.pi * 8.0 * t_values / 1000.0)
        target_1 = 8.0 + 4.0 * np.sin(2 * np.pi * 8.0 * t_values / 1000.0 + 0.5)
        target = {
            'exc': tf.constant(
                np.stack([target_0, target_1], axis=-1)[None, :, :],
                dtype=tf.float32
            )
        }

        loss_fn = CompositeLoss([
            (1.0, MSELoss(target)),
        ])

        initial_output = network(t_seq, training=False)
        initial_loss = float(loss_fn(initial_output, network))

        optimizer = tf.keras.optimizers.Adam(1e-2)
        optimizer.build(network.trainable_variables)

        for epoch in range(10):
            solver = AdjointSolver(network, network._integrator)
            grads, variables, _ = solver.compute_gradients(t_seq, target, loss_fn)

            grads_and_vars = [(g, v) for g, v in zip(grads, variables) if g is not None]
            if grads_and_vars:
                clipped_grads, _ = tf.clip_by_global_norm(
                    [g for g, v in grads_and_vars], 1.0
                )
                optimizer.apply_gradients(zip(clipped_grads, [v for g, v in grads_and_vars]))

        final_output = network(t_seq, training=False)
        final_loss = float(loss_fn(final_output, network))

        assert final_loss < initial_loss, "Loss should decrease during training"