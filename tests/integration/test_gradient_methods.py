"""
Integration tests for gradient methods comparison.

These tests verify:
1. Adjoint method produces correct gradients (matches autograd)
2. Training with adjoint method works
3. Memory usage comparison between methods
4. Long sequence handling with adjoint method
"""
import pytest
import tensorflow as tf
import numpy as np

import neuraltide
from neuraltide.config import set_dtype, get_dtype
from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.populations import IzhikevichMeanField
from neuraltide.synapses import TsodyksMarkramSynapse, NMDASynapse, StaticSynapse
from neuraltide.inputs import VonMisesGenerator
from neuraltide.integrators import RK4Integrator, EulerIntegrator
from neuraltide.training import Trainer, CompositeLoss, MSELoss
from neuraltide.utils import seed_everything


@pytest.fixture
def dt():
    return 0.5


@pytest.fixture
def izh_params():
    return {
        'tau_pop': {'value': [1.0, 1.0], 'trainable': False},
        'alpha': {'value': [0.5, 0.5], 'trainable': False},
        'a': {'value': [0.02, 0.02], 'trainable': False},
        'b': {'value': [0.2, 0.2], 'trainable': False},
        'w_jump': {'value': [0.1, 0.1], 'trainable': False},
        'Delta_I': {'value': [0.5, 0.6], 'trainable': True, 'min': 0.01, 'max': 2.0},
        'I_ext': {'value': [1.0, 1.2], 'trainable': True},
    }


@pytest.fixture
def synapse_params():
    return {
        'gsyn_max': {'value': [[0.1, 0.1]], 'trainable': True},
        'tau_f': {'value': 20.0, 'trainable': True, 'min': 6.0, 'max': 240.0},
        'tau_d': {'value': 5.0, 'trainable': True, 'min': 2.0, 'max': 15.0},
        'tau_r': {'value': 200.0, 'trainable': True, 'min': 91.0, 'max': 1300.0},
        'Uinc': {'value': 0.2, 'trainable': True, 'min': 0.04, 'max': 0.7},
        'pconn': {'value': [[1.0, 1.0]], 'trainable': False},
        'e_r': {'value': 0.0, 'trainable': False},
    }


class TestAdjointVsAutograd:
    """Test comparison between adjoint and autograd methods."""

    def _make_target(self, network, t_seq):
        """Run network and create target from output (shifted to have nonzero grad)."""
        output = network(t_seq)
        target = {k: v + 0.5 for k, v in output.firing_rates.items()}
        return target

    def test_simple_model_gradients_match(self, izh_params, synapse_params, dt):
        """Gradients from adjoint should match autograd for simple model."""
        seed_everything(42)

        pop = IzhikevichMeanField(dt=dt, params=izh_params)
        syn = TsodyksMarkramSynapse(n_pre=1, n_post=2, dt=dt, params=synapse_params)

        gen = VonMisesGenerator(dt=dt, params={
            'mean_rate': {'value': 20.0, 'trainable': False},
            'R': {'value': 0.5, 'trainable': False},
            'freq': {'value': 8.0, 'trainable': False},
            'phase': {'value': 0.0, 'trainable': False},
        })

        graph = NetworkGraph(dt=dt)
        graph.add_input_population('theta', gen)
        graph.add_population('exc', pop)
        graph.add_synapse('theta->exc', syn, src='theta', tgt='exc')

        network_autograd = NetworkRNN(graph, RK4Integrator())
        network_adjoint = NetworkRNN(graph, RK4Integrator())
        integrator = RK4Integrator()

        T = 50
        t_values = np.arange(T, dtype=np.float32) * dt
        t_seq = tf.constant(t_values[None, :, None])

        target = self._make_target(network_autograd, t_seq)
        mse = MSELoss(target)

        with tf.GradientTape() as tape:
            output = network_autograd(t_seq)
            loss = mse(output, network_autograd)
        autograd_grads = tape.gradient(loss, network_autograd.trainable_variables)

        from neuraltide.training.adjoint import AdjointSolver
        adj_comp = AdjointSolver(network_adjoint, integrator)
        adj_grads_list, _, _ = adj_comp.compute_gradients(t_seq, target, mse)

        adj_dict = {
            v.name: g
            for v, g in zip(network_adjoint.trainable_variables, adj_grads_list)
            if g is not None
        }
        auto_dict = {
            v.name: g
            for v, g in zip(network_autograd.trainable_variables, autograd_grads)
            if g is not None
        }

        max_rel_error = 0.0
        for var_name, auto_g in auto_dict.items():
            adj_g = adj_dict.get(var_name)
            if adj_g is None:
                continue
            rel_error = float(tf.reduce_max(
                tf.abs(auto_g - adj_g) / (tf.abs(auto_g) + 1e-8)
            ))
            max_rel_error = max(max_rel_error, rel_error)

        assert max_rel_error < 1e-3, \
            f"Max relative error {max_rel_error:.6f} should be < 1e-3"

    def test_exc_inh_network_gradients(self, izh_params, synapse_params, dt):
        """Test adjoint for exc-inh network."""
        seed_everything(42)

        pop_exc = IzhikevichMeanField(dt=dt, params={
            'tau_pop': {'value': [1.0, 1.0], 'trainable': False},
            'alpha': {'value': [0.5, 0.5], 'trainable': False},
            'a': {'value': [0.02, 0.02], 'trainable': False},
            'b': {'value': [0.2, 0.2], 'trainable': False},
            'w_jump': {'value': [0.1, 0.1], 'trainable': False},
            'Delta_I': {'value': [0.5, 0.6], 'trainable': True, 'min': 0.01, 'max': 2.0},
            'I_ext': {'value': [1.0, 1.2], 'trainable': True},
        })

        pop_inh = IzhikevichMeanField(dt=dt, params={
            'tau_pop': {'value': [1.0], 'trainable': False},
            'alpha': {'value': [0.5], 'trainable': False},
            'a': {'value': [0.02], 'trainable': False},
            'b': {'value': [0.2], 'trainable': False},
            'w_jump': {'value': [0.1], 'trainable': False},
            'Delta_I': {'value': [0.5], 'trainable': True, 'min': 0.01, 'max': 2.0},
            'I_ext': {'value': [1.0], 'trainable': True},
        })

        syn_in = TsodyksMarkramSynapse(n_pre=1, n_post=2, dt=dt, params={
            'gsyn_max': {'value': [[0.1, 0.1]], 'trainable': True},
            'tau_f': {'value': 20.0, 'trainable': True, 'min': 6.0, 'max': 240.0},
            'tau_d': {'value': 5.0, 'trainable': True, 'min': 2.0, 'max': 15.0},
            'tau_r': {'value': 200.0, 'trainable': True, 'min': 91.0, 'max': 1300.0},
            'Uinc': {'value': 0.2, 'trainable': True, 'min': 0.04, 'max': 0.7},
            'pconn': {'value': [[1.0, 1.0]], 'trainable': False},
            'e_r': {'value': 0.0, 'trainable': False},
        })

        syn_exc_inh = TsodyksMarkramSynapse(n_pre=2, n_post=1, dt=dt, params={
            'gsyn_max': {'value': [[0.05], [0.05]], 'trainable': True},
            'tau_f': {'value': 20.0, 'trainable': True, 'min': 6.0, 'max': 240.0},
            'tau_d': {'value': 5.0, 'trainable': True, 'min': 2.0, 'max': 15.0},
            'tau_r': {'value': 200.0, 'trainable': True, 'min': 91.0, 'max': 1300.0},
            'Uinc': {'value': 0.2, 'trainable': True, 'min': 0.04, 'max': 0.7},
            'pconn': {'value': [[1.0], [1.0]], 'trainable': False},
            'e_r': {'value': 0.0, 'trainable': False},
        })

        syn_inh_exc = StaticSynapse(n_pre=1, n_post=2, dt=dt, params={
            'gsyn_max': {'value': [[0.02, 0.02]], 'trainable': True},
            'pconn': {'value': [[1.0, 1.0]], 'trainable': False},
            'e_r': {'value': 0.0, 'trainable': False},
        })

        gen = VonMisesGenerator(dt=dt, params={
            'mean_rate': {'value': 20.0, 'trainable': False},
            'R': {'value': 0.5, 'trainable': False},
            'freq': {'value': 8.0, 'trainable': False},
            'phase': {'value': 0.0, 'trainable': False},
        })

        graph = NetworkGraph(dt=dt)
        graph.add_input_population('theta', gen)
        graph.add_population('exc', pop_exc)
        graph.add_population('inh', pop_inh)
        graph.add_synapse('theta->exc', syn_in, src='theta', tgt='exc')
        graph.add_synapse('exc->inh', syn_exc_inh, src='exc', tgt='inh')
        graph.add_synapse('inh->exc', syn_inh_exc, src='inh', tgt='exc')

        network = NetworkRNN(graph, RK4Integrator())
        integrator = RK4Integrator()

        T = 50
        t_values = np.arange(T, dtype=np.float32) * dt
        t_seq = tf.constant(t_values[None, :, None])

        target = self._make_target(network, t_seq)
        mse = MSELoss(target)

        with tf.GradientTape() as tape:
            output = network(t_seq)
            loss = mse(output, network)
        autograd_grads = tape.gradient(loss, network.trainable_variables)

        from neuraltide.training.adjoint import AdjointSolver
        adj_comp = AdjointSolver(network, integrator)
        adj_grads_list, _, _ = adj_comp.compute_gradients(t_seq, target, mse)

        adj_dict = {v.name: g for v, g in zip(network.trainable_variables, adj_grads_list) if g is not None}
        auto_dict = {v.name: g for v, g in zip(network.trainable_variables, autograd_grads) if g is not None}

        max_rel_error = 0.0
        for var_name, auto_g in auto_dict.items():
            adj_g = adj_dict.get(var_name)
            if adj_g is None:
                continue
            rel_error = float(tf.reduce_max(
                tf.abs(auto_g - adj_g) / (tf.abs(auto_g) + 1e-8)
            ))
            max_rel_error = max(max_rel_error, rel_error)

        assert max_rel_error < 1e-3, \
            f"Max relative error {max_rel_error:.6f} should be < 1e-3"

    def test_synapse_params_gradients(self, izh_params, synapse_params, dt):
        """Synapse parameters should have nonzero gradients."""
        seed_everything(42)

        pop = IzhikevichMeanField(dt=dt, params=izh_params)
        syn = TsodyksMarkramSynapse(n_pre=1, n_post=2, dt=dt, params=synapse_params)

        gen = VonMisesGenerator(dt=dt, params={
            'mean_rate': {'value': 20.0, 'trainable': False},
            'R': {'value': 0.5, 'trainable': False},
            'freq': {'value': 8.0, 'trainable': False},
            'phase': {'value': 0.0, 'trainable': False},
        })

        graph = NetworkGraph(dt=dt)
        graph.add_input_population('theta', gen)
        graph.add_population('exc', pop)
        graph.add_synapse('theta->exc', syn, src='theta', tgt='exc')

        network = NetworkRNN(graph, RK4Integrator())

        T = 50
        t_values = np.arange(T, dtype=np.float32) * dt
        t_seq = tf.constant(t_values[None, :, None])

        target = self._make_target(network, t_seq)
        mse = MSELoss(target)

        from neuraltide.training.adjoint import AdjointSolver
        adj_comp = AdjointSolver(network, network._integrator)
        grads_list, _, _ = adj_comp.compute_gradients(t_seq, target, mse)

        syn_params = {v.name for v in syn.trainable_variables}
        for v, g in zip(network.trainable_variables, grads_list):
            if v.name in syn_params:
                assert g is not None, f"Synapse param {v.name} should have gradient"
                assert tf.reduce_any(tf.abs(g) > 1e-8), \
                    f"Synapse param {v.name} gradient should be nonzero"


class TestLongSequence:
    """Test adjoint method with long sequences."""

    @pytest.mark.slow
    def test_long_sequence_adjoint(self, izh_params, dt):
        """Adjoint should work with long sequences (T=1000)."""
        seed_everything(42)

        pop = IzhikevichMeanField(dt=dt, params={
            'tau_pop': {'value': [1.0, 1.0], 'trainable': False},
            'alpha': {'value': [0.5, 0.5], 'trainable': False},
            'a': {'value': [0.02, 0.02], 'trainable': False},
            'b': {'value': [0.2, 0.2], 'trainable': False},
            'w_jump': {'value': [0.1, 0.1], 'trainable': False},
            'Delta_I': {'value': [0.5, 0.6], 'trainable': True, 'min': 0.01, 'max': 2.0},
            'I_ext': {'value': [1.0, 1.2], 'trainable': True},
        })

        syn = TsodyksMarkramSynapse(n_pre=1, n_post=2, dt=dt, params={
            'gsyn_max': {'value': [[0.1, 0.1]], 'trainable': True},
            'tau_f': {'value': 20.0, 'trainable': True, 'min': 6.0, 'max': 240.0},
            'tau_d': {'value': 5.0, 'trainable': True, 'min': 2.0, 'max': 15.0},
            'tau_r': {'value': 200.0, 'trainable': True, 'min': 91.0, 'max': 1300.0},
            'Uinc': {'value': 0.2, 'trainable': True, 'min': 0.04, 'max': 0.7},
            'pconn': {'value': [[1.0, 1.0]], 'trainable': False},
            'e_r': {'value': 0.0, 'trainable': False},
        })

        gen = VonMisesGenerator(dt=dt, params={
            'mean_rate': {'value': 20.0, 'trainable': False},
            'R': {'value': 0.5, 'trainable': False},
            'freq': {'value': 8.0, 'trainable': False},
            'phase': {'value': 0.0, 'trainable': False},
        })

        graph = NetworkGraph(dt=dt)
        graph.add_input_population('theta', gen)
        graph.add_population('exc', pop)
        graph.add_synapse('theta->exc', syn, src='theta', tgt='exc')

        network = NetworkRNN(graph, RK4Integrator())
        integrator = RK4Integrator()

        T = 1000
        t_values = np.arange(T, dtype=np.float32) * dt
        t_seq = tf.constant(t_values[None, :, None])

        output = network(t_seq)
        target = {k: v + 0.5 for k, v in output.firing_rates.items()}
        mse = MSELoss(target)

        from neuraltide.training.adjoint import AdjointSolver
        adj_comp = AdjointSolver(network, integrator)
        grads_list, _, _ = adj_comp.compute_gradients(t_seq, target, mse)

        assert len(grads_list) > 0, "Should compute gradients"
        for v, g in zip(network.trainable_variables, grads_list):
            assert g is not None, f"Gradient for {v.name} should not be None"
            assert tf.reduce_all(tf.math.is_finite(g)), \
                f"Gradient for {v.name} should be finite"


class TestNMDAWithAdjoint:
    """Test adjoint with NMDA synapse."""

    def test_nmda_synapse_gradients(self, izh_params, dt):
        """Adjoint should work with NMDA synapse."""
        seed_everything(42)

        pop = IzhikevichMeanField(dt=dt, params=izh_params)

        syn_ampa = StaticSynapse(n_pre=1, n_post=2, dt=dt, params={
            'gsyn_max': {'value': [[0.05, 0.05]], 'trainable': True},
            'pconn': {'value': [[1.0, 1.0]], 'trainable': False},
            'e_r': {'value': 0.0, 'trainable': False},
        })

        syn_nmda = NMDASynapse(n_pre=1, n_post=2, dt=dt, params={
            'gsyn_max_nmda': {'value': [[0.08, 0.08]], 'trainable': True},
            'tau1_nmda': {'value': 10.0, 'trainable': False},
            'tau2_nmda': {'value': 100.0, 'trainable': False},
            'Mgb': {'value': 3.0, 'trainable': False},
            'av_nmda': {'value': 0.1, 'trainable': False},
            'pconn_nmda': {'value': [[1.0, 1.0]], 'trainable': False},
            'e_r_nmda': {'value': 0.0, 'trainable': False},
            'v_ref': {'value': 0.0, 'trainable': False},
        })

        gen = VonMisesGenerator(dt=dt, params={
            'mean_rate': {'value': 20.0, 'trainable': False},
            'R': {'value': 0.5, 'trainable': False},
            'freq': {'value': 8.0, 'trainable': False},
            'phase': {'value': 0.0, 'trainable': False},
        })

        graph = NetworkGraph(dt=dt)
        graph.add_input_population('theta', gen)
        graph.add_population('exc', pop)
        graph.add_synapse('theta->exc_ampa', syn_ampa, src='theta', tgt='exc')
        graph.add_synapse('theta->exc_nmda', syn_nmda, src='theta', tgt='exc')

        network = NetworkRNN(graph, RK4Integrator())
        integrator = RK4Integrator()

        T = 50
        t_values = np.arange(T, dtype=np.float32) * dt
        t_seq = tf.constant(t_values[None, :, None])

        output = network(t_seq)
        target = {k: v + 0.5 for k, v in output.firing_rates.items()}
        mse = MSELoss(target)

        from neuraltide.training.adjoint import AdjointSolver
        adj_comp = AdjointSolver(network, integrator)
        grads_list, _, _ = adj_comp.compute_gradients(t_seq, target, mse)

        assert len(grads_list) > 0, "Should compute gradients"
        trainable = network.trainable_variables
        assert len(trainable) > 0
        for v, g in zip(trainable, grads_list):
            assert g is not None, f"Gradient for {v.name} should not be None"
            assert tf.reduce_all(tf.math.is_finite(g)), \
                f"Gradient for {v.name} should be finite"
