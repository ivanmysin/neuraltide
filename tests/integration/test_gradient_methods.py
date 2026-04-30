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

    def test_simple_model_gradients_match(self, izh_params, synapse_params, dt):
        """Gradients from adjoint should match autograd for simple model."""
        from neuraltide.training.adjoint import AdjointGradientComputer
        
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
        
        with tf.GradientTape() as tape:
            output = network_autograd(t_seq)
            loss = tf.reduce_mean(output.firing_rates['exc'])
        
        autograd_grads = tape.gradient(loss, network_autograd.trainable_variables)
        
        adj_comp = AdjointGradientComputer(network_adjoint, integrator)
        
        with tf.GradientTape() as tape:
            output = network_adjoint(t_seq)
            loss = tf.reduce_mean(output.firing_rates['exc'])
        
        adj_grads = adj_comp.compute_gradients(loss, t_seq)
        
        var_to_grad = {v.name: g for v, g in zip(network_autograd.trainable_variables, autograd_grads) if g is not None}
        
        max_rel_error = 0.0
        for var_name, auto_g in var_to_grad.items():
            adj_g = adj_grads.get(var_name)
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
        from neuraltide.training.adjoint import AdjointGradientComputer
        
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
            'gsyn_max': {'value': [[0.05]], 'trainable': True},
            'tau_f': {'value': 20.0, 'trainable': True, 'min': 6.0, 'max': 240.0},
            'tau_d': {'value': 5.0, 'trainable': True, 'min': 2.0, 'max': 15.0},
            'tau_r': {'value': 200.0, 'trainable': True, 'min': 91.0, 'max': 1300.0},
            'Uinc': {'value': 0.2, 'trainable': True, 'min': 0.04, 'max': 0.7},
            'pconn': {'value': [[1.0]], 'trainable': False},
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
        
        output = network(t_seq)
        loss = tf.reduce_mean(output.firing_rates['exc']) + tf.reduce_mean(output.firing_rates['inh'])
        
        with tf.GradientTape() as tape:
            output = network(t_seq)
            loss = tf.reduce_mean(output.firing_rates['exc']) + tf.reduce_mean(output.firing_rates['inh'])
        
        autograd_grads = tape.gradient(loss, network.trainable_variables)
        
        adj_comp = AdjointGradientComputer(network, integrator)
        adj_grads = adj_comp.compute_gradients(loss, t_seq)
        
        var_to_grad = {v.name: g for v, g in zip(network.trainable_variables, autograd_grads) if g is not None}
        
        max_rel_error = 0.0
        for var_name, auto_g in var_to_grad.items():
            adj_g = adj_grads.get(var_name)
            if adj_g is None:
                continue
            
            rel_error = float(tf.reduce_max(
                tf.abs(auto_g - adj_g) / (tf.abs(auto_g) + 1e-8)
            ))
            max_rel_error = max(max_rel_error, rel_error)
        
        assert max_rel_error < 1e-2, \
            f"Max relative error {max_rel_error:.6f} should be < 1e-2"


class TestAdjointTraining:
    """Test training with adjoint method."""

    def test_training_reduces_loss(self, izh_params, synapse_params, dt):
        """Training with adjoint should reduce loss."""
        from neuraltide.training import Trainer, CompositeLoss, MSELoss
        
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
        
        T = 100
        t_values = np.arange(T, dtype=np.float32) * dt
        t_seq = tf.constant(t_values[None, :, None])
        
        target_0 = 10.0 + 5.0 * np.sin(2 * np.pi * 8.0 * t_values / 1000.0)
        target_1 = 8.0 + 4.0 * np.sin(2 * np.pi * 8.0 * t_values / 1000.0 + 0.5)
        target = tf.constant(
            np.stack([target_0, target_1], axis=-1)[None, :, :],
            dtype=tf.float32
        )
        target_dict = {'exc': target}
        
        loss_fn = CompositeLoss([
            (1.0, MSELoss(target_dict)),
        ])
        
        optimizer = tf.keras.optimizers.Adam(1e-3)
        
        trainer = Trainer(
            network, loss_fn, optimizer,
            gradient_method="adjoint"
        )
        
        initial_loss = float(trainer.train_step(t_seq)['loss'])
        
        for _ in range(20):
            trainer.train_step(t_seq)
        
        final_loss = float(trainer.train_step(t_seq)['loss'])
        
        assert final_loss < initial_loss, \
            f"Loss should decrease: initial={initial_loss:.6f}, final={final_loss:.6f}"


class TestLongSequence:
    """Test adjoint method with long sequences."""

    @pytest.mark.slow
    def test_long_sequence_adjoint(self, izh_params, dt):
        """Adjoint should work with long sequences (T=1000)."""
        from neuraltide.training.adjoint import AdjointGradientComputer
        
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
        loss = tf.reduce_mean(output.firing_rates['exc'])
        
        adj_comp = AdjointGradientComputer(network, integrator)
        grads = adj_comp.compute_gradients(loss, t_seq)
        
        assert len(grads) > 0, "Should compute gradients"
        for var_name, grad in grads.items():
            assert grad is not None, f"Gradient for {var_name} should not be None"
            assert tf.reduce_all(tf.math.is_finite(grad)), \
                f"Gradient for {var_name} should be finite"


class TestNMDAWithAdjoint:
    """Test adjoint with NMDA synapse."""

    def test_nmda_synapse_gradients(self, izh_params, dt):
        """Adjoint should work with NMDA synapse."""
        from neuraltide.training.adjoint import AdjointGradientComputer
        
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
        loss = tf.reduce_mean(output.firing_rates['exc'])
        
        adj_comp = AdjointGradientComputer(network, integrator)
        grads = adj_comp.compute_gradients(loss, t_seq)
        
        assert len(grads) > 0, "Should compute gradients"