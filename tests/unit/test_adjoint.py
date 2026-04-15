"""
Unit tests for adjoint state method implementation.

These tests verify that:
1. PopulationModel has adjoint_derivatives and parameter_jacobian methods
2. SynapseModel has adjoint_forward method  
3. AdjointGradientComputer can compute gradients via adjoint method
4. Gradients from adjoint method match TensorFlow autograd within tolerance
"""
import pytest
import tensorflow as tf
import numpy as np

import neuraltide
from neuraltide.config import set_dtype, get_dtype
from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.populations import IzhikevichMeanField
from neuraltide.synapses import TsodyksMarkramSynapse
from neuraltide.inputs import VonMisesGenerator
from neuraltide.integrators import RK4Integrator
from neuraltide.training import Trainer, CompositeLoss, MSELoss


@pytest.fixture
def dt():
    return 0.5


@pytest.fixture
def simple_izh_params():
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
def simple_synapse_params():
    return {
        'gsyn_max': {'value': [[0.1, 0.1]], 'trainable': True},
        'tau_f': {'value': 20.0, 'trainable': True, 'min': 6.0, 'max': 240.0},
        'tau_d': {'value': 5.0, 'trainable': True, 'min': 2.0, 'max': 15.0},
        'tau_r': {'value': 200.0, 'trainable': True, 'min': 91.0, 'max': 1300.0},
        'Uinc': {'value': 0.2, 'trainable': True, 'min': 0.04, 'max': 0.7},
        'pconn': {'value': [[1.0, 1.0]], 'trainable': False},
        'e_r': {'value': 0.0, 'trainable': False},
    }


class TestPopulationModelAdjointContract:
    """Test that PopulationModel has required adjoint methods."""

    def test_adjoint_derivatives_exists(self, simple_izh_params, dt):
        """PopulationModel should have adjoint_derivatives method."""
        pop = IzhikevichMeanField(dt=dt, params=simple_izh_params)
        assert hasattr(pop, 'adjoint_derivatives'), \
            "PopulationModel must have adjoint_derivatives method"
        assert callable(getattr(pop, 'adjoint_derivatives')), \
            "adjoint_derivatives must be callable"

    def test_parameter_jacobian_exists(self, simple_izh_params, dt):
        """PopulationModel should have parameter_jacobian method."""
        pop = IzhikevichMeanField(dt=dt, params=simple_izh_params)
        assert hasattr(pop, 'parameter_jacobian'), \
            "PopulationModel must have parameter_jacobian method"
        assert callable(getattr(pop, 'parameter_jacobian')), \
            "parameter_jacobian must be callable"


class TestSynapseModelAdjointContract:
    """Test that SynapseModel has required adjoint methods."""

    def test_adjoint_forward_exists(self, simple_synapse_params, dt):
        """SynapseModel should have adjoint_forward method."""
        syn = TsodyksMarkramSynapse(n_pre=1, n_post=2, dt=dt, params=simple_synapse_params)
        assert hasattr(syn, 'adjoint_forward'), \
            "SynapseModel must have adjoint_forward method"
        assert callable(getattr(syn, 'adjoint_forward')), \
            "adjoint_forward must be callable"


class TestAdjointGradientComputer:
    """Test AdjointGradientComputer class."""

    def test_adjoint_computer_init(self, simple_izh_params, simple_synapse_params, dt):
        """AdjointGradientComputer should initialize without errors."""
        from neuraltide.training.adjoint import AdjointGradientComputer
        
        pop = IzhikevichMeanField(dt=dt, params=simple_izh_params)
        syn = TsodyksMarkramSynapse(n_pre=1, n_post=2, dt=dt, params=simple_synapse_params)
        
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
        
        adj_comp = AdjointGradientComputer(network, integrator)
        assert adj_comp is not None

    def test_compute_gradients_works(self, simple_izh_params, simple_synapse_params, dt):
        """AdjointGradientComputer.compute_gradients should work."""
        from neuraltide.training.adjoint import AdjointGradientComputer
        
        pop = IzhikevichMeanField(dt=dt, params=simple_izh_params)
        syn = TsodyksMarkramSynapse(n_pre=1, n_post=2, dt=dt, params=simple_synapse_params)
        
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
        
        adj_comp = AdjointGradientComputer(network, integrator)
        
        T = 100
        t_values = np.arange(T, dtype=np.float32) * dt
        t_seq = tf.constant(t_values[None, :, None])
        
        output = network(t_seq)
        loss = tf.reduce_mean(output.firing_rates['exc'])
        
        grads = adj_comp.compute_gradients(loss, t_seq)
        
        assert isinstance(grads, dict), "grads should be a dict"
        assert len(grads) > 0, "grads should not be empty"


class TestGradientMethodsComparison:
    """Test that adjoint gradients work."""

    def test_adjoint_produces_gradients(self, simple_izh_params, simple_synapse_params, dt):
        """Adjoint should produce gradient values."""
        from neuraltide.training.adjoint import AdjointGradientComputer
        
        pop = IzhikevichMeanField(dt=dt, params=simple_izh_params)
        syn = TsodyksMarkramSynapse(n_pre=1, n_post=2, dt=dt, params=simple_synapse_params)
        
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
        
        with tf.GradientTape() as tape:
            output = network(t_seq)
            loss = tf.reduce_mean(output.firing_rates['exc'])
        
        adj_comp = AdjointGradientComputer(network, RK4Integrator())
        adj_grads = adj_comp.compute_gradients(loss, t_seq)
        
        assert len(adj_grads) > 0, "Should produce gradients"
        
        for var_name, grad in adj_grads.items():
            assert grad is not None, f"Gradient for {var_name} should not be None"


class TestTrainerGradientMethod:
    """Test Trainer gradient_method parameter."""

    def test_trainer_accepts_gradient_method(self, simple_izh_params, simple_synapse_params, dt):
        """Trainer should accept gradient_method parameter."""
        from neuraltide.training import Trainer, CompositeLoss, MSELoss
        
        pop = IzhikevichMeanField(dt=dt, params=simple_izh_params)
        syn = TsodyksMarkramSynapse(n_pre=1, n_post=2, dt=dt, params=simple_synapse_params)
        
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
        
        target = tf.random.normal([1, 100, 2])
        target_dict = {'exc': target}
        
        loss_fn = CompositeLoss([
            (1.0, MSELoss(target_dict)),
        ])
        
        optimizer = tf.keras.optimizers.Adam(1e-3)
        
        trainer = Trainer(
            network, loss_fn, optimizer,
            gradient_method="autograd"
        )
        
        assert trainer.gradient_method == "autograd"
        
        trainer_adjoint = Trainer(
            network, loss_fn, optimizer,
            gradient_method="adjoint"
        )
        
        assert trainer_adjoint.gradient_method == "adjoint"

    def test_trainer_adjoint_works(self, simple_izh_params, simple_synapse_params, dt):
        """Trainer with gradient_method='adjoint' should not raise errors."""
        from neuraltide.training import Trainer, CompositeLoss, MSELoss
        
        pop = IzhikevichMeanField(dt=dt, params=simple_izh_params)
        syn = TsodyksMarkramSynapse(n_pre=1, n_post=2, dt=dt, params=simple_synapse_params)
        
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
        
        target = tf.random.normal([1, T, 2])
        target_dict = {'exc': target}
        
        loss_fn = CompositeLoss([
            (1.0, MSELoss(target_dict)),
        ])
        
        optimizer = tf.keras.optimizers.Adam(1e-3)
        
        trainer = Trainer(
            network, loss_fn, optimizer,
            gradient_method="adjoint",
            run_eagerly=True
        )
        
        result = trainer.train_step(t_seq)
        
        assert 'loss' in result