"""Tests for neuraltide.keras module — BrainModelKeras."""
import numpy as np
import pytest
import tensorflow as tf

import neuraltide
from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.populations import IzhikevichMeanField
from neuraltide.synapses import TsodyksMarkramSynapse
from neuraltide.inputs import VonMisesGenerator
from neuraltide.integrators import RK4Integrator
from neuraltide.training import MSELoss, CompositeLoss


def _make_simple_network(dt=0.05):
    """Create a simple network for testing."""
    pop = IzhikevichMeanField(dt=dt, params={
        'tau_pop':   {'value': [1.0],   'trainable': False},
        'alpha':     {'value': [0.5],   'trainable': False},
        'a':         {'value': [0.02],  'trainable': False},
        'b':         {'value': [0.2],   'trainable': False},
        'w_jump':    {'value': [0.1],   'trainable': False},
        'Delta_I':   {'value': [0.5],   'trainable': True},
        'I_ext':     {'value': [0.1],   'trainable': True},
    })

    gen = VonMisesGenerator(dt=dt, params={
        'mean_rate': 20.0,
        'R': 0.5,
        'freq': 8.0,
        'phase': 0.0,
    })

    syn = TsodyksMarkramSynapse(n_pre=1, n_post=1, dt=dt, params={
        'gsyn_max': {'value': [[0.1]], 'trainable': True},
        'tau_f':    {'value': 20.0,  'trainable': False},
        'tau_d':    {'value': 5.0,   'trainable': False},
        'tau_r':    {'value': 200.0, 'trainable': False},
        'Uinc':     {'value': 0.2,   'trainable': False},
        'pconn':    {'value': [[1.0]], 'trainable': False},
        'e_r':      {'value': 0.0,   'trainable': False},
    })

    graph = NetworkGraph(dt=dt)
    graph.declare_input('theta', n_units=gen.n_units)
    graph.add_population('exc', pop)
    graph.add_synapse('theta->exc', syn, src='theta', tgt='exc')

    return graph, gen


class TestBrainModelKeras:
    """Tests for BrainModelKeras wrapper."""

    def test_instantiation(self):
        """BrainModelKeras should be creatable."""
        from neuraltide.model import BrainModelKeras

        graph, gen = _make_simple_network()
        model = BrainModelKeras(graph, RK4Integrator(), dt=0.05)
        assert model is not None

    def test_call_returns_network_output(self):
        """call() should return NetworkOutput with firing_rates."""
        from neuraltide.model import BrainModelKeras

        graph, gen = _make_simple_network()
        model = BrainModelKeras(graph, RK4Integrator(), dt=0.05)

        t_seq = np.arange(50, dtype=np.float32) * 0.05
        inputs = gen(tf.constant(t_seq)[np.newaxis, :, np.newaxis])
        inputs = inputs.numpy()  # [1, 50, 1]

        output = model(inputs, training=False)
        assert hasattr(output, 'firing_rates')
        assert 'exc' in output.firing_rates
        assert output.firing_rates['exc'].shape == (1, 50, 1)

    def test_t_sequence_inferred(self):
        """t_sequence should be generated from dt and T."""
        from neuraltide.model import BrainModelKeras

        graph, gen = _make_simple_network(dt=0.1)
        model = BrainModelKeras(graph, RK4Integrator(), dt=0.1)

        T = 30
        t_seq = np.arange(T, dtype=np.float32) * 0.1
        inputs = gen(tf.constant(t_seq)[np.newaxis, :, np.newaxis]).numpy()

        output = model(inputs, training=False)
        # Should produce output with T=30
        assert output.firing_rates['exc'].shape[1] == T

    def test_trainable_variables(self):
        """trainable_variables should return network's trainable vars."""
        from neuraltide.model import BrainModelKeras

        graph, gen = _make_simple_network()
        model = BrainModelKeras(graph, RK4Integrator(), dt=0.05)

        vars_ = model.trainable_variables
        assert len(vars_) > 0
        # Should include population and synapse params
        names = [v.name for v in vars_]
        assert any('Delta_I' in n or 'I_ext' in n for n in names)
        assert any('gsyn_max' in n for n in names)

    def test_compile_and_fit(self):
        """model.compile + model.fit should work."""
        from neuraltide.model import BrainModelKeras

        graph, gen = _make_simple_network()
        dt = 0.05
        T = 50
        n_steps = int(T / dt)

        t_values = np.arange(n_steps, dtype=np.float32) * dt
        inputs = gen(tf.constant(t_values)[np.newaxis, :, np.newaxis]).numpy()

        target = {
            'exc': 10.0 + 5.0 * np.sin(2 * np.pi * 8.0 * t_values / 1000.0)
        }
        target_arr = target['exc'][np.newaxis, :, np.newaxis].astype(np.float32)

        model = BrainModelKeras(graph, RK4Integrator(), dt=dt)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-2))

        # Use dict target
        target_dict = {'exc': tf.constant(target_arr)}
        history = model.fit(inputs, target_dict, epochs=5, verbose=0)

        assert 'loss' in history.history
        assert len(history.history['loss']) == 5
        # Loss should decrease
        assert history.history['loss'][-1] < history.history['loss'][0]

    def test_with_composite_loss(self):
        """BrainModelKeras should work with CompositeLoss."""
        from neuraltide.model import BrainModelKeras

        graph, gen = _make_simple_network()
        dt = 0.05
        T = 50
        n_steps = int(T / dt)

        t_values = np.arange(n_steps, dtype=np.float32) * dt
        inputs = gen(tf.constant(t_values)[np.newaxis, :, np.newaxis]).numpy()

        target_arr = (10.0 + 5.0 * np.sin(2 * np.pi * 8.0 * t_values / 1000.0))
        target_dict = {'exc': tf.constant(target_arr[np.newaxis, :, np.newaxis])}

        loss_fn = CompositeLoss([
            (1.0, MSELoss(target_dict)),
        ])

        model = BrainModelKeras(graph, RK4Integrator(), dt=dt, loss_fn=loss_fn)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-2))

        history = model.fit(inputs, target_dict, epochs=5, verbose=0)
        assert history.history['loss'][-1] < history.history['loss'][0]

    def test_network_property(self):
        """model.network should return the inner NetworkRNN."""
        from neuraltide.model import BrainModelKeras

        graph, gen = _make_simple_network()
        model = BrainModelKeras(graph, RK4Integrator(), dt=0.05)

        assert isinstance(model.network, NetworkRNN)
        assert model.network._graph is graph
