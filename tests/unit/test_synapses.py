import pytest
import tensorflow as tf
import numpy as np

import neuraltide
import neuraltide.config
from neuraltide.core.base import SynapseModel
from neuraltide.integrators import EulerIntegrator, RK4Integrator
from neuraltide.synapses import TsodyksMarkramSynapse, NMDASynapse, StaticSynapse, CompositeSynapse


def _make_default_synapse_params(n_pre, n_post):
    return {
        'gsyn_max': np.ones((n_pre, n_post)) * 0.01,
        'tau_f': np.ones((n_pre, n_post)) * 100.0,
        'tau_d': np.ones((n_pre, n_post)) * 200.0,
        'tau_r': np.ones((n_pre, n_post)) * 10.0,
        'Uinc': np.ones((n_pre, n_post)) * 0.5,
        'pconn': np.ones((n_pre, n_post)),
        'e_r': np.zeros((n_pre, n_post)),
    }


def _make_default_nmda_params(n_pre, n_post):
    return {
        'gsyn_max_nmda': np.ones((n_pre, n_post)) * 0.01,
        'tau1_nmda': np.ones((n_pre, n_post)) * 5.0,
        'tau2_nmda': np.ones((n_pre, n_post)) * 70.0,
        'Mgb': np.ones((n_pre, n_post)) * 0.2,
        'av_nmda': np.ones((n_pre, n_post)) * 0.08,
        'pconn_nmda': np.ones((n_pre, n_post)),
        'e_r_nmda': np.zeros((n_pre, n_post)),
        'v_ref': np.zeros((n_pre, n_post)),
    }


def _make_default_static_params(n_pre, n_post):
    return {
        'gsyn_max': np.ones((n_pre, n_post)) * 0.01,
        'pconn': np.ones((n_pre, n_post)),
        'e_r': np.zeros((n_pre, n_post)),
    }


class TestTsodyksMarkramDerivatives:
    """Tests for TsodyksMarkramSynapse derivatives method."""

    def test_has_derivatives_method(self):
        """Synapse should have derivatives method."""
        n_pre, n_post = 2, 3
        params = _make_default_synapse_params(n_pre, n_post)
        syn = TsodyksMarkramSynapse(n_pre=n_pre, n_post=n_post, dt=0.1, params=params)

        assert hasattr(syn, 'derivatives'), "Synapse should have derivatives method"
        assert callable(syn.derivatives), "derivatives should be callable"

    def test_derivatives_shape(self):
        """derivatives should return list of same shape as state."""
        n_pre, n_post = 2, 3
        dt = 0.1
        params = _make_default_synapse_params(n_pre, n_post)
        syn = TsodyksMarkramSynapse(n_pre=n_pre, n_post=n_post, dt=dt, params=params)

        state = syn.get_initial_state()
        pre_rate = tf.ones([1, n_pre])
        post_v = tf.zeros([1, n_post])

        derivs = syn.derivatives(state, pre_rate, post_v)

        assert len(derivs) == len(state), "derivatives should have same length as state"
        for d, s in zip(derivs, state):
            assert d.shape == s.shape, f"Derivative shape {d.shape} should match state shape {s.shape}"

    def test_derivatives_zero_input_no_change(self):
        """With zero input, derivatives should reflect decay."""
        n_pre, n_post = 2, 3
        dt = 0.1
        params = _make_default_synapse_params(n_pre, n_post)
        syn = TsodyksMarkramSynapse(n_pre=n_pre, n_post=n_post, dt=dt, params=params)

        state = [
            tf.ones([n_pre, n_post]) * 0.5,
            tf.ones([n_pre, n_post]) * 0.1,
            tf.ones([n_pre, n_post]) * 0.2,
        ]
        pre_rate = tf.zeros([1, n_pre])
        post_v = tf.zeros([1, n_post])

        derivs = syn.derivatives(state, pre_rate, post_v)

        for d in derivs:
            assert not tf.reduce_any(tf.math.is_nan(d)), "Derivatives should not contain NaN"

    def test_derivatives_euler_integration(self):
        """Euler integration of derivatives should produce similar results to forward."""
        n_pre, n_post = 2, 3
        dt = 0.1
        params = _make_default_synapse_params(n_pre, n_post)
        syn = TsodyksMarkramSynapse(n_pre=n_pre, n_post=n_post, dt=dt, params=params)

        pre_rate = tf.ones([1, n_pre]) * 100.0
        post_v = tf.zeros([1, n_post])

        state = syn.get_initial_state()
        target_state_fwd = None
        for _ in range(100):
            current_dict, new_state = syn.forward(pre_rate, post_v, state, dt)
            target_state_fwd = new_state
            state = new_state

        state = syn.get_initial_state()
        integrator = EulerIntegrator()
        for _ in range(100):
            derivs = syn.derivatives(state, pre_rate, post_v)
            new_state = [
                state[i] + dt * derivs[i]
                for i in range(len(state))
            ]
            state = new_state

        for i in range(len(state)):
            diff = tf.reduce_mean(tf.abs(state[i] - target_state_fwd[i]))
            assert diff < 0.1, f"State {i}: Euler diff {diff:.4f} should be small"


class TestNMDADerivatives:
    """Tests for NMDASynapse derivatives method."""

    def test_has_derivatives_method(self):
        """Synapse should have derivatives method."""
        n_pre, n_post = 2, 3
        params = _make_default_nmda_params(n_pre, n_post)
        syn = NMDASynapse(n_pre=n_pre, n_post=n_post, dt=0.1, params=params)

        assert hasattr(syn, 'derivatives'), "Synapse should have derivatives method"
        assert callable(syn.derivatives), "derivatives should be callable"

    def test_derivatives_shape(self):
        """derivatives should return list of same shape as state."""
        n_pre, n_post = 2, 3
        dt = 0.1
        params = _make_default_nmda_params(n_pre, n_post)
        syn = NMDASynapse(n_pre=n_pre, n_post=n_post, dt=dt, params=params)

        state = syn.get_initial_state()
        pre_rate = tf.ones([1, n_pre])
        post_v = tf.zeros([1, n_post])

        derivs = syn.derivatives(state, pre_rate, post_v)

        assert len(derivs) == len(state), "derivatives should have same length as state"
        for d, s in zip(derivs, state):
            assert d.shape == s.shape, f"Derivative shape {d.shape} should match state shape {s.shape}"

    def test_derivatives_euler_integration(self):
        """Euler integration of derivatives should produce similar results to forward."""
        n_pre, n_post = 2, 3
        dt = 0.1
        params = _make_default_nmda_params(n_pre, n_post)
        syn = NMDASynapse(n_pre=n_pre, n_post=n_post, dt=dt, params=params)

        pre_rate = tf.ones([1, n_pre]) * 100.0
        post_v = tf.zeros([1, n_post])

        state = syn.get_initial_state()
        target_state_fwd = None
        for _ in range(100):
            current_dict, new_state = syn.forward(pre_rate, post_v, state, dt)
            target_state_fwd = new_state
            state = new_state

        state = syn.get_initial_state()
        for _ in range(100):
            derivs = syn.derivatives(state, pre_rate, post_v)
            new_state = [
                state[i] + dt * derivs[i]
                for i in range(len(state))
            ]
            state = new_state

        for i in range(len(state)):
            diff = tf.reduce_mean(tf.abs(state[i] - target_state_fwd[i]))
            assert diff < 0.1, f"State {i}: Euler diff {diff:.4f} should be small"


class TestStaticSynapseDerivatives:
    """Tests for StaticSynapse derivatives method."""

    def test_has_derivatives_method(self):
        """Synapse should have derivatives method."""
        n_pre, n_post = 2, 3
        params = _make_default_static_params(n_pre, n_post)
        syn = StaticSynapse(n_pre=n_pre, n_post=n_post, dt=0.1, params=params)

        assert hasattr(syn, 'derivatives'), "Synapse should have derivatives method"
        assert callable(syn.derivatives), "derivatives should be callable"

    def test_derivatives_returns_empty_or_zero(self):
        """Static synapse with no state should return empty derivatives."""
        n_pre, n_post = 2, 3
        dt = 0.1
        params = _make_default_static_params(n_pre, n_post)
        syn = StaticSynapse(n_pre=n_pre, n_post=n_post, dt=dt, params=params)

        state = []
        pre_rate = tf.ones([1, n_pre])
        post_v = tf.zeros([1, n_post])

        derivs = syn.derivatives(state, pre_rate, post_v)

        assert len(derivs) == 0, "Static synapse should have empty derivatives"


class TestCompositeSynapseDerivatives:
    """Tests for CompositeSynapse derivatives method."""

    def test_has_derivatives_method(self):
        """Synapse should have derivatives method."""
        n_pre, n_post = 2, 3

        tm_params = _make_default_synapse_params(n_pre, n_post)
        tm_syn = TsodyksMarkramSynapse(n_pre=n_pre, n_post=n_post, dt=0.1, params=tm_params)

        nmda_params = _make_default_nmda_params(n_pre, n_post)
        nmda_syn = NMDASynapse(n_pre=n_pre, n_post=n_post, dt=0.1, params=nmda_params)

        composite = CompositeSynapse(
            n_pre=n_pre, n_post=n_post, dt=0.1,
            components=[('tm', tm_syn), ('nmda', nmda_syn)]
        )

        assert hasattr(composite, 'derivatives'), "CompositeSynapse should have derivatives method"
        assert callable(composite.derivatives), "derivatives should be callable"

    def test_derivatives_shape(self):
        """derivatives should return list matching combined state size."""
        n_pre, n_post = 2, 3

        tm_params = _make_default_synapse_params(n_pre, n_post)
        tm_syn = TsodyksMarkramSynapse(n_pre=n_pre, n_post=n_post, dt=0.1, params=tm_params)

        nmda_params = _make_default_nmda_params(n_pre, n_post)
        nmda_syn = NMDASynapse(n_pre=n_pre, n_post=n_post, dt=0.1, params=nmda_params)

        composite = CompositeSynapse(
            n_pre=n_pre, n_post=n_post, dt=0.1,
            components=[('tm', tm_syn), ('nmda', nmda_syn)]
        )

        state = composite.get_initial_state()
        pre_rate = tf.ones([1, n_pre])
        post_v = tf.zeros([1, n_post])

        derivs = composite.derivatives(state, pre_rate, post_v)

        assert len(derivs) == len(state), "derivatives should have same length as state"
        for d, s in zip(derivs, state):
            assert d.shape == s.shape, f"Derivative shape {d.shape} should match state shape {s.shape}"


class TestSynapseIntegratorInterface:
    """Test that synapses can be integrated like populations."""

    def test_synapse_integrator_interface(self):
        """Synapse should work with integrator.step_synapse interface."""
        n_pre, n_post = 2, 3
        dt = 0.1
        params = _make_default_synapse_params(n_pre, n_post)
        syn = TsodyksMarkramSynapse(n_pre=n_pre, n_post=n_post, dt=dt, params=params)

        pre_rate = tf.ones([1, n_pre]) * 50.0
        post_v = tf.zeros([1, n_post])

        state = syn.get_initial_state()

        integrator = EulerIntegrator()
        new_state, local_error = integrator.step_synapse(syn, state, pre_rate, post_v, dt)

        assert len(new_state) == len(state), "New state should have same length"
        assert local_error is not None, "Local error should be returned"