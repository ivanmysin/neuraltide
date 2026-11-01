import numpy as np
import pytest
import tensorflow as tf

from neuraltide.integrators import EulerIntegrator
from neuraltide.populations.wilson_cowan import WilsonCowan


@pytest.fixture
def params_2units():
    return {
        "tau": {"value": [10.0, 10.0], "trainable": False},
        "a": {"value": [1.0, 1.2], "trainable": False},
        "theta": {"value": [0.0, 0.0], "trainable": False},
        "w": {"value": [1.0, 0.8], "trainable": False},
        "I_ext": {"value": [1.0, 0.5], "trainable": False},
        "max_rate": {"value": [100.0, 100.0], "trainable": False},
    }


class TestWilsonCowan:
    def test_n_units_inferred(self, params_2units):
        pop = WilsonCowan(dt=0.5, params=params_2units)
        assert pop.n_units == 2

    def test_initial_state(self, params_2units):
        pop = WilsonCowan(dt=0.5, params=params_2units)
        state = pop.get_initial_state()
        assert len(state) == 1
        assert state[0].shape == (1, 2)
        assert np.allclose(state[0].numpy(), 0.0)

    def test_derivatives_shape(self, params_2units):
        pop = WilsonCowan(dt=0.5, params=params_2units)
        state = pop.get_initial_state()
        zero = tf.zeros([1, 2], dtype=tf.float32)
        dE = pop.derivatives(state, {"I_syn": zero, "g_syn": zero})
        assert len(dE) == 1
        assert dE[0].shape == (1, 2)

    def test_missing_param_raises(self):
        with pytest.raises(ValueError, match="required parameter"):
            WilsonCowan(dt=0.5, params={"tau": 10.0})

    def test_silent_state_stays_silent(self, params_2units):
        """Сильный отрицательный вход: активность остаётся ~нулевой.

        (Ноль сам по себе не является фиксированной точкой модели:
        при E=0, I=0 сигмоида даёт F(−θ) ≠ 0.)
        """
        params = {
            **params_2units,
            "I_ext": {"value": [-20.0, -20.0], "trainable": False},
        }
        pop = WilsonCowan(dt=0.5, params=params)
        integrator = EulerIntegrator()
        state = pop.get_initial_state()
        zero = tf.zeros([1, 2], dtype=tf.float32)
        syn = {"I_syn": zero, "g_syn": zero}
        for _ in range(100):
            state, _ = integrator.step(pop, state, syn)
        assert np.allclose(state[0].numpy(), 0.0, atol=1e-6)

    def test_steady_state_matches_fixed_point(self, params_2units):
        """Установившееся значение при постоянном входе совпадает с
        аналитической фиксированной точкой E* = F(w·E* + I_ext)."""
        pop = WilsonCowan(dt=0.5, params=params_2units)
        integrator = EulerIntegrator()
        state = pop.get_initial_state()
        zero = tf.zeros([1, 2], dtype=tf.float32)
        syn = {"I_syn": zero, "g_syn": zero}
        for _ in range(2000):
            state, _ = integrator.step(pop, state, syn)
        E_sim = state[0].numpy()[0]

        params = params_2units
        a = np.array(params["a"]["value"], dtype=np.float64)
        theta = np.array(params["theta"]["value"], dtype=np.float64)
        w = np.array(params["w"]["value"], dtype=np.float64)
        I_ext = np.array(params["I_ext"]["value"], dtype=np.float64)
        E_star = np.zeros(2)
        for i in range(2):
            E = 0.5
            for _ in range(1000):
                E = 1.0 / (1.0 + np.exp(-a[i] * (w[i] * E + I_ext[i] - theta[i])))
            E_star[i] = E
        assert np.allclose(E_sim, E_star, atol=1e-3)

    def test_firing_rate(self, params_2units):
        pop = WilsonCowan(dt=0.5, params=params_2units)
        state = [tf.constant([[0.5, -0.2]], dtype=tf.float32)]
        rate = pop.get_firing_rate(state)
        assert rate.shape == (1, 2)
        assert np.allclose(rate.numpy(), [[50.0, 0.0]])

    def test_observables_no_v_mean(self, params_2units):
        pop = WilsonCowan(dt=0.5, params=params_2units)
        obs = pop.observables(pop.get_initial_state())
        assert "firing_rate" in obs
        assert "v_mean" not in obs

    def test_trainable_w_gradient(self, params_2units):
        params = {
            **params_2units,
            "w": {"value": [1.0, 0.8], "trainable": True},
            "I_ext": {"value": [1.0, 0.5], "trainable": True},
        }
        pop = WilsonCowan(dt=0.5, params=params)
        integrator = EulerIntegrator()
        zero = tf.zeros([1, 2], dtype=tf.float32)
        syn = {"I_syn": zero, "g_syn": zero}
        with tf.GradientTape() as tape:
            state = pop.get_initial_state()
            for _ in range(20):
                state, _ = integrator.step(pop, state, syn)
            loss = tf.reduce_sum(pop.get_firing_rate(state))
        grads = tape.gradient(loss, pop.trainable_variables)
        assert len(grads) == 2
        assert all(g is not None for g in grads)
        assert all(np.all(np.isfinite(g.numpy())) for g in grads)

    def test_tf_function_compatible(self, params_2units):
        pop = WilsonCowan(dt=0.5, params=params_2units)
        zero = tf.zeros([1, 2], dtype=tf.float32)
        syn = {"I_syn": zero, "g_syn": zero}

        @tf.function
        def step(state):
            return pop.derivatives(state, syn)

        dE = step(pop.get_initial_state())
        assert dE[0].shape == (1, 2)
