import numpy as np
import pytest
import tensorflow as tf
from scipy.integrate import quad
from scipy.special import erfcx

from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.integrators import EulerIntegrator
from neuraltide.populations.brunel_fp import BrunelFokkerPlanck


@pytest.fixture
def brunel_params():
    """Параметры Brunel (2000): τ=20 мс, θ=20 мВ, V_r=10 мВ, τ_rp=2 мс.

    Режим у порога: μ0=18 мВ, σ=3 мВ.
    """
    return {
        "tau_m": {"value": 20.0, "trainable": False},
        "v_theta": {"value": 20.0, "trainable": False},
        "v_reset": {"value": 10.0, "trainable": False},
        "tau_rp": {"value": 2.0, "trainable": False},
        "mu_0": {"value": 18.0, "trainable": False},
        "sigma_ext": {"value": 3.0, "trainable": False},
    }


def brunel_stationary_rate(mu, sigma, v_theta, v_reset, tau, tau_rp):
    """Стационарная частота ν0 (Гц) из самосогласованного уравнения
    (21) Brunel (2000) для чисто Feedforward случая."""
    y_r = (v_reset - mu) / sigma
    y_th = (v_theta - mu) / sigma
    integral, _ = quad(lambda u: erfcx(-u), y_r, y_th, epsabs=1e-12)
    nu_ms = 1.0 / (tau_rp + tau * np.sqrt(np.pi) * integral)
    return 1000.0 * nu_ms


def make_step_fn(pop):
    """Скомпилированный цикл интегрирования (быстрый в тестах)."""
    integrator = EulerIntegrator()
    zero = tf.zeros([1, 1], dtype=tf.float32)
    syn = {"I_syn": zero, "g_syn": zero}

    @tf.function
    def step_n(state, n):
        for _ in tf.range(n):
            state, _ = integrator.step(pop, state, syn)
        return state

    return step_n


class TestBrunelFokkerPlanck:
    def test_initial_state_normalization(self, brunel_params):
        pop = BrunelFokkerPlanck(
            dt=0.5, params=brunel_params, grid_size=250, v_min=-80.0
        )
        state = pop.get_initial_state()
        assert len(state) == 3
        P, p_r, buf = state
        assert P.shape == (1, 250)
        assert p_r.shape == (1, 1)
        assert buf.shape == (1, 4)  # tau_rp/dt = 2/0.5
        mass = np.sum(P.numpy()) * pop.dV + p_r.numpy()[0, 0]
        assert mass == pytest.approx(1.0, abs=1e-5)
        assert P.numpy()[0, -1] == 0.0  # поглощающий узел на пороге

    def test_missing_param_raises(self):
        with pytest.raises(ValueError, match="required parameter"):
            BrunelFokkerPlanck(dt=0.5, params={"tau_m": 20.0})

    def test_probability_conservation(self, brunel_params):
        """Нормировка ∫P dV + p_r = 1 сохраняется на протяжении прогона."""
        params = {**brunel_params, "mu_0": {"value": 0.0, "trainable": False}}
        pop = BrunelFokkerPlanck(dt=0.5, params=params, grid_size=250, v_min=-80.0)
        state = make_step_fn(pop)(pop.get_initial_state(), 2000)
        mass = np.sum(state[0].numpy()) * pop.dV + state[1].numpy()[0, 0]
        assert mass == pytest.approx(1.0, abs=1e-4)

    def test_stationary_rate_matches_brunel(self, brunel_params):
        """Установившаяся частота совпадает с уравнением (21) Brunel (2000)."""
        pop = BrunelFokkerPlanck(
            dt=0.5, params=brunel_params, grid_size=250, v_min=-80.0
        )
        step_n = make_step_fn(pop)
        state = step_n(pop.get_initial_state(), 2400)  # 1200 мс прогрев
        rates = []
        state_tuple = state
        integrator = EulerIntegrator()
        zero = tf.zeros([1, 1], dtype=tf.float32)
        syn = {"I_syn": zero, "g_syn": zero}
        for _ in range(400):  # 200 мс измерение
            state_tuple, _ = integrator.step(pop, state_tuple, syn)
            rates.append(pop.get_firing_rate(state_tuple).numpy()[0, 0])
        nu_sim = np.mean(rates)

        nu_ref = brunel_stationary_rate(
            mu=18.0,
            sigma=3.0,
            v_theta=20.0,
            v_reset=10.0,
            tau=20.0,
            tau_rp=2.0,
        )
        assert nu_sim == pytest.approx(nu_ref, rel=0.05)
        assert np.all(np.array(rates) >= 0.0)

    def test_grid_convergence(self, brunel_params):
        """В надпороговом режиме ошибка доминирует по пространству:
        уточнение сетки приближает частоту к аналитике."""
        params = {
            **brunel_params,
            "mu_0": {"value": 40.0, "trainable": False},
            "sigma_ext": {"value": 2.0, "trainable": False},
        }
        errs = []
        nu_ref = brunel_stationary_rate(
            mu=40.0,
            sigma=2.0,
            v_theta=20.0,
            v_reset=10.0,
            tau=20.0,
            tau_rp=2.0,
        )
        for grid in (250, 500):
            pop = BrunelFokkerPlanck(dt=0.5, params=params, grid_size=grid, v_min=-80.0)
            step_n = make_step_fn(pop)
            state = step_n(pop.get_initial_state(), 3000)
            integrator = EulerIntegrator()
            zero = tf.zeros([1, 1], dtype=tf.float32)
            syn = {"I_syn": zero, "g_syn": zero}
            rates = []
            for _ in range(400):
                state, _ = integrator.step(pop, state, syn)
                rates.append(pop.get_firing_rate(state).numpy()[0, 0])
            nu = float(np.mean(rates))
            errs.append(abs(nu - nu_ref) / nu_ref)
        assert errs[1] < errs[0]
        assert errs[1] < 0.06

    def test_sigma_increases_rate(self, brunel_params):
        """Рост σ при фиксированном μ повышает частоту разрядов."""
        rates = []
        for sigma in (2.0, 4.0):
            params = {
                **brunel_params,
                "sigma_ext": {"value": sigma, "trainable": False},
            }
            pop = BrunelFokkerPlanck(dt=0.5, params=params, grid_size=250, v_min=-80.0)
            state = make_step_fn(pop)(pop.get_initial_state(), 1600)
            rates.append(pop.get_firing_rate(state).numpy()[0, 0])
        assert rates[1] > rates[0]

    def test_synaptic_input_drives_rate(self, brunel_params):
        """I_syn добавляется к μ: положительный ток повышает частоту."""
        pop = BrunelFokkerPlanck(
            dt=0.5, params=brunel_params, grid_size=250, v_min=-80.0
        )
        zero = tf.zeros([1, 1], dtype=tf.float32)
        rates = []
        for i_syn in (0.0, 4.0):
            integrator = EulerIntegrator()
            syn = {"I_syn": tf.constant([[i_syn]], dtype=tf.float32), "g_syn": zero}

            @tf.function
            def step_n(state, n):
                for _ in tf.range(n):
                    state, _ = integrator.step(pop, state, syn)
                return state

            state = step_n(pop.get_initial_state(), 1200)
            rates.append(pop.get_firing_rate(state).numpy()[0, 0])
        assert rates[1] > rates[0]

    def test_observables(self, brunel_params):
        pop = BrunelFokkerPlanck(
            dt=0.5, params=brunel_params, grid_size=250, v_min=-80.0
        )
        obs = pop.observables(pop.get_initial_state())
        assert "firing_rate" in obs
        assert "v_mean" in obs
        assert "p_refractory" in obs
        assert obs["firing_rate"].shape == (1, 1)
        assert obs["v_mean"].shape == (1, 1)

    def test_trainable_mu_gradient(self, brunel_params):
        params = {
            **brunel_params,
            "mu_0": {"value": 18.0, "trainable": True, "min": 0.0, "max": 25.0},
        }
        pop = BrunelFokkerPlanck(dt=0.5, params=params, grid_size=100, v_min=-80.0)
        integrator = EulerIntegrator()
        zero = tf.zeros([1, 1], dtype=tf.float32)
        syn = {"I_syn": zero, "g_syn": zero}
        with tf.GradientTape() as tape:
            state = pop.get_initial_state()
            for _ in range(10):
                state, _ = integrator.step(pop, state, syn)
            loss = tf.reduce_sum(pop.get_firing_rate(state))
        grads = tape.gradient(loss, pop.trainable_variables)
        assert len(grads) == 1
        assert grads[0] is not None
        assert np.all(np.isfinite(grads[0].numpy()))

    def test_tf_function_compatible(self, brunel_params):
        pop = BrunelFokkerPlanck(
            dt=0.5, params=brunel_params, grid_size=100, v_min=-80.0
        )
        zero = tf.zeros([1, 1], dtype=tf.float32)
        syn = {"I_syn": zero, "g_syn": zero}

        @tf.function
        def step(state):
            return pop.derivatives(state, syn)

        dP, dp_r, dbuf = step(pop.get_initial_state())
        assert dP.shape == (1, 100)
        assert dp_r.shape == (1, 1)
        assert dbuf.shape == (1, 4)

    def test_network_smoke(self, brunel_params):
        """Прогон через NetworkRNN: firing_rates [1, T, 1], значения конечны."""
        pop = BrunelFokkerPlanck(
            dt=0.5, params=brunel_params, grid_size=100, v_min=-80.0
        )
        graph = NetworkGraph(dt=0.5)
        with pytest.warns(UserWarning, match="no incoming synapses"):
            graph.add_population("fp", pop)
            graph.validate()
        network = NetworkRNN(graph, integrator=EulerIntegrator())
        t_seq = tf.constant(np.arange(0, 100, 0.5, dtype=np.float32)[None, :, None])
        output = network(t_seq)
        rates = output.firing_rates["fp"]
        assert rates.shape == (1, 200, 1)
        assert np.all(np.isfinite(rates.numpy()))
