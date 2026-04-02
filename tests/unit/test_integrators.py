import pytest
import tensorflow as tf

import neuraltide
from neuraltide.core.base import PopulationModel
from neuraltide.integrators import EulerIntegrator, HeunIntegrator, RK4Integrator


class SimpleDecayPopulation(PopulationModel):
    """Простая модель: dr/dt = -r."""

    def __init__(self, n_units=1, dt=0.1, **kwargs):
        super().__init__(n_units=n_units, dt=dt, **kwargs)
        self.state_size = [tf.TensorShape([1, n_units])]

    def get_initial_state(self, batch_size=1):
        return [tf.ones([1, self.n_units], dtype=neuraltide.config.get_dtype())]

    def derivatives(self, state, total_synaptic_input):
        r = state[0]
        drdt = -r
        return [drdt]

    def get_firing_rate(self, state):
        return state[0]

    @property
    def parameter_spec(self):
        return {}


class NonlinearOscillatorPopulation(PopulationModel):
    """Нелинейный осциллятор для тестирования error estimate."""

    def __init__(self, n_units=1, dt=0.1, **kwargs):
        super().__init__(n_units=n_units, dt=dt, **kwargs)
        self.state_size = [tf.TensorShape([1, n_units])]

    def get_initial_state(self, batch_size=1):
        return [tf.constant([[2.0]], dtype=neuraltide.config.get_dtype())]

    def derivatives(self, state, total_synaptic_input):
        x = state[0]
        dxdt = x - x ** 3
        return [dxdt]

    def get_firing_rate(self, state):
        return state[0]

    @property
    def parameter_spec(self):
        return {}


class TestEulerIntegrator:
    def test_convergence(self):
        """Euler: dy/dt = -y, y(0)=1, ошибка < 5% за 1000 шагов dt=0.01."""
        dt = 0.01
        n_steps = 1000
        final_time = dt * n_steps
        analytical = float(tf.exp(-final_time))

        pop = SimpleDecayPopulation(n_units=1, dt=dt)
        integrator = EulerIntegrator()
        state = pop.get_initial_state()

        for _ in range(n_steps):
            state, _ = integrator.step(pop, state, {'I_syn': tf.zeros([1, 1]),
                                                     'g_syn': tf.zeros([1, 1])})

        numerical = float(state[0].numpy()[0, 0])
        error = abs(numerical - analytical)
        if analytical > 1e-10:
            relative_error = error / analytical
        else:
            relative_error = error
        assert relative_error < 0.05, f"Relative error {relative_error:.4f} > 0.05"

    def test_local_error_zero(self):
        """Euler local_error_estimate == 0.0."""
        pop = SimpleDecayPopulation(n_units=1, dt=0.1)
        integrator = EulerIntegrator()
        state = pop.get_initial_state()

        _, local_error = integrator.step(
            pop, state, {'I_syn': tf.zeros([1, 1]), 'g_syn': tf.zeros([1, 1])}
        )

        assert tf.reduce_all(local_error == 0.0)


class TestHeunIntegrator:
    def test_convergence(self):
        """Heun: dy/dt = -y, y(0)=1, ошибка < 0.1% за 1000 шагов dt=0.01."""
        dt = 0.01
        n_steps = 1000
        final_time = dt * n_steps
        analytical = float(tf.exp(-final_time))

        pop = SimpleDecayPopulation(n_units=1, dt=dt)
        integrator = HeunIntegrator()
        state = pop.get_initial_state()

        for _ in range(n_steps):
            state, _ = integrator.step(pop, state, {'I_syn': tf.zeros([1, 1]),
                                                      'g_syn': tf.zeros([1, 1])})

        numerical = float(state[0].numpy()[0, 0])
        error = abs(numerical - analytical)
        if analytical > 1e-10:
            relative_error = error / analytical
        else:
            relative_error = error
        assert relative_error < 0.001, f"Relative error {relative_error:.6f} > 0.001"


class TestRK4Integrator:
    def test_convergence(self):
        """RK4: dy/dt = -y, y(0)=1, ошибка < 1e-4% за 100 шагов dt=0.1."""
        dt = 0.1
        n_steps = 100
        final_time = dt * n_steps
        analytical = float(tf.exp(-final_time))

        pop = SimpleDecayPopulation(n_units=1, dt=dt)
        integrator = RK4Integrator()
        state = pop.get_initial_state()

        for _ in range(n_steps):
            state, _ = integrator.step(pop, state, {'I_syn': tf.zeros([1, 1]),
                                                      'g_syn': tf.zeros([1, 1])})

        numerical = float(state[0].numpy()[0, 0])
        error = abs(numerical - analytical)
        if analytical > 1e-10:
            relative_error = error / analytical
        else:
            relative_error = error
        assert relative_error < 1e-5, f"Relative error {relative_error:.6e} > 1e-5"

    def test_local_error_nonzero_nonlinear(self):
        """RK4 local_error_estimate ненулевой при нелинейной динамике."""
        pop = NonlinearOscillatorPopulation(n_units=1, dt=0.1)
        integrator = RK4Integrator()
        state = pop.get_initial_state()

        _, local_error = integrator.step(
            pop, state, {'I_syn': tf.zeros([1, 1]), 'g_syn': tf.zeros([1, 1])}
        )

        assert float(local_error[0]) > 0.0


class TestIntegratorsTFFunction:
    def test_euler_tf_function(self):
        """Euler совместим с tf.function."""
        pop = SimpleDecayPopulation(n_units=1, dt=0.1)
        integrator = EulerIntegrator()

        @tf.function
        def one_step():
            state = pop.get_initial_state()
            new_state, _ = integrator.step(
                pop, state, {'I_syn': tf.zeros([1, 1]), 'g_syn': tf.zeros([1, 1])}
            )
            return new_state

        result = one_step()
        assert len(result) == 1
        assert result[0].shape == (1, 1)

    def test_heun_tf_function(self):
        """Heun совместим с tf.function."""
        pop = SimpleDecayPopulation(n_units=1, dt=0.1)
        integrator = HeunIntegrator()

        @tf.function
        def one_step():
            state = pop.get_initial_state()
            new_state, _ = integrator.step(
                pop, state, {'I_syn': tf.zeros([1, 1]), 'g_syn': tf.zeros([1, 1])}
            )
            return new_state

        result = one_step()
        assert len(result) == 1
        assert result[0].shape == (1, 1)

    def test_rk4_tf_function(self):
        """RK4 совместим с tf.function."""
        pop = SimpleDecayPopulation(n_units=1, dt=0.1)
        integrator = RK4Integrator()

        @tf.function
        def one_step():
            state = pop.get_initial_state()
            new_state, _ = integrator.step(
                pop, state, {'I_syn': tf.zeros([1, 1]), 'g_syn': tf.zeros([1, 1])}
            )
            return new_state

        result = one_step()
        assert len(result) == 1
        assert result[0].shape == (1, 1)
