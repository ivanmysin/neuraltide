"""
Пример 4: FokkerPlanckPopulation.
Пользователь наследуется и реализует derivatives().
"""
import tensorflow as tf
import numpy as np

import neuraltide
from neuraltide.core.base import PopulationModel
from neuraltide.config import register_population
from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.integrators import EulerIntegrator
from neuraltide.training import Trainer, CompositeLoss, MSELoss
from neuraltide.utils import seed_everything

seed_everything(42)


class SimpleFokkerPlanck(PopulationModel):
    """
    Простая Fokker-Planck модель с Ornstein-Uhlenbeck процессом.

    dP/dt = D * d²P/dV² - mu * dP/dV
    """

    def __init__(self, n_units, dt, grid_size=100, D=1.0, mu=0.5, **kwargs):
        super().__init__(n_units=n_units, dt=dt, **kwargs)
        self.grid_size = grid_size
        self.D = D
        self.mu = mu
        self.v_min = -5.0
        self.v_max = 5.0
        self.dV = (self.v_max - self.v_min) / (grid_size - 1)
        self.state_size = [tf.TensorShape([1, grid_size])]

        v_grid = tf.linspace(self.v_min, self.v_max, grid_size)
        self.v_grid = tf.reshape(v_grid, [1, grid_size])

    def get_initial_state(self, batch_size=1):
        dtype = neuraltide.config.get_dtype()
        v = self.v_grid
        mean = tf.constant([[0.0]], dtype=dtype)
        std = tf.constant([[1.0]], dtype=dtype)
        P = tf.exp(-((v - mean) ** 2) / (2 * std ** 2))
        P = P / tf.reduce_sum(P, axis=-1, keepdims=True)
        return [P]

    def derivatives(self, state, total_synaptic_input):
        P = state[0]
        D = tf.constant(self.D, dtype=neuraltide.config.get_dtype())
        mu = tf.constant(self.mu, dtype=neuraltide.config.get_dtype())
        dV = tf.constant(self.dV, dtype=neuraltide.config.get_dtype())

        dP_dV = tf.gradient(P, self.v_grid)[0]
        d2P_dV2 = tf.gradient(dP_dV, self.v_grid)[0]

        diffusion = D * d2P_dV2
        drift = -mu * dP_dV

        dPdt = diffusion + drift
        return [dPdt]

    def get_firing_rate(self, state):
        P = state[0]
        threshold = tf.constant([self.v_max - self.dV], dtype=neuraltide.config.get_dtype())
        idx = tf.constant(self.grid_size - 1)
        J_out = -self.D * (P[:, :, idx:idx+1] - P[:, :, idx-1:idx]) / self.dV
        return J_out

    @property
    def parameter_spec(self):
        return {
            'D': {
                'shape': (),
                'trainable': False,
                'constraint': None,
                'units': 'mV^2/ms'
            },
            'mu': {
                'shape': (),
                'trainable': False,
                'constraint': None,
                'units': 'mV/ms'
            },
        }


register_population('SimpleFokkerPlanck', SimpleFokkerPlanck)

dt = 0.5
T = 500

pop = SimpleFokkerPlanck(n_units=1, dt=dt, grid_size=100)

graph = NetworkGraph(dt=dt)
graph.add_population('fp', pop)

network = NetworkRNN(graph, integrator=EulerIntegrator())

t_values = np.arange(T, dtype=np.float32) * dt
t_seq = tf.constant(t_values[None, :, None])

target_rate = 5.0 + 2.0 * np.sin(2 * np.pi * 5.0 * t_values / 1000.0)
target = {
    'fp': tf.constant(target_rate[None, :, None], dtype=tf.float32)
}

loss_fn = CompositeLoss([
    (1.0, MSELoss(target)),
])

trainer = Trainer(network, loss_fn,
                  optimizer=tf.keras.optimizers.Adam(1e-3))
history = trainer.fit(t_seq, epochs=50, verbose=1)

print("Fokker-Planck population trained successfully!")
