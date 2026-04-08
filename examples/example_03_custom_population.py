"""
Пример 3: Пользовательская популяционная модель.
Минимальный контракт: derivatives, get_initial_state, get_firing_rate, parameter_spec.
"""
import tensorflow as tf

import neuraltide
from neuraltide.core.base import PopulationModel
from neuraltide.config import register_population
from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.synapses import TsodyksMarkramSynapse
from neuraltide.inputs import ConstantRateGenerator
from neuraltide.integrators import EulerIntegrator
from neuraltide.training import Trainer, CompositeLoss, MSELoss
from neuraltide.utils import seed_everything

seed_everything(42)


class MyRateModel(PopulationModel):
    """tau * dr/dt = -r + tanh(I_ext + I_syn)"""

    def __init__(self, n_units, dt, params, **kwargs):
        super().__init__(n_units=n_units, dt=dt, **kwargs)
        self.tau = self._make_param(params, 'tau')
        self.I_ext = self._make_param(params, 'I_ext')
        self.state_size = [tf.TensorShape([1, n_units])]

    def get_initial_state(self, batch_size=1):
        return [tf.zeros([1, self.n_units],
                         dtype=neuraltide.config.get_dtype())]

    def derivatives(self, state, total_synaptic_input):
        r = state[0]
        I_syn = total_synaptic_input['I_syn']
        I_tot = self.I_ext + I_syn
        drdt = (-r + tf.nn.tanh(I_tot)) / self.tau
        return [drdt]

    def get_firing_rate(self, state):
        return tf.nn.relu(state[0]) * 100.0

    @property
    def parameter_spec(self):
        return {
            'tau': {
                'shape': (self.n_units,),
                'trainable': False,
                'constraint': None,
                'units': 'ms'
            },
            'I_ext': {
                'shape': (self.n_units,),
                'trainable': True,
                'constraint': None,
                'units': 'dimensionless'
            },
        }


register_population('MyRateModel', MyRateModel)

n_units = 2
dt = 0.5
T = 50

pop = MyRateModel(n_units=n_units, dt=dt, params={
    'tau': {'value': [10.0]*n_units, 'trainable': False},
    'I_ext': {'value': [0.5]*n_units, 'trainable': True},
})

gen = ConstantRateGenerator(
    dt=dt,
    params={'rate': 20.0}
)

syn = TsodyksMarkramSynapse(n_pre=1, n_post=n_units, dt=dt, params={
    'gsyn_max': {'value': [[0.1]*n_units], 'trainable': True},
    'tau_f':    {'value': 20.0,  'trainable': True, 'min': 6.0,  'max': 240.0},
    'tau_d':    {'value': 5.0,   'trainable': True, 'min': 2.0,  'max': 15.0},
    'tau_r':    {'value': 200.0, 'trainable': True, 'min': 91.0, 'max': 1300.0},
    'Uinc':     {'value': 0.2,   'trainable': True, 'min': 0.04, 'max': 0.7},
    'pconn':    {'value': [[1]*n_units], 'trainable': False},
    'e_r':      {'value': 0.0,   'trainable': False},
})

graph = NetworkGraph(dt=dt)
graph.add_input_population('input', gen)
graph.add_population('rate', pop)
graph.add_synapse('input->rate', syn, src='input', tgt='rate')

network = NetworkRNN(graph, integrator=EulerIntegrator())

t_values = tf.range(T, dtype=tf.float32) * dt
t_seq = tf.constant(t_values[None, :, None])

target_data = 10.0 + 5.0 * tf.sin(2 * 3.14159 * 2.0 * t_values / 1000.0)
target = {
    'rate': tf.constant(
        tf.stack([target_data, target_data * 0.8], axis=-1)[None, :, :],
        dtype=tf.float32
    )
}

loss_fn = CompositeLoss([
    (1.0, MSELoss(target)),
])

trainer = Trainer(network, loss_fn,
                  optimizer=tf.keras.optimizers.Adam(1e-3))
history = trainer.fit(t_seq, epochs=100, verbose=1)

print("Custom population model trained successfully!")
