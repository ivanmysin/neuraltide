"""
Пример 5: LFP Proxy readout.
Использование LFPProxyReadout для взвешенной суммы токов.
"""
import numpy as np
import tensorflow as tf

import neuraltide
from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.populations import IzhikevichMeanField
from neuraltide.synapses import TsodyksMarkramSynapse
from neuraltide.inputs import SinusoidalGenerator
from neuraltide.integrators import RK4Integrator
from neuraltide.training import Trainer, CompositeLoss, MSELoss
from neuraltide.training.readouts import LFPProxyReadout
from neuraltide.utils import seed_everything

seed_everything(42)

dt = 0.5
T = 20

exc_params = {
    'tau_pop':   {'value': [1.0]*4,   'trainable': False},
    'alpha':     {'value': [0.5]*4,   'trainable': False},
    'a':         {'value': [0.02]*4, 'trainable': False},
    'b':         {'value': [0.2]*4,   'trainable': False},
    'w_jump':    {'value': [0.1]*4,   'trainable': False},
    'Delta_I':   {'value': [0.5]*4,  'trainable': True, 'min': 0.01, 'max': 2.0},
    'I_ext':     {'value': [1.0]*4,  'trainable': True},
}

pop = IzhikevichMeanField(dt=dt, params=exc_params, name='exc')

gen = SinusoidalGenerator(amplitude=10.0, freq=8.0, phase=0.0, offset=10.0)

syn = TsodyksMarkramSynapse(n_pre=1, n_post=4, dt=dt, params={
    'gsyn_max': {'value': [[0.1]*4], 'trainable': True},
    'tau_f':    {'value': 20.0,  'trainable': True, 'min': 6.0,  'max': 240.0},
    'tau_d':    {'value': 5.0,   'trainable': True, 'min': 2.0,  'max': 15.0},
    'tau_r':    {'value': 200.0, 'trainable': True, 'min': 91.0, 'max': 1300.0},
    'Uinc':     {'value': 0.2,   'trainable': True, 'min': 0.04, 'max': 0.7},
    'pconn':    {'value': [[1]*4], 'trainable': False},
    'e_r':      {'value': 0.0,   'trainable': False},
})

graph = NetworkGraph(dt=dt)
graph.add_input_population('stim', gen)
graph.add_population('exc', pop)
graph.add_synapse('stim->exc', syn, src='stim', tgt='exc')

network = NetworkRNN(graph, integrator=RK4Integrator())

t_values = np.arange(T, dtype=np.float32) * dt
t_seq = tf.constant(t_values[None, :, None])

target_0 = 10.0 + 5.0 * np.sin(2 * np.pi * 8.0 * t_values / 1000.0)
target = {
    'exc': tf.constant(target_0[None, :, None], dtype=tf.float32)
}

loss_fn = CompositeLoss([
    (1.0, MSELoss(target)),
])

trainer = Trainer(network, loss_fn,
                  optimizer=tf.keras.optimizers.Adam(1e-3))
history = trainer.fit(t_seq, epochs=100, verbose=1)

output = trainer.predict(t_seq)
print(f"LFP output shape: {output.firing_rates['exc'].shape}")
print("LFP Proxy example completed successfully!")
