"""
Пример 1: Одна популяция IzhikevichMeanField (n_units=2),
тета-ритмический вход через обучаемый синапс.
Цель: воспроизвести заданную firing rate траекторию.
"""
import numpy as np
import tensorflow as tf

import neuraltide
from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.populations import IzhikevichMeanField
from neuraltide.synapses import TsodyksMarkramSynapse
from neuraltide.inputs import VonMisesGenerator
from neuraltide.integrators import RK4Integrator
from neuraltide.training import Trainer, CompositeLoss, MSELoss, StabilityPenalty
from neuraltide.utils import seed_everything, print_summary

seed_everything(42)

dt = 0.5
T = 2000

pop = IzhikevichMeanField(n_units=2, dt=dt, params={
    'alpha':     {'value': [0.5, 0.5],   'trainable': False},
    'a':         {'value': [0.02, 0.02], 'trainable': False},
    'b':         {'value': [0.2, 0.2],   'trainable': False},
    'w_jump':    {'value': [0.1, 0.1],   'trainable': False},
    'dt_nondim': {'value': [0.01, 0.01], 'trainable': False},
    'Delta_eta': {'value': [0.5, 0.6],   'trainable': True,
                  'min': 0.01, 'max': 2.0},
    'I_ext':     {'value': [1.0, 1.2],   'trainable': True},
})

gen = VonMisesGenerator(params=[
    {'MeanFiringRate': 20.0, 'R': 0.5, 'ThetaFreq': 8.0, 'ThetaPhase': 0.0},
], name='theta_gen')

syn_in = TsodyksMarkramSynapse(n_pre=1, n_post=2, dt=dt, params={
    'gsyn_max': {'value': [[0.1, 0.1]], 'trainable': True},
    'tau_f':    {'value': 20.0,  'trainable': True, 'min': 6.0,  'max': 240.0},
    'tau_d':    {'value': 5.0,   'trainable': True, 'min': 2.0,  'max': 15.0},
    'tau_r':    {'value': 200.0, 'trainable': True, 'min': 91.0, 'max': 1300.0},
    'Uinc':     {'value': 0.2,   'trainable': True, 'min': 0.04, 'max': 0.7},
    'pconn':    {'value': [[1.0, 1.0]], 'trainable': False},
    'e_r':      {'value': 0.0,   'trainable': False},
})

syn_rec = TsodyksMarkramSynapse(n_pre=2, n_post=2, dt=dt, params={
    'gsyn_max': {'value': 0.05,  'trainable': True},
    'tau_f':    {'value': 20.0,  'trainable': True, 'min': 6.0,  'max': 240.0},
    'tau_d':    {'value': 5.0,   'trainable': True, 'min': 2.0,  'max': 15.0},
    'tau_r':    {'value': 200.0, 'trainable': True, 'min': 91.0, 'max': 1300.0},
    'Uinc':     {'value': 0.2,   'trainable': True, 'min': 0.04, 'max': 0.7},
    'pconn':    {'value': [[1, 1], [1, 1]], 'trainable': False},
    'e_r':      {'value': 0.0,   'trainable': False},
})

graph = NetworkGraph(dt=dt)
graph.add_input_population('theta', gen)
graph.add_population('exc', pop)
graph.add_synapse('theta->exc', syn_in,  src='theta', tgt='exc')
graph.add_synapse('exc->exc',   syn_rec, src='exc',   tgt='exc')

network = NetworkRNN(graph, integrator=RK4Integrator(),
                     stability_penalty_weight=1e-3)
print_summary(network)

t_values = np.arange(T, dtype=np.float32) * dt
t_seq = tf.constant(t_values[None, :, None])

target_0 = 10.0 + 5.0*np.sin(2*np.pi*8.0*t_values/1000.0)
target_1 = 8.0  + 4.0*np.sin(2*np.pi*8.0*t_values/1000.0 + 0.5)
target = {
    'exc': tf.constant(
        np.stack([target_0, target_1], axis=-1)[None, :, :],
        dtype=tf.float32
    )
}

loss_fn = CompositeLoss([
    (1.0,  MSELoss(target)),
    (1e-3, StabilityPenalty()),
])
trainer = Trainer(network, loss_fn,
                  optimizer=tf.keras.optimizers.Adam(1e-3))
history = trainer.fit(t_seq, epochs=200, verbose=1)
