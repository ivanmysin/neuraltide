"""
Пример 2: exc (IzhikevichMF, n_units=4) + inh (IzhikevichMF, n_units=2).
Синапсы: AMPA+NMDA (exc→exc), AMPA (exc→inh), GABA_A (inh→exc).
Вход: тета-генератор через TsodyksMarkramSynapse.
Частичные наблюдения: target только для exc.
BandpassReadout: loss считается только в тета-диапазоне.
"""
import numpy as np
import tensorflow as tf

import neuraltide
from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.populations import IzhikevichMeanField
from neuraltide.synapses import TsodyksMarkramSynapse, NMDASynapse
from neuraltide.inputs import VonMisesGenerator
from neuraltide.integrators import RK4Integrator
from neuraltide.training import Trainer, CompositeLoss, MSELoss, StabilityPenalty
from neuraltide.training.readouts import BandpassReadout
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

inh_params = {
    'tau_pop':   {'value': [1.0]*2,   'trainable': False},
    'alpha':     {'value': [0.5]*2,   'trainable': False},
    'a':         {'value': [0.02]*2, 'trainable': False},
    'b':         {'value': [0.2]*2,   'trainable': False},
    'w_jump':    {'value': [0.1]*2,   'trainable': False},
    'Delta_I':   {'value': [0.6]*2,  'trainable': True, 'min': 0.01, 'max': 2.0},
    'I_ext':     {'value': [0.8]*2,  'trainable': True},
}

exc = IzhikevichMeanField(dt=dt, params=exc_params, name='exc')
inh = IzhikevichMeanField(dt=dt, params=inh_params, name='inh')

gen = VonMisesGenerator(
    dt=dt,
    params={
        'mean_rate': 20.0,
        'R': 0.5,
        'freq': 8.0,
        'phase': 0.0,
    },
    name='theta_gen'
)

ampa_ee = TsodyksMarkramSynapse(n_pre=4, n_post=4, dt=dt, params={
    'gsyn_max': {'value': 0.05,  'trainable': True},
    'tau_f':    {'value': 20.0,  'trainable': True, 'min': 6.0,  'max': 240.0},
    'tau_d':    {'value': 5.0,   'trainable': True, 'min': 2.0,  'max': 15.0},
    'tau_r':    {'value': 200.0, 'trainable': True, 'min': 91.0, 'max': 1300.0},
    'Uinc':     {'value': 0.2,   'trainable': True, 'min': 0.04, 'max': 0.7},
    'pconn':    {'value': [[1]*4]*4, 'trainable': False},
    'e_r':      {'value': 0.0,   'trainable': False},
})

nmda_ee = NMDASynapse(n_pre=4, n_post=4, dt=dt, params={
    'gsyn_max_nmda': {'value': 0.05,  'trainable': True},
    'tau1_nmda':     {'value': 2.0,   'trainable': False},
    'tau2_nmda':     {'value': 100.0, 'trainable': False},
    'Mgb':           {'value': 1.0,   'trainable': False},
    'av_nmda':       {'value': 0.062, 'trainable': False},
    'pconn_nmda':    {'value': [[1]*4]*4, 'trainable': False},
    'e_r_nmda':      {'value': 0.0,   'trainable': False},
    'v_ref':         {'value': 1.0,   'trainable': False},
})

exc_to_inh = TsodyksMarkramSynapse(n_pre=4, n_post=2, dt=dt, params={
    'gsyn_max': {'value': 0.08,  'trainable': True},
    'tau_f':    {'value': 20.0,  'trainable': True, 'min': 6.0,  'max': 240.0},
    'tau_d':    {'value': 5.0,   'trainable': True, 'min': 2.0,  'max': 15.0},
    'tau_r':    {'value': 200.0, 'trainable': True, 'min': 91.0, 'max': 1300.0},
    'Uinc':     {'value': 0.2,   'trainable': True, 'min': 0.04, 'max': 0.7},
    'pconn':    {'value': [[1]*2]*4, 'trainable': False},
    'e_r':      {'value': 0.0,   'trainable': False},
})

inh_to_exc = NMDASynapse(n_pre=2, n_post=4, dt=dt, params={
    'gsyn_max_nmda': {'value': 0.1,  'trainable': True},
    'tau1_nmda':     {'value': 2.0,   'trainable': False},
    'tau2_nmda':     {'value': 100.0, 'trainable': False},
    'Mgb':           {'value': 1.0,   'trainable': False},
    'av_nmda':       {'value': 0.062, 'trainable': False},
    'pconn_nmda':    {'value': [[1]*4]*2, 'trainable': False},
    'e_r_nmda':      {'value': -70.0,   'trainable': False},
    'v_ref':         {'value': 1.0,   'trainable': False},
})

theta_syn = TsodyksMarkramSynapse(n_pre=1, n_post=4, dt=dt, params={
    'gsyn_max': {'value': [[0.1]*4], 'trainable': True},
    'tau_f':    {'value': 20.0,  'trainable': True, 'min': 6.0,  'max': 240.0},
    'tau_d':    {'value': 5.0,   'trainable': True, 'min': 2.0,  'max': 15.0},
    'tau_r':    {'value': 200.0, 'trainable': True, 'min': 91.0, 'max': 1300.0},
    'Uinc':     {'value': 0.2,   'trainable': True, 'min': 0.04, 'max': 0.7},
    'pconn':    {'value': [[1]*4], 'trainable': False},
    'e_r':      {'value': 0.0,   'trainable': False},
})

graph = NetworkGraph(dt=dt)
graph.add_input_population('theta', gen)
graph.add_population('exc', exc)
graph.add_population('inh', inh)
graph.add_synapse('theta->exc', theta_syn, src='theta', tgt='exc')
graph.add_synapse('ampa_ee', ampa_ee, src='exc', tgt='exc')
graph.add_synapse('nmda_ee', nmda_ee, src='exc', tgt='exc')
graph.add_synapse('exc->inh', exc_to_inh, src='exc', tgt='inh')
graph.add_synapse('inh->exc', inh_to_exc, src='inh', tgt='exc')

network = NetworkRNN(graph, integrator=RK4Integrator(),
                     stability_penalty_weight=1e-3)

t_values = np.arange(T, dtype=np.float32) * dt
t_seq = tf.constant(t_values[None, :, None])

target_exc = 10.0 + 5.0*np.sin(2*np.pi*8.0*t_values/1000.0)
target_exc = tf.constant(target_exc[None, :, None], dtype=tf.float32)

theta_readout = BandpassReadout(f_low=4.0, f_high=12.0, dt=dt)
loss_fn = CompositeLoss([
    (1.0,  MSELoss({'exc': target_exc}, readout=theta_readout)),
    (1e-3, StabilityPenalty()),
])
trainer = Trainer(network, loss_fn,
                  optimizer=tf.keras.optimizers.Adam(1e-3))
history = trainer.fit(t_seq, epochs=200, verbose=1)
