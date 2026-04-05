"""
Пример: Динамика NMDASynapse с входом Von Mises.

Von Mises генератор (тета-ритм ~8Hz) подаётся на NMDA синапс.
Демонстрируется:
- gnmda (проводимость NMDA): нарастает и спадает с двумя экспонентами
- Mg block (магниевый блок): зависит от постсинаптического потенциала
- I_syn (синаптический ток): фильтрованный отклик на входную частоту
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.populations.izhikevich_mf import IzhikevichMeanField
from neuraltide.synapses.nmda import NMDASynapse
from neuraltide.inputs.von_mises import VonMisesGenerator
from neuraltide.integrators import RK4Integrator
from neuraltide.utils import seed_everything

seed_everything(42)

dt = 0.5
T = 200

gen = VonMisesGenerator(params=[
    {'MeanFiringRate': 30.0, 'R': 0.8, 'ThetaFreq': 8.0, 'ThetaPhase': 0.0},
], name='theta_gen')

syn = NMDASynapse(n_pre=1, n_post=2, dt=dt, params={
    'gsyn_max_nmda': {'value': [[0.1, 0.08]], 'trainable': False},
    'tau1_nmda':     {'value': 5.0,   'trainable': False},
    'tau2_nmda':     {'value': 80.0,  'trainable': False},
    'Mgb':           {'value': 0.1,   'trainable': False},
    'av_nmda':       {'value': 0.18,  'trainable': False},
    'pconn_nmda':    {'value': [[1.0, 1.0]], 'trainable': False},
    'e_r_nmda':      {'value': 0.0,   'trainable': False},
    'v_ref':         {'value': 0.0,   'trainable': False},
})

pop = IzhikevichMeanField(n_units=2, dt=dt, params={
    'alpha':     {'value': [0.5, 0.5],   'trainable': False},
    'a':         {'value': [0.02, 0.02], 'trainable': False},
    'b':         {'value': [0.2, 0.2],   'trainable': False},
    'w_jump':    {'value': [0.1, 0.1],   'trainable': False},
    'dt_nondim': {'value': [0.01, 0.01], 'trainable': False},
    'Delta_eta': {'value': [0.5, 0.5],   'trainable': False},
    'I_ext':     {'value': [1.5, 1.5],   'trainable': False},
})

graph = NetworkGraph(dt=dt)
graph.add_input_population('theta', gen)
graph.add_population('main', pop)
graph.add_synapse('theta->main', syn, src='theta', tgt='main')

network = NetworkRNN(graph, integrator=RK4Integrator())

t_values = np.arange(0, T, dt, dtype=np.float32)
t_seq = tf.constant(t_values[None, :, None], dtype=tf.float32)

output = network(t_seq, training=False)

gnmda_hist = []
dgnmda_hist = []
I_syn_hist = []
g_syn_hist = []
mg_block_hist = []

init_syn = list(network._init_syn_states)
syn_states = list(init_syn)
syn_states_dict = {}
idx = 0
for name in graph.synapse_names:
    entry = graph._synapses[name]
    n = len(entry.model.state_size)
    syn_states_dict[name] = syn_states[idx:idx + n]
    idx += n

pop_states_dict = {}
pop_states = list(network._init_pop_states)
idx = 0
for name in graph.population_names:
    p = graph._populations[name]
    n = len(p.state_size)
    pop_states_dict[name] = pop_states[idx:idx + n]
    idx += n

from neuraltide.core.network import _step_fn

for step in range(len(t_values)):
    t = t_seq[:, step:step+1, 0]

    for name in graph.input_population_names:
        pop_states_dict[name] = [t]

    tgt_pop = graph._populations['main']
    src_pop = graph._populations['theta']

    tgt_obs = tgt_pop.observables(pop_states_dict['main'])
    post_v = tgt_obs.get('v_mean', tf.zeros([1, tgt_pop.n_units], dtype=tf.float32))

    syn_entry = graph._synapses['theta->main']
    pre_rate = src_pop.get_firing_rate(pop_states_dict['theta'])
    syn_state = syn_states_dict['theta->main']

    current_dict, new_syn_state = syn_entry.model.forward(
        pre_rate, post_v, syn_state, syn_entry.model.dt
    )

    gnmda_hist.append(syn_state[0][0].numpy())
    dgnmda_hist.append(syn_state[1][0].numpy())
    I_syn_hist.append(current_dict['I_syn'][0].numpy())
    g_syn_hist.append(current_dict['g_syn'][0].numpy())

    gsyn_max_nmda = syn_entry.model.gsyn_max_nmda.numpy()
    Mgb_val = syn_entry.model.Mgb.numpy()[0, 0]
    av_nmda_val = syn_entry.model.av_nmda.numpy()[0, 0]
    v_ref_val = syn_entry.model.v_ref.numpy()[0, 0]

    post_v_flat = post_v[0].numpy()
    mg_block = 1.0 / (1.0 + Mgb_val * np.exp(-av_nmda_val * (post_v_flat - v_ref_val)))
    mg_block_hist.append(mg_block)

    new_pop, new_syn, _ = _step_fn(
        (tuple(pop_states), tuple(syn_states), tf.zeros([1], dtype=tf.float32)),
        t, graph, network._integrator
    )

    pop_states = list(new_pop)
    syn_states = list(new_syn)

    idx = 0
    for name in graph.population_names:
        p = graph._populations[name]
        n = len(p.state_size)
        pop_states_dict[name] = pop_states[idx:idx + n]
        idx += n
    idx = 0
    for name in graph.synapse_names:
        entry = graph._synapses[name]
        n = len(entry.model.state_size)
        syn_states_dict[name] = syn_states[idx:idx + n]
        idx += n

gnmda_hist = np.array(gnmda_hist)
dgnmda_hist = np.array(dgnmda_hist)
I_syn_hist = np.array(I_syn_hist)
g_syn_hist = np.array(g_syn_hist)
mg_block_hist = np.array(mg_block_hist)

rates = output.firing_rates['main'].numpy()[0]

fig, axes = plt.subplots(5, 1, figsize=(12, 12))

axes[0].plot(t_values, gnmda_hist[:, 0], color='tab:blue', linewidth=1.5, label='Unit 0')
axes[0].plot(t_values, gnmda_hist[:, 1], color='tab:orange', linewidth=1.5, linestyle='--', label='Unit 1')
axes[0].set_ylabel('gnmda')
axes[0].set_title('NMDA Conductance gnmda')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

axes[1].plot(t_values, dgnmda_hist[:, 0], color='tab:blue', linewidth=1.5, label='Unit 0')
axes[1].plot(t_values, dgnmda_hist[:, 1], color='tab:orange', linewidth=1.5, linestyle='--', label='Unit 1')
axes[1].set_ylabel('dgnmda')
axes[1].set_title('NMDA Conductance Derivative dgnmda')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

axes[2].plot(t_values, mg_block_hist[:, 0], color='tab:purple', linewidth=1.5, label='Unit 0')
axes[2].plot(t_values, mg_block_hist[:, 1], color='tab:brown', linewidth=1.5, linestyle='--', label='Unit 1')
axes[2].set_ylabel('Mg block')
axes[2].set_title('Magnesium Block (voltage-dependent)')
axes[2].legend(fontsize=9)
axes[2].grid(True, alpha=0.3)

axes[3].plot(t_values, I_syn_hist[:, 0], color='tab:red', linewidth=1.5, label='Unit 0')
axes[3].plot(t_values, I_syn_hist[:, 1], color='tab:purple', linewidth=1.5, linestyle='--', label='Unit 1')
axes[3].set_ylabel('I_syn')
axes[3].set_title('Synaptic Current I_syn')
axes[3].legend(fontsize=9)
axes[3].grid(True, alpha=0.3)

axes[4].plot(t_values, rates[:, 0], color='tab:blue', linewidth=1.5, label='Unit 0')
axes[4].plot(t_values, rates[:, 1], color='tab:orange', linewidth=1.5, linestyle='--', label='Unit 1')
axes[4].set_ylabel('Firing Rate (Hz)')
axes[4].set_xlabel('Time (ms)')
axes[4].set_title('Population Firing Rate')
axes[4].legend(fontsize=9)
axes[4].grid(True, alpha=0.3)

fig.suptitle(
    "NMDASynapse: Dynamics with Magnesium Block\n"
    "Input: Von Mises (theta ~8Hz, R=0.8)",
    fontsize=13
)
plt.tight_layout()
plt.savefig("example_syn_nmda.png", dpi=150)
plt.show()
print("Figure saved as example_syn_nmda.png")
