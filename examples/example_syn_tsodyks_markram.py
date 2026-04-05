"""
Пример: Динамика синапса TsodyksMarkramSynapse с входом Von Mises.

Von Mises генератор (тета-ритм ~8Hz) подаётся на синапс.
Демонстрируется:
- R (фракция доступных везикул): восстанавливается после высвобождения
- U (фракция использованных везикул): растёт при каждом спайке
- A (аккумулятор): растёт при высвобождении, затухает экспоненциально
- I_syn и g_syn как функции от U, R, A
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.populations.izhikevich_mf import IzhikevichMeanField
from neuraltide.synapses.tsodyks_markram import TsodyksMarkramSynapse
from neuraltide.inputs.von_mises import VonMisesGenerator
from neuraltide.integrators import RK4Integrator
from neuraltide.utils import seed_everything

seed_everything(42)

dt = 0.5
T = 200

gen = VonMisesGenerator(params=[
    {'MeanFiringRate': 30.0, 'R': 0.8, 'ThetaFreq': 8.0, 'ThetaPhase': 0.0},
], name='theta_gen')

syn = TsodyksMarkramSynapse(n_pre=1, n_post=2, dt=dt, params={
    'gsyn_max': {'value': [[0.15, 0.12]], 'trainable': False},
    'tau_f':    {'value': 30.0,  'trainable': False},
    'tau_d':    {'value': 8.0,   'trainable': False},
    'tau_r':    {'value': 300.0, 'trainable': False},
    'Uinc':     {'value': 0.3,   'trainable': False},
    'pconn':    {'value': [[1.0, 1.0]], 'trainable': False},
    'e_r':      {'value': 0.0,   'trainable': False},
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

R_hist = []
U_hist = []
A_hist = []
I_syn_hist = []
g_syn_hist = []

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

    r = pop_states_dict['main'][0]
    v = pop_states_dict['main'][1]
    w = pop_states_dict['main'][2]

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

    R_hist.append(syn_state[0][0].numpy())
    U_hist.append(syn_state[1][0].numpy())
    A_hist.append(syn_state[2][0].numpy())
    I_syn_hist.append(current_dict['I_syn'][0].numpy())
    g_syn_hist.append(current_dict['g_syn'][0].numpy())

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

R_hist = np.array(R_hist)
U_hist = np.array(U_hist)
A_hist = np.array(A_hist)
I_syn_hist = np.array(I_syn_hist)
g_syn_hist = np.array(g_syn_hist)

rates = output.firing_rates['main'].numpy()[0]

fig, axes = plt.subplots(5, 1, figsize=(12, 12))

axes[0].plot(t_values, R_hist[:, 0], color='tab:blue', linewidth=1.5, label='R (vesicle fraction)')
axes[0].set_ylabel('R')
axes[0].set_title('R: Fraction of Available Vesicles')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

axes[1].plot(t_values, U_hist[:, 0], color='tab:orange', linewidth=1.5, label='U (utilization)')
axes[1].set_ylabel('U')
axes[1].set_title('U: Fraction of Used Vesicles')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

axes[2].plot(t_values, A_hist[:, 0], color='tab:green', linewidth=1.5, label='A (accumulator)')
axes[2].set_ylabel('A')
axes[2].set_title('A: Release Accumulator')
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
    "TsodyksMarkramSynapse: Short-Term Plasticity Dynamics\n"
    "Input: Von Mises (theta ~8Hz, R=0.8)",
    fontsize=13
)
plt.tight_layout()
plt.savefig("example_syn_tsodyks_markram.png", dpi=150)
plt.show()
print("Figure saved as example_syn_tsodyks_markram.png")
