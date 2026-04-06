"""
Пример: Динамика StaticSynapse с входом Von Mises.

Von Mises генератор (тета-ритм ~8Hz) подаётся на статический синапс.
StaticSynapse не имеет внутренней динамики (состояние пустое).

Демонстрируется:
- I_syn = gsyn_max * pconn * FR * (e_r - v)
- Проводимость g_syn пропорциональна входной частоте
- Мгновенный отклик на изменение входной частоты
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.populations.izhikevich_mf import IzhikevichMeanField
from neuraltide.synapses.static import StaticSynapse
from neuraltide.inputs.von_mises import VonMisesGenerator
from neuraltide.integrators import RK4Integrator
from neuraltide.utils import seed_everything

seed_everything(42)

dt = 0.5
T = 200

gen = VonMisesGenerator(params=[
    {'MeanFiringRate': 40.0, 'R': 0.9, 'ThetaFreq': 8.0, 'ThetaPhase': 0.0},
], name='theta_gen')

syn = StaticSynapse(n_pre=1, n_post=2, dt=dt, params={
    'gsyn_max': {'value': [[0.2, 0.15]], 'trainable': False},
    'pconn':    {'value': [[1.0, 1.0]], 'trainable': False},
    'e_r':      {'value': 0.0,          'trainable': False},
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

I_syn_hist = []
g_syn_hist = []
pre_rate_hist = []

init_syn = list(network._init_syn_states)
syn_states = list(init_syn)

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

    pre_rate = src_pop.get_firing_rate(pop_states_dict['theta'])
    pre_rate_hist.append(pre_rate[0].numpy())

    syn_entry = graph._synapses['theta->main']
    current_dict, new_syn_state = syn_entry.model.forward(
        pre_rate, post_v, [], syn_entry.model.dt
    )

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

I_syn_hist = np.array(I_syn_hist)
g_syn_hist = np.array(g_syn_hist)
pre_rate_hist = np.array(pre_rate_hist)
rates = output.firing_rates['main'].numpy()[0]

fig, axes = plt.subplots(4, 1, figsize=(12, 10))

axes[0].plot(t_values, pre_rate_hist[:, 0], color='tab:blue', linewidth=1.5)
axes[0].set_ylabel('FRpre (Hz)')
axes[0].set_title('Presynaptic Firing Rate (Von Mises input)')
axes[0].grid(True, alpha=0.3)

axes[1].plot(t_values, g_syn_hist[:, 0], color='tab:blue', linewidth=1.5, label='Unit 0')
axes[1].plot(t_values, g_syn_hist[:, 1], color='tab:orange', linewidth=1.5, linestyle='--', label='Unit 1')
axes[1].set_ylabel('g_syn')
axes[1].set_title('Synaptic Conductance g_syn = gsyn_max * FR * pconn')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

axes[2].plot(t_values, I_syn_hist[:, 0], color='tab:blue', linewidth=1.5, label='Unit 0')
axes[2].plot(t_values, I_syn_hist[:, 1], color='tab:orange', linewidth=1.5, linestyle='--', label='Unit 1')
axes[2].set_ylabel('I_syn')
axes[2].set_title('Synaptic Current I_syn = g_syn * (e_r - v)')
axes[2].legend(fontsize=9)
axes[2].grid(True, alpha=0.3)

axes[3].plot(t_values, rates[:, 0], color='tab:blue', linewidth=1.5, label='Unit 0')
axes[3].plot(t_values, rates[:, 1], color='tab:orange', linewidth=1.5, linestyle='--', label='Unit 1')
axes[3].set_ylabel('Firing Rate (Hz)')
axes[3].set_xlabel('Time (ms)')
axes[3].set_title('Population Firing Rate')
axes[3].legend(fontsize=9)
axes[3].grid(True, alpha=0.3)

fig.suptitle(
    "StaticSynapse: Instantaneous Response to Von Mises Input\n"
    "No internal state — current directly proportional to presynaptic rate",
    fontsize=13
)
plt.tight_layout()
plt.savefig("example_syn_static.png", dpi=150)
plt.show()
print("Figure saved as example_syn_static.png")
