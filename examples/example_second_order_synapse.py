"""
Пример: Синапс второго порядка (SecondOrderSynapse).

Входной генератор (тета-ритм ~8Hz) подаётся на синапс второго порядка.
Демонстрируется:
- g_s (проводимость): нарастает и спадает с динамикой второго порядка
- dg_s (производная): скорость изменения проводимости
- I_syn (синаптический ток): фильтрованный отклик на входную частоту

Уравнение:
    tau1 * tau2 * d²g_s/dt² + (tau1 + tau2) * dg_s/dt + g_s = nu_pre
    I_syn = gsyn_max * g_s * (e_r - V)
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.populations.izhikevich_mf import IzhikevichMeanField
from neuraltide.synapses.second_order import SecondOrderSynapse
from neuraltide.inputs.von_mises import VonMisesGenerator
from neuraltide.integrators import RK4Integrator
from neuraltide.utils import seed_everything

seed_everything(42)

dt = 0.1
T = 200

gen = VonMisesGenerator(
    dt=dt,
    params={
        'mean_rate': 10.0,
        'R': 0.8,
        'freq': 8.0,
        'phase': 0.0,
    },
    name='theta_gen'
)

syn = SecondOrderSynapse(n_pre=1, n_post=2, dt=dt, params={
    'gsyn_max': {'value': [[5.0, 2.0]], 'trainable': False},
    'tau1':     {'value': 3.0,   'trainable': False},
    'tau2':     {'value': 10.0,  'trainable': False},
    'pconn':    {'value': [[1.0, 1.0]], 'trainable': False},
    'e_r':      {'value': 1.0,   'trainable': False},
})

pop = IzhikevichMeanField(dt=dt, params={
    'tau_pop':   {'value': [1.0, 1.0],   'trainable': False},
    'alpha':     {'value': [0.5, 0.5],   'trainable': False},
    'a':         {'value': [0.02, 0.02], 'trainable': False},
    'b':         {'value': [0.2, 0.2],   'trainable': False},
    'w_jump':    {'value': [0.1, 0.1],   'trainable': False},
    'Delta_I':   {'value': [0.005, 0.005],   'trainable': False},
    'I_ext':     {'value': [0.0, 0.0],   'trainable': False},
})

graph = NetworkGraph(dt=dt)
graph.declare_input('theta', n_units=gen.n_units)
graph.add_population('main', pop)
graph.add_synapse('theta->main', syn, src='theta', tgt='main')

network = NetworkRNN(graph, integrator=RK4Integrator())

t_values = np.arange(0, T, dt, dtype=np.float32)
t_seq = tf.constant(t_values[None, :, None], dtype=tf.float32)
inputs = graph.pack_inputs({'theta': gen(t_seq)})

output = network(t_seq, inputs=inputs, training=False)

g_s_hist = []
dg_s_hist = []
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

    for name in graph.input_names:
        pop_states_dict[name] = [t]

    tgt_pop = graph._populations['main']

    tgt_obs = tgt_pop.observables(pop_states_dict['main'])
    post_v = tgt_obs.get('v_mean', tf.zeros([1, tgt_pop.n_units], dtype=tf.float32))

    syn_entry = graph._synapses['theta->main']
    pre_rate = gen(t)
    syn_state = syn_states_dict['theta->main']

    current_dict, new_syn_state = syn_entry.model.forward(
        pre_rate, post_v, syn_state, syn_entry.model.dt
    )

    g_s_hist.append(syn_state[0][0].numpy())
    dg_s_hist.append(syn_state[1][0].numpy())
    I_syn_hist.append(current_dict['I_syn'][0].numpy())
    g_syn_hist.append(current_dict['g_syn'][0].numpy())

    new_pop, new_syn, _ = _step_fn(
        (tuple(pop_states), tuple(syn_states), tf.zeros([1], dtype=tf.float32)),
        t, {'theta': pre_rate}, graph, network._integrator
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

g_s_hist = np.array(g_s_hist)
dg_s_hist = np.array(dg_s_hist)
I_syn_hist = np.array(I_syn_hist)
g_syn_hist = np.array(g_syn_hist)

rates = output.firing_rates['main'].numpy()[0]

fig, axes = plt.subplots(4, 1, figsize=(12, 10))

axes[0].plot(t_values, g_s_hist[:, 0], color='tab:blue', linewidth=1.5, label='Unit 0')
axes[0].plot(t_values, g_s_hist[:, 1], color='tab:orange', linewidth=1.5, linestyle='--', label='Unit 1')
axes[0].set_ylabel('g_s')
axes[0].set_title('Second-Order Synapse Conductance g_s')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

axes[1].plot(t_values, dg_s_hist[:, 0], color='tab:blue', linewidth=1.5, label='Unit 0')
axes[1].plot(t_values, dg_s_hist[:, 1], color='tab:orange', linewidth=1.5, linestyle='--', label='Unit 1')
axes[1].set_ylabel('dg_s/dt')
axes[1].set_title('Conductance Derivative')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

axes[2].plot(t_values, I_syn_hist[:, 0], color='tab:red', linewidth=1.5, label='Unit 0')
axes[2].plot(t_values, I_syn_hist[:, 1], color='tab:purple', linewidth=1.5, linestyle='--', label='Unit 1')
axes[2].set_ylabel('I_syn')
axes[2].set_title('Synaptic Current I_syn')
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
    "SecondOrderSynapse: Second-Order Dynamics\n"
    "Input: Von Mises (theta ~8Hz, R=0.8)\n"
    r"$\tau_1\tau_2 \ddot{g}_s + (\tau_1+\tau_2) \dot{g}_s + g_s = \nu_{pre}$",
    fontsize=13
)
plt.tight_layout()
plt.savefig("example_second_order_synapse.png", dpi=150)
plt.show()
print("Figure saved as example_second_order_synapse.png")
