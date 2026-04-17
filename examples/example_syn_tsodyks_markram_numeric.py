"""
Пример: Динамика синапса TsodyksMarkramSynapse с численным интегрированием.

Синапсы теперь интегрируются численно через интегратор (RK4), 
аналогично популяциям. Это демонстрирует:
- R (фракция доступных везикул): восстанавлив��ется после высвобождения
- U (фракция использованных везикул): растёт при каждом спайке
- A (аккумулятор): растёт при высвобождении, затухает экспоненциально
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
from neuraltide.core.network import _step_fn

seed_everything(42)

dt = 0.05
T = 200

gen = VonMisesGenerator(
    dt=dt,
    params={
        'mean_rate': 3.0,
        'R': 0.8,
        'freq': 30.0,
        'phase': 0.0,
    },
    name='theta_gen'
)

syn = TsodyksMarkramSynapse(n_pre=1, n_post=2, dt=dt, params={
    'gsyn_max': [[0.15, 0.12]],
    'tau_f': 30.0,
    'tau_d': 8.0,
    'tau_r': 300.0,
    'Uinc': 0.3,
    'pconn': [[1.0, 1.0]],
    'e_r': 0.0,
})

pop = IzhikevichMeanField(dt=dt, params={
    'tau_pop': [1.0, 1.0],
    'alpha': [0.5, 0.5],
    'a': [0.02, 0.02],
    'b': [0.2, 0.2],
    'w_jump': [0.1, 0.1],
    'Delta_I': [0.5, 0.5],
    'I_ext': [1.5, 1.5],
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

integrator = network._integrator

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

    new_syn_state, local_err = integrator.step_synapse(
        syn_entry.model, syn_state, pre_rate, post_v, syn_entry.model.dt
    )
    current_dict = syn_entry.model.compute_current(new_syn_state, pre_rate, post_v)

    R_hist.append(new_syn_state[0][0].numpy())
    U_hist.append(new_syn_state[1][0].numpy())
    A_hist.append(new_syn_state[2][0].numpy())
    I_syn_hist.append(current_dict['I_syn'][0].numpy())

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

rates = output.firing_rates['main'].numpy()[0]

fig, axes = plt.subplots(5, 1, figsize=(12, 12))

axes[0].plot(t_values, R_hist[:, 0], color='tab:blue', linewidth=1.5, label='R (vesicle fraction)')
axes[0].set_ylabel('R')
axes[0].set_title('R: Fraction of Available Vesicles (Numerical Integration)')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

axes[1].plot(t_values, U_hist[:, 0], color='tab:orange', linewidth=1.5, label='U (utilization)')
axes[1].set_ylabel('U')
axes[1].set_title('U: Fraction of Used Vesicles (Numerical Integration)')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

axes[2].plot(t_values, A_hist[:, 0], color='tab:green', linewidth=1.5, label='A (accumulator)')
axes[2].set_ylabel('A')
axes[2].set_title('A: Release Accumulator (Numerical Integration)')
axes[2].legend(fontsize=9)
axes[2].grid(True, alpha=0.3)

axes[3].plot(t_values, I_syn_hist[:, 0], color='tab:red', linewidth=1.5, label='Unit 0')
axes[3].plot(t_values, I_syn_hist[:, 1], color='tab:purple', linewidth=1.5, linestyle='--', label='Unit 1')
axes[3].set_ylabel('I_syn')
axes[3].set_title('Synaptic Current I_syn (Numerical Integration)')
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
    "TsodyksMarkramSynapse: Short-Term Plasticity Dynamics (Numerical Integration)\n"
    "Input: Von Mises (theta ~8Hz, R=0.8)",
    fontsize=13
)
plt.tight_layout()
plt.savefig("example_syn_tsodyks_markram_numeric.png", dpi=150)
plt.show()
print("Figure saved as example_syn_tsodyks_markram_numeric.png")