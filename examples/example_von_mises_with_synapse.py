"""
Пример: VonMisesGenerator + TsodyksMarkramSynapse - динамика синаптических переменных.

Показывает:
- Выход VonMisesGenerator (1 канал)
- Динамику переменных синапса R, U, A
- Результирующий ток на популяцию
"""
import numpy as np
import matplotlib.pyplot as plt

try:
    import tensorflow as tf
    import neuraltide
    from neuraltide.core.network import NetworkGraph, NetworkRNN
    from neuraltide.populations import IzhikevichMeanField
    from neuraltide.synapses import TsodyksMarkramSynapse
    from neuraltide.inputs import VonMisesGenerator
    from neuraltide.integrators import RK4Integrator
    from neuraltide.populations.input_population import InputPopulation
    HAS_TF = True
except ImportError as e:
    print(f"TensorFlow not available: {e}")
    HAS_TF = False
    import sys
    sys.exit(0)

neuraltide.config.set_dtype(tf.float32)

dt = 0.1
T = 500

gen = VonMisesGenerator(
    dt=dt,
    params={
        'mean_rate': 20.0,
        'R': 0.9,
        'freq': 8.0,
        'phase': 0.0,
    }
)
print(f"VonMisesGenerator: n_units = {gen.n_units}")

pop = IzhikevichMeanField(dt=dt, params={
    'tau_pop':   {'value': 1.0,   'trainable': False},
    'alpha':     {'value': 0.5,   'trainable': False},
    'a':         {'value': 0.02,  'trainable': False},
    'b':         {'value': 0.2,   'trainable': False},
    'w_jump':    {'value': 0.1,   'trainable': False},
    'Delta_I':   {'value': 0.5,   'trainable': False},
    'I_ext':     {'value': 1.0,   'trainable': False},
})

syn = TsodyksMarkramSynapse(n_pre=1, n_post=1, dt=dt, params={
    'gsyn_max': {'value': [[0.5]], 'trainable': False},
    'tau_f':    {'value': 20.0,  'trainable': False},
    'tau_d':    {'value': 200.0, 'trainable': False},
    'tau_r':    {'value': 500.0, 'trainable': False},
    'Uinc':     {'value': 0.3,   'trainable': False},
    'pconn':    {'value': [[1.0]], 'trainable': False},
    'e_r':      {'value': 1.0,   'trainable': False},
})

graph = NetworkGraph(dt=dt)
graph.add_input_population('input', gen)
graph.add_population('exc', pop)
graph.add_synapse('input->exc', syn, src='input', tgt='exc')

network = NetworkRNN(graph, integrator=RK4Integrator())

t_values = np.arange(0, T, dt, dtype=np.float32)
t_seq = tf.constant(t_values[None, :, None])

output = network(t_seq, training=False)
firing_rates = output.firing_rates['exc'].numpy()[0]

gen_output = gen(t_seq).numpy()[0]

input_pop = graph._populations['input']
pop_model = graph._populations['exc']

syn_state = syn.get_initial_state(batch_size=1)
pop_state = pop_model.get_initial_state(batch_size=1)

R_history = []
U_history = []
A_history = []
I_syn_history = []

for t in t_values:
    t_tensor = tf.constant([[t]], dtype=tf.float32)
    
    pre_rate = input_pop.get_firing_rate([t_tensor])
    
    post_v = tf.zeros([1, 1], dtype=tf.float32)
    
    current_dict, new_syn_state = syn.forward(pre_rate, post_v, syn_state, dt)
    
    syn_state = new_syn_state
    
    R_val = syn_state[0].numpy()[0, 0]
    U_val = syn_state[1].numpy()[0, 0]
    A_val = syn_state[2].numpy()[0, 0]
    I_val = current_dict['I_syn'].numpy()[0, 0]
    
    R_history.append(R_val)
    U_history.append(U_val)
    A_history.append(A_val)
    I_syn_history.append(I_val)

fig, axes = plt.subplots(4, 1, figsize=(12, 12))

axes[0].plot(t_values, gen_output[:, 0], linewidth=2, color='tab:blue')
axes[0].set_ylabel("Input Rate (Hz)")
axes[0].set_title("VonMisesGenerator output")
axes[0].grid(True, alpha=0.3)

axes[1].plot(t_values, R_history, linewidth=1.5, label='R (fraction of ready vesicles)')
axes[1].plot(t_values, U_history, linewidth=1.5, label='U (release probability)')
axes[1].plot(t_values, A_history, linewidth=1.5, label='A (accumulated)')
axes[1].set_ylabel("Value")
axes[1].set_title("TsodyksMarkramSynapse state variables")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(t_values, I_syn_history, linewidth=1.5, color='tab:orange')
axes[2].set_ylabel("I_syn")
axes[2].set_title("Synaptic current to population")
axes[2].grid(True, alpha=0.3)

axes[3].plot(t_values, firing_rates[:, 0], linewidth=2, color='tab:green')
axes[3].set_xlabel("Time (ms)")
axes[3].set_ylabel("Firing Rate (Hz)")
axes[3].set_title("Population output")
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("von_mises_with_synapse.png", dpi=150)
plt.show()
print("\nFigure saved as von_mises_with_synapse.png")
