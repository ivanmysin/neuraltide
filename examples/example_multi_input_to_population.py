"""
Пример: два входа VonMisesGenerator → две популяции через синапс.

Логика связей:
- Генератор 1 (freq=30Hz) → Популяция 1
- Генератор 2 (freq=40Hz) → Популяция 2

Синапс: диагональная матрица связей (pconn), gsyn_max на диагонали = 50
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
    HAS_TF = True
except ImportError as e:
    print(f"TensorFlow not available: {e}")
    HAS_TF = False
    import sys
    sys.exit(0)

neuraltide.config.set_dtype(tf.float32)


dt = 0.05
T = 150

gen = VonMisesGenerator(
    dt=dt,
    params={
        'mean_rate': [20.0, 15.0],
        'R': [0.9, 0.7],
        'freq': [30.0, 40.0],
        'phase': [0.0, np.pi],
    }
)
print(f"VonMisesGenerator: n_units = {gen.n_units}")
print(f"  mean_rate = {gen.mean_rate.numpy()}")
print(f"  kappa = {gen.kappa.numpy()}")
print(f"  freq = {gen.freq.numpy()}")
print(f"  phase = {gen.phase.numpy()}")

pop = IzhikevichMeanField(dt=dt, params={
    'tau_pop':   {'value': [1.0, 1.0],   'trainable': False},
    'alpha':     {'value': [0.5, 0.5],   'trainable': False},
    'a':         {'value': [0.02, 0.02], 'trainable': False},
    'b':         {'value': [0.2, 0.2],   'trainable': False},
    'w_jump':    {'value': [0.1, 0.1],   'trainable': False},
    'Delta_I':   {'value': [0.05, 0.05],   'trainable': False},
    'I_ext':     {'value': [0.5, 0.5],   'trainable': False},
})
print(f"IzhikevichMeanField: n_units = {pop.n_units}")

syn = TsodyksMarkramSynapse(n_pre=2, n_post=2, dt=dt, params={
    'gsyn_max': {'value': [[50.0, 0.0],
                            [0.0, 50.0]], 'trainable': True},
    'tau_f':    {'value': 20.0,  'trainable': False},
    'tau_d':    {'value': 5.0,   'trainable': False},
    'tau_r':    {'value': 200.0, 'trainable': False},
    'Uinc':     {'value': 0.2,   'trainable': False},
    'pconn':    {'value': [[1.0, 0.0],
                            [0.0, 1.0]], 'trainable': False},
    'e_r':      {'value': 1.0,   'trainable': False},
})
print(f"TsodyksMarkramSynapse: n_pre={syn.n_pre}, n_post={syn.n_post}")

graph = NetworkGraph(dt=dt)
graph.add_input_population('inputs', gen)
graph.add_population('exc', pop)
graph.add_synapse('inputs->exc', syn, src='inputs', tgt='exc')

network = NetworkRNN(graph, integrator=RK4Integrator())

t_values = np.arange(0, T, dt, dtype=np.float32)
t_seq = tf.constant(t_values[None, :, None])

output = network(t_seq, training=False)
firing_rates = output.firing_rates['exc'].numpy()[0]

gen_output = gen(t_seq).numpy()[0]

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

for i in range(2):
    axes[0].plot(t_values, gen_output[:, i], linewidth=1.5, 
                 label=f"Input {i}: R={gen._params['R'][i]}, freq={gen._params['freq'][i]}")

axes[0].set_xlabel("Time (ms)")
axes[0].set_ylabel("Input Firing Rate (Hz)")
axes[0].set_title("VonMisesGenerator: 2 input channels")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

for i in range(2):
    axes[1].plot(t_values, firing_rates[:, i], linewidth=1.5, 
                 label=f"Population unit {i}")

axes[1].set_xlabel("Time (ms)")
axes[1].set_ylabel("Output Firing Rate (Hz)")
axes[1].set_title("IzhikevichMeanField output (2 units)")
axes[1].grid(True, alpha=0.3)
axes[1].legend()

axes[2].bar(['Input\nchannels', 'Synapse\nn_pre', 'Synapse\nn_post', 'Population\nn_units'],
            [gen.n_units, syn.n_pre, syn.n_post, pop.n_units],
            color=['tab:blue', 'tab:orange', 'tab:orange', 'tab:green'])
axes[2].set_ylabel("Count")
axes[2].set_title("Network dimensions")
for i, v in enumerate([gen.n_units, syn.n_pre, syn.n_post, pop.n_units]):
    axes[2].text(i, v + 0.1, str(v), ha='center', fontsize=12)

plt.tight_layout()
plt.savefig("example_multi_input_to_population.png", dpi=150)
plt.show()
print("\nFigure saved as example_multi_input_to_population.png")
