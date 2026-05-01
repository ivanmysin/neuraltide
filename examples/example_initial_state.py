"""
Пример: Управление начальным состоянием.

Демонстрирует:
1. Установку начальных условий (r=0.5, v=-1.0, w=0.0)
2. Продолжение симуляции через final_state из NetworkOutput
3. Сброс к нулевому начальному состоянию
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.populations import IzhikevichMeanField
from neuraltide.synapses import TsodyksMarkramSynapse
from neuraltide.inputs import SinusoidalGenerator
from neuraltide.integrators import RK4Integrator


dt = 0.05
T = 50
batch_size = 1

pop = IzhikevichMeanField(dt=dt, params={
    'tau_pop': {'value': 1.0, 'trainable': False},
    'alpha': {'value': 0.5, 'trainable': False},
    'a': {'value': 0.02, 'trainable': False},
    'b': {'value': 0.2, 'trainable': False},
    'w_jump': {'value': 0.1, 'trainable': False},
    'Delta_I': {'value': 0.5, 'trainable': False},
    'I_ext': {'value': 0.0, 'trainable': False},
})

gen = SinusoidalGenerator(
    dt=dt,
    params={
        'amplitude': 5.0,
        'freq': 8.0,
        'phase': 0.0,
        'offset': 10.0,
    },
    name='theta'
)

syn = TsodyksMarkramSynapse(n_pre=1, n_post=1, dt=dt, params={
    'gsyn_max': {'value': 0.1, 'trainable': False},
    'tau_f': {'value': 20.0, 'trainable': False},
    'tau_d': {'value': 5.0, 'trainable': False},
    'tau_r': {'value': 200.0, 'trainable': False},
    'Uinc': {'value': 0.2, 'trainable': False},
    'pconn': {'value': [[1.0]], 'trainable': False},
    'e_r': {'value': 0.0, 'trainable': False},
})

graph = NetworkGraph(dt=dt)
graph.add_input_population('theta', gen)
graph.add_population('exc', pop)
graph.add_synapse('theta->exc', syn, src='theta', tgt='exc')

network = NetworkRNN(graph, integrator=RK4Integrator())
print("Network created")

t_values = np.arange(T, dtype=np.float32) * dt
t_seq = tf.constant(t_values[None, :, None])
print(f"Time sequence: {t_values.shape}")

# Кастомные начальные условия
init_pop, init_syn = network.get_initial_state(batch_size)
init_pop[0] = tf.constant([[0.5]])    # r = 0.5
init_pop[1] = tf.constant([[-1.0]])   # v = -1.0
init_pop[2] = tf.constant([[0.0]])    # w = 0.0
custom_init = (init_pop, init_syn)
print(f"Custom init: r={init_pop[0].numpy()}, v={init_pop[1].numpy()}, "
      f"w={init_pop[2].numpy()}")

# Batch 1: custom start
output1 = network(t_seq, training=False, initial_state=custom_init)
rates1 = output1.firing_rates['exc'].numpy()[0, -1, 0]
final1 = output1.final_state
print(f"Batch 1 final r={float(final1[0][0][0, 0]):.4f}")

# Batch 2: продолжение из конечного состояния batch 1
output2 = network(t_seq, training=False, initial_state=final1)
rates2 = output2.firing_rates['exc'].numpy()[0, -1, 0]
print(f"Batch 2 final rate: {rates2:.2f} Hz (continuing from batch 1)")

# Batch 3: сброс — нулевое начальное состояние
output3 = network(t_seq, training=False)
rates3 = output3.firing_rates['exc'].numpy()[0, 0, 0]
print(f"Batch 3 start rate (reset): {rates3:.2f} Hz")

# Визуализация
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(t_values, output1.firing_rates['exc'].numpy()[0, :, 0],
             label='Batch 1 (custom init)', linewidth=1.5)
axes[0].plot(t_values, output2.firing_rates['exc'].numpy()[0, :, 0],
             label='Batch 2 (continued)', linewidth=1.5, linestyle='--')
axes[0].axhline(0.5, color='red', linestyle=':', alpha=0.5, label='Initial r=0.5')
axes[0].set_xlabel('Time (ms)')
axes[0].set_ylabel('Firing Rate (Hz)')
axes[0].set_title('Custom Initial Conditions')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(t_values, output2.firing_rates['exc'].numpy()[0, :, 0],
             label='Batch 2 (continued)', linewidth=1.5)
axes[1].plot(t_values, output3.firing_rates['exc'].numpy()[0, :, 0],
             label='Batch 3 (reset)', linewidth=1.5, linestyle='--')
axes[1].set_xlabel('Time (ms)')
axes[1].set_ylabel('Firing Rate (Hz)')
axes[1].set_title('Continuation vs Reset')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

fig.suptitle("Initial State Example")
plt.tight_layout()
plt.savefig("example_initial_state.png", dpi=150)
plt.show()
print("\nSaved to example_initial_state.png")
