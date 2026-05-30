"""
Базовый пример: 1 вход → 1 синапс → 1 популяция.

Цель: обучить сеть воспроизводить заданную траекторию firing rate.
Вход: синусоидальный генератор (8 Hz).
Популяция: IzhikevichMeanField.
Синапс: TsodyksMarkramSynapse (обучаемый).
"""
import numpy as np
import tensorflow as tf

from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.populations import IzhikevichMeanField
from neuraltide.synapses import TsodyksMarkramSynapse
from neuraltide.inputs import SinusoidalGenerator
from neuraltide.integrators import RK4Integrator
from neuraltide.training import Trainer, CompositeLoss, MSELoss
from neuraltide.utils import seed_everything

seed_everything(42)

# === 1. Параметры ===
dt = 0.1       # шаг интегрирования, мс
T = 1000        # время симуляции, мс
n_steps = int(T / dt)

# === 2. Генератор входа ===
gen = SinusoidalGenerator(dt=dt, params={
    'amplitude': 10.0,   # Гц
    'freq': 8.0,         # Гц
    'phase': 0.0,        # рад
    'offset': 10.0,      # Гц
})

# === 3. Популяция ===
pop = IzhikevichMeanField(dt=dt, params={
    'tau_pop':  {'value': 1.0,  'trainable': False},
    'alpha':    {'value': 0.5,  'trainable': False},
    'a':        {'value': 0.02, 'trainable': False},
    'b':        {'value': 0.2,  'trainable': False},
    'w_jump':   {'value': 0.1,  'trainable': False},
    'Delta_I':  {'value': 0.01,  'trainable': True, 'min': 0.01, 'max': 2.0},
    'I_ext':    {'value': 0.0,  'trainable': True},
})

# === 4. Синапс ===
syn = TsodyksMarkramSynapse(n_pre=1, n_post=1, dt=dt, params={
    'gsyn_max': {'value': [[0.1]], 'trainable': True, 'min': 0.0},
    'tau_f':    {'value': 20.0, 'trainable': False},
    'tau_d':    {'value': 5.0,  'trainable': False},
    'tau_r':    {'value': 200.0, 'trainable': False},
    'Uinc':     {'value': 0.2,  'trainable': False},
    'pconn':    {'value': [[1.0]], 'trainable': False},
    'e_r':      {'value': 1.0,  'trainable': False},
})

# === 5. Граф и сеть ===
graph = NetworkGraph(dt=dt)
graph.declare_input('input', n_units=gen.n_units)
graph.add_population('exc', pop)
graph.add_synapse('input->exc', syn, src='input', tgt='exc')

network = NetworkRNN(graph, RK4Integrator())

# === 6. Данные ===
t_values = np.arange(n_steps, dtype=np.float32) * dt
t_seq = tf.constant(t_values[None, :, None])  # [1, T, 1]

# Входные частоты
inputs = graph.pack_inputs({'input': gen(t_seq)})

# Целевая траектория: та же синусоида
target_rates = 10.0 + 5.0 * np.sin(2 * np.pi * 8.0 * t_values / 1000.0)
target = {
    'exc': tf.constant(target_rates[None, :, None], dtype=tf.float32)
}

# === 7. Обучение ===
loss_fn = CompositeLoss([(1.0, MSELoss(target))])
trainer = Trainer(network, loss_fn,
                  optimizer=tf.keras.optimizers.Adam(1e-2))

print("Обучение...")
history = trainer.fit(t_seq, inputs=inputs, epochs=50, verbose=0)

# === 8. Результат ===
output = network(t_seq, inputs=inputs, training=False)
pred = output.firing_rates['exc'].numpy()[0, :, 0]

print(f"Начальный loss: {history.loss_history[0]:.2f}")
print(f"Конечный loss:  {history.loss_history[-1]:.2f}")
print(f"Корреляция:     {np.corrcoef(target_rates, pred)[0, 1]:.4f}")

# === 9. Визуализация ===
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

input_rates = inputs.numpy()[0, :, 0]

fig, axes = plt.subplots(3, 1, figsize=(12, 8))

# Вход
axes[0].plot(t_values, input_rates, color='tab:blue', linewidth=1.5)
axes[0].set_ylabel('Input rate (Hz)')
axes[0].set_title('Вход (SinusoidalGenerator)')
axes[0].grid(True, alpha=0.3)

# Цель и выход
axes[1].plot(t_values, target_rates, color='tab:green', linewidth=2, label='Target', alpha=0.8)
axes[1].plot(t_values, pred, color='tab:red', linewidth=1.5, linestyle='--', label='Prediction', alpha=0.8)
axes[1].set_ylabel('Firing rate (Hz)')
axes[1].set_title('Цель vs Предсказание')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Loss
axes[2].plot(history.loss_history, color='tab:blue', linewidth=1.5)
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Loss')
axes[2].set_title('Training Loss')
axes[2].grid(True, alpha=0.3)

fig.suptitle('Базовый пример: input → synapse → population', fontsize=13)
plt.tight_layout()
plt.savefig('example_simple.png', dpi=150)
print("\nГрафик сохранён: example_simple.png")
