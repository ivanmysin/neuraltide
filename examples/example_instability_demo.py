"""
Пример: Демонстрация работы штрафа за численную нестабильность (StabilityPenalty).

Цель примера: показать, как StabilityPenalty влияет на обучаемые параметры.
Параметр 'I_ext' (внешний ток) влияет на жесткость динамики системы:
- Большой I_ext → более резкие переходы → выше локальная ошибка интегрирования
- Оптимизатор, минимизируя stability_loss, уменьшает I_ext

Сценарий:
- Две сети с одинаковым начальным I_ext=1.0
- Сеть 1: MSE loss + StabilityPenalty (stability_penalty_weight=10 в NetworkRNN)
- Сеть 2: MSE loss БЕЗ StabilityPenalty
- Целевая траектория: постоянный firing rate 0.3 (безразмерный)
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import neuraltide
from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.populations import IzhikevichMeanField
from neuraltide.synapses import TsodyksMarkramSynapse
from neuraltide.inputs import ConstantRateGenerator
from neuraltide.integrators import RK4Integrator
from neuraltide.training import Trainer, CompositeLoss, MSELoss, StabilityPenalty
from neuraltide.utils import seed_everything, print_summary

seed_everything(42)

dt = 0.5  # Больший dt для усиления эффекта нелинейности
T = 50
n_steps = int(T / dt)

def create_network(i_ext_value, stability_weight, name):
    """Создает сеть с заданным I_ext и весом stability penalty."""
    pop = IzhikevichMeanField(dt=dt, params={
    'tau_pop':   {'value': 1.0,   'trainable': False},
    'alpha':     {'value': 0.5,   'trainable': False},
    'a':         {'value': 0.02,  'trainable': False},
    'b':         {'value': 0.2,   'trainable': False},
    'w_jump':    {'value': 0.5,   'trainable': False},
        'Delta_I':   {'value': 0.5,   'trainable': False},
        'I_ext':     {'value': i_ext_value, 'trainable': True,
                      'min': 0.1, 'max': 5.0},
    }, name=f'pop_{name}')

    gen = ConstantRateGenerator(
        dt=dt,
        params={'rate': 15.0},
        name=f'input_{name}'
    )

    syn_in = TsodyksMarkramSynapse(n_pre=1, n_post=1, dt=dt, params={
        'gsyn_max': {'value': [[0.5]], 'trainable': True},
        'tau_f':    {'value': 20.0,  'trainable': False},
        'tau_d':    {'value': 5.0,   'trainable': False},
        'tau_r':    {'value': 200.0, 'trainable': False},
        'Uinc':     {'value': 0.2,   'trainable': False},
        'pconn':    {'value': [[1.0]], 'trainable': False},
        'e_r':      {'value': 0.0,   'trainable': False},
    })

    graph = NetworkGraph(dt=dt)
    graph.add_input_population('input', gen)
    graph.add_population('exc', pop)
    graph.add_synapse('input->exc', syn_in, src='input', tgt='exc')

    network = NetworkRNN(graph, integrator=RK4Integrator(),
                         stability_penalty_weight=stability_weight,
                         name=f'network_{name}')
    return network


i_ext_initial = 1.0  # Большой I_ext — система жесткая, высокая нестабильность
stability_weight = 10

network_with_penalty = create_network(i_ext_initial, stability_weight, 'with_penalty')
network_without_penalty = create_network(i_ext_initial, 0.0, 'without_penalty')

t_values = np.arange(n_steps, dtype=np.float32) * dt
t_seq = tf.constant(t_values[None, :, None])

target_rate = 0.3 * np.ones_like(t_values)
target = {
    'exc': tf.constant(target_rate[None, :, None], dtype=tf.float32)
}

print("=== Проверка stability_loss до обучения ===")
output_before = network_with_penalty(t_seq, training=False)
print(f"stability_loss со штрафом (I_ext={i_ext_initial}): {output_before.stability_loss.numpy()}")
output_before_no = network_without_penalty(t_seq, training=False)
print(f"stability_loss без штрафа: {output_before_no.stability_loss.numpy()}")

print("\n=== Сеть со StabilityPenalty ===")
print_summary(network_with_penalty)
print("\n=== Сеть без StabilityPenalty ===")
print_summary(network_without_penalty)

loss_fn_with = CompositeLoss([
    (1.0, MSELoss(target)),
    (1.0, StabilityPenalty()),  # Weight=1, потому что stability_weight уже учтен внутри NetworkRNN
])
loss_fn_without = CompositeLoss([
    (1.0, MSELoss(target)),
])

print("\n=== Обучение сети со StabilityPenalty ===")
i_ext_history_with = []
stability_history_with = []
loss_history_with = []
trainer_with = Trainer(network_with_penalty, loss_fn_with,
                       optimizer=tf.keras.optimizers.Adam(1e-3))

for epoch in range(50):
    result = trainer_with.train_step(t_seq)
    i_ext_history_with.append(network_with_penalty._graph._populations['exc'].I_ext.numpy()[0])
    loss_history_with.append(float(result['loss']))
    output = network_with_penalty(t_seq, training=False)
    stability_history_with.append(float(output.stability_loss.numpy()))
    if epoch % 10 == 0:
        i_ext_val = network_with_penalty._graph._populations['exc'].I_ext.numpy()[0]
        print(f"Epoch {epoch}/50: I_ext={i_ext_val:.4f}, stability_loss={output.stability_loss.numpy():.6f}")

print("\n=== Обучение сети без StabilityPenalty ===")
i_ext_history_without = []
stability_history_without = []
loss_history_without = []
trainer_without = Trainer(network_without_penalty, loss_fn_without,
                          optimizer=tf.keras.optimizers.Adam(1e-3))
for epoch in range(50):
    result = trainer_without.train_step(t_seq)
    i_ext_history_without.append(network_without_penalty._graph._populations['exc'].I_ext.numpy()[0])
    loss_history_without.append(float(result['loss']))
    output = network_without_penalty(t_seq, training=False)
    stability_history_without.append(float(output.stability_loss.numpy()))
    if epoch % 10 == 0:
        i_ext_val = network_without_penalty._graph._populations['exc'].I_ext.numpy()[0]
        print(f"Epoch {epoch}/50: I_ext={i_ext_val:.4f}, stability_loss={output.stability_loss.numpy():.6f}")

print(f"\n=== Проверка stability_loss после обучения ===")
output_after = network_with_penalty(t_seq, training=False)
print(f"stability_loss со штрафом: {output_after.stability_loss.numpy()}")
output_after_no = network_without_penalty(t_seq, training=False)
print(f"stability_loss без штрафа: {output_after_no.stability_loss.numpy()}")

i_ext_with_penalty = network_with_penalty._graph._populations['exc'].I_ext.numpy()[0]
i_ext_without_penalty = network_without_penalty._graph._populations['exc'].I_ext.numpy()[0]

print(f"\n=== Результаты ===")
print(f"Начальное I_ext: {i_ext_initial:.3f}")
print(f"I_ext со StabilityPenalty: {i_ext_with_penalty:.3f} (изменение: {i_ext_with_penalty - i_ext_initial:.3f})")
print(f"I_ext без StabilityPenalty: {i_ext_without_penalty:.3f} (изменение: {i_ext_without_penalty - i_ext_initial:.3f})")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

epochs = np.arange(1, len(loss_history_with) + 1)

axes[0, 0].plot(epochs, loss_history_with, color='tab:blue', linewidth=2, label='Со StabilityPenalty')
axes[0, 0].plot(epochs, loss_history_without, color='tab:orange', linewidth=2, label='Без StabilityPenalty')
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Total Loss")
axes[0, 0].set_title("Общий Loss по эпохам")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(epochs, i_ext_history_with, color='tab:blue', linewidth=2, label='Со StabilityPenalty')
axes[0, 1].plot(epochs, i_ext_history_without, color='tab:orange', linewidth=2, label='Без StabilityPenalty')
axes[0, 1].axhline(i_ext_initial, color='gray', linestyle='--', linewidth=1, label=f'Начальное I_ext={i_ext_initial}')
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("Параметр I_ext")
axes[0, 1].set_title("Параметр 'I_ext' по эпохам")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(epochs, stability_history_with, color='tab:blue', linewidth=2)
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_ylabel("Stability Loss")
axes[1, 0].set_title("Stability Loss (со StabilityPenalty)")
axes[1, 0].grid(True, alpha=0.3)

output_with = network_with_penalty(t_seq, training=False)
output_without = network_without_penalty(t_seq, training=False)

axes[1, 1].plot(t_values, target_rate, color='tab:green', linewidth=2, label='Target')
axes[1, 1].plot(t_values, output_with.firing_rates['exc'].numpy()[0, :, 0], 
                color='tab:blue', linewidth=1.5, linestyle='--', label='Со StabilityPenalty')
axes[1, 1].plot(t_values, output_without.firing_rates['exc'].numpy()[0, :, 0], 
                color='tab:orange', linewidth=1.5, linestyle='--', label='Без StabilityPenalty')
axes[1, 1].set_xlabel("Time (ms)")
axes[1, 1].set_ylabel("Firing Rate (Hz)")
axes[1, 1].set_title("Firing Rate после обучения")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

fig.suptitle(f"Демонстрация StabilityPenalty: I_ext меняется только со штрафом\n"
             f"I_ext: {i_ext_initial:.2f} → {i_ext_with_penalty:.3f} (со штрафом) vs {i_ext_without_penalty:.3f} (без)", 
             fontsize=13)
plt.tight_layout()
plt.savefig("example_instability_demo.png", dpi=150)
plt.show()

print("\nГрафик сохранен как example_instability_demo.png")