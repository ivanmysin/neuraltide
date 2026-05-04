"""
Пример: Две взаимно ингибирующие популяции fast-spiking нейронов.

Архитектура:
- Две однородные популяции IzhikevichMeanField (Pop1 и Pop2)
- Взаимное ингибирование через синапсы TsodyksMarkramSynapse (short-term depression)
- Внешний возбуждающий ток подаётся на каждую популяцию

Оптимизируемые параметры:
- g12, g21: максимальные синаптические проводимости
- I_ext1, I_ext2: внешние токи
- tau_d, tau_r, tau_f, U_inc: параметры кратковременной депрессии
"""

import time

import numpy as np
import tensorflow as tf

from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.inputs import VonMisesGenerator
from neuraltide.integrators import RK4Integrator
from neuraltide.populations import IzhikevichMeanField
from neuraltide.synapses import TsodyksMarkramSynapse
from neuraltide.training import Trainer, CompositeLoss, MSELoss, MSLELoss, StabilityPenalty
from copy import deepcopy


dt = 0.1
T_total = 1000
transient = 200
nepochs = 100

Loss = MSLELoss

n_transient_steps = int(transient / dt)

t_all = np.arange(0, T_total, dt)
t_seq_full = tf.constant(t_all[None, :, None], dtype=tf.float32)

print(f"Total: T={T_total}ms, dt={dt}ms, steps={t_all.shape[0]}, "
      f"transient={transient}ms, epochs={nepochs}")

# --- Популяции ---
pop_params_1 = {
    'tau_pop': {'value': [1.6622994,], 'trainable': False},
    'alpha': {'value': [0.38348085,], 'trainable': False},
    'a': {'value': [0.0083115,], 'trainable': False},
    'b': {'value': [0.00320795,], 'trainable': False},
    'w_jump': {'value': [0.00050604,], 'trainable': False},
    'Delta_I': {'value': [0.00632551,], 'trainable': False, 'min': 0.01, 'max': 2.0},
    'I_ext': {'value': [0.5,], 'trainable': True, 'min': -20.0, 'max': 20.0},
}

pop_params_2 = deepcopy(pop_params_1)
pop_params_2['I_ext']['value'] = [0.1,]

pop1 = IzhikevichMeanField(dt=dt, params=pop_params_1, name='pop1')
pop2 = IzhikevichMeanField(dt=dt, params=pop_params_2, name='pop2')

# --- Синапсы ---
syn_params_1 = {
    'gsyn_max': {'value': 10.5, 'trainable': True, 'min': 0.0, 'max': 100.0},
    'tau_d': {'value': 6.02, 'trainable': True, 'min': 2.0, 'max': 15.0},
    'tau_r': {'value': 200.0, 'trainable': True, 'min': 50.0, 'max': 500.0},
    'tau_f': {'value': 20.0, 'trainable': True, 'min': 5.0, 'max': 100.0},
    'Uinc': {'value': 0.3, 'trainable': True, 'min': 0.1, 'max': 0.6},
    'pconn': {'value': 1.0, 'trainable': False},
    'e_r': {'value': -0.15, 'trainable': False},
}

syn_params_2 = deepcopy(syn_params_1)
syn_params_2['gsyn_max']['value'] = 5.0

syn_1to2 = TsodyksMarkramSynapse(n_pre=1, n_post=1, dt=dt, params=syn_params_1,
                                 name='syn_1to2')
syn_2to1 = TsodyksMarkramSynapse(n_pre=1, n_post=1, dt=dt, params=syn_params_2,
                                 name='syn_2to1')

# --- Граф ---
graph = NetworkGraph(dt=dt)
graph.add_population('pop1', pop1)
graph.add_population('pop2', pop2)
graph.add_synapse('pop1->pop2', syn_1to2, src='pop1', tgt='pop2')
graph.add_synapse('pop2->pop1', syn_2to1, src='pop2', tgt='pop1')

network = NetworkRNN(graph, integrator=RK4Integrator(),
                     stability_penalty_weight=1e-2)

# --- Целевой сигнал (von Mises, противофазные осцилляции) ---
gen_target = VonMisesGenerator(dt=dt, params={
    'mean_rate': 50.0,
    'R': 0.6,
    'freq': 8.0,
    'phase': [0.0, 2.61],
})
target_full = gen_target(tf.constant(t_all[None, :, None], dtype=tf.float32))
target_full = target_full[0, :, :].numpy()  # [T, 2]

target = {
    'pop1': tf.constant(target_full[None, :, 0:1], dtype=tf.float32),
    'pop2': tf.constant(target_full[None, :, 1:2], dtype=tf.float32),
}

# --- Trainer ---
loss_fn = CompositeLoss([
    (1.0, Loss(target)),
    (1e-3, StabilityPenalty()),
])
optimizer = tf.keras.optimizers.Adam(1e-2)
trainer = Trainer(network, loss_fn, optimizer, grad_method='bptt')
                  #grad_clip_norm=1.0)

# --- Предварительный transient-прогон ---
print("\n=== Transient simulation ===")
t_transient = tf.constant(t_all[:n_transient_steps][None, :, None],
                          dtype=tf.float32)
transient_output = network(t_transient, training=False)
init_state = transient_output.final_state
print(f"Transient done: pop1 r={float(init_state[0][0][0, 0]):.4f}")

# --- Оптимизация ---
print(f"\n=== Optimization: {nepochs} epochs ===")
total_start = time.time()

for epoch in range(nepochs):
    epoch_start = time.time()

    result = trainer.train_step(t_seq_full,
                                initial_state=init_state)

    epoch_time = time.time() - epoch_start
    loss_val = float(result['loss'])

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d}/{nepochs} | loss: {loss_val:.6f} | "
              f"{epoch_time:.1f}s")

total_time = time.time() - total_start
print(f"\nTotal optimization time: {total_time:.1f}s "
      f"({total_time/nepochs:.2f}s/epoch)")

# --- Финальная симуляция ---
print("\n=== Final simulation ===")
final_output = network(t_seq_full, training=False,
                       initial_state=init_state)
rates_pop1 = final_output.firing_rates['pop1'].numpy()[0, :, 0]
rates_pop2 = final_output.firing_rates['pop2'].numpy()[0, :, 0]

n_eval_start = n_transient_steps
print(f"Pop1 mean rate (after transient): {rates_pop1[n_eval_start:].mean():.2f} Hz")
print(f"Pop2 mean rate (after transient): {rates_pop2[n_eval_start:].mean():.2f} Hz")
print(f"MSE vs target: "
      f"pop1={np.mean((rates_pop1[n_eval_start:] - target_full[n_eval_start:, 0])**2):.4f}, "
      f"pop2={np.mean((rates_pop2[n_eval_start:] - target_full[n_eval_start:, 1])**2):.4f}")

# --- Параметры ---
print("\n=== Optimized parameters ===")
for v in network.trainable_variables:
    print(f"  {v.name.split(':')[0]}: {float(v.numpy().flatten()[0]):.4f}")

# --- Экспорт ---
trainer.export_results('optimization_results.json')
trainer.export_results('optimization_results.csv', format='csv')
print("\nResults exported to optimization_results.json/.csv")

# --- Визуализация ---
print("\n=== Visualization ===")
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    ax = axes[0]
    ax.plot(t_all, target_full[:, 0], 'b--', label='Target Pop1', linewidth=2)
    ax.plot(t_all, rates_pop1, 'b-', label='Pop1', linewidth=1.2)
    ax.plot(t_all, target_full[:, 1], 'r--', label='Target Pop2', linewidth=2)
    ax.plot(t_all, rates_pop2, 'r-', label='Pop2', linewidth=1.2)
    ax.axvline(t_all[n_transient_steps], color='gray', linestyle=':',
               label=f'Transient end ({transient}ms)')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Firing Rate (Hz)')
    ax.set_title('Target vs Actual Firing Rates')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    var_names = [v.name.split(':')[0] for v in network.trainable_variables]
    var_values = [float(v.numpy().flatten()[0])
                  for v in network.trainable_variables]
    colors = ['steelblue' if 'gsyn' in n else 'coral' for n in var_names]
    ax.bar(range(len(var_names)), var_values, color=colors)
    ax.set_xticks(range(len(var_names)))
    ax.set_xticklabels(var_names, rotation=45, ha='right')
    ax.set_ylabel('Value')
    ax.set_title('Optimized Parameters')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('optimization_visualization.png', dpi=150)
    print("Saved to optimization_visualization.png")
    plt.close()

except ImportError:
    print("Matplotlib not available")
