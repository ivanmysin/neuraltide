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
from copy import deepcopy

import numpy as np
import tensorflow as tf

from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.inputs import VonMisesGenerator
from neuraltide.integrators import RK4Integrator
from neuraltide.populations import IzhikevichMeanField
from neuraltide.synapses import TsodyksMarkramSynapse
from neuraltide.training import Trainer, CompositeLoss, MSELoss, MSLELoss, StabilityPenalty


dt = 0.1
T_total = 400
transient = 0.1
nepochs = 500

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
    'Delta_I': {'value': [0.00632551,], 'trainable': False, 'min': 0.00001, 'max': 2.0},
    'I_ext': {'value': [0.2,], 'trainable': True, 'min': -2.0, 'max': 2.0},
}

pop_params_2 = deepcopy(pop_params_1)
pop_params_2['I_ext']['value'] = [0.3,]

pop1 = IzhikevichMeanField(dt=dt, params=pop_params_1, name='pop1')
pop2 = IzhikevichMeanField(dt=dt, params=pop_params_2, name='pop2')

# --- Синапсы ---
syn_params_1 = {
    'gsyn_max': {'value': 20.5, 'trainable': True, 'min': 0.0, 'max': 500.0},
    'tau_d': {'value': 5.02, 'trainable': True, 'min': 1.0, 'max': 15.0},
    'tau_r': {'value': 317.0, 'trainable': True, 'min': 50.0, 'max': 500.0},
    'tau_f': {'value': 4.0, 'trainable': True, 'min': 1.0, 'max': 100.0},
    'Uinc': {'value': 0.6, 'trainable': True, 'min': 0.1, 'max': 0.6},
    'pconn': {'value': 1.0, 'trainable': False},
    'e_r': {'value': -0.15, 'trainable': False},
}

syn_params_2 = deepcopy(syn_params_1)
syn_params_2['gsyn_max']['value'] = 20


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

# --- Предварительная визуализация (до обучения) ---
print("\n=== Pre-training simulation (collecting I_syn/g_syn) ===")
try:
    import matplotlib.pyplot as plt
    plt.ion()  # interactive mode

    # --- Быстрое сканирование с сохранением состояний ---
    _, _, _, _, all_pop_stacked, all_syn_stacked = \
        network._scan_forward_states(t_seq_full,
                                     tuple(init_state[0]),
                                     tuple(init_state[1]))

    # --- Индексация состояний ---
    pop_names = graph.population_names
    syn_names = graph.synapse_names

    # Собираем словари: имя → список индексов в stacked-кортеже
    pop_state_ranges = {}
    idx = 0
    for name in pop_names:
        n = len(graph._populations[name].state_size)
        pop_state_ranges[name] = (idx, idx + n)
        idx += n

    syn_state_ranges = {}
    idx = 0
    for name in syn_names:
        entry = graph._synapses[name]
        n = len(entry.model.state_size)
        syn_state_ranges[name] = (idx, idx + n)
        idx += n

    # --- Запускаем быстрый прогон для получения firing_rates ---
    pre_rates = {}
    pre_output = network(t_seq_full, training=False, initial_state=init_state)
    for name in graph.dynamic_population_names:
        pre_rates[name] = pre_output.firing_rates[name].numpy()[0, :, :]

    # --- I_syn и g_syn для каждого синапса ---
    dtype_np = np.float32
    T_steps = t_all.shape[0]
    syn_I = {}
    syn_g = {}

    for syn_name in syn_names:
        syn_I[syn_name] = np.zeros(T_steps, dtype=dtype_np)
        syn_g[syn_name] = np.zeros(T_steps, dtype=dtype_np)

    # Предвычисляем gsyn_max и e_r для каждого синапса
    syn_params_cache = {}
    for syn_name in syn_names:
        entry = graph._synapses[syn_name]
        m = entry.model
        syn_params_cache[syn_name] = {
            'gsyn_max': m.gsyn_max.numpy(),
            'e_r': m.e_r.numpy(),
            'n_pre': m.n_pre,
            'n_post': m.n_post,
            'src': entry.src,
            'tgt': entry.tgt,
        }

    # Предвычисляем firing rates для всех популяций на каждом шагу
    # из stacked состояний (быстрее, чем вызов model каждый раз)
    pop_rate_arrays = {}
    pop_v_arrays = {}
    for name in graph.dynamic_population_names:
        pop = graph._populations[name]
        start, end = pop_state_ranges[name]
        r_stacked = all_pop_stacked[start].numpy()  # [T, 1, n_units]
        v_stacked = all_pop_stacked[start + 1].numpy()  # [T, 1, n_units]
        tau_pop = pop.tau_pop.numpy()  # [n_units]
        pop_rate_arrays[name] = r_stacked[:, 0, :] / (tau_pop * 1e-3)  # [T, n_units] in Hz
        pop_v_arrays[name] = v_stacked[:, 0, :]  # [T, n_units]

    # --- Вычисление I_syn / g_syn ---
    for syn_name in syn_names:
        entry = graph._synapses[syn_name]
        m = entry.model
        src_name = entry.src
        tgt_name = entry.tgt

        gsyn_max = syn_params_cache[syn_name]['gsyn_max']  # [n_pre, n_post]
        e_r = syn_params_cache[syn_name]['e_r']  # [n_pre, n_post]
        n_pre = syn_params_cache[syn_name]['n_pre']
        n_post = syn_params_cache[syn_name]['n_post']

        start, end = syn_state_ranges[syn_name]
        # TsodyksMarkram state = [R, U, A], each [T, n_pre, n_post]
        A_stacked = all_syn_stacked[start + 2].numpy()  # [T, n_pre, n_post]

        post_v_stacked = pop_v_arrays[tgt_name]  # [T, n_post]

        for t in range(T_steps):
            A_t = A_stacked[t]  # [n_pre, n_post]
            g_eff = gsyn_max * A_t  # [n_pre, n_post]
            g_syn_val = np.sum(g_eff, axis=0)  # [n_post]
            post_v = post_v_stacked[t]  # [n_post]

            # e_r - post_v: [n_pre, n_post] - [n_post] → broadcast
            I_pair = g_eff * (e_r - post_v[np.newaxis, :])  # [n_pre, n_post]
            I_syn_val = np.sum(I_pair, axis=0)  # [n_post]

            syn_I[syn_name][t] = np.sum(I_syn_val)
            syn_g[syn_name][t] = np.sum(g_syn_val)

    # --- Интерактивная визуализация ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # 1) Firing rates
    ax = axes[0]
    for name in graph.dynamic_population_names:
        label = f'{name} (mean={pre_rates[name][n_transient_steps:].mean():.1f} Hz)'
        ax.plot(t_all, pre_rates[name][:, 0], label=label, linewidth=1.2)
    ax.axvline(t_all[n_transient_steps], color='gray', linestyle=':',
               label=f'Transient end ({transient}ms)')
    ax.set_ylabel('Firing Rate (Hz)')
    ax.set_title('Pre-training: Firing Rates')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 2) Synaptic currents I_syn
    ax = axes[1]
    for syn_name in syn_names:
        entry = graph._synapses[syn_name]
        label = f'{syn_name}  ({entry.src}→{entry.tgt})'
        ax.plot(t_all, syn_I[syn_name], label=label, linewidth=1.2)
    ax.axvline(t_all[n_transient_steps], color='gray', linestyle=':',
               label=f'Transient end ({transient}ms)')
    ax.set_ylabel('I_syn (arb. units)')
    ax.set_title('Pre-training: Synaptic Currents')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 3) Synaptic conductances g_syn
    ax = axes[2]
    for syn_name in syn_names:
        entry = graph._synapses[syn_name]
        label = f'{syn_name}  ({entry.src}→{entry.tgt})'
        ax.plot(t_all, syn_g[syn_name], label=label, linewidth=1.2)
    ax.axvline(t_all[n_transient_steps], color='gray', linestyle=':',
               label=f'Transient end ({transient}ms)')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('g_syn (arb. units)')
    ax.set_title('Pre-training: Synaptic Conductances')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show(block=True)

except ImportError:
    print("Matplotlib not available")
except Exception as e:
    print(f"Pre-training visualization error: {e}")


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
    # import matplotlib
    # matplotlib.use('Agg')
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
    plt.show()
    #plt.close()

except ImportError:
    print("Matplotlib not available")
