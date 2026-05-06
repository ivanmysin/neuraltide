"""
Example: Two mutually inhibiting populations with anti-phase loss.

Architecture:
- Two homogeneous IzhikevichMeanField populations (Pop1 and Pop2)
- Mutual inhibition via TsodyksMarkramSynapse (short-term depression)
- External excitatory current to each population

Loss: AntiPhaseLoss — drives populations toward anti-correlation
      while maintaining minimum activity.

Trainable parameters:
- gsyn_max (both synapses), tau_d, tau_r, tau_f, Uinc
- I_ext (both populations)
"""

import os
import sys
import time
from copy import deepcopy

import numpy as np
import tensorflow as tf

# sys.path.append('../')

from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.integrators import RK4Integrator
from neuraltide.populations import IzhikevichMeanField
from neuraltide.synapses import TsodyksMarkramSynapse
from neuraltide.training import (
    Trainer,
    CompositeLoss,
    AntiPhaseLoss,
    StabilityPenalty,
)

# ── Device selection ────────────────────────────────────────────────
print("=" * 60)
print("Device detection")

gpu_devices = tf.config.list_physical_devices("GPU")
cpu_devices = tf.config.list_physical_devices("CPU")

print(f"  CPUs visible: {len(cpu_devices)}")

if gpu_devices:
    print(f"  GPUs visible: {len(gpu_devices)}")
    for i, gpu in enumerate(gpu_devices):
        details = tf.config.experimental.get_device_details(gpu)
        name = details.get("device_name", gpu.name)
        mem_mb = details.get("memory_limit", None)
        mem_str = f", {mem_mb // (1024*1024)} MiB" if mem_mb else ""
        print(f"    [{i}] {name}{mem_str}")

    # Allow memory growth — avoids grabbing all VRAM at once
    for gpu in gpu_devices:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    device_type = "GPU"
    print("  → Running on GPU")
else:
    device_type = "CPU"
    print("  → Running on CPU")

# Force tf.function compilation on the selected device
if device_type == "GPU":
    logical_gpus = tf.config.list_logical_devices("GPU")
    if logical_gpus:
        device_name = logical_gpus[0].name.replace("physical_device:", "")
        print(f"  → Logical device: {device_name}")
else:
    logical_cpus = tf.config.list_logical_devices("CPU")
    if logical_cpus:
        print(f"  → Logical device: {logical_cpus[0].name.replace('physical_device:', '')}")

# Print TF build info
print(f"  TF version: {tf.__version__}")
print(f"  tf.config.list_logical_devices():")
for d in tf.config.list_logical_devices():
    print(f"    {d.name}  ({d.device_type})")
print("=" * 60)
sys.stdout.flush()

# ── Simulation parameters ──────────────────────────────────────────
dt = 0.1
T_total = 1000
transient = 200
nepochs = 80
log_every = 5

n_transient_steps = int(transient / dt)

t_all = np.arange(0, T_total, dt)
t_seq_full = tf.constant(t_all[None, :, None], dtype=tf.float32)

print(f"Total: T={T_total}ms, dt={dt}ms, steps={t_all.shape[0]}, "
      f"transient={transient}ms, epochs={nepochs}")

# ── Populations ────────────────────────────────────────────────────
pop_params_base = {
    'tau_pop': {'value': [1.6622994, ], 'trainable': False},
    'alpha':   {'value': [0.38348085, ], 'trainable': False},
    'a':       {'value': [0.0083115, ], 'trainable': False},
    'b':       {'value': [0.00320795, ], 'trainable': False},
    'w_jump':  {'value': [0.00050604, ], 'trainable': False},
    'Delta_I': {'value': [0.00632551, ], 'trainable': False,
                'min': 0.01, 'max': 2.0},
    'I_ext':   {'value': [0.5, ], 'trainable': True,
                'min': -20.0, 'max': 20.0},
}

pop_params_1 = deepcopy(pop_params_base)
pop_params_2 = deepcopy(pop_params_base)
pop_params_2['I_ext']['value'] = [0.3]

pop1 = IzhikevichMeanField(dt=dt, params=pop_params_1, name='pop1')
pop2 = IzhikevichMeanField(dt=dt, params=pop_params_2, name='pop2')

# ── Synapses ───────────────────────────────────────────────────────
syn_params_base = {
    'gsyn_max': {'value': 2.5, 'trainable': True, 'min': 0.0, 'max': 100.0},
    'tau_d':    {'value': 6.02, 'trainable': True, 'min': 2.0, 'max': 15.0},
    'tau_r':    {'value': 200.0, 'trainable': True, 'min': 50.0, 'max': 500.0},
    'tau_f':    {'value': 20.0, 'trainable': True, 'min': 5.0, 'max': 100.0},
    'Uinc':     {'value': 0.3, 'trainable': True, 'min': 0.1, 'max': 0.6},
    'pconn':    {'value': 1.0, 'trainable': False},
    'e_r':      {'value': -0.25, 'trainable': False},
}

syn_params_1 = deepcopy(syn_params_base)
syn_params_2 = deepcopy(syn_params_base)
syn_params_2['gsyn_max']['value'] = 30.5

syn_1to2 = TsodyksMarkramSynapse(n_pre=1, n_post=1, dt=dt,
                                  params=syn_params_1, name='syn_1to2')
syn_2to1 = TsodyksMarkramSynapse(n_pre=1, n_post=1, dt=dt,
                                  params=syn_params_2, name='syn_2to1')

# ── Graph ──────────────────────────────────────────────────────────
graph = NetworkGraph(dt=dt)
graph.add_population('pop1', pop1)
graph.add_population('pop2', pop2)
graph.add_synapse('pop1->pop2', syn_1to2, src='pop1', tgt='pop2')
graph.add_synapse('pop2->pop1', syn_2to1, src='pop2', tgt='pop1')

network = NetworkRNN(graph, integrator=RK4Integrator(),
                     stability_penalty_weight=1e-2)

# ── Loss ───────────────────────────────────────────────────────────
loss_fn = CompositeLoss([
    (1.0, AntiPhaseLoss(
        pop_pairs=[('pop1', 'pop2')],
        target_correlation=-1.0,
        correlation_weight=1.0,
        activity_target=20.0,
        activity_weight=0.5,
        transient_steps=n_transient_steps,
    )),
    (1e-3, StabilityPenalty()),
])

# ── Trainer ────────────────────────────────────────────────────────
optimizer = tf.keras.optimizers.Adam(1e-2)
trainer = Trainer(network, loss_fn, optimizer, grad_method='bptt')

# ── Transient ──────────────────────────────────────────────────────
print("\n=== Transient simulation ===")
t_transient = tf.constant(t_all[:n_transient_steps][None, :, None],
                          dtype=tf.float32)
transient_output = network(t_transient, training=False)
init_state = transient_output.final_state
print(f"Transient done: pop1 r={float(init_state[0][0][0, 0]):.4f}")

# ── Pre-training analysis ──────────────────────────────────────────
def _extend_syn_currents(network, graph, t_seq, init_state, t_all,
                         n_transient_steps):
    """Compute I_syn and g_syn from stacked states for visualization."""
    _, _, _, _, all_pop_stacked, all_syn_stacked = \
        network._scan_forward_states(t_seq,
                                     tuple(init_state[0]),
                                     tuple(init_state[1]))
    syn_names = graph.synapse_names
    dynamic_names = graph.dynamic_population_names

    # Index ranges
    pop_state_ranges = {}
    idx = 0
    for name in graph.population_names:
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

    # Population v_mean
    pop_v_arrays = {}
    for name in dynamic_names:
        start, _ = pop_state_ranges[name]
        v_stacked = all_pop_stacked[start + 1].numpy()  # [T, 1, n_units]
        pop_v_arrays[name] = v_stacked[:, 0, :]          # [T, n_units]

    # I_syn and g_syn per synapse
    T_steps = t_all.shape[0]
    syn_I = {}
    syn_g = {}
    for syn_name in syn_names:
        entry = graph._synapses[syn_name]
        m = entry.model
        tgt_name = entry.tgt
        start, _ = syn_state_ranges[syn_name]
        A_stacked = all_syn_stacked[start + 2].numpy()  # [T, n_pre, n_post]
        gsyn_max = m.gsyn_max.numpy()                   # [n_pre, n_post]
        e_r = m.e_r.numpy()                             # [n_pre, n_post]
        post_v = pop_v_arrays[tgt_name]                 # [T, n_post]

        g_syn_arr = np.zeros(T_steps, dtype=np.float32)
        I_syn_arr = np.zeros(T_steps, dtype=np.float32)
        for t in range(T_steps):
            g_eff   = gsyn_max * A_stacked[t]           # [n_pre, n_post]
            g_syn_t = np.sum(g_eff, axis=0)              # [n_post]
            I_pair  = g_eff * (e_r - post_v[t][np.newaxis, :])
            I_syn_t = np.sum(I_pair, axis=0)             # [n_post]
            g_syn_arr[t] = np.sum(g_syn_t)
            I_syn_arr[t] = np.sum(I_syn_t)
        syn_I[syn_name] = I_syn_arr
        syn_g[syn_name] = g_syn_arr

    return syn_I, syn_g

def _run_and_collect(network, graph, t_seq, init_state, t_all,
                     n_transient_steps):
    """Run simulation and collect rates, I_syn, g_syn."""
    output = network(t_seq, training=False, initial_state=init_state)
    rates = {}
    for name in graph.dynamic_population_names:
        rates[name] = output.firing_rates[name].numpy()[0, :, 0]

    syn_I, syn_g = _extend_syn_currents(
        network, graph, t_seq, init_state, t_all, n_transient_steps)

    return rates, syn_I, syn_g

def _compute_correlation(rates, pop_pairs, n_skip):
    corrs = {}
    for a, b in pop_pairs:
        ra = rates[a][n_skip:]
        rb = rates[b][n_skip:]
        ra_c = ra - ra.mean()
        rb_c = rb - rb.mean()
        cov = np.mean(ra_c * rb_c)
        std = np.std(ra) * np.std(rb) + 1e-8
        corrs[f'{a}↔{b}'] = cov / std
    return corrs

print("\n=== Pre-training analysis ===")
pre_rates, pre_I, pre_g = _run_and_collect(
    network, graph, t_seq_full, init_state, t_all, n_transient_steps)
pre_corrs = _compute_correlation(pre_rates, [('pop1', 'pop2')],
                                  n_transient_steps)
print(f"Correlation: {pre_corrs}")
for name in graph.dynamic_population_names:
    print(f"  {name} mean rate: "
          f"{pre_rates[name][n_transient_steps:].mean():.2f} Hz")

# ── Optimization ───────────────────────────────────────────────────
print(f"\n=== Optimization: {nepochs} epochs ===")
total_start = time.time()
history = {'loss': [], 'corr': [], 'rate_pop1': [], 'rate_pop2': []}

for epoch in range(nepochs):
    epoch_start = time.time()
    result = trainer.train_step(t_seq_full, initial_state=init_state)
    epoch_time = time.time() - epoch_start
    loss_val = float(result['loss'])

    # Periodic evaluation
    if (epoch + 1) % log_every == 0 or epoch == 0:
        eval_rates, _, _ = _run_and_collect(
            network, graph, t_seq_full, init_state, t_all,
            n_transient_steps)
        eval_corrs = _compute_correlation(
            eval_rates, [('pop1', 'pop2')], n_transient_steps)
        r1 = float(eval_rates['pop1'][n_transient_steps:].mean())
        r2 = float(eval_rates['pop2'][n_transient_steps:].mean())

        history['loss'].append(loss_val)
        history['corr'].append(eval_corrs.get('pop1↔pop2', 0))
        history['rate_pop1'].append(r1)
        history['rate_pop2'].append(r2)

        print(f"Epoch {epoch+1:3d}/{nepochs} | loss: {loss_val:.4f} | "
              f"corr: {eval_corrs.get('pop1↔pop2', 0):+.3f} | "
              f"r1: {r1:.1f} r2: {r2:.1f} | {epoch_time:.1f}s")

total_time = time.time() - total_start
print(f"\nTotal optimization time: {total_time:.1f}s "
      f"({total_time/nepochs:.2f}s/epoch)")

# ── Final simulation ───────────────────────────────────────────────
print("\n=== Final simulation ===")
final_rates, final_I, final_g = _run_and_collect(
    network, graph, t_seq_full, init_state, t_all, n_transient_steps)

n_eval = n_transient_steps
for name in graph.dynamic_population_names:
    print(f"  {name} mean rate: "
          f"{final_rates[name][n_eval:].mean():.2f} Hz")

final_corrs = _compute_correlation(final_rates, [('pop1', 'pop2')], n_eval)
print(f"  Correlation (after transient): "
      f"{final_corrs.get('pop1↔pop2', 0):+.3f}")

# ── Optimized parameters ───────────────────────────────────────────
print("\n=== Optimized parameters ===")
for v in network.trainable_variables:
    print(f"  {v.name.split(':')[0]}: {float(v.numpy().flatten()[0]):.4f}")

# ── Export ─────────────────────────────────────────────────────────
trainer.export_results('optimization_results.json')
trainer.export_results('optimization_results.csv', format='csv')
print("\nResults exported to optimization_results.json/.csv")

# ── Visualization ──────────────────────────────────────────────────
print("\n=== Visualization ===")
try:
    import matplotlib.pyplot as plt
    plt.ion()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # (0,0) — Pre-training firing rates
    ax = axes[0, 0]
    for name in graph.dynamic_population_names:
        ax.plot(t_all, pre_rates[name], label=name, linewidth=1.0)
    ax.axvline(t_all[n_transient_steps], color='gray', linestyle=':',
               label='transient end')
    ax.set_title(f'Pre-training: Firing Rates  (corr={pre_corrs.get("pop1↔pop2",0):+.2f})')
    ax.set_ylabel('Firing Rate (Hz)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (0,1) — Pre-training synaptic currents
    ax = axes[0, 1]
    for sn in graph.synapse_names:
        entry = graph._synapses[sn]
        ax.plot(t_all, pre_I[sn], label=f'{sn} ({entry.src}→{entry.tgt})',
                linewidth=1.0)
    ax.axvline(t_all[n_transient_steps], color='gray', linestyle=':')
    ax.set_title('Pre-training: I_syn')
    ax.set_ylabel('I_syn')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (0,2) — Pre-training synaptic conductances
    ax = axes[0, 2]
    for sn in graph.synapse_names:
        entry = graph._synapses[sn]
        ax.plot(t_all, pre_g[sn], label=f'{sn} ({entry.src}→{entry.tgt})',
                linewidth=1.0)
    ax.axvline(t_all[n_transient_steps], color='gray', linestyle=':')
    ax.set_title('Pre-training: g_syn')
    ax.set_ylabel('g_syn')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,0) — Post-training firing rates
    ax = axes[1, 0]
    for name in graph.dynamic_population_names:
        ax.plot(t_all, final_rates[name], label=name, linewidth=1.0)
    ax.axvline(t_all[n_transient_steps], color='gray', linestyle=':')
    ax.set_title(f'Post-training: Firing Rates  '
                 f'(corr={final_corrs.get("pop1↔pop2",0):+.2f})')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Firing Rate (Hz)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,1) — Post-training synaptic currents
    ax = axes[1, 1]
    for sn in graph.synapse_names:
        entry = graph._synapses[sn]
        ax.plot(t_all, final_I[sn], label=f'{sn} ({entry.src}→{entry.tgt})',
                linewidth=1.0)
    ax.axvline(t_all[n_transient_steps], color='gray', linestyle=':')
    ax.set_title('Post-training: I_syn')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('I_syn')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,2) — Training curves
    ax = axes[1, 2]
    epochs_ix = np.arange(1, len(history['loss']) + 1) * log_every

    ax_loss = ax
    ax_corr = ax.twinx()

    line1, = ax_loss.plot(epochs_ix, history['loss'], 'b.-',
                           label='Loss', linewidth=1.0)
    line2, = ax_corr.plot(epochs_ix, history['corr'], 'r.-',
                           label='Correlation', linewidth=1.0)
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss', color='b')
    ax_corr.set_ylabel('Corr (pop1↔pop2)', color='r')
    ax_loss.tick_params(axis='y', labelcolor='b')
    ax_corr.tick_params(axis='y', labelcolor='r')
    ax.set_title('Training Progress')
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('optimization_visualization.png', dpi=150)
    print("Saved to optimization_visualization.png")
    plt.show(block=True)

except ImportError:
    print("Matplotlib not available")
