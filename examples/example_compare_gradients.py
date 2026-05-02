"""
Example: Compare gradients computed via BPTT vs Adjoint (discrete adjoint backward pass).

This example demonstrates:
1. Creating a network with trainable parameters
2. Computing gradients via standard BPTT (tf.GradientTape over entire sequence)
3. Computing gradients via discrete adjoint backward_pass (O(1) graph memory)
4. Comparing time, correctness, and memory footprint

Key difference from BPTT:
  BPTT:    stores TF computational graph for all T steps simultaneously
  Adjoint: stores states_sequence (plain tensors), rebuilds local graph one step at a time.
           Peak vRAM grows as O(1) steps, not O(T).
"""
import gc
import time
import tracemalloc

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.populations import IzhikevichMeanField
from neuraltide.synapses import TsodyksMarkramSynapse
from neuraltide.inputs import VonMisesGenerator
from neuraltide.integrators import RK4Integrator
from neuraltide.training import MSELoss, StabilityPenalty, CompositeLoss
from neuraltide.training.adjoint import AdjointSolver
from neuraltide.utils import print_summary


# ── Network configuration ─────────────────────────────────────────────────────
dt = 0.1     # ms
T  = 1000.0   # ms  (100 steps)
n_steps = int(T / dt)

pop = IzhikevichMeanField(dt=dt, params={
    'tau_pop':   {'value': [1.0, 1.0],   'trainable': False},
    'alpha':     {'value': [0.5, 0.5],   'trainable': False},
    'a':         {'value': [0.02, 0.02], 'trainable': False},
    'b':         {'value': [0.2, 0.2],   'trainable': False},
    'w_jump':    {'value': [0.1, 0.1],   'trainable': False},
    'Delta_I':   {'value': [0.5, 0.5],   'trainable': True,
                  'min': 0.01, 'max': 2.0},
    'I_ext':     {'value': [0.1, 0.1],   'trainable': True,
                  'min': -2.0, 'max': 2.0},
})

gen = VonMisesGenerator(
    dt=dt,
    params={'mean_rate': 20.0, 'R': 0.5, 'freq': 8.0, 'phase': 0.0},
    name='theta_gen',
)

syn = TsodyksMarkramSynapse(n_pre=1, n_post=2, dt=dt, params={
    'gsyn_max': {'value': [[0.1, 0.1]], 'trainable': True},
    'tau_f':    {'value': 20.0,  'trainable': True, 'min': 6.0,  'max': 240.0},
    'tau_d':    {'value': 5.0,   'trainable': True, 'min': 2.0,  'max': 15.0},
    'tau_r':    {'value': 200.0, 'trainable': False},
    'Uinc':     {'value': 0.2,   'trainable': False},
    'pconn':    {'value': [[1.0, 1.0]], 'trainable': False},
    'e_r':      {'value': 0.0,   'trainable': False},
})

graph = NetworkGraph(dt=dt)
graph.add_input_population('theta', gen)
graph.add_population('exc', pop)
graph.add_synapse('theta->exc', syn, src='theta', tgt='exc')

network = NetworkRNN(graph, integrator=RK4Integrator(), stability_penalty_weight=1e-3)
print_summary(network)
print(f"\nSequence length T={T}ms, {n_steps} steps, {len(network.trainable_variables)} trainable variables\n")

# ── Targets ───────────────────────────────────────────────────────────────────
t_values = np.arange(n_steps, dtype=np.float32) * dt
t_seq    = tf.constant(t_values[None, :, None])

target_0 = 10.0 + 5.0 * np.sin(2 * np.pi * 8.0 * t_values / 1000.0)
target_1 =  8.0 + 4.0 * np.sin(2 * np.pi * 8.0 * t_values / 1000.0 + 0.5)
target = {
    'exc': tf.constant(
        np.stack([target_0, target_1], axis=-1)[None, :, :],
        dtype=tf.float32,
    )
}

loss_fn = CompositeLoss([
    (1.0,   MSELoss(target)),
    (1e-3,  StabilityPenalty()),
])


# ── Helper: peak RAM via tracemalloc ─────────────────────────────────────────
def measure_peak_ram_mb(fn):
    """Run fn(), return (result, peak_ram_MB)."""
    gc.collect()
    tracemalloc.start()
    result = fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, peak / 1e6


# ── Helper: GPU memory (only on GPU builds) ───────────────────────────────────
def gpu_mem_mb():
    try:
        return tf.config.experimental.get_memory_info('GPU:0')['current'] / 1e6
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 1.  BPTT
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Method 1: BPTT  (tf.GradientTape over all T steps)")
print("=" * 60)

def run_bptt():
    with tf.GradientTape() as tape:
        out   = network(t_seq, training=False)
        loss  = loss_fn(out, network)
    grads = tape.gradient(loss, network.trainable_variables)
    return grads, out, float(loss)

# warm-up (TF graph tracing)
_ = run_bptt()
gc.collect()

gpu_before_bptt = gpu_mem_mb()
t0 = time.perf_counter()
(bptt_grads, bptt_output, bptt_loss_val), bptt_peak_ram = measure_peak_ram_mb(run_bptt)
bptt_time = time.perf_counter() - t0
gpu_after_bptt = gpu_mem_mb()

print(f"  Loss  : {bptt_loss_val:.6f}")
print(f"  Time  : {bptt_time:.3f}s")
print(f"  Peak RAM (Python heap) : {bptt_peak_ram:.1f} MB")
if gpu_before_bptt is not None:
    print(f"  GPU memory : {gpu_after_bptt:.1f} MB")
print("\n  Gradients (mean |g|):")
for v, g in zip(network.trainable_variables, bptt_grads):
    if g is not None:
        print(f"    {v.name}: {float(np.mean(np.abs(g.numpy()))):.6f}")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Discrete Adjoint backward_pass
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Method 2: Discrete Adjoint backward_pass")
print("  Forward : stores states_sequence (plain tensors, no TF graph)")
print("  Backward: one local GradientTape per step, deleted after each step")
print("=" * 60)

solver = AdjointSolver(network, network._integrator)

def run_adjoint():
    grads, variables, out = solver.compute_gradients(t_seq, target, loss_fn)
    return grads, variables, out, float(loss_fn(out, network))

# warm-up
_ = run_adjoint()
gc.collect()

gpu_before_adj = gpu_mem_mb()
t0 = time.perf_counter()
(adj_grads, variables, adj_output, adj_loss_val), adj_peak_ram = measure_peak_ram_mb(run_adjoint)
adj_time = time.perf_counter() - t0
gpu_after_adj = gpu_mem_mb()

print(f"  Loss  : {adj_loss_val:.6f}")
print(f"  Time  : {adj_time:.3f}s")
print(f"  Peak RAM (Python heap) : {adj_peak_ram:.1f} MB")
if gpu_before_adj is not None:
    print(f"  GPU memory : {gpu_after_adj:.1f} MB")
print("\n  Gradients (mean |g|):")
for v, g in zip(variables, adj_grads):
    if g is not None:
        print(f"    {v.name}: {float(np.mean(np.abs(g.numpy()))):.6f}")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Comparison
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Comparison")
print("=" * 60)

max_abs_diff = 0.0
max_rel_diff = 0.0

print(f"\n{'Variable':<20} {'|BPTT-Adj| abs':>15} {'rel %':>10}")
print("-" * 48)
for v, bg, ag in zip(network.trainable_variables, bptt_grads, adj_grads):
    if bg is None or ag is None:
        continue
    abs_diff = float(np.mean(np.abs(bg.numpy() - ag.numpy())))
    bptt_mag = float(np.mean(np.abs(bg.numpy()))) + 1e-12
    rel_diff = abs_diff / bptt_mag * 100.0
    max_abs_diff = max(max_abs_diff, abs_diff)
    max_rel_diff = max(max_rel_diff, rel_diff)
    print(f"  {v.name:<18} {abs_diff:>15.2e} {rel_diff:>9.4f}%")

print()
print(f"  Max absolute gradient difference : {max_abs_diff:.2e}")
print(f"  Max relative gradient difference : {max_rel_diff:.4f}%")
print()
print(f"  BPTT time        : {bptt_time:.3f}s")
print(f"  Adjoint time     : {adj_time:.3f}s")
print(f"  Time ratio       : {adj_time / bptt_time:.2f}x  (Adjoint / BPTT)")
print()
print(f"  BPTT peak RAM    : {bptt_peak_ram:.1f} MB")
print(f"  Adjoint peak RAM : {adj_peak_ram:.1f} MB")
ram_ratio = adj_peak_ram / (bptt_peak_ram + 1e-9)
print(f"  RAM ratio        : {ram_ratio:.2f}x  (Adjoint / BPTT)")
print()
if max_abs_diff < 1e-3:
    print("  [OK] Gradients match (|diff| < 1e-3)")
else:
    print(f"  [WARN] Large gradient difference: {max_abs_diff:.2e}")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Visualization
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

# — firing rate traces —
for unit, (tgt, lbl_t) in enumerate(
    zip([target_0, target_1], ['Target 0', 'Target 1'])
):
    ax = axes[0, unit]
    ax.plot(t_values, tgt, 'g-', lw=2, label=lbl_t)
    ax.plot(t_values, bptt_output.firing_rates['exc'].numpy()[0, :, unit],
            'b--', lw=1.5, label='BPTT')
    ax.plot(t_values, adj_output.firing_rates['exc'].numpy()[0, :, unit],
            'r:', lw=1.5, label='Adjoint')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Firing Rate (Hz)')
    ax.set_title(f'Population {unit}: firing rates')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# — gradient magnitudes bar chart —
ax = axes[0, 2]
var_labels = [v.name[:16] for v in network.trainable_variables]
bptt_norms = [float(np.mean(np.abs(g.numpy()))) if g is not None else 0.0
              for g in bptt_grads]
adj_norms  = [float(np.mean(np.abs(g.numpy()))) if g is not None else 0.0
              for g in adj_grads]
x = np.arange(len(var_labels))
w = 0.35
ax.bar(x - w/2, bptt_norms, w, label='BPTT',    color='blue',  alpha=0.75)
ax.bar(x + w/2, adj_norms,  w, label='Adjoint', color='red',   alpha=0.75)
ax.set_xticks(x)
ax.set_xticklabels(var_labels, rotation=40, ha='right', fontsize=8)
ax.set_ylabel('Mean |gradient|')
ax.set_title('Gradient magnitudes')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

# — absolute difference bar chart —
ax = axes[1, 0]
abs_diffs = [
    float(np.mean(np.abs(bg.numpy() - ag.numpy())))
    if bg is not None and ag is not None else 0.0
    for bg, ag in zip(bptt_grads, adj_grads)
]
ax.bar(x, abs_diffs, color='purple', alpha=0.75)
ax.set_xticks(x)
ax.set_xticklabels(var_labels, rotation=40, ha='right', fontsize=8)
ax.set_ylabel('|BPTT − Adjoint|')
ax.set_title('Gradient absolute difference')
ax.grid(True, alpha=0.3, axis='y')

# — time comparison —
ax = axes[1, 1]
ax.bar(['BPTT', 'Adjoint'], [bptt_time, adj_time],
       color=['blue', 'red'], alpha=0.75)
ax.set_ylabel('Time (s)')
ax.set_title(f'Computation time  (T={n_steps} steps)')
for i, v in enumerate([bptt_time, adj_time]):
    ax.text(i, v + 0.01, f'{v:.2f}s', ha='center', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# — peak RAM comparison —
ax = axes[1, 2]
ax.bar(['BPTT', 'Adjoint'], [bptt_peak_ram, adj_peak_ram],
       color=['blue', 'red'], alpha=0.75)
ax.set_ylabel('Peak RAM (MB, Python heap)')
ax.set_title('Peak memory usage')
for i, v in enumerate([bptt_peak_ram, adj_peak_ram]):
    ax.text(i, v + 0.01, f'{v:.1f}MB', ha='center', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

fig.suptitle(
    f'BPTT vs Discrete Adjoint — T={n_steps} steps, {len(network.trainable_variables)} params',
    fontsize=13,
)
plt.tight_layout()
plt.savefig('example_compare_gradients.png', dpi=150)
plt.show()
print('\nFigure saved: example_compare_gradients.png')

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Summary
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '=' * 60)
print('SUMMARY')
print('=' * 60)
print(f'  Network          : {len(network.trainable_variables)} trainable parameters')
print(f'  Sequence length  : T={n_steps} steps ({T} ms)')
print()
print(f'  BPTT   loss  : {bptt_loss_val:.6f}  time: {bptt_time:.3f}s  RAM: {bptt_peak_ram:.1f}MB')
print(f'  Adjoint loss : {adj_loss_val:.6f}  time: {adj_time:.3f}s  RAM: {adj_peak_ram:.1f}MB')
print()
print(f'  Max gradient diff: {max_abs_diff:.2e}  '
      f'({"MATCH" if max_abs_diff < 1e-3 else "MISMATCH"})')
print()
print('  Memory note:')
print('    BPTT stores the full TF computational graph (scales with T).')
print('    Adjoint stores states_sequence (plain tensors) + one local step graph.')
print('    On GPU the vRAM difference is significant for large T.')
