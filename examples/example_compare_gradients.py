"""
Example: Compare gradients computed via BPTT vs Adjoint methods.

This example demonstrates:
1. Creating a simple network
2. Computing gradients via standard BPTT (tf.GradientTape)
3. Computing gradients via AdjointSolver
4. Comparing the two methods numerically and visually
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import neuraltide
from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.populations import IzhikevichMeanField
from neuraltide.synapses import TsodyksMarkramSynapse
from neuraltide.inputs import VonMisesGenerator
from neuraltide.integrators import RK4Integrator
from neuraltide.training import Trainer, MSELoss, StabilityPenalty, CompositeLoss
from neuraltide.training.adjoint import AdjointSolver
from neuraltide.utils import seed_everything, print_summary


dt = 0.1
T = 500
n_steps = int(T / dt)

pop = IzhikevichMeanField(dt=dt, params={
    'tau_pop':   {'value': [1.0, 1.0],   'trainable': False},
    'alpha':     {'value': [0.5, 0.5],   'trainable': False},
    'a':         {'value': [0.02, 0.02], 'trainable': False},
    'b':         {'value': [0.2, 0.2],   'trainable': False},
    'w_jump':    {'value': [0.1, 0.1],   'trainable': False},
    'Delta_I':   {'value': [0.05, 0.05],   'trainable': True,
                  'min': 0.01, 'max': 2.0},
    'I_ext':     {'value': [0.1, 0.1],   'trainable': True,
                  'min': -2.0, 'max': 2.0},
})

gen = VonMisesGenerator(
    dt=dt,
    params={
        'mean_rate': 20.0,
        'R': 0.5,
        'freq': 8.0,
        'phase': 0.0,
    },
    name='theta_gen'
)

syn = TsodyksMarkramSynapse(n_pre=1, n_post=2, dt=dt, params={
    'gsyn_max': {'value': [[100.1, 100.1]], 'trainable': True},
    'tau_f':    {'value': 20.0,  'trainable': False},
    'tau_d':    {'value': 5.0,   'trainable': False},
    'tau_r':    {'value': 200.0, 'trainable': False},
    'Uinc':     {'value': 0.2,   'trainable': False},
    'pconn':    {'value': [[1.0, 1.0]], 'trainable': False},
    'e_r':      {'value': 1.0,   'trainable': False},
})

graph = NetworkGraph(dt=dt)
graph.add_input_population('theta', gen)
graph.add_population('exc', pop)
graph.add_synapse('theta->exc', syn, src='theta', tgt='exc')

network = NetworkRNN(graph, integrator=RK4Integrator())
print_summary(network)

t_values = np.arange(n_steps, dtype=np.float32) * dt
t_seq = tf.constant(t_values[None, :, None])

target_0 = 10.0 + 5.0 * np.sin(2 * np.pi * 8.0 * t_values / 1000.0)
target_1 = 8.0 + 4.0 * np.sin(2 * np.pi * 8.0 * t_values / 1000.0 + 0.5)
target = {
    'exc': tf.constant(
        np.stack([target_0, target_1], axis=-1)[None, :, :],
        dtype=tf.float32
    )
}

print("\n" + "="*60)
print("Computing gradients via BPTT...")
print("="*60)

bptt_loss_fn = MSELoss(target)

with tf.GradientTape() as bptt_tape:
    bptt_output = network(t_seq, training=False)
    bptt_loss = bptt_loss_fn(bptt_output, network)

bptt_grads = bptt_tape.gradient(bptt_loss, network.trainable_variables)
bptt_loss_val = float(bptt_loss)

print(f"BPTT Loss: {bptt_loss_val:.6f}")
print("\nBPTT Gradients:")
for v, g in zip(network.trainable_variables, bptt_grads):
    if g is not None:
        print(f"  {v.name}: {np.mean(np.abs(g.numpy())):.6f}")

print("\n" + "="*60)
print("Computing gradients via Adjoint...")
print("="*60)

adjoint_solver = AdjointSolver(network, network._integrator)
adjoint_loss_fn = MSELoss(target)

adj_grads, variables, adj_output = adjoint_solver.compute_gradients(t_seq, target, adjoint_loss_fn)
adj_loss_val = float(adjoint_loss_fn(adj_output, network))

print(f"Adjoint Loss: {adj_loss_val:.6f}")
print("\nAdjoint Gradients:")
for v, g in zip(variables, adj_grads):
    if g is not None:
        print(f"  {v.name}: {np.mean(np.abs(g.numpy())):.6f}")

print("\n" + "="*60)
print("Comparison")
print("="*60)

print("\nGradient difference (|BPTT - Adjoint|):")
max_diff = 0.0
for v, bg, ag in zip(network.trainable_variables, bptt_grads, adj_grads):
    if bg is not None and ag is not None:
        diff = np.mean(np.abs(bg.numpy() - ag.numpy()))
        max_diff = max(max_diff, diff)
        rel_diff = diff / (np.mean(np.abs(bg.numpy())) + 1e-8) * 100
        print(f"  {v.name}:")
        print(f"    Absolute: {diff:.8f}")
        print(f"    Relative: {rel_diff:.4f}%")

print(f"\nMax gradient difference: {max_diff:.8f}")

print("\n" + "="*60)
print("Visualization")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(t_values, target_0, 'g-', linewidth=2, label='Target 0')
axes[0, 0].plot(t_values, bptt_output.firing_rates['exc'].numpy()[0, :, 0], 
               'b--', linewidth=1.5, label='BPTT')
axes[0, 0].plot(t_values, adj_output.firing_rates['exc'].numpy()[0, :, 0], 
               'r:', linewidth=1.5, label='Adjoint')
axes[0, 0].set_xlabel("Time (ms)")
axes[0, 0].set_ylabel("Firing Rate (Hz)")
axes[0, 0].set_title("Population 0: Target vs Predictions")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(t_values, target_1, 'g-', linewidth=2, label='Target 1')
axes[0, 1].plot(t_values, bptt_output.firing_rates['exc'].numpy()[0, :, 1], 
               'b--', linewidth=1.5, label='BPTT')
axes[0, 1].plot(t_values, adj_output.firing_rates['exc'].numpy()[0, :, 1], 
               'r:', linewidth=1.5, label='Adjoint')
axes[0, 1].set_xlabel("Time (ms)")
axes[0, 1].set_ylabel("Firing Rate (Hz)")
axes[0, 1].set_title("Population 1: Target vs Predictions")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

var_names = [v.name.split(':')[0] for v in network.trainable_variables]
bptt_grad_norms = [np.mean(np.abs(g.numpy())) if g is not None else 0 for g in bptt_grads]
adj_grad_norms = [np.mean(np.abs(g.numpy())) if g is not None else 0 for g in adj_grads]

x = np.arange(len(var_names))
width = 0.35
axes[1, 0].bar(x - width/2, bptt_grad_norms, width, label='BPTT', color='blue', alpha=0.7)
axes[1, 0].bar(x + width/2, adj_grad_norms, width, label='Adjoint', color='red', alpha=0.7)
axes[1, 0].set_xlabel("Variable")
axes[1, 0].set_ylabel("Mean |Gradient|")
axes[1, 0].set_title("Gradient Magnitudes Comparison")
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels([n[:15] for n in var_names], rotation=45, ha='right')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3, axis='y')

grad_diffs = [np.mean(np.abs(bg.numpy() - ag.numpy())) if bg is not None and ag is not None else 0 
             for bg, ag in zip(bptt_grads, adj_grads)]
axes[1, 1].bar(var_names, grad_diffs, color='purple', alpha=0.7)
axes[1, 1].set_xlabel("Variable")
axes[1, 1].set_ylabel("|BPTT - Adjoint|")
axes[1, 1].set_title("Gradient Difference")
axes[1, 1].set_xticks(range(len(var_names)))
axes[1, 1].set_xticklabels([n[:15] for n in var_names], rotation=45, ha='right')
axes[1, 1].grid(True, alpha=0.3, axis='y')

fig.suptitle("BPTT vs Adjoint: Gradient Comparison", fontsize=14)
plt.tight_layout()
plt.savefig("example_compare_gradients.png", dpi=150)
plt.show()
print("\nFigure saved as example_compare_gradients.png")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"BPTT Loss: {bptt_loss_val:.6f}")
print(f"Adjoint Loss: {adj_loss_val:.6f}")
print(f"Max Gradient Diff: {max_diff:.8f}")
print("Results match!" if max_diff < 1e-3 else "WARNING: Large gradient difference!")