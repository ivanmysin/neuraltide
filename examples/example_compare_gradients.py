"""
Пример: Сравнение autograd и adjoint методов.

Запускать: python examples/example_compare_gradients.py
"""
import numpy as np
import tensorflow as tf
from neuraltide.training.adjoint import AdjointGradientComputer
from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.populations import IzhikevichMeanField
from neuraltide.synapses import TsodyksMarkramSynapse
from neuraltide.inputs import VonMisesGenerator
from neuraltide.integrators import RK4Integrator

dt = 0.05
T = 100

print("=" * 60)
print("Пример: Сравнение градиентов autograd vs adjoint")
print(f"Параметры: dt={dt}, T={T}")
print("=" * 60)

pop = IzhikevichMeanField(dt=dt, params={
    'tau_pop': {'value': [1.0], 'trainable': False},
    'alpha': {'value': [0.5], 'trainable': False},
    'a': {'value': [0.02], 'trainable': False},
    'b': {'value': [0.2], 'trainable': False},
    'w_jump': {'value': [0.1], 'trainable': False},
    'Delta_I': {'value': [0.5], 'trainable': True},
    'I_ext': {'value': [1.0], 'trainable': True},
})

syn = TsodyksMarkramSynapse(n_pre=1, n_post=1, dt=dt, params={
    'gsyn_max': {'value': 0.1, 'trainable': True},
    'tau_f': {'value': 20.0, 'trainable': True},
    'tau_d': {'value': 5.0, 'trainable': True},
    'tau_r': {'value': 200.0, 'trainable': True},
    'Uinc': {'value': 0.2, 'trainable': True},
    'pconn': {'value': 1.0, 'trainable': False},
    'e_r': {'value': 0.0, 'trainable': False},
})

gen = VonMisesGenerator(dt=dt, params={
    'mean_rate': {'value': 20.0, 'trainable': False},
    'R': {'value': 0.5, 'trainable': False},
    'freq': {'value': 8.0, 'trainable': False},
    'phase': {'value': 0.0, 'trainable': False},
})

graph = NetworkGraph(dt=dt)
graph.add_input_population('theta', gen)
graph.add_population('exc', pop)
graph.add_synapse('theta->exc', syn, src='theta', tgt='exc')

network = NetworkRNN(graph, RK4Integrator())
print(f"Trainable variables: {len(network.trainable_variables)}")

t_values = np.arange(T, dtype=np.float32) * dt
t_seq = tf.constant(t_values[None, :, None])

with tf.GradientTape() as tape:
    output = network(t_seq)
    loss = tf.reduce_mean(output.firing_rates['exc'])

autograd_grads = tape.gradient(loss, network.trainable_variables)

print("\n--- TensorFlow autograd ---")
for var, g in zip(network.trainable_variables, autograd_grads):
    if g is not None:
        print(f"  {var.name}: {g.numpy().mean():.6e}")

adj_comp = AdjointGradientComputer(network, RK4Integrator())
adj_grads = adj_comp.compute_gradients(loss, t_seq)

print("\n--- Adjoint method ---")
for var_name, g in adj_grads.items():
    if g is not None:
        print(f"  {var_name}: {g.numpy().mean():.6e}")

print("\n--- Comparison ---")
for var, g_auto in zip(network.trainable_variables, autograd_grads):
    g_adj = adj_grads.get(var.name)
    if g_auto is not None and g_adj is not None:
        auto_val = g_auto.numpy().mean()
        adj_val = g_adj.numpy().mean()
        if abs(auto_val) > 1e-8:
            rel_err = abs(auto_val - adj_val) / abs(auto_val)
            print(f"  {var.name}: auto={auto_val:.4e}, adj={adj_val:.4e}, rel_err={rel_err:.4f}")

print("\n" + "=" * 60)
print(" Adjoint method готов!")
print("=" * 60)