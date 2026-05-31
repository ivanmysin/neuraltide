"""
Пример: Прямое использование AdjointSolver.

Демонстрирует:
- Прямое использование AdjointSolver без Trainer
- Сравнение градиентов между autograd и adjoint
- Вычисление градиентов для длинных последовательностей
"""
import numpy as np
import tensorflow as tf
import neuraltide
from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.populations import IzhikevichMeanField
from neuraltide.synapses import TsodyksMarkramSynapse, StaticSynapse
from neuraltide.inputs import VonMisesGenerator
from neuraltide.integrators import RK4Integrator
from neuraltide.training.adjoint import AdjointSolver
from neuraltide.training import MSELoss, CompositeLoss
from neuraltide.utils import seed_everything

seed_everything(42)

dt = 0.5

print("=" * 60)
print("Пример 1: Простая модель с одним синапсом")
print("=" * 60)

pop = IzhikevichMeanField(dt=dt, params={
    'tau_pop': {'value': [1.0, 1.0], 'trainable': False},
    'alpha': {'value': [0.5, 0.5], 'trainable': False},
    'a': {'value': [0.02, 0.02], 'trainable': False},
    'b': {'value': [0.2, 0.2], 'trainable': False},
    'w_jump': {'value': [0.1, 0.1], 'trainable': False},
    'Delta_I': {'value': [0.5, 0.6], 'trainable': True, 'min': 0.01, 'max': 2.0},
    'I_ext': {'value': [1.0, 1.2], 'trainable': True},
})

syn = TsodyksMarkramSynapse(n_pre=1, n_post=2, dt=dt, params={
    'gsyn_max': {'value': [[0.1, 0.1]], 'trainable': True},
    'tau_f': {'value': 20.0, 'trainable': True},
    'tau_d': {'value': 5.0, 'trainable': True},
    'tau_r': {'value': 200.0, 'trainable': True},
    'Uinc': {'value': 0.2, 'trainable': True},
    'pconn': {'value': [[1.0, 1.0]], 'trainable': False},
    'e_r': {'value': 0.0, 'trainable': False},
})

gen = VonMisesGenerator(dt=dt, params={
    'mean_rate': {'value': 20.0, 'trainable': False},
    'R': {'value': 0.5, 'trainable': False},
    'freq': {'value': 8.0, 'trainable': False},
    'phase': {'value': 0.0, 'trainable': False},
})

graph = NetworkGraph(dt=dt)
graph.declare_input('theta', n_units=gen.n_units)
graph.add_population('exc', pop)
graph.add_synapse('theta->exc', syn, src='theta', tgt='exc')

network = NetworkRNN(graph, RK4Integrator())

T = 100
t_values = np.arange(T, dtype=np.float32) * dt
t_seq = tf.constant(t_values[None, :, None])
inputs = graph.pack_inputs({'theta': gen(t_seq)})

target = {
    'exc': tf.constant(
        (10.0 + 5.0 * np.sin(2 * np.pi * 8.0 * t_values / 1000.0))[None, :, None].repeat(2, axis=-1),
        dtype=tf.float32
    )
}

# Autograd
with tf.GradientTape() as tape:
    output = network(t_seq, inputs=inputs)
    loss_fn = MSELoss(target)
    loss = loss_fn(output, network)

autograd_grads = tape.gradient(loss, network.trainable_variables)

print("\nGradients via TensorFlow autograd:")
for var, g in zip(network.trainable_variables, autograd_grads):
    if g is not None:
        print(f"  {var.name}: norm={np.linalg.norm(g.numpy()):.6f}")

# Adjoint
adj_comp = AdjointSolver(network, RK4Integrator())
adj_grads_list, adj_vars, adj_out = adj_comp.compute_gradients(
    t_seq, inputs, target, loss_fn)

print("\nGradients via Adjoint method:")
for v, g in zip(adj_vars, adj_grads_list):
    if g is not None:
        print(f"  {v.name}: norm={np.linalg.norm(g.numpy()):.6f}")

print("\n" + "=" * 60)
print("Пример 2: Длинная последовательность (T=500)")
print("=" * 60)

seed_everything(42)

pop3 = IzhikevichMeanField(dt=dt, params={
    'tau_pop': {'value': [1.0], 'trainable': False},
    'alpha': {'value': [0.5], 'trainable': False},
    'a': {'value': [0.02], 'trainable': False},
    'b': {'value': [0.2], 'trainable': False},
    'w_jump': {'value': [0.1], 'trainable': False},
    'Delta_I': {'value': [0.5], 'trainable': True, 'min': 0.01, 'max': 2.0},
    'I_ext': {'value': [1.0], 'trainable': True},
})

syn3 = StaticSynapse(n_pre=1, n_post=1, dt=dt, params={
    'gsyn_max': {'value': [[0.1]], 'trainable': True},
    'pconn': {'value': [[1.0]], 'trainable': False},
    'e_r': {'value': 0.0, 'trainable': False},
})

gen3 = VonMisesGenerator(dt=dt, params={
    'mean_rate': {'value': 20.0, 'trainable': False},
    'R': {'value': 0.5, 'trainable': False},
    'freq': {'value': 8.0, 'trainable': False},
    'phase': {'value': 0.0, 'trainable': False},
})

graph3 = NetworkGraph(dt=dt)
graph3.declare_input('theta', n_units=gen3.n_units)
graph3.add_population('exc', pop3)
graph3.add_synapse('theta->exc', syn3, src='theta', tgt='exc')

network3 = NetworkRNN(graph3, RK4Integrator())

T_long = 500
t_values = np.arange(T_long, dtype=np.float32) * dt
t_seq_long = tf.constant(t_values[None, :, None])
inputs3 = graph3.pack_inputs({'theta': gen3(t_seq_long)})

print(f"T = {T_long} шагов ({T_long * dt} мс)")

output3 = network3(t_seq_long, inputs=inputs3)
target3 = {'exc': output3.firing_rates['exc']}
loss_fn3 = MSELoss(target3)

adj_comp3 = AdjointSolver(network3, RK4Integrator())
adj_grads_list3, adj_vars3, adj_out3 = adj_comp3.compute_gradients(
    t_seq_long, inputs3, target3, loss_fn3)

print(f"Вычислено градиентов для {len(adj_grads_list3)} переменных")
for v, g in zip(adj_vars3, adj_grads_list3):
    if g is not None:
        print(f"  {v.name}: shape={g.shape}, norm={np.linalg.norm(g.numpy()):.6f}")

print("\n" + "=" * 60)
print("Все примеры выполнены успешно!")
print("=" * 60)
