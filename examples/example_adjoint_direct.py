"""
Пример 2: Прямое использование AdjointGradientComputer.

Демонстрирует:
- Прямое использование AdjointGradientComputer без Trainer
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
from neuraltide.training.adjoint import AdjointGradientComputer
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
graph.add_input_population('theta', gen)
graph.add_population('exc', pop)
graph.add_synapse('theta->exc', syn, src='theta', tgt='exc')

network = NetworkRNN(graph, RK4Integrator())
integrator = RK4Integrator()

T = 100
t_values = np.arange(T, dtype=np.float32) * dt
t_seq = tf.constant(t_values[None, :, None])

output = network(t_seq)
loss = tf.reduce_mean(output.firing_rates['exc'])

with tf.GradientTape() as tape:
    output = network(t_seq)
    loss = tf.reduce_mean(output.firing_rates['exc'])

autograd_grads = tape.gradient(loss, network.trainable_variables)

print("\nGradients via TensorFlow autograd:")
for var, g in zip(network.trainable_variables, autograd_grads):
    if g is not None:
        print(f"  {var.name}: {g.numpy()}")

adj_comp = AdjointGradientComputer(network, integrator)
adj_grads = adj_comp.compute_gradients(loss, t_seq)

print("\nGradients via Adjoint method:")
for var_name, g in adj_grads.items():
    if g is not None:
        print(f"  {var_name}: {g.numpy()}")

print("\n" + "=" * 60)
print("Пример 2: Большая сеть (exc-inh) с NMDA")
print("=" * 60)

seed_everything(42)

pop_exc = IzhikevichMeanField(dt=dt, params={
    'tau_pop': {'value': [1.0, 1.0, 1.0, 1.0], 'trainable': False},
    'alpha': {'value': [0.5, 0.5, 0.5, 0.5], 'trainable': False},
    'a': {'value': [0.02, 0.02, 0.02, 0.02], 'trainable': False},
    'b': {'value': [0.2, 0.2, 0.2, 0.2], 'trainable': False},
    'w_jump': {'value': [0.1, 0.1, 0.1, 0.1], 'trainable': False},
    'Delta_I': {'value': [0.5, 0.6, 0.5, 0.6], 'trainable': True, 'min': 0.01, 'max': 2.0},
    'I_ext': {'value': [1.0, 1.2, 1.0, 1.2], 'trainable': True},
})

pop_inh = IzhikevichMeanField(dt=dt, params={
    'tau_pop': {'value': [1.0, 1.0], 'trainable': False},
    'alpha': {'value': [0.5, 0.5], 'trainable': False},
    'a': {'value': [0.02, 0.02], 'trainable': False},
    'b': {'value': [0.2, 0.2], 'trainable': False},
    'w_jump': {'value': [0.1, 0.1], 'trainable': False},
    'Delta_I': {'value': [0.5, 0.6], 'trainable': True, 'min': 0.01, 'max': 2.0},
    'I_ext': {'value': [1.0, 1.2], 'trainable': True},
})

syn_ampa = StaticSynapse(n_pre=1, n_post=4, dt=dt, params={
    'gsyn_max': {'value': [[0.05, 0.05, 0.05, 0.05]], 'trainable': True},
    'pconn': {'value': [[1.0, 1.0, 1.0, 1.0]], 'trainable': False},
    'e_r': {'value': 0.0, 'trainable': False},
})

syn_exc_inh = TsodyksMarkramSynapse(n_pre=4, n_post=2, dt=dt, params={
    'gsyn_max': {'value': [[0.05], [0.05]], 'trainable': True},
    'tau_f': {'value': 20.0, 'trainable': True},
    'tau_d': {'value': 5.0, 'trainable': True},
    'tau_r': {'value': 200.0, 'trainable': True},
    'Uinc': {'value': 0.2, 'trainable': True},
    'pconn': {'value': [[1.0], [1.0], [1.0], [1.0]], 'trainable': False},
    'e_r': {'value': -0.07, 'trainable': False},
})

syn_inh_exc = StaticSynapse(n_pre=2, n_post=4, dt=dt, params={
    'gsyn_max': {'value': [[0.02, 0.02], [0.02, 0.02], [0.02, 0.02], [0.02, 0.02]], 'trainable': True},
    'pconn': {'value': [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], 'trainable': False},
    'e_r': {'value': -0.07, 'trainable': False},
})

gen = VonMisesGenerator(dt=dt, params={
    'mean_rate': {'value': 20.0, 'trainable': False},
    'R': {'value': 0.5, 'trainable': False},
    'freq': {'value': 8.0, 'trainable': False},
    'phase': {'value': 0.0, 'trainable': False},
})

graph2 = NetworkGraph(dt=dt)
graph2.add_input_population('theta', gen)
graph2.add_population('exc', pop_exc)
graph2.add_population('inh', pop_inh)
graph2.add_synapse('theta->exc', syn_ampa, src='theta', tgt='exc')
graph2.add_synapse('exc->inh', syn_exc_inh, src='exc', tgt='inh')
graph2.add_synapse('inh->exc', syn_inh_exc, src='inh', tgt='exc')

network2 = NetworkRNN(graph2, RK4Integrator())

print(f"\nЧисло trainable переменных: {len(network2.trainable_variables)}")

output = network2(t_seq)
loss = tf.reduce_mean(output.firing_rates['exc']) + tf.reduce_mean(output.firing_rates['inh'])

adj_comp2 = AdjointGradientComputer(network2, RK4Integrator())
adj_grads = adj_comp2.compute_gradients(loss, t_seq)

print("Gradients via Adjont method:")
for var_name, g in list(adj_grads.items())[:5]:
    if g is not None:
        print(f"  {var_name}: shape={g.shape}, mean={g.numpy().mean():.6f}")
print("  ...")

print("\n" + "=" * 60)
print("Пример 3: Длинная последовательность (T=500)")
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
graph3.add_input_population('theta', gen3)
graph3.add_population('exc', pop3)
graph3.add_synapse('theta->exc', syn3, src='theta', tgt='exc')

network3 = NetworkRNN(graph3, RK4Integrator())

T_long = 500
t_values = np.arange(T_long, dtype=np.float32) * dt
t_seq_long = tf.constant(t_values[None, :, None])

print(f"T = {T_long} шагов ({T_long * dt} мс)")

output = network3(t_seq_long)
loss = tf.reduce_mean(output.firing_rates['exc'])

adj_comp3 = AdjointGradientComputer(network3, RK4Integrator())
adj_grads = adj_comp3.compute_gradients(loss, t_seq_long)

print(f"Вычислено градиентов для {len(adj_grads)} переменных")
for var_name, g in adj_grads.items():
    if g is not None:
        print(f"  {var_name}: shape={g.shape}, norm={np.linalg.norm(g.numpy()):.6f}")

print("\n" + "=" * 60)
print("Все примеры выполнены успешно!")
print("=" * 60)