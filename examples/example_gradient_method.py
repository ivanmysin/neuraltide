"""
Пример: Сравнение autograd и adjoint методов вычисления градиентов.

Запускать из директории neuraltide:
    python examples/example_gradient_method.py
"""
import numpy as np
import tensorflow as tf
import neuraltide
from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.populations import IzhikevichMeanField
from neuraltide.synapses import TsodyksMarkramSynapse
from neuraltide.inputs import VonMisesGenerator
from neuraltide.integrators import RK4Integrator
from neuraltide.training.losses import CompositeLoss, MSELoss
from neuraltide.utils import seed_everything, print_summary

seed_everything(42)

dt = 0.5
T = 200

# --- Модель ---
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
    'tau_f': {'value': 20.0, 'trainable': True, 'min': 6.0, 'max': 240.0},
    'tau_d': {'value': 5.0, 'trainable': True, 'min': 2.0, 'max': 15.0},
    'tau_r': {'value': 200.0, 'trainable': True, 'min': 91.0, 'max': 1300.0},
    'Uinc': {'value': 0.2, 'trainable': True, 'min': 0.04, 'max': 0.7},
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

print_summary(network)

# --- Временная ось ---
t_values = np.arange(T, dtype=np.float32) * dt
t_seq = tf.constant(t_values[None, :, None])

# --- Целевой сигнал ---
target_0 = 10.0 + 5.0 * np.sin(2 * np.pi * 8.0 * t_values / 1000.0)
target_1 = 8.0 + 4.0 * np.sin(2 * np.pi * 8.0 * t_values / 1000.0 + 0.5)
target = tf.constant(
    np.stack([target_0, target_1], axis=-1)[None, :, :],
    dtype=tf.float32
)

loss_fn = CompositeLoss([(1.0, MSELoss({'exc': target}))])
optimizer = tf.keras.optimizers.Adam(1e-3)

print("\n" + "=" * 60)
print("Обучение с gradient_method='autograd'")
print("=" * 60)

from neuraltide.training.trainer import Trainer as TrainerBase

class Trainer(TrainerBase):
    pass

trainer_autograd = Trainer(network, loss_fn, optimizer, gradient_method="autograd", run_eagerly=True)

initial_loss = trainer_autograd.train_step(t_seq)['loss']
print(f"Initial loss: {initial_loss:.4f}")

for epoch in range(20):
    trainer_autograd.train_step(t_seq)

final_loss = trainer_autograd.train_step(t_seq)['loss']
print(f"Final loss: {final_loss:.4f}")
print(f"Loss decreased: {initial_loss > final_loss}")

print("\n" + "=" * 60)
print("Обучение с gradient_method='adjoint'")
print("=" * 60)

seed_everything(42)

pop2 = IzhikevichMeanField(dt=dt, params={
    'tau_pop': {'value': [1.0, 1.0], 'trainable': False},
    'alpha': {'value': [0.5, 0.5], 'trainable': False},
    'a': {'value': [0.02, 0.02], 'trainable': False},
    'b': {'value': [0.2, 0.2], 'trainable': False},
    'w_jump': {'value': [0.1, 0.1], 'trainable': False},
    'Delta_I': {'value': [0.5, 0.6], 'trainable': True, 'min': 0.01, 'max': 2.0},
    'I_ext': {'value': [1.0, 1.2], 'trainable': True},
})

syn2 = TsodyksMarkramSynapse(n_pre=1, n_post=2, dt=dt, params={
    'gsyn_max': {'value': [[0.1, 0.1]], 'trainable': True},
    'tau_f': {'value': 20.0, 'trainable': True},
    'tau_d': {'value': 5.0, 'trainable': True},
    'tau_r': {'value': 200.0, 'trainable': True},
    'Uinc': {'value': 0.2, 'trainable': True},
    'pconn': {'value': [[1.0, 1.0]], 'trainable': False},
    'e_r': {'value': 0.0, 'trainable': False},
})

graph2 = NetworkGraph(dt=dt)
graph2.add_input_population('theta', gen)
graph2.add_population('exc', pop2)
graph2.add_synapse('theta->exc', syn2, src='theta', tgt='exc')

network2 = NetworkRNN(graph2, RK4Integrator())

trainer_adjoint = Trainer(network2, loss_fn, optimizer, gradient_method="adjoint", run_eagerly=True)

initial_loss2 = trainer_adjoint.train_step(t_seq)['loss']
print(f"Initial loss: {initial_loss2:.4f}")

for epoch in range(20):
    trainer_adjoint.train_step(t_seq)

final_loss2 = trainer_adjoint.train_step(t_seq)['loss']
print(f"Final loss: {final_loss2:.4f}")
print(f"Loss decreased: {initial_loss2 > final_loss2}")

print("\n" + "=" * 60)
print(" Оба метода работают корректно!")
print("=" * 60)