"""
Пример 1: Одна популяция IzhikevichMeanField (n_units=2),
тета-ритмический вход через обучаемый синапс.
Цель: воспроизвести заданную firing rate траекторию.
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
from neuraltide.training import Trainer, CompositeLoss, MSELoss, StabilityPenalty
from neuraltide.utils import seed_everything, print_summary

seed_everything(42)

dt = 0.05
T = 20

pop = IzhikevichMeanField(dt=dt, params={
    'tau_pop':   {'value': [1.0, 1.0],   'trainable': False},
    'alpha':     {'value': [0.5, 0.5],   'trainable': False},
    'a':         {'value': [0.02, 0.02], 'trainable': False},
    'b':         {'value': [0.2, 0.2],   'trainable': False},
    'w_jump':    {'value': [0.1, 0.1],   'trainable': False},
    'Delta_I':   {'value': [0.5, 0.6],   'trainable': True,
                  'min': 0.01, 'max': 2.0},
    'I_ext':     {'value': [0.1, 0.2],   'trainable': True,
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

syn_in = TsodyksMarkramSynapse(n_pre=1, n_post=2, dt=dt, params={
    'gsyn_max': {'value': [[0.1, 0.1]], 'trainable': True, 'min': 0.0},
    'tau_f':    {'value': 20.0,  'trainable': True, 'min': 6.0,  'max': 240.0},
    'tau_d':    {'value': 5.0,   'trainable': True, 'min': 2.0,  'max': 15.0},
    'tau_r':    {'value': 200.0, 'trainable': True, 'min': 91.0, 'max': 1300.0},
    'Uinc':     {'value': 0.2,   'trainable': True, 'min': 0.04, 'max': 0.7},
    'pconn':    {'value': [[1.0, 1.0]], 'trainable': False},
    'e_r':      {'value': 1.0,   'trainable': False},
})

syn_rec = TsodyksMarkramSynapse(n_pre=2, n_post=2, dt=dt, params={
    'gsyn_max': {'value': 0.05,  'trainable': True},
    'tau_f':    {'value': 20.0,  'trainable': True, 'min': 6.0,  'max': 240.0},
    'tau_d':    {'value': 5.0,   'trainable': True, 'min': 2.0,  'max': 15.0},
    'tau_r':    {'value': 200.0, 'trainable': True, 'min': 91.0, 'max': 1300.0},
    'Uinc':     {'value': 0.2,   'trainable': True, 'min': 0.04, 'max': 0.7},
    'pconn':    {'value': [[1, 1], [1, 1]], 'trainable': False},
    'e_r':      {'value': 1.0,   'trainable': False},
})

graph = NetworkGraph(dt=dt)
graph.add_input_population('theta', gen)
graph.add_population('exc', pop)
graph.add_synapse('theta->exc', syn_in,  src='theta', tgt='exc')
graph.add_synapse('exc->exc',   syn_rec, src='exc',   tgt='exc')

network = NetworkRNN(graph, integrator=RK4Integrator(),
                     stability_penalty_weight=1e-3)
print_summary(network)

t_values = np.arange(T, dtype=np.float32) * dt
t_seq = tf.constant(t_values[None, :, None])

target_0 = 10.0 + 5.0*np.sin(2*np.pi*8.0*t_values/1000.0)
target_1 = 8.0  + 4.0*np.sin(2*np.pi*8.0*t_values/1000.0 + 0.5)
target = {
    'exc': tf.constant(
        np.stack([target_0, target_1], axis=-1)[None, :, :],
        dtype=tf.float32
    )
}

output_before = network(t_seq, training=False)

loss_fn = CompositeLoss([
    (1.0,  MSELoss(target)),
    (1e-3, StabilityPenalty()),
])
trainer = Trainer(network, loss_fn,
                  optimizer=tf.keras.optimizers.Adam(1e-3))
history = trainer.fit(t_seq, epochs=2000, verbose=2)

output_after = network(t_seq, training=False)

tgt = target['exc'].numpy()[0]
pred_before = output_before.firing_rates['exc'].numpy()[0]
pred_after = output_after.firing_rates['exc'].numpy()[0]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

epochs = np.arange(1, len(history.loss_history) + 1)
axes[0, 0].plot(epochs, history.loss_history, color='tab:blue', linewidth=1.5)
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].set_title(f"Training Loss (final: {history.loss_history[-1]:.2f})")
axes[0, 0].grid(True, alpha=0.3)

for unit in range(2):
    axes[0, 1].plot(t_values, tgt[:, unit], color='tab:green', linewidth=2, label=f'Target unit {unit}', alpha=0.8)
    axes[0, 1].plot(t_values, pred_before[:, unit], color=f'C{unit}', linewidth=1.5, linestyle='--', label=f'Before unit {unit}', alpha=0.8)
axes[0, 1].set_xlabel("Time (ms)")
axes[0, 1].set_ylabel("Firing Rate (Hz)")
axes[0, 1].set_title("Before Training: Target vs Prediction")
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(True, alpha=0.3)

for unit in range(2):
    axes[0, 2].plot(t_values, tgt[:, unit], color='tab:green', linewidth=2, label=f'Target unit {unit}', alpha=0.8)
    axes[0, 2].plot(t_values, pred_after[:, unit], color=f'C{unit}', linewidth=1.5, linestyle='--', label=f'After unit {unit}', alpha=0.8)
axes[0, 2].set_xlabel("Time (ms)")
axes[0, 2].set_ylabel("Firing Rate (Hz)")
axes[0, 2].set_title("After Training: Target vs Prediction")
axes[0, 2].legend(fontsize=8)
axes[0, 2].grid(True, alpha=0.3)

for unit in range(2):
    axes[1, 1].plot(t_values, pred_before[:, unit] - tgt[:, unit],
                    color=f'C{unit}', linewidth=1.5, linestyle='--',
                    label=f'Unit {unit}', alpha=0.8)
axes[1, 1].axhline(0, color='black', linewidth=0.5)
axes[1, 1].set_xlabel("Time (ms)")
axes[1, 1].set_ylabel("Prediction - Target (Hz)")
axes[1, 1].set_title("Prediction Error Before Training")
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(True, alpha=0.3)

for unit in range(2):
    axes[1, 2].plot(t_values, pred_after[:, unit] - tgt[:, unit],
                    color=f'C{unit}', linewidth=1.5, linestyle='--',
                    label=f'Unit {unit}', alpha=0.8)
axes[1, 2].axhline(0, color='black', linewidth=0.5)
axes[1, 2].set_xlabel("Time (ms)")
axes[1, 2].set_ylabel("Prediction - Target (Hz)")
axes[1, 2].set_title("Prediction Error After Training")
axes[1, 2].legend(fontsize=8)
axes[1, 2].grid(True, alpha=0.3)

axes[1, 0].axis('off')

fig.suptitle("Example 1: IzhikevichMeanField + TsodyksMarkramSynapse Training", fontsize=13)
plt.tight_layout()
plt.savefig("example_01_training_results.png", dpi=150)
plt.show()
print("Figure saved as example_01_training_results.png")
