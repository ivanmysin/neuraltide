"""
Пример: Две взаимно ингибирующие популяции fast-spiking нейронов.

Архитектура:
- Две однородные популяции IzhikevichMeanField (Pop1 и Pop2)
- Взаимное ингибирование через синапсы TsodyksMarkramSynapse (short-term depression)
- Внешний возбуждающий ток подаётся на каждую популяцию

Оптимизируемые параметры:
- g12, g21: максимальные синаптические проводимости
- I_ext1, I_ext2: внешние токи
- tau_d, tau_r, tau_f, U_inc: параметры короткоtemporary депрессии
"""

import numpy as np
import tensorflow as tf
from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.populations import IzhikevichMeanField
from neuraltide.synapses import TsodyksMarkramSynapse
from neuraltide.inputs import VonMisesGenerator
from neuraltide.integrators import RK4Integrator
from neuraltide.training import Trainer, CompositeLoss, MSELoss, StabilityPenalty
from neuraltide.training.callbacks import DivergenceDetector

dt = 0.1
T = 30
transient = 10
t = np.arange(0, T, dt)
print(f"Total steps: {len(t)}, transient after step {int(transient/dt)}")
t_seq = tf.constant(t[None, :, None], dtype=tf.float32)

pop_params = {
    'Cm': 114.0,
    'K': 1.19,
    'V_rest': -57.63,
    'V_T': -35.53,
    'V_reset': -48.7,
    'A': 0.005,
    'B': 0.22,
    'W_jump': 2.0,
    'Delta_I': 20.0,
    'I_ext': {'value': 400.0, 'trainable': True, 'min': 100.0, 'max': 1000.0},
}

pop1 = IzhikevichMeanField(dt=dt, params=pop_params.copy(), name='pop1')
pop2 = IzhikevichMeanField(dt=dt, params=pop_params.copy(), name='pop2')

syn_1to2 = TsodyksMarkramSynapse(n_pre=1, n_post=1, dt=dt, params={
    'gsyn_max': {'value': 100.0, 'trainable': True, 'min': 10.0, 'max': 500.0},
    'tau_d': {'value': 6.02, 'trainable': True, 'min': 2.0, 'max': 15.0},
    'tau_r': {'value': 200.0, 'trainable': True, 'min': 50.0, 'max': 500.0},
    'tau_f': {'value': 20.0, 'trainable': True, 'min': 5.0, 'max': 100.0},
    'Uinc': {'value': 0.3, 'trainable': True, 'min': 0.1, 'max': 0.6},
    'pconn': {'value': 1.0, 'trainable': False},
    'e_r': {'value': -75.0, 'trainable': False},
}, name='syn_1to2')

syn_2to1 = TsodyksMarkramSynapse(n_pre=1, n_post=1, dt=dt, params={
    'gsyn_max': {'value': 100.0, 'trainable': True, 'min': 10.0, 'max': 500.0},
    'tau_d': {'value': 6.02, 'trainable': True, 'min': 2.0, 'max': 15.0},
    'tau_r': {'value': 200.0, 'trainable': True, 'min': 50.0, 'max': 500.0},
    'tau_f': {'value': 20.0, 'trainable': True, 'min': 5.0, 'max': 100.0},
    'Uinc': {'value': 0.3, 'trainable': True, 'min': 0.1, 'max': 0.6},
    'pconn': {'value': 1.0, 'trainable': False},
    'e_r': {'value': -75.0, 'trainable': False},
}, name='syn_2to1')

graph = NetworkGraph(dt=dt)
graph.add_population('pop1', pop1)
graph.add_population('pop2', pop2)
graph.add_synapse('pop1->pop2', syn_1to2, src='pop1', tgt='pop2')
graph.add_synapse('pop2->pop1', syn_2to1, src='pop2', tgt='pop1')

network = NetworkRNN(graph, integrator=RK4Integrator(), stability_penalty_weight=1e-3)

if transient > 0:
    t_idx = t >= transient
else:
    t_idx = np.ones_like(t, dtype=bool)

gen_target1 = VonMisesGenerator(dt=dt, params={
    'mean_rate': 50.0,
    'R': 0.6,
    'freq': 8.0,
    'phase': 0.0,
})

gen_target2 = VonMisesGenerator(dt=dt, params={
    'mean_rate': 50.0,
    'R': 0.6,
    'freq': 8.0,
    'phase': 2.61,
})

t_target = tf.constant(t[None, :, None], dtype=tf.float32)
target_1_full = gen_target1(t_target).numpy().flatten()
target_2_full = gen_target2(t_target).numpy().flatten()

if transient > 0:
    trans_idx = t < transient
    target_1_full[trans_idx] = 0.0
    target_2_full[trans_idx] = 0.0

target = {
    'pop1': tf.constant(target_1_full[None, :, None], dtype=tf.float32),
    'pop2': tf.constant(target_2_full[None, :, None], dtype=tf.float32),
}

loss_fn = CompositeLoss([
    (1.0, MSELoss(target)),
    (1e-3, StabilityPenalty()),
])

optimizer = tf.keras.optimizers.Adam(1e-3)
trainer = Trainer(network, loss_fn, optimizer, grad_method='bptt', grad_clip_norm=1.0)

transient_idx = int(transient / dt)
t_transient = t[:transient_idx]
t_seq_transient = tf.constant(t_transient[None, :, None], dtype=tf.float32)

print("Running transient simulation...")
_ = network(t_seq_transient, training=False)
print("Transient done")

final_pop_state, final_syn_state = network.get_state(force_compute=True)
print(f"Got state")

print(f"Transient final state: pop1 r={final_pop_state[0][0,0]:.4f}")
print("Resetting state...")
network.reset_state()
print("Reset done")
print("\nStarting optimization with state from transient...")

callbacks = [DivergenceDetector()]

history = trainer.fit(t_seq, epochs=20, callbacks=callbacks, verbose=2, initial_state=(final_pop_state, final_syn_state))

print(f"\nInitial loss: {history.loss_history[0]:.4f}")
print(f"Final loss: {history.loss_history[-1]:.4f}")

output = network(t_seq, training=False)
rates = output.firing_rates

print("\n=== Trainable variables ===")
for v in network.trainable_variables:
    print(f"{v.name}: {float(v.numpy()[0, 0]):.4f}")

output1 = rates['pop1'].numpy().flatten()
output2 = rates['pop2'].numpy().flatten()

print(f"\nPop1 mean rate (after transient): {np.mean(output1[t_idx]):.2f} Hz")
print(f"Pop2 mean rate (after transient): {np.mean(output2[t_idx]):.2f} Hz")

trainer.export_results('optimization_results.json')
trainer.export_results('optimization_results.csv', format='csv')
print("\nResults exported to optimization_results.json and optimization_results.csv")