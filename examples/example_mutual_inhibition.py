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
T_total = 1000  # ms - total simulation time
batch_size = 25   # ms per batch
transient = 20     # ms transient to skip
nbatches = 2       # number of batches to optimize (reduced for test)

n_batches = T_total // batch_size
print(f"Total: T={T_total}ms, batch={batch_size}ms ({n_batches} batches), transient={transient}ms")

t_all = np.arange(0, T_total, dt)
n_steps_per_batch = int(batch_size / dt)
n_transient_steps = int(transient / dt)

t_seq_full = tf.constant(t_all[None, :, None], dtype=tf.float32)

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

gen_target = VonMisesGenerator(dt=dt, params={
    'mean_rate': 50.0,
    'R': 0.6,
    'freq': 8.0,
    'phase': [0.0, 2.61],
})

t_target = tf.constant(t_all[None, :, None], dtype=tf.float32)
target_full = gen_target(t_target).numpy()
target_full = target_full[0, :, :]  # [time, 2]

t_eval_start = n_transient_steps
t_eval_end = t_all.shape[0]
t_eval_mask = np.zeros(t_all.shape[0], dtype=bool)
t_eval_mask[t_eval_start:] = True

target_eval_pop1 = target_full[t_eval_mask, 0]
target_eval_pop2 = target_full[t_eval_mask, 1]

target = {
    'pop1': tf.constant(target_full[None, :, 0:1], dtype=tf.float32),
    'pop2': tf.constant(target_full[None, :, 1:2], dtype=tf.float32),
}

loss_fn = CompositeLoss([
    (1.0, MSELoss(target)),
    (1e-3, StabilityPenalty()),
])

optimizer = tf.keras.optimizers.Adam(1e-3)
trainer = Trainer(network, loss_fn, optimizer, grad_method='bptt', grad_clip_norm=1.0)

print("\n=== Transient simulation ===")
t_transient = t_all[:n_transient_steps]
t_seq_transient = tf.constant(t_transient[None, :, None], dtype=tf.float32)
_ = network(t_seq_transient, training=False)

initial_pop_state, initial_syn_state = network.get_state(force_compute=True)
print(f"Transient done: pop1 r={initial_pop_state[0][0,0]:.4f}")

network.reset_state()

def get_batch_tensor(batch_idx):
    start = batch_idx * n_steps_per_batch
    end = start + n_steps_per_batch
    return tf.constant(t_all[start:end][None, :, None], dtype=tf.float32)

def get_batch_target(batch_idx):
    start = batch_idx * n_steps_per_batch
    end = start + n_steps_per_batch
    return {
        'pop1': tf.constant(target_full[start:end, 0:1][None, :], dtype=tf.float32),
        'pop2': tf.constant(target_full[start:end, 1:2][None, :], dtype=tf.float32),
    }

def run_batched_simulation(n_batches_to_run, initial_state=None):
    all_outputs = {'pop1': [], 'pop2': []}
    current_pop_state = initial_state[0] if initial_state else None
    current_syn_state = initial_state[1] if initial_state else None
    
    for b in range(n_batches_to_run):
        t_batch = get_batch_tensor(b)
        
        if current_pop_state is not None:
            output = network(t_batch, training=False, initial_state=(current_pop_state, current_syn_state))
        else:
            output = network(t_batch, training=False)
        
        rates = output.firing_rates
        all_outputs['pop1'].append(rates['pop1'].numpy())
        all_outputs['pop2'].append(rates['pop2'].numpy())
        
        current_pop_state, current_syn_state = network.get_state(force_compute=True)
    
    pop1_rates = np.concatenate(all_outputs['pop1'], axis=1)
    pop2_rates = np.concatenate(all_outputs['pop2'], axis=1)
    
    return pop1_rates, pop2_rates

print("\n=== Batched simulation (before opt) ===")
rates_pop1, rates_pop2 = run_batched_simulation(nbatches)
print(f"Pop1 mean: {rates_pop1[0,t_eval_start:].mean():.2f} Hz")
print(f"Pop2 mean: {rates_pop2[0,t_eval_start:].mean():.2f} Hz")

print(f"\n=== Optimization ({nbatches} batches, {nbatches} epochs) ===")

print(f"Starting {nbatches}-batch optimization...")

current_pop_state = initial_pop_state
current_syn_state = initial_syn_state

from neuraltide.training.losses import MSELoss

for epoch in range(nbatches):
    for b in range(nbatches):
        t_batch = get_batch_tensor(b)
        target_batch = get_batch_target(b)
        
        loss_batch = CompositeLoss([
            (1.0, MSELoss(target_batch)),
            (1e-3, StabilityPenalty()),
        ])
        
        with tf.GradientTape() as tape:
            output = network(t_batch, training=True, initial_state=(current_pop_state, current_syn_state))
            loss = loss_batch(output, network)
        
        grads = tape.gradient(loss, network.trainable_variables)
        grads_and_vars = [(g, v) for g, v in zip(grads, network.trainable_variables) if g is not None]
        
        if grads_and_vars:
            grads_only = [g for g, v in grads_and_vars]
            clipped_grads, _ = tf.clip_by_global_norm(grads_only, 1.0)
            grads_and_vars = [(g, v) for g, (_, v) in zip(clipped_grads, grads_and_vars)]
            optimizer.apply_gradients(grads_and_vars)
        
        current_pop_state, current_syn_state = network.get_state(force_compute=True)
        print(f"  Batch {b}: loss={loss.numpy():.4f}")
    
    print(f"Epoch {epoch+1}/{nbatches} done")

rates_pop1, rates_pop2 = run_batched_simulation(nbatches)
rates_full = rates_pop1[0, :]
n_eval_start = n_transient_steps
target_for_eval = target_full[:n_steps_per_batch * nbatches, 0].flatten()
loss_val = np.mean((rates_full[n_eval_start:] - target_for_eval[n_eval_start:])**2)
print(f"Epoch {epoch+1}/{nbatches} - loss: {loss_val:.4f}, pop1: {rates_pop1[0,n_eval_start:].mean():.2f} Hz")

final_rates_pop1, final_rates_pop2 = run_batched_simulation(nbatches)
rates_full = final_rates_pop1[0, :]
n_eval_start = n_transient_steps

print("\n=== Results ===")
print(f"Pop1 mean rate: {rates_full[n_eval_start:].mean():.2f} Hz")
print(f"Pop2 mean rate: {final_rates_pop2[0,n_eval_start:].mean():.2f} Hz")

print("\n=== Trainable variables ===")
for v in network.trainable_variables:
    print(f"{v.name}: {float(v.numpy()[0, 0]):.4f}")

trainer.export_results('optimization_results.json')
trainer.export_results('optimization_results.csv', format='csv')
print("\nResults exported to optimization_results.json and optimization_results.csv")