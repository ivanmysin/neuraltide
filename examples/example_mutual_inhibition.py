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
from neuraltide.training import Trainer, CompositeLoss, MSELoss, MSLELoss, StabilityPenalty
import time


dt = 0.1
T_total = 1000  # ms - total simulation time
batch_size = 25   # ms per batch
transient = 20     # ms transient to skip
nbatches = 1       # number of batches to optimize (reduced for test)
nepochs = 200

n_batches = T_total // batch_size
print(f"Total: T={T_total}ms, batch={batch_size}ms ({n_batches} batches), transient={transient}ms")

t_all = np.arange(0, T_total, dt)
n_steps_per_batch = int(batch_size / dt)
n_transient_steps = int(transient / dt)

t_seq_full = tf.constant(t_all[None, :, None], dtype=tf.float32)

# pop_params = {
#     'Cm': 114.0,
#     'K': 1.19,
#     'V_rest': -57.63,
#     'V_T': -35.53,
#     'V_reset': -48.7,
#     'A': 0.005,
#     'B': 0.22,
#     'W_jump': 2.0,
#     'Delta_I': 20.0,
#     'I_ext': {'value': 400.0, 'trainable': True, 'min': 0.0, 'max': 1000.0},
# }

pop_params = {'tau_pop':   {'value': [1.0, ],   'trainable': False},
              'alpha':     {'value': [0.5, ],   'trainable': False},
              'a':         {'value': [0.02, ], 'trainable': False},
              'b':         {'value': [0.2, ],   'trainable': False},
              'w_jump':    {'value': [0.1, ],   'trainable': False},
              'Delta_I':   {'value': [0.5, ],   'trainable': False,
                            'min': 0.01, 'max': 2.0},
              'I_ext':     {'value': [0.1, ],   'trainable': True,
                            'min': -2.0, 'max': 2.0},}

pop1 = IzhikevichMeanField(dt=dt, params=pop_params.copy(), name='pop1')
pop2 = IzhikevichMeanField(dt=dt, params=pop_params.copy(), name='pop2')

syn_1to2 = TsodyksMarkramSynapse(n_pre=1, n_post=1, dt=dt, params={
    'gsyn_max': {'value': 0.1, 'trainable': True, 'min': 0.0, 'max': 0.5},
    'tau_d': {'value': 6.02, 'trainable': True, 'min': 2.0, 'max': 15.0},
    'tau_r': {'value': 200.0, 'trainable': True, 'min': 50.0, 'max': 500.0},
    'tau_f': {'value': 20.0, 'trainable': True, 'min': 5.0, 'max': 100.0},
    'Uinc': {'value': 0.3, 'trainable': True, 'min': 0.1, 'max': 0.6},
    'pconn': {'value': 1.0, 'trainable': False},
    'e_r': {'value': -0.2, 'trainable': False},
}, name='syn_1to2')

syn_2to1 = TsodyksMarkramSynapse(n_pre=1, n_post=1, dt=dt, params={
    'gsyn_max': {'value': 0.1, 'trainable': True, 'min': 0.0, 'max': 0.5},
    'tau_d': {'value': 6.02, 'trainable': True, 'min': 2.0, 'max': 15.0},
    'tau_r': {'value': 200.0, 'trainable': True, 'min': 50.0, 'max': 500.0},
    'tau_f': {'value': 20.0, 'trainable': True, 'min': 5.0, 'max': 100.0},
    'Uinc': {'value': 0.3, 'trainable': True, 'min': 0.1, 'max': 0.6},
    'pconn': {'value': 1.0, 'trainable': False},
    'e_r': {'value': -0.2, 'trainable': False},
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
    (1.0, MSLELoss(target)),
    (1e-3, StabilityPenalty()),
])

optimizer = tf.keras.optimizers.Adam(1e-4)
trainer = Trainer(network, loss_fn, optimizer, grad_method='bptt', grad_clip_norm=1.0)

print("\n=== Pre-training: Full transient simulation ===")
print("Running full T=1000ms simulation to observe pre-training dynamics...")

network.reset_state()
t_seq_full_tf = tf.constant(t_all[None, :, None], dtype=tf.float32)

timer=time.time()
_ = network(t_seq_full_tf, training=False)
print(f"Full simulation time: {time.time()-timer:.2f} sec")
pre_train_pop_state, pre_train_syn_state = network.get_state(force_compute=True)

# print(f"Pre-training state after full simulation:")
# print(f"  Pop1: r={float(pre_train_pop_state[0].numpy()[0,0]):.4f}, v={float(pre_train_pop_state[1].numpy()[0,0]):.4f}, w={float(pre_train_pop_state[2].numpy()[0,0]):.4f}")
# print(f"  Pop2: r={float(pre_train_pop_state[3].numpy()[0,0]):.4f}, v={float(pre_train_pop_state[4].numpy()[0,0]):.4f}, w={float(pre_train_pop_state[5].numpy()[0,0]):.4f}")
# print(f"  Syn1to2: R={float(pre_train_syn_state[0].numpy()[0,0]):.4f}, U={float(pre_train_syn_state[1].numpy()[0,0]):.4f}, A={float(pre_train_syn_state[2].numpy()[0,0]):.4f}")
# print(f"  Syn2to1: R={float(pre_train_syn_state[3].numpy()[0,0]):.4f}, U={float(pre_train_syn_state[4].numpy()[0,0]):.4f}, A={float(pre_train_syn_state[5].numpy()[0,0]):.4f}")

# print("\n=== Visualization: Pre-training dynamics ===")
# try:
#     import matplotlib
#     matplotlib.use('Agg')
#     import matplotlib.pyplot as plt
#
#     network.reset_state()
#     pop_hist = {
#         'pop1': {'r': [], 'v': [], 'w': []},
#         'pop2': {'r': [], 'v': [], 'w': []},
#     }
#     syn_hist = {
#         'syn_1to2': {'R': [], 'U': [], 'A': []},
#         'syn_2to1': {'R': [], 'U': [], 'A': []},
#     }
#     pop_states, syn_states = None, None
#     transient_states = [pre_train_pop_state, pre_train_syn_state]
#
#     for step in range(t_all.shape[0]):
#         t = tf.constant([[t_all[step]]], dtype=tf.float32)
#         if step == 0:
#             output = network(t, training=False, initial_state=transient_states)
#         else:
#             output = network(t, training=False, initial_state=(pop_states, syn_states))
#         pop_states, syn_states = network.get_state(force_compute=True)
#         pop_hist['pop1']['r'].append(pop_states[0][0].numpy())
#         pop_hist['pop1']['v'].append(pop_states[1][0].numpy())
#         pop_hist['pop1']['w'].append(pop_states[2][0].numpy())
#         pop_hist['pop2']['r'].append(pop_states[3][0].numpy())
#         pop_hist['pop2']['v'].append(pop_states[4][0].numpy())
#         pop_hist['pop2']['w'].append(pop_states[5][0].numpy())
#         syn_hist['syn_1to2']['R'].append(syn_states[0][0].numpy())
#         syn_hist['syn_1to2']['U'].append(syn_states[1][0].numpy())
#         syn_hist['syn_1to2']['A'].append(syn_states[2][0].numpy())
#         syn_hist['syn_2to1']['R'].append(syn_states[3][0].numpy())
#         syn_hist['syn_2to1']['U'].append(syn_states[4][0].numpy())
#         syn_hist['syn_2to1']['A'].append(syn_states[5][0].numpy())
#
#     fig, axes = plt.subplots(4, 2, figsize=(14, 12))
#
#     ax = axes[0, 0]
#     r_p1 = np.squeeze(np.array(pop_hist['pop1']['r']))
#     r_p2 = np.squeeze(np.array(pop_hist['pop2']['r']))
#     ax.plot(t_all, r_p1, 'b-', linewidth=1.5, label='pop1')
#     ax.plot(t_all, r_p2, 'r-', linewidth=1.5, label='pop2')
#     ax.set_xlabel('Time (ms)')
#     ax.set_ylabel('r (dimensionless)')
#     ax.set_title('Pre-training: Firing Rate r (dimensionless)')
#     ax.legend()
#     ax.grid(True, alpha=0.3)
#
#     ax = axes[0, 1]
#     ax.plot(t_all, target_full[:, 0], 'b--', linewidth=2, label='Target pop1')
#     ax.plot(t_all, r_p1 / (dt * 1e-3), 'b-', linewidth=1.5, alpha=0.8, label='pop1')
#     ax.plot(t_all, target_full[:, 1], 'r--', linewidth=2, label='Target pop2')
#     ax.plot(t_all, r_p2 / (dt * 1e-3), 'r-', linewidth=1.5, alpha=0.8, label='pop2')
#     ax.set_xlabel('Time (ms)')
#     ax.set_ylabel('Firing Rate (Hz)')
#     ax.set_title('Pre-training: Firing Rate (Hz) with Target')
#     ax.legend()
#     ax.grid(True, alpha=0.3)
#
#     ax = axes[1, 0]
#     ax.plot(t_all, np.squeeze(np.array(pop_hist['pop1']['v'])), 'b-', linewidth=1.5, label='v pop1')
#     ax.plot(t_all, np.squeeze(np.array(pop_hist['pop2']['v'])), 'r-', linewidth=1.5, label='v pop2')
#     ax.set_xlabel('Time (ms)')
#     ax.set_ylabel('v (dimensionless)')
#     ax.set_title('Pre-training: Mean Membrane Potential v')
#     ax.legend()
#     ax.grid(True, alpha=0.3)
#
#     ax = axes[1, 1]
#     ax.plot(t_all, np.squeeze(np.array(pop_hist['pop1']['w'])), 'b-', linewidth=1.5, label='w pop1')
#     ax.plot(t_all, np.squeeze(np.array(pop_hist['pop2']['w'])), 'r-', linewidth=1.5, label='w pop2')
#     ax.set_xlabel('Time (ms)')
#     ax.set_ylabel('w (dimensionless)')
#     ax.set_title('Pre-training: Adaptation Current w')
#     ax.legend()
#     ax.grid(True, alpha=0.3)
#
#     ax = axes[2, 0]
#     ax.plot(t_all, np.squeeze(np.array(syn_hist['syn_1to2']['R'])), 'b-', linewidth=1.5, label='R')
#     ax.plot(t_all, np.squeeze(np.array(syn_hist['syn_1to2']['U'])), 'g-', linewidth=1.5, label='U')
#     ax.plot(t_all, np.squeeze(np.array(syn_hist['syn_1to2']['A'])), 'm-', linewidth=1.5, label='A')
#     ax.set_xlabel('Time (ms)')
#     ax.set_ylabel('State')
#     ax.set_title('Pre-training: Synapse Pop1->Pop2: R, U, A')
#     ax.legend()
#     ax.grid(True, alpha=0.3)
#
#     ax = axes[2, 1]
#     ax.plot(t_all, np.squeeze(np.array(syn_hist['syn_2to1']['R'])), 'b-', linewidth=1.5, label='R')
#     ax.plot(t_all, np.squeeze(np.array(syn_hist['syn_2to1']['U'])), 'g-', linewidth=1.5, label='U')
#     ax.plot(t_all, np.squeeze(np.array(syn_hist['syn_2to1']['A'])), 'm-', linewidth=1.5, label='A')
#     ax.set_xlabel('Time (ms)')
#     ax.set_ylabel('State')
#     ax.set_title('Pre-training: Synapse Pop2->Pop1: R, U, A')
#     ax.legend()
#     ax.grid(True, alpha=0.3)
#
#     ax = axes[3, 0]
#     rates_p1 = r_p1 / (dt * 1e-3)
#     rates_p2 = r_p2 / (dt * 1e-3)
#     ax.plot(t_all, rates_p1, 'b-', linewidth=1.5, label='Pop1 (Hz)')
#     ax.plot(t_all, rates_p2, 'r-', linewidth=1.5, label='Pop2 (Hz)')
#     ax.set_xlabel('Time (ms)')
#     ax.set_ylabel('Firing Rate (Hz)')
#     ax.set_title('Pre-training: Firing Rates in Hz')
#     ax.legend()
#     ax.grid(True, alpha=0.3)
#
#     ax = axes[3, 1]
#     phase = np.arctan2(rates_p2, rates_p1)
#     ax.plot(t_all, phase, 'k-', linewidth=1.5)
#     ax.set_xlabel('Time (ms)')
#     ax.set_ylabel('Phase (rad)')
#     ax.set_title('Pre-training: Phase (arctan2(Pop2, Pop1))')
#     ax.grid(True, alpha=0.3)
#
#     plt.tight_layout()
#     plt.savefig('pretraining_dynamics.png', dpi=150)
#     print("Pre-training dynamics saved to pretraining_dynamics.png")
#     plt.close()
#
# except ImportError as e:
#     print(f"Matplotlib not available: {e}")

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

print("\n=== Optimization ({nbatches} batches, {nepochs} epochs) ===")

print(f"Starting {nepochs}-epoch optimization with {nbatches} batches each...")

current_pop_state = pre_train_pop_state
current_syn_state = pre_train_syn_state

train_log = []

for epoch in range(nepochs):
    epoch_losses = []
    epoch_r = []
    epoch_grads_norm = []
    nan_detected = False

    for b in range(nbatches):
        t_batch = get_batch_tensor(b)
        target_batch = get_batch_target(b)

        loss_batch = CompositeLoss([
            (1.0, MSLELoss(target_batch)),
            (1e-3, StabilityPenalty()),
        ])

        with tf.GradientTape() as tape:
            output = network(t_batch, training=True, initial_state=(current_pop_state, current_syn_state))
            
            # msle_loss = MSLELoss(target_batch)(output, network)
            msle_loss = MSLELoss(target_batch)(output, network)
            stab_loss = StabilityPenalty()(output, network)
            

            
            loss = loss_batch(output, network)

        grads = tape.gradient(loss, network.trainable_variables)

        grads_and_vars = [(g, v) for g, v in zip(grads, network.trainable_variables) if g is not None]

        if grads_and_vars:
            grads_only = [g for g, v in grads_and_vars]
            clipped_grads, _ = tf.clip_by_global_norm(grads_only, 1.0)
            grads_and_vars = [(g, v) for g, (_, v) in zip(clipped_grads, grads_and_vars)]

            optimizer.apply_gradients(grads_and_vars)

        current_pop_state, current_syn_state = network.get_state(force_compute=True)
        
        r_val = float(current_pop_state[0].numpy()[0, 0])

        epoch_losses.append(loss.numpy())
        epoch_r.append(r_val)


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
    print(f"{v.name}: {float(v.numpy().flatten()[0]):.4f}")

print("\n=== Visualization ===")
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    t_values = t_all[:n_steps_per_batch * nbatches]

    ax = axes[0]
    ax.plot(t_values, target_full[:n_steps_per_batch * nbatches, 0], 'b--', label='Target Pop1', linewidth=2)
    ax.plot(t_values, final_rates_pop1[0, :], 'b-', label='Actual Pop1', linewidth=1.5, alpha=0.8)
    ax.plot(t_values, target_full[:n_steps_per_batch * nbatches, 1], 'r--', label='Target Pop2', linewidth=2)
    ax.plot(t_values, final_rates_pop2[0, :], 'r-', label='Actual Pop2', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Firing Rate (Hz)')
    ax.set_title('Target vs Actual Firing Rates (after optimization)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    diff_pop1 = target_full[:n_steps_per_batch * nbatches, 0] - final_rates_pop1[0, :]
    diff_pop2 = target_full[:n_steps_per_batch * nbatches, 1] - final_rates_pop2[0, :]
    ax.plot(t_values, diff_pop1, 'b-', label='Error Pop1', linewidth=1.5)
    ax.plot(t_values, diff_pop2, 'r-', label='Error Pop2', linewidth=1.5)
    ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Error (Hz)')
    ax.set_title('Prediction Error (Target - Actual)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    var_names = [v.name.split(':')[0] for v in network.trainable_variables]
    var_values = [float(v.numpy().flatten()[0]) for v in network.trainable_variables]
    colors = ['steelblue' if 'gsyn' in n else 'coral' for n in var_names]
    ax.bar(range(len(var_names)), var_values, color=colors)
    ax.set_xticks(range(len(var_names)))
    ax.set_xticklabels(var_names, rotation=45, ha='right')
    ax.set_ylabel('Value')
    ax.set_title('Optimized Parameters')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('optimization_visualization.png', dpi=150)
    print("Visualization saved to optimization_visualization.png")
    plt.close()

except ImportError as e:
    print(f"Matplotlib not available: {e}")

trainer.export_results('optimization_results.json')
trainer.export_results('optimization_results.csv', format='csv')
print("\nResults exported to optimization_results.json and optimization_results.csv")

print("\n=== Full simulation with optimal parameters ===")

network.reset_state()
t_seq_full = tf.constant(t_all[None, :, None], dtype=tf.float32)

pop_hist = {
    'pop1': {'r': [], 'v': [], 'w': []},
    'pop2': {'r': [], 'v': [], 'w': []},
}
syn_hist = {
    'syn_1to2': {'R': [], 'U': [], 'A': []},
    'syn_2to1': {'R': [], 'U': [], 'A': []},
}

pop_states, syn_states = None, None

for step in range(t_all.shape[0]):
    t = tf.constant([[t_all[step]]], dtype=tf.float32)
    
    if step == 0:
        output = network(t, training=False, initial_state=(initial_pop_state, initial_syn_state))
    else:
        output = network(t, training=False, initial_state=(pop_states, syn_states))
    
    pop_states, syn_states = network.get_state(force_compute=True)
    
    pop_hist['pop1']['r'].append(pop_states[0][0].numpy())
    pop_hist['pop1']['v'].append(pop_states[1][0].numpy())
    pop_hist['pop1']['w'].append(pop_states[2][0].numpy())
    pop_hist['pop2']['r'].append(pop_states[3][0].numpy())
    pop_hist['pop2']['v'].append(pop_states[4][0].numpy())
    pop_hist['pop2']['w'].append(pop_states[5][0].numpy())
    
    syn_hist['syn_1to2']['R'].append(syn_states[0][0].numpy())
    syn_hist['syn_1to2']['U'].append(syn_states[1][0].numpy())
    syn_hist['syn_1to2']['A'].append(syn_states[2][0].numpy())
    syn_hist['syn_2to1']['R'].append(syn_states[3][0].numpy())
    syn_hist['syn_2to1']['U'].append(syn_states[4][0].numpy())
    syn_hist['syn_2to1']['A'].append(syn_states[5][0].numpy())

print(f"Simulation completed: {t_all.shape[0]} steps")

print("\n=== Optimal Dynamics Visualization ===")
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 2, figsize=(14, 12))

    ax = axes[0, 0]
    r_pop1 = np.squeeze(np.array(pop_hist['pop1']['r']))
    r_pop2 = np.squeeze(np.array(pop_hist['pop2']['r']))
    ax.plot(t_all, r_pop1, 'b-', linewidth=1.5, label='pop1')
    ax.plot(t_all, r_pop2, 'r-', linewidth=1.5, label='pop2')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('r (dimensionless)')
    ax.set_title('Firing Rate r (dimensionless)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    rates_pop1_hz = r_pop1 / (dt * 1e-3)
    rates_pop2_hz = r_pop2 / (dt * 1e-3)
    ax.plot(t_all, target_full[:, 0], 'b--', linewidth=2, label='Target pop1')
    ax.plot(t_all, rates_pop1_hz, 'b-', linewidth=1.5, alpha=0.8, label='pop1')
    ax.plot(t_all, target_full[:, 1], 'r--', linewidth=2, label='Target pop2')
    ax.plot(t_all, rates_pop2_hz, 'r-', linewidth=1.5, alpha=0.8, label='pop2')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Firing Rate (Hz)')
    ax.set_title('Firing Rate (Hz) with Target')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    v_pop1 = np.squeeze(np.array(pop_hist['pop1']['v']))
    v_pop2 = np.squeeze(np.array(pop_hist['pop2']['v']))
    ax.plot(t_all, v_pop1, 'b-', linewidth=1.5, label='v pop1')
    ax.plot(t_all, v_pop2, 'r-', linewidth=1.5, label='v pop2')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('v (dimensionless)')
    ax.set_title('Mean Membrane Potential v')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    w_pop1 = np.squeeze(np.array(pop_hist['pop1']['w']))
    w_pop2 = np.squeeze(np.array(pop_hist['pop2']['w']))
    ax.plot(t_all, w_pop1, 'b-', linewidth=1.5, label='w pop1')
    ax.plot(t_all, w_pop2, 'r-', linewidth=1.5, label='w pop2')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('w (dimensionless)')
    ax.set_title('Adaptation Current w')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2, 0]
    R_1to2 = np.squeeze(np.array(syn_hist['syn_1to2']['R']))
    U_1to2 = np.squeeze(np.array(syn_hist['syn_1to2']['U']))
    A_1to2 = np.squeeze(np.array(syn_hist['syn_1to2']['A']))
    ax.plot(t_all, R_1to2, 'b-', linewidth=1.5, label='R')
    ax.plot(t_all, U_1to2, 'g-', linewidth=1.5, label='U')
    ax.plot(t_all, A_1to2, 'm-', linewidth=1.5, label='A')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('State')
    ax.set_title('Synapse Pop1->Pop2: R, U, A')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    R_2to1 = np.squeeze(np.array(syn_hist['syn_2to1']['R']))
    U_2to1 = np.squeeze(np.array(syn_hist['syn_2to1']['U']))
    A_2to1 = np.squeeze(np.array(syn_hist['syn_2to1']['A']))
    ax.plot(t_all, R_2to1, 'b-', linewidth=1.5, label='R')
    ax.plot(t_all, U_2to1, 'g-', linewidth=1.5, label='U')
    ax.plot(t_all, A_2to1, 'm-', linewidth=1.5, label='A')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('State')
    ax.set_title('Synapse Pop2->Pop1: R, U, A')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[3, 0]
    ax.plot(t_all, rates_pop1_hz, 'b-', linewidth=1.5, label='Pop1 (Hz)')
    ax.plot(t_all, rates_pop2_hz, 'r-', linewidth=1.5, label='Pop2 (Hz)')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Firing Rate (Hz)')
    ax.set_title('Final Firing Rates in Hz')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[3, 1]
    phase = np.arctan2(rates_pop2_hz, rates_pop1_hz)
    ax.plot(t_all, phase, 'k-', linewidth=1.5)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Phase (rad)')
    ax.set_title('Phase (arctan2(Pop2, Pop1))')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('optimal_dynamics.png', dpi=150)
    print("Optimal dynamics visualization saved to optimal_dynamics.png")
    plt.close()

except ImportError as e:
    print(f"Matplotlib not available: {e}")