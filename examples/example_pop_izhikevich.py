"""
Example: IzhikevichMeanField population dynamics with constant input current.

This example demonstrates:
1. Using dimensionless parameters (recommended approach)
2. Using dimensional parameters (backward compatibility)
3. Vectorized operations with multiple units (n_units > 1)
4. Trainable parameters with constraints

The model implements the mean-field Izhikevich equations from Chen & Campbell 2022:
    tau_pop * d_nu/dt = Delta_I/pi + 2*nu*<v> - (alpha + g_syn_tot)*nu
    tau_pop * d<v>/dt = <v>^2 - alpha*<v> - <w> + I_ext + I_syn - (pi*nu)^2
    tau_pop * d<w>/dt = a*(b*<v> - <w>) + w_jump*nu

Mean-field simulation only (no point-neuron simulation).
"""

import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from neuraltide.populations.izhikevich_mf import IzhikevichMeanField
from neuraltide.integrators import RK4Integrator
import neuraltide.config


DT = 0.05
T = 200

integrator = RK4Integrator()
dtype = neuraltide.config.get_dtype()

t_values = np.arange(0, T, DT, dtype=np.float32)
n_steps = len(t_values)

print("=" * 60)
print("Example 1: Dimensionless vs Dimensional parameters")
print("=" * 60)
print("\nThis example demonstrates that running IzhikevichMeanField with")
print("dimensional and dimensionless parameters gives identical results.\n")

V_rest = -57.63
K = 1.19
Cm = 114
V_T = -35.53
A = 0.005
B = 0.22
V_peak = 20.0
V_reset = -50.0
W_jump = 2.0
Delta_I_dimensional = 25
I_ext_dimensional = 500

print("Dimensional Parameters (Basket Cell):")
print(f"  V_rest = {V_rest} mV, V_T = {V_T} mV, V_peak = {V_peak} mV, V_reset = {V_reset} mV")
print(f"  Cm = {Cm:.4f} pF, K = {K} nS/mV")
print(f"  A = {A:.6f} ms^-1, B = {B:.4f} nS")
print(f"  W_jump = {W_jump:.4f} pA, Delta_I = {Delta_I_dimensional:.4f} pA, I_ext = {I_ext_dimensional:.4f} pA")

dim_params = dict(
    V_rest=V_rest,
    V_T=V_T,
    V_peak=V_peak,
    V_reset=V_reset,
    Cm=Cm,
    K=K,
    A=A,
    B=B,
    W_jump=W_jump,
    Delta_I=Delta_I_dimensional,
    I_ext=I_ext_dimensional,
)

pop_dimensional = IzhikevichMeanField(dt=DT, params=dim_params)

dimless_params = pop_dimensional._build_params_from_dimensional(dim_params)

def to_float(v):
    if hasattr(v, 'numpy'):
        arr = v.numpy()
        if arr.ndim == 1:
            arr = arr[0]
        return float(arr)
    return float(v)

print("\nConverted Dimensionless Parameters:")
print(f"  tau_pop = {to_float(dimless_params['tau_pop']):.6f}")
print(f"  alpha = {to_float(dimless_params['alpha']):.6f}")
print(f"  a = {to_float(dimless_params['a']):.6f}")
print(f"  b = {to_float(dimless_params['b']):.6f}")
print(f"  w_jump = {to_float(dimless_params['w_jump']):.6f}")
print(f"  Delta_I = {to_float(dimless_params['Delta_I']):.6f}")
print(f"  I_ext = {to_float(dimless_params['I_ext']):.6f}")

params_dimensionless = {
    'tau_pop': {'value': to_float(dimless_params['tau_pop']), 'trainable': False},
    'alpha': {'value': to_float(dimless_params['alpha']), 'trainable': False},
    'a': {'value': to_float(dimless_params['a']), 'trainable': False},
    'b': {'value': to_float(dimless_params['b']), 'trainable': False},
    'w_jump': {'value': to_float(dimless_params['w_jump']), 'trainable': False},
    'Delta_I': {'value': to_float(dimless_params['Delta_I']), 'trainable': False},
    'I_ext': {'value': to_float(dimless_params['I_ext']), 'trainable': False},
}

pop_dimensionless = IzhikevichMeanField(dt=DT, params=params_dimensionless)

print("\nVerifying converted parameters match:")
print(f"  tau_pop: dimensionless={pop_dimensionless.tau_pop.numpy()[0]:.6f}, dimensional={pop_dimensional.tau_pop.numpy()[0]:.6f}")
print(f"  alpha: dimensionless={pop_dimensionless.alpha.numpy()[0]:.6f}, dimensional={pop_dimensional.alpha.numpy()[0]:.6f}")
print(f"  a: dimensionless={pop_dimensionless.a.numpy()[0]:.6f}, dimensional={pop_dimensional.a.numpy()[0]:.6f}")
print(f"  b: dimensionless={pop_dimensionless.b.numpy()[0]:.6f}, dimensional={pop_dimensional.b.numpy()[0]:.6f}")
print(f"  w_jump: dimensionless={pop_dimensionless.w_jump.numpy()[0]:.6f}, dimensional={pop_dimensional.w_jump.numpy()[0]:.6f}")
print(f"  Delta_I: dimensionless={pop_dimensionless.Delta_I.numpy()[0]:.6f}, dimensional={pop_dimensional.Delta_I.numpy()[0]:.6f}")
print(f"  I_ext: dimensionless={pop_dimensionless.I_ext.numpy()[0]:.6f}, dimensional={pop_dimensional.I_ext.numpy()[0]:.6f}")

state_dimless = pop_dimensionless.get_initial_state()
state_dim = pop_dimensional.get_initial_state()

fr_dimless_hist = []
fr_dim_hist = []

zero_I = tf.zeros([1, 1], dtype=dtype)
zero_syn = {'I_syn': zero_I, 'g_syn': zero_I}

for step in range(n_steps):
    fr_dimless_hist.append(pop_dimensionless.get_firing_rate(state_dimless)[0].numpy())
    fr_dim_hist.append(pop_dimensional.get_firing_rate(state_dim)[0].numpy())

    state_dimless, _ = integrator.step(pop_dimensionless, state_dimless, zero_syn)
    state_dim, _ = integrator.step(pop_dimensional, state_dim, zero_syn)

fr_dimless_hist = np.array(fr_dimless_hist)
fr_dim_hist = np.array(fr_dim_hist)

max_diff = np.max(np.abs(fr_dimless_hist - fr_dim_hist))
print(f"\nSimulation results comparison:")
print(f"  Max difference in firing rates: {max_diff:.10f}")
print(f"  Results are identical: {max_diff < 1e-9}")


print("\n" + "=" * 60)
print("Example 2: Multi-unit simulation with dimensionless parameters")
print("=" * 60)

params_multi = {
    'tau_pop': {'value': [1.0, 1.0], 'trainable': False},
    'alpha': {'value': [0.5, 0.5], 'trainable': False},
    'a': {'value': [0.02, 0.02], 'trainable': False},
    'b': {'value': [0.2, 0.2], 'trainable': False},
    'w_jump': {'value': [0.1, 0.1], 'trainable': False},
    'Delta_I': {'value': [0.5, 0.6], 'trainable': True},
    'I_ext': {'value': [1.0, 1.2], 'trainable': True},
}

pop_multi = IzhikevichMeanField(dt=DT, params=params_multi)

print(f"Created IzhikevichMeanField with {pop_multi.n_units} units")
print(f"Trainable parameters: {[v.name for v in pop_multi.trainable_variables]}")
print(f"Parameter spec: {pop_multi.parameter_spec}")

state_multi = pop_multi.get_initial_state()
r_hist = []
v_hist = []
w_hist = []
fr_hist = []

for step in range(n_steps):
    r, v, w = state_multi
    r_hist.append(r[0].numpy())
    v_hist.append(v[0].numpy())
    w_hist.append(w[0].numpy())
    fr_hist.append(pop_multi.get_firing_rate(state_multi)[0].numpy())
    state_multi, _ = integrator.step(pop_multi, state_multi, zero_syn)

r_hist = np.array(r_hist)
v_hist = np.array(v_hist)
w_hist = np.array(w_hist)
fr_hist = np.array(fr_hist)

print(f"\nSimulation completed:")
print(f"  Steps: {n_steps}")
print(f"  Time: {T} ms")
print(f"  dt: {DT} ms")
print(f"  Final firing rate (unit 0): {fr_hist[-1, 0]:.4f}")
print(f"  Final firing rate (unit 1): {fr_hist[-1, 1]:.4f}")
print(f"  Final v (unit 0): {v_hist[-1, 0]:.4f}")
print(f"  Final v (unit 1): {v_hist[-1, 1]:.4f}")

fig, axes = plt.subplots(4, 1, figsize=(14, 12))

axes[0].plot(t_values, fr_hist[:, 0], color='tab:blue', linewidth=1.0, label='Unit 0')
axes[0].plot(t_values, fr_hist[:, 1], color='tab:orange', linewidth=1.0, label='Unit 1')
axes[0].set_ylabel('r (dimensionless)')
axes[0].set_title(r'Population Firing Rate $r$ ($\tau_{{pop}}$ = {:.2f} ms)'.format(pop_multi.tau_pop.numpy()[0]))
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim([0, T])

axes[1].plot(t_values, v_hist[:, 0], color='tab:blue', linewidth=1.0, label='Unit 0')
axes[1].plot(t_values, v_hist[:, 1], color='tab:orange', linewidth=1.0, label='Unit 1')
axes[1].axhline(pop_multi.alpha.numpy()[0], color='red', linestyle=':', linewidth=1,
                label=f'alpha={pop_multi.alpha.numpy()[0]:.2f}')
axes[1].set_ylabel('v (dimensionless)')
axes[1].set_title(r'Dimensionless Mean Membrane Potential ($\langle v \rangle$)')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim([0, T])

axes[2].plot(t_values, r_hist[:, 0], color='tab:blue', linewidth=1.0, alpha=0.7, label='Unit 0')
axes[2].plot(t_values, r_hist[:, 1], color='tab:orange', linewidth=1.0, alpha=0.7, label='Unit 1')
axes[2].set_ylabel('r (dimensionless)')
axes[2].set_title(r'Dimensionless Firing Rate ($\nu$)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)
axes[2].set_xlim([0, T])

axes[3].plot(t_values, w_hist[:, 0], color='tab:blue', linewidth=1.0, label='Unit 0')
axes[3].plot(t_values, w_hist[:, 1], color='tab:orange', linewidth=1.0, label='Unit 1')
axes[3].set_ylabel('w (dimensionless)')
axes[3].set_title(r'Adaptation Variable ($\langle w \rangle$)')
axes[3].set_xlabel('Time (ms)')
axes[3].legend()
axes[3].grid(True, alpha=0.3)
axes[3].set_xlim([0, T])

fig.suptitle(
    f"IzhikevichMeanField: Mean-Field Simulation\n"
    f"$\\alpha$={pop_multi.alpha.numpy()[0]:.2f}, $a$={pop_multi.a.numpy()[0]:.2f}, "
    f"$b$={pop_multi.b.numpy()[0]:.2f}, $\\Delta_I$={pop_multi.Delta_I.numpy()[0]:.2f}, "
    f"$I_{{ext}}$={pop_multi.I_ext.numpy()[0]:.2f}",
    fontsize=12
)
plt.tight_layout()
plt.savefig("example_pop_izhikevich.png", dpi=150)
print(f"\nFigure saved as example_pop_izhikevich.png")

print("\n" + "=" * 60)
print("Example 3: Multi-unit simulation with different parameters per unit")
print("=" * 60)

n_units_multi = 4
params_multi_diff = {
    'tau_pop': {'value': 1.0, 'trainable': False},
    'alpha': {'value': 0.5, 'trainable': False},
    'a': {'value': 0.02, 'trainable': False},
    'b': {'value': 0.2, 'trainable': False},
    'w_jump': {'value': 0.1, 'trainable': False},
    'Delta_I': {'value': [0.3, 0.5, 0.7, 0.9], 'trainable': True},
    'I_ext': {'value': [0.8, 1.0, 1.2, 1.4], 'trainable': True},
}

pop_multi_diff = IzhikevichMeanField(dt=DT, params=params_multi_diff)

print(f"Created multi-unit population with {pop_multi_diff.n_units} units")
print(f"I_ext per unit: {pop_multi_diff.I_ext.numpy()}")
print(f"Delta_I per unit: {pop_multi_diff.Delta_I.numpy()}")

state_multi_diff = pop_multi_diff.get_initial_state()
fr_multi_hist = []

zero_I_multi = tf.zeros([1, n_units_multi], dtype=dtype)
zero_syn_multi = {'I_syn': zero_I_multi, 'g_syn': zero_I_multi}

for step in range(n_steps):
    fr_multi_hist.append(pop_multi_diff.get_firing_rate(state_multi_diff)[0].numpy())
    state_multi_diff, _ = integrator.step(pop_multi_diff, state_multi_diff, zero_syn_multi)

fr_multi_hist = np.array(fr_multi_hist)

fig2, axes2 = plt.subplots(2, 1, figsize=(14, 8))

for i in range(n_units_multi):
    axes2[0].plot(t_values, fr_multi_hist[:, i], linewidth=1.0,
                   label=f'Unit {i+1} (I_ext={params_multi_diff["I_ext"]["value"][i]:.1f})')

axes2[0].set_ylabel('r (dimensionless)')
axes2[0].set_title('Multi-Unit Firing Rates (different I_ext per unit)')
axes2[0].legend()
axes2[0].grid(True, alpha=0.3)
axes2[0].set_xlim([0, T])

state_v = pop_multi_diff.get_initial_state()
v_hist_multi = []
for step in range(n_steps):
    v_hist_multi.append(state_v[1].numpy())
    state_v, _ = integrator.step(pop_multi_diff, state_v, zero_syn_multi)
v_hist_multi = np.array(v_hist_multi)
for i in range(n_units_multi):
    axes2[1].plot(t_values, v_hist_multi[:, 0, i], linewidth=1.0, label=f'Unit {i+1}')

axes2[1].set_ylabel('v (dimensionless)')
axes2[1].set_xlabel('Time (ms)')
axes2[1].set_title('Dimensionless Mean Membrane Potential')
axes2[1].legend()
axes2[1].grid(True, alpha=0.3)
axes2[1].set_xlim([0, T])

fig2.suptitle(
    f"Multi-Unit IzhikevichMeanField Simulation\n"
    f"Different I_ext and Delta_I per unit, vectorized computation",
    fontsize=12
)
plt.tight_layout()
plt.savefig("example_pop_izhikevich_multi.png", dpi=150)
print(f"\nMulti-unit figure saved as example_pop_izhikevich_multi.png")


print("\n" + "=" * 60)
print("All examples completed successfully!")
print("=" * 60)
