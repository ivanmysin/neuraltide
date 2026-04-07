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
from neuraltide.utils import seed_everything
import neuraltide.config

seed_everything(42)

DT = 0.1
T = 100

print("=" * 60)
print("Example 1: Dimensionless parameters (recommended)")
print("=" * 60)

n_units = 2
params_dimless = {
    'tau_pop': {'value': [1.0, 1.0], 'trainable': False},
    'alpha': {'value': [0.5, 0.5], 'trainable': False},
    'a': {'value': [0.02, 0.02], 'trainable': False},
    'b': {'value': [0.2, 0.2], 'trainable': False},
    'w_jump': {'value': [0.1, 0.1], 'trainable': False},
    'Delta_I': {'value': [0.5, 0.6], 'trainable': True},
    'I_ext': {'value': [1.0, 1.2], 'trainable': True},
}

pop_dimless = IzhikevichMeanField(
    n_units=n_units,
    dt=DT,
    params=params_dimless,
)

print(f"Created IzhikevichMeanField with {pop_dimless.n_units} units")
print(f"Trainable parameters: {[v.name for v in pop_dimless.trainable_variables]}")
print(f"Parameter spec: {pop_dimless.parameter_spec}")

print("\n" + "=" * 60)
print("Example 2: Dimensional parameters (backward compatibility)")
print("=" * 60)

V_R = -57.6
V_T = -35.5
V_peak = 21.7
V_reset = -48.7
C = 114.0
K = 1.194
A = 0.0046
B = 0.2157
W_jump = 2.0
Delta_I_dimensional = 20.0
I_ext_dimensional = 500.0

pop_dim = IzhikevichMeanField(
    n_units=1,
    dt=DT,
    V_R=V_R,
    V_T=V_T,
    V_peak=V_peak,
    V_reset=V_reset,
    C=C,
    K=K,
    A=A,
    B=B,
    W_jump=W_jump,
    Delta_I=Delta_I_dimensional,
    I_ext=I_ext_dimensional,
)

print("\nDimensional Parameters (Basket Cell):")
print(f"  V_R = {V_R} mV, V_T = {V_T} mV, V_peak = {V_peak} mV, V_reset = {V_reset} mV")
print(f"  C = {C} pF, K = {K} nS/mV")
print(f"  A = {A} ms^-1, B = {B} nS")
print(f"  W_jump = {W_jump} pA, Delta_I = {Delta_I_dimensional} pA, I_ext = {I_ext_dimensional} pA")

print("\nConverted Dimensionless Parameters:")
print(f"  tau_pop = {pop_dim.tau_pop.numpy()[0]:.4f} ms")
print(f"  alpha = {pop_dim.alpha.numpy()[0]:.4f}")
print(f"  a = {pop_dim.a.numpy()[0]:.6f}")
print(f"  b = {pop_dim.b.numpy()[0]:.6f}")
print(f"  w_jump = {pop_dim.w_jump.numpy()[0]:.6f}")
print(f"  Delta_I = {pop_dim.Delta_I.numpy()[0]:.6f}")
print(f"  I_ext = {pop_dim.I_ext.numpy()[0]:.6f}")

print("\nVerifying conversion matches static method:")
dimless_from_static = IzhikevichMeanField.dimensional_to_dimensionless(
    V_R, V_T, V_peak, V_reset, C, K, A, B, W_jump, Delta_I_dimensional, I_ext_dimensional
)
print(f"  Static method Delta_I: {dimless_from_static['Delta_I']:.6f}")
print(f"  Model Delta_I: {pop_dim.Delta_I.numpy()[0]:.6f}")

print("\n" + "=" * 60)
print("Example 3: Simulation run with dimensionless parameters")
print("=" * 60)

integrator = RK4Integrator()

pop = IzhikevichMeanField(
    n_units=1,
    dt=DT,
    params={
        'tau_pop': {'value': 1.0, 'trainable': False},
        'alpha': {'value': 0.5, 'trainable': False},
        'a': {'value': 0.02, 'trainable': False},
        'b': {'value': 0.2, 'trainable': False},
        'w_jump': {'value': 0.1, 'trainable': False},
        'Delta_I': {'value': 0.5, 'trainable': False},
        'I_ext': {'value': 1.0, 'trainable': False},
    }
)

t_values = np.arange(0, T, DT, dtype=np.float32)
n_steps = len(t_values)

state = pop.get_initial_state()
r_hist = []
v_hist = []
w_hist = []
fr_hist = []

dtype = neuraltide.config.get_dtype()
zero_I = tf.zeros([1, pop.n_units], dtype=dtype)
zero_syn = {'I_syn': zero_I, 'g_syn': zero_I}

for step in range(n_steps):
    r, v, w = state
    r_hist.append(r[0].numpy())
    v_hist.append(v[0].numpy())
    w_hist.append(w[0].numpy())
    fr_hist.append(pop.get_firing_rate(state)[0].numpy())

    state, _ = integrator.step(pop, state, zero_syn)

r_hist = np.array(r_hist)
v_hist = np.array(v_hist)
w_hist = np.array(w_hist)
fr_hist = np.array(fr_hist)

print(f"\nSimulation completed:")
print(f"  Steps: {n_steps}")
print(f"  Time: {T} ms")
print(f"  dt: {DT} ms")
print(f"  Final firing rate (dimensionless): {fr_hist[-1, 0]:.4f}")
print(f"  Final v (dimensionless): {v_hist[-1, 0]:.4f}")
print(f"  Final w (dimensionless): {w_hist[-1, 0]:.4f}")

fig, axes = plt.subplots(4, 1, figsize=(14, 12))

axes[0].plot(t_values, fr_hist[:, 0], color='tab:blue', linewidth=1.0)
axes[0].set_ylabel('r (dimensionless)')
axes[0].set_title(r'Population Firing Rate $r$ ($\tau_{{pop}}$ = {:.2f} ms)'.format(pop.tau_pop.numpy()[0]))
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim([0, T])

axes[1].plot(t_values, v_hist[:, 0], color='tab:orange', linewidth=1.0)
axes[1].axhline(pop.alpha.numpy()[0], color='red', linestyle=':', linewidth=1, 
                label=f'alpha={pop.alpha.numpy()[0]:.2f}')
axes[1].set_ylabel('v (dimensionless)')
axes[1].set_title(r'Dimensionless Mean Membrane Potential ($\langle v \rangle$)')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim([0, T])

axes[2].plot(t_values, r_hist[:, 0], color='tab:blue', linewidth=1.0, alpha=0.7)
axes[2].set_ylabel('r (dimensionless)')
axes[2].set_title(r'Dimensionless Firing Rate ($\nu$)')
axes[2].grid(True, alpha=0.3)
axes[2].set_xlim([0, T])

axes[3].plot(t_values, w_hist[:, 0], color='tab:green', linewidth=1.0)
axes[3].set_ylabel('w (dimensionless)')
axes[3].set_title(r'Adaptation Variable ($\langle w \rangle$)')
axes[3].set_xlabel('Time (ms)')
axes[3].grid(True, alpha=0.3)
axes[3].set_xlim([0, T])

fig.suptitle(
    f"IzhikevichMeanField: Mean-Field Simulation\n"
    f"$\\alpha$={pop.alpha.numpy()[0]:.2f}, $a$={pop.a.numpy()[0]:.2f}, "
    f"$b$={pop.b.numpy()[0]:.2f}, $\\Delta_I$={pop.Delta_I.numpy()[0]:.2f}, "
    f"$I_{{ext}}$={pop.I_ext.numpy()[0]:.2f}",
    fontsize=12
)
plt.tight_layout()
plt.savefig("example_pop_izhikevich.png", dpi=150)
print(f"\nFigure saved as example_pop_izhikevich.png")

print("\n" + "=" * 60)
print("Example 4: Multi-unit simulation with different parameters")
print("=" * 60)

n_units_multi = 4
params_multi = {
    'tau_pop': {'value': 1.0, 'trainable': False},
    'alpha': {'value': 0.5, 'trainable': False},
    'a': {'value': 0.02, 'trainable': False},
    'b': {'value': 0.2, 'trainable': False},
    'w_jump': {'value': 0.1, 'trainable': False},
    'Delta_I': {'value': [0.3, 0.5, 0.7, 0.9], 'trainable': True},
    'I_ext': {'value': [0.8, 1.0, 1.2, 1.4], 'trainable': True},
}

pop_multi = IzhikevichMeanField(
    n_units=n_units_multi,
    dt=DT,
    params=params_multi,
)

print(f"Created multi-unit population with {pop_multi.n_units} units")
print(f"I_ext per unit: {pop_multi.I_ext.numpy()}")
print(f"Delta_I per unit: {pop_multi.Delta_I.numpy()}")

state_multi = pop_multi.get_initial_state()
fr_multi_hist = []

for step in range(n_steps):
    fr_multi_hist.append(pop_multi.get_firing_rate(state_multi)[0].numpy())
    state_multi, _ = integrator.step(pop_multi, state_multi, zero_syn)

fr_multi_hist = np.array(fr_multi_hist)

fig2, axes2 = plt.subplots(2, 1, figsize=(14, 8))

for i in range(n_units_multi):
    axes2[0].plot(t_values, fr_multi_hist[:, i], linewidth=1.0,
                   label=f'Unit {i+1} (I_ext={params_multi["I_ext"]["value"][i]:.1f})')

axes2[0].set_ylabel('r (dimensionless)')
axes2[0].set_title('Multi-Unit Firing Rates (different I_ext per unit)')
axes2[0].legend()
axes2[0].grid(True, alpha=0.3)
axes2[0].set_xlim([0, T])

state_v = pop_multi.get_initial_state()
v_hist_multi = []
for step in range(n_steps):
    v_hist_multi.append(state_v[1].numpy())
    state_v, _ = integrator.step(pop_multi, state_v, zero_syn)
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
print("Example 5: Dimensional parameters with per-unit vectors")
print("=" * 60)

pop_vec_dim = IzhikevichMeanField(
    n_units=3,
    dt=DT,
    V_R=[-57.6, -55.0, -53.0],
    V_T=[-35.5, -34.0, -33.0],
    V_peak=[21.7, 20.0, 19.0],
    V_reset=[-48.7, -45.0, -42.0],
    C=[114.0, 100.0, 90.0],
    K=[1.194, 1.2, 1.1],
    A=[0.0046, 0.005, 0.004],
    B=[0.2157, 0.2, 0.18],
    W_jump=[2.0, 1.8, 1.5],
    Delta_I=[20.0, 18.0, 15.0],
    I_ext=[120.0, 100.0, 80.0],
)

print(f"Created population with 3 units from vectorial dimensional params")
print(f"I_ext per unit: {pop_vec_dim.I_ext.numpy()}")
print(f"Delta_I per unit: {pop_vec_dim.Delta_I.numpy()}")
print(f"tau_pop per unit: {pop_vec_dim.tau_pop.numpy()}")

print("\n" + "=" * 60)
print("All examples completed successfully!")
print("=" * 60)
