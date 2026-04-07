"""
Пример: Динамика популяции IzhikevichMeanField с постоянным входным током.

Используются параметры нейронов Basket из статьи (таблицы 4, 5).
Демонстрируется:
- Как IzhikevichMeanField отвечает на постоянный ток
- Переменные r (частота), v (безразмерный потенциал), w (безразмерная адаптация)
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

dt = 0.01
T = 100

V_R = -57.6
V_T = -35.5
V_peak = 21.7
V_reset = -48.7
C = 114.0
K = 1.194
A = 0.0046
B = 0.2157
W_jump = 2.0
Delta_eta = 20
eta_bar = 120
E_r = 0.0

V_R_abs = abs(V_R)

pop = IzhikevichMeanField(
    n_units=1,
    dt=dt,
    V_R=V_R,
    V_T=V_T,
    V_peak=V_peak,
    V_reset=V_reset,
    C=C,
    K=K,
    A=A,
    B=B,
    W_jump=W_jump,
    E_r=E_r,
    Delta_I=Delta_eta,
    I_ext=eta_bar,
    eta_bar=eta_bar,
    use_dimensionless=False,
)

dimless = IzhikevichMeanField.dimensional_to_dimensionless(
    V_R, V_T, V_peak, V_reset, C, K, A, B, W_jump, Delta_eta, eta_bar, E_r
)

print("=== Basket Cell Dimensional Parameters ===")
print(f"V_R = {V_R} mV, V_T = {V_T} mV, V_peak = {V_peak} mV, V_reset = {V_reset} mV")
print(f"C = {C} pF, K = {K} nS/mV")
print(f"A = {A} ms, B = {B} nS, W_jump = {W_jump} pA")
print(f"Delta_eta (table) = {Delta_eta}, eta_bar (table) = {eta_bar}")
print()
print("=== Dimensionless Parameters (from model) ===")
print(f"  tau_pop = {pop.tau_pop.numpy()[0]:.4f} ms")
print(f"  alpha = {pop.alpha.numpy()[0]:.4f}")
print(f"  a = {pop.a.numpy()[0]:.6f}")
print(f"  b = {pop.b.numpy()[0]:.6f}")
print(f"  w_jump = {pop.w_jump.numpy()[0]:.6f}")
print(f"  v_peak = {pop.v_peak.numpy()[0]:.4f}")
print(f"  v_reset = {pop.v_reset.numpy()[0]:.4f}")
print(f"  Delta_eta = {pop.Delta_eta.numpy()[0]:.6f} (target: {Delta_eta})")
print(f"  eta_bar = {pop.eta_bar.numpy()[0]:.6f} (target: {eta_bar})")

integrator = RK4Integrator()

t_values = np.arange(0, T, dt, dtype=np.float32)
n_steps = len(t_values)

state = pop.get_initial_state()
r_hist = []
v_hist = []
w_hist = []

dtype = neuraltide.config.get_dtype()
zero_I = tf.zeros([1, pop.n_units], dtype=dtype)
zero_syn = {'I_syn': zero_I, 'g_syn': zero_I}

v_peak_val = pop.v_peak.numpy()[0]
v_reset_val = pop.v_reset.numpy()[0]
w_jump_val = pop.w_jump.numpy()[0]

for step in range(n_steps):
    r = state[0]
    v = state[1]
    w = state[2]
    r_hist.append(r[0].numpy())
    v_hist.append(v[0].numpy())
    w_hist.append(w[0].numpy())

    state, _ = integrator.step(pop, state, zero_syn)

    v_new = state[1]
    v_cond = v_new > v_peak_val
    v_reset_tensor = tf.constant([[v_reset_val]], dtype=dtype)
    w_jump_tensor = tf.constant([[w_jump_val]], dtype=dtype)
    v_reset_final = tf.where(v_cond, v_reset_tensor, v_new)
    w_add = tf.where(v_cond, w_jump_tensor, tf.zeros_like(w))
    state = [state[0], v_reset_final, state[2] + w_add]

r_hist = np.array(r_hist)
v_hist = np.array(v_hist)
w_hist = np.array(w_hist)

v_voltage = V_R_abs * (v_hist - 1.0)

spike_mask = (v_voltage[:, 0] > V_peak).astype(float)
spike_times = t_values[spike_mask > 0]

fig, axes = plt.subplots(4, 1, figsize=(14, 10))

axes[0].plot(t_values, r_hist[:, 0], color='tab:blue', linewidth=1.0)
axes[0].set_ylabel('r (dimensionless)')
axes[0].set_title(r'Population Firing Rate $r$ ($\tau_{{pop}}$ = {:.2f} ms)'.format(pop.tau_pop.numpy()[0]))
axes[0].grid(True, alpha=0.3)

axes[1].plot(t_values, v_voltage[:, 0], color='tab:orange', linewidth=1.0)
axes[1].axhline(V_peak, color='red', linestyle=':', linewidth=1, label=f'V_peak={V_peak} mV')
axes[1].axhline(V_reset, color='green', linestyle=':', linewidth=1, label=f'V_reset={V_reset} mV')
axes[1].axhline(V_R, color='gray', linestyle='--', linewidth=1, label=f'V_R={V_R} mV')
axes[1].set_ylabel('V (mV)')
axes[1].set_title(r'Mean Membrane Potential ($v = 1 + V/|V_R|$)')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

axes[2].plot(t_values, w_hist[:, 0], color='tab:green', linewidth=1.0)
axes[2].set_ylabel('w (dimensionless)')
axes[2].set_title(r'Adaptation Variable ($w = W/(K|V_R|^2)$)')
axes[2].grid(True, alpha=0.3)

axes[3].eventplot(spike_times, lineoffsets=0.5, linelength=0.8, color='tab:red')
axes[3].set_ylabel('Spikes')
axes[3].set_xlabel('Time (ms)')
axes[3].set_title('Spike Times (when V > V_peak)')
axes[3].set_xlim([0, T])
axes[3].grid(True, alpha=0.3)

fig.suptitle(
    f"Basket Cell (IzhikevichMeanField): Constant Current\n"
    f"$V_R$={V_R} mV, $V_T$={V_T} mV, $C$={C} pF, $K$={K} nS/mV, "
    f"$W_{{jump}}$={W_jump} pA, $\\bar{{\\eta}}$={pop.eta_bar.numpy()[0]:.4f}",
    fontsize=12
)
plt.tight_layout()
plt.savefig("example_pop_izhikevich.png", dpi=150)
print(f"\nFigure saved as example_pop_izhikevich.png")
print(f"Number of spikes: {len(spike_times)}")