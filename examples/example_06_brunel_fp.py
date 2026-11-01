"""
Пример: BrunelFokkerPlanck — популяция ЛИФ-нейронов в приближении
Фоккера–Планка (Brunel, 2000, модель A).

Демонстрируется:
- Сходимость к стационарной частоте разрядов
- Сравнение с аналитическим решением (уравнение (21) из статьи)
- Сохранение полной вероятности ∫P dV + p_r = 1
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.integrate import quad
from scipy.special import erfcx

from neuraltide.populations.brunel_fp import BrunelFokkerPlanck
from neuraltide.integrators import EulerIntegrator
from neuraltide.utils import seed_everything

seed_everything(42)

dt = 0.5
T = 1500

# Параметры Brunel (2000): τ=20 мс, θ=20 мВ, V_r=10 мВ, τ_rp=2 мс.
# Режим у порога: μ0=18 мВ, σ=3 мВ.
tau, tau_rp, mu_0, sigma_ext = 20.0, 2.0, 18.0, 3.0

pop = BrunelFokkerPlanck(
    dt=dt,
    params={
        "tau_m": {"value": tau, "trainable": False},
        "v_theta": {"value": 20.0, "trainable": False},
        "v_reset": {"value": 10.0, "trainable": False},
        "tau_rp": {"value": tau_rp, "trainable": False},
        "mu_0": {"value": mu_0, "trainable": False},
        "sigma_ext": {"value": sigma_ext, "trainable": False},
    },
    grid_size=250,
    v_min=-80.0,
)

integrator = EulerIntegrator()
zero = tf.zeros([1, 1], dtype=tf.float32)
syn = {"I_syn": zero, "g_syn": zero}

t_values = np.arange(0, T, dt, dtype=np.float32)
rate_hist = []
mass_hist = []

state = pop.get_initial_state()
for step in range(len(t_values)):
    rate_hist.append(pop.get_firing_rate(state).numpy()[0, 0])
    mass_hist.append(np.sum(state[0].numpy()) * pop.dV + state[1].numpy()[0, 0])
    state, _ = integrator.step(pop, state, syn)

rate_hist = np.array(rate_hist)
mass_hist = np.array(mass_hist)

# Аналитическая стационарная частота (уравнения (20)-(21) Brunel 2000)
y_r = (10.0 - mu_0) / sigma_ext
y_th = (20.0 - mu_0) / sigma_ext
integral, _ = quad(lambda u: erfcx(-u), y_r, y_th, epsabs=1e-12)
nu_ref = 1000.0 / (tau_rp + tau * np.sqrt(np.pi) * integral)
nu_sim = np.mean(rate_hist[-400:])
print(
    f"Стационарная частота: симуляция = {nu_sim:.2f} Гц, "
    f"аналитика (21) = {nu_ref:.2f} Гц"
)
print(
    f"Сохранение нормы ∫P dV + p_r: "
    f"min = {mass_hist.min():.6f}, max = {mass_hist.max():.6f}"
)

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].plot(
    t_values,
    rate_hist,
    linewidth=1.0,
    color="tab:blue",
    label="BrunelFokkerPlanck rate",
)
axes[0].axhline(
    nu_ref,
    color="red",
    linestyle="--",
    linewidth=1.5,
    label=f"аналитика (21): {nu_ref:.1f} Гц",
)
axes[0].set_ylabel("Firing rate (Hz)")
axes[0].set_title("Частота разрядов: установление к стационару")
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

axes[1].plot(t_values, mass_hist, linewidth=1.0, color="tab:green")
axes[1].set_ylabel("∫P dV + p_r")
axes[1].set_xlabel("Time (ms)")
axes[1].set_title("Сохранение полной вероятности")
axes[1].grid(True, alpha=0.3)

fig.suptitle(
    f"Brunel-FokkerPlanck: μ0 = {mu_0:.1f} мВ, σ = {sigma_ext:.1f} мВ, "
    f"θ = 20 мВ, V_r = 10 мВ",
    fontsize=13,
)
plt.tight_layout()
plt.savefig("example_brunel_fp.png", dpi=150)
plt.show()
print("Figure saved as example_brunel_fp.png")
