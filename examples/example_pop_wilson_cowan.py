"""
Пример: Динамика популяции Wilson-Cowan с постоянным входным током.

Каждая популяция описывается одной переменной E:
    τ · dE/dt = −E + F(w·E + I_ext + I_syn),  F — сигмоида
Частота разрядов: relu(E) · max_rate.

Демонстрируется:
- Запуск активности после подачи постоянного тока I_ext
- Установление в стационарное значение (фиксированная точка)
- Разные параметры у двух независимых популяций
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from neuraltide.populations.wilson_cowan import WilsonCowan
from neuraltide.integrators import RK4Integrator
from neuraltide.utils import seed_everything
import neuraltide.config

seed_everything(42)

dt = 0.5
T = 200

pop = WilsonCowan(
    dt=dt,
    params={
        "tau": {"value": [10.0, 10.0], "trainable": False},
        "a": {"value": [1.0, 1.2], "trainable": False},
        "theta": {"value": [0.0, 0.0], "trainable": False},
        "w": {"value": [1.0, 0.8], "trainable": False},
        "I_ext": {"value": [1.0, 0.5], "trainable": False},
        "max_rate": {"value": [100.0, 100.0], "trainable": False},
    },
)

integrator = RK4Integrator()

t_values = np.arange(0, T, dt, dtype=np.float32)
n_steps = len(t_values)

state = pop.get_initial_state()
E_hist = []

dtype = neuraltide.config.get_dtype()
zero_I = tf.zeros([1, pop.n_units], dtype=dtype)
zero_syn = {"I_syn": zero_I, "g_syn": zero_I}

for step in range(n_steps):
    E_hist.append(state[0][0].numpy())
    state, _ = integrator.step(pop, state, zero_syn)

E_hist = np.array(E_hist)
rates = np.maximum(E_hist, 0.0) * pop.max_rate.numpy()

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

for u in range(2):
    axes[0].plot(t_values, E_hist[:, u], label=f"Unit {u}", linewidth=1.5)
axes[0].set_ylabel("E (normalized activity)")
axes[0].set_title("Population Activity E")
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

for u in range(2):
    axes[1].plot(t_values, rates[:, u], label=f"Unit {u}", linewidth=1.5)
axes[1].set_ylabel("Firing Rate (Hz)")
axes[1].set_xlabel("Time (ms)")
axes[1].set_title("Output Firing Rate (relu(E) * max_rate)")
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

fig.suptitle(
    "Wilson-Cowan (single-population rate model)\nI_ext = [1.0, 0.5] (constant)",
    fontsize=13,
)
plt.tight_layout()
plt.savefig("example_pop_wilson_cowan.png", dpi=150)
plt.show()
print("Figure saved as example_pop_wilson_cowan.png")
