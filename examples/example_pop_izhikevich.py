"""
Пример: Динамика популяции IzhikevichMeanField с постоянным входным током.

Постоянный ток I_ext подаётся через параметр при создании.
Демонстрируется:
- Как Izhikevich отвечает на постоянный ток: начальная вспышка с последующим затуханием
- Переменные r (частота), v (мембранный потенциал), w (адаптация)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from neuraltide.populations.izhikevich_mf import IzhikevichMeanField
from neuraltide.integrators import RK4Integrator
from neuraltide.utils import seed_everything
import neuraltide.config

seed_everything(42)

dt = 0.05
T = 100

pop = IzhikevichMeanField(n_units=2, dt=dt, params={
    'alpha':     {'value': [0.5, 0.5],   'trainable': False},
    'a':         {'value': [0.02, 0.02], 'trainable': False},
    'b':         {'value': [0.2, 0.2],   'trainable': False},
    'w_jump':    {'value': [0.1, 0.1],   'trainable': False},
    'dt_nondim': {'value': [0.01, 0.01], 'trainable': False},
    'Delta_eta': {'value': [0.5, 0.6],   'trainable': False},
    'I_ext':     {'value': [1.5, 0.5],   'trainable': False},
})

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

for step in range(n_steps):
    r = state[0]
    v = state[1]
    w = state[2]
    r_hist.append(r[0].numpy())
    v_hist.append(v[0].numpy())
    w_hist.append(w[0].numpy())

    state, _ = integrator.step(pop, state, zero_syn)

    v = state[1]
    v_reset = tf.where(v > 30.0, tf.constant([[-65.0, -65.0]], dtype=dtype), v)
    w_add = tf.where(v > 30.0, tf.constant([[0.1, 0.1]], dtype=dtype), tf.zeros_like(w))
    state = [state[0], v_reset, state[2] + w_add]

r_hist = np.array(r_hist)
v_hist = np.array(v_hist)
w_hist = np.array(w_hist)

rates = r_hist * (pop.dt_nondim.numpy() / (dt * 1e-3))

fig, axes = plt.subplots(4, 1, figsize=(12, 10))

for u in range(2):
    axes[0].plot(t_values, r_hist[:, u], label=f'Unit {u}', linewidth=1.5)
axes[0].set_ylabel('r (rate, dimensionless)')
axes[0].set_title('Firing Rate r (dimensionless)')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

for u in range(2):
    axes[1].plot(t_values, v_hist[:, u], label=f'Unit {u}', linewidth=1.5)
axes[1].axhline(30, color='red', linestyle=':', linewidth=1, label='V_thresh=30mV')
axes[1].set_ylabel('v (mV)')
axes[1].set_title('Membrane Potential v (spike reset at 30mV)')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

for u in range(2):
    axes[2].plot(t_values, w_hist[:, u], label=f'Unit {u}', linewidth=1.5)
axes[2].set_ylabel('w (adaptation)')
axes[2].set_title('Adaptation Variable w')
axes[2].legend(fontsize=9)
axes[2].grid(True, alpha=0.3)

for u in range(2):
    axes[3].plot(t_values, rates[:, u], label=f'Unit {u}', linewidth=1.5)
axes[3].set_ylabel('Firing Rate (Hz)')
axes[3].set_xlabel('Time (ms)')
axes[3].set_title('Output Firing Rate')
axes[3].legend(fontsize=9)
axes[3].grid(True, alpha=0.3)

fig.suptitle(
    f"IzhikevichMeanField: Constant Current Response\n"
    f"I_ext = 1.5 (constant)",
    fontsize=13
)
plt.tight_layout()
plt.savefig("example_pop_izhikevich.png", dpi=150)
plt.show()
print("Figure saved as example_pop_izhikevich.png")
