"""
Пример: Динамика популяции Wilson-Cowan с постоянным входным током.

Постоянный ток I_ext_E подаётся через параметр при создании.
Демонстрируется:
- Возбуждающая и тормозная активность (E, I)
- Ответ на постоянный ток: запуск с последующим установлением
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from neuraltide.populations.wilson_cowan import WilsonCowan
from neuraltide.integrators import RK4Integrator
from neuraltide.utils import seed_everything
import neuraltide.config

seed_everything(42)

dt = 0.5
T = 100

pop = WilsonCowan(n_units=2, dt=dt, params={
    'tau_E':     {'value': [10.0, 10.0],  'trainable': False},
    'tau_I':     {'value': [10.0, 10.0],  'trainable': False},
    'a_E':       {'value': [1.0, 1.2],    'trainable': False},
    'a_I':       {'value': [1.0, 1.0],    'trainable': False},
    'theta_E':   {'value': [0.0, 0.0],    'trainable': False},
    'theta_I':   {'value': [0.0, 0.0],    'trainable': False},
    'w_EE':      {'value': [1.0, 1.0],    'trainable': False},
    'w_IE':      {'value': [1.0, 1.0],    'trainable': False},
    'w_EI':      {'value': [1.0, 1.0],    'trainable': False},
    'w_II':      {'value': [1.0, 1.0],    'trainable': False},
    'I_ext_E':   {'value': [1.0, 1.0],    'trainable': False},
    'I_ext_I':   {'value': [0.0, 0.0],    'trainable': False},
    'max_rate':  {'value': [100.0, 100.0], 'trainable': False},
})

integrator = RK4Integrator()

t_values = np.arange(0, T, dt, dtype=np.float32)
n_steps = len(t_values)

state = pop.get_initial_state()
E_hist = []
I_hist = []

dtype = neuraltide.config.get_dtype()
zero_I = tf.zeros([1, pop.n_units], dtype=dtype)
zero_syn = {'I_syn': zero_I, 'g_syn': zero_I}

for step in range(n_steps):
    E = state[0]
    I = state[1]
    E_hist.append(E[0].numpy())
    I_hist.append(I[0].numpy())

    state, _ = integrator.step(pop, state, zero_syn)

E_hist = np.array(E_hist)
I_hist = np.array(I_hist)

rates = np.maximum(E_hist, 0.0) * pop.max_rate.numpy()

fig, axes = plt.subplots(3, 1, figsize=(12, 8))

for u in range(2):
    axes[0].plot(t_values, E_hist[:, u], label=f'Unit {u}', linewidth=1.5)
axes[0].set_ylabel('E (excitatory)')
axes[0].set_title('Excitatory Activity E')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

for u in range(2):
    axes[1].plot(t_values, I_hist[:, u], label=f'Unit {u}', linewidth=1.5)
axes[1].set_ylabel('I (inhibitory)')
axes[1].set_title('Inhibitory Activity I')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

for u in range(2):
    axes[2].plot(t_values, rates[:, u], label=f'Unit {u}', linewidth=1.5)
axes[2].set_ylabel('Firing Rate (Hz)')
axes[2].set_xlabel('Time (ms)')
axes[2].set_title('Output Firing Rate (E * max_rate)')
axes[2].legend(fontsize=9)
axes[2].grid(True, alpha=0.3)

fig.suptitle(
    f"Wilson-Cowan: Constant Current Response\n"
    f"I_ext_E = 1.0 (constant)",
    fontsize=13
)
plt.tight_layout()
plt.savefig("example_pop_wilson_cowan.png", dpi=150)
plt.show()
print("Figure saved as example_pop_wilson_cowan.png")
