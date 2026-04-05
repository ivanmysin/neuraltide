"""
Пример: Динамика популяции Wilson-Cowan с ступенчатым входным током.

Ступенчатый ток (Heaviside step) подаётся через I_ext_E.
При t < 50ms: I_ext_E = 0
При t >= 50ms: I_ext_E = 1.0

Демонстрируется:
- Возбуждающая и тормозная активность (E, I)
- Ответ на включение тока: запуск с последующим установлением
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from neuraltide.integrators import RK4Integrator
from neuraltide.utils import seed_everything
import neuraltide.config

seed_everything(42)

dt = 0.5
T = 100
STEP_TIME = 50.0
STEP_MAGNITUDE = 1.0


def make_population():
    from neuraltide.populations.wilson_cowan import WilsonCowan

    class WilsonCowanStep(WilsonCowan):
        def derivatives(self, state, total_synaptic_input):
            E, I = state
            I_syn = total_synaptic_input['I_syn']

            dtype = neuraltide.config.get_dtype()
            tau_E = tf.cast(self.tau_E, dtype)
            tau_I = tf.cast(self.tau_I, dtype)
            a_E = tf.cast(self.a_E, dtype)
            a_I = tf.cast(self.a_I, dtype)
            theta_E = tf.cast(self.theta_E, dtype)
            theta_I = tf.cast(self.theta_I, dtype)
            w_EE = tf.cast(self.w_EE, dtype)
            w_IE = tf.cast(self.w_IE, dtype)
            w_EI = tf.cast(self.w_EI, dtype)
            w_II = tf.cast(self.w_II, dtype)
            I_ext_E_raw = tf.cast(self.I_ext_E, dtype)
            I_ext_I = tf.cast(self.I_ext_I, dtype)

            step_on = tf.cast(self._t[0, 0] >= STEP_TIME, dtype=dtype)
            I_ext_E = I_ext_E_raw + tf.constant(STEP_MAGNITUDE, dtype=dtype) * step_on

            x_E = w_EE * E - w_IE * I + I_ext_E + I_syn
            x_I = w_EI * E - w_II * I + I_ext_I

            def sigmoid(x, a, theta):
                return 1.0 / (1.0 + tf.exp(-a * (x - theta)))

            F_E = sigmoid(x_E, a_E, theta_E)
            F_I = sigmoid(x_I, a_I, theta_I)

            dEdt = (-E + F_E) / tau_E
            dIdt = (-I + F_I) / tau_I

            return [dEdt, dIdt]

    pop = WilsonCowanStep(n_units=2, dt=dt, params={
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
        'I_ext_E':   {'value': [0.0, 0.0],    'trainable': False},
        'I_ext_I':   {'value': [0.0, 0.0],    'trainable': False},
        'max_rate':  {'value': [100.0, 100.0], 'trainable': False},
    })
    return pop


pop = make_population()
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
    t = tf.constant([[t_values[step]]], dtype=dtype)
    pop._t = t

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
axes[0].axvline(STEP_TIME, color='black', linestyle='--', linewidth=1, label=f'Step at {STEP_TIME}ms')
axes[0].set_ylabel('E (excitatory)')
axes[0].set_title('Excitatory Activity E')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

for u in range(2):
    axes[1].plot(t_values, I_hist[:, u], label=f'Unit {u}', linewidth=1.5)
axes[1].axvline(STEP_TIME, color='black', linestyle='--', linewidth=1)
axes[1].set_ylabel('I (inhibitory)')
axes[1].set_title('Inhibitory Activity I')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

for u in range(2):
    axes[2].plot(t_values, rates[:, u], label=f'Unit {u}', linewidth=1.5)
axes[2].axvline(STEP_TIME, color='black', linestyle='--', linewidth=1)
axes[2].set_ylabel('Firing Rate (Hz)')
axes[2].set_xlabel('Time (ms)')
axes[2].set_title('Output Firing Rate (E * max_rate)')
axes[2].legend(fontsize=9)
axes[2].grid(True, alpha=0.3)

fig.suptitle(
    f"Wilson-Cowan: Step Current Response\n"
    f"Step: I_ext_E = 0 → {STEP_MAGNITUDE} at t = {STEP_TIME}ms",
    fontsize=13
)
plt.tight_layout()
plt.savefig("example_pop_wilson_cowan.png", dpi=150)
plt.show()
print("Figure saved as example_pop_wilson_cowan.png")
