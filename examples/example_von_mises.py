"""
Пример: VonMisesGenerator - зависимость выхода от времени.
"""
import numpy as np
import matplotlib.pyplot as plt

try:
    import tensorflow as tf
    import neuraltide
    from neuraltide.inputs import VonMisesGenerator
    HAS_TF = True
except ImportError as e:
    print(f"TensorFlow not available: {e}")
    HAS_TF = False
    import sys
    sys.exit(0)

dt = 0.5
T = 500

gen = VonMisesGenerator(
    dt=dt,
    params={
        'mean_rate': 20.0,
        'R': 0.8,
        'freq': 8.0,
        'phase': 0.0,
    }
)
print(f"VonMisesGenerator: n_units = {gen.n_units}")
print(f"  mean_rate = {gen.mean_rate.numpy()}")
print(f"  kappa = {gen.kappa.numpy()}")
print(f"  freq = {gen.freq.numpy()}")
print(f"  phase = {gen.phase.numpy()}")

gen_multi = VonMisesGenerator(
    dt=dt,
    params={
        'mean_rate': [20.0, 15.0, 10.0],
        'R': [0.8, 0.5, 0.2],
        'freq': [8.0, 8.0, 8.0],
        'phase': [0.0, np.pi/2, np.pi],
    }
)
print(f"\nVonMisesGenerator (multi): n_units = {gen_multi.n_units}")

t_values = np.arange(0, T, dt, dtype=np.float32)
t_seq = tf.constant(t_values[None, :, None])

output_single = gen(t_seq).numpy()[0]
output_multi = gen_multi(t_seq).numpy()[0]

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].plot(t_values, output_single, linewidth=2, color='tab:blue')
axes[0].set_xlabel("Time (ms)")
axes[0].set_ylabel("Firing Rate (Hz)")
axes[0].set_title("VonMisesGenerator: Single input (R=0.8, freq=8Hz)")
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=20.0, color='gray', linestyle='--', alpha=0.5, label='mean_rate')
axes[0].legend()

for i in range(3):
    axes[1].plot(t_values, output_multi[:, i], linewidth=1.5, label=f"ch{i}: R={gen_multi._params['R'][i]}, freq={gen_multi._params['freq'][i]}")

axes[1].set_xlabel("Time (ms)")
axes[1].set_ylabel("Firing Rate (Hz)")
axes[1].set_title("VonMisesGenerator: Multiple inputs (3 channels)")
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.savefig("von_mises_generator_output.png", dpi=150)
plt.show()
print("\nFigure saved as von_mises_generator_output.png")
