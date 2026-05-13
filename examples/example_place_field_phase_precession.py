"""
Example: PlaceFieldGenerator — direct generator output with phase precession.

Demonstrates:
1. Direct PlaceFieldGenerator calls with (x, y) position input in cm.
2. Phase precession: rate phase shifts as animal moves through place field.
3. Diagnostic printout of all generator input parameters.
4. Visualization: field crossing with theta, phase precession plot, arena map.

All lengths: cm.  All speeds: cm/s.  All times: ms.
"""
import numpy as np
import tensorflow as tf

try:
    import matplotlib
    matplotlib.use('qt5agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

if not HAS_MPL:
    print("matplotlib not available, skipping example")
    import sys
    sys.exit(0)

from neuraltide.inputs import PlaceFieldGenerator


# ════════════════ Parameters (cm, cm/s, ms, Hz) ════════════════
dt = 0.5                  # ms
T_total = 30000            # ms
n_place_cells = 2

# Trajectory parameters
speed = 20.0              # cm/s
r_base = 60.0             # cm — base radius of quasi-circular trajectory
r_mod_ampl = 20.0         # cm — amplitude of radius modulation
r_noise_std = 2.0         # cm — std of radius noise
theta_noise_std = 0.1    # rad — std of angular noise

# Arena parameters
arena_radius = 100.0      # cm
arena_size = ((-arena_radius, arena_radius), (-arena_radius, arena_radius))

# Place field parameters (in cm)
center_x = [0.0, -50.0]          # cm
center_y = [-75.0, 55.0]         # cm
radius =   [20.0, 15.0]          # cm
peak_rate = [5.0, 8.0]           # Hz
bg_rate =   [0.5, 1.0]           # Hz
theta_mod_factor = [1.0, 1.0]    # 1 = full theta modulation outside field

# Phase precession parameters
precession_slope = [5.0, 15.0]          # deg/cm
precession_init_phase = [0.0, 45.0]     # degrees
phase_outside = [0.0, 180.0]            # degrees

# Theta rhythm parameters
R_theta = [0.25, 0.45]
freq_theta = 8.0                        # Hz

# ════════════════ Diagnostic print ════════════════
print("=" * 60)
print("PlaceFieldGenerator — input parameters")
print("=" * 60)
print(f"  dt                  = {dt:.1f} ms")
print(f"  T_total             = {T_total:.0f} ms")
print(f"  n_place_cells       = {n_place_cells}")
print(f"  arena_size          = ({arena_size[0][0]:.0f}, {arena_size[0][1]:.0f}) "
      f"× ({arena_size[1][0]:.0f}, {arena_size[1][1]:.0f}) cm")
print(f"  arena_radius        = {arena_radius:.0f} cm")
print()
print("-- Trajectory --")
print(f"  speed               = {speed:.1f} cm/s")
print(f"  r_base              = {r_base:.1f} cm")
print(f"  r_mod_ampl          = {r_mod_ampl:.1f} cm")
print(f"  r_noise_std         = {r_noise_std:.1f} cm")
print(f"  theta_noise_std     = {theta_noise_std:.3f} rad")
print()
for i in range(n_place_cells):
    print(f"-- Cell {i} --")
    print(f"  center              = ({center_x[i]:+.1f}, {center_y[i]:+.1f}) cm")
    print(f"  radius              = {radius[i]:.1f} cm")
    print(f"  peak_rate           = {peak_rate[i]:.1f} Hz")
    print(f"  background_rate     = {bg_rate[i]:.1f} Hz")
    print(f"  theta_mod_factor    = {theta_mod_factor[i]:.1f}")
    print(f"  precession_slope    = {precession_slope[i]:.1f} deg/cm")
    print(f"  precession_total    = {2.0 * precession_slope[i] * radius[i]:.0f} deg "
          f"(full field crossing)")
    print(f"  precession_init_ph  = {precession_init_phase[i]:.0f} deg = "
          f"{np.deg2rad(precession_init_phase[i]):.2f} rad")
    print()
    print("-- Theta rhythm --")
    print(f"  R                   = {R_theta[i]:.2f}")
    print(f"  phase_outside       = {phase_outside[i]:.0f} deg = "
          f"{np.deg2rad(phase_outside[i]):.2f} rad")
print()
print(f"  freq_theta          = {freq_theta:.1f} Hz")

# ════════════════ Build generator ════════════════
gen = PlaceFieldGenerator(
    dt=dt,
    params={
        'center_x': center_x,
        'center_y': center_y,
        'radius': radius,
        'peak_rate': peak_rate,
        'background_rate': bg_rate,
        'theta_modulation_factor': theta_mod_factor,
        'precession_slope': precession_slope,
        'precession_init_phase': precession_init_phase,
        'phase_outside': phase_outside,
        'R': R_theta,
        'freq': freq_theta,
    },
    arena_size=arena_size,
    arena_radius=arena_radius,
    name='place_cells',
)

print(f"Generator created:  n_units = {gen.n_units}")
print(f"  arena_size = {gen.arena_size}")
print(f"  center_x   = {gen.center_x.numpy()}  cm")
print(f"  center_y   = {gen.center_y.numpy()}  cm")
print(f"  radius     = {gen.radius.numpy()}    cm")
print(f"  peak_rate  = {gen.peak_rate.numpy()} Hz")
print(f"  bg_rate    = {gen.background_rate.numpy()} Hz")
print(f"  tmf        = {gen.theta_modulation_factor.numpy()}")
print(f"  slope_rad  = {gen.precession_slope_rad.numpy()} rad/cm")
print(f"    (= {np.array2string(gen.precession_slope_rad.numpy() * 180.0 / np.pi, formatter={'float_kind': lambda x: f'{x:.1f}'})} deg/cm)")
print(f"  ph0        = {gen.precession_init_phase.numpy()} rad")
print(f"    (= {np.array2string(np.rad2deg(gen.precession_init_phase.numpy()), formatter={'float_kind': lambda x: f'{x:.0f}'})} deg)")
print(f"  ph_outside = {gen.phase_outside.numpy()} rad")
print(f"    (= {np.array2string(np.rad2deg(gen.phase_outside.numpy()), formatter={'float_kind': lambda x: f'{x:.0f}'})} deg)")
print(f"  kappa      = {gen.kappa.numpy()}")
print(f"  i0(kappa)  = {gen.i0_kappa.numpy()}")
print(f"  freq       = {gen.freq.numpy()} Hz")
print()

# ════════════════ Noisy quasi-circular trajectory ════════════════
n_steps = int(T_total / dt)
t_values = np.arange(n_steps, dtype=np.float32) * dt

# n_cycles from speed along circular path
T_total_sec = T_total / 1000.0
n_cycles = speed * T_total_sec / (2.0 * np.pi * r_base)

np.random.seed(42)
r_mod = r_mod_ampl * np.sin(
    2.0 * np.pi * np.arange(n_steps, dtype=np.float32) / n_steps * 0.5)
r_noise = r_noise_std * np.random.randn(n_steps).astype(np.float32)
r_traj = r_base + r_mod + r_noise

theta_noise = theta_noise_std * np.random.randn(n_steps).astype(np.float32)
theta_traj = (2.0 * np.pi * np.arange(n_steps, dtype=np.float32) / n_steps
              * n_cycles + theta_noise)
pos_x = r_traj * np.cos(theta_traj)   # cm
pos_y = r_traj * np.sin(theta_traj)   # cm

# Path diagnostics
path_deltas = np.sqrt(np.diff(pos_x)**2 + np.diff(pos_y)**2)
actual_path = np.sum(path_deltas)
actual_speed = actual_path / T_total_sec
print("Trajectory diagnostics:")
print(f"  n_cycles     = {n_cycles:.2f} over {T_total_sec:.1f} s")
print(f"  actual path  = {actual_path:.0f} cm")
print(f"  actual speed = {actual_speed:.1f} cm/s (target = {speed:.1f} cm/s)")
print(f"  x range      = [{pos_x.min():.0f}, {pos_x.max():.0f}] cm")
print(f"  y range      = [{pos_y.min():.0f}, {pos_y.max():.0f}] cm")
print()

# ════════════════ Run generator ════════════════
t_seq = tf.constant(t_values[np.newaxis, :, np.newaxis], dtype=tf.float32)    # [1, T, 1]
extra_xy = np.stack([pos_x, pos_y], axis=-1).astype(np.float32)[np.newaxis, :, :]  # [1, T, 2]
extra_inputs = tf.constant(extra_xy)

rates = gen(t_seq, extra_inputs=extra_inputs).numpy()[0]  # [T, n_units]

print(f"Generator output shape: {rates.shape}")
print(f"  Rate range cell 0: [{rates[:, 0].min():.2f}, {rates[:, 0].max():.2f}] Hz")
print(f"  Rate range cell 1: [{rates[:, 1].min():.2f}, {rates[:, 1].max():.2f}] Hz")
print()

# Also run without extra_inputs (returns background rate only)
rates_no_pos = gen(t_seq, extra_inputs=None).numpy()[0]
print(f"No-position output: cell 0 range = [{rates_no_pos[:, 0].min():.2f}, "
      f"{rates_no_pos[:, 0].max():.2f}] Hz")

# ════════════════ Field crossing diagnostics ════════════════
print()
print("--- Field crossing diagnostics ---")
for i in range(gen.n_units):
    peak_idx = np.argmax(rates[:, i])
    peak_time_ms = peak_idx * dt
    peak_rate_val = rates[peak_idx, i]
    cx_i = gen.center_x.numpy()[i]
    cy_i = gen.center_y.numpy()[i]
    r_i = gen.radius.numpy()[i]
    dist_peak = np.sqrt((pos_x[peak_idx] - cx_i)**2 + (pos_y[peak_idx] - cy_i)**2)
    slope_deg = gen.precession_slope_rad.numpy()[i] * 180.0 / np.pi
    inside_field = dist_peak < r_i

    print(f"  Cell {i}:")
    print(f"    peak_rate = {peak_rate_val:.3f} Hz at t = {peak_time_ms:.0f} ms")
    print(f"    peak_position = ({pos_x[peak_idx]:.1f}, {pos_y[peak_idx]:.1f}) cm")
    print(f"    dist_to_center = {dist_peak:.1f} cm, inside_field = {inside_field}")
    print("    expected max = peak_rate * spatial * theta_mod_inside "
          "+ bg * ...")
    print(f"    slope = {slope_deg:.1f} deg/cm, "
          f"phi0 = {precession_init_phase[i]:.0f} deg")

# ════════════════ Figure 1: Field crossing — rate overlaid on theta ════════════════
cell_id = 0
cx0 = center_x[cell_id]
cy0 = center_y[cell_id]
r0 = radius[cell_id]
dists_to_cell0 = np.sqrt((pos_x - cx0)**2 + (pos_y - cy0)**2)
inside_mask = dists_to_cell0 < r0


t_hr = np.arange(0, T_total, dt, dtype=np.float32)
pos_x_hr = np.interp(t_hr, t_values, pos_x).astype(np.float32)
pos_y_hr = np.interp(t_hr, t_values, pos_y).astype(np.float32)

t_hr_seq = tf.constant(t_hr[np.newaxis, :, np.newaxis])
extra_hr = tf.constant(np.stack([pos_x_hr, pos_y_hr], axis=-1)[np.newaxis, :, :])
rates_hr = gen(t_hr_seq, extra_inputs=extra_hr).numpy()[0]

init_phase0 = precession_init_phase[cell_id]
theta_ref = np.cos(2.0 * np.pi * freq_theta * t_hr / 1000.0 + np.deg2rad(init_phase0))

fig1, axes = plt.subplots(1, n_place_cells, figsize=(14, 4.5))

for cell_id in range(n_place_cells):
    ax1 = axes[cell_id]
    color_rate = 'steelblue'
    ax1.plot(t_hr, rates_hr[:, cell_id], color=color_rate, linewidth=1.2,
             label='PlaceFieldGenerator rate')
    ax1.fill_between(t_hr, rates_hr[:, cell_id], alpha=0.1, color=color_rate)
    ax1.axhline(y=bg_rate[cell_id], color='gray', linestyle=':', alpha=0.5,
                label=f'bg={bg_rate[cell_id]:.1f} Hz')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Firing rate (Hz)', color=color_rate)
    ax1.tick_params(axis='y', labelcolor=color_rate)

    ax1b = ax1.twinx()
    ax1b.plot(t_hr, theta_ref, 'k--', linewidth=0.8, alpha=0.7, label='theta (8 Hz)')
    ax1b.set_ylabel('Theta phase (cos)', color='gray')
    ax1b.tick_params(axis='y', labelcolor='gray')
    ax1b.set_ylim(-1.2, 1.2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines1b, labels1b = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines1b, labels1 + labels1b, fontsize=9, loc='upper right')
    ax1.set_title(
        f'Cell {cell_id} — Place field crossing with theta\n'
        f'slope={precession_slope[cell_id]:+.0f} deg/cm, '
        f'phi0={init_phase0:.0f}deg, '
        f'Δφ={2.0 * precession_slope[cell_id] * radius[cell_id]:.0f}deg',
        fontsize=11)
    ax1.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('place_field_cell_theta.png', dpi=150)
print(f"\nFigure 1 saved as place_field_cell_theta.png )")

# ════════════════ Figure 2: Phase precession plot ════════════════
# x-axis: normalized x-distance through place field (pos_x - cx) / r
# y-axis: preferred theta LFP phase (rad)
#   Rate peaks when cos(2πft + φ₀ + dphi) = 1 → LFP phase = -(φ₀ + dphi) mod 2π

fig2, axes2 = plt.subplots(1, gen.n_units, figsize=(8 * gen.n_units, 5))
if gen.n_units == 1:
    axes2 = [axes2]

for i in range(gen.n_units):
    ax = axes2[i]
    cx_i = gen.center_x.numpy()[i]
    cy_i = gen.center_y.numpy()[i]
    r_i = gen.radius.numpy()[i]
    slope_rad_i = gen.precession_slope_rad.numpy()[i]
    phi0_rad_i = gen.precession_init_phase.numpy()[i]

    norm_x = (pos_x - cx_i) / r_i
    dphi = -slope_rad_i * (pos_x - cx_i)
    pref_phase = (-phi0_rad_i - dphi) % (2.0 * np.pi)

    dist_raw = np.sqrt((pos_x - cx_i)**2 + (pos_y - cy_i)**2)
    inside = dist_raw < r_i

    rate_norm = rates[:, i] / (rates[:, i].max() + 1e-8)

    dphi_total = 2.0 * slope_rad_i * r_i * 180.0 / np.pi

    ax.scatter(norm_x[inside], pref_phase[inside],
               s=8, alpha=0.4, c=rate_norm[inside],
               cmap='plasma', vmin=0, vmax=1)
    ax.scatter(norm_x[~inside], pref_phase[~inside],
               s=3, alpha=0.15, c='gray')

    ax.set_xlabel('Normalized x-distance  (pos_x − cx) / r', fontsize=10)
    ax.set_ylabel('Preferred theta phase (rad)', fontsize=10)
    ax.set_title(
        f'Cell {i}\n'
        f'center=({cx_i:.0f},{cy_i:.0f}) cm  '
        f'slope={precession_slope[i]:+.0f} deg/cm  '
        f'φ₀={precession_init_phase[i]:.0f}°  '
        f'Δφ={dphi_total:.0f}°',
        fontsize=11, fontweight='bold')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.1, 2.0 * np.pi + 0.1)
    ax.set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax.set_yticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
    ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('place_field_phase_precession.png', dpi=150)
print("Figure 2 saved as place_field_phase_precession.png")

# ════════════════ Figure 3: Arena map ════════════════
fig3, ax3 = plt.subplots(1, 1, figsize=(7, 7))

x_grid = np.linspace(arena_size[0][0], arena_size[0][1], 100)
y_grid = np.linspace(arena_size[1][0], arena_size[1][1], 100)
X, Y = np.meshgrid(x_grid, y_grid)
colors = plt.cm.tab10(np.arange(gen.n_units))

for i in range(gen.n_units):
    cx = gen.center_x.numpy()[i]
    cy = gen.center_y.numpy()[i]
    r = gen.radius.numpy()[i]
    Z = np.exp(-0.5 * ((X - cx)**2 + (Y - cy)**2) / r**2)
    ax3.contourf(X, Y, Z, levels=5, alpha=0.3, colors=[colors[i]])
    ax3.plot(cx, cy, 'o', color=colors[i], markersize=8, label=f'cell {i}')
    circle = plt.Circle((cx, cy), r, color=colors[i], fill=False,
                        linewidth=2, linestyle='--')
    ax3.add_patch(circle)

ax3.plot(pos_x, pos_y, 'k-', linewidth=0.6, alpha=0.5, label='trajectory')
ax3.set_xlabel('X (cm)')
ax3.set_ylabel('Y (cm)')
ax3.set_title('Place Fields + Noisy Trajectory', fontsize=12, fontweight='bold')
ax3.set_aspect('equal')
ax3.legend(loc='upper right', fontsize=8)
ax3.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('place_field_arena.png', dpi=150)
print("Figure 3 saved as place_field_arena.png")

plt.show()

print()
print("All figures saved!")
print()
print("Key points:")
print(f"  1. Arena: {arena_radius:.0f} cm radius, trajectory: quasi-circular "
      f"(r_base={r_base:.0f} cm, {n_cycles:.1f} cycles)")
print("  2. Position (x, y) in cm passed directly to PlaceFieldGenerator.call()")
print("  3. Phase precession: dphi = -slope * (x - cx) shifts theta phase inside field")
print("  4. Outside field: rate = bg_rate * (1 - tmf + tmf * theta_mod_outside)")
print("  5. theta_modulation_factor = 1 -> full theta modulation outside field")
print(f"  6. phase_outside = {phase_outside}° — separate theta phase outside field")
