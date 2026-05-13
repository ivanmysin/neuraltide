"""
Example: PlaceFieldGenerator with phase precession via NetworkRNN extra_inputs_seq.

Demonstrates:
1. Passing position (x, y) coordinates through extra_inputs_seq in NetworkRNN.call()
2. PlaceFieldGenerator with phase precession working inside the network pipeline
3. Visualizations: cell output + theta, classic phase precession plot, arena map
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


from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.integrators import RK4Integrator
from neuraltide.populations import IzhikevichMeanField
from neuraltide.synapses import StaticSynapse
from neuraltide.inputs import PlaceFieldGenerator


# ════════════════ Parameters ════════════════
dt = 0.5          # ms
T_total = 300    # ms
n_place_cells = 2

# Trajectory parameters
speed = 0.15        # m/s — average linear speed along the path
r_base = 0.6       # m — base radius of quasi-circular trajectory
r_mod_ampl = 0.2   # m — amplitude of radius modulation
r_noise_std = 0.05 # m — std of radius noise
theta_noise_std = 0.01  # rad — std of angular noise

# Place field parameters
center_x = [0.0, -0.5]
center_y = [-0.75,  0.55]
radius =   [0.2, 0.15]
peak_rate = [5.0, 8.0]
bg_rate =   [0.5,  1.0]
theta_mod_factor = 1.0

# Phase precession parameters (slope in deg/cm)
precession_slope = [5.0, 15.0]
precession_init_phase = [0.0, 45.0]

# Theta rhythm parameters
R_theta = 0.6
freq_theta = 8.0

# ════════════════ Build network ════════════════

# 1. Create PlaceFieldGenerator (wraps 6 place cells with phase precession)
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
        'R': R_theta,
        'freq': freq_theta,
    },
    arena_size=((-1.0, 1.0), (-1.0, 1.0)),
    arena_radius=1.0,
    name='place_cells',
)

# 2. Create readout population — IzhikevichMeanField driven by place cell input.
#    I_ext=0 so the readout responds only to synaptic input from place cells.
#    Strong gsyn_max and e_r provide clear modulation reflecting place fields.
pop = IzhikevichMeanField(dt=dt, params={
    'tau_pop': {'value': [1.0]  * n_place_cells, 'trainable': False},
    'alpha':   {'value': [0.5]  * n_place_cells, 'trainable': False},
    'a':       {'value': [0.02] * n_place_cells, 'trainable': False},
    'b':       {'value': [0.2]  * n_place_cells, 'trainable': False},
    'w_jump':  {'value': [0.1]  * n_place_cells, 'trainable': False},
    'Delta_I': {'value': [0.05] * n_place_cells, 'trainable': False},
    'I_ext':   {'value': [0.0]  * n_place_cells, 'trainable': False},
})

# 3. Diagonal synapse: each place cell (unit i) → corresponding readout unit i.
#    gsyn_max=500 with e_r=5.0 provides strong excitatory drive that makes
#    the readout rates clearly reflect the place field + phase precession pattern.
syn_weights = [[1.0 if i == j else 0.0 for j in range(n_place_cells)]
               for i in range(n_place_cells)]
syn = StaticSynapse(n_pre=2, n_post=n_place_cells, dt=dt, params={
    'gsyn_max': {'value': syn_weights, 'trainable': False},
    'pconn':    {'value': 1.0, 'trainable': False},
    'e_r':      {'value': 5.0, 'trainable': False},
})

# 4. Build graph
graph = NetworkGraph(dt=dt)
graph.add_input_population('place', gen)
graph.add_population('readout', pop)
graph.add_synapse('place->readout', syn, src='place', tgt='readout')

# 5. Create NetworkRNN
network = NetworkRNN(graph, integrator=RK4Integrator())

print(f"PlaceFieldGenerator: n_units = {gen.n_units}")
print(f"  arena_size = {gen.arena_size}")
print(f"  center_x = {gen.center_x.numpy()}")
print(f"  center_y = {gen.center_y.numpy()}")
print(f"  precession_slope (deg/cm) = {gen.precession_slope_rad.numpy() * 180.0 / np.pi}")
print(f"Graph populations: {graph.population_names}")
print(f"Graph synapses: {graph.synapse_names}")

# ════════════════ Noisy quasi-circular trajectory ════════════════
n_steps = int(T_total / dt)
t_values = np.arange(n_steps, dtype=np.float32) * dt

# n_cycles derived from speed:  speed = 2π·r_base·n_cycles / T_total_sec
T_total_sec = T_total / 1000.0
n_cycles = speed * T_total_sec / (2.0 * np.pi * r_base)

# Trajectory: noisy quasi-circular path with variable radius
arena_center_x = 0.0
arena_center_y = 0.0
np.random.seed(42)
r_mod = r_mod_ampl * np.sin(2.0 * np.pi * np.arange(n_steps, dtype=np.float32) / n_steps * 0.5)
r_noise = r_noise_std * np.random.randn(n_steps).astype(np.float32)
r_traj = r_base + r_mod + r_noise
theta_noise = theta_noise_std * np.random.randn(n_steps).astype(np.float32)
theta_traj = (2.0 * np.pi * np.arange(n_steps, dtype=np.float32) / n_steps * n_cycles
              + theta_noise)
pos_x = arena_center_x + r_traj * np.cos(theta_traj)
pos_y = arena_center_y + r_traj * np.sin(theta_traj)

# Compute actual path length and average speed for diagnostics
path_deltas = np.sqrt(np.diff(pos_x)**2 + np.diff(pos_y)**2)
actual_path = np.sum(path_deltas)
actual_speed = actual_path / T_total_sec
print(f"\nTrajectory: r_base={r_base:.2f} m, speed param={speed:.2f} m/s")
print(f"  n_cycles = {n_cycles:.2f}, actual path = {actual_path:.2f} m, "
      f"actual speed = {actual_speed:.2f} m/s")

# ════════════════ Build extra_inputs_seq — only (x, y) ════════════════
# Time is provided via t_sequence (first argument to network.call).
# Extra columns carry position data for InputPopulation generators.
x_col = pos_x[:, np.newaxis]   # [T, 1]
y_col = pos_y[:, np.newaxis]   # [T, 1]
extra_xy = np.concatenate([x_col, y_col], axis=-1).astype(np.float32)  # [T, 2]
extra_inputs_seq = tf.constant(extra_xy[np.newaxis, :, :])              # [1, T, 2]

# Time sequence
t_seq = tf.constant(t_values[np.newaxis, :, np.newaxis])                # [1, T, 1]

print(f"\nt_sequence shape: {t_seq.shape}")
print(f"extra_inputs_seq shape: {extra_inputs_seq.shape}")
print("  columns: (x, y) position in arena units")

# ════════════════ Run simulation ════════════════
output = network(t_seq, extra_inputs_seq=extra_inputs_seq, training=False)
readout_rates = output.firing_rates['readout'].numpy()[0]  # [T, n_units]

# Also run without extra_inputs for comparison (uses default circular trajectory)
output_default = network(t_seq, extra_inputs_seq=None, training=False)
readout_rates_default = output_default.firing_rates['readout'].numpy()[0]

print(f"Output shape: {readout_rates.shape}")
print(f"Time range: {t_values[0]:.1f} - {t_values[-1]:.1f} ms")

# ════════════════ Direct generator output for comparison ════════════════
t_array = tf.constant(t_values[np.newaxis, :, np.newaxis])              # [1, T, 1]
extra_xy_direct = tf.constant(extra_xy[np.newaxis, :, :])                # [1, T, 2]
direct_rates = gen(t_array, extra_inputs=extra_xy_direct).numpy()[0]    # [T, n_units]

# ════════════════ Phase precession numeric check ════════════════
print("\n--- Phase precession check (readout rates) ---")
for i in range(gen.n_units):
    peak_idx = np.argmax(readout_rates[:, i])
    peak_time_ms = peak_idx * dt
    peak_rate_val = readout_rates[peak_idx, i]
    peak_x = pos_x[peak_idx]
    peak_y = pos_y[peak_idx]
    cx_i = gen.center_x.numpy()[i]
    cy_i = gen.center_y.numpy()[i]
    dist_peak = np.sqrt((peak_x - cx_i)**2 + (peak_y - cy_i)**2)
    slope_deg = gen.precession_slope_rad.numpy()[i] * 180.0 / np.pi

    inside_field = dist_peak < gen.radius.numpy()[i]

    print(f"  Cell {i}: peak_rate={peak_rate_val:.3f} at t={peak_time_ms:.0f}ms")
    print(f"    center=({cx_i:.1f},{cy_i:.1f}), peak_pos=({peak_x:.2f},{peak_y:.2f})")
    print(f"    dist_to_center={dist_peak:.2f}, inside_field={inside_field}")
    print(f"    slope={slope_deg:.1f} deg/cm, phi0={precession_init_phase[i]:.0f}deg")
print("Phase precession check complete!")

# ════════════════ Figure 1: Field crossing — rate overlaid on theta ════════════════

# Find a field crossing window for cell 0
cell_id = 0
cx0 = center_x[cell_id]
cy0 = center_y[cell_id]
r0 = radius[cell_id]
dists_to_cell0 = np.sqrt((pos_x - cx0)**2 + (pos_y - cy0)**2)
inside_mask = dists_to_cell0 < r0

# Find contiguous inside segments
changes = np.diff(np.concatenate([[False], inside_mask, [False]]).astype(int))
enter_idx = np.where(changes == 1)[0]
exit_idx = np.where(changes == -1)[0] - 1

margin_ms = 50.0
t_start = 0.0
t_end = T_total
for e_in, e_out in zip(enter_idx, exit_idx):
    dur_ms = (e_out - e_in) * dt
    if dur_ms > 100:
        t_start = max(0.0, float(t_values[e_in]) - margin_ms)
        t_end = min(float(T_total), float(t_values[min(e_out, n_steps - 1)]) + margin_ms)
        break

# High-res time grid — interpolate noisy trajectory to dt_hr
dt_hr = 0.1
t_hr = np.arange(t_start, t_end, dt_hr, dtype=np.float32)
pos_x_hr = np.interp(t_hr, t_values, pos_x).astype(np.float32)
pos_y_hr = np.interp(t_hr, t_values, pos_y).astype(np.float32)
extra_hr = tf.constant(np.stack([pos_x_hr, pos_y_hr], axis=-1)[np.newaxis, :, :])
t_hr_seq = tf.constant(t_hr[np.newaxis, :, np.newaxis])

# Run high-res simulation
out_hr = network(t_hr_seq, extra_inputs_seq=extra_hr, training=False)
readout_hr = out_hr.firing_rates['readout'].numpy()[0]
direct_hr = gen(t_hr_seq, extra_inputs=extra_hr).numpy()[0]

# Theta reference sinusoid (normalized)
init_phase0 = precession_init_phase[cell_id]
theta_ref = np.cos(2.0 * np.pi * freq_theta * t_hr / 1000.0 + np.deg2rad(init_phase0))

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

# -- Left: readout (network) overlaid on theta --
color_rate = 'steelblue'
ax1.plot(t_hr, readout_hr[:, cell_id], color=color_rate, linewidth=1.2,
         label='readout rate (network)')
ax1.fill_between(t_hr, readout_hr[:, cell_id], alpha=0.1, color=color_rate)
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
ax1.legend(lines1 + lines1b, labels1 + labels1b, fontsize=8, loc='upper right')
ax1.set_title(
    f'Cell {cell_id} — Readout rate vs theta\n'
    f'slope={precession_slope[cell_id]:+.0f} deg/cm, phi0={init_phase0:.0f}deg',
    fontsize=11)
ax1.grid(True, alpha=0.2)

# -- Right: direct generator overlaid on theta --
color_dir = 'coral'
ax2.plot(t_hr, direct_hr[:, cell_id], color=color_dir, linewidth=1.2,
         label='generator rate (direct)')
ax2.fill_between(t_hr, direct_hr[:, cell_id], alpha=0.1, color=color_dir)
ax2.axhline(y=bg_rate[cell_id], color='gray', linestyle=':', alpha=0.5,
            label=f'bg={bg_rate[cell_id]:.1f} Hz')
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('Firing rate (Hz)', color=color_dir)
ax2.tick_params(axis='y', labelcolor=color_dir)

ax2b = ax2.twinx()
ax2b.plot(t_hr, theta_ref, 'k--', linewidth=0.8, alpha=0.7, label='theta (8 Hz)')
ax2b.set_ylabel('Theta phase (cos)', color='gray')
ax2b.tick_params(axis='y', labelcolor='gray')
ax2b.set_ylim(-1.2, 1.2)

lines2, labels2 = ax2.get_legend_handles_labels()
lines2b, labels2b = ax2b.get_legend_handles_labels()
ax2.legend(lines2 + lines2b, labels2 + labels2b, fontsize=8, loc='upper right')
ax2.set_title(
    f'Cell {cell_id} — Generator rate vs theta\n'
    f'slope={precession_slope[cell_id]:+.0f} deg/cm, phi0={init_phase0:.0f}deg',
    fontsize=11)
ax2.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('place_field_cell_theta.png', dpi=150)
print(f"\nFigure 1 saved as place_field_cell_theta.png  (t = {t_start:.0f}–{t_end:.0f} ms)")
# plt.close(fig1)

# ════════════════ Figure 2: Classic phase precession plot ════════════════
# x-axis: normalized x-distance through place field  (pos_x - cx) / r
#   (0 = center, negative = entering, positive = exiting)
# y-axis: preferred theta LFP phase — the LFP phase at which the cell fires
#   maximally.  Rate peaks when cos(2πft + φ₀ + dphi) = 1 → LFP phase
#   2πft = -(φ₀ + dphi) mod 2π.  This is purely spatial and directly reveals
#   the precession tilt.
# Rows = cells, columns = readout / direct generator

fig2, axes2 = plt.subplots(gen.n_units, 2, figsize=(16, 5 * gen.n_units))

for i in range(gen.n_units):
    cx_i = gen.center_x.numpy()[i]
    cy_i = gen.center_y.numpy()[i]
    r_i = gen.radius.numpy()[i]
    slope_rad_i = gen.precession_slope_rad.numpy()[i]
    phi0_rad_i = gen.precession_init_phase.numpy()[i]

    norm_x = (pos_x - cx_i) / r_i

    # Precession-induced phase shift (matches generator: dphi = -slope * (pos_x - cx))
    dphi = -slope_rad_i * (pos_x - cx_i)

    # Preferred LFP theta phase: -(φ₀ + dphi) mod 2π
    pref_phase = (-phi0_rad_i - dphi) % (2.0 * np.pi)

    dist_raw = np.sqrt((pos_x - cx_i)**2 + (pos_y - cy_i)**2)
    inside = dist_raw < r_i

    # Total precession shift across the field (degrees)
    dphi_total = 2.0 * slope_rad_i * r_i * 180.0 / np.pi

    for col_idx, (rates, title_suffix) in enumerate([
        (readout_rates, 'Readout (network)'),
        (direct_rates, 'Generator (direct)'),
    ]):
        ax = axes2[i, col_idx] if gen.n_units > 1 else axes2[col_idx]

        rate_norm = rates[:, i] / (rates[:, i].max() + 1e-8)

        ax.scatter(norm_x[inside], pref_phase[inside],
                   s=8, alpha=0.4, c=rate_norm[inside],
                   cmap='plasma', vmin=0, vmax=1)

        ax.set_xlabel('Normalized x-distance  (pos_x − cx) / r', fontsize=10)
        ax.set_ylabel('Preferred theta phase (rad)', fontsize=10)
        ax.set_title(
            f'Cell {i} — {title_suffix}\n'
            f'center=({cx_i:.2f},{cy_i:.2f})  '
            f'slope={precession_slope[i]:+.0f} deg/cm  '
            f'φ₀={precession_init_phase[i]:.0f}°  '
            f'Δφ_across_field={dphi_total:.0f}°',
            fontsize=11, fontweight='bold')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-0.1, 2.0 * np.pi + 0.1)
        ax.set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
        ax.set_yticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
        ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('place_field_phase_precession.png', dpi=150)
print("Figure 2 saved as place_field_phase_precession.png")
# plt.close(fig2)

# ════════════════ Figure 3: Arena map ════════════════
fig3, axes3 = plt.subplots(1, 2, figsize=(12, 6))

x_grid = np.linspace(-1, 1, 100)
y_grid = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x_grid, y_grid)
colors = plt.cm.tab10(np.arange(gen.n_units))

for ax_idx, title_suffix in enumerate([
    'extra_inputs_seq', 'generator direct',
]):
    ax = axes3[ax_idx]
    for i in range(gen.n_units):
        cx = gen.center_x.numpy()[i]
        cy = gen.center_y.numpy()[i]
        r = gen.radius.numpy()[i]
        Z = np.exp(-0.5 * ((X - cx)**2 + (Y - cy)**2) / r**2)
        ax.contourf(X, Y, Z, levels=5, alpha=0.3, colors=[colors[i]])
        ax.plot(cx, cy, 'o', color=colors[i], markersize=5, label=f'cell {i}')
        circle = plt.Circle((cx, cy), r, color=colors[i], fill=False,
                            linewidth=2, linestyle='--')
        ax.add_patch(circle)

    ax.plot(pos_x, pos_y, 'k-', linewidth=0.6, alpha=0.5,
            label='noisy trajectory')
    ax.set_xlabel('X (arena units)')
    ax.set_ylabel('Y (arena units)')
    ax.set_title(f'Place Fields — {title_suffix}', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=6)
    ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('place_field_arena.png', dpi=150)
print("Figure 3 saved as place_field_arena.png")
# plt.close(fig3)
plt.show()

print("\nAll figures saved!")
print("\nKey demonstration:")
print(f"  1. Noisy quasi-circular trajectory: r_base={r_base:.1f} m, speed={speed:.1f} m/s"
      f" ({n_cycles:.1f} cycles)")
print("  2. Position (x, y) passed via extra_inputs_seq → InputPopulation → PlaceFieldGenerator")
print("  3. PlaceFieldGenerator computes rate with phase precession using those coordinates")
print("  4. Rate flows through StaticSynapse → IzhikevichMeanField readout")
print("  5. Readout firing rate reflects place field + phase precession pattern")
print("  6. Default trajectory: network runs with extra_inputs_seq=None → circular default")
