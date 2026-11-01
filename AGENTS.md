# AGENTS.md — NeuralTide

## Build / test / lint

```bash
pip install -e ".[dev]"          # editable install with dev deps
pytest                           # run all tests
pytest tests/unit/test_integrators.py -k test_euler_convergence
ruff check .                     # lint
ruff format --check .            # format check (black-equivalent, line-length=88)
```

Python **>=3.12** (pyproject.toml; README says 3.9 but that is outdated).

## Architecture

- **`neuraltide.populations`** — PopulationModel subclasses (IzhikevichMeanField, WilsonCowan, FokkerPlanckPopulation, BrunelFokkerPlanck, InputPopulation (deprecated)).
- **`neuraltide.synapses`** — SynapseModel subclasses (TsodyksMarkram, NMDA, Static, Composite).
- **`neuraltide.inputs`** — BaseInputGenerator subclasses (VonMises, Sinusoidal, ConstantRate, PlaceField). Внешние входы предвычисляются: `NetworkGraph.declare_input()` + `graph.pack_inputs()` → `NetworkRNN.call(t_seq, inputs=...)`.
- **`neuraltide.integrators`** — Euler, Heun, RK4.
- **`neuraltide.core.network`** — `NetworkGraph` (topology), `NetworkRNN` (simulation via `tf.while_loop` + TensorArray).
- **`neuraltide.training`** — Losses, readouts, Trainer, callbacks, adjoint solver.
- **`neuraltide.config`** — Global dtype (float32 default), debug_numerics flag, class registries.
- **`neuraltide.constraints`** — MinMaxConstraint, NonNegConstraint, UnitIntervalConstraint.

## Key conventions

### `n_units` is inferred, not passed
`IzhikevichMeanField(dt=..., params={...})` and `WilsonCowan(dt=..., params={...})` — `n_units` is inferred from the max length of parameter arrays. Do **not** pass `n_units` as a keyword argument. (`FokkerPlanckPopulation`/`BrunelFokkerPlanck` are single-population objects, `n_units=1`.)

### WilsonCowan — single-population rate model
`τ·dE/dt = −E + F(w·E + I_ext + I_syn)`, one state variable `E` per unit, output `relu(E)·max_rate` (same interface as IzhikevichMeanField). Inhibition comes only through incoming synapses (including negative-weight inputs from inhibitory populations). Params (all required): `tau`, `a`, `theta`, `w`, `I_ext`, `max_rate`.

### BrunelFokkerPlanck
Classical Brunel (2000) model A: FP equation on a V-grid with absorbing threshold, reinjection of ν(t−τ_rp) at v_reset, refractory mass p_r. Coupling: `μ(t) = mu_0 + I_syn`, σ is the `sigma_ext` parameter. State: `[P, p_r, nu_buf]`. Internal CFL-based substepping in `derivatives()`; **use EulerIntegrator/HeunIntegrator** (the ν-delay buffer shift is exact for Euler). Stationary rate matches Brunel eq. (21) (~0.2% at grid_size=250 near-threshold; ~4% suprathreshold at grid 500).

### Input generators
All generators return `[batch, T, n_units]` and accept `t` as `[batch, T, 1]`, `[batch, T]` or `[T]`. `PlaceFieldGenerator` additionally takes `extra_inputs` with (x, y) or (x, y, vx, vy) columns; rate = `peak·spatial·θ_in + bg·(1−spatial)·(1−tmf+tmf·θ_out)` where tmf = theta_modulation_factor. In `VonMisesGenerator` `R` may be trainable (kappa is recomputed in `call`). `BaseInputGenerator` sets `autocast = False`, so float64 works through Keras `__call__`.

### `params` dict format
Every parameter key can be:
- A raw scalar or list (→ non-trainable, no constraints)
- A dict: `{'value': ..., 'trainable': bool, 'min': float, 'max': float}`

**Constraints are named `min`/`max` in the user `params` dict**, but the internal `MinMaxConstraint` constructor uses `min_val`/`max_val`. This translation is handled by `_make_param`.

### Global dtype
`neuraltide.config.set_dtype(tf.float64)` must be called **before** creating any model objects. The conftest auto-resets to `float32` before every test.

### debug_numerics
`neuraltide.config.set_debug_numerics(True)` enables `tf.debugging.check_numerics` on critical ops — useful for debugging NaNs during training.

### Two gradient computation modes
Trainer accepts `grad_method="bptt"` (default) or `grad_method="adjoint"`. The adjoint method requires models to implement `adjoint_derivatives()` and `synaptic_coupling()`. Not all built-in models support the adjoint method yet.

### NetworkRNN uses `tf.while_loop`, not Keras RNN
`NetworkRNN` wraps a `tf.while_loop` over TensorArrays (not `tf.keras.layers.RNN`). Its `call(t_sequence, inputs=None, initial_state=None)` returns a `NetworkOutput` with `.firing_rates: dict[str, Tensor]`, `.stability_loss`, and `.final_state`. `inputs` (shape `[batch, T, total_input_units]`) carries the precomputed external input rates, packed via `graph.pack_inputs()`; the graph's external inputs are declared with `NetworkGraph.declare_input(name, n_units)`.

### Input time shape
Time sequence: `tf.Tensor` shape `[batch, T, 1]`. If passed as `[batch, T]`, it is automatically expanded.

### Auto-registration on import
Importing `neuraltide.populations`, `neuraltide.synapses`, or `neuraltide.inputs` auto-registers classes into the global `POPULATION_REGISTRY`, `SYNAPSE_REGISTRY`, `INPUT_REGISTRY` dicts in `neuraltide.config`. Required for the config-first API (`config/schema.py`).

### State shapes
All population state tensors have leading batch dimension of 1: `[1, n_units]`. The batch axis reflects the Keras convention, but multi-sample batches beyond 1 are not supported in the current sequential `tf.scan`.

## Outdated docs
- `README.md`: example code uses the old `IzhikevichMeanField(dt=..., params={...})` with `n_units` as positional (n_units was removed from this constructor — it is now inferred), and the old E/I-pair WilsonCowan (`n_units` positional, `tau_E/tau_I/...` params) — WilsonCowan is now a single-population rate model, see `docs/populations.md`.
- `MASTER_TZ.md`: test file names do not match actual files (e.g. references `test_network.py`, `test_populations.py` — these were renamed/consolidated); the WilsonCowan/FokkerPlanck sections were updated to match the current code.
