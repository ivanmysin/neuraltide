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

- **`neuraltide.populations`** — PopulationModel subclasses (IzhikevichMeanField, WilsonCowan, FokkerPlanckPopulation, InputPopulation).
- **`neuraltide.synapses`** — SynapseModel subclasses (TsodyksMarkram, NMDA, Static, Composite).
- **`neuraltide.inputs`** — BaseInputGenerator subclasses (VonMises, Sinusoidal, ConstantRate).
- **`neuraltide.integrators`** — Euler, Heun, RK4.
- **`neuraltide.core.network`** — `NetworkGraph` (topology), `NetworkRNN` (simulation via `tf.scan`).
- **`neuraltide.training`** — Losses, readouts, Trainer, callbacks, adjoint solver.
- **`neuraltide.config`** — Global dtype (float32 default), debug_numerics flag, class registries.
- **`neuraltide.constraints`** — MinMaxConstraint, NonNegConstraint, UnitIntervalConstraint.

## Key conventions

### `n_units` is inferred, not passed
`IzhikevichMeanField(dt=..., params={...})` — `n_units` is inferred from the max length of parameter arrays. Do **not** pass `n_units` as a keyword argument.

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
Trainer accepts `grad_method="bptt"` (default, `tf.scan`-based) or `grad_method="adjoint"`. The adjoint method requires models to implement `adjoint_derivatives()` and `synaptic_coupling()`. Not all built-in models support the adjoint method yet.

### NetworkRNN uses `tf.scan`, not Keras RNN
`NetworkRNN` wraps a pure `tf.scan` loop (not `tf.keras.layers.RNN`). Its `call(t_sequence)` returns a `NetworkOutput` with `.firing_rates: dict[str, Tensor]`, `.stability_loss`, and `.final_state`.

### Input time shape
Time sequence: `tf.Tensor` shape `[batch, T, 1]`. If passed as `[batch, T]`, it is automatically expanded.

### Auto-registration on import
Importing `neuraltide.populations`, `neuraltide.synapses`, or `neuraltide.inputs` auto-registers classes into the global `POPULATION_REGISTRY`, `SYNAPSE_REGISTRY`, `INPUT_REGISTRY` dicts in `neuraltide.config`. Required for the config-first API (`config/schema.py`).

### State shapes
All population state tensors have leading batch dimension of 1: `[1, n_units]`. The batch axis reflects the Keras convention, but multi-sample batches beyond 1 are not supported in the current sequential `tf.scan`.

## Outdated docs
- `README.md`: example code uses the old `IzhikevichMeanField(dt=..., params={...})` with `n_units` as positional (n_units was removed from this constructor — it is now inferred).
- `MASTER_TZ.md`: test file names do not match actual files (e.g. references `test_network.py`, `test_populations.py` — these were renamed/consolidated).
