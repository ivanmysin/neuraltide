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
- **`neuraltide.synapses`** — SynapseModel subclasses (TsodyksMarkram, NMDA, Static, SecondOrderSynapse (непластичный), Composite).
- **`neuraltide.inputs`** — BaseInputGenerator subclasses (VonMises, Sinusoidal, ConstantRate, PlaceField). Внешние входы предвычисляются: `NetworkGraph.declare_input()` + `graph.pack_inputs()` → `NetworkRNN.call(t_seq, inputs=...)`.
- **`neuraltide.integrators`** — Euler, Heun, RK4.
- **`neuraltide.core.network`** — `NetworkGraph` (topology), `NetworkRNN` (simulation via `tf.while_loop` + TensorArray).
- **`neuraltide.training`** — Losses, readouts, Trainer, callbacks, adjoint solver.
- **`neuraltide.optimization`** — слой калибровки (P0/P1/P2 дорожной карты `docs/optimization_roadmap.md`): `CalibrationProblem`/`ParameterSpace`/`transforms`/`objectives`/`results`/`diagnostics` (P0) и `local` (SciPy L-BFGS-B, least_squares TRF/LM), `sensitivities` (forward JVP/якобиан, checkpointed/truncated BPTT), `shooting` (multiple shooting, penalty method), `strategies` (sample_phi0s, multistart, dt/subset continuation), `schedules` (Stage, ContinuationScheduler, teacher forcing с обязательным alpha=0), `identifiability` (FIM, likelihood_profile, select_subset) (P1); `global_search` (de/cma/spsa/basinhopping/nsga2, extras `global`/`pareto`), `robust` (веса компонент, PCGrad/CAGrad, balanced_train с baseline), `constraints` (ConstraintSpec, augmented_lagrangian, epsilon_constraint), `uq` (residual_diagnostics, laplace, bootstrap, predictive_intervals), `parallel` (SerialPool/ProcessPool, evaluate_candidates, make_evaluator) (P2). Пакетный `__init__.py` ленивый (PEP-562 `_LAZY_MAP`); всё API — eager-режим, batch=1. Ограничения P2: TF forward-AD чувствительностей несовместим с предшествующими gradient-прогонами в пределах одного NetworkRNN (UQ — на свежем экземпляре); exact-fit (нулевые невязки) вырожден для residual-UQ; балансировка — BPTT-only; process parallelism окупается только на дорогих оценках; tensor candidate-batch вне P2.
- **`neuraltide.config`** — Global dtype (float32 default), debug_numerics flag, class registries.
- **`neuraltide.constraints`** — MinMaxConstraint, NonNegConstraint, UnitIntervalConstraint.

## Key conventions

### `n_units` is inferred, not passed
`IzhikevichMeanField(dt=..., params={...})` and `WilsonCowan(dt=..., params={...})` — `n_units` is inferred from the max length of parameter arrays. Do **not** pass `n_units` as a keyword argument. (`FokkerPlanckPopulation`/`BrunelFokkerPlanck` are single-population objects, `n_units=1`.)

### WilsonCowan — single-population rate model
`τ·dE/dt = −E + F(w·E + I_ext + I_syn)`, one state variable `E` per unit, output `relu(E)·max_rate` (same interface as IzhikevichMeanField). Inhibition comes only through incoming synapses (including negative-weight inputs from inhibitory populations). Params (all required): `tau`, `a`, `theta`, `w`, `I_ext`, `max_rate`.

### BrunelFokkerPlanck
Classical Brunel (2000) model A: FP equation on a V-grid with absorbing threshold, reinjection of ν(t−τ_rp) at v_reset, refractory mass p_r. Coupling: `μ(t) = mu_0 + I_syn`, σ is the `sigma_ext` parameter. State: `[P, p_r, nu_buf]`. Internal CFL-based substepping in `derivatives()`; **use EulerIntegrator/HeunIntegrator** (the ν-delay buffer shift is exact for Euler). Stationary rate matches Brunel eq. (21) (~0.2% at grid_size=250 near-threshold; ~4% suprathreshold at grid 500).

### FokkerPlanckPopulation (базовый)
Абстрактная основа: `get_initial_state()` бросает `NotImplementedError` — подкласс обязан задать начальное распределение (валидный вариант: `self._gaussian_initial_state(center, sigma)` — нормированная гауссиана с обнулённым узлом порога).

### Dtype-конвенция (float64)
Все вычислительные методы (derivatives/compute_current/adjoint/forward) обязаны приводить веса к `neuraltide.config.get_dtype()` через `neuraltide.core.types.cast_if_needed` (не голый `tf.cast` по месту без нужды, но и не сырые `self.<param>` в арифметике с состоянием — иначе float64 падает на смешении dtype). Проверки NaN — через `neuraltide.config.maybe_check_numerics`.

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
All population state tensors have leading batch dimension of 1: `[1, n_units]`. The batch axis reflects the Keras convention, but multi-sample batches beyond 1 are not supported in the current sequential `tf.while_loop`.

## Docs status
- `README.md`: актуален (quick start через `declare_input`/`pack_inputs`, новые классы, Python >= 3.12).
- `SPEC.md` (бывший `MASTER_TZ.md`): историческая дизайн-спецификация v0.1.0. Расхождения с кодом помечены **[устарело]** в шапке и соответствующих разделах; источник истины по API — `docs/`.
