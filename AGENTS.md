# AGENTS.md — NeuralTide

## Build / test / lint

```bash
pip install -e ".[dev]"          # editable install with dev deps
pytest                           # run all tests
pytest tests/unit/test_integrators.py -k test_euler_convergence
pytest tests/unit/test_graph_mode.py        # граф-эквивалентность NetworkRNN (этап 1a)
pytest tests/unit/test_compiled_problem.py  # compiled-ядро optimization-слоя (этап 1b)
ruff check .                     # lint
ruff format --check .            # format check (black-equivalent, line-length=88)
```

Python **>=3.12** (pyproject.toml; README says 3.9 but that is outdated).

## Architecture

- **`neuraltide.populations`** — PopulationModel subclasses (IzhikevichMeanField, WilsonCowan, FokkerPlanckPopulation, BrunelFokkerPlanck, InputPopulation (deprecated)).
- **`neuraltide.synapses`** — SynapseModel subclasses (TsodyksMarkram, NMDA, Static, SecondOrderSynapse (непластичный), Composite).
- **`neuraltide.inputs`** — BaseInputGenerator subclasses (VonMises, Sinusoidal, ConstantRate, PlaceField). Внешние входы предвычисляются: `NetworkGraph.declare_input()` + `graph.pack_inputs()` → `NetworkRNN.call(t_seq, inputs=...)`.
- **`neuraltide.integrators`** — Euler, Heun, RK4; P2: AdaptiveRK (BS23+PI), IMEX-Euler, Rosenbrock ROS2, Exponential Euler, `stiffness_report`. Network-wide интегрирование: `NetworkRNN(integration_mode='split'|'network')` (default split); режим 'network' — единый шаг `step_network(rhs, pop_states, syn_states, dt)` по `_JointNetworkRHS`, поддерживают Euler/Heun/RK4/Rosenbrock (`supports_network_step=True`; у AdaptiveRK/IMEX/Exponential — ValueError); RK4 network-wide на связанной TM-сети порядок ~4 (split ~2), ROS2 network-wide устойчив на жёсткой связанной задаче (плотный [D,D] якобиан); `stiffness_report(scope='populations'|'network')` — network scope строит полный спектр связанной системы (входы ZOH). Ограничения network-режима: `stability_loss` не оценивается (== 0 — step_network не возвращает per-step local_error); analytical adjoint в network-режиме NotImplementedError (discrete adjoint/BPTT — оба режима); InputPopulation внутри полного scan не поддержан (как и раньше); BrunelFP — split Euler/Heun с ν-буфером; внешние входы ZOH внутри шага — порядок 4 заявляется для автономной/константной части.
- **`neuraltide.core.network`** — `NetworkGraph` (topology), `NetworkRNN` (simulation via `tf.while_loop` + TensorArray).
- **`neuraltide.training`** — Losses, readouts, Trainer, callbacks, adjoint solver.
- **`neuraltide.optimization`** — слой калибровки (P0/P1/P2 дорожной карты `context/optimization_roadmap.md`): `CalibrationProblem`/`ParameterSpace`/`transforms`/`objectives`/`results`/`diagnostics` (P0) и `local` (SciPy L-BFGS-B, least_squares TRF/LM), `sensitivities` (forward JVP/якобиан, checkpointed/truncated BPTT), `shooting` (multiple shooting, penalty method; `fix_first_segment=True` фиксирует сегмент 0 за base initial state — без записей `x0s/0/...`), `strategies` (sample_phi0s, multistart, dt/subset continuation), `schedules` (Stage, ContinuationScheduler, teacher forcing с обязательным alpha=0), `identifiability` (FIM, likelihood_profile, select_subset) (P1); `global_search` (de/cma/spsa/basinhopping/nsga2, extras `global`/`pareto`), `robust` (веса компонент, PCGrad/CAGrad, balanced_train с baseline), `constraints` (ConstraintSpec, augmented_lagrangian, epsilon_constraint), `uq` (residual_diagnostics, laplace, bootstrap, predictive_intervals), `parallel` (SerialPool/ProcessPool, evaluate_candidates, make_evaluator) (P2). Пакетный `__init__.py` ленивый (PEP-562 `_LAZY_MAP`); batch=1; по умолчанию eager, `compiled=True` — графовый путь (этап 1b, см. «Компилируемые режимы»). Ограничения P2: TF forward-AD чувствительностей несовместим с предшествующими gradient-прогонами в пределах одного NetworkRNN (UQ — на свежем экземпляре); статистика (FIM/Лаплас/профили) — в единицах полного NLL (`likelihood_row_scale`, `likelihood_normalization`), хотя `ResidualLoss` по умолчанию делит вклад на N; split-режим (по умолчанию) интегрирует связанную сеть расщеплением с порядком не выше 2; режим `integration_mode='network'` снимает это ограничение (RK4 ≈4 на связанной TM-сети), но там `stability_loss` не оценивается и analytical adjoint не поддержан (docs/integrators.md); балансировка — BPTT-only; process parallelism окупается только на дорогих оценках; tensor candidate-batch вне P2; `MultipleShootingProblem` строится только поверх `CalibrationProblem` (не поверх `MultiExperimentProblem`); эксперимент-специфичные начальные состояния `x0/<experiment>/<pop>/<slot>` (`ParameterSpace.add_experiment_initial_state_entries`, `StateLocator.experiment`) материализуются только в `MultiExperimentProblem` (приоритет exp → common → default; orphan-записи и использование вне экспериментного контекста — UserWarning), sensitivities/UQ exp-x0 пока не материализуют. Параметр `ParameterEntry.trainable` значим: вектор φ строится только по trainable-записям (`phi_entries`/`trainable_names`), non-trainable записи — реестровые константы вне φ (`entries`/`names` — интроспекция; старые phi смешанных пространств могут несовпасть по размеру). `minimize_lbfgsb` ftol=None — dtype-aware auto (float64 2.2e-9 / float32 ≈eps32) с warning при float32 ftol < 1e-7; worker-процессы `parallel` по умолчанию ограничены `tf_threads=1` (None — не трогать TF, RuntimeError — fallback на дефолтные потоки).
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

### Компилируемые режимы (этап 1)
GPU-запуск: docs/optimization.md «Запуск на GPU» (float32 рекомендован; tf.function-совместимость полная — один граф на любые T/B). Скан `NetworkRNN` и train step `Trainer` всегда исполняются через `tf.function` (eager-нормализация в `call` → `_scan_compiled` с закреплёнными input_signature; T и батч B динамические, одна сигнатура на любые T/B); опция `jit_compile=True` у `NetworkRNN`/`Trainer` — XLA (градиент под XLA не компилируется у `AdaptiveRKIntegrator`/`ExponentialEulerIntegrator` и split-шага `RosenbrockIntegrator` при динамических батч-осях; `run_eagerly`+`jit_compile` — ValueError, adjoint+`jit_compile` — NotImplementedError; сеть с `jit_compile=True` внутри `Trainer` без jit не сочетать — TensorList не пересекает границу XLA/TF). Слой `optimization`: флаг `compiled=True` у `CalibrationProblem`/`MultiExperimentProblem` (граф прогона+ленты+градиента через `_CompiledCore`, default False = прежний eager); мутация модели (assign), счётчики и numpy-`Evaluation` — снаружи eager; `simulate`/`validate` — eager в обоих режимах; кэш ядер по (experiment, id(objective), id(t), id(inputs)), view-хелперы строят новые ядра лениво. Проверки/точки входа: `tests/unit/test_graph_mode.py`, `tests/unit/test_compiled_problem.py`; замеры — `benchmarks/g0_graph_mode.py`.

### Batch B>1 (этап 2)
`NetworkRNN` поддерживает произвольный batch B: состояния популяций `[B, n]`, синапсов `[B, n_pre, n_post]`; один батч-прогон эквивалентен B индивидуальным (значения и градиенты); `initial_state=None` broadcast'ит per-sample состояние на B; `state_size` моделей — per-sample формы (публичный атрибут не меняется). Одинаковый протокол для всех слотов батча; `InputPopulation` внутри scan не поддержан. Популяционные шаги жёстких интеграторов (`Rosenbrock`/`IMEX`/`Exponential` split) — только batch=1 (guard ValueError/assert); `integration_mode='network'` (Euler/Heun/RK4/Rosenbrock) батч-агностичен. `AntiPhaseLoss` — корреляция per-batch-item. Тесты: `tests/unit/test_batch_mode.py`.

### Векторный режим MultiExperimentProblem (этап 3)
`MultiExperimentProblem(..., vectorize=True)`: все обучающие эксперименты одним батч-прогоном (требуются одинаковые T и согласованный профиль входов; default False = прежний цикловой режим). Потеря/градиент — СРЕДНЕЕ по экспериментам (цикловой — сумма: переключение меняет масштаб в E раз, эквивалентно весам 1/E); n_forward += 1 за батч-операцию. x0-записи материализуются батчево (`_materialize_batched_initial_state` в `_state.py`): common — broadcast на все слоты (градиент аккумулирует вклады), exp-specific — one-hot маска своего слота (изоляция). `simulate()` — срез эксперимента (None — первого); `validate()` — батчем для held-out той же длины/профиля, иначе по одному. adjoint несовместим (`_VectorizedObjective.per_step_loss` → NotImplementedError). compiled=True + vectorize=True — одно `_CompiledCore` на батч (x0-подстановка внутри графа через `x0_substitute`). Тесты: `tests/unit/test_vectorized_experiments.py`.

### Чанкинг длинных протоколов (этап 4)
`chunk_length=L` в `NetworkRNN.call` (и проброс в `Trainer`/`CalibrationProblem`/`MultiExperimentProblem`, включая compiled-ядра — параметры в ключе кэша): прогон сегментами длины L с carry=final_state, память градиента O(L). `chunk_grad='checkpointed'` (default) — точный BPTT через `tf.custom_gradient` с пересчётом сегмента (tf.recompute_grad теряет градиенты по переменным сети внутри вложенного tf.function — используется ручной custom_gradient, как fallback в sensitivities); `'truncated'` — stop_gradient (дёшево, смещён). Требует статической T; несовместим с adjoint (ValueError); `return_rates=False` — predict-only. stability_loss чанкованного прогона — взвешенное по длинам среднее (при weight=0 эквивалентен целому). Тесты: `tests/unit/test_chunking.py`.

### NetworkRNN uses `tf.while_loop`, not Keras RNN
`NetworkRNN` wraps a `tf.while_loop` over TensorArrays (not `tf.keras.layers.RNN`). Its `call(t_sequence, inputs=None, initial_state=None)` returns a `NetworkOutput` with `.firing_rates: dict[str, Tensor]`, `.stability_loss`, and `.final_state`. `inputs` (shape `[batch, T, total_input_units]`) carries the precomputed external input rates, packed via `graph.pack_inputs()`; the graph's external inputs are declared with `NetworkGraph.declare_input(name, n_units)`.

### Input time shape
Time sequence: `tf.Tensor` shape `[batch, T, 1]`. If passed as `[batch, T]`, it is automatically expanded.

### Auto-registration on import
Importing `neuraltide.populations`, `neuraltide.synapses`, or `neuraltide.inputs` auto-registers classes into the global `POPULATION_REGISTRY`, `SYNAPSE_REGISTRY`, `INPUT_REGISTRY` dicts in `neuraltide.config`. Required for the config-first API (`config/schema.py`).

### State shapes
All population state tensors carry a leading batch axis: `[B, n_units]` (synapses: `[B, n_pre, n_post]`). `B=1` is the per-sample default; arbitrary `B > 1` is supported by `NetworkRNN` (one batch run == B individual runs with a shared protocol). `state_size` attributes stay per-sample (`[1, n]` populations, `[n_pre, n_post]` synapses) — the batch axis is added/broadcast by `NetworkRNN` (`_build`/call).

## Docs status
- `README.md`: актуален (quick start через `declare_input`/`pack_inputs`, новые классы, Python >= 3.12).
- `SPEC.md` (бывший `MASTER_TZ.md`): историческая дизайн-спецификация v0.1.0. Расхождения с кодом помечены **[устарело]** в шапке и соответствующих разделах; источник истины по API — `docs/`.
