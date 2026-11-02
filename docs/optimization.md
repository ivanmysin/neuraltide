# Оптимизация и калибровка

Модуль `neuraltide.optimization` — слой калибровки поверх моделей сети. Он
реализует этапы **P0/P1** дорожной карты (`docs/optimization_roadmap.md`):
корректную постановку задачи и проверяемые градиенты (P0), затем практический
набор методов калибровки (P1): сильные локальные оптимизаторы SciPy,
мультистарт, режимы производных для длинных горизонтов, multiple shooting,
continuation-расписания и анализ идентифицируемости. `Trainer` остаётся
простым совместимым API градиентного обучения; все новые методы работают
поверх общего `CalibrationProblem`.

## Преобразования координат параметров (`transforms`)

Физические параметры модели θ связаны с координатами оптимизатора φ
дифференцируемым поэлементным преобразованием θ = forward(φ):

| Преобразование | Назначение |
|---|---|
| `identity` | текущее поведение params-формата без изменений |
| `affine` | центрирование и масштабирование (θ = a·φ + b) |
| `log` | положительные параметры, меняющиеся по порядкам величины (θ = exp φ) |
| `logit`, `scaled_sigmoid` | параметры ограниченного интервала |
| `softplus` | неотрицательные параметры с достижимой окрестностью нуля |
| `bounded_log` | логарифмическая координата внутри диапазона |

Фабрика `make_transform(kind, **kwargs)` и регистр `TRANSFORM_REGISTRY`.
Обратное преобразование `inverse(θ)` клиппит θ в допустимую область с
безопасным численным margin; `jacobian(φ)` возвращает dθ/dφ.

Предупреждение: sigmoid-образные преобразования насыщаются у границ
интервала (производная → 0), поэтому это не универсальный способ задания
ограничений — настоящие bounds (L-BFGS-B/TRF) появятся в P1.

## Реестр параметров (`ParameterSpace`)

`ParameterSpace.from_network(network, transforms={full_name: Transform})`
сканирует обучаемые переменные популяций и синапсов графа и строит плоский
вектор φ единый для всех оптимизаторов:

- `pack(values=None)` — θ → φ; `None` — текущие значения переменных модели;
- `unpack(phi)` — φ → θ (дифференцируемо, GradientTape-совместимо);
- `apply_to_model(theta_dict)` — запись θ в переменные модели;
- `project(phi)` — проекция φ в допустимую область (через физические
  min/max из params-формата).

Стабильные полные имена имеют вид `<owner>/<param>`, например
`exc/Delta_I` или `inh->exc/tau_f`. Параметры входных генераторов и
readout-слои автоматически не охватываются; nuisance-параметры (сдвиги,
масштабы, параметры шума) регистрируются вручную через `add_entry`.

## Наблюдения и невязки (`objectives`)

`ObservationSpec` описывает одно наблюдение: популяция, целевая траектория
`[batch, T, n_units]`, модель шума `sigma`, вес, маска, readout и вид
штрафа `kind`. Поддерживаются виды:

- `l2` — 0.5·e², e = pred − target;
- `relative` — 0.5·(e / (|target| + eps))²;
- `msle` — 0.5·(log1p(pred + eps) − log1p(target + eps))²;
- `huber` и `pseudo_huber` — порог `delta`, устойчивость к выбросам.

Нормировка — masked mean: `weight · sum(mask · psi) / (N · sigma²)`,
где N = max(sum(mask), eps) либо число элементов траектории. Маска масштаб
не меняет. `ResidualLoss` агрегирует набор спецификаций и разделяет два
интерфейса:

- скалярный: `__call__`, `components()` (вклады по именам), `target`
  (совместимо с `Trainer`), аддитивный `per_step_loss`;
- векторный: `residual(predictions)` — честные невязки r, такие что
  `0.5 · ||r||² == loss` с учётом sigma/weight/mask/kind. Это постановка
  для nonlinear least squares (Gauss–Newton / LM / TRF в P1).

## Задача калибровки (`CalibrationProblem`)

Инкапсулирует сеть, ParameterSpace, протокол (`t_sequence`, `inputs`,
`initial_state`) и objective (callable, `BaseLoss`/`CompositeLoss`;
`None` допускается с последующим присваиванием `problem.objective`):

- `simulate(phi=None)` — прямой прогон;
- `evaluate(phi=None)` — `Evaluation`: loss, компоненты, невязки,
  счётчики `n_forward`/`n_gradient`, wall time, dtype, dt;
- `value_and_gradient(phi=None)` — `(Evaluation, grad)` одним прогоном;
- `residual(phi=None)` — вектор невязок для GN/LM/TRF;
- `bounds_phi()` — φ-границы из физических min/max (заготовка для
  SciPy-адаптеров).

Контракт мутации модели: `phi=None` модель не изменяет; `phi != None`
явно переводит модель в состояние θ = unpack(phi) перед прогоном и
оставляет её в нём. Для сохранения/восстановления состояния:
`saved = problem.pack()` и
`space.apply_to_model(space.unpack(saved))`.

Режимы градиента (`grad_mode`): `'assign'` (по умолчанию) и `'chain'`. В
текущем eager TensorFlow градиент через `variable.assign` недоступен, поэтому
оба режима вычисляют dloss/dφ как dloss/dθ по переменным модели, умноженный
на dθ/dφ (jacobians transforms); результаты совпадают, что контролируется
taylor_test'ом.

## Результаты (`results`)

- `Evaluation` — результат одной оценки (см. выше), `to_dict()` с
  numpy → list для JSON;
- `OptimizationResult` — единый формат результата оптимизации: `phi_best`,
  физические `theta_best`, `loss_best`, история, seed, конфигурация,
  версии пакетов, dtype; `to_dict()`/`from_dict()`/`save()`/`load()`
  (JSON, numpy → list; сериализация самой модели не входит в контракт).

## Диагностика градиентов (`diagnostics`)

Пользовательские численные проверки (см. также `tests/unit/test_diagnostics.py`):

- `finite_difference_scan` — центральные конечные разности со сканированием h;
- `taylor_test` — направленный тест порядков 1 и 2 до численного предела;
- `dot_product_test` — сопряжённость прямого и транспонированного операторов;
- `dt_convergence` — сходимость потери/градиента при уточнении dt;
- `dtype_consistency_check` — согласованность float32/float64 через config;
- `compare_bptt_adjoint` — равенство потери и градиентов BPTT и adjoint.

## Локальные оптимизаторы (`local`, P1)

Два SciPy-адаптера с единым контрактом (возвращают `LocalOptimizerResult`):

```python
from neuraltide.optimization import least_squares, minimize_lbfgsb

res = least_squares(problem, phi0, method="trf")   # невязки problem.residual
res = minimize_lbfgsb(problem, phi0, maxiter=200)  # value_and_gradient
print(res.base.loss_best, res.n_forward, res.bounds_active)
```

- `minimize_lbfgsb` — `scipy.optimize.minimize(method='L-BFGS-B', jac=True)`
  поверх `value_and_gradient`; bounds из `problem.bounds_phi()`;
- `least_squares` — `scipy.optimize.least_squares` поверх `problem.residual`:
  `'trf'` с bounds как основной метод, `'lm'` только без bounds (честный
  `ValueError` с подсказкой перейти на `'trf'`); `jacobian` —
  `None | '2-point' | '3-point' | callable` (callable режим лениво
  связывается с `sensitivities.residual_jacobian`);
- оба проверяют финитность стартовой точки/потери/невязок до запуска
  (`ValueError` вместо безмолвных больших штрафов), пишут validation-метрики
  при непустом `held_out`, счётчики — дельта счётчиков задачи за запуск.

`least_squares` требует objective с честным residual-интерфейсом
(`ResidualLoss`), такой что `0.5·||r||² == loss`; иначе `TypeError` с
направлением на L-BFGS-B.

## Мультистарт и continuation (`strategies`, P1)

```python
from neuraltide.optimization import multistart, sample_phi0s, dt_continuation

ms = multistart(problem, n_starts=16, method="sobol", refine_top_k=4,
                local="least_squares", seed=7)
print(ms.base.loss_best, ms.clustering["n_clusters"], ms.solutions)

result = dt_continuation(build_problem, dts=[2.0, 1.0, 0.5],
                         optimizer="lbfgsb", maxiter=100)
```

- `sample_phi0s(space, n, method='sobol'|'lhs'|'random', seed=...)` —
  детерминированная выборка начальных φ: ограниченные записи равномерно (или
  лог-равномерно через `log_uniform=[names]`) в физических границах,
  неограниченные — гауссово вокруг текущего φ с `UserWarning`;
- `multistart` — дизайн -> screening (`screen_evals` оценок на старт) ->
  локальная доводка top-k; все старты сохраняются со статусами
  ('ok' / 'screen_rejected' / 'failed'), удачные кластеризуются greedy по
  близости φ (`dedup_rtol`); checkpoint/resume через `checkpoint_path` +
  `resume=True` (seed и конфигурация сверяются с чекпоинтом);
- `dt_continuation(build_problem, dts, ...)` — «coarse dt -> fine dt»:
  перенос θ между ParameterSpace соседних задач через unpack/pack;
- `subset_continuation(problem, stages_free_names, ...)` — цепочка
  `FrozenProblem` «reduced subset -> joint»; финальная совместная доводка
  добавляется автоматически, если последняя стадия не покрывает все записи.

## Чувствительности и длинные горизонты (`sensitivities`, P1)

```python
from neuraltide.optimization import (residual_jacobian,
                                     value_and_gradient_checkpointed,
                                     truncated_bptt)

jac = residual_jacobian(problem, phi_best)          # JacobianResult (J, r, loss)
ckpt = value_and_gradient_checkpointed(problem, phi0, segment_length=100)
tb = truncated_bptt(problem, phi0, tf.keras.optimizers.Adam(0.05),
                    horizon=100, epochs=3)
print(tb.free_run_loss)   # обязательный полный прогон после обучения
```

- `residual_jvp(problem, phi, v)` — одна forward-mode JVP-оценка `J·v`
  через `tf.autodiff.ForwardAccumulator` (primals — переменные модели и φ);
- `residual_jacobian(problem, phi, force=False, max_params=200)` — полный
  якобиан невязок по столбцам через n unit-JVP; при `n > max_params` без
  `force` — `ValueError` с честной оценкой стоимости и альтернативами;
- `value_and_gradient_checkpointed` — сегментный прогон с переносом
  final_state и checkpointing (`tf.recompute_grad` при подтверждённом probe,
  иначе ручной `tf.custom_gradient` fallback); градиент совпадает с plain BPTT
  — это критерий приёмки; `peak_memory` — best-effort (tracemalloc, нижняя
  оценка);
- `truncated_bptt` — явно приближённый режим: обрыв градиента
  (`tf.stop_gradient`) на границе окна, carry — финальное состояние
  предыдущего окна; после обучения всегда полный free-run.

## Multiple shooting (`shooting`, P1)

```python
from neuraltide.optimization import MultipleShootingProblem, ShootingConfig

ms = MultipleShootingProblem(problem, ShootingConfig(4), continuity_weight=1.0)
phi = ms.initialize_from_single_shooting()   # gaps ≈ 0
ev, grad = ms.value_and_gradient(phi)
check = ms.finalize(phi)                     # SingleShootingCheck
print(check.consistent, check.gaps["max"])
```

Горизонт разбивается на сегменты с независимыми начальными состояниями `s_k`
(записи `x0s/<k>/<pop>/<slot>`), общими физическими параметрами и
continuity-штрафом `0.5·w·Σ‖x_end,k − s_{k+1}‖²` (penalty method — не
augmented Lagrangian). Residual-раскладка `r = [r_data; √w·gap_k]` даёт
`0.5·‖r‖² == loss_ms` точно. `finalize` выполняет честный свободный
single-shooting прогон с `x0 = s_0` и проверяет разрывы.
`BrunelFokkerPlanck` не поддерживается (буфер задержки и CFL-субшагивание
несовместимы с точной сегментацией) — честный `NotImplementedError`.

## Continuation-расписания (`schedules`, P1)

```python
from neuraltide.optimization import (ContinuationScheduler, horizon_growth,
                                     teacher_forcing_schedule)

stages = horizon_growth(problem, [40, 80, 160])
stages = teacher_forcing_schedule(problem, target_rates, alphas=[0.5, 0.0])
result = ContinuationScheduler(stages).run(problem)
print(result.free_run_check.loss)   # оценка на исходных входах
```

- `Stage(name, problem, optimizer='lbfgsb'|'least_squares'|callable, maxiter,
  meta)` — стадия-view базовой задачи (тот же ParameterSpace, общие
  счётчики);
- `ContinuationScheduler(stages).run(problem)` — последовательный запуск с
  переносом φ и обязательным `free_run_check` на оригинальной задаче;
- helpers: `horizon_growth` (срез целей и временных масок ResidualLoss через
  observations), `teacher_forcing_schedule` (смешивание только во входах; последний alpha
  обязан быть 0 — иначе `ValueError` ещё на построении), `objective_schedule`,
  `smooth_objective` (скользящее среднее целей, инициализационная стадия).

## Идентифицируемость (`identifiability`, P1)

```python
from neuraltide.optimization import compute_fim, likelihood_profile, select_subset

jac = residual_jacobian(problem, phi_best)
fim = compute_fim(jac, param_names=space.names)
print(fim.effective_rank, fim.condition_number, fim.stiff_dirs)

profile = likelihood_profile(problem, "exc/tau", grid=np.linspace(4, 14, 7))
print(profile.intervals["0.95"], profile.warnings)

subset = select_subset(problem, jac, keep=["exc/tau", "exc/I_ext"])
print(subset.loss_before, subset.loss_after, subset.prediction_delta)
```

- `compute_fim(jac, energy=0.99, collinearity_threshold=0.95, max_dirs=10)` —
  FIM `F = JᵀJ` по готовому якобиану невязок (нормированным на sigma):
  спектр, `condition_number` = κ(J), `condition_number_fim` = κ(F),
  effective rank по накопленной энергии собственных значений, жёсткие/мягкие
  направления с вкладом параметров, коллинеарные пары столбцов J, нормы
  чувствительности. Якобиан здесь не вычисляется — передайте `JacobianResult`;
- `likelihood_profile(problem, name, phi=None, values|grid, optimizer='lbfgsb',
  warm_start=True)` — одномерный профиль по физическому θ фиксированной
  сеткой (не адаптивной): координата замораживается `FrozenProblem`, остальные
  доводятся локально (warm start от соседнего успешного узла). `phi=None` —
  старт из `problem.pack()`; если `pack()` невозможен (tensor-backed записи
  без переменных модели, например начальное состояние), передайте стартовую
  точку явно через `phi=...` — иначе `ValueError` с пояснением. Интервал
  `intervals['0.95']` по порогу `χ²_0.95(1)/2 ≈ 1.9207` на `rel_loss` для
  гауссова шума с известной sigma; неуспешные узлы честно фиксируются
  (losses=NaN, successes=False), немонотонность/несвязность/минимум на
  границе — в `warnings`. Только скалярные записи (numel == 1);
- `select_subset(problem, jac, keep, refit=True)` — совместная доводка
  подмножества поверх FrozenProblem с честной оценкой смещения: дельты θ
  (фиксированные структурно нулевые — свойство постановки), сдвиг потерь,
  prediction_delta по firing_rates обучающего протокола и validation_delta
  по held-out (None с пояснением в notes при отсутствии).

## Ограничения этапа P1

- **Eager-режим**: все методы P1 работают в eager TensorFlow; `tf.function`
  вокруг методов задачи не поддерживается;
- **batch=1**: контракт NetworkRNN; ансамблевая векторизация — этап P2;
- **Shooting**: только penalty method — continuity обеспечивается
  взвешенным штрафом, не augmented Lagrangian; всегда завершайте
  `finalize`-чеком. Periodic boundary conditions не реализованы (сегменты
  стартуют из независимых состояний `s_k`, свободных/замкнутых кольцевых
  условий нет); инициализация shooting-состояний `x0s/*` из данных не
  реализована — доступен только перенос `initialize_from_single_shooting()`;
- **FIM по якобиану**: `compute_fim` принимает готовый `JacobianResult`;
  качество FIM — качество J (forward sensitivities либо конечные разности);
  интервалы профиля — для гауссова шума с известной sigma, для huber/robust
  видов штрафа эвристичны;
- **Teacher forcing honesty**: смешивание выполняется только во входах,
  последний alpha обязан быть 0, итоговая оценка — `free_run_check` на
  исходных входах; truncated BPTT — приближённый режим (сравнивать
  `free_run_loss`, не `final_loss`);
- **Мультистарт последователен**: старты прогоняются по очереди (shared
  counters); параллельность — этап P2.

Сквозные примеры: `examples/example_07_calibration.py` (P0),
`examples/example_08_p1_workflow.py` (P1: мультистарт -> якобиан -> FIM ->
профиль -> валидация). Бенчмарки L0-L2: `benchmarks/`.
