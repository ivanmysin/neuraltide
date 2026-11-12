# Оптимизация и калибровка

Модуль `neuraltide.optimization` — слой калибровки поверх моделей сети.
Все методы работают поверх общего `CalibrationProblem` и покрывают:

- дифференцируемые преобразования координат параметров (θ ↔ φ) и единое
  пространство параметров;
- нормированные невязки и постановку задачи калибровки (в том числе
  мультиэкспериментную), диагностику градиентов;
- локальные оптимизаторы SciPy (L-BFGS-B, least squares TRF/LM),
  мультистарт и continuation-стратегии;
- производные режимы для длинных горизонтов (forward sensitivities,
  checkpointed и truncated BPTT), multiple shooting, continuation-расписания
  (включая teacher forcing);
- анализ идентифицируемости (FIM, профили правдоподобия, выбор подмножества);
- глобальный поиск (DE, CMA-ES, SPSA, basinhopping, NSGA-II), балансировку
  компонент целей и робастные градиенты (PCGrad/CAGrad);
- ограничения на прогнозы (augmented Lagrangian, epsilon-constraint),
  оценку неопределённости (Лаплас, бутстреп, прогнозные интервалы) и
  пакетную оценку кандидатов в нескольких процессах.

`Trainer` остаётся простым совместимым API градиентного обучения.

## Преобразования координат параметров (`transforms`)

Физические параметры модели θ связаны с координатами оптимизатора φ
дифференцируемым поэлементным преобразованием θ = forward(φ):

| Преобразование | Назначение |
|---|---|
| `identity` | params-формат без изменений |
| `affine` | центрирование и масштабирование (θ = a·φ + b) |
| `log` | положительные параметры, меняющиеся по порядкам величины (θ = exp φ) |
| `logit`, `scaled_sigmoid` | параметры ограниченного интервала |
| `softplus` | неотрицательные параметры с достижимой окрестностью нуля |
| `bounded_log` | логарифмическая координата внутри диапазона, 0 < min: θ(0) = √(min·max) |

Фабрика `make_transform(kind, **kwargs)` и регистр `TRANSFORM_REGISTRY`.
Обратное преобразование `inverse(θ)` клиппит θ в допустимую область с
безопасным численным margin; `jacobian(φ)` возвращает dθ/dφ.

Жёсткие границы параметров задаются через `min`/`max` в params-формате:
они поддерживаются локальными решателями с bounds (L-BFGS-B, TRF) и
учитываются при сэмплировании стартов. Sigmoid-образные преобразования
насыщаются у границ интервала (производная → 0) — это мягкий способ
удержания в диапазоне, а не замена bounds.

## Реестр параметров (`ParameterSpace`)

`ParameterSpace.from_network(network, transforms={full_name: Transform})`
сканирует обучаемые переменные популяций и синапсов графа и строит плоский
вектор φ, единый для всех оптимизаторов:

- `pack(values=None)` — θ → φ; `None` — текущие значения переменных модели;
- `unpack(phi)` — φ → θ (дифференцируемо, GradientTape-совместимо);
- `apply_to_model(theta_dict)` — запись θ в переменные модели;
- `project(phi)` — проекция φ в допустимую область (через физические
  min/max из params-формата).

Стабильные полные имена имеют вид `<owner>/<param>`, например
`exc/Delta_I` или `inh->exc/tau_f`. Параметры входных генераторов и
readout-слои автоматически не охватываются; nuisance-параметры (сдвиги,
масштабы, параметры шума) регистрируются вручную через `add_entry`.

Флаг `ParameterEntry.trainable` значим: вектор φ строится **только** по
записям с `trainable=True`. `entries`/`names` содержат все записи реестра
(интроспекция, duplicate-check), а `phi_entries`/`trainable_names` —
trainable-only view в порядке регистрации; именно оно определяет координаты
φ. Записи с `trainable=False` (ручные `add_entry(..., trainable=False)`,
а также переменные модели с `trainable=False`) — реестровые константы вне
φ: `size`, `pack`, `unpack`, `project`, `apply_to_model` и все φ-операции
задач их игнорируют. В смешанных пространствах с ручными non-trainable
записями размер φ может отличаться от числа всех записей — стартовые
точки пересобирайте через `pack()`; обращение к non-trainable имени там,
где ожидается координата φ (`likelihood_profile`, `select_subset`,
`sample_phi0s(log_uniform=...)`, `FrozenProblem`), отклоняется `ValueError`.

## Наблюдения и невязки (`objectives`)

`ObservationSpec` описывает одно наблюдение: популяция, целевая траектория
`[batch, T, n_units]`, модель шума `sigma`, вес, маска, readout и вид
штрафа `kind`. Поддерживаются виды:

- `l2` — 0.5·e², e = pred − target;
- `relative` — 0.5·(e / (|target| + eps))²;
- `msle` — 0.5·(log1p(pred + eps) − log1p(target + eps))²;
- `huber` и `pseudo_huber` — порог `delta`, устойчивость к выбросам;
- `cauchy` и `tukey` — масштаб `scale` c: cauchy с логарифмическим
  хвостом, tukey обнуляет вклад выбросов при |e| > c (нулевой градиент).

Вес `weight` — неотрицательный float либо 0-d Tensor/Variable (тогда
приводится к глобальному dtype и используется как есть — обучаемый или
адаптивный вес).

Нормировка — masked mean: `weight · sum(mask · psi) / (N · sigma²)`,
где N = max(sum(mask), eps) либо число элементов траектории. Маска масштаб
не меняет. `ResidualLoss` агрегирует набор спецификаций и разделяет два
интерфейса:

- скалярный: `__call__`, `components()` (вклады по именам), `target`
  (совместимо с `Trainer`), аддитивный `per_step_loss`;
- векторный: `residual(predictions)` — честные невязки r, такие что
  `0.5 · ||r||² == loss` с учётом sigma/weight/mask/kind. Это постановка
  для nonlinear least squares (Gauss–Newton / LM / TRF).

## Задача калибровки (`CalibrationProblem`)

Инкапсулирует сеть, ParameterSpace, протокол (`t_sequence`, `inputs`,
`initial_state`) и objective (callable, `BaseLoss`/`CompositeLoss`;
`None` допускается с последующим присваиванием `problem.objective`):

- `simulate(phi=None)` — прямой прогон;
- `evaluate(phi=None)` — `Evaluation`: loss, компоненты, невязки,
  счётчики `n_forward`/`n_gradient`, wall time, dtype, dt;
- `value_and_gradient(phi=None)` — `(Evaluation, grad)` одним прогоном;
- `residual(phi=None)` — вектор невязок для GN/LM/TRF;
- `bounds_phi()` — φ-границы из физических min/max (используются
  SciPy-адаптерами).

Все φ-операции задачи (`pack`/`unpack`/`bounds_phi`/`search_box_phi`,
материализация x0, `FrozenProblem`) работают только с trainable-записями
ParameterSpace (`phi_entries`/`trainable_names`): non-trainable записи —
реестровые константы вне вектора φ, их размерность не меняют.

Контракт мутации модели: `phi=None` модель не изменяет; `phi != None`
явно переводит модель в состояние θ = unpack(phi) перед прогоном и
оставляет её в нём. Для сохранения/восстановления состояния:
`saved = problem.pack()` и
`space.apply_to_model(space.unpack(saved))`.

Режимы градиента (`grad_mode`): `'assign'` (по умолчанию) и `'chain'`.
Оба режима вычисляют dloss/dφ как dloss/dθ по переменным модели,
умноженный на dθ/dφ (jacobians transforms), и дают совпадающий результат
(проверяется `taylor_test`).

## Мультиэкспериментная калибровка (`MultiExperimentProblem`)

Несколько экспериментов (`Experiment`: `name`, `t_sequence`, `inputs`,
`objective`, опциональный `initial_state`) над одной сетью и одним
`ParameterSpace`. Потеря — сумма потерь обучающих экспериментов, все
прогоны выполняются в одной `GradientTape`, поэтому градиенты общих
параметров суммируются по протоколам автоматически; `residual` —
конкатенация невязок, компоненты несут префикс `'{name}/'`. Held-out
эксперименты (`held_out=[Experiment]`) входят только в `validate`.

```python
from neuraltide.optimization import Experiment, MultiExperimentProblem

problem = MultiExperimentProblem(
    network, space,
    experiments=[exp_a, exp_b],      # у каждого — свой objective
    grad_mode="assign",              # или "chain"
    held_out=[exp_holdout],
)
```

**Общий и эксперимент-специфичный начальный вектор `x0`.** Компоненты
начального состояния регистрируются в ParameterSpace двумя способами:

- общие: `space.add_initial_state_entries(network)` → записи
  `x0/<population>/<slot>` (участвуют во всех экспериментах);
- эксперимент-специфичные:
  `space.add_experiment_initial_state_entries(network, "exp1")` → записи
  `x0/<experiment>/<population>/<slot>`; аргумент `include` принимает
  полные exp-имена, кортежи `(population, slot)` и
  `(experiment, population, slot)`; непотреблённые include-имена
  (опечатки) отклоняются `ValueError` со списком неизвестных имён —
  до любой регистрации; `transforms`/`bounds` задаются по
  exp-именам (записи корректно проходят через transforms, bounds и
  φ-операции).

При материализации начального состояния эксперимента действует приоритет
exp-specific → common → начальное состояние протокола/сети по умолчанию;
exp-запись перезаписывает слот common-записи (её тензор выходит из
состояния и из chain-градиента). Градиент по exp-specific `x0` идёт
только через соответствующий эксперимент.

Held-out внутри `MultiExperimentProblem` — обычный эксперимент по своему
имени: в `validate()` ему применяется его exp-specific `x0` (при
наличии), как обучающим. Отдельный `CalibrationProblem.validate()`
(held-out эксперименты вне multi-контекста) материализует только common
`x0`: exp-specific записи пропускаются с одноразовым `UserWarning`.

Правила предупреждений: записи, имя эксперимента которых не совпадает ни с
одним train/held-out экспериментом, никогда не материализуются — при каждом
конструировании `MultiExperimentProblem` выдаётся `UserWarning` со
списком таких «осиротевших» экспериментов (предупреждение повторяется
при каждом конструировании). Вне экспериментного контекста
(`CalibrationProblem`, shooting, robust) trainable exp-записи
пропускаются с одноразовым `UserWarning` (материализуются только у
эксперимента с совпадающим именем, включая held-out в `validate`).

Имена обучающих экспериментов должны быть уникальны и не пересекаться с
held-out — иначе `ValueError` при конструировании (два одинаковых имени
делили бы exp-specific `x0` одного имени молча).

Ограничения: sensitivities/UQ не материализуют exp-specific `x0`
(внеэкспериментный контекст — см. warning выше); `MultipleShootingProblem`
не строится поверх `MultiExperimentProblem` (shooting — над
`CalibrationProblem`).

## Результаты (`results`)

- `Evaluation` — результат одной оценки (см. выше), `to_dict()` с
  numpy → list для JSON;
- `OptimizationResult` — единый формат результата оптимизации: `phi_best`,
  физические `theta_best`, `loss_best`, история, seed, конфигурация,
  версии пакетов, dtype; `to_dict()`/`from_dict()`/`save()`/`load()`
  (JSON, numpy → list; сериализация самой модели не входит в контракт).

## Диагностика градиентов (`diagnostics`)

Пользовательские численные проверки:

- `finite_difference_scan` — центральные конечные разности со сканированием h;
- `taylor_test` — направленный тест порядков 1 и 2 до численного предела;
- `dot_product_test` — сопряжённость прямого и транспонированного операторов;
- `dt_convergence` — сходимость потери/градиента при уточнении dt;
- `dtype_consistency_check` — согласованность float32/float64 через config;
- `compare_bptt_adjoint` — равенство потери и градиентов BPTT и adjoint.

## Локальные оптимизаторы (`local`)

Два SciPy-адаптера с единым контрактом (возвращают `LocalOptimizerResult`):

```python
from neuraltide.optimization import least_squares, minimize_lbfgsb

res = least_squares(problem, phi0, method="trf")   # невязки problem.residual
res = minimize_lbfgsb(problem, phi0, maxiter=200)  # value_and_gradient
print(res.base.loss_best, res.n_forward, res.bounds_active)
```

- `minimize_lbfgsb` — `scipy.optimize.minimize(method='L-BFGS-B', jac=True)`
  поверх `value_and_gradient`; bounds из `problem.bounds_phi()`; `ftol=None`
  — dtype-aware auto: float64 → 2.2e-9, float32 → ≈eps32 (1.19e-7). Явный
  `ftol`/`options['ftol']` чтится без изменений (приоритет у `options`;
  `options['ftol']=None` — как «не задано»: fallback на явный `ftol`/auto,
  `None` в scipy не передаётся), resolved-значение фиксируется в
  `config['options']['ftol']`; при float32-координатах φ и эффективном
  `ftol < 1e-7` — одноразовый `UserWarning` (риск
  `ABNORMAL_TERMINATION_IN_LNSRCH`), значение не переопределяется;
- `least_squares` — `scipy.optimize.least_squares` поверх `problem.residual`:
  `'trf'` с bounds как основной метод, `'lm'` только без bounds (честный
  `ValueError` с подсказкой перейти на `'trf'`); `jacobian` —
  `None | '2-point' | '3-point' | callable` (callable режим лениво
  связывается с `sensitivities.residual_jacobian`). Дефолт (`jacobian=None`):
  forward-якобиан через `sensitivities.residual_jacobian` (1 оценка невязок
  + n JVP-прогонов) при `problem.size <= jacobian_max_params` (по умолчанию
  200) и полном контракте `CalibrationProblem`; иначе (превышение порога,
  недоступность sensitivities, view/duck-typed задача типа `FrozenProblem`)
  — честный fallback на `'2-point'`. Фактический режим фиксируется в
  `config["jacobian"]` (`'forward' | '2-point' | '3-point' | 'callable'`);
  `callback(intermediate_result)` — пользовательский callback итераций
  (поддержка scipy >= 1.16; на 1.15 не передаётся в scipy с `UserWarning`,
  у `method='lm'` не поддержан scipy — тоже `UserWarning`; история тогда
  стартовая + финальная). `maxiter` — унифицированный бюджет итераций как у
  `minimize_lbfgsb`: scipy у `least_squares` его не поддерживает, поэтому
  при `max_nfev=None` бюджет маппируется как
  `max_nfev = maxiter * (problem.size + 2)` (~n+1 оценка невязок на
  итерацию: n на конечно-разностный якобиан + пробные шаги; эффективное
  значение — в `config["max_nfev"]`); одновременная передача `maxiter` и
  `max_nfev` отклоняется `ValueError`. scipy-defaults ftol/xtol/gtol не
  меняются; явный `kwargs['ftol']` при float32-φ ниже 1e-7 даёт тот же
  одноразовый `UserWarning`, что и в `minimize_lbfgsb` (значение не
  переопределяется);
- оба проверяют финитность стартовой точки/потери/невязок до запуска
  (`ValueError` вместо безмолвных больших штрафов), пишут validation-метрики
  при непустом `held_out`, счётчики — дельта счётчиков задачи за запуск.

`least_squares` требует objective с честным residual-интерфейсом
(`ResidualLoss`), такой что `0.5·||r||² == loss`; иначе `TypeError` с
направлением на L-BFGS-B.

Правило forward-режима (`jacobian=None` по умолчанию): TF forward-AD
(`residual_jacobian`) несовместим с предшествующими gradient-прогонами в
пределах одного `NetworkRNN`; если задача уже использовалась для
BPTT-градиентов (`problem.n_gradient > 0` — L-BFGS-B, shooting, truncated
BPTT, градиентный screen-refine в `multistart`), адаптер сам переключается
на `'2-point'` с `UserWarning` (фактический режим — в
`config["jacobian"]`); для forward-режима передайте `jacobian=None` на
свежем экземпляре задачи или явный `jacobian='2-point'`/`'3-point'`.

## Мультистарт и continuation (`strategies`)

```python
from neuraltide.optimization import multistart, sample_phi0s, dt_continuation

ms = multistart(problem, n_starts=16, method="sobol", refine_top_k=4,
                local="least_squares", seed=7, screen_refine_steps=2)
print(ms.base.loss_best, ms.clustering["n_clusters"], ms.solutions)

result = dt_continuation(build_problem, dts=[2.0, 1.0, 0.5],
                         optimizer="lbfgsb", maxiter=100)
```

- `sample_phi0s(space, n, method='sobol'|'lhs'|'random', seed=...)` —
  детерминированная выборка начальных φ: ограниченные записи равномерно (или
  лог-равномерно через `log_uniform=[names]`) в физических границах,
  неограниченные — гауссово вокруг текущего φ с `UserWarning`;
- `multistart` — дизайн -> screening (по умолчанию один `evaluate` на
  старт; `screen_evals` > 1 — среднее по прогонам) -> опциональный
  screen-refine (`screen_refine_steps` > 0: короткий прогон `screen_local`
  от того же phi0 с `maxiter=screen_refine_steps`, `validate=False`; улучшает
  только оценку старта — `screened_loss = min(screened, refined)`, полная
  доводка по-прежнему стартует из исходного phi0) -> локальная доводка
  top-k; все старты сохраняются со статусами
  ('ok' / 'screen_rejected' / 'failed'), удачные кластеризуются greedy по
  близости φ (`dedup_rtol`); checkpoint/resume через `checkpoint_path` +
  `resume=True` (seed и конфигурация — включая `screen_refine_steps` и
  `screen_local` — сверяются с чекпоинтом, старты с `screen_refined=True`
  повторно screen-refine не проходят); чекпоинты, записанные без
  screen-refine в конфигурации, не resume-ятся: сверка конфигурации
  отклонит их с `ValueError` — прогон надо начать заново;
- замечание по совместимости режимов: градиентный screen-refine
  (`screen_local='lbfgsb'` или gradient-based callable) делает
  `problem.n_gradient > 0`, после чего default forward-якобиан полной
  доводки `local='least_squares'` (`jacobian=None`) на том же экземпляре
  `problem` несовместим с `NetworkRNN` — адаптер честно переключится на
  `'2-point'` с `UserWarning`. Для forward-режима используйте свежий
  экземпляр задачи или явный `jacobian='2-point'`; screen-refine через
  `screen_local='least_squares'` градиентов не трогает и с этим совместим;
- `dt_continuation(build_problem, dts, ...)` — «coarse dt -> fine dt»:
  перенос θ между ParameterSpace соседних задач через unpack/pack;
- `subset_continuation(problem, stages_free_names, ...)` — цепочка
  `FrozenProblem` «reduced subset -> joint»; финальная совместная доводка
  добавляется автоматически, если последняя стадия не покрывает все записи.

## Чувствительности и длинные горизонты (`sensitivities`)

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
  якобиан невязок по столбцам через n unit-JVP; невязки считаются один раз
  и переиспользуются для всех столбцов, поэтому стоимость — n+1 прямых
  прогонов (а не 2n); при `n > max_params` без `force` — `ValueError` с
  честной оценкой стоимости и альтернативами;
- `value_and_gradient_checkpointed` — сегментный прогон с переносом
  final_state и checkpointing (`tf.recompute_grad` при подтверждённом probe,
  иначе ручной `tf.custom_gradient` fallback); градиент совпадает с plain BPTT;
  `peak_memory` — best-effort (tracemalloc, нижняя оценка);
- `truncated_bptt` — явно приближённый режим: обрыв градиента
  (`tf.stop_gradient`) на границе окна, carry — финальное состояние
  предыдущего окна; после обучения всегда полный free-run.

## Multiple shooting (`shooting`)

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
`0.5·‖r‖² == loss_ms` точно. Состояния синапсов (например, R, U, A у
`TsodyksMarkramSynapse`) по умолчанию тоже shooting-переменные
(`include_syn_states=True`); при `False` они сбрасываются к начальным на
каждом узле, и склейка не совпадает со свободным прогоном (выдаётся
предупреждение). `finalize` выполняет честный свободный single-shooting
прогон с `x0 = s_0`, проверяет разрывы и сравнивает `loss_ms` с
`loss_single` (`loss_rtol`, `loss_atol`). Начальное состояние первого
сегмента `s_0` по умолчанию тоже свободно, то есть оценивается, а не
берётся из протокола.

`fix_first_segment=True` закрепляет сегмент 0 за начальным состоянием
базовой задачи (base initial state протокола с общими `x0/...`-записями
базового пространства): записи `x0s/0/...` не создаются, размерность φ
уменьшается ровно на число начальных слотов первого сегмента, стыки
`0->1, 1->2, ...` и `finalize`/`to_single_shooting_problem` сохраняются
(single shooting стартует из base state). `initialize_from_single_shooting`
не создаёт переменные для фиксированного сегмента 0.
`BrunelFokkerPlanck` не поддерживается (буфер задержки и CFL-субшагивание
несовместимы с точной сегментацией) — честный `NotImplementedError`.

## Continuation-расписания (`schedules`)

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
  observations), `teacher_forcing_schedule` (смешивание только во входах;
  последний alpha обязан быть 0 — иначе `ValueError` ещё на построении),
  `objective_schedule`,
  `smooth_objective` (скользящее среднее целей, инициализационная стадия).

## Идентифицируемость (`identifiability`)

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
  FIM `F = JᵀJ` по готовому якобиану невязок в единицах полного NLL:
  `ResidualLoss` по умолчанию делит вклад наблюдения на число точек N, и
  `compute_fim` снимает это деление через `JacobianResult.likelihood_scale`
  (строки J умножаются на sqrt(N)); без этого F была бы FIM/N, а интервалы
  шире в sqrt(N) раз;
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
  точку явно через `phi=...` — иначе `ValueError` с пояснением. Профиль
  считается в единицах полного NLL (контекст
  `objectives.likelihood_normalization` временно ставит `ResidualLoss`
  в `normalization='sum'`; `losses` — в этих единицах). Интервал
  `intervals['0.95']` по порогу `χ²_0.95(1)/2 ≈ 1.9207` на `rel_loss` для
  гауссова шума с известной sigma; шаг сетки должен быть меньше полуширины
  интервала, иначе интервал схлопывается в узел минимума; неуспешные узлы
  честно фиксируются (losses=NaN, successes=False), немонотонность/несвязность/минимум на
  границе — в `warnings`. Только скалярные записи (numel == 1);
- `select_subset(problem, jac, keep, refit=True)` — совместная доводка
  подмножества поверх FrozenProblem с честной оценкой смещения: дельты θ
  (фиксированные структурно нулевые — свойство постановки), сдвиг потерь,
  prediction_delta по firing_rates обучающего протокола и validation_delta
  по held-out (None с пояснением в notes при отсутствии).

## Глобальный поиск (`global_search`)

Единая точка входа для безградиентных методов поверх `CalibrationProblem`:

```python
from neuraltide.optimization import global_search

res = global_search(problem, method="de", max_evals=1500, seed=11,
                    refine_top_k=3, local="least_squares",
                    options={"popsize": 12, "polish": False})
print(res.base.loss_best, res.n_evals, res.message)
```

- методы: `'de'` (scipy Differential Evolution), `'cma'` (CMA-ES,
  optional extra `'global'`: `pip install 'neuraltide[global]'`),
  `'spsa'` (без границ — авто-масштаб по bounds_phi/phi0), `'basinhopping'`
  (scipy), `'nsga2'` (NSGA-II, optional extra `'pareto'` с `pymoo`;
  результат — `ParetoResult`, не `GlobalOptimizerResult`);
- `'de'`/`'cma'`/`'nsga2'` требуют конечных границ по всем координатам
  (явные `min`/`max` в params), иначе честный `ValueError` с советом;
- область поиска — `problem.search_box_phi()`: физический диапазон без
  1 % у каждого края, переведённый в φ. `bounds_phi()` для этого не годится:
  у sigmoid-образных преобразований там граница машинной точности (±11 во
  float32, ±31 во float64), и равномерная выборка ложилась бы к краям
  диапазона, а результат зависел бы от dtype. Локальная доводка использует
  полные `bounds_phi()`; явный аргумент `bounds` имеет приоритет;
- единый бюджет `max_evals`; `refine_top_k` — локальная доводка лучших
  кандидатов (`'lbfgsb'` | `'least_squares'` | callable);
- checkpoint/resume — только `'cma'`/`'spsa'` (scipy DE и basinhopping из
  seed не восстанавливаются); `resume=True` досчитывает бюджет;
- NaN/±inf потеря считается отказом кандидата (loss=+inf со счётчиком),
  без безмолвного завершения;
- `evaluator` — внешний пакетный callable для параллелизма (несовместим
  с `'nsga2'`); см. также `parallel.make_evaluator` ниже;
- отсутствие optional-зависимости поднимает `OptionalDependencyError` с
  сообщением об установке.

## Балансировка компонент и робастные градиенты (`robust`)

```python
from neuraltide.optimization import (GradNormWeights, balanced_train,
                                     pcgrad_project)

bt = balanced_train(problem, phi0, GradNormWeights(alpha=1.0),
                    optimizer="adam", max_steps=50,
                    projector=pcgrad_project)
print(bt.base.loss_best, bt.weight_history[-1], bt.baseline["best_loss"])
```

- `WeightStrategy` — протокол весов компонент; реализации `FixedWeights`,
  `UncertaintyWeighting` (w ∝ 1/sqrt(loss)), `SelfAdaptiveWeights`
  (w ∝ loss^(-exponent)), `GradNormWeights` (выравнивание норм градиентов).
  Это детерминированные суррогаты: веса пересчитываются аналитически из
  наблюдённых losses/градиентов шага, а не обучаются совместно с моделью;
- `balanced_gradient(problem, phi, weights, projector=None,
  projector_kwargs=None)` — один прогон с per-component GradientTape:
  (losses, взвешенный градиент); `projector_kwargs` — keyword-only mapping
  дополнительных аргументов проектора (напр. `{'seed': 0}` для PCGrad),
  без `projector` не допускается;
- `balanced_train(...)` — шаги Keras-оптимизатора с обновлением стратегии
  и логированием весов (`'w:<имя>'` в history); принимает те же `projector`/
  `projector_kwargs` — baseline-прогон использует их же, значение
  сохраняется в `base.config['projector_kwargs']` и упоминается в notes;
  обязательный baseline `FixedWeights(1.0)` из той же phi0 с тем же числом
  шагов — адаптивные веса не гарантируют улучшения, сравнивайте `best_loss`;
- `pcgrad_project(grads, *, seed=None)` / `cagrad_project(grads, c=0.5)` —
  проекции конфликтующих градиентов (Yu et al. 2020 / Liu et al. 2021) по
  карте {имя: grad} или списку градиентов. Обе модификации `pcgrad_project`
  проецируют против ИСХОДНЫХ `g_j` (как в оригинале), различие — порядок пар:
  `seed=None` (умолчание) — детерминированная batch-модификация
  (фиксированный порядок j=0..n-1); `seed=int` — оригинальный
  случайно-упорядоченный PCGrad статьи: порядок задач j случаен через
  `np.random.default_rng(seed)`, затем среднее; одинаковый seed
  воспроизводит результат, разные seed допустимо дают разные результаты.
  CAGrad решает `max_d min_i <g_i, d>` в шаре `||d − g0|| <= c||g0||`,
  поэтому худшая задача убывает не медленнее, чем при простом усреднении;
- `adaptive_window_sampler(problem, window_size, focus, boost)` — выбор
  временного окна с большим остатком и focused view задачи
  (`with_objective`, маска усилена в окне; исходные spec не мутируются);
- только BPTT-режим: adjoint/`Trainer` — `NotImplementedError`.

## Ограничения на прогнозы (`constraints`)

Ограничения вида c(x)=0 / g(x)≤0 на прогнозы сети (не на параметры —
bounds параметров это transforms + локальные решатели):

```python
from neuraltide.optimization import (ConstraintSpec, ConstrainedProblem,
                                     augmented_lagrangian, epsilon_constraint)

con = ConstraintSpec("inh_cap", "ineq",
                     lambda preds, net: preds.firing_rates["inh"] - 60.0)
result = augmented_lagrangian(problem, [con], max_outer=8, local="lbfgsb")
print(result.base.loss_best, result.violation_history, result.state.lambdas)
```

- `ConstraintSpec(name, kind, fn(predictions, network) -> Tensor,
  jacobian=None)`; для CalibrationProblem градиенты ограничений получаются
  автоматически в общей ленте (data loss + AL-член одним прогоном);
  generic-путь — `jacobian='fd'`|callable (см. `fd_jacobian`);
- `ConstrainedProblem(problem, constraints, state, augmented_residual=False)`
  — duck-обёртка с тем же контрактом; `loss = data + AL-штраф`, components
  дополнены `'al/penalty'` и `'constraint/<name>'`; `augmented_residual=True`
  дополняет residual блоками ограничений для least-squares постановки
  (осторожно: FD-якобиан штрафа на таких блоках шумный);
- `augmented_lagrangian(...)` — внешние итерации (обновление λ, рост μ при
  недостаточном сокращении нарушения), checkpoint/resume;
- `epsilon_constraint(problem, objectives, index, eps, ...)` — компонент
  `objectives[index]` становится ограничением `<= eps`, цель — сумма
  остальных (или явный `objective`).

## Неопределённость (`uq`)

```python
from neuraltide.optimization import (laplace_approximation,
                                     predictive_intervals,
                                     residual_diagnostics)

lap = laplace_approximation(problem, phi_map, method="gn")   # JᵀJ + prior
print(lap.intervals["exc/tau"], lap.pd, lap.jitter_used)
rng = np.random.default_rng(0)
samples = rng.multivariate_normal(lap.phi_map, lap.covariance + 1e-12*eye, 48)
pred = predictive_intervals(problem, samples, level=0.95)    # per-(pop, t, unit)
```

- `residual_diagnostics(problem, phi, max_lag, n_bins, windows)` — вектор
  невязок по ObservationSpec: mean/std/max|r|, временной ряд, автокорреляция,
  гетероскедастичность (Levene/Brown–Forsythe по временным бинам), сводки по
  окнам; fallback на плоский псевдо-spec без метаданных observations;
- `laplace_approximation(problem, phi_map, jac=None, prior_cov=None,
  method='gn'|'fd-hessian')` — гауссово приближение постериора: covariance =
  (JᵀJ + P)⁻¹ в единицах полного NLL (`'gn'`; J — готовый `JacobianResult`
  или внутренний `residual_jacobian` с капом `max_params`); `'fd-hessian'` —
  только sanity-check (2n+1 прогонов, кап `max_fd_params=50`). PD-проверка
  через Cholesky с нарастающим jitter (`pd`, `jitter_used`); маргинальные
  интервалы в физических θ (`intervals`) и в φ (`phi_intervals`) — границы
  min/max записей в них **не вшиты**;
- `bootstrap_experiments(problem, n_draws, evaluator=None)` — бутстреп по
  экспериментам/записям; `residual_bootstrap(problem, n_draws)` — по
  остаткам (с оговорками в notes). Для `MultiExperimentProblem` каждая
  draw — подвыборка экспериментов с повторением, где копии получают
  уникальные служебные имена `boot_{draw}_{position}_{original_name}`
  ('/' в имени sanitize'ится в '_'), counts подвыборки — в `weights`;
  пространства с эксперимент-специфичными x0-записями
  (`add_experiment_initial_state_entries`) отклоняются `ValueError` —
  такие записи на служебные имена не переносятся;
- `parameter_ensemble(results)` — ансамбль φ/θ из готовых результатов
  (`.base` любых результатов оптимизации), выборочная ковариация;
- `predictive_intervals(problem, phi_samples, level)` — прогнозные квантили
  firing_rates по ансамблю φ; batch обязан быть 1; нефинитные строки
  пропускаются (`n_skipped_nonfinite`);
- `noise_sensitivity(laplace_result, sigma_factors)` — чувствительность
  posterior к масштабу шума.

Все статистические величины (FIM, Лаплас, профили) — в единицах полного
NLL. Якобиан невязок конечен и при точном совпадении (exact-fit):
для l2/relative/msle невязка — сама ошибка `e`, для робастных видов
`sign(e)·sqrt(2ψ)` заменяется на `e` в точках ψ = 0. Прогнозные интервалы
`predictive_intervals` по сэмплам постериора — неопределённость параметров,
без шума наблюдений: их сравнивают с бесшумной траекторией, а не с данными.

## Пакетная оценка кандидатов (`parallel`)

```python
from neuraltide.optimization import evaluate_candidates, make_evaluator

records = evaluate_candidates(build_problem, candidates, n_workers=4, seed=0)
evaluator = make_evaluator(build_problem, n_workers=4, seed=0)  # для global_search
try:
    res = global_search(problem, method="de", evaluator=evaluator, ...)
finally:
    evaluator.close()
```

- `build_problem` — picklable фабрика уровня модуля (lambda/замыкания не
  сериализуются — честный `ValueError` до запуска процессов);
- `CandidateTask(index, phi, seed=None)` / `EvalRecord(index, loss,
  components, wall_time, n_forward, error)` — пары задача/результат;
  исключение кандидата → `loss=+inf`, `error='<Тип>: <сообщение>'`, остальные
  кандидаты вычисляются;
- `SerialPool` — лениво строит задачу в родителе, ordered игнорируется;
  `ProcessPool` — spawn-процессы, каждый со своей копией задачи (TF-граф
  строится в worker'е один раз), родительская модель не мутирует;
  `ordered=False` — выдача по мере завершения (индексы сохраняются);
- `tf_threads` (у `ProcessPool`, `evaluate_candidates`, `make_evaluator`;
  по умолчанию 1) — ограничение intra-op и inter-op пулов потоков
  TensorFlow в каждом worker-процессе: без него n_workers процессов
  держали бы пулы TF на все ядра CPU (oversubscription). `None` — не
  трогать потоки TF. Если пулы TF в worker'е уже инициализированы и
  настройка бросает `RuntimeError`, worker остаётся на дефолтной
  конфигурации потоков (fallback) — семантика оценки не меняется.
  `SerialPool` TF не затрагивает; при `n_workers=1` параметр
  валидируется, но игнорируется;
- один seed → одинаковые per-candidate сиды → потери воспроизводимы между
  SerialPool и ProcessPool;
- экономика: spawn + импорт TF в каждом worker'е стоят секунд — выигрыш
  есть при дорогих оценках (длинный горизонт, жёсткие интеграторы) и
  живом пуле между батчами; на мелких задачах serial быстрее;
- tensor candidate-batch (batch>1 через NetworkRNN в одном прогоне) не
  поддержан — параллелится только «один кандидат = один прогон».

## Компилируемый режим (tf.function)

По умолчанию всё считается в eager TensorFlow — удобно для отладки, но
каждая итерация оптимизатора платит Python-оверхед за построение графа
операций. Два независимых механизма компиляции снимают большую его часть
(оба опциональны; eager остаётся fallback):

**1. Keras-путь: `NetworkRNN(jit_compile=...)` / `Trainer(jit_compile=...)`.**
Скан симуляции и train step всегда исполняются через `tf.function` с
закреплёнными сигнатурами (см. `core/network.py`); `jit_compile=True`
дополнительно компилирует их через XLA. Ограничения XLA (TF 2.21, CPU):
прямой прогон компилируется для всех встроенных интеграторов; градиент
(train step) — для всех, кроме `AdaptiveRKIntegrator` и
`ExponentialEulerIntegrator` (InvalidArgumentError «fixed tensor list
size»). `run_eagerly=True` и `jit_compile=True` несовместимы (ValueError);
`grad_method='adjoint'` с `jit_compile=True` пока не поддержан
(NotImplementedError).

**2. Optimization-путь: `CalibrationProblem(..., compiled=True)` /
`MultiExperimentProblem(..., compiled=True)`.**
`value_and_gradient` / `evaluate` / `residual` выполняются через
скомпилированное ядро (`_CompiledCore`, `optimization/_compiled.py`):
граф прогона сети + GradientTape + градиента по φ трассируется один раз
на комбинацию (эксперимент × objective × протокол) и кэшируется; ядро
без XLA по умолчанию. Снаружи в eager остаются: мутация модели
(`assign` φ → θ), счётчики, numpy-сборка `Evaluation`, `simulate()` и
`validate()`. Ключ кэша — id объектов (задача держит ссылки): view-хелперы
(`time_slice` / `with_objective` / `with_inputs` / `with_initial_state`)
создают новые ядра лениво; явное `initial_state` протокола — аргумент
ядра, in-place смена отражается немедленно; мутирование objective по
месту ядром не отслеживается.

**Когда включать.** Длинные горизонты T и много итераций (L-BFGS-B,
DE, мультистарт): compiled/jit убирают Python-оверхед за итерацию
(замеры — `benchmarks/g0_graph_mode.py`). **Когда не включать.** Отладка
(стеки ошибок внутри графа менее читабельны); FD/Taylor-диагностика и
forward-чувствительности (forward-AD несовместим с предшествующими
gradient-прогонами на одном экземпляре сети — см. ограничения ниже);
сценарии с частой сменой objective по месту.

## Запуск на GPU

Пакет полностью `tf.function`-совместим (этап 1): скан `NetworkRNN`
исполняется через закреплённый граф с динамическими T и batch — один
граф на любые длины протокола и размеры батча, ретрейсинга при смене
размеров нет. На GPU дополнительных изменений кода не требуется.

**Проверка окружения.** TensorFlow должен видеть устройство:

```python
import tensorflow as tf
print(tf.config.list_physical_devices("GPU"))
```

Если список пуст — установлена сборка TF без CUDA-поддержки:
переустановите по официальной инструкции
(https://www.tensorflow.org/install/pip, `pip install tensorflow` или
`tensorflow[and-cuda]` под вашу ОС/драйвер). Пакет сам размещает
вычисления на GPU (placement по умолчанию TF); при нескольких
устройствах явное размещение — `with tf.device("/GPU:0"): ...`.

**dtype.** На GPU рекомендуется float32: float64 работает, но в 3–10 и
больше раз медленнее. Тип задаётся до создания любых объектов:

```python
import neuraltide
neuraltide.config.set_dtype(tf.float32)  # до построения сети/задач
```

**Рекомендуемая связка режимов.** `CalibrationProblem(...,
compiled=True)` / `MultiExperimentProblem(..., compiled=True)` —
компилируемое ядро прогона+градиента; `Trainer(jit_compile=True)` —
XLA-поверх (проверено на TF 2.21: градиент под XLA не компилируется у
`AdaptiveRKIntegrator`/`ExponentialEulerIntegrator`/split-`Rosenbrock`
— снимите `jit_compile` для них, прямой прогон под XLA работает у всех);
батч `B>1` в `NetworkRNN` и `vectorize=True` у многоэкспериментной
задачи — параллелизм протоколов одним прогоном; чанкинг
`chunk_length=L` для длинных временных рядов (память градиента O(L)).
Сочетание `jit_compile` + чанкинг не валидировано — используйте одно из
двого.

**Память.** Рост памяти GPU по умолчанию — стандартные средства TF:
`tf.config.experimental.set_memory_growth(device, True)` при
необходимости совместного использования карты. OOM видеопамяти на
длинном T: уменьшите `chunk_length`, размер батча, используйте
`return_rates=False` для predict-only либо отключите `jit_compile`.

**Проверка и замеры.** GPU smoke-тест: `pytest tests/unit/test_graph_mode.py
-k gpu` (skip, если GPU нет). Бенчмарк режимов:
`python benchmarks/g0_graph_mode.py` — штатные цифры CPU; на GPU
перезапустите для своей машины.

## Чанкинг длинных протоколов (этап 4)

Длинный горизонт T держит активации всех шагов для BPTT (память O(T)).
`chunk_length=L` разбивает прогон на сегменты длины L (carry = final_state
предыдущего сегмента; последний может быть короче) — память под активации
градиента O(L):

- `chunk_grad='checkpointed'` (default) — точный BPTT: каждый сегмент
  обёрнут в checkpointing, граничные carry сохраняются, активации сегмента
  пересчитываются при backward. Стоимость ~2× прямого прохода на сегмент;
  итоговый градиент совпадает с целым прогоном.
- `chunk_grad='truncated'` — stop_gradient между сегментами: дёшево
  (без пересчёта), но градиент смещён (вклад динамики через границы
  сегментов теряется).

**Как выбрать L.** Память ~ O(L); время растёт с числом сегментов
(checkpointed ~ 2× стоимость BPTT на шаг). Практика: начните с L=100–500
и уменьшайте при OOM; truncated берите, когда устойчивость важнее точности
градиента.

**Где включать.** `NetworkRNN.call(..., chunk_length=L, chunk_grad=...)`;
`Trainer(chunk_length=...)`, `CalibrationProblem`/`MultiExperimentProblem`
(`chunk_length=...`; view-хелперы наследуют — срез чанкуется внутри себя);
`return_rates=False` для predict-only без конкатенации rates.
Несовместимости: `grad_method='adjoint'` (у adjoint собственная память
O(1) — ValueError); чанкинг требует статической длины T; XLA-путь
(`jit_compile`) с чанкингом не валидирован. stability-штраф чанкованного
прогона — взвешенное среднее сегментных итогов Σ(stab_k·L_k)/T; при
`stability_penalty_weight>0` результат зависит от разбиения (в отличие
от целого пути, который берёт накопленный штраф последнего шага) — держите
weight=0 (default).

## Ограничения

- **Eager по умолчанию**: методы задач работают в eager TensorFlow;
  компилируемый режим (`compiled=True`) — опция, см. раздел
  «Компилируемый режим (tf.function)» выше;
- **Batch сети B>1 (этап 2)**: `NetworkRNN` поддерживает произвольный
  батч (состояния `[B, n]`, синапсы `[B, n_pre, n_post]`); один батч-прогон
  эквивалентен B индивидуальным с общим протоколом (значения и градиенты).
  Все слоты батча исполняют один и тот же протокол; `InputPopulation`
  внутри полного скана не поддержан; split-шаги жёстких интеграторов
  (`Rosenbrock`/`IMEX`/`Exponential` population step) — только batch=1.
  Пакетная оценка кандидатов optimization-слоя по-прежнему послойная
  (`evaluate_candidates`), не тензорный batch; `predictive_intervals` —
  по одному кандидату за раз;
- **Векторный режим экспериментов (этап 3)**:
  `MultiExperimentProblem(..., vectorize=True)` прогоняет все обучающие
  эксперименты одним батч-прогоном (одинаковые T и профиль входов —
  обязательны, иначе ValueError; default False = цикловой режим без
  изменений). Потеря и градиент — среднее по экспериментам, а не сумма:
  переключение режимов меняет масштаб loss/grad ровно в E раз
  (эквивалентно весам 1/E в цикловом). x0-записи ParameterSpace
  материализуются батчево (common — broadcast на слоты, exp-specific —
  one-hot в свой слот, изоляция вкладов сохраняется); residual —
  конкатенация, компоненты с префиксом '{exp}/'. Счётчик n_forward
  векторного режима: +1 за батч-прогон (семантика отличается от
  циклового +E — учитывайте при сравнении бюджетов). adjoint-метод
  несовместим с векторным режимом (per-step разложения нет).
- **TF forward-AD против GradientTape в пределах одного NetworkRNN**:
  скан — закреплённый `tf.function` (`_scan_compiled`, см.
  `core/network.py`); после gradient-прогона
  (value_and_gradient/локальные решатели/балансировка) forward-режим
  чувствительностей (`residual_jvp`/`residual_jacobian` внутри Laplace)
  в этом экземпляре падает (`variant/int64` в while_loop). Порядок
  «Laplace/якобиан ДО градиентных методов» или свежий экземпляр задачи
  через фабрику (как в `examples/example_09_p2_workflow.py`) — рабочие
  обходные пути; это ограничение TF+кэша tf.function;
- **Shooting**: только penalty method — continuity обеспечивается
  взвешенным штрафом, не augmented Lagrangian; всегда завершайте
  `finalize`-чеком. Periodic boundary conditions не реализованы (сегменты
  стартуют из независимых состояний `s_k`, свободных/замкнутых кольцевых
  условий нет); инициализация shooting-состояний `x0s/*` из данных не
  реализована — доступен только перенос `initialize_from_single_shooting()`;
  `MultipleShootingProblem` строится только поверх `CalibrationProblem`;
- **FIM по якобиану**: `compute_fim` принимает готовый `JacobianResult`;
  качество FIM — качество J (forward sensitivities либо конечные разности);
  интервалы профиля — для гауссова шума с известной sigma, для huber/robust
  видов штрафа эвристичны;
- **Teacher forcing honesty**: смешивание выполняется только во входах,
  последний alpha обязан быть 0, итоговая оценка — `free_run_check` на
  исходных входах; truncated BPTT — приближённый режим (сравнивать
  `free_run_loss`, не `final_loss`);
- **Мультистарт последователен**: старты прогоняются по очереди (shared
  counters); параллельность кандидатов — через `parallel`;
- **Optional deps**: `'cma'` — extra `global` (`pip install
  'neuraltide[global]'`), `'pymoo'` — extra `pareto`; отсутствие поднимает
  `OptionalDependencyError` с сообщением об установке;
- **Процессный параллелизм**: spawn-процессы заново импортируют TF — на
  мелких задачах serial быстрее; выигрыш при дорогих оценках и живом пуле
  (`make_evaluator`), потолок — число ядер с учётом конкуренции за
  intra-op потоки (worker'ы по умолчанию ограничены `tf_threads=1`, см.
  раздел `parallel`); tensor candidate-batch не поддержан;
- **Балансировка — BPTT-only**: PCGrad/CAGrad/GradNorm/self-adaptive веса
  не работают с adjoint-режимом (`NotImplementedError`); аналитический
  adjoint Trainer для этих целей не поддержан;
- **Адаптивные веса — суррогаты**: пересчёт по наблюдённым losses/градиентам,
  не совместное обучение log-sigma; baseline FixedWeights обязателен для
  честного сравнения;
- **Laplace — гауссово приближение**: в плоских/скошенных направлениях
  posterior интервалы приближённы (границы min/max не вшиты); `'fd-hessian'`
  — sanity-check с капом 50 параметров; bootstrap-выводы опираются на
  число независимых экспериментов/остатков;
- **Bootstrap и exp-specific x0**: `bootstrap_experiments` для
  `MultiExperimentProblem` переименовывает копии экспериментов draw в
  уникальные служебные имена, поэтому эксперимент-специфичные начальные
  состояния `x0/<experiment>/<pop>/<slot>` на них не переносятся —
  пространства с такими записями отклоняются `ValueError` до цикла draw;
- **Sensitivities/UQ и exp-specific x0**: exp-specific начальные состояния
  материализуются только в `MultiExperimentProblem` (вне его — warning и
  пропуск, см. выше);
- **Жёсткие интеграторы** (`docs/integrators.md`): substepping внутри
  `derivatives()`, синапсы остаются явными, while_loop-adjoint для них не
  поддержан — совместимость с чувствительностями/BPTT проверяйте
  отдельно (`stiffness_report`).

Сквозные примеры: `examples/example_07_calibration.py` (постановка задачи
и верификация градиентов), `examples/example_08_p1_workflow.py`
(мультистарт -> якобиан -> FIM -> профиль -> валидация),
`examples/example_09_p2_workflow.py` (DE -> L-BFGS-B -> балансировка ->
Laplace -> прогнозные интервалы + SerialPool).
