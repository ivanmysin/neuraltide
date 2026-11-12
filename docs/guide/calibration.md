# Калибровка по данным

Центральный рецепт NeuralTide: от экспериментальной записи и сети — к
подогнанным параметрам с проверкой результата. Раздел идёт по порядку
построения задачи: данные и целевая функция → пространство параметров →
задача калибровки → верификация → оптимизаторы → стратегии поиска →
мультиэкспериментные постановки. Анализ идентифицируемости и
неопределённость — в [Расширенной калибровке](advanced_calibration.md).

## Обзор потока

```
данные (target, sigma, mask)
        │
        ▼
ObservationSpec + ResidualLoss          ← что сравниваем с чем
        │
ParameterSpace (+ transforms)           ← что оптимизируем, границы
        │
        ▼
CalibrationProblem                      ← simulate / evaluate / residual / ∇
        │
        ▼
верификация (taylor, dt, adjoint)  →  оптимизация  →  проверка (held-out, free-run)
```

По умолчанию вычисления — в eager-режиме (компилируемый режим — см.
ниже); ведущая размерность батча равна 1.

## Данные и целевая функция

**Когда использовать.** `ObservationSpec` описывает одно сопоставление
«популяция ↔ данные»: целевую траекторию, масштаб шума `sigma`, маску,
вес и вид штрафа. `ResidualLoss` агрегирует набор спецификаций в одну
потерю с честным residual-интерфейсом `0.5·||r||² == loss` — этот
вектор невязок нужен методам нелинейных наименьших квадратов
(TRF/LM/Gauss–Newton, см. ниже).

**Минимальный пример:**

```python
from neuraltide.optimization import ObservationSpec, ResidualLoss

objective = ResidualLoss([
    ObservationSpec(
        name="exc",             # уникальное имя (ключ в components)
        population="exc",       # имя популяции в NetworkGraph
        target=target_exc,      # [batch, T, n_units]
        sigma=2.0,              # масштаб шума; вклад делится на sigma**2
        weight=1.0,
        kind="l2",              # l2|relative|msle|huber|pseudo_huber|cauchy|tukey
    ),
])
ev_components = objective.components(predictions, network)   # вклады по именам
r = objective.residual(predictions)                          # плоский вектор невязок
```

Полезные поля `ObservationSpec`:

- `mask` — тензор, broadcastable к `target` (например `[1, T, 1]`);
  семантика masked mean, масштаб маски значение не меняет — удобно
  выключать участки записи;
- `kind` — вид штрафа: `l2` (по умолчанию), `relative` (для целей около
  нуля), `msle` (мультипликативные ошибки), `huber`/`pseudo_huber`
  (порог `delta`, робастность к выбросам), `cauchy`/`tukey` (масштаб
  `scale`, тяжёлые хвосты; у `tukey` вне `|e| > scale` вклад в градиент
  нулевой);
- `readout` — опциональный `BaseReadout` (например, полосовая фильтрация
  тета-диапазона перед сравнением);
- `sigma` — оценка стандартного отклонения шума записи; неверный `sigma`
  искажает относительные веса наблюдений и последующую статистику.

Нормировка `ResidualLoss(observations, normalization=...)`: `'mean'`
(по умолчанию) — вклад делится на число элементов, масштаб потери не
зависит от длительности записи (удобно для оптимизации); `'sum'` —
полный NLL без деления. Модули анализа (FIM, Лаплас, профили) сами
переводят невязки в единицы полного NLL.

**Типичные ошибки и ограничения:**

- Имена наблюдений должны быть уникальны — они же ключи словаря
  компонент.
- `msle` требует неотрицательных предсказаний; `cauchy`/`tukey` —
  положительного `scale`.
- Робастные виды (`huber` и далее) снижают влияние выбросов, но меняют
  статистическую интерпретацию — для итоговых интервалов проверяйте
  остатки (см. [Расширенную калибровку](advanced_calibration.md#оценка-неопределённости)).

## Пространство параметров и преобразования

**Когда использовать.** `ParameterSpace` собирает плоский вектор φ из
обучаемых (`trainable: True`) параметров сети и связывает его с
переменными модели. Преобразования (`Transform`) переводят физические
параметры θ (с границами) в неограниченные координаты φ, в которых
работают оптимизаторы: θ = forward(φ), φ = inverse(θ).

**Минимальный пример:**

```python
from neuraltide.optimization import (
    BoundedLogTransform, LogTransform, ParameterSpace,
)

space = ParameterSpace.from_network(network, transforms={
    "exc/tau": LogTransform(),                      # theta > 0
    "in->exc/gsyn_max": LogTransform(),
    "exc/Delta_I": BoundedLogTransform(min=0.01, max=2.0),  # 0 < min < max
})

phi = space.pack()                # текущие θ -> φ
theta = space.unpack(phi)         # φ -> θ (дифференцируемо)
space.apply_to_model({"exc/tau": tf.constant([[12.0]])})   # явная запись θ
lo, hi = problem.bounds_phi()     # границы по φ (problem — см. ниже)
```

Выбор преобразования:

| Преобразование | θ = | Когда использовать |
|----------------|-----|--------------------|
| `IdentityTransform()` | φ | Параметры без ограничений |
| `AffineTransform(a, b)` | a·φ + b | Сдвиг/масштаб координат |
| `LogTransform(floor)` | exp(φ) (+ floor) | Строго положительные величины без верхней границы |
| `SoftplusTransform(floor)` | softplus(φ) (+ floor) | Как `log`, но мягче у нуля |
| `LogitTransform(min, max)` | min + (max−min)·σ(φ) | Параметры внутри интервала |
| `ScaledSigmoidTransform(min, max, s)` | то же, с масштабом s | Интервал + контроль крутизны |
| `BoundedLogTransform(min, max)` | exp-интерполяция в (min, max) | Положительные величины внутри интервала |

Записи с `trainable: False` остаются в реестре (`space.entries`,
`space.names`) как константы, но не входят в вектор φ: размер φ —
`space.size`, координаты — `space.trainable_names`.

**Типичные ошибки и ограничения:**

- Sigmoid-образные преобразования насыщаются у границ: производная
  dθ/dφ → 0, градиент по φ затухает, оптимизатор «прилипает» у края.
  У нижней границы `log`/`softplus` ведут себя так же (dθ/dφ ≈ e^φ);
  их преимущество — отсутствие верхней границы.
- Старые векторы φ, сохранённые при другом составе пространства, могут
  не совпасть по размеру — храните вместе с φ состав `trainable_names`.
- Начальное значение положительного параметра при `BoundedLogTransform`
  при φ = 0 равно √(min·max) — учитывайте при задании стартовых точек.

## Задача калибровки: CalibrationProblem

**Когда использовать.** `CalibrationProblem` — единая точка входа для
всех оптимизаторов: один и тот же объект принимают L-BFGS-B, TRF/LM,
мультистарт, глобальный поиск, continuation и методы анализа.

**Минимальный пример:**

```python
from neuraltide.optimization import CalibrationProblem

problem = CalibrationProblem(
    network,
    space,
    t_seq,
    inputs=inputs,            # упакованные входы (или None)
    objective=objective,      # ResidualLoss или callable(predictions, network)
    initial_state=None,       # (pop, syn) — при необходимости
)

ev = problem.evaluate()                    # Evaluation: loss, components, residuals
out = problem.simulate()                   # прямой прогон NetworkOutput
ev, grad = problem.value_and_gradient()    # потеря и ∇ по φ
r = problem.residual()                     # вектор невязок (np.ndarray)
lo, hi = problem.bounds_phi()              # физические границы в координатах φ
```

Контракт мутации: вызов с `phi=None` не меняет модель; вызов с явным
`phi` переводит модель в θ = unpack(φ) и оставляет в ней. Счётчики
`problem.n_forward`/`problem.n_gradient` общие для всех операций —
удобно оценивать бюджет.

**Типичные ошибки и ограничения:**

- Если у сети объявлены внешние входы, `inputs` обязателен — прогон без
  входов невозможен.
- `residual()` требует objective с residual-интерфейсом (`ResidualLoss`);
  иначе статус `Evaluation.status == "no_residual_interface"`.
- API по умолчанию eager (есть опция `compiled=True`, см. ниже) и
  пакетный размер 1; прогоны длинных горизонтов дорогие — об этом ниже
  (continuation, shooting).

**Компилируемый режим (`compiled=True`).** Для длинных горизонтов и
многоитерационной оптимизации (L-BFGS-B, DE, мультистарт) включайте
компиляцию задачи:

```python
problem = CalibrationProblem(network, space, t_seq, inputs=inputs,
                             objective=loss, compiled=True)
```

`value_and_gradient` / `evaluate` / `residual` пойдут через
скомпилированное ядро — один граф прогона+ленты+градиента на комбинацию
(objective × протокол); Python-оверхед на итерацию заметно ниже (замеры —
`benchmarks/g0_graph_mode.py`). Мутация модели, счётчики и `Evaluation`
остаются eager; `simulate` / `validate` не меняются. Не включайте при
отладке и FD/Taylor-диагностике (forward-AD несовместим с
gradient-прогонами на одном экземпляре сети). Подробности и связка с
`Trainer(jit_compile=True)` — в
[optimization.md](../optimization.md#компилируемый-режим-tffunction).

## Верификация задачи до оптимизации

**Когда использовать.** Обязательный шаг перед любой оптимизацией:
проверить, что градиент корректен и что задача не чувствительна к шагу
интегрирования. Дешевле поймать ошибку здесь, чем после сотен прогонов
L-BFGS-B.

**Минимальный пример:**

```python
import numpy as np
from neuraltide.optimization import (
    compare_bptt_adjoint, dt_convergence, taylor_test,
)

# 1. Направленный Taylor test: наклоны ≈ 1 и ≈ 2
def value_and_grad(phi_np):
    ev, grad = problem.value_and_gradient(phi_np)
    return ev.loss, np.asarray(grad)

report = taylor_test(value_and_grad, np.asarray(problem.pack()) + 0.25)
print(report["slope_R0_median"], report["slope_R1_median"])

# 2. Сходимость по dt: потеря и градиент на уточняющейся сетке
rows = dt_convergence(build_and_run, dts=[0.5, 0.25, 0.125])

# 3. Согласованность BPTT и adjoint (если используете adjoint-метод)
check = compare_bptt_adjoint(network, loss_fn, t_seq, inputs=inputs, target=target)
```

`build_and_run(dt)` — ваша фабрика пересборки задачи при заданном шаге:
`callable(dt) -> {"loss": ..., "grads": ..., "outputs": ...}`. Доступны
также `finite_difference_scan` (конечные разности по скалярным h),
`dot_product_test` (сопряжённость JVP/VJP) и `dtype_consistency_check`
(согласованность float32/float64).

**Типичные ошибки и ограничения:**

- В окрестности точного минимума R0 вырождается в O(h²) — проверку
  проводите в точке вдали от минимума (как в примере выше, со сдвигом φ).
- Наклоны, зависшие у 0 или у предела округления, — повод проверить
  dtype (float32 округляет рано), шаг и границы параметров.
- `dt_convergence` не предназначен для `BrunelFokkerPlanck`.

## Локальные оптимизаторы: L-BFGS-B и least_squares

**Когда использовать какой метод.** Оба — обёртки SciPy поверх
`CalibrationProblem` с границами из `ParameterSpace`.

| Метод | Когда использовать |
|-------|--------------------|
| `minimize_lbfgsb` | Основной локальный метод: потеря общего вида, границы, квазиньютоновская сходимость |
| `least_squares(method="trf")` | Когда цель — сумма квадратов невязок (всегда для `ResidualLoss`): границы + доверительный регион |
| `least_squares(method="lm")` | Быстрая сходимость в гладкой окрестности минимума; только без границ |

**Минимальный пример:**

```python
from neuraltide.optimization import least_squares, minimize_lbfgsb

result = minimize_lbfgsb(problem, maxiter=200, gtol=1e-6)
print(result.base.loss_best, result.success, result.message)

# Gauss–Newton/TRF по вектору невязок (forward-якобиан при size <= 200)
fit = least_squares(problem, method="trf")
print(fit.base.loss_best, fit.bounds_active)   # координаты на границе
```

Детали, влияющие на устойчивость:

- `minimize_lbfgsb`: `ftol=None` включает dtype-aware авто-порог
  (float64 → 2.2e-9, float32 → ≈eps32). При float32 и эффективном
  `ftol < 1e-7` выдаётся предупреждение о риске сбоя линейного поиска —
  рекомендуется `ftol >= 1e-7` или float64.
- `least_squares`: якобиан по умолчанию — forward-чувствительности
  (`residual_jacobian`) при `problem.size <= jacobian_max_params` (200),
  иначе `'2-point'`; фактический режим — в `fit.base.config["jacobian"]`.
  Forward-режим несовместим с уже выполненными gradient-прогонами на том
  же экземпляре `NetworkRNN` — будет честный fallback на `'2-point'` с
  предупреждением.
- `least_squares(method="lm")` — только без границ; нефинитный старт или
  невязки отклоняются `ValueError` до запуска.

Оба возвращают `LocalOptimizerResult` с полями `base` (единый
`OptimizationResult`: `phi_best`, `theta_best`, `loss_best`), `success`,
`nit`/`nfev`/`njev`, `grad_norm`, `bounds_active`, счётчиками и
`validation` — потерями на held-out протоколах (см. ниже).

**Типичные ошибки и ограничения:**

- Локальный метод сходится к ближайшему локальному минимуму — для
  мультимодальных задач запускайте из нескольких стартов (следующий
  раздел).
- `bounds_active` не пустой — решение упёрлось в границу: расширяйте
  границы осмысленно или пересматривайте постановку.
- `validate=True` (по умолчанию) добавляет оценку held-out протоколов —
  держите их небольшими, это прямые прогоны.

## Стартовые точки, мультистарт и глобальный поиск

**Когда использовать какой инструмент.** Локальная оптимизация
чувствительна к стартовой точке. Если есть сомнения в выпуклости —
начинайте с разведки.

**Минимальный пример:**

```python
from neuraltide.optimization import (
    global_search, multistart, sample_phi0s,
)

# Дизайн стартовых точек в допустимой области:
starts = sample_phi0s(space, n=16, method="sobol", seed=7)

# Мультистарт: дизайн -> screening -> локальная доводка top-k
ms = multistart(
    problem,
    n_starts=16,
    method="sobol",
    refine_top_k=4,
    local="least_squares",
    seed=7,
)
best = ms.solutions[ms.best_index]
print(ms.base.loss_best, ms.clustering["n_clusters"])

# Безградиентный глобальный поиск (бюджет в оценках потери)
gs = global_search(
    problem,
    method="de",
    max_evals=500,
    refine_top_k=1,          # доводка лучших локальным методом
    local="lbfgsb",
    seed=7,
    options={"popsize": 10, "polish": False},
)
```

`multistart` строит детерминированный по `(method, seed)` дизайн
(`sobol` | `lhs` | `random`), оценивает все точки (screening), лучшие
`refine_top_k` доводит локальным методом; поддерживает checkpoint/resume
и кластеризацию решений. `sample_phi0s` — только генерация точек
(`clip_to_bounds=True` проецирует в область; `log_uniform` — список имён
для лог-равномерного разброса).

`global_search(method=...)`: `'de'` (дифференциальная эволюция),
`'cma'` (требуется пакет `cma`), `'spsa'` (без градиентов, шумоустойчив),
`'basinhopping'` (локальный поиск с прыжками), `'nsga2'` (многоцелевой,
требуется `pymoo`; возвращает `ParetoResult` с фронтом по компонентам
целей). `'de'`/`'cma'`/`'nsga2'` требуют конечных границ; checkpoint/resume
— у `'cma'`/`'spsa'`. Отсутствующий дополнительный пакет поднимает
`OptionalDependencyError`.

**Типичные ошибки и ограничения:**

- Глобальный поиск дорог: бюджет `max_evals` — это прямые прогоны. Для
  дешёвых задач это нормально; для дорогих — мультистарт с малым
  `screen_evals` и пакетная оценка кандидатов (см.
  [Параллельную оценку](advanced_calibration.md#пакетная-оценка-кандидатов)).
- Если все старты сходятся в одну точку с разбросом ниже
  `dedup_rtol` — задача, вероятно, одномодальна в области поиска.

## Continuation и расписания

**Когда использовать.** Длинные горизонты, жёсткие постановки и
протоколы со многими стадиями решаются ступенчато: от простого к сложному.
Два готовых помощника — `dt_continuation` (по шагу интегрирования) и
`subset_continuation` (по набору свободных параметров); полный контроль
даёт `ContinuationScheduler` над `Stage`.

**Минимальный пример:**

```python
from neuraltide.optimization import (
    ContinuationScheduler, Stage, dt_continuation, horizon_growth,
    objective_schedule, smooth_objective, subset_continuation,
    teacher_forcing_schedule,
)

# 1. Сначала грубый шаг, затем уточнение:
res_dt = dt_continuation(build_problem, dts=[2.0, 1.0, 0.5], optimizer="lbfgsb")

# 2. Сначала подгоняем главное, затем остальное:
res_sub = subset_continuation(
    problem,
    stages_free_names=[["exc/tau"], ["exc/tau", "in->exc/gsyn_max"]],
)

# 3. Расписание стадий с переносом phi между ними:
scheduler = ContinuationScheduler([
    Stage("rough",  problem.with_objective(smooth_objective(objective, window=5)), maxiter=50),
    Stage("fine",   problem, maxiter=200),
])
schedule = scheduler.run(problem)          # ScheduleResult с free_run_check
```

Teacher forcing — отдельный случай: для автономных (замкнутых) сетей
смешивание выполняется только во входах, `inputs_α = (1−α)·u + α·target`,
и последний `alpha` обязан быть 0 — финальная стадия работает на
исходных входах:

```python
stages = teacher_forcing_schedule(problem, target_rates=recorded_rates,
                                  alphas=[0.8, 0.5, 0.2, 0.0])
result = ContinuationScheduler(stages).run(problem)
```

`horizon_growth(base_problem, horizons, build_inputs=None)` строит
стадии по удлинению горизонта; `objective_schedule(base_problem,
objectives)` — по смене целевой функции; scheduler гарантирует
`free_run_check` на исходной задаче — честный прогон без смешивания.

**Типичные ошибки и ограничения:**

- `teacher_forcing_schedule` без `alphas[-1] == 0` — `ValueError`: иначе
  итог нельзя интерпретировать как свободный прогон.
- View-задачи стадий (`problem.with_objective`, `with_inputs`,
  `time_slice`, `with_initial_state`) делят счётчики с базовой задачей —
  бюджет учитывается общий.
- Перенос φ между стадиями не проецируется в границы автоматически —
  не меняйте границы между стадиями без необходимости.

## Multiple shooting

**Когда использовать.** Длинные горизонты, где сквозной прогон
чувствителен к начальному состоянию: горизонт разбивается на сегменты с
независимыми начальными состояниями, стыки стягиваются штрафом
(метод штрафа). Особенно полезно в сочетании с угадываемым начальным
состоянием.

**Минимальный пример:**

```python
from neuraltide.optimization import (
    MultipleShootingProblem, ShootingConfig, minimize_lbfgsb,
)

ms_config = ShootingConfig(nodes=5)               # или явные границы [0, t1, ..., T]
ms_problem = MultipleShootingProblem(
    problem, ms_config, continuity_weight=1.0, include_syn_states=True,
)
phi_ms = ms_problem.initialize_from_single_shooting()   # gaps ≈ 0
res = minimize_lbfgsb(ms_problem, phi_ms, maxiter=200)
check = ms_problem.finalize(res.base.phi_best)    # SingleShootingCheck
print(check.consistent, check.gaps["max"])
```

`fix_first_segment=True` закрепляет первый сегмент за начальным
состоянием базовой задачи (меньше размерность φ). Residual-раскладка
честная: `r = [r_data; √w·gap_k]`, `0.5·||r||² == loss_ms` точно.
Метод duck-совместим с `CalibrationProblem` — его принимают те же
оптимизаторы.

**Типичные ошибки и ограничения:**

- `BrunelFokkerPlanck` в постановке не поддержан (`NotImplementedError`).
- После доводки обязателен `finalize`: он сравнивает потерю multiple
  shooting с честным свободным прогоном и проверяет стыки (`gaps`,
  `consistent`).
- Больше сегментов = больше размерность φ и дороже шаг; начинайте с
  3–5 сегментов.

## Несколько экспериментов и начальные состояния

**Когда использовать.** Если данные сняты в нескольких протоколах
(разные стимулы, разные частоты входа) — соберите `Experiment` для
каждого и оптимизируйте сумму потерь в одной ленте градиента:
`MultiExperimentProblem`. Отложенные протоколы передаются в `held_out` —
их потеря считается в `validate()` без включения в оптимизацию.

**Минимальный пример:**

```python
from neuraltide.optimization import (
    Experiment, MultiExperimentProblem, ParameterSpace,
)

train_a = Experiment(name="freq8",  t_sequence=t_seq, inputs=inputs_a, objective=obj_a)
train_b = Experiment(name="freq16", t_sequence=t_seq, inputs=inputs_b, objective=obj_b)
holdout = Experiment(name="holdout", t_sequence=t_seq, inputs=inputs_h, objective=obj_h)

problem = MultiExperimentProblem(
    network, space, experiments=[train_a, train_b], held_out=[holdout],
)
ev, grad = problem.value_and_gradient()      # сумма потерь в одной GradientTape
validation = problem.validate(phi_best)      # {'holdout': loss}
```

Начальные состояния тоже могут быть параметрами. Общие компоненты
участвуют во всех экспериментах, эксперимент-специфичные — только в своём
(с приоритетом над общими):

```python
space = ParameterSpace.from_network(network)
space.add_initial_state_entries(network)                 # x0/<pop>/<slot> — общие
space.add_experiment_initial_state_entries(network, "stim")  # x0/stim/<pop>/<slot>

space_c = ParameterSpace.from_network(network)
space_c.add_initial_state_entries(network, include=[("exc", 0)])   # только слот 0
```

`FrozenProblem(problem, free_names=..., fixed=...)` строит задачу над
подпространством φ: свободные координаты доводятся, остальные заморожены
(см. `select_subset` в [Расширенной калибровке](advanced_calibration.md#анализ-идентифицируемости)).

**Типичные ошибки и ограничения:**

- Exp-specific записи `x0/<experiment>/<pop>/<slot>` материализуются
  только внутри `MultiExperimentProblem`: одинокие (orphan) записи и
  использование вне экспериментного контекста дают `UserWarning`.
- Компоненты потерь мультиэкспериментной задачи имеют префикс
  `{name}/` — учитывайте при разборе `ev.components`.
- Записи с `/` в имени эксперимента не допускаются (`ValueError`).
- Полный пример с разложением градиентов по общему и специфичному x0 —
  в `examples/example_10_multiexperiment_x0.py`.

## Полезные ссылки

- [Оптимизация](../optimization.md) — справочник по модулю калибровки;
- [Расширенная калибровка](advanced_calibration.md) —
  идентифицируемость, неопределённость, ограничения на прогнозы,
  параллельная оценка;
- [Примеры](../examples.md) — `example_07_calibration.py` (первая
  калибровка) и сквозной процесс «от данных к интервалам параметров»;
  исходники — в `examples/`;
- [API Reference](../api_reference.md#neuraltideoptimization) — все
  сигнатуры;
- [Решение проблем](../troubleshooting.md) — если калибровка не сходится
  или градиенты подозрительны;
- [Данные](../data.md) — сохранение/загрузка записей HDF5.
