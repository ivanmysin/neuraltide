# Расширенная калибровка

Продолжение [Калибровки по данным](calibration.md): что делать после того,
как решение найдено — проверить, какие параметры вообще определимы по
данным, оценить неопределённость, добавить ограничения на прогнозы и
ускорить массовые оценки. Каждый раздел содержит рекомендации «когда
использовать» и «как интерпретировать».

## Анализ идентифицируемости

**Когда использовать.** После первой успешной оптимизации, до того как
сообщать численные значения параметров. Идентифицируемость отвечает на
вопрос: какие параметры (и их комбинации) влияют на прогноз существенно
сильнее уровня шума, а какие данные почти не чувствуют. Плохо
идентифицируемые параметры нельзя интерпретировать, даже если оптимизатор
сообщил «сходимость».

**Минимальный пример.** Точка входа — якобиан невязок по φ
(forward-чувствительности), дальше матрица информации Фишера F = JᵀJ.
Forward-режим несовместим с ранее выполненными gradient-прогонами на
том же экземпляре `NetworkRNN`, поэтому считаем на свежем экземпляре
задачи из той же фабрики (см. ограничение в разделе про
неопределённость):

```python
from neuraltide.optimization import compute_fim, residual_jacobian

problem_id = build_problem()                  # свежий экземпляр той же задачи
jac = residual_jacobian(problem_id, phi_best)  # J: [n_residuals, size]
fim = compute_fim(jac, param_names=space.trainable_names)

print(fim.effective_rank, "/", problem_id.size)  # сколько направлений реально видно
print(fim.condition_number)                     # kappa(J) по ненулевым сингулярным
for pair in fim.collinearity:                   # почти линейно зависимые пары
    print(pair.first, "~", pair.second, pair.cosine)
for direction in fim.soft_dirs[:3]:             # «мягкие» направления
    print(direction.top_params)
```

Как интерпретировать:

- `effective_rank` заметно меньше размерности φ — данные определяют
  только часть параметров; остальные произвольны в пределах «мягких»
  направлений.
- `condition_number` большой (≳1e6–1e8) — задача плохо обусловлена:
  малые изменения данных сильно двигают решение.
- `collinearity` — пары параметров, входящие в прогноз почти
  одинаково (косинус ≈ 1): их нельзя оценивать по отдельности, только
  комбинацию.
- `stiff_dirs`/`soft_dirs` с разложением `top_params` — конкретные
  комбинации параметров, которые данные видят хорошо и плохо.

Профиль правдоподобия — одномерная проверка по конкретному параметру:
для сетки значений θ параметр фиксируется, остальные доводятся, и
строится ΔNLL(θ):

```python
from neuraltide.optimization import likelihood_profile

profile = likelihood_profile(
    problem_id, "exc/tau", grid=np.linspace(4.0, 14.0, 11),
    optimizer="lbfgsb", warm_start=True,
)
print(profile.intervals["0.95"])   # chi2-интервал 95% (порог 1.921 по rel_loss)
```

Интервал честный: если он широкий или граничный (уходит в min/max),
параметр по данным не определён. Узлы, где доводка не сошлась, видны в
`profile.successes`/`profile.warnings` (`losses` там NaN).

`select_subset` отвечает на практический вопрос «а что если заморозить
всё, кроме главных?»:

```python
from neuraltide.optimization import select_subset

sub = select_subset(problem_id, jac, keep=["exc/tau", "in->exc/gsyn_max"])
print(sub.loss_after - sub.loss_before)     # цена фиксации остальных
print(sub.prediction_delta)                  # ||Δr||/||r|| по популяциям
print(sub.validation_delta)                  # дельта потерь на held-out
```

`prediction_delta` мало — подмножество достаточно; `theta_delta_kept`
показывает смещение оценок при доводке в меньшем пространстве.

**Типичные ошибки и ограничения:**

- Вся статистика здесь — в единицах полного NLL; модули сами переводят
  невязки из нормировки потери (это не требует действий пользователя, но
  объясняет, почему числа отличаются от `ev.loss` при
  `normalization="mean"`).
- `likelihood_profile`/`select_subset` принимают только обучаемые имена
  из `ParameterSpace` — имя константы (`trainable: False`) отклоняется.
- Forward-якобиан несовместим с ранее выполненными gradient-прогонами на
  том же экземпляре `NetworkRNN` (см. ограничение в разделе про
  неопределённость ниже).

## Валидация решения

**Когда использовать.** Всегда: перед отчётом результата и перед
переходом к неопределённости. Подгонка и проверка — на разных протоколах.

**Минимальный пример.** Отложенные протоколы задаются при постановке
задачи и оцениваются `validate`:

```python
from neuraltide.optimization import CalibrationProblem, Experiment

holdout = Experiment(name="holdout_f16", t_sequence=t_seq,
                     inputs=inputs_holdout, objective=objective_holdout)
problem = CalibrationProblem(network, space, t_seq, inputs=inputs_train,
                             objective=objective_train, held_out=[holdout])

local = minimize_lbfgsb(problem)
print(local.validation)                     # {'holdout_f16': loss}
print(problem.validate(phi_best))
```

Если в обучении использовался teacher forcing — финальный честный
прогон выполняется на исходных входах без смешивания: либо
`free_run_check` у `ContinuationScheduler`, либо явная оценка
`problem.evaluate(phi_best)` после завершения расписания.

Как интерпретировать:

- Потеря на held-out заметно выше обучающей — переобучение или
  неверная постановка (пропущенная динамика, неверный `sigma`);
- Провал локально, на участке записи — смотрите оконную диагностику
  остатков (`residual_diagnostics(..., windows=...)` ниже) и
  `adaptive_window_sampler` для фокусировки доводки.

**Типичные ошибки и ограничения:**

- Held-out протокол должен отличаться по динамике (другой вход, другой
  режим), иначе проверка ничего не измеряет.
- `validate` — это прямые прогоны: не делайте отложенный набор огромным.

## Оценка неопределённости

**Когда использовать какой инструмент.**

| Инструмент | Что даёт | Когда использовать |
|------------|----------|--------------------|
| `residual_diagnostics` | Свойства остатков в точке решения | Первая проверка адекватности: шум vs систематика |
| `laplace_approximation` | Ковариация posterior в MAP, интервалы параметров | Гладкая задача, умеренная размерность φ, есть якобиан |
| `residual_bootstrap` | Распределение φ при сомнении в гауссовости | Остатки приблизительно iid и адекватны |
| `bootstrap_experiments` | Распределение φ по повторным подвыборкам | Есть несколько экспериментов |
| `parameter_ensemble` | Ансамбль из готовых решений | Несколько стартов/методов уже посчитаны |
| `predictive_intervals` | Квантили firing rates по ансамблю φ | Итоговая визуализация неопределённости прогноза |
| `noise_sensitivity` | Интервалы при масштабировании шума | Проверка чувствительности выводов к `sigma` |

**Минимальный пример.** Начните с остатков — дальнейшая статистика
опирается на их адекватность:

```python
from neuraltide.optimization import residual_diagnostics

diag = residual_diagnostics(problem, phi_best, max_lag=10,
                            windows=[(0, 400), (400, 800)])   # окна в индексах шагов [0, T)
for spec in diag.specs:
    print(spec.name, spec.std, spec.max_abs, spec.autocorr)
```

Интерпретация: `std` остатков сопоставим с `sigma` наблюдений —
масштаб шума задан верно; высокая автокорреляция — в остатках осталась
динамика, которую модель не воспроизводит (систематическая ошибка), и
гауссова статистика ниже формально некорректна; окна
(`WindowDiagnostics`) локализуют проблемные участки записи.

Лаплас-аппроксимация posterior в MAP-точке:

```python
from neuraltide.optimization import laplace_approximation

lap = laplace_approximation(problem, phi_map, method="gn", level=0.95)
print(lap.pd, lap.jitter_used)          # обратилась ли precision без jitter
for name in space.trainable_names:
    print(name, lap.intervals[name])    # интервалы в физических theta
```

`method="gn"` строит ковариацию через якобиан невязок (нужен
`problem.size <= max_params`, по умолчанию 200); `"fd-hessian"` — через
конечные разности Гессиана (`max_fd_params`, медленнее). Интервалы
физических параметров — в `lap.intervals`, в координатах φ — в
`lap.phi_intervals`.

Bootstrap — альтернатива без гауссова предположения:

```python
from neuraltide.optimization import (
    bootstrap_experiments, parameter_ensemble, predictive_intervals,
    residual_bootstrap,
)

boot = residual_bootstrap(problem, n_draws=100, seed=7)      # остаточный
boot_exp = bootstrap_experiments(problem, n_draws=100, seed=7)  # по экспериментам

ensemble = parameter_ensemble([local.base, ms.base])         # из готовых решений
pred = predictive_intervals(problem, ensemble.phi_samples, level=0.95)
print(pred.lower["exc"].shape)   # [T, n_units] квантили прогноза
```

**Важное ограничение.** Forward-режим чувствительностей (внутри
`residual_jacobian`, а значит и `laplace_approximation`) несовместим с
уже выполненными gradient-прогонами на том же экземпляре `NetworkRNN`:
TensorFlow кэширует граф прямого прогона, и forward-AD после GradientTape
падает. Поэтому блоки UQ запускайте на свежем экземпляре задачи из той
же picklable-фабрики:

```python
problem_uq = build_problem()                 # та же фабрика, что и для parallel
lap = laplace_approximation(problem_uq, phi_map, method="gn")
```

Как интерпретировать:

- `lap.pd == False` или большой `jitter_used` — posterior-ковариация
  вырождена: см. `effective_rank` FIM; интервалы формально есть, но их
  ширина не имеет смысла.
- Интервал параметра шире практической значимости — параметр не
  определён данными; не интерпретируйте точечную оценку.
- Покрытие прогнозных интервалов: доля точек цели внутри
  `[lower, upper]` при верной модели шума близка к `level`.

**Типичные ошибки и ограничения:**

- Exact-fit (потеря ровно 0) — вырожденный случай: sqrt-нормировка
  невязок недифференцируема в нуле, forward-чувствительности дают NaN.
  Калибруйте по зашумлённым данным или добавьте реалистичный `sigma`.
- `residual_bootstrap` ресемплирует iid и игнорирует автокорреляцию —
  при систематических остатках интервалы слишком узкие.
- `bootstrap_experiments` с эксперимент-специфичными записями x0 в
  пространстве отклоняется `ValueError`.
- Статистика (FIM/Лаплас/профили/bootstrap) — в единицах полного NLL;
  потеря `ev.loss` при `normalization="mean"` отличается масштабом —
  это не ошибка.

## Робастные и сбалансированные потери

**Когда использовать.** Когда целевая функция состоит из нескольких
конфликтующих компонент (несколько популяций, несколько экспериментов)
и градиенты тянут параметры в разные стороны. Базовые рычаги — веса
компонент (`ObservationSpec.weight`), далее адаптивные стратегии и
проекции конфликтующих градиентов.

**Минимальный пример.**

```python
from neuraltide.optimization import (
    GradNormWeights, balanced_gradient, balanced_train, pcgrad_project,
)

# Одна итерация: взвешенный градиент с проекцией PCGrad
components, grad = balanced_gradient(
    problem, phi, weights={"exc": 1.0, "inh": 2.0}, projector=pcgrad_project,
)

# Полный цикл с адаптивными весами (с обязательным baseline сравнения):
bt = balanced_train(
    problem, phi0, GradNormWeights(alpha=1.0),
    optimizer="adam", max_steps=200, projector=pcgrad_project, seed=7,
)
print(bt.base.loss_best, bt.baseline["best_loss"])   # vs FixedWeights(1.0)
```

Стратегии весов (`WeightStrategy`): `FixedWeights` (константные веса —
и baseline для сравнения), `UncertaintyWeighting` (вес ~ 1/потеря),
`SelfAdaptiveWeights` (степенная адаптация), `GradNormWeights` (веса из
норм градиентов). Проекторы: `pcgrad_project` (PCGrad: конфликтующие
компоненты проецируются), `cagrad_project` (CAGrad с параметром `c`),
либо `None` (только взвешивание).

`adaptive_window_sampler` — отдельный инструмент для записей, где
ошибка сосредоточена в коротком окне:

```python
from neuraltide.optimization import adaptive_window_sampler

focus = adaptive_window_sampler(problem, window_size=50)   # поиск окна
focused_view = focus(phi)          # view задачи с усиленной маской в окне
```

Как интерпретировать: `balanced_train` всегда прогоняет baseline
`FixedWeights(1.0)` из той же стартовой точки — улучшение есть, только
если `loss_best` лучше baseline; `weight_history` показывает динамику
весов (рост веса компонента = компонент систематически хуже).

**Типичные ошибки и ограничения:**

- Балансировка — детерминированные суррогаты (пересчёт по наблюдённым
  потерям/градиентам), а не совместное обучение; только BPTT-режим.
- Проекторы не делают несовместные цели совместными — они выбирают
  компромисс; если baseline сопоставим, используйте простые веса.
- `balanced_train` мутирует модель по общему контракту — верните
  лучшую точку явной оценкой, если дальше работаете с MAP.

## Ограничения на прогнозы

**Когда использовать.** Когда помимо потери есть требования к самому
прогнозу: например, средняя частота популяции в заданном диапазоне,
запрет отрицательных токов, фазовые соотношения. Это ограничения на
выход сети — в отличие от ограничений на параметры (`min`/`max` в
params и `MinMaxConstraint`).

**Минимальный пример.** Ограничение задаётся `ConstraintSpec`
(`kind='eq'` для c(x)=0, `'ineq'` для g(x)≤0), далее — метод
внешних штрафов с множителями Лагранжа:

```python
from neuraltide.optimization import (
    ConstraintSpec, augmented_lagrangian,
)

def mean_rate_bound(predictions, network):
    return tf.reduce_mean(predictions.firing_rates["inh"]) - 15.0  # <= 0

constraints = [ConstraintSpec(name="inh_rate", kind="ineq", fn=mean_rate_bound)]
al = augmented_lagrangian(problem, constraints, local="lbfgsb", max_outer=10)
print(al.base.loss_best)            # потеря ДАННЫХ в phi_best (без AL-членов)
for row in al.violation_history:
    print(row)
```

Метод увеличивает `mu` внешними итерациями и корректирует множители;
`ALResult.state` (`AugmentedLagrangianState`: `lambdas`, `mu`) пригоден
для checkpoint/resume. `ConstrainedProblem` — та же задача с AL-членом
одной лентой (принимается локальными оптимизаторами напрямую), если
нужен ручной контроль над циклом внешних итераций.

Эпсилон-ограничение — второй способ: один из компонентов цели становится
ограничением, остальные — взвешенной суммой:

```python
from neuraltide.optimization import epsilon_constraint

constrained = epsilon_constraint(
    problem,
    objectives=["exc", "inh"],   # имена компонентов objective.components
    index=1,                      # 'inh' <= eps
    eps=1.0,
)
```

Якобиан ограничений: `ConstraintSpec.jacobian` — `None | 'fd' | callable`;
`fd_jacobian(value_fn, phi, ...)` строит конечно-разностный якобиан для
методов, которым он нужен.

**Типичные ошибки и ограничения:**

- Ограничения на прогнозы действуют на прямой прогон — они удорожают
  каждую оценку; формулируйте только нужные.
- `epsilon_constraint`: выбранный компонент выбывает из целевой функции;
  если таких компонентов все — передайте явный `objective=...`, иначе
  `ValueError`.
- Эпсилон-ограничение ищет допустимую область по компоненту — слишком
  жёсткий `eps` может сделать задачу неразрешимой; следите за
  `violation_history`.

## Пакетная оценка кандидатов

**Когда использовать.** Массовые оценки одинаковой задачи в разных
точках φ: дизайны стартов, глобальный поиск, bootstrap с внешним
evaluator, сравнение решений из мультистарта. Пул процессов окупается
только на дорогих прогонах — для дешёвых используйте `SerialPool`.

**Минимальный пример.**

```python
from neuraltide.optimization import (
    CandidateTask, ProcessPool, SerialPool, evaluate_candidates, make_evaluator,
)

# Один вызов — список записей (loss=+inf и error при исключении в worker'е):
records = evaluate_candidates(build_problem, candidates, n_workers=4, seed=0)

# Явный пул для повторного использования:
pool = ProcessPool(build_problem, fn="evaluate", n_workers=4, tf_threads=1)
records = pool.map([CandidateTask(index=i, phi=phi) for i, phi in enumerate(candidates)])
pool.close()

# Инъекция в глобальный поиск:
evaluator = make_evaluator(build_problem, n_workers=4, seed=0)
gs = global_search(problem, method="de", max_evals=500, evaluator=evaluator)
evaluator.close()
```

`build_problem` — picklable-фабрика уровня модуля, строящая свежую задачу
(см. ограничение про forward-AD в разделе неопределённости: фабрика как
раз решает и его). Один seed даёт одинаковые потери в обоих пулах —
удобно для сверки с ручным циклом `problem.evaluate`.

**Типичные ошибки и ограничения:**

- Worker-процессы ограничены `tf_threads=1` (защита от oversubscription);
  `None` не трогает потоки TF.
- `evaluator` несовместим с `method="nsga2"` (мультицелевой метод
  управляет оценками сам).
- Пул процессов имеет смысл, когда прогон дорогой; иначе накладные
  расходы на spawn съедают выигрыш.

## Сводка: когда что использовать

| Задача | Инструмент |
|--------|-----------|
| Проверить, какие параметры определимы | `residual_jacobian` + `compute_fim` |
| Интервал по одному параметру | `likelihood_profile` |
| Оценить цену фиксации части параметров | `select_subset` |
| Проверка адекватности остатков | `residual_diagnostics` |
| Интервалы параметров (гауссова approx) | `laplace_approximation` на свежем экземпляре задачи |
| Негауссово распределение решений | `residual_bootstrap` / `bootstrap_experiments` |
| Неопределённость прогноза | `parameter_ensemble` + `predictive_intervals` |
| Чувствительность выводов к `sigma` | `noise_sensitivity` |
| Конфликт компонент цели | `balanced_train` (+ `pcgrad_project`) / адаптивные веса |
| Ошибка сосредоточена во временном окне | `adaptive_window_sampler` |
| Требования к прогнозу | `augmented_lagrangian` / `epsilon_constraint` |
| Массовые оценки кандидатов | `evaluate_candidates` / `make_evaluator` |

## Полезные ссылки

- [Калибровка по данным](calibration.md) — постановка задачи и
  оптимизация;
- [Оптимизация](../optimization.md) — справочник модуля калибровки;
- [Примеры](../examples.md) — сквозные сценарии: идентифицируемость,
  профили правдоподобия и валидация, глобальный поиск, балансировка,
  Лаплас-аппроксимация и пакетная оценка; исходники — в `examples/`;
- [API Reference](../api_reference.md#neuraltideoptimization) —
  сигнатуры всех функций и DTO результатов;
- [Решение проблем](../troubleshooting.md) — интерпретация странных
  интервалов и провалов валидации.
