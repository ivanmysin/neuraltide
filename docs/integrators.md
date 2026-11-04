# Интеграторы

NeuralTide предоставляет несколько методов численного интегрирования ОДУ:

- **EulerIntegrator** — метод Эйлера (1-го порядка)
- **HeunIntegrator** — метод Хьюна (2-го порядка)
- **RK4Integrator** — метод Рунге-Кутты 4-го порядка

Все интеграторы являются подклассами `BaseIntegrator`.

> **Порядок на уровне сети.** Формальный порядок относится к изолированной
> популяции при фиксированном синаптическом входе. Шаг сети расщеплён: сначала
> синапсы интегрируются с пресинаптическими частотами начала шага, затем
> популяции — с синаптическим током, посчитанным по новому состоянию синапсов
> и замороженным на весь шаг. Для связанной сети порядок поэтому не выше
> второго: на E–E сети Уилсона–Коуэна с синапсом Цодыкса–Маркрама измерено
> 1.0 для Euler, 2.0 для Heun и 2.0 для RK4 (без связи RK4 даёт 4.0).
> По той же причине `AdaptiveRKIntegrator` удерживает `rtol` только для
> динамики внутри популяции, а `stiffness_report` видит спектр популяции
> при фиксированном входе и не видит жёсткости быстрых синапсов.
> Регрессионный тест `test_rk4_network_order_with_tm_coupling` помечен
> `xfail(strict=True)` до появления интегрирования сети целиком.

---

## BaseIntegrator

**Файл**: `neuraltide/integrators/base.py`

Абстрактный базовый класс интегратора ОДУ.

### Концепция

Интегратор отдельно от популяционной модели — пользователь может менять схему интегрирования не изменяя модель.

### Метод step

```python
step(
    population: PopulationModel,
    state: StateList,
    total_synaptic_input: Dict[str, TensorType],
) -> Tuple[StateList, TensorType]
```

**Аргументы**:
- `population`: экземпляр PopulationModel
- `state`: текущее состояние популяции
- `total_synaptic_input`: `{'I_syn': [1, n], 'g_syn': [1, n]}`

**Возвращает**:
```python
(new_state, local_error_estimate)
```
- `new_state`: новое состояние популяции
- `local_error_estimate`: `tf.Tensor` shape [1] — оценка локальной ошибки

---

## EulerIntegrator

**Файл**: `neuraltide/integrators/euler.py`

Явный метод Эйлера — простейший метод интегрирования.

### Уравнения

```
new_state[i] = state[i] + dt * deriv[i]
local_error = tf.zeros([1])
```

### Достоинства

- Простая реализация
- Высокая скорость вычислений
- Минимальные затраты памяти

### Недостатки

- Низкая точность (1-й порядок)
- Может быть неустойчив при больших dt

### Использование

```python
from neuraltide.integrators import EulerIntegrator

integrator = EulerIntegrator()
network = NetworkRNN(graph, integrator=integrator)
```

---

## HeunIntegrator

**Файл**: `neuraltide/integrators/heun.py`

Метод Хьюна (предсказывающий-корректирующий) — метод 2-го порядка.

### Уравнения

```
k1 = derivatives(state)
k2 = derivatives(state + dt * k1)
new_state[i] = state[i] + dt/2 * (k1[i] + k2[i])

euler_state[i] = state[i] + dt * k1[i]
local_error = mean(||new_state - euler_state||²)
```

### Достоинства

- Более высокая точность (2-й порядок)
- Оценка локальной ошибки
- Лучшая устойчивость чем Euler

### Недостатки

- В 2 раза больше вычислений производных чем Euler
- Оценка ошибки требует дополнительных вычислений

### Использование

```python
from neuraltide.integrators import HeunIntegrator

integrator = HeunIntegrator()
network = NetworkRNN(graph, integrator=integrator)
```

---

## RK4Integrator

**Файл**: `neuraltide/integrators/rk4.py`

Метод Рунге-Кутты 4-го порядка — стандартный метод высокой точности.

### Уравнения

```
k1 = derivatives(state)
k2 = derivatives(state + dt/2 * k1)
k3 = derivatives(state + dt/2 * k2)
k4 = derivatives(state + dt * k3)

new_state[i] = state[i] + dt/6 * (k1+2*k2+2*k3+k4)[i]

heun_state[i] = state[i] + dt/2 * (k1+k2)[i]
local_error = mean(||new_state - heun_state||²)
```

### Достоинства

- Высокая точность (4-й порядок)
- Очень хорошая устойчивость
- Стандартный выбор для большинства задач

### Недостатки

- В 4 раза больше вычислений производных чем Euler
- Более высокие требования к памяти

### Использование

```python
from neuraltide.integrators import RK4Integrator

integrator = RK4Integrator()
network = NetworkRNN(graph, integrator=RK4Integrator())
```

---

## Интеграторы P2: жёсткие системы

**Файлы**: `neuraltide/integrators/{adaptive_rk,imex,rosenbrock,exponential,stiffness}.py`

Этап P2 добавляет методы для жёстких (stiff) дифференциальных систем, где
явные схемы вынуждены брать шаг по жёсткой компоненте. Все они — drop-in
замена `BaseIntegrator` и работают как с прямыми вызовами `step()`, так и
внутри `NetworkRNN` (шаги трассируются в граф `@tf.function`).

### Общая конструкция: substepping внутри шага сети

`NetworkRNN` и его `tf.while_loop` **не изменяются**. Каждый P2-интегратор
внутри одного вызова `step()` делает несколько субшагов по `population.dt`,
синаптический вход при этом заморожен (как и у явных методов). Наружу
возвращается состояние на границе шага сети и накопленный `local_error`
(оценка встроенной пары / индикатор отклонения от явного Эйлера — единая
с RK4 конвенция `mean` по переменным суммы квадратов, используется в
`stability_loss`).

**Синапсы в P2 остаются явными**: `step_synapse` всех P2-интеграторов
делегирует явному Эйлеру.

### AdaptiveRKIntegrator — RK 3(2) Богаки–Шампина с PI-контроллером

Встроенная пара даёт оценку локальной ошибки; PI-контроллер (Gustafsson)
управляет субшагом; неудачные пробы отклоняются. Контур управления
обёрнут в `tf.stop_gradient` и реализован через `tf.while_loop`, поэтому
метод работает и в eager, и под `tf.function`. Статистика последнего шага
(в eager): `last_n_substeps`, `last_mean_substep`, `last_local_error`,
`last_failed`.

```python
from neuraltide.integrators import AdaptiveRKIntegrator

integrator = AdaptiveRKIntegrator(rtol=1e-5, atol=1e-7)
# Фиксированный субшаг (для сходимости по порядку / отладки):
fixed = AdaptiveRKIntegrator(adapt=False, h0=0.1)
```

Замечание: встроенная разность пары BS23 имеет порядок O(h³): на линейных
скалярных задачах оценка ошибки ненулевая и контроллер работает штатно
(только на точном равновесии, где правая часть тождественно нулевая,
пробы дают нулевую ошибку и шаг вырастает до границы сети — это
вырожденный случай, а не общее свойство линейных задач).

### IMEXEulerIntegrator — IMEX-Euler по аддитивному разделению

Нежёсткая часть — явный Эйлер, жёсткая — неявный через линейный оператор
популяции. Популяция переопределяет два аддитивных хука в
`PopulationModel` (по умолчанию оба вырождены — см. «Хуки разделения»):

```python
class MyPopulation(PopulationModel):
    def derivatives_split(self, state, total_synaptic_input):
        f = self.derivatives(state, total_synaptic_input)
        f_stiff = [...]  # жёсткая компонента, та же структура что f
        return ([f[i] - f_stiff[i] for i in range(len(f))], f_stiff)

    def linear_operator(self, state, total_synaptic_input):
        # f_stiff(y) ≈ A @ y_flat + b; y_flat: [n_units, d]
        A = ...  # [n_units, d, d]
        b = ...  # [n_units, d], можно None-эквивалент нуля
        return A, b
```

Честные fallback'и (с warning в eager, вычислительный путь идентичен):
нулевой split без `linear_operator` → тождественно явный Эйлер;
`f_stiff ≠ 0` без оператора → жёсткая часть обрабатывается явно.

### RosenbrockIntegrator — ROS2 (порядок 2, L-устойчивый)

Двухстадийный линейно-неявный метод с якобианом правой части, вычисленным
блочно по юнитам через `ForwardAccumulator` (d JVP на субшаг) и батчевыми
решениями `(I − γhJ)` (`tf.linalg.solve`). Устойчив при шагах, где RK4
расходится. Не требует разделения правой части — только дифференцируемой
`derivatives()`.

```python
from neuraltide.integrators import RosenbrockIntegrator

integrator = RosenbrockIntegrator(n_substeps=4)  # h = dt / n_substeps
```

### ExponentialEulerIntegrator — экспоненциальный Эйлер

Субшаг решается точно для линеаризации `f(y) ≈ f(y0) + J·(y − y0)`:
`y_new = y0 + h·φ1(hJ)·f(y0)` через матричную экспоненту расширенного
блока `[[hJ, h·f], [0, 0]]` (`tf.linalg.expm`, блочно по юнитам
`[n_units, d+1, d+1]`). На линейной динамике популяции — точен до машинного
нуля при любом субшаге. При `d+1 > max_block_size` — документированная
ошибка (`on_large_block='raise'`, по умолчанию) или честный fallback на
явный Эйлер (`on_large_block='euler'`).

### Хуки разделения (core/base.py)

`PopulationModel` получил два аддитивных хука с безопасными дефолтами
(поведение существующих моделей не изменилось):

- `derivatives_split(state, inp) -> (f_nonstiff, f_stiff)` — по умолчанию
  `(derivatives, zeros)`;
- `linear_operator(state, inp) -> (A, b) | None` — по умолчанию `None`.

Контракт: `f_nonstiff + f_stiff == derivatives` покомпонентно.

### stiffness_report — диагностика жёсткости

```python
from neuraltide.integrators import stiffness_report, classify_failure

report = stiffness_report(problem_or_network, phi=None, t_points=8, method='jvp')
# report['populations'][name]['eigenvalues']  — спектр [P, n, d] в точках траектории
# report['stiffness_ratio']                   — max |Re λ|max / |Re λ|min
# report['explicit_dt_limit']                 — min 2|Re λ|/|λ|² (0.0, если есть мода Re λ >= 0)
# report['euler_stable']                      — dt сети <= explicit_dt_limit
```

Прогон разбивается на `t_points` сегментов; в конце каждого для каждой
динамической популяции вычисляется якобиан блока (`method='jvp'` через
`ForwardAccumulator` или `'fd'` — центральные разности) и собственные
значения. Спектр — для блоков популяций при замороженных синаптических
входах (`I_syn = g_syn = 0`); межюнитная синаптическая связность в
якобиан не входит (это соответствует линеаризации самих P2-интеграторов).

`classify_failure(output)` классифицирует результат прогона:
`{'status': 'ok' | 'nan' | 'diverged', 'max_abs': float}`.

### Градиенты

Градиенты параметров доступны через развёртку дифференцируемых шагов
(BPTT/AD, `GradientTape`) — включая адаптивный контур (`tf.while_loop`
с `stop_gradient` в контуре управления шагом). Аналитический
while_loop-adjoint для P2-интеграторов **не поддерживается**: метод
`grad_method='adjoint'` в `Trainer` с ними не используйте. Для жёстких
задач рекомендуется глобальный dtype `float64`
(`neuraltide.config.set_dtype(tf.float64)` до создания моделей).

### Ограничения P2

- substepping — внутри одного `integrator.step()`; временна́я сетка
  `NetworkRNN` не адаптируется;
- синапсы — явные (явный Эйлер на шаге сети, без substepping); связанная
  сеть интегрируется расщеплением с порядком не выше 2 (см. начало раздела);
- ни одна встроенная популяция не переопределяет `derivatives_split` /
  `linear_operator`, поэтому `IMEXEulerIntegrator` для них совпадает с явным
  Эйлером (выдаётся предупреждение);
- `local_error` у IMEX/Rosenbrock/Exponential — индикатор уровня
  отклонения от явного Эйлера (не контролируемая оценка); контроль
  ошибки — только у AdaptiveRK;
- предупреждения о fallback и статистика субшагов доступны в eager-режиме
  (весь API пакета — eager); под `tf.function` вычислительный путь
  идентичен, без предупреждений.

---

## Выбор интегратора

### Рекомендации

| Интегратор | Когда использовать |
|------------|-------------------|
| **Euler** | Отладка, быстрые эксперименты, большие dt |
| **Heun** | Баланс скорости и точности, адаптивные методы |
| **RK4** | Высокая точность, долгосрочные симуляции |

### Сравнение точности

Для типичной популяционной модели:

| dt (мс) | Euler ошибка | Heun ошибка | RK4 ошибка |
|---------|--------------|-------------|------------|
| 0.1 | ~1% | ~0.1% | ~0.001% |
| 0.5 | ~5% | ~0.5% | ~0.01% |
| 1.0 | ~10% | ~2% | ~0.1% |

### Сравнение скорости

Относительное время выполнения (RK4 = 1.0):

| Интегратор | Относительная скорость |
|------------|----------------------|
| Euler | ~0.25x |
| Heun | ~0.5x |
| RK4 | 1.0x |

---

## Пример: влияние интегратора на результат

```python
import numpy as np
import tensorflow as tf
from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.populations import IzhikevichMeanField
from neuraltide.synapses import StaticSynapse
from neuraltide.inputs import ConstantRateGenerator
from neuraltide.integrators import EulerIntegrator, HeunIntegrator, RK4Integrator

# Создание сети
gen = ConstantRateGenerator(dt=0.5, params={'rate': 20.0})
pop = IzhikevichMeanField(dt=0.5, params={
    'tau_pop': [1.0], 'alpha': [0.5], 'a': [0.02],
    'b': [0.2], 'w_jump': [0.1], 'Delta_I': [0.5], 'I_ext': [1.0],
})
syn = StaticSynapse(n_pre=1, n_post=1, dt=0.5, params={
    'gsyn_max': 0.1, 'pconn': [[1.0]], 'e_r': 0.0,
})

graph = NetworkGraph(dt=0.5)
graph.declare_input('input', n_units=gen.n_units)
graph.add_population('exc', pop)
graph.add_synapse('input->exc', syn, src='input', tgt='exc')

# Симуляция с разными интеграторами
t_seq = tf.constant(np.arange(0, 1000, 0.5)[None, :, None])
inputs = graph.pack_inputs({'input': gen(t_seq)})

for name, Integrator in [('Euler', EulerIntegrator), 
                          ('Heun', HeunIntegrator), 
                          ('RK4', RK4Integrator)]:
    network = NetworkRNN(graph, integrator=Integrator())
    output = network(t_seq, inputs=inputs)
    rates = output.firing_rates['exc'].numpy()
    print(f"{name}: final rate = {rates[0, -1, 0]:.2f} Hz")
```

---

## Оценка стабильности

NetworkRNN поддерживает `stability_penalty_weight` для оценки стабильности интегрирования:

```python
network = NetworkRNN(
    graph,
    integrator=RK4Integrator(),
    stability_penalty_weight=1e-3  # штраф за нестабильность
)
```

Оценка основана на разнице между RK4 и Heun решениями на каждом шаге. Большая ошибка указывает на numerical instability.