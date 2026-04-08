# Интеграторы

NeuralTide предоставляет несколько методов численного интегрирования ОДУ:

- **EulerIntegrator** — метод Эйлера (1-го порядка)
- **HeunIntegrator** — метод Хьюна (2-го порядка)
- **RK4Integrator** — метод Рунге-Кутты 4-го порядка

Все интеграторы являются подклассами `BaseIntegrator`.

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
graph.add_input_population('input', gen)
graph.add_population('exc', pop)
graph.add_synapse('input->exc', syn, src='input', tgt='exc')

# Симуляция с разными интеграторами
t_seq = tf.constant(np.arange(0, 1000, 0.5)[None, :, None])

for name, Integrator in [('Euler', EulerIntegrator), 
                          ('Heun', HeunIntegrator), 
                          ('RK4', RK4Integrator)]:
    network = NetworkRNN(graph, integrator=Integrator())
    output = network(t_seq)
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