# Сеть (NetworkGraph и NetworkRNN)

NeuralTide предоставляет два основных класса для построения и симуляции нейронных сетей:

- **NetworkGraph** — описание топологии сети
- **NetworkRNN** — симуляция сети на временной оси

---

## NetworkGraph

**Файл**: `neuraltide/core/network.py`

Класс для описания топологии сети: популяции и синаптические проекции.

### Концепция

NetworkGraph хранит:
- Все популяции (динамические и входные)
- Все синаптические проекции
- Проверку корректности топологии

### Конструктор

```python
NetworkGraph(dt: float)
```

| Параметр | Тип | Описание |
|----------|-----|-----------|
| `dt` | float | Шаг интегрирования в мс |

### Методы

#### `add_population(name, model)`

Регистрирует популяцию (динамическую или InputPopulation).

```python
graph.add_population('exc', pop)
```

**Raises**: `ValueError` если имя уже занято.

#### `add_input_population(name, generator)`

Регистрирует входной генератор как псевдо-популяцию InputPopulation.

```python
graph.add_input_population('theta_input', gen)
```

Это эквивалентно:
```python
input_pop = InputPopulation(generator=generator, name=name + '_input_pop')
graph.add_population(name, input_pop)
```

#### `add_synapse(name, model, src, tgt)`

Регистрирует синаптическую проекцию.

```python
graph.add_synapse('input_to_exc', syn, src='theta_input', tgt='exc')
```

| Параметр | Тип | Описание |
|----------|-----|-----------|
| `name` | str | Уникальное имя синапса |
| `model` | SynapseModel | Экземпляр синапса |
| `src` | str | Имя исходной популяции |
| `tgt` | str | Имя целевой популяции |

**Raises**:
- `ValueError` если имя уже занято
- `ValueError` если src или tgt не зарегистрированы
- `ValueError` если tgt — InputPopulation (входные популяции не могут быть целями)

#### `validate()`

Проверяет корректность топологии перед построением NetworkRNN.

Проверки:
1. `n_pre` каждого синапса == `n_units` исходной популяции
2. `n_post` каждого синапса == `n_units` целевой популяции
3. Все динамические популяции имеют хотя бы один входящий синапс (警告)

```python
graph.validate()  # Raises ValueError при ошибке
```

### Свойства

#### `population_names`

Список всех имён популяций.

```python
graph.population_names  # ['theta_input', 'exc', 'inh']
```

#### `synapse_names`

Список всех имён синапсов.

```python
graph.synapse_names  # ['input->exc', 'exc->inh', 'inh->exc']
```

#### `dynamic_population_names`

Список имён только динамических популяций (без InputPopulation).

```python
graph.dynamic_population_names  # ['exc', 'inh']
```

#### `input_population_names`

Список имён только входных популяций.

```python
graph.input_population_names  # ['theta_input']
```

### Пример полного графа

```python
import numpy as np
import tensorflow as tf
from neuraltide.core.network import NetworkGraph
from neuraltide.populations import IzhikevichMeanField
from neuraltide.synapses import TsodyksMarkramSynapse, StaticSynapse
from neuraltide.inputs import VonMisesGenerator
from neuraltide.integrators import RK4Integrator

dt = 0.05

# Генератор тета-ритма
gen = VonMisesGenerator(
    dt=dt,
    params={
        'mean_rate': 20.0,
        'R': 0.5,
        'freq': 8.0,
        'phase': 0.0,
    },
    name='theta_gen'
)

# Возбуждающая популяция
pop_exc = IzhikevichMeanField(dt=dt, params={
    'tau_pop':   {'value': [1.0, 1.0],   'trainable': False},
    'alpha':     {'value': [0.5, 0.5],   'trainable': False},
    'a':         {'value': [0.02, 0.02], 'trainable': False},
    'b':         {'value': [0.2, 0.2],   'trainable': False},
    'w_jump':    {'value': [0.1, 0.1],   'trainable': False},
    'Delta_I':   {'value': [0.5, 0.6],   'trainable': True,
                  'min': 0.01, 'max': 2.0},
    'I_ext':     {'value': [0.1, 0.2],   'trainable': True,
                  'min': -2.0, 'max': 2.0},
})

# Входной синапс (Tsodyks-Markram)
syn_in = TsodyksMarkramSynapse(n_pre=1, n_post=2, dt=dt, params={
    'gsyn_max': {'value': [[0.1, 0.1]], 'trainable': True},
    'tau_f':    {'value': 20.0,  'trainable': True, 'min': 6.0,  'max': 240.0},
    'tau_d':    {'value': 5.0,   'trainable': True, 'min': 2.0,  'max': 15.0},
    'tau_r':    {'value': 200.0, 'trainable': True, 'min': 91.0, 'max': 1300.0},
    'Uinc':     {'value': 0.2,   'trainable': True, 'min': 0.04, 'max': 0.7},
    'pconn':    {'value': [[1.0, 1.0]], 'trainable': False},
    'e_r':      {'value': 0.0,   'trainable': False},
})

# Рекуррентный синапс
syn_rec = TsodyksMarkramSynapse(n_pre=2, n_post=2, dt=dt, params={
    'gsyn_max': {'value': 0.05,  'trainable': True},
    'tau_f':    {'value': 20.0,  'trainable': True, 'min': 6.0,  'max': 240.0},
    'tau_d':    {'value': 5.0,   'trainable': True, 'min': 2.0,  'max': 15.0},
    'tau_r':    {'value': 200.0, 'trainable': True, 'min': 91.0, 'max': 1300.0},
    'Uinc':     {'value': 0.2,   'trainable': True, 'min': 0.04, 'max': 0.7},
    'pconn':    {'value': [[1, 1], [1, 1]], 'trainable': False},
    'e_r':      {'value': 0.0,   'trainable': False},
})

# Построение графа
graph = NetworkGraph(dt=dt)
graph.add_input_population('theta', gen)
graph.add_population('exc', pop_exc)
graph.add_synapse('theta->exc', syn_in,  src='theta', tgt='exc')
graph.add_synapse('exc->exc',   syn_rec, src='exc',   tgt='exc')

# Валидация
graph.validate()
```

---

## NetworkRNN

**Файл**: `neuraltide/core/network.py`

Слой Keras для симуляции сети на временно́й оси.

### Концепция

NetworkRNN использует `tf.scan` для пошагового интегрирования сети. На каждом временном шаге:
1. Вычисляются синаптические токи от всех проекций
2. Обновляются состояния динамических популяций через интегратор
3. Обновляются состояния синапсов

### Конструктор

```python
NetworkRNN(
    graph: NetworkGraph,
    integrator: BaseIntegrator,
    return_sequences: bool = True,
    return_hidden_states: bool = False,
    stability_penalty_weight: float = 0.0,
    stateful: bool = False,
    **kwargs
)
```

| Параметр | Тип | Описание | По умолчанию |
|----------|-----|---------|--------------|
| `graph` | NetworkGraph | Граф сети | — |
| `integrator` | BaseIntegrator | Интегратор | — |
| `return_sequences` | bool | Возвращать последовательности | True |
| `return_hidden_states` | bool | Возвращать скрытые состояния | False |
| `stability_penalty_weight` | float | Вес штрафа за стабильность | 0.0 |
| `stateful` | bool | Сохранять состояние между батчами | False |

### Вызов сети

```python
output = network(t_sequence, initial_state=None, training=False)
```

**Аргументы**:
- `t_sequence`: tf.Tensor shape `[batch, T, 1]` или `[batch, T]` — временная последовательность в мс
- `initial_state`: Optional[Tuple[pop_states, syn_states]] — начальное состояние. Если None, используется нулевое.
- `training`: bool — режим (влияет на dropout и т.д.)
- `reset_state`: bool — сбросить состояние перед прогоном (полезно для не-stateful режима)

**Возвращает**: `NetworkOutput`

### NetworkOutput

```python
@dataclass
class NetworkOutput:
    firing_rates: Dict[str, TensorType]       # {pop_name: [batch, T, n_units]}
    hidden_states: Optional[Dict[str, Dict[str, TensorType]]]  # или None
    stability_loss: TensorType  # скаляр
```

### Свойства

#### `trainable_variables`

Агрегирует trainable_variables из всех популяций и синапсов графа.

```python
network.trainable_variables
# [<tf.Variable ...>, ...]
```

### Методы

#### `get_initial_state(batch_size=1)`

Возвращает начальное состояние сети.

```python
init_pop, init_syn = network.get_initial_state(batch_size=2)
```

**Возвращает**: `Tuple[StateList, StateList]`

#### `set_initial_state(state)`

Устанавливает начальное состояние сети.

```python
init_pop, init_syn = network.get_initial_state()
init_pop[0] = tf.constant([[0.5, 0.5]])  # r = 0.5
network.set_initial_state((init_pop, init_syn))
```

**Args**:
- `state`: Tuple[pop_states, syn_states]

#### `get_state()`

Возвращает текущее сохранённое состояние (после прогона в stateful режиме).

```python
output = network(t_seq)
current_state = network.get_state()
```

**Возвращает**: `Tuple[StateList, StateList]` или `None`

#### `reset_state()`

Сбрасывает состояние в нулевое.

```python
network.reset_state()
```

---

## Симуляция

### Простейший пример

```python
import numpy as np
import tensorflow as tf

# ... создание графа ...

network = NetworkRNN(graph, integrator=RK4Integrator())

# Временная последовательность
T = 1000  # мс
dt = 0.5
t_values = np.arange(0, T, dt, dtype=np.float32)
t_seq = tf.constant(t_values[None, :, None])  # [1, T/dt, 1]

# Запуск
output = network(t_seq)

# Результат
print(output.firing_rates['exc'].shape)  # [1, 2000, 2]
rates = output.firing_rates['exc'].numpy()
```

### Симуляция с пользовательским начальным состоянием

```python
# Получение начального состояния
init_pop, init_syn = network.get_initial_state()

# Модификация состояния
init_pop[0] = tf.constant([[0.5, 0.5]])  # r = 0.5
init_pop[1] = tf.constant([[0.1, 0.1]])  # v = 0.1

# Запуск с пользовательским состоянием
output = network(t_seq, initial_state=(init_pop, init_syn))
```

### Мониторинг стабильности

```python
network = NetworkRNN(
    graph,
    integrator=RK4Integrator(),
    stability_penalty_weight=1e-3
)

output = network(t_seq)

# Штраф за стабильность
print(output.stability_loss)  # скаляр
```

### Доступ к скрытым состояниям

```python
network = NetworkRNN(
    graph,
    integrator=RK4Integrator(),
    return_hidden_states=True
)

output = network(t_seq)

# hidden_states содержит состояния всех популяций и синапсов
# Формат зависит от конкретной модели
```

---

## Внутренняя структура (_step_fn)

Функция `_step_fn` выполняет один шаг симуляции:

```python
def _step_fn(states, t, graph, integrator):
    # 1. Распаковка состояний
    pop_states_dict = {...}
    syn_states_dict = {...}
    
    # 2. Обновление времени для InputPopulation
    for name in input_population_names:
        pop_states_dict[name] = [t]
    
    # 3. Вычисление синаптических токов
    for syn_name, entry in synapses:
        pre_rate = src_pop.get_firing_rate(src_state)
        post_v = tgt_pop.observables(tgt_state)['v_mean']
        current_dict, new_syn_state = synapse.forward(...)
        
        syn_I[tgt] += current_dict['I_syn']
        syn_g[tgt] += current_dict['g_syn']
    
    # 4. Интегрирование популяций
    for name in dynamic_population_names:
        total_syn = {'I_syn': syn_I[name], 'g_syn': syn_g[name]}
        new_pop_state, local_err = integrator.step(pop, pop_state, total_syn)
    
    return (new_pop_states, new_syn_states, stability_error)
```

Эта функция вызывается `tf.scan` для всех временных шагов.

---

## Ограничения

### InputPopulation не может быть целью синапса

```python
# Ошибка!
graph.add_synapse('exc->input', syn, src='exc', tgt='input')
# ValueError: InputPopulation cannot be a synaptic target
```

### Все динамические популяции должны иметь вход

```python
# Предупреждение (не ошибка)
graph.validate()
# UserWarning: Population 'exc' has no incoming synapses.
```

### Согласование n_units

```python
# Ошибка: n_pre != n_units источника
pop = IzhikevichMeanField(dt=0.5, params={...})  # n_units=4
syn = StaticSynapse(n_pre=2, n_post=4, ...)  # n_pre=2 != 4
graph.add_synapse('syn', syn, src='pop', tgt='pop2')
# ValueError: n_pre=2 != src 'pop' n_units=4
```