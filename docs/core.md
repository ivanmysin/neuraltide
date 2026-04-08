# Базовые классы

NeuralTide определяет три основных абстрактных класса, от которых наследуются все модели:

- `PopulationModel` — для популяций
- `SynapseModel` — для синапсов
- `BaseInputGenerator` — для генераторов входов

Все они являются подклассами `tf.keras.layers.Layer`.

---

## PopulationModel

**Файл**: `neuraltide/core/base.py`

Абстрактный базовый класс для популяционной модели нейронов.

### Семантика n_units

`n_units` — число независимых популяций одного типа внутри одного объекта. Например, `IzhikevichMeanField(n_units=4)` описывает 4 независимые популяции с потенциально разными параметрами. Все n_units обрабатываются одной батчированной матричной операцией.

### Контракт подкласса

При реализации своего подкласса необходимо:

1. Установить `self.state_size` в `__init__` до завершения `super().__init__()`
2. Реализовать `get_initial_state()`
3. Реализовать `derivatives()`
4. Реализовать `get_firing_rate()`
5. Опционально переопределить `observables()`
6. Реализовать `parameter_spec` (property)

### Аргументы конструктора

| Параметр | Тип | Описание |
|----------|-----|-----------|
| `n_units` | int | Число независимых популяций |
| `dt` | float | Шаг интегрирования в мс |
| `name` | str | Имя слоя Keras |

### Методы

#### `_make_param(params, name)`

Регистрирует параметр популяции через `add_weight()`.

**Форматы params[name]**:

```python
# Без словаря (trainable=False):
params[name] = 0.5
params[name] = [0.5, 0.6, 0.4, 0.7]

# Словарь с полным контролем:
params[name] = {
    'value': 0.5,           # скаляр или список длины n_units
    'trainable': True,      # по умолчанию False
    'min': 0.01,            # нижняя граница (опционально)
    'max': 2.0,             # верхняя граница (опционально)
}
```

**Правила broadcast**:
- Скаляр → `tf.fill([n_units], scalar)`
- Список длины n_units → используется as-is
- Список другой длины → ValueError

**Возвращает**: `tf.Variable`, зарегистрированная как вес слоя.

**Raises**: `ValueError` если name не найден в params или длина списка не совпадает с n_units.

#### `get_initial_state(batch_size=1)`

Возвращает начальное состояние популяции.

**Возвращает**: `StateList` — список тензоров. Для большинства моделей: `[tf.zeros([1, n_units]), ...]`

#### `derivatives(state, total_synaptic_input)`

Вычисляет производные состояния популяции.

**Аргументы**:
- `state`: текущее состояние — список тензоров
- `total_synaptic_input`: словарь агрегированных синаптических сигналов:
  - `'I_syn'`: tf.Tensor [1, n_units] — суммарный синаптический ток
  - `'g_syn'`: tf.Tensor [1, n_units] — суммарная проводимость

**Возвращает**: `StateList` — производные состояния

#### `get_firing_rate(state)`

Извлекает частоту разрядов из состояния.

**Возвращает**: `tf.Tensor`, shape = [1, n_units], в Гц

#### `observables(state)`

Возвращает словарь наблюдаемых переменных.

**По умолчанию**:
```python
{'firing_rate': self.get_firing_rate(state)}
```

**Опциональные ключи**:
- `'v_mean'` — средний мембранный потенциал, shape [1, n_units]
- `'w_mean'` — среднее адаптационное переменное
- `'lfp_proxy'` — прокси LFP

#### `parameter_spec` (property)

Спецификация параметров модели для summary и сериализации.

**Возвращает**:
```python
{
    'param_name': {
        'shape': tuple,
        'trainable': bool,
        'constraint': str or None,
        'units': str,
    },
    ...
}
```

---

## SynapseModel

**Файл**: `neuraltide/core/base.py`

Абстрактный базовый класс для синаптической модели.

### Семантика

Синапс описывает проекцию от `n_pre` пресинаптических популяций к `n_post` постсинаптическим.

### Матрица параметров

Все весовые матрицы имеют форму `[n_pre, n_post]`. Элемент `[i, j]` описывает связь от популяции `i` к популяции `j`.

### Маскирование через pconn

`pconn` [n_pre, n_post] — матрица вероятностей/масок соединений, обычно `trainable=False`.

### Аргументы конструктора

| Параметр | Тип | Описание |
|----------|-----|-----------|
| `n_pre` | int | Число пресинаптических популяций |
| `n_post` | int | Число постсинаптических популяций |
| `dt` | float | Шаг интегрирования в мс |
| `name` | str | Имя слоя Keras |

### Методы

#### `_make_param(params, name)`

Регистрирует параметр синапса через `add_weight()`.

**Форматы и правила broadcast к [n_pre, n_post]**:

```python
# Скаляр:
{'value': 0.1, 'trainable': True}
→ tf.fill([n_pre, n_post], 0.1)

# Вектор длины n_pre:
{'value': [0.1, 0.2, 0.15, 0.1], 'trainable': True}
→ reshape [n_pre, 1] → broadcast [n_pre, n_post]

# Вектор длины n_post (только если n_pre != n_post):
{'value': [0.1, 0.2, 0.15], 'trainable': True}
→ reshape [1, n_post] → broadcast [n_pre, n_post]

# Матрица [n_pre, n_post]:
→ используется as-is
```

#### `get_initial_state(batch_size=1)`

Возвращает начальное состояние синапса.

**Возвращает**: `StateList`

#### `forward(pre_firing_rate, post_voltage, state, dt)`

Вычисляет новое состояние синапса и возвращает синаптический ток.

**Аргументы**:
- `pre_firing_rate`: частота пресинаптических популяций, shape = [1, n_pre], в Гц
- `post_voltage`: средний потенциал постсинаптических популяций, shape = [1, n_post]
- `state`: текущее внутреннее состояние синапса
- `dt`: шаг интегрирования в мс

**Возвращает**: `Tuple[Dict[str, TensorType], StateList]`
```python
(
    {
        'I_syn': tf.Tensor [1, n_post],
        'g_syn': tf.Tensor [1, n_post],
    },
    new_state  # list of tf.Tensor
)
```

#### `parameter_spec` (property)

Аналогично PopulationModel.

---

## BaseInputGenerator

**Файл**: `neuraltide/core/base.py`

Базовый класс для генератора входных сигналов. Генератор оборачивается в InputPopulation и подключается к динамическим популяциям через полноценные синапсы.

### Семантика n_units

`n_units` — число независимых входных каналов одного типа. Например, `VonMisesGenerator` с `n_units=4` описывает 4 независимых входных сигнала. Все каналы обрабатываются одной векторизованной операцией.

### Особенность: автоматический вывод n_units

В отличие от PopulationModel и SynapseModel, параметр `n_units` **не передаётся явно**. Он автоматически выводится из размерности параметров.

### Аргументы конструктора

| Параметр | Тип | Описание |
|----------|-----|-----------|
| `params` | Dict | Словарь параметров генератора |
| `dt` | float | Шаг интегрирования в мс |
| `name` | str | Имя слоя Keras |

### Методы

#### `_infer_n_units_from_params()`

Определяет n_units из размерности параметров. `n_units` = максимальная длина среди всех параметров.

#### `_validate_param_dimensions()`

Проверяет согласованность размерностей параметров. Все параметры должны иметь длину 1 или n_units.

#### `_make_param(params, name)`

Аналогично PopulationModel, но для генераторов. Broadcast к `n_units`.

#### `call(t)` (abstractmethod)

Вычисляет выход генератора в момент времени t.

**Аргументы**:
- `t`: текущее время в мс. shape = [batch, 1]

**Возвращает**: `tf.Tensor`, shape = [batch, n_units], в Гц

#### `parameter_spec` (property)

Спецификация параметров генератора.

---

## Пример: создание собственного генератора

```python
import tensorflow as tf
from neuraltide.core.base import BaseInputGenerator
from neuraltide.core.types import TensorType

class MyGenerator(BaseInputGenerator):
    def __init__(self, dt: float, params: dict, name: str = "my_generator", **kwargs):
        super().__init__(params=params, dt=dt, name=name, **kwargs)
        
        # Регистрация параметров
        self.amplitude = self._make_param(self._params, 'amplitude')
        self.freq = self._make_param(self._params, 'freq')
    
    def call(self, t: TensorType) -> TensorType:
        # Реализация генератора
        two_pi = tf.constant(2.0 * 3.141592653589793)
        amplitude = tf.reshape(self.amplitude, [1, self.n_units])
        freq = tf.reshape(self.freq, [1, self.n_units])
        
        rate = amplitude * tf.sin(two_pi * freq * t / 1000.0)
        return tf.nn.relu(rate)
    
    @property
    def parameter_spec(self) -> dict:
        return {
            'amplitude': {
                'shape': (self.n_units,),
                'trainable': self.amplitude.trainable,
                'constraint': None,
                'units': 'Hz',
            },
            'freq': {
                'shape': (self.n_units,),
                'trainable': self.freq.trainable,
                'constraint': None,
                'units': 'Hz',
            },
        }
```