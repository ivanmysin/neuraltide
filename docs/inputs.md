# Генераторы входов

NeuralTide предоставляет несколько типов генераторов входных сигналов:

- **SinusoidalGenerator** — синусоидальный генератор
- **ConstantRateGenerator** — генератор постоянной частоты
- **VonMisesGenerator** — генератор на основе распределения фон Мизаса

Все генераторы являются подклассами `BaseInputGenerator` и поддерживают векторизацию.

---

## Общие принципы

### Векторизация

Генераторы автоматически выводят `n_units` из размерности параметров:

```python
# n_units = 1 (скалярные параметры)
gen = SinusoidalGenerator(dt=0.5, params={
    'amplitude': 10.0, 'freq': 8.0, 'phase': 0.0, 'offset': 5.0,
})

# n_units = 2 (векторные параметры)
gen = SinusoidalGenerator(dt=0.5, params={
    'amplitude': [10.0, 15.0], 'freq': [8.0, 10.0],
    'phase': [0.0, 1.5], 'offset': [5.0, 7.0],
})
```

### Вызов генератора

```python
# t: shape [batch, 1] — текущее время в мс
output = generator(t)  # Returns: shape [batch, n_units] в Гц
```

### Интеграция с NetworkGraph

Генераторы оборачиваются в InputPopulation:

```python
graph = NetworkGraph(dt=0.5)
graph.add_input_population('my_input', generator)
```

---

## SinusoidalGenerator

**Файл**: `neuraltide/inputs/sinusoidal.py`

Синусоидальный генератор.

### Уравнение

```
rate(t) = max(0, amplitude * sin(2π*freq*t/1000 + phase) + offset)
```

### Конструктор

```python
SinusoidalGenerator(
    dt: float,
    params: Dict[str, Any],
    name: str = "sinusoidal_generator",
    **kwargs
)
```

### Параметры (params)

| Параметр | Тип | Описание | Единицы |
|----------|-----|-----------|---------|
| `amplitude` | float/list | Амплитуда | Гц |
| `freq` | float/list | Частота | Гц |
| `phase` | float/list | Начальная фаза | рад |
| `offset` | float/list | Смещение | Гц |

### Примеры

**Один канал (n_units=1)**:
```python
gen = SinusoidalGenerator(
    dt=0.5,
    params={
        'amplitude': 10.0,
        'freq': 8.0,
        'phase': 0.0,
        'offset': 5.0,
    }
)
```

**Несколько каналов (n_units=3)**:
```python
gen = SinusoidalGenerator(
    dt=0.5,
    params={
        'amplitude': [10.0, 15.0, 8.0],
        'freq': [8.0, 10.0, 6.0],
        'phase': [0.0, 1.5, 3.0],
        'offset': [5.0, 7.0, 4.0],
    }
)
```

**Обучаемые параметры**:
```python
gen = SinusoidalGenerator(
    dt=0.5,
    params={
        'amplitude': {'value': 10.0, 'trainable': True, 'min': 0.0, 'max': 50.0},
        'freq': {'value': 8.0, 'trainable': True, 'min': 1.0, 'max': 100.0},
        'phase': {'value': 0.0, 'trainable': False},
        'offset': 5.0,
    }
)
```

### ParameterSpec

```python
{
    'amplitude': {'shape': (n_units,), 'trainable': ..., 'constraint': ..., 'units': 'Hz'},
    'freq':      {'shape': (n_units,), 'trainable': ..., 'constraint': ..., 'units': 'Hz'},
    'phase':     {'shape': (n_units,), 'trainable': ..., 'constraint': ..., 'units': 'rad'},
    'offset':    {'shape': (n_units,), 'trainable': ..., 'constraint': ..., 'units': 'Hz'},
}
```

---

## ConstantRateGenerator

**Файл**: `neuraltide/inputs/constant.py`

Генератор постоянной частоты.

### Уравнение

```
rate(t) = constant_rate (независимо от t)
```

### Конструктор

```python
ConstantRateGenerator(
    dt: float,
    params: Dict[str, Any],
    name: str = "constant_rate_generator",
    **kwargs
)
```

### Параметры (params)

| Параметр | Тип | Описание | Единицы |
|----------|-----|-----------|---------|
| `rate` | float/list | Постоянная частота | Гц |

### Примеры

**Один канал**:
```python
gen = ConstantRateGenerator(
    dt=0.5,
    params={
        'rate': 10.0,
    }
)
```

**Несколько каналов**:
```python
gen = ConstantRateGenerator(
    dt=0.5,
    params={
        'rate': [10.0, 15.0, 20.0],
    }
)
```

**Обучаемая частота**:
```python
gen = ConstantRateGenerator(
    dt=0.5,
    params={
        'rate': {'value': 10.0, 'trainable': True, 'min': 0.0, 'max': 100.0},
    }
)
```

### ParameterSpec

```python
{
    'rate': {'shape': (n_units,), 'trainable': ..., 'constraint': ..., 'units': 'Hz'},
}
```

---

## VonMisesGenerator

**Файл**: `neuraltide/inputs/von_mises.py`

Генератор тета-ритмического входа на основе распределения фон Мизаса.

### Уравнение

```
rate(t) = (MeanFiringRate / I0(kappa)) * exp(kappa * cos(2π*freq*t/1000 - phase))
```

где:
- `I0(kappa)` — модифицированная функция Бесселя нулевого порядка
- `kappa` вычисляется из R (R-value) через формулу r2kappa

### Формула r2kappa

Преобразование R → kappa использует кусочную аппроксимацию:

```python
if R < 0.53:
    kappa = 2*R + R³ + (5/6)*R⁵
elif R < 0.85:
    kappa = -0.4 + 1.39*R + 0.43/(1 - R)
else:
    kappa = 1 / (3*R - 4*R² + R³)
```

### Конструктор

```python
VonMisesGenerator(
    dt: float,
    params: Dict[str, Any],
    name: str = "von_mises_generator",
    **kwargs
)
```

### Параметры (params)

| Параметр | Тип | Описание | Единицы |
|----------|-----|-----------|---------|
| `mean_rate` | float/list | Средняя частота | Гц |
| `R` | float/list | R-value (0-1), характеризует концентрированность | безразмерный |
| `freq` | float/list | Частота тета-ритма | Гц |
| `phase` | float/list | Начальная фаза | рад |

**Примечание**: Параметр `R` не является обучаемым — он используется только для вычисления `kappa`, который сохраняется как константа.

### Примеры

**Один канал**:
```python
gen = VonMisesGenerator(
    dt=0.5,
    params={
        'mean_rate': 20.0,
        'R': 0.5,
        'freq': 8.0,
        'phase': 0.0,
    }
)
```

**Несколько каналов**:
```python
gen = VonMisesGenerator(
    dt=0.5,
    params={
        'mean_rate': [20.0, 15.0],
        'R': [0.5, 0.8],
        'freq': [8.0, 10.0],
        'phase': [0.0, 1.5],
    }
)
```

**Словарь (без trainable)**:
```python
gen = VonMisesGenerator(
    dt=0.5,
    params={
        'mean_rate': {'value': 20.0, 'trainable': False},
        'R': 0.5,
        'freq': {'value': 8.0, 'trainable': True, 'min': 1.0, 'max': 20.0},
        'phase': {'value': 0.0, 'trainable': False},
    }
)
```

### ParameterSpec

```python
{
    'mean_rate': {'shape': (n_units,), 'trainable': ..., 'constraint': ..., 'units': 'Hz'},
    'kappa':     {'shape': (n_units,), 'trainable': False, 'constraint': ..., 'units': 'dimensionless'},
    'freq':      {'shape': (n_units,), 'trainable': ..., 'constraint': ..., 'units': 'Hz'},
    'phase':     {'shape': (n_units,), 'trainable': ..., 'constraint': ..., 'units': 'rad'},
}
```

### Зависимости

Для вычисления `I0(kappa)` требуется scipy:
```python
from scipy.special import i0
```

---

## Создание собственного генератора

### Шаблон

```python
import tensorflow as tf
from neuraltide.core.base import BaseInputGenerator
from neuraltide.core.types import TensorType, Dict, Any

class MyGenerator(BaseInputGenerator):
    def __init__(self, dt: float, params: Dict[str, Any], 
                 name: str = "my_generator", **kwargs):
        super().__init__(params=params, dt=dt, name=name, **kwargs)
        
        # Регистрация параметров через _make_param
        self.param1 = self._make_param(self._params, 'param1')
        self.param2 = self._make_param(self._params, 'param2')
    
    def call(self, t: TensorType) -> TensorType:
        """
        Args:
            t: текущее время в мс. shape = [batch, 1]
        
        Returns:
            tf.Tensor, shape = [batch, n_units], в Гц
        """
        # Реализация генератора
        # Используйте tf.reshape для broadcasting параметров
        param1 = tf.reshape(self.param1, [1, self.n_units])
        
        # Вычисление выхода
        output = param1 * t  # пример
        
        return tf.nn.relu(output)  # частота должна быть >= 0
    
    @property
    def parameter_spec(self) -> Dict[str, Dict[str, Any]]:
        return {
            'param1': {
                'shape': (self.n_units,),
                'trainable': self.param1.trainable,
                'constraint': self._get_constraint_name(self.param1),
                'units': 'Hz',  # ваши единицы
            },
            'param2': {
                'shape': (self.n_units,),
                'trainable': self.param2.trainable,
                'constraint': self._get_constraint_name(self.param2),
                'units': 'dimensionless',
            },
        }
    
    def _get_constraint_name(self, var: tf.Variable) -> str:
        if var.constraint is not None:
            return var.constraint.__class__.__name__
        return None
```

### Регистрация генератора

Для использования с конфигурационным API:

```python
import neuraltide.config

neuraltide.config.register_input('MyGenerator', MyGenerator)
```

---

## Использование генераторов в сети

### Пример: один вход → одна популяция

```python
import tensorflow as tf
import numpy as np
from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.populations import IzhikevichMeanField
from neuraltide.synapses import StaticSynapse
from neuraltide.inputs import SinusoidalGenerator
from neuraltide.integrators import RK4Integrator

# Создание компонентов
gen = SinusoidalGenerator(dt=0.5, params={
    'amplitude': 10.0, 'freq': 8.0, 'phase': 0.0, 'offset': 5.0,
})

pop = IzhikevichMeanField(dt=0.5, params={
    'tau_pop': [1.0], 'alpha': [0.5], 'a': [0.02],
    'b': [0.2], 'w_jump': [0.1], 'Delta_I': [0.5], 'I_ext': [1.0],
})

syn = StaticSynapse(n_pre=1, n_post=1, dt=0.5, params={
    'gsyn_max': 0.1, 'pconn': [[1.0]], 'e_r': 0.0,
})

# Построение сети
graph = NetworkGraph(dt=0.5)
graph.add_input_population('input', gen)
graph.add_population('exc', pop)
graph.add_synapse('input->exc', syn, src='input', tgt='exc')

network = NetworkRNN(graph, integrator=RK4Integrator())

# Запуск
t_seq = tf.constant(np.arange(0, 1000, 0.5)[None, :, None])
output = network(t_seq)
print(output.firing_rates['exc'].shape)  # [1, 2000, 1]
```

### Пример: несколько входов

```python
gen1 = SinusoidalGenerator(dt=0.5, params={
    'amplitude': 10.0, 'freq': 8.0, 'phase': 0.0, 'offset': 5.0,
})

gen2 = VonMisesGenerator(dt=0.5, params={
    'mean_rate': 15.0, 'R': 0.7, 'freq': 5.0, 'phase': 1.0,
})

pop = IzhikevichMeanField(dt=0.5, params={
    'tau_pop': [1.0, 1.0], 'alpha': [0.5, 0.5], 'a': [0.02, 0.02],
    'b': [0.2, 0.2], 'w_jump': [0.1, 0.1], 
    'Delta_I': [0.5, 0.5], 'I_ext': [1.0, 1.0],
})

# Один синапс от gen1, другой от gen2
syn1 = StaticSynapse(n_pre=1, n_post=2, dt=0.5, params={
    'gsyn_max': [[0.1, 0.1]], 'pconn': [[1.0, 1.0]], 'e_r': 0.0,
})

syn2 = StaticSynapse(n_pre=1, n_post=2, dt=0.5, params={
    'gsyn_max': [[0.05, 0.08]], 'pconn': [[1.0, 1.0]], 'e_r': 0.0,
})

graph = NetworkGraph(dt=0.5)
graph.add_input_population('sin_input', gen1)
graph.add_input_population('vm_input', gen2)
graph.add_population('exc', pop)
graph.add_synapse('sin->exc', syn1, src='sin_input', tgt='exc')
graph.add_synapse('vm->exc', syn2, src='vm_input', tgt='exc')
```