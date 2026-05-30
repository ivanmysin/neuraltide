# Генераторы входов

NeuralTide предоставляет несколько типов генераторов входных сигналов:

- **SinusoidalGenerator** — синусоидальный генератор
- **ConstantRateGenerator** — генератор постоянной частоты
- **VonMisesGenerator** — генератор на основе распределения фон Мизаса

Все генераторы являются подклассами `BaseInputGenerator` и поддерживают векторизацию.

> **Примечание**: `InputPopulation` deprecated. Используйте `NetworkGraph.declare_input()` и предвычисленные входы вместо оборачивания генераторов в `InputPopulation`.

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

Генераторы подключаются к сети через объявление входа и предвычисление частот:

```python
graph = NetworkGraph(dt=0.5)
graph.declare_input('my_input', n_units=gen.n_units)

# Предвычисление частот разрядов
inputs = graph.pack_inputs({'my_input': gen(t_seq)})

# Запуск сети
output = network(t_seq, inputs=inputs)
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

### call() сигнатура

```python
def call(self, t: TensorType, extra_inputs: Optional[TensorType] = None) -> TensorType:
```

- `t`: форма `[batch, 1]`, `[T]`, или `[batch, T, 1]` — время в мс
- `extra_inputs`: опционально — дополнительные данные (игнорируется этим генератором)
- Возвращает: `[batch, n_units]` в Hz

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

## PlaceFieldGenerator

**Файл**: `neuraltide/inputs/place_field.py`

Генератор place fields гиппокампа с модуляцией тета-ритмом и фазовой прецессией.

Все пространственные координаты — в сантиметрах (см). Время — в миллисекундах (мс).

### Уравнение

```
rate_i(t) = peak_rate * spatial * theta_inside
          + bg_rate * (1 - spatial) * (1 - tmf + tmf * theta_outside)

spatial = exp(-0.5 * ((x-cx)/r)^2 + ((y-cy)/r)^2)

theta_inside  = exp(kappa * cos(2π·freq·t/1000 + ph0 + dphi)) / I0(kappa)
theta_outside = exp(kappa * cos(2π·freq·t/1000 + ph_out)) / I0(kappa)

dphi = -slope_rad * (x - cx)   — фазовая прецессия
tmf  = theta_modulation_factor (0 → нет модуляции вне поля, 1 → полная)
```

### Особенности

- Требует пространственных координат `(x, y)` (см) через `extra_inputs`
- Без координат (`extra_inputs=None` или n_cols < 2) — spatial=0 (чисто фоновая частота)
- Поддерживает фазовую прецессию (сдвиг фазы тета-ритма в зависимости от позиции)
- Раздельные фазы тета-ритма внутри поля (`precession_init_phase`) и вне поля (`phase_outside`)

### Конструктор

```python
PlaceFieldGenerator(
    dt: float,
    params: Dict[str, Any],
    arena_size: Tuple[Tuple[float, float], Tuple[float, float]] = ((-100.0, 100.0), (-100.0, 100.0)),
    arena_radius: float = 100.0,
    name: str = "place_field_generator",
    **kwargs
)
```

### Параметры

| Параметр | Тип | Единицы | Описание |
|----------|-----|---------|----------|
| `center_x` | скаляр / `[n_units]` | см | Центр place field (X) |
| `center_y` | скаляр / `[n_units]` | см | Центр place field (Y) |
| `radius` | скаляр / `[n_units]` | см | Ширина place field (σ гауссианы) |
| `peak_rate` | скаляр / `[n_units]` | Hz | Пиковая частота в центре поля (по умолчанию 20.0) |
| `background_rate` | скаляр / `[n_units]` | Hz | Фоновая частота вне поля (по умолчанию 0.0) |
| `theta_modulation_factor` | скаляр / `[n_units]` | — | Сила тета-модуляции вне поля (0 = нет, 1 = полная) |
| `precession_slope` | скаляр / `[n_units]` | deg/cm | Наклон фазовой прецессии (>0 = классическая) |
| `precession_init_phase` | скаляр / `[n_units]` | градусы | Фаза тета-ритма в центре поля |
| `phase_outside` | скаляр / `[n_units]` | градусы | Фаза тета-ритма вне поля (по умолчанию 0.0) |
| `R` | скаляр / `[n_units]` | — | Концентрация Von Mises для тета-ритма (0–1) |
| `freq` | скаляр / `[n_units]` | Hz | Частота тета-ритма |

### call() сигнатура

```python
def call(self, t: TensorType, extra_inputs: Optional[TensorType] = None) -> TensorType:
```

- `t`: форма `[batch, T, 1]`, `[batch, T]`, или `[T]` — время в мс
- `extra_inputs`: опционально, форма `[batch, T, n_cols]` или `[batch, n_cols]`
  - Колонка 0: x-координата (см)
  - Колонка 1: y-координата (см)
  - Если `None` или `n_cols < 2` — spatial=0, работает как фоновая частота
- Возвращает: `[batch, T, n_units]` в Hz

### Пример (прямой вызов)

```python
import numpy as np
import tensorflow as tf
from neuraltide.inputs import PlaceFieldGenerator

gen = PlaceFieldGenerator(dt=0.5, params={
    'center_x': [40.0, -50.0], 'center_y': [30.0, 40.0],
    'radius': [35.0, 40.0], 'peak_rate': [25.0, 30.0],
    'background_rate': [2.0, 3.0], 'theta_modulation_factor': 0.0,
    'precession_slope': [30.0, 35.0], 'precession_init_phase': [0.0, 90.0],
    'phase_outside': 0.0,
    'R': 0.6, 'freq': 8.0,
})

# Время (мс) + координаты (см)
t = tf.constant(np.arange(0, 1000, 0.5, dtype=np.float32)[None, :, None])
x = 70.0 * tf.cos(2*np.pi * tf.range(2000, dtype=tf.float32) / 2000 * 2)
y = 70.0 * tf.sin(2*np.pi * tf.range(2000, dtype=tf.float32) / 2000 * 2)
extra = tf.stack([x, y], axis=-1)[None, :, :]  # [1, T, 2]

rates = gen(t, extra_inputs=extra)  # [1, T, 2]
```

### Пример (полный скрипт)

См. `examples/example_place_field_phase_precession.py` — демонстрирует прямой вызов
генератора с зашумлённой круговой траекторией, диагностический вывод параметров
и три графика (rate vs theta, фазовая прецессия, карта арены).

### Использование через NetworkRNN

При интеграции в сеть координаты сначала передаются в генератор для вычисления частот, затем частоты упаковываются через `pack_inputs()`:

```python
# Вычисление входных частот с учётом координат
rates = gen(t_seq, extra_inputs=extra)
inputs = graph.pack_inputs({'place': rates})

# Запуск сети
output = network(t_seq, inputs=inputs)
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
graph.declare_input('input', n_units=gen.n_units)
graph.add_population('exc', pop)
graph.add_synapse('input->exc', syn, src='input', tgt='exc')

network = NetworkRNN(graph, integrator=RK4Integrator())

# Запуск
t_seq = tf.constant(np.arange(0, 1000, 0.5)[None, :, None])
inputs = graph.pack_inputs({'input': gen(t_seq)})
output = network(t_seq, inputs=inputs)
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
graph.declare_input('sin_input', n_units=gen1.n_units)
graph.declare_input('vm_input', n_units=gen2.n_units)
graph.add_population('exc', pop)
graph.add_synapse('sin->exc', syn1, src='sin_input', tgt='exc')
graph.add_synapse('vm->exc', syn2, src='vm_input', tgt='exc')

# Упаковка входов
inputs = graph.pack_inputs({
    'sin_input': gen1(t_seq),
    'vm_input': gen2(t_seq),
})
output = network(t_seq, inputs=inputs)
```

### Пример: позиционно-зависимый вход через pack_inputs

PlaceFieldGenerator использует координаты `(x, y)`, которые передаются в генератор для вычисления частот, затем частоты упаковываются через `pack_inputs()`.

```python
import numpy as np
import tensorflow as tf
from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.populations import IzhikevichMeanField
from neuraltide.synapses import StaticSynapse
from neuraltide.inputs import PlaceFieldGenerator
from neuraltide.integrators import RK4Integrator

gen = PlaceFieldGenerator(dt=0.5, params={
    'center_x': [40.0, -50.0], 'center_y': [30.0, 40.0],
    'radius': [35.0, 40.0], 'peak_rate': [25.0, 30.0],
    'background_rate': [2.0, 3.0], 'theta_modulation_factor': 0.0,
    'precession_slope': [30.0, 35.0], 'precession_init_phase': [0.0, 90.0],
    'phase_outside': 0.0,
    'R': 0.6, 'freq': 8.0,
})

pop = IzhikevichMeanField(dt=0.5, params={
    'tau_pop': [1.0, 1.0], 'alpha': [0.5, 0.5], 'a': [0.02, 0.02],
    'b': [0.2, 0.2], 'w_jump': [0.1, 0.1], 'Delta_I': [0.05, 0.05],
    'I_ext': [0.0, 0.0],
})
syn = StaticSynapse(n_pre=2, n_post=2, dt=0.5, params={
    'gsyn_max': [[1.0, 0.0], [0.0, 1.0]], 'pconn': 1.0, 'e_r': 5.0,
})

graph = NetworkGraph(dt=0.5)
graph.declare_input('place', n_units=gen.n_units)
graph.add_population('readout', pop)
graph.add_synapse('place->readout', syn, src='place', tgt='readout')

network = NetworkRNN(graph, integrator=RK4Integrator())

# Время
t_seq = tf.constant(np.arange(0, 5000, 0.5, dtype=np.float32)[None, :, None])

# Координаты (см)
n_steps = t_seq.shape[1]
pos_x = 70.0 * np.cos(2 * np.pi * np.arange(n_steps) / n_steps * 2)
pos_y = 70.0 * np.sin(2 * np.pi * np.arange(n_steps) / n_steps * 2)
extra = tf.constant(
    np.stack([pos_x, pos_y], axis=-1).astype(np.float32)[None, :, :]
)

# Вычисление частот с учётом координат, затем упаковка
rates = gen(t_seq, extra_inputs=extra)
inputs = graph.pack_inputs({'place': rates})

output = network(t_seq, inputs=inputs)
print(output.firing_rates['readout'].shape)  # [1, n_steps, 2]
```
