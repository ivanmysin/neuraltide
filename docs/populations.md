# Популяции

NeuralTide предоставляет несколько типов популяционных моделей:

- **IzhikevichMeanField** — модель Ижикевича (mean-field)
- **WilsonCowan** — модель Вильсона-Коуэна
- **FokkerPlanckPopulation** — базовый класс для модели Фоккера-Планка
- **InputPopulation** — псевдо-популяция для входных генераторов

---

## IzhikevichMeanField

**Файл**: `neuraltide/populations/izhikevich_mf.py`

Модель Izhikevich в приближении mean-field (Montbrio-Pazo-Roxin).

### Уравнения

```
τ_pop * dr/dt = Δ_I/π + 2*r*v - (α + g_syn)*r
τ_pop * dv/dt = v² - α*v - w + I_ext + I_syn - (π*r)²
τ_pop * dw/dt = a*(b*v - w) + w_jump*r
```

### Переменные состояния

Модель использует безразмерные переменные:
- `r` — безразмерная частота разрядов
- `v` — безразмерный средний мембранный потенциал (относительно V_rest)
- `w` — безразмерный ток адаптации

### Конструктор

```python
IzhikevichMeanField(
    dt: float,
    params: Optional[Dict[str, Any]] = None,
    name: str = "izhikevich_mf",
    **kwargs
)
```

### Параметры (params)

Модель поддерживает два режима:

#### Режим 1: Безразмерные параметры (рекомендуемый)

| Параметр | Тип | Описание | Типичные значения |
|----------|-----|-----------|------------------|
| `tau_pop` | list[float] | Постоянная времени популяции [мс] | [1.0, ...] |
| `alpha` | list[float] | Параметр порога (безразмерный) | [0.5, ...] |
| `a` | list[float] | Скорость адаптации (безразмерный) | [0.02, ...] |
| `b` | list[float] | Связь адаптации (безразмерный) | [0.2, ...] |
| `w_jump` | list[float] | Скачок адаптации (безразмерный) | [0.1, ...] |
| `Delta_I` | list[float] | Спред тока Лоренца (безразмерный) | [0.5, ...] |
| `I_ext` | list[float] | Внешний ток (безразмерный) | [1.0, ...] |

**Пример**:
```python
pop = IzhikevichMeanField(
    dt=0.5,
    params={
        'tau_pop':   {'value': [1.0, 1.0, 1.0, 1.0], 'trainable': False},
        'alpha':     {'value': [0.5, 0.5, 0.5, 0.5], 'trainable': False},
        'a':         {'value': [0.02, 0.02, 0.02, 0.02], 'trainable': False},
        'b':         {'value': [0.2, 0.2, 0.2, 0.2], 'trainable': False},
        'w_jump':    {'value': [0.1, 0.1, 0.1, 0.1], 'trainable': False},
        'Delta_I':   {'value': [0.5, 0.6, 0.5, 0.6], 'trainable': True,
                      'min': 0.01, 'max': 2.0},
        'I_ext':     {'value': [1.0, 1.2, 1.0, 1.2], 'trainable': True},
    }
)
```

#### Режим 2: Размерные параметры (для обратной совместимости)

| Параметр | Тип | Описание |
|----------|-----|-----------|
| `V_rest` | float | Потенциал покоя [мВ] |
| `V_T` | float | Пороговый потенциал [мВ] |
| `V_peak` | float | Пиковый потенциал [мВ] (игнорируется) |
| `V_reset` | float | Потенциал сброса [мВ] (игнорируется) |
| `Cm` | float | Мембранная ёмкость [пФ] |
| `K` | float | Параметр масштабирования [нС/мВ] |
| `A` | float | Параметр адаптации A |
| `B` | float | Параметр адаптации B |
| `W_jump` | float | Скачок адаптации |
| `Delta_I` | float | Спред тока [пА] |
| `I_ext` | float | Внешний ток [пА] |

**Пример**:
```python
pop = IzhikevichMeanField(
    dt=0.5,
    params={
        'V_rest': -57.6, 'V_T': -35.5, 'V_peak': 21.7, 'V_reset': -48.7,
        'Cm': 114.0, 'K': 1.194, 'A': 0.0046, 'B': 0.2157, 'W_jump': 2.0,
        'Delta_I': 20.0, 'I_ext': 120.0
    }
)
```

### Состояние

```python
state_size = [
    tf.TensorShape([1, n_units]),  # r
    tf.TensorShape([1, n_units]),  # v
    tf.TensorShape([1, n_units]),  # w
]
```

### Начальное состояние

`get_initial_state()` возвращает `[r=0, v=0, w=0]`, что соответствует состоянию покоя.

### Observables

```python
{
    'firing_rate': r,           # безразмерная частота (для Hz: r / (dt * 1e-3))
    'v_mean': v,                # безразмерный потенциал
    'w_mean': w,                # безразмерный ток адаптации
}
```

### Преобразование размеров

Статический метод `dimensionless_to_dimensional()` преобразует безразмерные параметры обратно в размерные:

```python
IzhikevichMeanField.dimensionless_to_dimensional(
    tau_pop=1.0,
    alpha=0.5,
    a=0.02,
    b=0.2,
    w_jump=0.1,
    Delta_I=0.5,
    I_ext=1.0,
    V_rest=-57.6,
    K=1.194,
)
# Returns: {'V_T': ..., 'Cm': ..., 'A': ..., 'B': ..., 'W_jump': ..., 'Delta_I': ..., 'I_ext': ...}
```

---

## WilsonCowan

**Файл**: `neuraltide/populations/wilson_cowan.py`

Модель Wilson-Cowan для взаимодействия возбуждающих и тормозных популяций.

### Уравнения

```
τ_E * dE/dt = -E + F_E(w_EE*E - w_IE*I + I_ext_E + I_syn)
τ_I * dI/dt = -I + F_I(w_EI*E - w_II*I + I_ext_I)

F(x) = 1 / (1 + exp(-a_coeff*(x - theta)))
```

### Конструктор

```python
WilsonCowan(
    n_units: int,
    dt: float,
    params: dict,
    **kwargs
)
```

### Параметры (params)

| Параметр | Тип | Описание | Единицы |
|----------|-----|-----------|---------|
| `tau_E` | list[float] | Постоянная времени E-популяции | мс |
| `tau_I` | list[float] | Постоянная времени I-популяции | мс |
| `a_E` | list[float] | Коэффициент сигмоиды E | 1/мВ |
| `a_I` | list[float] | Коэффициент сигмоиды I | 1/мВ |
| `theta_E` | list[float] | Порог сигмоиды E | мВ |
| `theta_I` | list[float] | Порог сигмоиды I | мВ |
| `w_EE` | list[float] | Вес E→E | безразмерный |
| `w_IE` | list[float] | Вес I→E | безразмерный |
| `w_EI` | list[float] | Вес E→I | безразмерный |
| `w_II` | list[float] | Вес I→I | безразмерный |
| `I_ext_E` | list[float] | Внешний ток E | мВ |
| `I_ext_I` | list[float] | Внешний ток I | мВ |
| `max_rate` | list[float] | Максимальная частота | Гц |

**Пример**:
```python
pop = WilsonCowan(
    n_units=2,
    dt=0.5,
    params={
        'tau_E': [10.0, 10.0],
        'tau_I': [10.0, 10.0],
        'a_E': [1.0, 1.0],
        'a_I': [1.0, 1.0],
        'theta_E': [0.0, 0.0],
        'theta_I': [0.0, 0.0],
        'w_EE': [1.0, 1.0],
        'w_IE': [1.0, 1.0],
        'w_EI': [1.0, 1.0],
        'w_II': [1.0, 1.0],
        'I_ext_E': [0.5, 0.5],
        'I_ext_I': [0.0, 0.0],
        'max_rate': [100.0, 100.0],
    }
)
```

### Состояние

```python
state_size = [
    tf.TensorShape([1, n_units]),  # E
    tf.TensorShape([1, n_units]),  # I
]
```

### Observables

```python
{'firing_rate': E * max_rate}  # с учётом max_rate
```

---

## FokkerPlanckPopulation

**Файл**: `neuraltide/populations/fokker_planck.py`

Базовый класс для популяций с дискретизованным распределением P(V, t).

### Концепция

Модель описывает эволюцию распределения вероятностей потенциалов на дискретной сетке. Пользователь наследует класс и реализует `derivatives()` с дискретизованным оператором Фоккера-Планка.

### Конструктор

```python
FokkerPlanckPopulation(
    n_units: int,
    dt: float,
    grid_size: int = 100,
    v_min: float = -100.0,
    v_max: float = 50.0,
    **kwargs
)
```

### Параметры

| Параметр | Тип | Описание |
|----------|-----|-----------|
| `grid_size` | int | Число точек на сетке (по умолчанию 100) |
| `v_min` | float | Минимальное значение потенциала [мВ] |
| `v_max` | float | Максимальное значение потенциала [мВ] |

### Состояние

```python
state_size = [tf.TensorShape([1, grid_size])]  # P(V)
```

### Методы

#### `get_boundary_flux(P, dV)`

Вычисляет поток через правую границу (пороговый потенциал):

```
J_out = -D * dP/dV|_{V=V_threshold}
```

### Пример: создание собственной Fokker-Planck модели

```python
class LeakyIF(FokkerPlanckPopulation):
    def __init__(self, n_units: int, dt: float, params: dict, **kwargs):
        super().__init__(n_units=n_units, dt=dt, **kwargs)
        
        self.v_reset = self._make_param(params, 'v_reset')
        self.v_thresh = self._make_param(params, 'v_thresh')
        self.tau_m = self._make_param(params, 'tau_m')
        
    def derivatives(self, state, total_synaptic_input):
        P = state[0]
        I_syn = total_synaptic_input['I_syn']
        
        # Дискретизованный оператор Фоккера-Планка
        # ... (реализация пользователя)
        
        return [dPdt]
    
    def get_firing_rate(self, state):
        P = state[0]
        return self.get_boundary_flux(P, self.dV)
```

---

## InputPopulation

**Файл**: `neuraltide/populations/input_population.py`

Псевдо-популяция без динамики, оборачивающая BaseInputGenerator.

### Назначение

Позволяет подключать внешние входные сигналы к сети через полноценные обучаемые синапсы, единообразно с рекуррентными проекциями между динамическими популяциями.

### Семантика

- Не имеет уравнений динамики (`derivatives()` возвращает `[]`)
- Не обновляется интегратором
- Состояние: `[t_current]`, shape [1, 1] — текущее время в мс
- `get_firing_rate(state)` вызывает `generator(state[0])`
- Не может быть целью синапса (только источником)

### Конструктор

```python
InputPopulation(
    generator: BaseInputGenerator,
    **kwargs
)
```

### Пример

```python
from neuraltide.inputs import SinusoidalGenerator
from neuraltide.populations import InputPopulation

gen = SinusoidalGenerator(dt=0.5, params={
    'amplitude': 10.0,
    'freq': 8.0,
    'phase': 0.0,
    'offset': 5.0,
})

input_pop = InputPopulation(generator=gen)
# input_pop.n_units = 1 (выводится из параметров генератора)
# input_pop.dt = 0.5 (берётся из генератора)
```

### Создание через NetworkGraph

```python
graph = NetworkGraph(dt=0.5)
graph.add_input_population('my_input', generator)
```

Это эквивалентно:
```python
input_pop = InputPopulation(generator=generator, name='my_input_input_pop')
graph.add_population('my_input', input_pop)
```