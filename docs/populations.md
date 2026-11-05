# Популяции

NeuralTide предоставляет несколько типов популяционных моделей:

- **IzhikevichMeanField** — модель Ижикевича (mean-field)
- **WilsonCowan** — модель Вильсона-Коуэна
- **FokkerPlanckPopulation** — базовый класс для модели Фоккера-Планка
- **InputPopulation** — псевдо-популяция для входных генераторов (deprecated)

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

Модель Вильсона–Коуэна в форме одной популяции (rate model). Одна
популяция описывается одной переменной E — нормированной активностью
(0..1). Торможение учитывается через входящие синаптические токи
(в том числе от отдельных тормозных популяций с отрицательным весом).

### Уравнения

```
τ * dE/dt = -E + F(w*E + I_ext + I_syn)

F(x) = 1 / (1 + exp(-a*(x - theta)))
```

### Конструктор

```python
WilsonCowan(
    dt: float,
    params: dict,
    **kwargs
)
```

`n_units` выводится из максимальной длины параметров (как у
`IzhikevichMeanField`), явно не передаётся.

### Параметры (params)

| Параметр | Тип | Описание | Единицы |
|----------|-----|-----------|---------|
| `tau` | list[float] | Постоянная времени | мс |
| `a` | list[float] | Крутизна сигмоиды | безразмерная |
| `theta` | list[float] | Порог сигмоиды | мВ |
| `w` | list[float] | Рекуррентный вес | безразмерный |
| `I_ext` | list[float] | Внешний ток | мВ |
| `max_rate` | list[float] | Масштаб частоты разрядов | Гц |

**Пример**:
```python
pop = WilsonCowan(
    dt=0.5,
    params={
        'tau': [10.0, 10.0],
        'a': [1.0, 1.0],
        'theta': [0.0, 0.0],
        'w': [1.0, 1.0],
        'I_ext': [0.5, 0.5],
        'max_rate': [100.0, 100.0],
    }
)
# n_units=2
```

### Состояние

```python
state_size = [
    tf.TensorShape([1, n_units]),  # E
]
```

### Observables

```python
{'firing_rate': relu(E) * max_rate}  # нет 'v_mean'
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

## BrunelFokkerPlanck

**Файл**: `neuraltide/populations/brunel_fp.py`

Готовая популяция ЛИФ-нейронов в приближении Фоккера–Планка по
Brunel (2000), модель A (одинаковые нейроны). Уравнение для плотности
распределения потенциалов `P(V, t)` на сетке `[v_min, v_theta]`:

```
τ_m · ∂P/∂t = (σ²/2) · ∂²P/∂V² + ∂/∂V[(V − μ)·P]

μ(t)  = mu_0 + I_syn(t)     — синаптический ток добавляется к дрейфу
σ²(t) = sigma_ext²          — параметр
```

Граничные условия: поглощающий порог `v_theta` (частота разрядов —
диффузионный поток через порог), реинжекция потока `ν(t − τ_rp)` в
точке `v_reset`, рефрактерная масса `p_r = ∫_{t−τ_rp}^{t} ν(u) du`.
Нормировка `∫P dV + p_r = 1` сохраняется схемой точно.

Дискретизация: консервативная конечно-объёмная схема (диффузия —
центральные разности, дрейф — против потока) с внутренним
субшагиванием по Куранту и защитой от отрицательной плотности.
Задержка `τ_rp` — кольцевой буфер из `τ_rp/dt` значений ν.
Рекомендуются `EulerIntegrator`/`HeunIntegrator`.

### Конструктор

```python
BrunelFokkerPlanck(
    dt: float,
    params: dict,
    grid_size: int = 300,
    v_min: float = -100.0,
    **kwargs
)
```

Один объект — одна FP-популяция (`n_units = 1`).

### Параметры (params)

| Параметр | Тип | Описание | Единицы |
|----------|-----|-----------|---------|
| `tau_m` | float | Мембранная постоянная времени | мс |
| `v_theta` | float | Пороговый потенциал | мВ |
| `v_reset` | float | Потенциал сброса | мВ |
| `tau_rp` | float | Рефрактерный период | мс |
| `mu_0` | float | Средний дрейф в покое | мВ |
| `sigma_ext` | float | Стандартное отклонение шума | мВ |

### Состояние

```python
state_size = [
    tf.TensorShape([1, grid_size]),  # P(V)
    tf.TensorShape([1, 1]),          # p_r
    tf.TensorShape([1, tau_rp/dt]),  # буфер задержки ν
]
```

### Observables

```python
{'firing_rate', 'v_mean', 'p_refractory'}
```

### Проверка

Стационарная частота совпадает с самосогласованным уравнением (21)
Brunel (2000) с точностью ~0.2% (сетка 250, режим у порога);
пример запуска — [`examples/example_06_brunel_fp.py`](../examples/example_06_brunel_fp.py).

---

## InputPopulation (deprecated)

**Файл**: `neuraltide/populations/input_population.py`

> **Внимание**: `InputPopulation` deprecated. Используйте `NetworkGraph.declare_input()` и предвычисленные входы вместо оборачивания генераторов в `InputPopulation`.

Псевдо-популяция без динамики, оборачивающая BaseInputGenerator.

### Назначение

Позволяет подключать внешние входные сигналы к сети через полноценные обучаемые синапсы, единообразно с рекуррентными проекциями между динамическими популяциями.

### Семантика

- Не имеет уравнений динамики (`derivatives()` возвращает `[]`)
- Не обновляется интегратором
- Состояние: `[t_current, extra_inputs]` — тензоры формы `[1, 1]` (время в мс) и `[1, n_cols]` (дополнительные данные — координаты, параметры стимула и т.д.)
- `get_firing_rate(state)` вызывает `generator(state[0], extra_inputs=state[1])`
- Не может быть целью синапса (только источником)

### Новый подход

Вместо `InputPopulation` используйте:

```python
# 1. Объявление входа
graph = NetworkGraph(dt=0.5)
graph.declare_input('theta', n_units=gen.n_units)

# 2. Предвычисление входных частот
inputs = graph.pack_inputs({'theta': gen(t_seq)})

# 3. Передача в сеть
output = network(t_seq, inputs=inputs)
```

### Конструктор (deprecated)

```python
InputPopulation(
    generator: BaseInputGenerator,
    **kwargs
)
```

### Пример (deprecated)

```python
from neuraltide.inputs import SinusoidalGenerator
from neuraltide.populations import InputPopulation

gen = SinusoidalGenerator(dt=0.5, params={
    'amplitude': 10.0,
    'freq': 8.0,
    'phase': 0.0,
    'offset': 5.0,
})

# Не рекомендуется:
input_pop = InputPopulation(generator=gen)

# Рекомендуется:
graph.declare_input('my_input', n_units=gen.n_units)
inputs = graph.pack_inputs({'my_input': gen(t_seq)})
output = network(t_seq, inputs=inputs)
```
