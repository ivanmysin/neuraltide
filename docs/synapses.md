# Синапсы

NeuralTide предоставляет несколько типов синаптических моделей:

- **StaticSynapse** — статический синапс без пластичности
- **NMDASynapse** — NMDA синапс с магниевым блоком
- **TsodyksMarkramSynapse** — синапс с кратковременной пластичностью (STP)
- **CompositeSynapse** — композитный синапс, объединяющий несколько компонент

---

## Общие принципы

### Матрица соединений pconn

Матрица `pconn` имеет форму `[n_pre, n_post]`:
- `pconn[i, j]` — вероятность/маска связи **от** популяции `i` **к** популяции `j`
- Обычно `trainable=False`

**Пример**: 2 входа → 2 нейрона, полносвязная
```python
pconn = [[1.0, 1.0],
         [1.0, 1.0]]
```

**Пример**: 2 входа → 2 нейрона, диагональная (1→1, 2→2)
```python
pconn = [[1.0, 0.0],
         [0.0, 1.0]]
```

### Выход синапса

Метод `forward()` возвращает словарь:
```python
{
    'I_syn': tf.Tensor [1, n_post],  # синаптический ток
    'g_syn': tf.Tensor [1, n_post],  # синаптическая проводимость
}
```

---

## StaticSynapse

**Файл**: `neuraltide/synapses/static.py`

Статический синапс без пластичности.

### Уравнения

```
I_syn = gsyn_max * FRpre_normed * (e_r - post_v)
g_syn = gsyn_max * FRpre_normed
```

где `FRpre_normed = pconn * (dt * pre_firing_rate / 1000)`

### Конструктор

```python
StaticSynapse(
    n_pre: int,
    n_post: int,
    dt: float,
    params: dict,
    **kwargs
)
```

### Параметры (params)

| Параметр | Тип | Описание | Единицы |
|----------|-----|-----------|---------|
| `gsyn_max` | float/matrix | Максимальная проводимость | мС/см² |
| `pconn` | matrix [n_pre, n_post] | Матрица соединений | безразмерный |
| `e_r` | float/matrix | Потенциал реверса | мВ |

**Пример**:
```python
syn = StaticSynapse(
    n_pre=1,
    n_post=2,
    dt=0.5,
    params={
        'gsyn_max': {'value': [[0.1, 0.15]], 'trainable': True},
        'pconn':    {'value': [[1.0, 1.0]], 'trainable': False},
        'e_r':      {'value': 0.0, 'trainable': False},
    }
)
```

### Состояние

```python
state_size = []  # пустой список, нет состояния
```

### ParameterSpec

```python
{
    'gsyn_max': {'shape': (n_pre, n_post), 'trainable': ..., 'constraint': 'NonNegConstraint', 'units': 'mS/cm^2'},
    'pconn':    {'shape': (n_pre, n_post), 'trainable': False, 'constraint': 'UnitIntervalConstraint', 'units': 'dimensionless'},
    'e_r':      {'shape': (n_pre, n_post), 'trainable': False, 'constraint': None, 'units': 'mV'},
}
```

---

## NMDASynapse

**Файл**: `neuraltide/synapses/nmda.py`

NMDA синапс с двойной экспонентой и магниевым блоком.

### Уравнения

```
# Кинетика двойной экспоненты:
dgnmda_new = dgnmda + dt*(s_input - gnmda - (tau1 + tau2)*dgnmda) / (tau1*tau2)
gnmda_new = gnmda + dt * dgnmda_new

# Магниевый блок:
mg_block = 1 / (1 + Mgb * exp(-av*(post_v - v_ref)))

# Синаптический ток:
I_syn = gsyn_max_nmda * gnmda_new * mg_block * (e_r - post_v)
g_syn = gsyn_max_nmda * gnmda_new * mg_block
```

### Конструктор

```python
NMDASynapse(
    n_pre: int,
    n_post: int,
    dt: float,
    params: dict,
    **kwargs
)
```

### Параметры (params)

| Параметр | Тип | Описание | Единицы |
|----------|-----|-----------|---------|
| `gsyn_max_nmda` | float/matrix | Максимальная проводимость NMDA | мС/см² |
| `tau1_nmda` | float/matrix | Время нарастания | мс |
| `tau2_nmda` | float/matrix | Время спада | мс |
| `Mgb` | float/matrix | Концентрация магния | мМ |
| `av_nmda` | float/matrix | Коэффициент блока | 1/мВ |
| `pconn_nmda` | matrix [n_pre, n_post] | Матрица соединений | безразмерный |
| `e_r_nmda` | float/matrix | Потенциал реверса NMDA | мВ |
| `v_ref` | float/matrix | Опорный потенциал для блока | мВ |

**Пример**:
```python
syn_nmda = NMDASynapse(
    n_pre=1,
    n_post=2,
    dt=0.5,
    params={
        'gsyn_max_nmda': {'value': 0.15, 'trainable': True},
        'tau1_nmda':     {'value': 5.0, 'trainable': False},
        'tau2_nmda':     {'value': 150.0, 'trainable': False},
        'Mgb':           {'value': 1.0, 'trainable': False},
        'av_nmda':       {'value': 0.08, 'trainable': False},
        'pconn_nmda':    {'value': [[1.0, 1.0]], 'trainable': False},
        'e_r_nmda':      {'value': 0.0, 'trainable': False},
        'v_ref':         {'value': 0.0, 'trainable': False},
    }
)
```

### Состояние

```python
state_size = [
    tf.TensorShape([n_pre, n_post]),  # gnmda
    tf.TensorShape([n_pre, n_post]),  # dgnmda
]
```

### ParameterSpec

```python
{
    'gsyn_max_nmda': {'shape': (n_pre, n_post), 'trainable': ..., 'constraint': 'NonNegConstraint', 'units': 'mS/cm^2'},
    'tau1_nmda':     {'shape': (n_pre, n_post), 'trainable': False, 'constraint': None, 'units': 'ms'},
    'tau2_nmda':     {'shape': (n_pre, n_post), 'trainable': False, 'constraint': None, 'units': 'ms'},
    'Mgb':           {'shape': (n_pre, n_post), 'trainable': False, 'constraint': None, 'units': 'mM'},
    'av_nmda':       {'shape': (n_pre, n_post), 'trainable': False, 'constraint': None, 'units': 'mV^-1'},
    'pconn_nmda':    {'shape': (n_pre, n_post), 'trainable': False, 'constraint': 'UnitIntervalConstraint', 'units': 'dimensionless'},
    'e_r_nmda':      {'shape': (n_pre, n_post), 'trainable': False, 'constraint': None, 'units': 'mV'},
    'v_ref':         {'shape': (n_pre, n_post), 'trainable': False, 'constraint': None, 'units': 'mV'},
}
```

---

## TsodyksMarkramSynapse

**Файл**: `neuraltide/synapses/tsodyks_markram.py`

Синапс Цойдокса-Маркрама с кратковременной пластичностью (Short-Term Plasticity, STP).

### Уравнения (аналитическое решение)

```
a_  = A * exp(-dt/tau_d)
r_  = 1 + (R - 1 + tau_d/(tau_d - tau_r)*A) * exp(-dt/tau_r) - tau_d/(tau_d - tau_r)*A
u_  = U * exp(-dt/tau_f)

R_new = r_ - U * r_ * FRpre_normed
U_new = u_ + Uinc*(1 - u_)*FRpre_normed
A_new = a_ + U * r_ * FRpre_normed

I_syn = gsyn_max * A_new * (e_r - post_v)
g_syn = gsyn_max * A_new
```

### Переменные состояния

- `R` — доступный ресурс (0-1)
- `U` — вероятность использования (0-1)
- `A` — активность (текущая эффективность)

### Конструктор

```python
TsodyksMarkramSynapse(
    n_pre: int,
    n_post: int,
    dt: float,
    params: dict,
    **kwargs
)
```

### Параметры (params)

| Параметр | Тип | Описание | Единицы |
|----------|-----|-----------|---------|
| `gsyn_max` | float/matrix | Максимальная проводимость | мС/см² |
| `tau_f` | float/matrix | Время фасилитации | мс |
| `tau_d` | float/matrix | Время депрессии | мс |
| `tau_r` | float/matrix | Время восстановления | мс |
| `Uinc` | float/matrix | Инкремент использования | безразмерный |
| `pconn` | matrix [n_pre, n_post] | Матрица соединений | безразмерный |
| `e_r` | float/matrix | Потенциал реверса | мВ |

**Типичные значения**:
- Фасилитация: `tau_f > 0`, `tau_d ≈ tau_r`, `Uinc` мало
- Депрессия: `tau_d > tau_f`, `Uinc` высокое

**Пример** (депрессия):
```python
syn_stp = TsodyksMarkramSynapse(
    n_pre=1,
    n_post=2,
    dt=0.5,
    params={
        'gsyn_max': {'value': [[0.1, 0.1]], 'trainable': True},
        'tau_f':    {'value': 20.0,  'trainable': True, 'min': 6.0,  'max': 240.0},
        'tau_d':    {'value': 5.0,   'trainable': True, 'min': 2.0,  'max': 15.0},
        'tau_r':    {'value': 200.0, 'trainable': True, 'min': 91.0, 'max': 1300.0},
        'Uinc':     {'value': 0.2,   'trainable': True, 'min': 0.04, 'max': 0.7},
        'pconn':    {'value': [[1.0, 1.0]], 'trainable': False},
        'e_r':      {'value': 0.0,   'trainable': False},
    }
)
```

### Состояние

```python
state_size = [
    tf.TensorShape([n_pre, n_post]),  # R (ресурс)
    tf.TensorShape([n_pre, n_post]),  # U (использование)
    tf.TensorShape([n_pre, n_post]),  # A (активность)
]
```

### Начальное состояние

```python
[R=1.0, U=0.0, A=0.0]  # полностью восстановленный синапс
```

### ParameterSpec

```python
{
    'gsyn_max': {'shape': (n_pre, n_post), 'trainable': ..., 'constraint': 'NonNegConstraint', 'units': 'mS/cm^2'},
    'tau_f':    {'shape': (n_pre, n_post), 'trainable': ..., 'constraint': 'MinMaxConstraint', 'units': 'ms'},
    'tau_d':    {'shape': (n_pre, n_post), 'trainable': ..., 'constraint': 'MinMaxConstraint', 'units': 'ms'},
    'tau_r':    {'shape': (n_pre, n_post), 'trainable': ..., 'constraint': 'MinMaxConstraint', 'units': 'ms'},
    'Uinc':     {'shape': (n_pre, n_post), 'trainable': ..., 'constraint': 'MinMaxConstraint', 'units': 'dimensionless'},
    'pconn':    {'shape': (n_pre, n_post), 'trainable': False, 'constraint': 'UnitIntervalConstraint', 'units': 'dimensionless'},
    'e_r':      {'shape': (n_pre, n_post), 'trainable': False, 'constraint': None, 'units': 'mV'},
}
```

---

## CompositeSynapse

**Файл**: `neuraltide/synapses/composite.py`

Композитный синапс, объединяющий несколько синаптических компонент.

### Концепция

Токи от компонент суммируются:
```
I_syn_total = sum(I_syn_i for i in components)
g_syn_total = sum(g_syn_i for i in components)
```

### Конструктор

```python
CompositeSynapse(
    n_pre: int,
    n_post: int,
    dt: float,
    components: List[Tuple[str, SynapseModel]],
    **kwargs
)
```

### Параметры

| Параметр | Тип | Описание |
|----------|-----|-----------|
| `components` | List[Tuple[str, SynapseModel]] | Список кортежей (name, SynapseModel) |

### Пример: NMDA + Tsodyks-Markram

```python
syn_static = StaticSynapse(
    n_pre=1, n_post=2, dt=0.5,
    params={...}
)

syn_nmda = NMDASynapse(
    n_pre=1, n_post=2, dt=0.5,
    params={...}
)

syn_stp = TsodyksMarkramSynapse(
    n_pre=1, n_post=2, dt=0.5,
    params={...}
)

composite = CompositeSynapse(
    n_pre=1,
    n_post=2,
    dt=0.5,
    components=[
        ('static', syn_static),
        ('nmda', syn_nmda),
        ('stp', syn_stp),
    ]
)
```

### Состояние

Объединяет состояния всех компонент:
```python
state_size = static.state_size + nmda.state_size + stp.state_size
```

### ParameterSpec

```python
{
    'static_gsyn_max': {...},
    'static_pconn': {...},
    'nmda_gsyn_max_nmda': {...},
    'nmda_tau1_nmda': {...},
    # ... и т.д.
    'stp_gsyn_max': {...},
    'stp_tau_f': {...},
}
```

---

## Синаптическая динамика в сети

Синапсы обновляются на каждом временном шаге в `_step_fn`:

```python
# Для каждого синапса:
pre_rate = src_pop.get_firing_rate(src_state)  # [1, n_pre]
post_v = tgt_pop.observables(tgt_state)['v_mean']  # [1, n_post]

current_dict, new_syn_state = synapse.forward(
    pre_rate, post_v, syn_state, dt
)

# Накопление токов к целевой популяции:
syn_I[tgt] += current_dict['I_syn']
syn_g[tgt] += current_dict['g_syn']
```