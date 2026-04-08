# Ограничения (Constraints)

NeuralTide предоставляет систему ограничений для параметров моделей. Ограничения применяются к весам слоёв TensorFlow во время обучения и могут использоваться для:

- Ограничения диапазона значений (MinMaxConstraint)
- Ограничения неотрицательности (NonNegConstraint)
- Ограничения на единичный интервал (UnitIntervalConstraint)

---

## MinMaxConstraint

**Файл**: `neuraltide/constraints/param_constraints.py`

Ограничение параметра на отрезок `[min_val, max_val]`.

### Класс

```python
class MinMaxConstraint(tf.keras.constraints.Constraint):
    def __init__(self, min_val: Optional[float], max_val: Optional[float])
```

### Параметры

| Параметр | Тип | Описание |
|----------|-----|-----------|
| `min_val` | Optional[float] | Минимальное значение (включительно) |
| `max_val` | Optional[float] | Максимальное значение (включительно) |

### Реализация

```python
def __call__(self, w: tf.Tensor) -> tf.Tensor:
    if self.min_val is not None and self.max_val is not None:
        return tf.clip_by_value(w, self.min_val, self.max_val)
    elif self.min_val is not None:
        return tf.maximum(w, self.min_val)
    elif self.max_val is not None:
        return tf.minimum(w, self.max_val)
    return w
```

### Примеры использования

**Ограничение диапазона [0, 1]**:
```python
params = {
    'rate': {
        'value': 0.5,
        'trainable': True,
        'min': 0.0,
        'max': 1.0,
    }
}
```

**Только нижняя граница**:
```python
params = {
    'gsyn_max': {
        'value': 0.1,
        'trainable': True,
        'min': 0.0,  # только min, max не ограничен
    }
}
```

**Только верхняя граница**:
```python
params = {
    'Delta_I': {
        'value': 0.5,
        'trainable': True,
        'max': 2.0,  # только max, min не ограничен
    }
}
```

### Использование с популяциями

```python
pop = IzhikevichMeanField(dt=0.5, params={
    'Delta_I': {'value': [0.5, 0.6], 'trainable': True, 'min': 0.01, 'max': 2.0},
    'I_ext': {'value': [1.0, 1.2], 'trainable': True, 'min': -2.0, 'max': 2.0},
})
```

### Использование с синапсами

```python
syn = TsodyksMarkramSynapse(n_pre=1, n_post=2, dt=0.5, params={
    'gsyn_max': {'value': 0.1, 'trainable': True},
    'tau_f': {'value': 20.0, 'trainable': True, 'min': 6.0, 'max': 240.0},
    'tau_d': {'value': 5.0, 'trainable': True, 'min': 2.0, 'max': 15.0},
    'tau_r': {'value': 200.0, 'trainable': True, 'min': 91.0, 'max': 1300.0},
    'Uinc': {'value': 0.2, 'trainable': True, 'min': 0.04, 'max': 0.7},
    'pconn': {'value': [[1.0, 1.0]], 'trainable': False},
    'e_r': {'value': 0.0, 'trainable': False},
})
```

---

## NonNegConstraint

**Файл**: `neuraltide/constraints/param_constraints.py`

Ограничение: неотрицательные значения (через ReLU).

### Класс

```python
class NonNegConstraint(tf.keras.constraints.Constraint):
    def __call__(self, w: tf.Tensor) -> tf.Tensor:
        return tf.nn.relu(w)
```

### Когда использовать

- Максимальная проводимость `gsyn_max`
- Постоянные времени `tau_*` (всегда положительные)
- Любые параметры, которые должны быть >= 0

### Пример

```python
syn = StaticSynapse(n_pre=1, n_post=2, dt=0.5, params={
    'gsyn_max': {'value': 0.1, 'trainable': True},  # автоматически NonNegConstraint
    'pconn': {'value': [[1.0, 1.0]], 'trainable': False},
    'e_r': {'value': 0.0, 'trainable': False},
})
```

При этом ограничение `gsyn_max` автоматически применяется через `parameter_spec`.

---

## UnitIntervalConstraint

**Файл**: `neuraltide/constraints/param_constraints.py`

Ограничение на отрезок `[0, 1]`. Эквивалентна `MinMaxConstraint(0, 1)`.

### Класс

```python
class UnitIntervalConstraint(tf.keras.constraints.Constraint):
    def __init__(self):
        self._inner = MinMaxConstraint(0.0, 1.0)
    
    def __call__(self, w: tf.Tensor) -> tf.Tensor:
        return self._inner(w)
```

### Когда использовать

- Матрица соединений `pconn` (вероятности от 0 до 1)
- Любые вероятности или нормированные величины

### Пример

```python
syn = NMDASynapse(n_pre=1, n_post=2, dt=0.5, params={
    # ...
    'pconn_nmda': {'value': [[1.0, 1.0]], 'trainable': False},  # UnitIntervalConstraint
})
```

---

## Ограничения в parameter_spec

Каждый параметр в `parameter_spec` содержит информацию об ограничении:

```python
{
    'gsyn_max': {
        'shape': (n_pre, n_post),
        'trainable': True,
        'constraint': 'NonNegConstraint',  # имя класса ограничения
        'units': 'mS/cm^2',
    },
    'pconn': {
        'shape': (n_pre, n_post),
        'trainable': False,
        'constraint': 'UnitIntervalConstraint',
        'units': 'dimensionless',
    },
}
```

### Получение имени ограничения

```python
def _get_constraint_name(self, var: tf.Variable) -> Optional[str]:
    if var.constraint is not None:
        return var.constraint.__class__.__name__
    return None
```

---

## Создание собственного ограничения

### Пример: положительное с максимумом

```python
class PositiveMaxConstraint(tf.keras.constraints.Constraint):
    def __init__(self, max_val: float):
        self.max_val = max_val
    
    def __call__(self, w: tf.Tensor) -> tf.Tensor:
        return tf.clip_by_value(w, 0.0, self.max_val)
```

### Пример: ограничение на нормацию

```python
class UnitNormConstraint(tf.keras.constraints.Constraint):
    def __call__(self, w: tf.Tensor) -> tf.Tensor:
        return tf.math.l2_normalize(w, axis=-1)
```

### Пример: симметричное ограничение

```python
class SymmetricConstraint(tf.keras.constraints.Constraint):
    def __call__(self, w: tf.Tensor) -> tf.Tensor:
        # Обеспечение симметрии матрицы
        return 0.5 * (w + tf.transpose(w))
```

---

## Таблица ограничений по типам параметров

| Параметр | Рекомендуемое ограничение | Причина |
|----------|--------------------------|---------|
| `gsyn_max` | NonNegConstraint | Проводимость ≥ 0 |
| `pconn` | UnitIntervalConstraint | Вероятность 0-1 |
| `tau_*` | MinMaxConstraint(0, None) | Время > 0 |
| `Uinc` | UnitIntervalConstraint | Вероятность 0-1 |
| `e_r` | None | Потенциал реверса произвольный |
| `Delta_I` | MinMaxConstraint(0, None) | Спред тока ≥ 0 |
| `I_ext` | None | Ток может быть отрицательным |

---

## Отладка ограничений

### Проверка применённых ограничений

```python
# После создания модели
pop = IzhikevichMeanField(dt=0.5, params={...})

for var in pop.variables:
    print(f"{var.name}: constraint = {var.constraint}")
```

### Визуализация значений параметров

```python
import numpy as np

# Получение значений
values = pop.Delta_I.numpy()
print(f"Delta_I values: {values}")
print(f"Delta_I constraint: {pop.Delta_I.constraint}")
```

### Проверка нарушений

```python
def check_constraints(network):
    for var in network.trainable_variables:
        if var.constraint is not None:
            constrained = var.constraint(var.value())
            diff = tf.reduce_max(tf.abs(constrained - var.value()))
            if diff > 1e-6:
                print(f"Warning: {var.name} violates constraint")
                print(f"  Original: {var.value()}")
                print(f"  Constrained: {constrained}")

check_constraints(network)
```