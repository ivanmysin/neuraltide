# Данные (neuraltide.data)

Модуль для работы с данными: сохранение, загрузка и визуализация в формате HDF5.

**Файл**: `neuraltide/data/__init__.py`

---

## Обзор

`neuraltide.data` предоставляет удобный интерфейс для сохранения и загрузки входных данных и целевых firing rates. Формат HDF5 обеспечивает эффективное хранение больших массивов данных.

### Структура HDF5 файла

```
data.h5
├── inputs              [T, total_input_units]
├── target              [T, total_target_units]
├── time_seq            [T]
└── metadata
    ├── dt              float
    ├── input_names     JSON list
    ├── input_n_units   JSON dict
    ├── target_names    JSON list
    ├── target_n_units  JSON dict
    └── generator_params JSON dict
```

---

## Dataset

Контейнер для данных обучения.

```python
@dataclass
class Dataset:
    inputs: np.ndarray          # [T, total_input_units]
    target: np.ndarray          # [T, total_target_units]
    time_seq: np.ndarray        # [T]
    dt: float
    input_names: List[str]
    input_n_units: Dict[str, int]
    target_names: List[str]
    target_n_units: Dict[str, int]
    generator_params: Dict[str, Any] = field(default_factory=dict)
```

### Свойства

| Свойство | Тип | Описание |
|----------|-----|-----------|
| `T` | int | Число временных шагов |
| `total_input_units` | int | Общее число входных каналов |
| `total_target_units` | int | Общее число целевых каналов |

### Методы

#### `input_slice(name)`

Возвращает срез входа по имени.

```python
data.input_slice('theta')  # [T, n_units_theta]
```

#### `target_slice(name)`

Возвращает срез цели по имени.

```python
data.target_slice('exc')  # [T, n_units_exc]
```

---

## save_dataset

Сохраняет dataset в HDF5 (.h5).

```python
def save_dataset(
    path: str,
    inputs: Dict[str, np.ndarray],
    target: Dict[str, np.ndarray],
    dt: float,
    generator_params: Optional[Dict[str, Any]] = None,
) -> None:
```

### Аргументы

| Параметр | Тип | Описание |
|----------|-----|-----------|
| `path` | str | Путь к .h5 файлу |
| `inputs` | Dict[str, np.ndarray] | `{name: array[T, n_units_i]}` — firing rates входов |
| `target` | Dict[str, np.ndarray] | `{name: array[T, n_units_i]}` — целевые firing rates |
| `dt` | float | Шаг интегрирования (мс) |
| `generator_params` | Optional[Dict] | Параметры генераторов (для reproducibility) |

### Пример

```python
import numpy as np
from neuraltide.data import save_dataset

# Генерация данных
T = 10000
dt = 0.1
t = np.arange(T) * dt

# Входные данные
theta_rate = 10.0 + 5.0 * np.sin(2 * np.pi * 8.0 * t / 1000.0)
place_rate = np.where(
    np.sqrt((t % 500 - 250)**2) < 100,
    20.0,
    2.0
)

# Целевые данные
exc_rate = 0.5 * theta_rate + 0.3 * place_rate

save_dataset(
    path='training_data.h5',
    inputs={
        'theta': theta_rate[:, np.newaxis],
        'place': place_rate[:, np.newaxis],
    },
    target={
        'exc': exc_rate[:, np.newaxis],
    },
    dt=dt,
    generator_params={
        'theta': {'freq': 8.0, 'amplitude': 5.0},
        'place': {'center': 250.0, 'width': 100.0},
    }
)
```

---

## load_dataset

Загружает dataset из HDF5.

```python
def load_dataset(path: str) -> Dataset:
```

### Аргументы

| Параметр | Тип | Описание |
|----------|-----|-----------|
| `path` | str | Путь к .h5 файлу |

### Возвращает

`Dataset` с inputs, target и metadata.

### Пример

```python
from neuraltide.data import load_dataset

data = load_dataset('training_data.h5')

print(f"T = {data.T}")
print(f"dt = {data.dt}")
print(f"Inputs: {data.input_names}")
print(f"Targets: {data.target_names}")

# Доступ к данным
theta_data = data.input_slice('theta')  # [T, 1]
exc_data = data.target_slice('exc')     # [T, 1]
```

---

## plot_dataset

Строит графики для визуальной проверки данных.

```python
def plot_dataset(
    data: Dataset,
    max_t: Optional[float] = None,
    figsize: tuple = (14, 8),
) -> None:
```

### Аргументы

| Параметр | Тип | Описание |
|----------|-----|-----------|
| `data` | Dataset | Датасет для отображения |
| `max_t` | Optional[float] | Максимальное время для отображения (мс). Если None — всё. |
| `figsize` | tuple | Размер фигуры |

### Пример

```python
from neuraltide.data import load_dataset, plot_dataset

data = load_dataset('training_data.h5')
plot_dataset(data, max_t=500)  # Показать первые 500 мс
```

---

## Интеграция с обучением

### Подготовка данных для Trainer

```python
import numpy as np
import tensorflow as tf
from neuraltide.data import load_dataset

data = load_dataset('training_data.h5')

# Конвертация в тензоры
t_seq = tf.constant(data.time_seq[None, :, None], dtype=tf.float32)
inputs = tf.constant(data.inputs[None, :, :], dtype=tf.float32)
target = {name: tf.constant(data.target_slice(name)[None, :, :], dtype=tf.float32)
          for name in data.target_names}

# Обучение
trainer = Trainer(network, loss_fn, optimizer)
history = trainer.fit(t_seq, inputs=inputs, target=target, epochs=100)
```

### Сохранение результатов симуляции

```python
from neuraltide.data import save_dataset

# Запуск симуляции
output = network(t_seq, inputs=inputs)

# Сохранение результатов
save_dataset(
    path='simulation_results.h5',
    inputs={'input': inputs.numpy()[0]},
    target={name: rate.numpy()[0] for name, rate in output.firing_rates.items()},
    dt=network._graph.dt,
)
```
