# Keras-модель (neuraltide.model)

Keras-совместимая обёртка для NetworkRNN.

**Файл**: `neuraltide/model/__init__.py`

---

## Обзор

`BrainModelKeras` оборачивает `NetworkRNN` в стандартный интерфейс `tf.keras.Model`, позволяя использовать привычные методы `compile()` и `fit()`.

---

## BrainModelKeras

```python
class BrainModelKeras(tf.keras.Model):
```

### Конструктор

```python
BrainModelKeras(
    graph: NetworkGraph,
    integrator: BaseIntegrator,
    dt: float,
    loss_fn: Optional[BaseLoss] = None,
    stability_penalty_weight: float = 0.0,
    **kwargs
)
```

| Параметр | Тип | Описание |
|----------|-----|-----------|
| `graph` | NetworkGraph | Граф сети (с `declare_input` и `add_population/add_synapse`) |
| `integrator` | BaseIntegrator | Интегратор (EulerIntegrator, RK4Integrator и т.д.) |
| `dt` | float | Шаг интегрирования (мс) |
| `loss_fn` | Optional[BaseLoss] | Функция потерь |
| `stability_penalty_weight` | float | Вес штрафа за стабильность |

### Пример

```python
import numpy as np
import tensorflow as tf
from neuraltide.core.network import NetworkGraph
from neuraltide.populations import IzhikevichMeanField
from neuraltide.synapses import StaticSynapse
from neuraltide.inputs import SinusoidalGenerator
from neuraltide.integrators import RK4Integrator
from neuraltide.model import BrainModelKeras
from neuraltide.training import MSELoss

dt = 0.5

# Создание компонентов
gen = SinusoidalGenerator(dt=dt, params={
    'amplitude': 10.0, 'freq': 8.0, 'phase': 0.0, 'offset': 5.0,
})

pop = IzhikevichMeanField(dt=dt, params={
    'tau_pop': [1.0], 'alpha': [0.5], 'a': [0.02],
    'b': [0.2], 'w_jump': [0.1], 'Delta_I': [0.5], 'I_ext': [1.0],
})

syn = StaticSynapse(n_pre=1, n_post=1, dt=dt, params={
    'gsyn_max': 0.1, 'pconn': [[1.0]], 'e_r': 0.0,
})

# Построение графа
graph = NetworkGraph(dt=dt)
graph.declare_input('input', n_units=gen.n_units)
graph.add_population('exc', pop)
graph.add_synapse('input->exc', syn, src='input', tgt='exc')

# Целевые данные
T = 1000
t_values = np.arange(0, T, dt, dtype=np.float32)
target_rates = 10.0 + 5.0 * np.sin(2 * np.pi * 8.0 * t_values / 1000.0)
target = {
    'exc': tf.constant(target_rates[None, :, None], dtype=tf.float32)
}

# Входные данные
t_seq = tf.constant(t_values[None, :, None])
inputs = graph.pack_inputs({'input': gen(t_seq)})

# Создание Keras-модели
model = BrainModelKeras(
    graph, RK4Integrator(), dt=dt,
    loss_fn=MSELoss(target),
    stability_penalty_weight=1e-3,
)

# Обучение
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
history = model.fit(inputs, target, epochs=100, verbose=1)

# Предсказание
output = model(inputs, training=False)
print(output.firing_rates['exc'].shape)  # [1, T, 1]
```

---

## Методы

### `call(inputs, training=False, initial_state=None)`

Forward pass.

**Аргументы**:
- `inputs`: tf.Tensor `[T, total_input_units]` или `[batch, T, total_input_units]` — firing rates входов. NetworkRNN поддерживает batch=1.
- `training`: bool — флаг обучения
- `initial_state`: Optional[Tuple[StateList, StateList]] — начальное состояние

**Возвращает**: `NetworkOutput`

```python
output = model(inputs, training=False)
# output.firing_rates  — Dict[str, Tensor]
# output.stability_loss — Tensor (скаляр)
```

### `train_step(data)`

Один шаг обучения (Keras convention).

**Аргументы**:
- `data`: tuple `(x, y)` или просто `x`

**Возвращает**: `Dict[str, Tensor]` с ключом `'loss'`

### `test_step(data)`

Один шаг валидации.

**Аргументы**:
- `data`: tuple `(x, y)` или просто `x`

**Возвращает**: `Dict[str, Tensor]` с ключом `'loss'`

---

## Свойства

### `network`

Доступ к внутреннему `NetworkRNN`.

```python
model.network  # NetworkRNN
```

### `trainable_variables`

Обучаемые переменные сети.

```python
model.trainable_variables  # List[tf.Variable]
```

---

## Использование с Keras API

### Callbacks

```python
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

callbacks = [
    EarlyStopping(patience=20, restore_best_weights=True),
    ModelCheckpoint('best_model.keras', save_best_only=True),
]

history = model.fit(
    inputs, target,
    epochs=500,
    batch_size=1,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1,
)
```

### Экспорт модели

```python
# Сохранение весов
model.save_weights('model_weights.keras')

# Загрузка весов
model.load_weights('model_weights.keras')
```

### Доступ к NetworkRNN после обучения

```python
# Получение обученных параметров
for var in model.network.trainable_variables:
    print(f"{var.name}: {var.numpy()}")

# Запуск симуляции с пользовательским начальным состоянием
init_pop, init_syn = model.network.get_initial_state()
output = model.network(t_seq, inputs=inputs, initial_state=(init_pop, init_syn))
```

---

## Сравнение с Trainer

| Особенность | BrainModelKeras | Trainer |
|-------------|-----------------|---------|
| Интерфейс | Keras `model.fit()` | `trainer.fit()` |
| Callbacks | Keras callbacks | NeuralTide callbacks |
| Методы градиентов | Только BPTT | BPTT + Adjoint |
| Экспорт результатов | Нет | `export_results()` |
| Рекомендуется для | Интеграция с Keras экосистемой | Полный контроль обучения |
