# Обучение нейронных сетей

NeuralTide поддерживает два метода вычисления градиентов для обучения:

1. **BPTT** (Backpropagation Through Time) — стандартный метод через TensorFlow GradientTape
2. **Adjoint** ( метод сопряжённых состояний) — метод с низким потреблением памяти

По умолчанию используется BPTT. Для переключения на adjoint см. раздел [Метод сопряжённых состояний](#метод-сопряжённых-состояний).

---

## Trainer

Обучение сети выполняется через класс `Trainer`:

```python
from neuraltide.training import Trainer, CompositeLoss, MSELoss, StabilityPenalty

loss_fn = CompositeLoss([
    (1.0, MSELoss(target)),
    (1e-3, StabilityPenalty()),
])
trainer = Trainer(network, loss_fn, optimizer=tf.keras.optimizers.Adam(1e-3))
history = trainer.fit(t_seq, epochs=epochs, verbose=1)
```

### Параметры Trainer

| Параметр | Тип | Описание |
|----------|-----|----------|
| `network` | NetworkRNN | Обучаемая сеть |
| `loss_fn` | BaseLoss | Функция потерь |
| `optimizer` | tf.optimizers | Оптимизатор TensorFlow |
| `grad_method` | str | Метод градиентов: `"bptt"` (по умолчанию) или `"adjoint"` |

### Методы Trainer

- `fit(t_seq, epochs, verbose)` — обучение сети
- `evaluate(t_seq)` — вычисление loss без обновления весов

---

## Функции потерь

### MSELoss

Среднеквадратичная ошибка между предсказанием и целью:

```python
from neuraltide.training import MSELoss

loss = MSELoss(target)
```

### StabilityPenalty

Штраф за численную нестабильность (NaN, Inf, слишком большие значения):

```python
from neuraltide.training import StabilityPenalty

loss = StabilityPenalty()
```

### CompositeLoss

Комбинация нескольких функций потерь с весами:

```python
from neuraltide.training import CompositeLoss, MSELoss, StabilityPenalty

loss_fn = CompositeLoss([
    (1.0, MSELoss(target)),      # основная ошибка
    (1e-3, StabilityPenalty()),  # штраф за нестабильность
])
```

---

## Оптимизаторы

Используются стандартные оптимизаторы TensorFlow:

```python
optimizer = tf.keras.optimizers.Adam(1e-3)
optimizer = tf.keras.optimizers.SGD(1e-2, momentum=0.9)
optimizer = tf.keras.optimizers.RMSprop(1e-3)
```

---

## Колбэки

### DivergenceDetector

Останавливает обучение при расходимости:

```python
from neuraltide.training import DivergenceDetector

callbacks = [DivergenceDetector(threshold=1e6)]
trainer.fit(t_seq, epochs=500, callbacks=callbacks)
```

### GradientMonitor

Логирует градиенты:

```python
from neuraltide.training import GradientMonitor

callbacks = [GradientMonitor(every_n_steps=50)]
```

### ExperimentLogger

Сохраняет историю обучения:

```python
from neuraltide.training import ExperimentLogger

callbacks = [ExperimentLogger(log_dir='./logs')]
```

---

## Метод сопряжённых состояний

Метод сопряжённых состояний (adjoint state method) — это техника вычисления градиентов, которая требует значительно меньше памяти чем BPTT.

### Зачем нужен adjoint?

BPTT хранит все состояния сети на всех временных шагах для вычисления градиентов через backpropagation. Это требует O(T) памяти, где T — число шагов.

Adjoint вычисляет градиенты через backward pass, аналогично методу сопряжённых состояний в численной оптимизации. Требует O(1) дополнительной памяти.

### Когда использовать adjoint?

- **Длинные временные последовательности** (T > 500)
- **Ограниченная видеопамять** (GPU)
- **Большие модели** с many parameters

### benchmarks

| T | BPTT RAM | Adjoint RAM | Уменьшение |
|---|----------|------------|-------------|
| 100 | 15 MB | 2 MB | 7.5x |
| 500 | 72 MB | 3 MB | 24x |
| 1000 | 144 MB | 4 MB | 36x |

Время вычисления adjoint немного больше, чем BPTT, из-за Python loop.

### Использование adjoint

```python
from neuraltide.training import Trainer

trainer = Trainer(
    network,
    loss_fn,
    optimizer=tf.keras.optimizers.Adam(1e-3),
    grad_method='adjoint'
)

history = trainer.fit(t_seq, epochs=100)
```

### Сравнение градиентов

Для верификации можно сравнить градиенты adjoint и BPTT:

```python
from neuraltide.training.adjoint import compute_gradients as adjoint_grads
from neuraltide.training.trainer import compute_gradients as bptt_grads

grads_adjoint = adjoint_grads(network, t_seq, loss_fn)
grads_bptt = bptt_grads(network, t_seq, loss_fn)

# Максимальная разница ~1e-7
for (name_a, g_a), (name_b, g_b) in zip(grads_adjoint, grads_bptt):
    diff = tf.abs(g_a - g_b).numpy().max()
    print(f"{name}: {diff:.2e}")
```

---

## Пример: Полное обучение с adjoint

```python
import numpy as np
import tensorflow as tf
from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.populations import IzhikevichMeanField
from neuraltide.synapses import TsodyksMarkramSynapse
from neuraltide.inputs import VonMisesGenerator
from neuraltide.integrators import RK4Integrator
from neuraltide.training import Trainer, CompositeLoss, MSELoss, StabilityPenalty

dt = 0.05
T = 50
epochs = 200

# Создание сети
gen = VonMisesGenerator(dt=dt, params={
    'mean_rate': 20.0, 'R': 0.5, 'freq': 8.0, 'phase': 0.0,
})

pop = IzhikevichMeanField(dt=dt, params={
    'tau_pop': [1.0, 1.0],
    'alpha': [0.5, 0.5],
    'a': [0.02, 0.02],
    'b': [0.2, 0.2],
    'w_jump': [0.1, 0.1],
    'Delta_I': {'value': [0.5, 0.6], 'trainable': True, 'min': 0.01, 'max': 2.0},
    'I_ext': {'value': [0.1, 0.2], 'trainable': True, 'min': -2.0, 'max': 2.0},
})

syn = TsodyksMarkramSynapse(n_pre=1, n_post=2, dt=dt, params={
    'gsyn_max': {'value': [[0.1, 0.1]], 'trainable': True},
    'tau_f': {'value': 20.0, 'trainable': True, 'min': 6.0, 'max': 240.0},
    'tau_d': {'value': 5.0, 'trainable': True, 'min': 2.0, 'max': 15.0},
    'tau_r': {'value': 200.0, 'trainable': True, 'min': 91.0, 'max': 1300.0},
    'Uinc': {'value': 0.2, 'trainable': True, 'min': 0.04, 'max': 0.7},
    'pconn': {'value': [[1.0, 1.0]], 'trainable': False},
    'e_r': {'value': 0.0, 'trainable': False},
})

graph = NetworkGraph(dt=dt)
graph.add_input_population('theta', gen)
graph.add_population('exc', pop)
graph.add_synapse('theta->exc', syn, src='theta', tgt='exc')

network = NetworkRNN(graph, integrator=RK4Integrator(), stability_penalty_weight=1e-3)

# Целевые данные
t_values = np.arange(T, dtype=np.float32) * dt
target_0 = 10.0 + 5.0 * np.sin(2*np.pi*8.0*t_values/1000.0)
target_1 = 8.0 + 4.0 * np.sin(2*np.pi*8.0*t_values/1000.0 + 0.5)
target = {
    'exc': tf.constant(np.stack([target_0, target_1], axis=-1)[None, :, :])
}

t_seq = tf.constant(t_values[None, :, None])

# Обучение с adjoint
loss_fn = CompositeLoss([
    (1.0, MSELoss(target)),
    (1e-3, StabilityPenalty()),
])
trainer = Trainer(network, loss_fn, optimizer=tf.keras.optimizers.Adam(1e-3), grad_method='adjoint')
history = trainer.fit(t_seq, epochs=epochs, verbose=50)

print(f"Initial loss: {history.loss_history[0]:.4f}")
print(f"Final loss: {history.loss_history[-1]:.4f}")
```

---

## Ограничения

1. **Adjoint работает только для моделей с дискретным временем** — непрерывные методы (CTNG) не поддерживаются
2. **Точность градиентов ~1e-7** — достаточна для обучения, но не для теоретических исследований
3. **Медленнее BPTT** — из-за Python loop, рекомендуется для длинных последовательностей где память важнее скорости