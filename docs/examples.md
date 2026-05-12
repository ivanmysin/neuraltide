# Примеры использования

В этом разделе приведены типичные сценарии использования NeuralTide.

---

## Пример 1: Простая сеть — один вход, одна популяция

```python
import numpy as np
import tensorflow as tf
from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.populations import IzhikevichMeanField
from neuraltide.synapses import StaticSynapse
from neuraltide.inputs import SinusoidalGenerator
from neuraltide.integrators import RK4Integrator
from neuraltide.utils import print_summary

# Параметры
dt = 0.5
T = 1000  # мс

# Создание генератора
gen = SinusoidalGenerator(
    dt=dt,
    params={
        'amplitude': 10.0,
        'freq': 8.0,
        'phase': 0.0,
        'offset': 5.0,
    }
)

# Создание популяции
pop = IzhikevichMeanField(
    dt=dt,
    params={
        'tau_pop':   [1.0],
        'alpha':     [0.5],
        'a':         [0.02],
        'b':         [0.2],
        'w_jump':    [0.1],
        'Delta_I':   [0.5],
        'I_ext':     [1.0],
    }
)

# Создание синапса
syn = StaticSynapse(
    n_pre=1,
    n_post=1,
    dt=dt,
    params={
        'gsyn_max': 0.1,
        'pconn': [[1.0]],
        'e_r': 0.0,
    }
)

# Построение графа
graph = NetworkGraph(dt=dt)
graph.add_input_population('input', gen)
graph.add_population('exc', pop)
graph.add_synapse('input->exc', syn, src='input', tgt='exc')

# Создание сети
network = NetworkRNN(graph, integrator=RK4Integrator())
print_summary(network)

# Симуляция
t_values = np.arange(0, T, dt, dtype=np.float32)
t_seq = tf.constant(t_values[None, :, None])

output = network(t_seq)
print(f"Output shape: {output.firing_rates['exc'].shape}")
# (1, 2000, 1)
```

---

## Пример 2: Две популяции (E-I) с синапсами

```python
import numpy as np
import tensorflow as tf
from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.populations import IzhikevichMeanField
from neuraltide.synapses import StaticSynapse
from neuraltide.inputs import ConstantRateGenerator
from neuraltide.integrators import RK4Integrator

dt = 0.5
T = 1000

# Входной генератор (постоянная частота)
gen = ConstantRateGenerator(dt=dt, params={'rate': 20.0})

# Возбуждающая популяция (2 единицы)
pop_exc = IzhikevichMeanField(dt=dt, params={
    'tau_pop':   [1.0, 1.0],
    'alpha':     [0.5, 0.5],
    'a':         [0.02, 0.02],
    'b':         [0.2, 0.2],
    'w_jump':    [0.1, 0.1],
    'Delta_I':   [0.5, 0.6],
    'I_ext':     [1.0, 1.2],
})

# Тормозная популяция (2 единицы)
pop_inh = IzhikevichMeanField(dt=dt, params={
    'tau_pop':   [1.0, 1.0],
    'alpha':     [0.5, 0.5],
    'a':         [0.02, 0.02],
    'b':         [0.2, 0.2],
    'w_jump':    [0.1, 0.1],
    'Delta_I':   [0.5, 0.5],
    'I_ext':     [0.3, 0.3],  # меньший внешний ток
})

# Синапс: вход -> возбуждающая
syn_in_exc = StaticSynapse(n_pre=1, n_post=2, dt=dt, params={
    'gsyn_max': 0.1,
    'pconn': [[1.0, 1.0]],
    'e_r': 0.0,
})

# Синапс: возбуждающая -> тормозная
syn_exc_inh = StaticSynapse(n_pre=2, n_post=2, dt=dt, params={
    'gsyn_max': 0.08,
    'pconn': [[1.0, 1.0], [1.0, 1.0]],
    'e_r': 0.0,
})

# Синапс: тормозная -> возбуждающая (отрицательная проводимость)
syn_inh_exc = StaticSynapse(n_pre=2, n_post=2, dt=dt, params={
    'gsyn_max': -0.3,
    'pconn': [[1.0, 1.0], [1.0, 1.0]],
    'e_r': -70.0,  # тормозный потенциал реверса
})

# Построение графа
graph = NetworkGraph(dt=dt)
graph.add_input_population('input', gen)
graph.add_population('exc', pop_exc)
graph.add_population('inh', pop_inh)
graph.add_synapse('input->exc', syn_in_exc, src='input', tgt='exc')
graph.add_synapse('exc->inh', syn_exc_inh, src='exc', tgt='inh')
graph.add_synapse('inh->exc', syn_inh_exc, src='inh', tgt='exc')

network = NetworkRNN(graph, integrator=RK4Integrator())

# Симуляция
t_values = np.arange(0, T, dt, dtype=np.float32)
t_seq = tf.constant(t_values[None, :, None])
output = network(t_seq)

print(f"Exc rates shape: {output.firing_rates['exc'].shape}")
print(f"Inh rates shape: {output.firing_rates['inh'].shape}")
```

---

## Пример 3: VonMises генератор + Tsodyks-Markram синапс

```python
import numpy as np
import tensorflow as tf
from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.populations import IzhikevichMeanField
from neuraltide.synapses import TsodyksMarkramSynapse
from neuraltide.inputs import VonMisesGenerator
from neuraltide.integrators import RK4Integrator

dt = 0.05
T = 500

# VonMises генератор (тета-ритм)
gen = VonMisesGenerator(
    dt=dt,
    params={
        'mean_rate': 20.0,
        'R': 0.5,
        'freq': 8.0,
        'phase': 0.0,
    }
)

# Популяция
pop = IzhikevichMeanField(dt=dt, params={
    'tau_pop':   [1.0, 1.0],
    'alpha':     [0.5, 0.5],
    'a':         [0.02, 0.02],
    'b':         [0.2, 0.2],
    'w_jump':    [0.1, 0.1],
    'Delta_I':   [0.5, 0.6],
    'I_ext':     [0.1, 0.2],
})

# Tsodyks-Markram синапс с кратковременной пластичностью
syn = TsodyksMarkramSynapse(
    n_pre=1,
    n_post=2,
    dt=dt,
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

graph = NetworkGraph(dt=dt)
graph.add_input_population('theta', gen)
graph.add_population('exc', pop)
graph.add_synapse('theta->exc', syn, src='theta', tgt='exc')

network = NetworkRNN(graph, integrator=RK4Integrator())

# Симуляция
t_values = np.arange(0, T, dt, dtype=np.float32)
t_seq = tf.constant(t_values[None, :, None])
output = network(t_seq)

# Результат
rates = output.firing_rates['exc'].numpy()
print(f"Final rates: {rates[0, -1, :]}")
```

---

## Пример 4: NMDA синапс

```python
import numpy as np
import tensorflow as tf
from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.populations import IzhikevichMeanField
from neuraltide.synapses import NMDASynapse
from neuraltide.inputs import SinusoidalGenerator
from neuraltide.integrators import RK4Integrator

dt = 0.5

gen = SinusoidalGenerator(dt=dt, params={
    'amplitude': 15.0, 'freq': 5.0, 'phase': 0.0, 'offset': 5.0,
})

pop = IzhikevichMeanField(dt=dt, params={
    'tau_pop': [1.0], 'alpha': [0.5], 'a': [0.02],
    'b': [0.2], 'w_jump': [0.1], 'Delta_I': [0.5], 'I_ext': [1.0],
})

# NMDA синапс
syn_nmda = NMDASynapse(
    n_pre=1,
    n_post=1,
    dt=dt,
    params={
        'gsyn_max_nmda': {'value': 0.15, 'trainable': True},
        'tau1_nmda':     {'value': 5.0, 'trainable': False},
        'tau2_nmda':     {'value': 150.0, 'trainable': False},
        'Mgb':           {'value': 1.0, 'trainable': False},
        'av_nmda':       {'value': 0.08, 'trainable': False},
        'pconn_nmda':    {'value': [[1.0]], 'trainable': False},
        'e_r_nmda':      {'value': 0.0, 'trainable': False},
        'v_ref':         {'value': 0.0, 'trainable': False},
    }
)

graph = NetworkGraph(dt=dt)
graph.add_input_population('input', gen)
graph.add_population('exc', pop)
graph.add_synapse('input->exc', syn_nmda, src='input', tgt='exc')

network = NetworkRNN(graph, integrator=RK4Integrator())

# Симуляция
t_values = np.arange(0, 1000, dt, dtype=np.float32)
t_seq = tf.constant(t_values[None, :, None])
output = network(t_seq)

print(f"NMDA output shape: {output.firing_rates['exc'].shape}")
```

---

## Пример 5: Composite синапс (NMDA + STP)

```python
import numpy as np
import tensorflow as tf
from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.populations import IzhikevichMeanField
from neuraltide.synapses import NMDASynapse, TsodyksMarkramSynapse, CompositeSynapse
from neuraltide.inputs import ConstantRateGenerator
from neuraltide.integrators import RK4Integrator

dt = 0.5

gen = ConstantRateGenerator(dt=dt, params={'rate': 20.0})

pop = IzhikevichMeanField(dt=dt, params={
    'tau_pop': [1.0], 'alpha': [0.5], 'a': [0.02],
    'b': [0.2], 'w_jump': [0.1], 'Delta_I': [0.5], 'I_ext': [1.0],
})

# Компоненты композитного синапса
syn_nmda = NMDASynapse(
    n_pre=1, n_post=1, dt=dt,
    params={
        'gsyn_max_nmda': 0.1, 'tau1_nmda': 5.0, 'tau2_nmda': 150.0,
        'Mgb': 1.0, 'av_nmda': 0.08, 'pconn_nmda': [[1.0]],
        'e_r_nmda': 0.0, 'v_ref': 0.0,
    }
)

syn_stp = TsodyksMarkramSynapse(
    n_pre=1, n_post=1, dt=dt,
    params={
        'gsyn_max': 0.05, 'tau_f': 20.0, 'tau_d': 5.0,
        'tau_r': 200.0, 'Uinc': 0.2, 'pconn': [[1.0]], 'e_r': 0.0,
    }
)

# Композитный синапс
composite = CompositeSynapse(
    n_pre=1,
    n_post=1,
    dt=dt,
    components=[
        ('nmda', syn_nmda),
        ('stp', syn_stp),
    ]
)

graph = NetworkGraph(dt=dt)
graph.add_input_population('input', gen)
graph.add_population('exc', pop)
graph.add_synapse('input->exc', composite, src='input', tgt='exc')

network = NetworkRNN(graph, integrator=RK4Integrator())

t_values = np.arange(0, 1000, dt, dtype=np.float32)
t_seq = tf.constant(t_values[None, :, None])
output = network(t_seq)

print(f"Composite output shape: {output.firing_rates['exc'].shape}")
```

---

## Пример 6: Обучение сети (fit)

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
T = 20
epochs = 500

# Сеть
gen = VonMisesGenerator(dt=dt, params={
    'mean_rate': 20.0, 'R': 0.5, 'freq': 8.0, 'phase': 0.0,
})

pop = IzhikevichMeanField(dt=dt, params={
    'tau_pop': [1.0, 1.0], 'alpha': [0.5, 0.5], 'a': [0.02, 0.02],
    'b': [0.2, 0.2], 'w_jump': [0.1, 0.1],
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
target_0 = 10.0 + 5.0*np.sin(2*np.pi*8.0*t_values/1000.0)
target_1 = 8.0 + 4.0*np.sin(2*np.pi*8.0*t_values/1000.0 + 0.5)
target = {
    'exc': tf.constant(np.stack([target_0, target_1], axis=-1)[None, :, :], dtype=tf.float32)
}

t_seq = tf.constant(t_values[None, :, None])

# До обучения
output_before = network(t_seq, training=False)

# Обучение
loss_fn = CompositeLoss([
    (1.0, MSELoss(target)),
    (1e-3, StabilityPenalty()),
])
trainer = Trainer(network, loss_fn, optimizer=tf.keras.optimizers.Adam(1e-3))
history = trainer.fit(t_seq, epochs=epochs, verbose=1)

# После обучения
output_after = network(t_seq, training=False)

print(f"Initial loss: {history.loss_history[0]:.4f}")
print(f"Final loss: {history.loss_history[-1]:.4f}")
```

---

## Пример 7: Wilson-Cowan популяция

```python
import numpy as np
import tensorflow as tf
from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.populations import WilsonCowan
from neuraltide.synapses import StaticSynapse
from neuraltide.inputs import ConstantRateGenerator
from neuraltide.integrators import RK4Integrator

dt = 1.0
T = 500

# Wilson-Cowan: E и I в одном объекте
pop = WilsonCowan(
    n_units=1,
    dt=dt,
    params={
        'tau_E': [10.0],
        'tau_I': [10.0],
        'a_E': [1.0],
        'a_I': [1.0],
        'theta_E': [0.0],
        'theta_I': [0.0],
        'w_EE': [1.5],
        'w_IE': [1.0],
        'w_EI': [1.0],
        'w_II': [0.5],
        'I_ext_E': [0.5],
        'I_ext_I': [0.0],
        'max_rate': [100.0],
    }
)

gen = ConstantRateGenerator(dt=dt, params={'rate': 20.0})

syn = StaticSynapse(n_pre=1, n_post=1, dt=dt, params={
    'gsyn_max': 0.1,
    'pconn': [[1.0]],
    'e_r': 0.0,
})

graph = NetworkGraph(dt=dt)
graph.add_input_population('input', gen)
graph.add_population('EI', pop)
graph.add_synapse('input->EI', syn, src='input', tgt='EI')

network = NetworkRNN(graph, integrator=RK4Integrator())

t_values = np.arange(0, T, dt, dtype=np.float32)
t_seq = tf.constant(t_values[None, :, None])
output = network(t_seq)

print(f"Wilson-Cowan output shape: {output.firing_rates['EI'].shape}")
```

---

## Пример 8: Использование конфигурации

```python
from neuraltide.config.schema import (
    NetworkConfig, PopulationConfig, SynapseConfig, InputConfig,
    build_network_from_config
)

config = NetworkConfig(
    dt=0.5,
    integrator='rk4',
    inputs=[
        InputConfig(
            name='input',
            generator_class='SinusoidalGenerator',
            params={
                'dt': 0.5,
                'params': {
                    'amplitude': 10.0,
                    'freq': 8.0,
                    'phase': 0.0,
                    'offset': 5.0,
                },
            }
        )
    ],
    populations=[
        PopulationConfig(
            name='exc',
            model_class='IzhikevichMeanField',
            dt=0.5,
            params={
                'tau_pop': [1.0], 'alpha': [0.5], 'a': [0.02],
                'b': [0.2], 'w_jump': [0.1], 'Delta_I': [0.5], 'I_ext': [1.0],
            }
        )
    ],
    synapses=[
        SynapseConfig(
            name='input->exc',
            synapse_class='StaticSynapse',
            src='input',
            tgt='exc',
            dt=0.5,
            params={
                'n_pre': 1,
                'n_post': 1,
                'gsyn_max': 0.1,
                'pconn': [[1.0]],
                'e_r': 0.0,
            }
        )
    ]
)

network = build_network_from_config(config)

# Далее обычная симуляция
import numpy as np
import tensorflow as tf

t_seq = tf.constant(np.arange(0, 1000, 0.5)[None, :, None])
output = network(t_seq)
print(output.firing_rates['exc'].shape)
```

---

## Пример: Управление начальным состоянием и Stateful режим

NeuralTide позволяет:
1. Устанавливать пользовательские начальные условия
2. Сохранять состояние между батчами (stateful режим)
3. Сбрасывать состояние для нового эксперимента

### Пользовательские начальные условия

```python
network = NetworkRNN(graph, integrator=RK4Integrator())

# Получение начального состояния
init_pop, init_syn = network.get_initial_state(batch_size=1)

# Модификация: r=0.5, v=-1.0, w=0.0 (для IzhikevichMeanField)
init_pop[0] = tf.constant([[0.5]])  # r
init_pop[1] = tf.constant([[-1.0]])  # v (относительно rest=0)
init_pop[2] = tf.constant([[0.0]])  # w

# Установка состояния
network.set_initial_state((init_pop, init_syn))

# Запуск симуляции
output = network(t_seq, initial_state=(init_pop, init_syn))
```

### Stateful режим

```python
# Создание сети с stateful=True
network = NetworkRNN(graph, integrator=RK4Integrator(), stateful=True)

# Первый батч
output1 = network(t_seq)
final_state = network.get_state()

# Второй батч продолжает с финального состояния первого
output2 = network(t_seq)

# Сброс для нового эксперимента
network.reset_state()
```

---

## Пример 9: Взаимное ингибирование двух популяций

Две популяции fast-spiking нейронов с взаимным ингибированием через синапсыshort-term depression (Tsodyks-Markram). Оптимизируются синаптические проводимости и внешние токи.

```python
from neuraltide.examples.example_mutual_inhibition import *
```

Полное описание примера в файле [`examples/example_mutual_inhibition.py`](examples/example_mutual_inhibition.py).

### Архитектура

- Две популяции `IzhikevichMeanField` (pop1, pop2) с dimensionalными параметрами
- Каждая популяция получает вход от `VonMisesGenerator` со сдвигом фазы 150° (2.61 рад)
- Взаимное ингибирование через `TsodyksMarkramSynapse` (short-term depression)

### Оптимизируемые параметры

| Параметр | Начальное | Диапазон |
|----------|-----------|----------|
| `gsyn_max` (syn_1to2, syn_2to1) | 3000 пСм | 100–5000 |
| `I_ext` (pop1, pop2) | 200 пА | 50–500 |
| `tau_d` | 6.02 мс | 2–15 |
| `tau_r` | 359.8 мс | 91–1300 |
| `tau_f` | 21.0 мс | 6–240 |
| `U_inc` | 0.25 | 0.04–0.7 |

### Экспорт результатов

```python
trainer.export_results('results.json')
trainer.export_results('results.csv', format='csv')
```

---

## Пример 10: PlaceFieldGenerator с фазовой прецессией через extra_inputs_seq

Полный пример: `examples/example_place_field_phase_precession.py`

Демонстрирует передачу внешних пространственных координат `(x, y)` в генератор через `extra_inputs_seq` параметр `NetworkRNN.call()`.

```python
import numpy as np
import tensorflow as tf
from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.integrators import RK4Integrator
from neuraltide.populations import IzhikevichMeanField
from neuraltide.synapses import StaticSynapse
from neuraltide.inputs import PlaceFieldGenerator

n_place_cells = 6
dt = 0.5

# PlaceFieldGenerator с фазовой прецессией
gen = PlaceFieldGenerator(dt=dt, params={
    'center_x': [0.4, -0.5, 0.2, -0.3, 0.6, -0.1],
    'center_y': [0.3,  0.4, -0.5, 0.1, -0.2, 0.5],
    'radius':   [0.35, 0.4, 0.3, 0.35, 0.3, 0.35],
    'peak_rate': [25.0, 30.0, 20.0, 22.0, 18.0, 15.0],
    'background_rate': [2.0]*6,
    'theta_modulation_factor': 0.0,
    'precession_slope': [30.0, 35.0, 25.0, -20.0, 40.0, 28.0],
    'precession_init_phase': [0.0, 45.0, 90.0, 135.0, 180.0, 225.0],
    'R': 0.6, 'freq': 8.0,
}, arena_size=((-1.0, 1.0), (-1.0, 1.0)), arena_radius=1.0)

# Readout-популяция
pop = IzhikevichMeanField(dt=dt, params={
    'tau_pop': [1.0]*6, 'alpha': [0.5]*6, 'a': [0.02]*6,
    'b': [0.2]*6, 'w_jump': [0.1]*6, 'Delta_I': [0.05]*6, 'I_ext': [0.0]*6,
})

# Диагональный синапс: каждая place cell → своя readout unit
syn = StaticSynapse(n_pre=6, n_post=6, dt=dt, params={
    'gsyn_max': [[1.0 if i==j else 0.0 for j in range(6)] for i in range(6)],
    'pconn': 1.0, 'e_r': 5.0,
})

graph = NetworkGraph(dt=dt)
graph.add_input_population('place', gen)
graph.add_population('readout', pop)
graph.add_synapse('place->readout', syn, src='place', tgt='readout')

network = NetworkRNN(graph, integrator=RK4Integrator())

# Время
T_total = 5000
t_values = np.arange(0, T_total, dt, dtype=np.float32)
t_seq = tf.constant(t_values[None, :, None])  # [1, T, 1]

# Координаты (x, y) — только они, время отдельно в t_seq
n_steps = len(t_values)
r_traj = 0.7
theta = 2.0 * np.pi * np.arange(n_steps) / n_steps * 2
pos_x = r_traj * np.cos(theta)
pos_y = r_traj * np.sin(theta)
extra_inputs_seq = tf.constant(
    np.stack([pos_x, pos_y], axis=-1).astype(np.float32)[None, :, :]
)  # [1, T, 2]

# Запуск с пространственным входом
output = network(t_seq, extra_inputs_seq=extra_inputs_seq)
rates = output.firing_rates['readout'].numpy()[0]  # [T, n_units]

print(f"Readout shape: {rates.shape}")  # [T, 6]

# Без extra_inputs_seq генератор использует встроенную круговую траекторию
output_default = network(t_seq, extra_inputs_seq=None)
```

### Поток данных

```
extra_inputs_seq [batch, T, 2]  ──┐
                                   ├──→ InputPopulation ──→ PlaceFieldGenerator.call(t, extra_inputs=extra)
t_sequence      [batch, T, 1]  ──┘           │
                                              ▼
                                     firing_rate [batch, n_units]
                                              │
                                              ▼
                                     StaticSynapse ──→ IzhikevichMeanField
                                              │
                                              ▼
                                     output.firing_rates['readout']
```

### Примечания

- `extra_inputs_seq` опционален (по умолчанию `None`). При `None` генераторы получают `[batch, 0]` тензор.
- Если `extra_inputs_seq` имеет rank 2, он автоматически расширяется до `[batch, T, 1]`.
- `extra_inputs` передаётся **только** во `InputPopulation`-генераторы, не в динамические популяции.
- Генераторы `SinusoidalGenerator`, `ConstantRateGenerator`, `VonMisesGenerator` игнорируют `extra_inputs`.