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