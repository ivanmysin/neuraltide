# NeuralTide Documentation

Добро пожаловать в NeuralTide — библиотеку для моделирования нейронных сетей на основе популяционных моделей с использованием TensorFlow.

## Описание

NeuralTide предоставляет гибкий инструментарий для создания и симуляции нейронных сетей с:

- **Популяционными моделями**: Izhikevich Mean-Field, Wilson-Cowan, Fokker-Planck
- **Синаптическими моделями**: Static, NMDA, Tsodyks-Markram (STP), Composite
- **Входными генераторами**: Sinusoidal, ConstantRate, VonMises
- **Интеграторами**: Euler, Heun, RK4
- **Модулями данных**: `neuraltide.data` (сохранение/загрузка HDF5)
- **Keras-моделью**: `neuraltide.model.BrainModelKeras`

## Быстрый старт

```python
import numpy as np
import tensorflow as tf
from neuraltide.populations import IzhikevichMeanField
from neuraltide.synapses import StaticSynapse
from neuraltide.inputs import SinusoidalGenerator
from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.integrators import RK4Integrator

# Создание популяции
pop = IzhikevichMeanField(dt=0.5, params={
    'tau_pop': [1.0], 'alpha': [0.5], 'a': [0.02],
    'b': [0.2], 'w_jump': [0.1], 'Delta_I': [0.5], 'I_ext': [1.0],
})

# Создание генератора входного сигнала
gen = SinusoidalGenerator(dt=0.5, params={
    'amplitude': 10.0, 'freq': 8.0, 'phase': 0.0, 'offset': 5.0,
})

# Создание синапса
syn = StaticSynapse(n_pre=1, n_post=1, dt=0.5, params={
    'gsyn_max': 0.1, 'pconn': [[1.0]], 'e_r': 0.0,
})

# Построение сети
graph = NetworkGraph(dt=0.5)
graph.declare_input('input', n_units=gen.n_units)
graph.add_population('exc', pop)
graph.add_synapse('input->exc', syn, src='input', tgt='exc')

network = NetworkRNN(graph, integrator=RK4Integrator())

# Запуск симуляции
t_seq = tf.constant(np.arange(0, 100, 0.5)[None, :, None])
inputs = graph.pack_inputs({'input': gen(t_seq)})
output = network(t_seq, inputs=inputs)
```

## Структура документации

### Основные разделы

| Раздел | Описание |
|--------|----------|
| [Архитектура](architecture.md) | Общее устройство библиотеки |
| [Базовые классы](core.md) | PopulationModel, SynapseModel, BaseInputGenerator |
| [Популяции](populations.md) | Все типы популяционных моделей |
| [Синапсы](synapses.md) | Все типы синаптических моделей |
| [Генераторы](inputs.md) | Все типы входных генераторов |
| [Интеграторы](integrators.md) | Методы численного интегрирования |
| [Сеть](network.md) | NetworkGraph и NetworkRNN |
| [Конфигурация](config.md) | Система конфигурации и реестры |
| [Обучение](training.md) | Обучение сети: Trainer, loss functions, adjoint method |
| [Данные](data.md) | Модуль neuraltide.data: сохранение и загрузка HDF5 |
| [Модель](model.md) | Модуль neuraltide.model: BrainModelKeras |
| [Ограничения](constraints.md) | Ограничения на параметры |
| [Примеры](examples.md) | Полные примеры использования |
| [API Reference](api_reference.md) | Полный справочник API |

## Установка

```bash
pip install neuraltide
```

или из исходников:

```bash
git clone https://github.com/your-repo/neuraltide.git
cd neuraltide
pip install -e .
```

## Требования

- Python 3.12+
- TensorFlow 2.x
- NumPy
- (опционально) Matplotlib для визуализации
- (опционально) SciPy для VonMisesGenerator
- (опционально) h5py для neuraltide.data

## Версия

0.2.0
