# NeuralTide

NeuralTide — Python-библиотека для дифференцируемого моделирования и обучения нейронных сетей на основе популяционных моделей.

## Основные возможности

- **Популяционные модели**: Izhikevich mean-field, Wilson–Cowan (rate model), Fokker–Planck (Brunel 2000, модель A)
- **Синаптическая динамика**: Tsodyks–Markram (STP), NMDA с Mg²⁺ блоком, SecondOrder (двойная экспонента), статические и композитные синапсы
- **Входные генераторы**: Sinusoidal, ConstantRate, VonMises, PlaceField (клетки места с фазовой прецессией)
- **Внешние входы**: предвычисляются генераторами и подаются в сеть через `declare_input()` + `pack_inputs()`
- **Интеграторы**: Euler, Heun, RK4
- **Дифференцируемость**: все вычисления через TensorFlow; градиенты BPTT или adjoint-методом
- **Расширяемость**: наследование от базовых классов с минимальным контрактом

## Документация

Полная документация доступна в директории `docs/`:

| Раздел | Описание |
|--------|----------|
| [docs/index.md](docs/index.md) | Главная страница документации |
| [docs/architecture.md](docs/architecture.md) | Архитектура библиотеки |
| [docs/core.md](docs/core.md) | Базовые классы (PopulationModel, SynapseModel, BaseInputGenerator) |
| [docs/populations.md](docs/populations.md) | Популяционные модели |
| [docs/synapses.md](docs/synapses.md) | Синаптические модели |
| [docs/inputs.md](docs/inputs.md) | Входные генераторы |
| [docs/integrators.md](docs/integrators.md) | Интеграторы ОДУ |
| [docs/network.md](docs/network.md) | NetworkGraph и NetworkRNN |
| [docs/config.md](docs/config.md) | Система конфигурации |
| [docs/constraints.md](docs/constraints.md) | Ограничения параметров |
| [docs/examples.md](docs/examples.md) | Примеры использования |
| [docs/api_reference.md](docs/api_reference.md) | Полный API reference |

## Установка

```bash
pip install neuraltide
```

или из исходников:

```bash
git clone https://github.com/ivanmysin/neuraltide.git
cd neuraltide
pip install -e ".[dev]"
```

## Быстрый старт

```python
import numpy as np
import tensorflow as tf
from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.populations import IzhikevichMeanField
from neuraltide.synapses import StaticSynapse
from neuraltide.inputs import SinusoidalGenerator
from neuraltide.integrators import RK4Integrator
from neuraltide.utils import print_summary

# Создание генератора (n_units выводится автоматически)
gen = SinusoidalGenerator(
    dt=0.5,
    params={
        'amplitude': 10.0,
        'freq': 8.0,
        'phase': 0.0,
        'offset': 5.0,
    }
)

# Создание популяции (безразмерные параметры, n_units выводится из params)
pop = IzhikevichMeanField(
    dt=0.5,
    params={
        'tau_pop': [1.0],
        'alpha': [0.5],
        'a': [0.02],
        'b': [0.2],
        'w_jump': [0.1],
        'Delta_I': [0.5],
        'I_ext': [1.0],
    }
)

# Создание синапса
syn = StaticSynapse(
    n_pre=1,
    n_post=1,
    dt=0.5,
    params={
        'gsyn_max': 0.1,
        'pconn': [[1.0]],
        'e_r': 0.0,
    }
)

# Построение сети
graph = NetworkGraph(dt=0.5)
graph.declare_input('input', n_units=gen.n_units)   # внешний вход
graph.add_population('exc', pop)
graph.add_synapse('input->exc', syn, src='input', tgt='exc')

network = NetworkRNN(graph, integrator=RK4Integrator())
print_summary(network)

# Симуляция: входные частоты предвычисляются и упаковываются
t_seq = tf.constant(np.arange(0, 1000, 0.5)[None, :, None], dtype=tf.float32)
input_rates = gen(t_seq)                            # [1, T, n_units]
inputs = graph.pack_inputs({'input': input_rates})

output = network(t_seq, inputs=inputs)
print(f"Output shape: {output.firing_rates['exc'].shape}")
```

## Примеры

Подробные примеры доступны в директории `examples/`:

- `example_01_single_population.py` — одна популяция IzhikevichMeanField с тета-входом
- `example_02_exc_inh_nmda.py` — E-I сеть с NMDA синапсами и BandpassReadout
- `example_03_custom_population.py` — создание собственной популяции
- `example_04_fokker_planck.py` — своя Fokker–Planck модель через наследование
- `example_05_lfp_proxy.py` — LFP-прокси как взвешенная сумма тета-фильтрованной активности
- `example_06_brunel_fp.py` — BrunelFokkerPlanck: сходимость к стационару vs аналитика
- `example_place_field_phase_precession.py` — клетки места с фазовой прецессией
- `example_pop_wilson_cowan.py` — популяция Wilson–Cowan (rate model)

См. также [docs/examples.md](docs/examples.md) для подробных описаний.

## Требования

- Python >= 3.12
- TensorFlow >= 2.16
- NumPy >= 1.26
- SciPy >= 1.12 (для VonMisesGenerator / PlaceFieldGenerator)

## Лицензия

MIT
