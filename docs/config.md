# Конфигурация

NeuralTide предоставляет систему конфигурации для управления глобальными параметрами и реестрами классов.

---

## neuraltide.config

**Файл**: `neuraltide/config/__init__.py`

### Управление типом данных

#### `set_dtype(dtype)`

Устанавливает глобальный тип данных. Должен вызываться до создания любых объектов.

```python
import tensorflow as tf
import neuraltide.config as config

config.set_dtype(tf.float64)  # по умолчанию tf.float32
```

**Аргументы**:
- `dtype`: tf.DType — тип данных TensorFlow (tf.float32 или tf.float64)

#### `get_dtype()`

Возвращает текущий глобальный тип данных.

```python
dtype = config.get_dtype()  # tf.float32
```

### Реестры классов

NeuralTide использует реестры для динамического создания объектов по имени:

```python
config.POPULATION_REGISTRY   # {name: class}
config.SYNAPSE_REGISTRY      # {name: class}
config.INPUT_REGISTRY        # {name: class}
```

#### `register_population(name, cls)`

Регистрирует пользовательский класс PopulationModel для config-first API.

```python
from neuraltide.config import register_population

class MyPopulation(PopulationModel):
    ...

register_population('MyPopulation', MyPopulation)
```

#### `register_synapse(name, cls)`

Регистрирует пользовательский класс SynapseModel.

```python
from neuraltide.config import register_synapse

class MySynapse(SynapseModel):
    ...

register_synapse('MySynapse', MySynapse)
```

#### `register_input(name, cls)`

Регистрирует пользовательский класс BaseInputGenerator.

```python
from neuraltide.config import register_input

class MyGenerator(BaseInputGenerator):
    ...

register_input('MyGenerator', MyGenerator)
```

---

## Конфигурационное API (schema.py)

**Файл**: `neuraltide/config/schema.py`

Библиотека поддерживает конфигурацию сети через структуры данных dataclass.

### PopulationConfig

```python
@dataclass
class PopulationConfig:
    name: str              # Имя популяции в сети
    model_class: str       # Имя класса из POPULATION_REGISTRY
    dt: float              # Шаг интегрирования
    params: Dict[str, Any] # Параметры популяции
```

### SynapseConfig

```python
@dataclass
class SynapseConfig:
    name: str                              # Имя синапса
    synapse_class: str                     # Имя класса из SYNAPSE_REGISTRY
    src: str                               # Имя исходной популяции
    tgt: str                               # Имя целевой популяции
    dt: float                              # Шаг интегрирования
    params: Dict[str, Any]                 # Параметры синапса
    components: Optional[List['SynapseConfig']] = None  # Для CompositeSynapse
```

### InputConfig

```python
@dataclass
class InputConfig:
    name: str              # Имя входной популяции
    generator_class: str  # Имя класса из INPUT_REGISTRY
    params: Dict[str, Any] # Параметры генератора
```

### NetworkConfig

```python
@dataclass
class NetworkConfig:
    dt: float                                  # Шаг интегрирования
    integrator: str                            # 'euler', 'heun', или 'rk4'
    populations: List[PopulationConfig]        # Список популяций
    synapses: List[SynapseConfig]             # Список синапсов
    inputs: List[InputConfig]                  # Список входов
    stability_penalty_weight: float = 0.0      # Вес штрафа за стабильность
    return_hidden_states: bool = False        # Возвращать скрытые состояния
```

### build_network_from_config(config)

Строит NetworkRNN из конфигурации.

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
```

### Пример: полная конфигурация сети E-I с NMDA

```python
config = NetworkConfig(
    dt=0.1,
    integrator='rk4',
    inputs=[
        InputConfig(
            name='theta_input',
            generator_class='VonMisesGenerator',
            params={
                'dt': 0.1,
                'params': {
                    'mean_rate': 20.0,
                    'R': 0.5,
                    'freq': 8.0,
                    'phase': 0.0,
                }
            }
        )
    ],
    populations=[
        PopulationConfig(
            name='exc',
            model_class='IzhikevichMeanField',
            dt=0.1,
            params={
                'tau_pop': [1.0, 1.0],
                'alpha': [0.5, 0.5],
                'a': [0.02, 0.02],
                'b': [0.2, 0.2],
                'w_jump': [0.1, 0.1],
                'Delta_I': [0.5, 0.5],
                'I_ext': [1.0, 1.0],
            }
        ),
        PopulationConfig(
            name='inh',
            model_class='IzhikevichMeanField',
            dt=0.1,
            params={
                'tau_pop': [1.0, 1.0],
                'alpha': [0.5, 0.5],
                'a': [0.02, 0.02],
                'b': [0.2, 0.2],
                'w_jump': [0.1, 0.1],
                'Delta_I': [0.5, 0.5],
                'I_ext': [0.5, 0.5],  # меньше внешнего тока
            }
        ),
    ],
    synapses=[
        # Вход -> возбуждающая
        SynapseConfig(
            name='input_exc',
            synapse_class='NMDASynapse',
            src='theta_input',
            tgt='exc',
            dt=0.1,
            params={
                'n_pre': 1,
                'n_post': 2,
                'gsyn_max_nmda': 0.15,
                'tau1_nmda': 5.0,
                'tau2_nmda': 150.0,
                'Mgb': 1.0,
                'av_nmda': 0.08,
                'pconn_nmda': [[1.0, 1.0]],
                'e_r_nmda': 0.0,
                'v_ref': 0.0,
            }
        ),
        # Возбуждающая -> тормозная
        SynapseConfig(
            name='exc_inh',
            synapse_class='StaticSynapse',
            src='exc',
            tgt='inh',
            dt=0.1,
            params={
                'n_pre': 2,
                'n_post': 2,
                'gsyn_max': 0.1,
                'pconn': [[1.0, 1.0], [1.0, 1.0]],
                'e_r': 0.0,
            }
        ),
        # Тормозная -> возбуждающая
        SynapseConfig(
            name='inh_exc',
            synapse_class='StaticSynapse',
            src='inh',
            tgt='exc',
            dt=0.1,
            params={
                'n_pre': 2,
                'n_post': 2,
                'gsyn_max': -0.5,  # тормозный (отрицательный)
                'pconn': [[1.0, 1.0], [1.0, 1.0]],
                'e_r': -70.0,  # тормозный потенциал реверса
            }
        ),
    ]
)

network = build_network_from_config(config)
```

---

## Сериализация и десериализация

### Сохранение конфигурации

```python
import json
from neuraltide.config.schema import NetworkConfig

# Создание конфигурации
config = NetworkConfig(...)

# Сериализация в JSON
config_dict = {
    'dt': config.dt,
    'integrator': config.integrator,
    'stability_penalty_weight': config.stability_penalty_weight,
    'return_hidden_states': config.return_hidden_states,
    'inputs': [
        {'name': i.name, 'generator_class': i.generator_class, 'params': i.params}
        for i in config.inputs
    ],
    'populations': [
        {'name': p.name, 'model_class': p.model_class, 'dt': p.dt, 'params': p.params}
        for p in config.populations
    ],
    'synapses': [
        {'name': s.name, 'synapse_class': s.synapse_class, 'src': s.src, 'tgt': s.tgt, 
         'dt': s.dt, 'params': s.params}
        for s in config.synapses
    ],
}

with open('network_config.json', 'w') as f:
    json.dump(config_dict, f, indent=2)
```

### Загрузка конфигурации

```python
import json
from neuraltide.config.schema import NetworkConfig

with open('network_config.json', 'r') as f:
    config_dict = json.load(f)

# Восстановление dataclass
config = NetworkConfig(
    dt=config_dict['dt'],
    integrator=config_dict['integrator'],
    stability_penalty_weight=config_dict.get('stability_penalty_weight', 0.0),
    return_hidden_states=config_dict.get('return_hidden_states', False),
    inputs=[InputConfig(**i) for i in config_dict['inputs']],
    populations=[PopulationConfig(**p) for p in config_dict['populations']],
    synapses=[SynapseConfig(**s) for s in config_dict['synapses']],
)

network = build_network_from_config(config)
```