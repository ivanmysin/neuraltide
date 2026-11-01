# API Reference

Полный справочник по всем классам, методам и функциям NeuralTide.

---

## neuraltide (корневой модуль)

### Функции

#### `seed_everything(seed)`

Устанавливает seed для воспроизводимости.

```python
from neuraltide.utils import seed_everything
seed_everything(42)
```

**Аргументы**:
- `seed` (int): seed для NumPy и TensorFlow

#### `print_summary(network)`

Выводит summary модели.

```python
from neuraltide.utils import print_summary
print_summary(network)
```

**Аргументы**:
- `network` (NetworkRNN): сеть для отображения

---

## neuraltide.config

### Функции

#### `set_dtype(dtype)`

```python
config.set_dtype(tf.float64)
```

#### `get_dtype()`

```python
dtype = config.get_dtype()
```

#### `register_population(name, cls)`

```python
config.register_population('MyPopulation', MyPopulation)
```

#### `register_synapse(name, cls)`

```python
config.register_synapse('MySynapse', MySynapse)
```

#### `register_input(name, cls)`

```python
config.register_input('MyGenerator', MyGenerator)
```

### Реестры

- `POPULATION_REGISTRY: Dict[str, Type]`
- `SYNAPSE_REGISTRY: Dict[str, Type]`
- `INPUT_REGISTRY: Dict[str, Type]`

---

## neuraltide.core.base

### PopulationModel

```python
class PopulationModel(tf.keras.layers.Layer):
```

**Атрибуты**:
- `n_units` (int)
- `dt` (float)
- `state_size` (list[TensorShape])

**Методы**:
- `_make_param(params, name) -> tf.Variable`
- `get_initial_state(batch_size=1) -> StateList`
- `derivatives(state, total_synaptic_input) -> StateList`
- `get_firing_rate(state) -> TensorType`
- `observables(state) -> Dict[str, TensorType]`
- `parameter_spec -> Dict`

### SynapseModel

```python
class SynapseModel(tf.keras.layers.Layer):
```

**Атрибуты**:
- `n_pre` (int)
- `n_post` (int)
- `dt` (float)
- `state_size` (list[TensorShape])

**Методы**:
- `_make_param(params, name) -> tf.Variable`
- `get_initial_state(batch_size=1) -> StateList`
- `forward(pre_firing_rate, post_voltage, state, dt) -> (Dict, StateList)`
- `parameter_spec -> Dict`

### BaseInputGenerator

```python
class BaseInputGenerator(tf.keras.layers.Layer):
```

**Атрибуты**:
- `n_units` (int)
- `dt` (float)

**Методы**:
- `_make_param(params, name) -> tf.Variable`
- `call(t) -> TensorType`
- `parameter_spec -> Dict`

---

## neuraltide.core.network

### NetworkGraph

```python
class NetworkGraph:
```

**Конструктор**:
```python
NetworkGraph(dt: float)
```

**Методы**:
- `add_population(name: str, model: PopulationModel) -> None`
- `declare_input(name: str, n_units: int) -> None`
- `pack_inputs(input_dict: Dict[str, TensorType]) -> TensorType`
- `add_synapse(name: str, model: SynapseModel, src: str, tgt: str) -> None`
- `validate() -> None`

**Свойства**:
- `population_names -> List[str]`
- `synapse_names -> List[str]`
- `dynamic_population_names -> List[str]`
- `input_names -> List[str]`
- `input_offsets -> Dict[str, int]`
- `total_input_units -> int`

### NetworkRNN

```python
class NetworkRNN(tf.keras.layers.Layer):
```

**Конструктор**:
```python
NetworkRNN(
    graph: NetworkGraph,
    integrator: BaseIntegrator,
    return_hidden_states: bool = False,
    stability_penalty_weight: float = 0.0
)
```

**Методы**:
- `call(t_sequence, inputs=None, initial_state=None, training=False) -> NetworkOutput`
- `get_initial_state(batch_size=1) -> Tuple[StateList, StateList]`

**Свойства**:
- `trainable_variables -> List[tf.Variable]`

### NetworkOutput

```python
@dataclass
class NetworkOutput:
    firing_rates: Dict[str, TensorType]
    hidden_states: Optional[Dict[str, Dict[str, TensorType]]]
    stability_loss: TensorType
    final_state: Tuple[StateList, StateList]
```

---

## neuraltide.populations

### IzhikevichMeanField

```python
class IzhikevichMeanField(PopulationModel):
```

**Конструктор**:
```python
IzhikevichMeanField(
    dt: float,
    params: Optional[Dict[str, Any]] = None,
    name: str = "izhikevich_mf",
    **kwargs
)
```

**Параметры (dimensionless)**:
- `tau_pop`, `alpha`, `a`, `b`, `w_jump`, `Delta_I`, `I_ext`

**Параметры (dimensional)**:
- `V_rest`, `V_T`, `Cm`, `K`, `A`, `B`, `W_jump`, `Delta_I`, `I_ext`

**Статические методы**:
- `dimensionless_to_dimensional(...) -> Dict`

### WilsonCowan

```python
class WilsonCowan(PopulationModel):
```

**Конструктор**:
```python
WilsonCowan(
    n_units: int,
    dt: float,
    params: dict,
    **kwargs
)
```

**Параметры**:
- `tau_E`, `tau_I`, `a_E`, `a_I`, `theta_E`, `theta_I`
- `w_EE`, `w_IE`, `w_EI`, `w_II`
- `I_ext_E`, `I_ext_I`, `max_rate`

### FokkerPlanckPopulation

```python
class FokkerPlanckPopulation(PopulationModel):
```

**Конструктор**:
```python
FokkerPlanckPopulation(
    n_units: int,
    dt: float,
    grid_size: int = 100,
    v_min: float = -100.0,
    v_max: float = 50.0,
    **kwargs
)
```

**Методы**:
- `get_boundary_flux(P, dV) -> TensorType`

### InputPopulation (deprecated)

```python
class InputPopulation(PopulationModel):
```

> **Внимание**: `InputPopulation` deprecated. Используйте `NetworkGraph.declare_input()` и предвычисленные входы вместо оборачивания генераторов в `InputPopulation`.

Псевдо-популяция, оборачивающая `BaseInputGenerator`. Не имеет собственной динамики.

**Конструктор**:
```python
InputPopulation(
    generator: BaseInputGenerator,
    **kwargs
)
```

**Методы**:
- `get_firing_rate(state) -> TensorType` — вызывает `generator(state[0], extra_inputs=state[1])`

---

## neuraltide.synapses

### StaticSynapse

```python
class StaticSynapse(SynapseModel):
```

**Конструктор**:
```python
StaticSynapse(
    n_pre: int,
    n_post: int,
    dt: float,
    params: dict,
    **kwargs
)
```

**Параметры**:
- `gsyn_max`, `pconn`, `e_r`

### NMDASynapse

```python
class NMDASynapse(SynapseModel):
```

**Конструктор**:
```python
NMDASynapse(
    n_pre: int,
    n_post: int,
    dt: float,
    params: dict,
    **kwargs
)
```

**Параметры**:
- `gsyn_max_nmda`, `tau1_nmda`, `tau2_nmda`, `Mgb`, `av_nmda`
- `pconn_nmda`, `e_r_nmda`, `v_ref`

### TsodyksMarkramSynapse

```python
class TsodyksMarkramSynapse(SynapseModel):
```

**Конструктор**:
```python
TsodyksMarkramSynapse(
    n_pre: int,
    n_post: int,
    dt: float,
    params: dict,
    **kwargs
)
```

**Параметры**:
- `gsyn_max`, `tau_f`, `tau_d`, `tau_r`, `Uinc`, `pconn`, `e_r`

### CompositeSynapse

```python
class CompositeSynapse(SynapseModel):
```

**Конструктор**:
```python
CompositeSynapse(
    n_pre: int,
    n_post: int,
    dt: float,
    components: List[Tuple[str, SynapseModel]],
    **kwargs
)
```

---

## neuraltide.inputs

### SinusoidalGenerator

```python
class SinusoidalGenerator(BaseInputGenerator):
```

**Конструктор**:
```python
SinusoidalGenerator(
    dt: float,
    params: Dict[str, Any],
    name: str = "sinusoidal_generator",
    **kwargs
)
```

**Параметры**:
- `amplitude`, `freq`, `phase`, `offset`

### ConstantRateGenerator

```python
class ConstantRateGenerator(BaseInputGenerator):
```

**Конструктор**:
```python
ConstantRateGenerator(
    dt: float,
    params: Dict[str, Any],
    name: str = "constant_rate_generator",
    **kwargs
)
```

**Параметры**:
- `rate`

### VonMisesGenerator

```python
class VonMisesGenerator(BaseInputGenerator):
```

**Конструктор**:
```python
VonMisesGenerator(
    dt: float,
    params: Dict[str, Any],
    name: str = "von_mises_generator",
    **kwargs
)
```

**Параметры**:
- `mean_rate`, `R`, `freq`, `phase`

---

## neuraltide.integrators

### BaseIntegrator

```python
class BaseIntegrator(ABC):
```

**Методы**:
- `step(population, state, total_synaptic_input) -> (StateList, TensorType)`

### EulerIntegrator

```python
class EulerIntegrator(BaseIntegrator):
```

### HeunIntegrator

```python
class HeunIntegrator(BaseIntegrator):
```

### RK4Integrator

```python
class RK4Integrator(BaseIntegrator):
```

---

## neuraltide.constraints

### MinMaxConstraint

```python
class MinMaxConstraint(tf.keras.constraints.Constraint):
```

**Конструктор**:
```python
MinMaxConstraint(min_val: Optional[float], max_val: Optional[float])
```

### NonNegConstraint

```python
class NonNegConstraint(tf.keras.constraints.Constraint):
```

### UnitIntervalConstraint

```python
class UnitIntervalConstraint(tf.keras.constraints.Constraint):
```

---

## neuraltide.config.schema

### PopulationConfig

```python
@dataclass
class PopulationConfig:
    name: str
    model_class: str
    dt: float
    params: Dict[str, Any]
```

### SynapseConfig

```python
@dataclass
class SynapseConfig:
    name: str
    synapse_class: str
    src: str
    tgt: str
    dt: float
    params: Dict[str, Any]
    components: Optional[List['SynapseConfig']] = None
```

### InputConfig

```python
@dataclass
class InputConfig:
    name: str
    generator_class: str
    params: Dict[str, Any]
```

### NetworkConfig

```python
@dataclass
class NetworkConfig:
    dt: float
    integrator: str
    populations: List[PopulationConfig]
    synapses: List[SynapseConfig]
    inputs: List[InputConfig]
    stability_penalty_weight: float = 0.0
    return_hidden_states: bool = False
```

### build_network_from_config

```python
def build_network_from_config(config: NetworkConfig) -> Tuple[NetworkRNN, Dict[str, BaseInputGenerator]]:
```

Возвращает кортеж `(network, generators)`, где `generators` — словарь `{name: generator}` для предвычисления входных частот.

---

## neuraltide.data

Модуль для работы с данными: сохранение, загрузка, визуализация в формате HDF5.

### Dataset

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
    generator_params: Dict[str, Any]
```

**Свойства**:
- `T -> int` — число временных шагов
- `total_input_units -> int` — общее число входных каналов
- `total_target_units -> int` — общее число целевых каналов

**Методы**:
- `input_slice(name) -> np.ndarray` — срез входа по имени: `[T, n_units_i]`
- `target_slice(name) -> np.ndarray` — срез цели по имени: `[T, n_units_i]`

### save_dataset

```python
def save_dataset(
    path: str,
    inputs: Dict[str, np.ndarray],
    target: Dict[str, np.ndarray],
    dt: float,
    generator_params: Optional[Dict[str, Any]] = None,
) -> None:
```

Сохраняет dataset в HDF5 (.h5).

**Аргументы**:
- `path`: путь к .h5 файлу
- `inputs`: `{name: array[T, n_units_i]}` — firing rates входов
- `target`: `{name: array[T, n_units_i]}` — целевые firing rates
- `dt`: шаг интегрирования (мс)
- `generator_params`: параметры генераторов (для reproducibility)

### load_dataset

```python
def load_dataset(path: str) -> Dataset:
```

Загружает dataset из HDF5.

### plot_dataset

```python
def plot_dataset(
    data: Dataset,
    max_t: Optional[float] = None,
    figsize: tuple = (14, 8),
) -> None:
```

Строит графики для визуальной проверки данных.

---

## neuraltide.model

Keras-совместимая обёртка для NetworkRNN.

### BrainModelKeras

```python
class BrainModelKeras(tf.keras.Model):
```

**Конструктор**:
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

**Методы**:
- `call(inputs, training=False, initial_state=None) -> NetworkOutput`
- `train_step(data) -> Dict`
- `test_step(data) -> Dict`

**Свойства**:
- `network -> NetworkRNN` — доступ к внутреннему NetworkRNN
- `trainable_variables -> List[tf.Variable]` — обучаемые переменные сети

**Пример**:
```python
from neuraltide.model import BrainModelKeras
from neuraltide.training import MSELoss

graph = NetworkGraph(dt=0.5)
graph.declare_input('theta', n_units=1)
graph.add_population('exc', pop)
graph.add_synapse('theta->exc', syn, src='theta', tgt='exc')

model = BrainModelKeras(graph, RK4Integrator(), dt=0.5,
                        loss_fn=MSELoss(target))
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
model.fit(inputs, targets, epochs=100)
```

---

## neuraltide.core.types

### Типы

```python
TensorType = tf.Tensor
StateList = List[TensorType]
ParamDict = Dict[str, Any]
```

### Функции

```python
def get_pi() -> tf.Tensor:
    """Возвращает π с текущим глобальным dtype."""
```

---

## neuraltide.training

### Trainer

```python
class Trainer:
```

**Конструктор**:
```python
Trainer(
    model: tf.keras.Model,
    loss_fn: callable,
    optimizer: tf.keras.optimizers.Optimizer,
    callbacks: Optional[List] = None,
    grad_method: str = "bptt"
)
```

**Методы**:
- `fit(t_sequence, inputs=None, epochs, verbose=1) -> History`
- `predict(t_sequence, inputs) -> NetworkOutput`
- `export_results(path, format="json")` — экспорт результатов в JSON/CSV

### CompositeLoss

```python
class CompositeLoss:
```

**Конструктор**:
```python
CompositeLoss(components: List[Tuple[float, callable]])
```

### MSELoss

```python
class MSELoss:
```

**Конструктор**:
```python
MSELoss(target: Dict[str, TensorType])
```

### StabilityPenalty

```python
class StabilityPenalty:
```

### AdjointSolver

```python
class AdjointSolver:
```

**Методы**:
- `compute_gradients(t_sequence, inputs, target, loss_fn) -> Tuple[List, List, NetworkOutput]`

---

## neuraltide.utils

### seed_everything

```python
def seed_everything(seed: int) -> None:
```

### print_summary

```python
def print_summary(network, rich_library: Optional[object] = None) -> None:
```

---

## neuraltide.integrators.__init__

Экспорты:
- `EulerIntegrator`
- `HeunIntegrator`  
- `RK4Integrator`
- `BaseIntegrator`

---

## neuraltide.synapses.__init__

Экспорты:
- `StaticSynapse`
- `NMDASynapse`
- `TsodyksMarkramSynapse`
- `CompositeSynapse`

---

## neuraltide.populations.__init__

Экспорты:
- `IzhikevichMeanField`
- `WilsonCowan`
- `FokkerPlanckPopulation`
- `InputPopulation` (deprecated)

---

## neuraltide.inputs.__init__

Экспорты:
- `BaseInputGenerator`
- `SinusoidalGenerator`
- `ConstantRateGenerator`
- `VonMisesGenerator`
