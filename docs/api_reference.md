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

Модель Вильсона–Коуэна в форме одной популяции (rate model):
`τ·dE/dt = −E + F(w·E + I_ext + I_syn)`, выход `relu(E)·max_rate`.

**Конструктор**:
```python
WilsonCowan(
    dt: float,
    params: dict,
    **kwargs
)
```

`n_units` выводится из параметров.

**Параметры**:
- `tau`, `a`, `theta`, `w`, `I_ext`, `max_rate`

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

### BrunelFokkerPlanck

```python
class BrunelFokkerPlanck(FokkerPlanckPopulation):
```

Готовая реализация Brunel (2000), модель A: дрейф-диффузия на сетке
`[v_min, v_theta]`, поглощающий порог, реинжекция `ν(t − τ_rp)` в
`v_reset`, рефрактерная масса. Связь с сетью: `μ = mu_0 + I_syn`,
σ — параметр.

**Конструктор**:
```python
BrunelFokkerPlanck(
    dt: float,
    params: dict,
    grid_size: int = 300,
    v_min: float = -100.0,
    **kwargs
)
```

**Параметры**:
- `tau_m`, `v_theta`, `v_reset`, `tau_rp`, `mu_0`, `sigma_ext`

**Состояние**: `[P, p_r, nu_buf]` (n_units = 1). Рекомендуются
Euler/Heun (буфер задержки точен для Эйлера).

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

### SecondOrderSynapse

```python
class SecondOrderSynapse(SynapseModel):
```

Непластичный синапс с фильтром второго порядка (двойная экспонента):
`τ₁·τ₂·g̈_s + (τ₁+τ₂)·ġ_s + g_s = ν_pre`. Состояние: `[g_s, dg_s]`.
Поддерживает adjoint-метод градиентов.

**Конструктор**:
```python
SecondOrderSynapse(
    n_pre: int,
    n_post: int,
    dt: float,
    params: dict,
    **kwargs
)
```

**Параметры**:
- `gsyn_max`, `tau1`, `tau2`, `pconn`, `e_r`

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

## neuraltide.optimization

Слой калибровки поверх сети (этапы P0/P1): дифференцируемые преобразования
координат параметров (θ ↔ φ), единое пространство параметров, нормированные
невязки, постановка задачи калибровки, диагностика градиентов (P0); SciPy
L-BFGS-B / least_squares, мультистарт и continuation, forward sensitivities и
checkpointed/truncated BPTT, multiple shooting, расписания и анализ
идентифицируемости (P1). Публичные имена P1 доступны лениво (PEP-562)
через `neuraltide.optimization`.

### Transform

```python
class Transform:
```

Базовый класс преобразования координат θ = forward(φ).

**Методы**:
- `forward(phi) -> tf.Tensor` — φ → θ (дифференцируемо)
- `inverse(theta) -> tf.Tensor` — θ → φ с клиппингом θ в допустимую область
- `jacobian(phi) -> tf.Tensor` — dθ/dφ поэлементно, форма как у phi

### Реализации Transform

- `IdentityTransform()` — θ = φ
- `AffineTransform(a=1.0, b=0.0)` — θ = a·φ + b
- `LogTransform(floor=None)` — θ = exp(φ), положительные параметры
- `LogitTransform(min, max)` — θ = lo + (hi−lo)·sigmoid(φ)
- `ScaledSigmoidTransform(min, max, s=1.0)` — θ = lo + (hi−lo)·sigmoid(φ/s)
- `SoftplusTransform(floor=None)` — θ = softplus(φ) + floor
- `BoundedLogTransform(min, max)` — лог-координата внутри диапазона, 0 < min < max:
  θ = exp(log min + (log max − log min)·sigmoid(φ)), θ(0) = √(min·max)

> **Примечание**: sigmoid-образные преобразования (logit, scaled_sigmoid,
> bounded_log) насыщаются у границ интервала — производная dθ/dφ стремится
> к нулю. У нижней границы (и у `softplus`/`log` около нуля) dθ/dφ ≈ e^φ:
> по затуханию градиента у нуля `softplus` и `log` не лучше sigmoid-образных;
> их преимущество — отсутствие верхней границы.

### Фабрика и реестр

```python
def make_transform(kind: str, **kwargs) -> Transform
def register_transform(kind: str, cls: Type[Transform]) -> None
TRANSFORM_REGISTRY: Dict[str, Type[Transform]]
```

### ParameterEntry

```python
@dataclass
class ParameterEntry:
    full_name: str                 # стабильное имя вида 'exc/Delta_I'
    shape: tf.TensorShape
    transform: Transform = IdentityTransform()
    min: Optional[float] = None    # физические границы
    max: Optional[float] = None
    group: str = "population"      # 'population' | 'synapse' | пользовательская
    trainable: bool = True         # False — реестровая константа вне вектора φ
    variable: Optional[tf.Variable] = None  # None для ручных записей
```

`trainable=False` — реестровая константа: запись остаётся в `entries`/
`names` (интроспекция, duplicate-check), но не входит в `size`, `pack`,
`unpack`, `project`, `apply_to_model` и любые φ-операции задач.

### StateLocator

```python
@dataclass
class StateLocator:
    population: str                  # имя популяции в NetworkGraph
    slot: int                        # индекс тензора в state_size популяции
    experiment: Optional[str] = None # None — common 'x0/<pop>/<slot>';
                                     # str — exp-specific
                                     # 'x0/<experiment>/<pop>/<slot>'
```

Адрес компоненты начального состояния в плоском списке pop-состояний сети
(`start + slot` по `network.pop_state_offsets[population]`, не зависит от
experiment). Классификация common vs exp-specific определяется полем
`experiment`, а не разбором имени записи.

### ParameterSpace

```python
class ParameterSpace:
```

Плоский вектор φ оптимизатора над trainable-параметрами сети. φ строится
только по записям с `trainable=True`; записи с `trainable=False` —
реестровые константы вне φ.

**Построение**:
- `ParameterSpace.from_network(network, transforms=None) -> ParameterSpace` —
  сборка из NetworkRNN; `transforms: {full_name: Transform}` (иначе identity)
- `add_entry(entry: ParameterEntry) -> None` — ручная регистрация
  (например, nuisance-параметров без переменной модели)
- `add_initial_state_entries(network, include=None, transforms=None,
  bounds=None) -> List[str]` — общие компоненты начального состояния
  `x0/<population>/<slot>` (kind='initial_state', locator с
  `experiment=None`); `include` — полные имена и/или кортежи
  `(population, slot)`; без него — все слоты всех динамических популяций
- `add_experiment_initial_state_entries(network, experiment, include=None,
  transforms=None, bounds=None) -> List[str]` — эксперимент-специфичные
  компоненты `x0/<experiment>/<population>/<slot>` (locator с
  `experiment=<experiment>`; `'/'` в имени эксперимента — `ValueError`);
  `include` дополнительно принимает кортежи `(experiment, population,
  slot)`. Общие и exp-записи могут сосуществовать в одном реестре;
  дубли ловятся по полному имени

**Методы**:
- `pack(values=None) -> tf.Tensor` — физические θ → плоский φ = inverse(θ)
- `unpack(phi) -> Dict[str, tf.Tensor]` — φ → θ = forward(φ); дифференцируемо по phi
- `apply_to_model(theta_dict) -> None` — запись θ в переменные модели
- `project(phi) -> tf.Tensor` — проекция φ в допустимую область (клиппинг θ в [min, max])

**Свойства**:
- `entries` — все записи реестра {full_name: ParameterEntry}
  (интроспекция, duplicate-check);
- `names` — имена всех записей реестра;
- `phi_entries` — ordered trainable-only view {full_name: ParameterEntry};
  подмножество, определяющее вектор φ (его длина — `size`; pack/unpack/
  project/apply_to_model работают только с этими записями);
- `trainable_names` — имена trainable-записей в порядке регистрации
  (координаты φ);
- `size` — длина вектора φ (суммарное число элементов trainable-записей);
- `shape(name)`, `group(name)`; также `model_entry_names`,
  `tensor_entry_names` (записи начального состояния)

### ObservationSpec

```python
@dataclass
class ObservationSpec:
    name: str                       # уникальное имя наблюдения
    population: str                 # имя популяции в NetworkGraph
    target: TensorType              # [batch, T, n_units]
    sigma: float = 1.0              # масштаб шума; вклад делится на sigma**2
    mask: Optional[TensorType] = None
    weight: float = 1.0
    kind: str = "l2"                # l2 | relative | msle | huber | pseudo_huber
    readout: Optional[BaseReadout] = None
    eps: float = 1e-8
    delta: float = 1.0
```

### ResidualLoss

```python
class ResidualLoss(BaseLoss):
```

Сумма нормированных невязок по набору ObservationSpec. Скалярный интерфейс
совместим с Trainer; residual-интерфейс честный:
`0.5 * ||residual(predictions)||^2 == __call__(predictions, model)`.

**Конструктор**:
```python
ResidualLoss(observations: Sequence[ObservationSpec])
```

**Методы**:
- `__call__(predictions, model) -> TensorType` — полная loss
- `components(predictions, model) -> Dict[str, TensorType]` — взвешенные вклады по именам наблюдений
- `per_step_loss(firing_rates, target, mask=None) -> TensorType` — аддитивный вклад временного шага
- `residual(predictions) -> TensorType` — плоский вектор невязок r

**Свойства**:
- `target`, `mask` — dict {spec.name: ...} в формате Trainer

### CalibrationProblem

```python
class CalibrationProblem:
```

Задача калибровки: сеть + ParameterSpace + протокол + objective; единые
операции для всех оптимизаторов (Keras, SciPy, безградиентные методы).

**Конструктор**:
```python
CalibrationProblem(
    network: NetworkRNN,
    parameter_space: ParameterSpace,
    t_sequence: TensorType,
    inputs: Optional[TensorType] = None,
    objective: Optional[Callable] = None,   # (predictions, network) -> scalar
    initial_state: Optional[tuple] = None,  # (pop, syn)
    grad_mode: str = "assign",              # "assign" | "chain"
)
```

**Методы**:
- `simulate(phi=None) -> NetworkOutput` — прямой прогон сети
- `evaluate(phi=None) -> Evaluation` — потеря, компоненты и невязки
- `value_and_gradient(phi=None) -> Tuple[Evaluation, tf.Tensor]` — потеря и градиент по φ
- `residual(phi=None) -> np.ndarray` — вектор невязок для GN/LM/TRF
- `bounds_phi() -> Tuple[np.ndarray, np.ndarray]` — физические границы в координатах φ
- `pack(phi=None) -> tf.Tensor`

Контракт мутации: `phi=None` — модель не изменяется; `phi != None` — модель
явно переводится в θ = unpack(phi) и остаётся в нём после вызова.

### Evaluation

```python
@dataclass
class Evaluation:
    loss: float
    components: Dict[str, float] = {}
    residuals: np.ndarray          # нулевой длины, если residual-интерфейса нет
    grad: Optional[np.ndarray] = None
    status: str = "ok"             # 'ok' | 'no_residual_interface'
    n_forward: int = 0
    n_gradient: int = 0
    wall_time: float = 0.0
    dtype: str = "float32"
    dt: float = 0.0
```

Результат одного вычисления CalibrationProblem. `to_dict()` — JSON-сериализуемый словарь.

### OptimizationResult

```python
@dataclass
class OptimizationResult:
    phi_best: np.ndarray
    theta_best: Mapping[str, np.ndarray]
    loss_best: float
    history: List[Dict[str, Any]] = []
    seed: Optional[int] = None
    config: Dict[str, Any] = {}
    versions: Dict[str, str] = {}  # по умолчанию заполняются автоматически
    dtype: str = "float32"
```

Единый формат результата оптимизации. `to_dict()` / `from_dict()`,
`save(path)` / `load(path)` — JSON-сериализация (numpy → list).

### DTO результатов (P1)

```python
@dataclass
class LocalOptimizerResult:   # результат local-адаптера
    base: OptimizationResult
    method: str = "L-BFGS-B"       # 'L-BFGS-B' | 'TRF' | 'LM'
    success: bool = False
    status: int = 0
    message: str = ""
    nit: int = 0
    nfev: int = 0
    njev: int = 0
    grad_norm: float = 0.0
    bounds_active: List[str] = []  # имена координат на границе
    n_forward: int = 0             # дельта счётчиков задачи за запуск
    n_gradient: int = 0
    wall_time: float = 0.0
    validation: Dict[str, float] = {}   # held-out протоколы

@dataclass
class StartResult:            # один старт мультистарта
    index: int = 0
    x0_origin: str = "manual"  # 'sobol' | 'lhs' | 'random' | ...
    phi0: np.ndarray = ...
    phi: np.ndarray = ...
    loss: float = 0.0
    status: str = "pending"    # 'ok' | 'screen_rejected' | 'failed'
    message: str = ""
    screened_loss: Optional[float] = None
    refined: bool = False
    screen_refined: bool = False     # короткий screen-refine выполнен
    local: Optional[LocalOptimizerResult] = None
    wall_time: float = 0.0

@dataclass
class MultistartResult:       # итог мультистарта
    base: OptimizationResult
    solutions: List[StartResult] = []
    best_index: int = -1
    n_starts: int = 0
    n_refined: int = 0
    n_failed: int = 0
    seed: Optional[int] = None
    clustering: Dict[str, Any] = {}    # n_clusters / labels / representatives
    wall_time: float = 0.0

@dataclass
class RunCheckpoint:          # чекпоинт прерванного запуска
    seed: Optional[int] = None
    config: Dict[str, Any] = {}
    completed: List[Any] = []
    pending: List[Any] = []
    best_index: int = -1

@dataclass
class JacobianResult:         # якобиан невязок (forward sensitivities)
    J: np.ndarray = ...        # [n_residuals, size]
    r: np.ndarray = ...
    loss: float = 0.0          # 0.5 * ||r||^2
    phi: np.ndarray = ...
    n_forward: int = 0
    wall_time: float = 0.0
    method: str = "forward"
```

Все DTO: `to_dict()` / `from_dict()` / `save(path)` / `load(path)`.

### Experiment / MultiExperimentProblem / FrozenProblem

```python
@dataclass
class Experiment:             # протокол одного эксперимента
    name: str
    t_sequence: TensorType
    inputs: Optional[TensorType] = None
    objective: Optional[Callable] = None   # None -> objective задачи (validate)
    initial_state: Optional[tuple] = None  # (pop, syn)

class MultiExperimentProblem:
    # одна сеть + один ParameterSpace; сумма потерь в одной GradientTape,
    # residual — конкатенация, компоненты с префиксом '{name}/'
    def __init__(self, network, parameter_space, experiments,
                 grad_mode="assign", held_out=None): ...
    # duck parity с CalibrationProblem: size/pack/unpack/bounds_phi/
    # evaluate/value_and_gradient/residual/simulate/validate/counters
    # начальные состояния: общие 'x0/<pop>/<slot>' — во всех экспериментах;
    # exp-specific 'x0/<experiment>/<pop>/<slot>' (см.
    # ParameterSpace.add_experiment_initial_state_entries) — только у
    # эксперимента с совпадающим именем (включая held-out в validate),
    # с приоритетом над общими; orphan/exp-вне-контекста — UserWarning

class FrozenProblem:
    # подпространство φ: free_names свободны, fixed заморожены на явных θ,
    # прочие — на текущих значениях модели на момент конструирования
    def __init__(self, problem, free_names=None, fixed=None): ...
    # evaluate/value_and_gradient (reduced-градиент)/residual/validate/
    # simulate/bounds_phi/pack/unpack; счётчики общие с базовой задачей
```

### Локальные оптимизаторы (P1)

```python
def minimize_lbfgsb(problem, phi0=None, *, maxiter=200, gtol=1e-6,
                    ftol=None, validate=True, options=None
                    ) -> LocalOptimizerResult
def least_squares(problem, phi0=None, *, method="trf", jacobian=None,
                  jacobian_max_params=200, callback=None, x_scale="jac",
                  max_nfev=None, maxiter=None, validate=True, **kwargs
                  ) -> LocalOptimizerResult
```

`minimize_lbfgsb`: `ftol=None` — dtype-aware auto (float64 → 2.2e-9,
прежнее поведение scipy; float32 → ≈eps32, 1.19e-7). Явный `ftol`/
`options['ftol']` чтится без изменений (приоритет у `options`), resolved
значение фиксируется в `config['options']['ftol']`. При float32-координатах
φ и эффективном `ftol < 1e-7` — одноразовый `UserWarning` о риске
`ABNORMAL_TERMINATION_IN_LNSRCH` (значение не переопределяется;
рекомендуется `ftol >= 1e-7` или dtype float64).

`least_squares`: `method` — `'trf'` (bounds) | `'lm'` (только без bounds);
`jacobian` — `None | '2-point' | '3-point' | callable(x) -> [m, n]`
(callable лениво связывается с `sensitivities.residual_jacobian`); при
`None` по умолчанию — forward-якобиан sensitivities при
`size <= jacobian_max_params` (200) и полном контракте `CalibrationProblem`,
иначе `'2-point'`; фактический режим —
в `config["jacobian"]`. Forward-режим несовместим с предшествующими
gradient-прогонами на том же экземпляре (`problem.n_gradient > 0`, TF
forward-AD state одного `NetworkRNN`): честный fallback на `'2-point'` с
`UserWarning`. `callback(intermediate_result)` — callback итераций
(scipy >= 1.16; на 1.15 игнорируется с предупреждением; у `'lm'` не
поддержан scipy — `UserWarning`). `maxiter` — унифицированный бюджет
итераций (как у `minimize_lbfgsb`): при `max_nfev=None` маппируется как
`max_nfev = maxiter * (problem.size + 2)`; одновременная передача `maxiter`
и `max_nfev` — `ValueError`. scipy-defaults ftol/xtol/gtol не меняются;
явный `kwargs['ftol']` при float32-φ ниже 1e-7 — одноразовый `UserWarning`
(значение не переопределяется). Нефинитный
старт/невязки отклоняются `ValueError` до запуска решателя.

### Чувствительности (P1)

```python
def residual_jvp(problem, phi, v) -> Tuple[np.ndarray, np.ndarray]
    # (r, J·v) — numpy float64, [n_residuals]
def residual_jacobian(problem, phi, *, force=False, max_params=200
                      ) -> JacobianResult
def value_and_gradient_checkpointed(problem, phi, segment_length
                                    ) -> CheckpointedGradient
def truncated_bptt(problem, phi0, optimizer, *, horizon, epochs,
                   warm_steps=0) -> TruncatedResult

@dataclass
class CheckpointedGradient:
    loss: float
    grad: np.ndarray
    segments: int
    wall_time: float
    peak_memory: Optional[float] = None   # tracemalloc, нижняя оценка

@dataclass
class TruncatedResult:
    history: List[Dict[str, float]]
    final_loss: float      # последнее обучающее окно (приближённо)
    free_run_loss: float   # полный свободный прогон после обучения
    notes: str             # документация приближённости режима
```

### Multiple shooting (P1)

```python
@dataclass
class ShootingConfig:
    nodes: Union[int, Sequence[int]]   # сегменты или явные границы [0..T]
    def resolve(self, T: int) -> List[int]: ...

@dataclass
class SingleShootingCheck:
    loss_ms: float
    loss_single: float
    gaps: Dict[str, Any]       # max/mean/l2/finite/joints
    consistent: bool           # разрывы <= gap_tol, потери конечны

class MultipleShootingProblem:
    def __init__(self, problem, config, *, continuity_weight=1.0,
                 include_syn_states=True, fix_first_segment=False): ...
    # duck parity с CalibrationProblem + специфичные:
    def simulate_segment(self, phi, k) -> NetworkOutput: ...
    def continuity_gaps(self, phi=None) -> Dict[str, Any]: ...
    def initialize_from_single_shooting(self, phi_base=None) -> tf.Tensor: ...
    def to_single_shooting_problem(self, phi) -> CalibrationProblem: ...
    def finalize(self, phi, gap_tol=1e-6, loss_rtol=1e-6,
                 loss_atol=1e-10) -> SingleShootingCheck: ...  # + сравнение loss_ms и loss_single
```

Residual-раскладка `r = [r_data; √w·gap_k]`, `0.5·‖r‖² == loss_ms` точно.
`fix_first_segment=True` закрепляет сегмент 0 за начальным состоянием
базовой задачи: записей `x0s/0/...` нет, размерность φ меньше на число
слотов первого сегмента, стыки и `finalize` сохраняются.
`BrunelFokkerPlanck` не поддерживается (`NotImplementedError`).

### Стратегии и расписания (P1)

```python
def sample_phi0s(space, n, *, method="sobol", seed=None,
                 log_uniform=None, clip_to_bounds=True) -> np.ndarray
def multistart(problem, *, n_starts=20, method="sobol", screen_evals=1,
               screen_refine_steps=0, screen_local="lbfgsb",
               refine_top_k=5, local="lbfgsb", seed=None,
               checkpoint_path=None, resume=False, dedup_rtol=1e-4
               ) -> MultistartResult
def dt_continuation(build_problem, dts, *, optimizer="lbfgsb", **kwargs
                    ) -> ContinuationResult
def subset_continuation(problem, stages_free_names, *, local="lbfgsb",
                        **kwargs) -> ContinuationResult

@dataclass
class ContinuationResult:     # общий DTO continuation-цепочек
    stage_results: List[Any] = []
    final_result: Any = None
    final_dt: float = 0.0
    wall_time: float = 0.0
    history: List[Dict[str, Any]] = []

@dataclass
class Stage:
    name: str
    problem: Any               # обычно view базовой CalibrationProblem
    optimizer: Union[str, Callable] = "lbfgsb"
    maxiter: int = 200
    meta: Dict[str, Any] = {}  # 'alpha' зарезервирован для teacher forcing

class ContinuationScheduler:
    def __init__(self, stages: Sequence[Stage]): ...
    def run(self, problem) -> ScheduleResult:
        # перенос phi между стадиями + обязательный free_run_check
        ...

@dataclass
class ScheduleResult:
    stage_results: List[Any] = []
    final_result: Any = None
    alpha_history: List[float] = []
    free_run_check: Optional[Evaluation] = None
    history: List[Dict[str, Any]] = []
    wall_time: float = 0.0

def horizon_growth(base_problem, horizons, *, build_inputs=None) -> List[Stage]
def teacher_forcing_schedule(base_problem, target_rates, alphas) -> List[Stage]
def objective_schedule(base_problem, objectives) -> List[Stage]
def smooth_objective(objective, window: int) -> ResidualLoss
```

### Идентифицируемость (P1)

```python
def compute_fim(jac: JacobianResult, *, energy=0.99, param_names=None,
                collinearity_threshold=0.95, max_dirs=10) -> FIMResult
def likelihood_profile(problem, name, *, values=None, n_points=11, grid=None,
                       optimizer="lbfgsb", warm_start=True) -> ProfileResult
def select_subset(problem, jac, keep, *, refit=True) -> SubsetResult
```

`likelihood_profile` и `select_subset` работают только с trainable-именами
ParameterSpace (координатами φ): non-trainable имя (реестровая константа
вне вектора φ) отклоняется `ValueError`, как и неизвестное.

@dataclass
class FIMDirection:
    index: int
    singular_value: float
    eigenvalue: float
    energy_fraction: float
    cumulative_energy: float
    top_params: List[Tuple[str, float]]   # до 5 крупнейших компонент

@dataclass
class CollinearityPair:
    first: str
    second: str
    cosine: float

@dataclass
class FIMResult:              # F = JᵀJ по готовому якобиану
    fim: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    singular_values: np.ndarray
    condition_number: float       # kappa(J) по ненулевым s
    condition_number_fim: float   # kappa(F)
    effective_rank: int           # по накопленной энергии lambda
    energy: float = 0.99
    stiff_dirs: List[FIMDirection] = []
    soft_dirs: List[FIMDirection] = []
    collinearity: List[CollinearityPair] = []
    param_sensitivity: List[Tuple[str, float]] = []
    param_names: List[str] = []
    phi: np.ndarray = ...
    loss: float = 0.0
    J_shape: Tuple[int, int] = (0, 0)

@dataclass
class ProfileResult:         # одномерный профиль правдоподобия
    name: str
    theta_values: List[float]
    losses: List[float]           # NaN для неуспешных узлов
    rel_loss: List[float]
    successes: List[bool]
    phi_path: List[Optional[np.ndarray]]
    nfev: List[int]
    messages: List[str]
    intervals: Dict[str, Optional[Tuple[float, float]]]  # '0.95' по chi2
    chi2_threshold: float         # 3.8414588 (2*dNLL ~ chi2(1))
    rel_loss_threshold: float     # 1.9207294
    warnings: List[str] = []
    optimizer: str = "lbfgsb"
    warm_start: bool = True
    n_forward: int = 0
    n_gradient: int = 0
    wall_time: float = 0.0
    notes: str = ""

@dataclass
class SubsetResult:          # доводка подмножества с оценкой смещения
    kept_names: List[str]
    fixed_names: List[str]
    refit_result: Optional[LocalOptimizerResult]
    loss_before: float
    loss_after: Optional[float]
    theta_delta_kept: Optional[Dict[str, np.ndarray]]
    theta_delta_fixed: Optional[Dict[str, np.ndarray]]  # структурно 0
    prediction_delta: Optional[Dict[str, float]]   # ||dr||/||r|| по популяциям
    validation_delta: Optional[Dict[str, float]]   # dloss на held-out
    notes: str = ""
    n_forward: int = 0
    n_gradient: int = 0
    wall_time: float = 0.0
```

### Глобальный поиск (P2)

```python
def global_search(problem, *, method="de", max_evals=None, bounds=None,
                  phi0=None, seed=None, refine_top_k=0, local="lbfgsb",
                  checkpoint_path=None, resume=False, evaluator=None,
                  options=None, objective_names=None, validate=True
                  ) -> GlobalOptimizerResult | ParetoResult
```

`method`: `'de'` | `'cma'` (extra `global`) | `'spsa'` | `'basinhopping'`
| `'nsga2'` (extra `pareto`, возвращает `ParetoResult` по компонентам целей).
`'de'`/`'cma'`/`'nsga2'` требуют конечных границ. checkpoint/resume —
только `'cma'`/`'spsa'`. `evaluator` — внешний пакетный callable
(`parallel.make_evaluator`), несовместим с `'nsga2'`.

```python
class OptionalDependencyError(ImportError): ...   # cma / pymoo не установлены

@dataclass
class GlobalOptimizerResult:
    base: OptimizationResult      # phi_best/theta_best/loss_best/история
    method: str                   # 'de' | 'cma' | 'spsa' | 'basinhopping'
    success: bool
    message: str
    n_evals: int                  # оценок глобальной фазы (<= max_evals)
    n_forward: int
    n_gradient: int
    wall_time: float
    history: List[Dict[str, Any]]   # {"n_evals", "loss", "failed"}
    candidates: List[StartResult]   # доводка refine_top_k
    validation: Dict[str, float]

@dataclass
class ParetoResult:               # method='nsga2'
    base: OptimizationResult      # knee-точка фронта (loss_best — сумма целей)
    frontier: Dict[str, np.ndarray]  # {objective_name: значения на фронте}
    objective_names: List[str]    # порядок столбцов frontier
    method: str = "nsga2"
    success: bool = False         # непустой фронт
    message: str = ""
    n_evals: int = 0
    n_forward: int = 0
    wall_time: float = 0.0
    history: List[Dict[str, Any]] = []
    seed: Optional[int] = None
    config: Dict[str, Any] = {}
```

### Балансировка компонент и робастные градиенты (P2)

```python
class WeightStrategy:                 # протокол: initial_weights/update
    def initial_weights(self, names) -> np.ndarray: ...
    def update(self, losses, grads, step) -> np.ndarray: ...

class FixedWeights(WeightStrategy):
    def __init__(self, weights): ...  # float | {имя: вес} | вектор
class UncertaintyWeighting(WeightStrategy):
    def __init__(self, eps=1e-8, min_weight=1e-3, max_weight=1e3): ...
class SelfAdaptiveWeights(WeightStrategy):
    def __init__(self, exponent=1.0, eps=1e-8, min_weight=1e-3,
                 max_weight=1e3): ...
class GradNormWeights(WeightStrategy):
    def __init__(self, alpha=1.0, step_size=1.0, eps=1e-8,
                 min_weight=1e-3, max_weight=1e3): ...

def pcgrad_project(grads, *, seed=None) -> np.ndarray
    # Mapping|Sequence -> mean proj; seed=None — детерминированная
    # batch-модификация (порядок j фиксирован), seed=int — оригинальный
    # случайно-упорядоченный PCGrad Yu et al. 2020 (np.random.default_rng)
def cagrad_project(grads, c: float = 0.5) -> np.ndarray
def balanced_gradient(problem, phi, weights, projector=None, *,
                      projector_kwargs=None
                      ) -> Tuple[Dict[str, float], np.ndarray]
def adaptive_window_sampler(problem, window_size, focus="large_residual",
                            boost=3.0) -> AdaptiveWindowFocus

@dataclass
class BalancedResult:
    base: OptimizationResult
    weight_history: List[Dict[str, Any]]  # 'step'/'loss'/'grad_norm'/'w:<имя>'/'l:<имя>'
    notes: str = ""
    baseline: Dict[str, Any] = {}         # {'method','best_loss','final_loss',...}

def balanced_train(problem, phi0, strategy, *, optimizer=None, max_steps,
                   projector=None, projector_kwargs=None, seed=None,
                   history_path=None) -> BalancedResult
```

Все стратегии — детерминированные суррогаты (пересчёт по наблюдённым
losses/grads, не совместное обучение); только BPTT-режим. `balanced_train`
всегда прогоняет baseline FixedWeights(1.0) из той же phi0; общие
`projector`/`projector_kwargs` применяются и в основном, и в baseline-
прогонах (kwargs сохраняются в `base.config['projector_kwargs']`).

### Ограничения на прогнозы (P2)

```python
@dataclass
class ConstraintSpec:                 # kind: 'eq' (c(x)=0) | 'ineq' (g(x)<=0)
    name: str
    kind: str
    fn: Callable                      # (predictions, network) -> Tensor
    jacobian: Optional[Any] = None    # None | 'fd' | callable(phi) -> [m, n]

@dataclass
class AugmentedLagrangianState:
    lambdas: Dict[str, np.ndarray]
    mu: float = 10.0
    mu_factor: float = 10.0
    def to_dict(self) -> Dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data) -> "AugmentedLagrangianState": ...

class ConstrainedProblem:             # duck parity с CalibrationProblem
    def __init__(self, problem, constraints, state=None, *,
                 augmented_residual=False): ...

def fd_jacobian(value_fn, phi, *, h=None, h_values=None,
                base_value=None) -> np.ndarray
def epsilon_constraint(problem, objectives, index, eps, *, objective=None,
                       weights=None) -> ConstrainedProblem
def augmented_lagrangian(problem, constraints, *, local="lbfgsb",
                         max_outer=10, mu0=10.0, eta=0.25, seed=None,
                         checkpoint_path=None, resume=False) -> ALResult

@dataclass
class ALResult:
    base: OptimizationResult      # data loss исходной задачи в phi_best
    state: AugmentedLagrangianState
    violation_history: List[Dict[str, Any]]
    n_outer: int
    validation: Dict[str, float]
    success: bool
    wall_time: float
```

### Неопределённость (P2)

```python
def residual_diagnostics(problem, phi, *, max_lag=10, n_bins=10,
                         windows=None) -> ResidualDiagnostics
def laplace_approximation(problem, phi_map, *, jac=None, prior_cov=None,
                          method="gn", level=0.95, max_params=200,
                          fd_h=1e-5, max_fd_params=50) -> LaplaceResult
def bootstrap_experiments(problem, *, n_draws=100, seed=None,
                          evaluator=None) -> BootstrapResult
def residual_bootstrap(problem, *, n_draws=100, seed=None) -> BootstrapResult
def parameter_ensemble(results) -> EnsembleResult
def predictive_intervals(problem, phi_samples, *, level=0.95) -> PredictiveResult
def noise_sensitivity(laplace_result, sigma_factors) -> SensitivityResult

@dataclass
class ResidualDiagnostics:
    specs: List[SpecResidualDiagnostics]   # per ObservationSpec
    windows: List[WindowDiagnostics]
    max_abs: float
    phi: np.ndarray
    loss: float                    # 0.5*||r||^2 в точке
    n_residuals: int
    max_lag: int
    n_bins: int
    has_observation_metadata: bool
    n_forward: int
    wall_time: float
    notes: str = ""

@dataclass
class SpecResidualDiagnostics: ...
@dataclass
class WindowDiagnostics: ...

@dataclass
class LaplaceResult:               # гауссово приближение постериора
    phi_map: np.ndarray
    covariance: np.ndarray         # [n, n] в phi-координатах
    precision_gn: np.ndarray
    prior_precision: Optional[np.ndarray]
    eigvals: np.ndarray
    eigvecs: np.ndarray
    std: np.ndarray
    param_names: List[str]
    intervals: Dict[str, Tuple[float, float]]        # в физических theta
    phi_intervals: Dict[str, Tuple[float, float]]    # phi_map +- z*std
    level: float
    z: float
    method: str                    # 'gn' | 'fd-hessian'
    pd: bool                       # precision обратилась через Cholesky
    jitter_used: Optional[float]
    loss: float
    n_forward: int
    wall_time: float
    max_params: int
    notes: str = ""

@dataclass
class BootstrapResult: ...
@dataclass
class EnsembleResult:              # ансамбль phi/theta из готовых решений
    phi_samples: np.ndarray        # [n_samples, n]
    theta_samples: Optional[Dict[str, np.ndarray]]  # {full_name: [n, *shape]}
    mean: np.ndarray
    std: np.ndarray
    cov: Optional[np.ndarray]      # выборочная ковариация phi (ddof=1)
    losses: List[Optional[float]]
    n_samples: int
    size: int
    notes: str = ""

@dataclass
class PredictiveResult:            # квантили по ансамблю phi
    level: float
    populations: List[str]
    lower: Dict[str, np.ndarray]   # {pop: [T, n_units]}
    upper: Dict[str, np.ndarray]
    central: Dict[str, np.ndarray] # поканальная медиана
    mean: Dict[str, np.ndarray]
    n_samples: int
    n_skipped_nonfinite: int
    n_forward: int
    wall_time: float
    notes: str = ""

@dataclass
class NoiseFactorResult:           # posterior при масштабе шума f*sigma
    sigma_factor: float
    covariance: np.ndarray
    std: np.ndarray
    phi_intervals: Dict[str, Tuple[float, float]]
    pd: bool
    jitter_used: Optional[float]

@dataclass
class SensitivityResult:           # noise_sensitivity: per-factor записи
    entries: List[NoiseFactorResult]
    phi_map: np.ndarray
    param_names: List[str]
    level: float
    z: float
    notes: str = ""
```

### Пакетная оценка кандидатов (P2)

```python
@dataclass
class CandidateTask:
    index: int
    phi: np.ndarray
    seed: Optional[int] = None

@dataclass
class EvalRecord:
    index: int
    loss: float                    # inf при исключении в worker'е
    components: Dict[str, float]
    wall_time: float
    n_forward: int
    error: Optional[str]           # None | '<ТипИсключения>: <сообщение>'

class SerialPool:
    def __init__(self, build_problem, fn="evaluate"): ...
    def map(self, tasks, *, ordered=True) -> List[EvalRecord]: ...
    def close(self) -> None: ...

class ProcessPool:                 # spawn-процессы, TF-граф в worker'е
    def __init__(self, build_problem, fn="evaluate", n_workers=2,
                 tf_threads=1): ...
    def map(self, tasks, *, ordered=True) -> List[EvalRecord]: ...
    def close(self) -> None: ...
    def terminate(self) -> None: ...

def evaluate_candidates(build_problem, candidates, *, fn="evaluate",
                        n_workers=1, seed=None, ordered=True, tf_threads=1
                        ) -> List[EvalRecord]
def make_evaluator(build_problem, n_workers=1, seed=None, *,
                   fn="evaluate", tf_threads=1
                   ) -> Callable[..., Sequence[float]]
    # у callable есть .close(); инъекция в global_search(evaluator=...)
```

`tf_threads` — ограничение intra-op и inter-op пулов потоков TensorFlow в
каждом worker-процессе (по умолчанию 1 — защита от CPU oversubscription
между n_workers процессами); `None` — не трогать потоки TF (поведение до
появления параметра). Если пулы TF в worker'е уже инициализированы и
настройка бросает `RuntimeError`, worker остаётся на дефолтной
конфигурации потоков (fallback) — семантика оценки не меняется. При
`n_workers=1` параметр валидируется, но игнорируется (SerialPool потоки
родителя не трогает); недопустимые значения (< 1) — `ValueError`.

`build_problem` — picklable фабрика уровня модуля; исключение кандидата
даёт `loss=+inf` и `error`, остальные вычисляются; один seed → одинаковые
потери в SerialPool и ProcessPool.

### Диагностика градиентов

```python
def finite_difference_scan(func, phi0, h_values=None) -> Dict[str, object]
def taylor_test(value_and_grad, phi0, v=None, h_values=None) -> Dict[str, object]
def dot_product_test(jvp_fn, vjp_fn, x, y, tol=1e-5) -> Dict[str, object]
def dt_convergence(build_and_run, dts) -> List[Dict[str, object]]
def dtype_consistency_check(build_and_run, dtypes=(tf.float32, tf.float64)) -> List[Dict[str, object]]
def compare_bptt_adjoint(network, loss_fn, t_seq, inputs=None, target=None, initial_state=None) -> Dict[str, object]
```

- `finite_difference_scan` — центральные конечные разности градиента со сканированием h
- `taylor_test` — направленный тест порядков 1/2 (медианные наклоны ≈ 1 и ≈ 2)
- `dot_product_test` — сопряжённость пар операторов: `<Jx, y>` против `<x, Jᵀy>`
- `dt_convergence` — сходимость потери/градиентов по уточняющейся сетке dt (не для BrunelFokkerPlanck)
- `dtype_consistency_check` — согласованность float32/float64 через config.set_dtype
- `compare_bptt_adjoint` — равенство потери и градиентов BPTT и adjoint на одной сети

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

Экспорты (P0/P1):
- `BaseIntegrator`
- `EulerIntegrator`
- `HeunIntegrator`
- `RK4Integrator`

Экспорты P2 (жёсткие системы, см. `docs/integrators.md`):
- `AdaptiveRKIntegrator`
- `IMEXEulerIntegrator`
- `RosenbrockIntegrator`
- `ExponentialEulerIntegrator`
- `stiffness_report`
- `classify_failure`

---

## neuraltide.synapses.__init__

Экспорты:
- `StaticSynapse`
- `NMDASynapse`
- `TsodyksMarkramSynapse`
- `SecondOrderSynapse`
- `CompositeSynapse`

---

## neuraltide.populations.__init__

Экспорты:
- `IzhikevichMeanField`
- `WilsonCowan`
- `FokkerPlanckPopulation`
- `BrunelFokkerPlanck`
- `InputPopulation` (deprecated)

---

## neuraltide.inputs.__init__

Экспорты:
- `BaseInputGenerator`
- `SinusoidalGenerator`
- `ConstantRateGenerator`
- `VonMisesGenerator`
