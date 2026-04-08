# NeuralTide: Техническая документация v0.1.0

## Полная спецификация для кодогенерации

---

# Часть I. Обзор проекта

## 1.1 Назначение

`neuraltide` — Python-пакет для дифференцируемого моделирования и обучения популяционных нейронных сетей. Основная идея: популяционная частота разрядов непрерывна, что устраняет проблему недифференцируемости спайков. Пакет позволяет строить модели сетей мозга из популяций нейронов (Wilson–Cowan, Izhikevich mean-field, Fokker–Planck и т.д.), соединять популяции синапсами с произвольной динамикой (включая NMDA), задавать обучаемые и фиксированные параметры, запускать оптимизацию через backpropagation through time (BPTT).

## 1.2 Целевая аудитория

Нейробиологи и вычислительные нейробиологи, строящие популяционные модели сетей мозга.

## 1.3 Ключевые принципы

- **Разделение ответственности**: популяционная модель, синаптическая модель, интегратор, граф сети, тренер — независимые абстракции.
- **Расширяемость**: пользователь наследуется от базовых классов и реализует минимальный контракт.
- **Differentiable-first**: все вычисления через TensorFlow ops для автоматического градиента.
- **Батчирование по популяциям**: `n_units` в одном объекте `PopulationModel` означает число независимых популяций одного типа, обрабатываемых одной матричной операцией.
- **Научная воспроизводимость**: фиксация seeds, версий библиотек, полная сериализация эксперимента.

## 1.4 Технический стек

- **Python**: >= 3.12
- **Backend**: TensorFlow >= 2.16
- **Дифференцирование**: backpropagation through time (BPTT) через явный пошаговый интегратор внутри `tf.keras.layers.RNN`
- **Устройства**: CPU, GPU, multi-GPU (через `tf.distribute`)
- **Зависимости**: numpy >= 1.26, scipy >= 1.12

---

# Часть II. Архитектура

## 2.1 Структура репозитория

```
neuraltide/
│
├── core/
│   ├── __init__.py
│   ├── base.py          # Базовые классы: PopulationModel, SynapseModel, BaseInputGenerator
│   ├── network.py       # NetworkGraph, NetworkRNNCell, NetworkRNN, NetworkOutput
│   ├── state.py         # NetworkState
│   └── types.py         # DTYPE, get_pi(), алиасы типов
│
├── populations/
│   ├── __init__.py
│   ├── izhikevich_mf.py    # IzhikevichMeanField
│   ├── wilson_cowan.py     # WilsonCowan
│   ├── fokker_planck.py    # FokkerPlanckPopulation
│   └── input_population.py # InputPopulation
│
├── synapses/
│   ├── __init__.py
│   ├── tsodyks_markram.py  # TsodyksMarkramSynapse
│   ├── nmda.py             # NMDASynapse
│   ├── static.py           # StaticSynapse
│   └── composite.py        # CompositeSynapse
│
├── inputs/
│   ├── __init__.py
│   ├── base.py             # BaseInputGenerator
│   ├── von_mises.py        # VonMisesGenerator
│   ├── sinusoidal.py       # SinusoidalGenerator
│   └── constant.py         # ConstantRateGenerator
│
├── integrators/
│   ├── __init__.py
│   ├── base.py             # BaseIntegrator
│   ├── euler.py            # EulerIntegrator
│   ├── heun.py             # HeunIntegrator
│   └── rk4.py              # RK4Integrator
│
├── training/
│   ├── __init__.py
│   ├── losses.py           # BaseLoss, MSELoss, StabilityPenalty,
│   │                       # L2RegularizationLoss, ParameterBoundLoss, CompositeLoss
│   ├── readouts.py         # BaseReadout, IdentityReadout, LinearReadout,
│   │                       # BandpassReadout, LFPProxyReadout, HemodynamicReadout
│   ├── trainer.py          # Trainer, TrainingHistory
│   └── callbacks.py        # DivergenceDetector, GradientMonitor, ExperimentLogger
│
├── constraints/
│   ├── __init__.py
│   └── param_constraints.py  # MinMaxConstraint, NonNegConstraint, UnitIntervalConstraint
│
├── utils/
│   ├── __init__.py
│   ├── summary.py            # print_summary()
│   ├── reproducibility.py    # seed_everything(), log_versions(), save_experiment_state()
│   └── sparse.py             # SparseMask (v0.2)
│
├── config/
│   ├── __init__.py           # set_dtype(), get_dtype(),
│   │                         # register_population(), register_synapse(),
│   │                         # POPULATION_REGISTRY, SYNAPSE_REGISTRY, INPUT_REGISTRY
│   └── schema.py             # NetworkConfig, PopulationConfig, SynapseConfig,
│                             # InputConfig, build_network_from_config()
│
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_config.py
│   │   ├── test_constraints.py
│   │   ├── test_populations.py
│   │   ├── test_synapses.py
│   │   ├── test_integrators.py
│   │   ├── test_network.py
│   │   ├── test_inputs.py
│   │   ├── test_losses.py
│   │   ├── test_readouts.py
│   │   └── test_training.py
│   └── integration/
│       ├── test_gradient_check.py
│       ├── test_analytic_solutions.py
│       └── test_full_pipeline.py
│
├── examples/
│   ├── example_01_single_population.py
│   ├── example_02_exc_inh_nmda.py
│   ├── example_03_custom_population.py
│   ├── example_04_fokker_planck.py
│   └── example_05_lfp_proxy.py
│
├── pyproject.toml
├── README.md
└── .gitignore
```

## 2.2 Поток данных

```
t (scalar) ──► InputPopulation.get_firing_rate(state=[t])
                        │
                        ▼ pre_firing_rate
               SynapseModel.forward()  ◄── post_voltage (от целевой популяции)
                        │
                        ▼ I_syn, g_syn
               накопление токов по всем входящим синапсам
                        │
                        ▼ total_synaptic_input
               PopulationModel.derivatives()
                        │
                        ▼
               Integrator.step()  →  new_population_state
                        │
                        ▼
               NetworkOutput:
                 firing_rates: dict[str, Tensor[batch, T, n_units]]
                 hidden_states: dict[str, dict[str, Tensor]] (optional)
                 stability_loss: scalar Tensor
                        │
                        ▼
               ReadoutLayer  (optional preprocessing)
                        │
                        ▼
               Loss(predicted, target, model_params)
                        │
                        ▼
               Optimizer → updated trainable variables
```

---

# Часть III. Детальная спецификация API

## 3.1 `config/__init__.py`

```python
import tensorflow as tf

_DTYPE = tf.float32

def set_dtype(dtype: tf.DType) -> None:
    """Устанавливает глобальный тип данных. Должен вызываться до создания любых объектов."""
    global _DTYPE
    _DTYPE = dtype

def get_dtype() -> tf.DType:
    """Возвращает текущий глобальный тип данных."""
    return _DTYPE
```

**Реестры пользовательских классов:**

```python
from typing import Dict, Type

POPULATION_REGISTRY: Dict[str, Type] = {}   # заполняется при импорте встроенных классов
SYNAPSE_REGISTRY:    Dict[str, Type] = {}
INPUT_REGISTRY:      Dict[str, Type] = {}

def register_population(name: str, cls: Type) -> None:
    """Регистрирует пользовательский класс PopulationModel для config-first API."""
    POPULATION_REGISTRY[name] = cls

def register_synapse(name: str, cls: Type) -> None:
    """Регистрирует пользовательский класс SynapseModel."""
    SYNAPSE_REGISTRY[name] = cls

def register_input(name: str, cls: Type) -> None:
    """Регистрирует пользовательский класс BaseInputGenerator."""
    INPUT_REGISTRY[name] = cls
```

---

## 3.2 `core/types.py`

```python
import tensorflow as tf
from typing import Dict, List, Tuple, Optional, Any, Type
import neuraltide.config as _config

def get_pi() -> tf.Tensor:
    """Возвращает π с текущим глобальным dtype."""
    return tf.constant(3.14159265358979323846, dtype=_config.get_dtype())

# Алиасы типов
TensorType  = tf.Tensor
StateList   = List[TensorType]
ParamDict   = Dict[str, Any]
```

---

## 3.3 `core/base.py` — Базовые классы

### 3.3.1 `PopulationModel`

```python
class PopulationModel(tf.keras.layers.Layer):
    """
    Абстрактный базовый класс для популяционной модели.

    Семантика n_units:
        n_units — число независимых популяций одного типа внутри одного объекта.
        Например, IzhikevichMeanField(n_units=4) описывает 4 независимые
        популяции нейронов Ижикевича с (потенциально) разными параметрами.
        Все n_units популяций обрабатываются одной батчированной матричной
        операцией — один kernel launch на объект.

    Параметры:
        Все параметры регистрируются через self._make_param(params, name).
        Обучаемость, начальное значение и ограничения задаются через
        словарь params при создании экземпляра.

    Контракт подкласса:
        1. Установить self.state_size в __init__ до завершения super().__init__().
        2. Реализовать get_initial_state().
        3. Реализовать derivatives().
        4. Реализовать get_firing_rate().
        5. Опционально переопределить observables() для доступа к v_mean и т.д.
        6. Реализовать parameter_spec (property).

    Args:
        n_units (int): число независимых популяций данного типа.
        dt (float): шаг интегрирования в мс.
        name (str): имя слоя Keras.
    """

    def __init__(self, n_units: int, dt: float,
                 name: str = "population", **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_units = n_units
        self.dt = dt
        # state_size ДОЛЖЕН быть задан в подклассе:
        # self.state_size = [tf.TensorShape([1, n_units]), ...]

    def _make_param(self, params: dict, name: str) -> tf.Variable:
        """
        Регистрирует параметр популяции через add_weight().

        Форматы params[name]:

            Без словаря (только значение, trainable=False):
                params[name] = 0.5
                params[name] = [0.5, 0.6, 0.4, 0.7]

            Словарь с полным контролем:
                params[name] = {
                    'value':     0.5,           # скаляр или список длины n_units
                    'trainable': True,           # по умолчанию False
                    'min':       0.01,           # нижняя граница (опционально)
                    'max':       2.0,            # верхняя граница (опционально)
                }

        Правила broadcast:
            Скаляр → tf.fill([n_units], scalar)
            Список длины n_units → используется as-is
            Список другой длины → ValueError

        Returns:
            tf.Variable, зарегистрированная как вес слоя.

        Raises:
            ValueError: если name не найден в params.
            ValueError: если длина списка не совпадает с n_units.
        """
        spec = params.get(name)
        if spec is None:
            raise ValueError(
                f"PopulationModel '{self.name}': "
                f"parameter '{name}' not found in params."
            )
        if not isinstance(spec, dict):
            spec = {'value': spec, 'trainable': False}

        raw   = spec['value']
        train = spec.get('trainable', False)
        lo    = spec.get('min', None)
        hi    = spec.get('max', None)

        value = tf.constant(raw, dtype=neuraltide.config.get_dtype())
        if value.shape.rank == 0:
            value = tf.fill([self.n_units], value)
        else:
            if value.shape[0] != self.n_units:
                raise ValueError(
                    f"PopulationModel '{self.name}': parameter '{name}' "
                    f"has length {value.shape[0]}, expected {self.n_units}."
                )

        from neuraltide.constraints import MinMaxConstraint
        constraint = MinMaxConstraint(lo, hi) \
                     if (lo is not None or hi is not None) else None

        return self.add_weight(
            shape=(self.n_units,),
            initializer=tf.keras.initializers.Constant(value.numpy()),
            trainable=train,
            constraint=constraint,
            dtype=neuraltide.config.get_dtype(),
            name=name,
        )

    @abstractmethod
    def get_initial_state(self, batch_size: int = 1) -> StateList:
        """
        Возвращает список тензоров начального состояния.

        Каждый тензор совместим с соответствующим элементом self.state_size.
        Стандартная инициализация — нулевые тензоры.

        Returns:
            list of tf.Tensor. Для большинства моделей:
                [tf.zeros([1, n_units]), ...]  — по одному тензору на переменную состояния.
        """
        raise NotImplementedError

    @abstractmethod
    def derivatives(
        self,
        state: StateList,
        total_synaptic_input: Dict[str, TensorType],
    ) -> StateList:
        """
        Вычисляет производные состояния популяции.

        Все внешние воздействия поступают через синапсы и содержатся
        в total_synaptic_input. Прямого аргумента external_input нет —
        внешние входы подключаются через InputPopulation и синапсы.

        Args:
            state: текущее состояние — список тензоров.
            total_synaptic_input: словарь агрегированных синаптических сигналов:
                {
                    'I_syn': tf.Tensor [1, n_units],  # суммарный синаптический ток
                    'g_syn': tf.Tensor [1, n_units],  # суммарная проводимость
                }
                Если входящих синапсов нет, оба тензора равны нулю.
                'g_syn' используется моделями где g_syn_tot входит в уравнение
                явно (например, IzhikevichMeanField).

        Returns:
            list of tf.Tensor — производные состояния, той же структуры что state.

        Требования:
            - Должен быть traceable tf.function (без Python side effects).
            - Не должен изменять state на месте.
        """
        raise NotImplementedError

    @abstractmethod
    def get_firing_rate(self, state: StateList) -> TensorType:
        """
        Извлекает частоту разрядов из состояния популяции.

        Returns:
            tf.Tensor, shape = [1, n_units], в единицах Гц (спайков/с).
            Значения неотрицательны (гарантируется реализацией через relu или abs).
        """
        raise NotImplementedError

    def observables(self, state: StateList) -> Dict[str, TensorType]:
        """
        Возвращает словарь наблюдаемых переменных состояния.

        Обязательный ключ: 'firing_rate'.
        Опциональные ключи (если модель поддерживает):
            'v_mean'  — средний мембранный потенциал, shape [1, n_units].
                        Используется NMDASynapse для расчёта магниевого блока.
            'w_mean'  — среднее адаптационное переменное.
            'lfp_proxy' — прокси LFP.

        Если 'v_mean' отсутствует, NetworkRNNCell передаёт
        tf.zeros([1, n_units]) в post_voltage синапсов.

        По умолчанию возвращает только {'firing_rate': get_firing_rate(state)}.
        Подкласс переопределяет для добавления дополнительных наблюдаемых.
        """
        return {'firing_rate': self.get_firing_rate(state)}

    @property
    @abstractmethod
    def parameter_spec(self) -> Dict[str, Dict[str, Any]]:
        """
        Спецификация параметров модели для summary и сериализации.

        Returns:
            {
                'param_name': {
                    'shape':      tuple,
                    'trainable':  bool,
                    'constraint': str or None,
                    'units':      str,   # физические единицы: 'mV', 'ms', 'Hz', ...
                },
                ...
            }
        """
        raise NotImplementedError
```

---

### 3.3.2 `SynapseModel`

```python
class SynapseModel(tf.keras.layers.Layer):
    """
    Абстрактный базовый класс для синаптической модели.

    Синапс описывает проекцию от n_pre пресинаптических популяций
    к n_post постсинаптическим популяциям.

    Матрица параметров:
        Все весовые матрицы имеют форму [n_pre, n_post].
        Элемент [i, j] описывает связь от пресинаптической популяции i
        к постсинаптической популяции j.

    Маскирование через pconn:
        pconn [n_pre, n_post] — матрица вероятностей/масок соединений,
        trainable=False. Применяется к firing_probs в forward():
            FRpre_normed = pconn * firing_probs_broadcast
        Нулевой элемент pconn[i,j] означает отсутствие связи i→j.
        gsyn_max задаётся пользователем полностью; обнуление несуществующих
        связей обеспечивается через pconn, а не через gsyn_max.

    Рекуррентные синапсы:
        Допустимы (src == tgt, n_pre == n_post).
        Диагональные элементы pconn управляют самовозбуждением.

    Args:
        n_pre (int): число пресинаптических популяций.
        n_post (int): число постсинаптических популяций.
        dt (float): шаг интегрирования в мс.
        name (str): имя слоя Keras.
    """

    def __init__(self, n_pre: int, n_post: int, dt: float,
                 name: str = "synapse", **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_pre  = n_pre
        self.n_post = n_post
        self.dt     = dt
        # state_size ДОЛЖЕН быть задан в подклассе

    def _make_param(self, params: dict, name: str) -> tf.Variable:
        """
        Регистрирует параметр синапса через add_weight().

        Форматы params[name] и правила broadcast до [n_pre, n_post]:

            Скаляр:
                {'value': 0.1, 'trainable': True}
                → tf.fill([n_pre, n_post], 0.1)

            Вектор длины n_pre:
                {'value': [0.1, 0.2, 0.15, 0.1], 'trainable': True}
                → reshape [n_pre, 1] → broadcast [n_pre, n_post]
                Интерпретация: параметр зависит от пресинаптической популяции.

            Вектор длины n_post (только если n_pre != n_post):
                {'value': [0.1, 0.2, 0.15], 'trainable': True}
                → reshape [1, n_post] → broadcast [n_pre, n_post]
                Интерпретация: параметр зависит от постсинаптической популяции.
                ВАЖНО: если n_pre == n_post, вектор трактуется как [n_pre].
                Для явного [n_post] при n_pre==n_post используй матрицу [1, n_post].

            Матрица [n_pre, n_post]:
                {'value': [[0.1, 0.2], [0.0, 0.3]], 'trainable': True}
                → используется as-is.

        Raises:
            ValueError: если name не найден в params.
            ValueError: если форма значения несовместима с [n_pre, n_post].
        """
        spec = params.get(name)
        if spec is None:
            raise ValueError(
                f"SynapseModel '{self.name}': "
                f"parameter '{name}' not found in params."
            )
        if not isinstance(spec, dict):
            spec = {'value': spec, 'trainable': False}

        raw   = spec['value']
        train = spec.get('trainable', False)
        lo    = spec.get('min', None)
        hi    = spec.get('max', None)

        value = tf.constant(raw, dtype=neuraltide.config.get_dtype())
        value = self._broadcast_to_matrix(value, name)   # → [n_pre, n_post]

        from neuraltide.constraints import MinMaxConstraint
        constraint = MinMaxConstraint(lo, hi) \
                     if (lo is not None or hi is not None) else None

        return self.add_weight(
            shape=(self.n_pre, self.n_post),
            initializer=tf.keras.initializers.Constant(value.numpy()),
            trainable=train,
            constraint=constraint,
            dtype=neuraltide.config.get_dtype(),
            name=name,
        )

    def _broadcast_to_matrix(self, value: TensorType, name: str) -> TensorType:
        """
        Приводит тензор произвольной формы к [n_pre, n_post].

        Raises:
            ValueError: при несовместимой форме.
        """
        rank = value.shape.rank

        if rank == 0:
            return tf.broadcast_to(
                tf.reshape(value, [1, 1]),
                [self.n_pre, self.n_post]
            )

        elif rank == 1:
            length = int(value.shape[0])
            if length == self.n_pre:
                return tf.broadcast_to(
                    tf.reshape(value, [self.n_pre, 1]),
                    [self.n_pre, self.n_post]
                )
            elif length == self.n_post and self.n_pre != self.n_post:
                return tf.broadcast_to(
                    tf.reshape(value, [1, self.n_post]),
                    [self.n_pre, self.n_post]
                )
            else:
                raise ValueError(
                    f"SynapseModel '{self.name}': parameter '{name}' "
                    f"has length {length}, expected {self.n_pre} (n_pre) "
                    f"or {self.n_post} (n_post)."
                )

        elif rank == 2:
            if value.shape != (self.n_pre, self.n_post):
                raise ValueError(
                    f"SynapseModel '{self.name}': parameter '{name}' "
                    f"has shape {value.shape}, "
                    f"expected [{self.n_pre}, {self.n_post}]."
                )
            return value

        else:
            raise ValueError(
                f"SynapseModel '{self.name}': parameter '{name}' "
                f"has unsupported rank {rank}. "
                f"Expected scalar, 1D vector, or 2D matrix."
            )

    @abstractmethod
    def get_initial_state(self, batch_size: int = 1) -> StateList:
        """Возвращает список тензоров начального состояния синапса."""
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        pre_firing_rate: TensorType,
        post_voltage:    TensorType,
        state:           StateList,
        dt:              float,
    ) -> Tuple[Dict[str, TensorType], StateList]:
        """
        Вычисляет новое состояние синапса и возвращает синаптический ток.

        Args:
            pre_firing_rate: частота пресинаптических популяций.
                shape = [1, n_pre], в Гц.
            post_voltage: средний потенциал постсинаптических популяций.
                shape = [1, n_post].
                Если целевая популяция не предоставляет 'v_mean' в observables(),
                NetworkRNNCell передаёт tf.zeros([1, n_post]).
                Используется NMDASynapse; другие синапсы игнорируют этот аргумент.
            state: текущее внутреннее состояние синапса.
            dt: шаг интегрирования в мс.

        Returns:
            Tuple:
                current_dict: {
                    'I_syn': tf.Tensor [1, n_post],  # суммарный ток в постсинаптику
                    'g_syn': tf.Tensor [1, n_post],  # суммарная проводимость
                }
                new_state: list of tf.Tensor — новое состояние синапса.

        Требования:
            - Должен быть traceable tf.function.
            - I_syn и g_syn должны иметь форму [1, n_post].
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def parameter_spec(self) -> Dict[str, Dict[str, Any]]:
        """Аналогично PopulationModel.parameter_spec."""
        raise NotImplementedError
```

---

### 3.3.3 `BaseInputGenerator`

```python
class BaseInputGenerator(tf.keras.layers.Layer):
    """
    Базовый класс для генератора входных сигналов.

    Генератор оборачивается в InputPopulation и подключается к
    динамическим популяциям через полноценные синапсы (SynapseModel).

    Семантика n_units:
        n_units — число независимых входных каналов одного типа.
        Например, VonMisesGenerator с n_units=4 описывает 4 независимых
        входных сигнала с (потенциально) разными параметрами.
        Все n_units каналов обрабатываются одной векторизованной операцией.

    Параметры:
        Все параметры регистрируются через self._make_param(params, name).
        Формат аналогичен PopulationModel:
            - скаляр (broadcast к n_units)
            - список длины n_units
            - словарь {'value': ..., 'trainable': bool, 'min':, 'max':}

    Args:
        params: словарь параметров генератора.
        dt: шаг интегрирования в мс.
        name: имя слоя Keras.
    """

    def __init__(self, params: Dict[str, Any], dt: float,
                 name: str = "input_generator", **kwargs):
        super().__init__(name=name, **kwargs)
        self.dt = dt
        self._params = params

        self._infer_n_units_from_params()
        self._validate_param_dimensions()

        self.n_units = self._n_units

    def _infer_n_units_from_params(self) -> None:
        """Определяет n_units из размерности параметров."""
        max_len = 1
        for key, spec in self._params.items():
            if isinstance(spec, dict):
                value = spec.get('value', None)
            else:
                value = spec
            if value is not None and isinstance(value, (list, tuple)):
                max_len = max(max_len, len(value))
        self._n_units = max_len

    def _make_param(self, params: dict, name: str) -> tf.Variable:
        """
        Регистрирует параметр генератора через add_weight().
        """
        raise NotImplementedError

    @abstractmethod
    def call(self, t: TensorType) -> TensorType:
        """
        Args:
            t: текущее время в мс. shape = [batch, 1].

        Returns:
            tf.Tensor, shape = [batch, n_units], в Гц.
        """
        raise NotImplementedError
```

---

## 3.4 `populations/input_population.py`

```python
class InputPopulation(PopulationModel):
    """
    Псевдо-популяция без динамики. Оборачивает BaseInputGenerator.

    Назначение:
        Позволяет подключать внешние входные сигналы к сети через
        полноценные обучаемые синапсы, единообразно с рекуррентными
        проекциями между динамическими популяциями.

    Семантика:
        - Не имеет уравнений динамики (derivatives() возвращает []).
        - Не обновляется интегратором.
        - Состояние: [t_current], shape [1, 1] — текущее время в мс.
        - NetworkRNNCell обновляет состояние напрямую: new_state = [t].
        - get_firing_rate(state) вызывает generator(state[0]).
        - Не может быть целью синапса (только источником).

    Note:
        n_units и dt берутся из generator автоматически.
        Пользователь не задаёт их вручную.
    """

    def __init__(self, generator: BaseInputGenerator, **kwargs):
        super().__init__(
            n_units=generator.n_units,
            dt=generator.dt,
            **kwargs
        )
        self.generator = generator
        self.state_size = [tf.TensorShape([1, 1])]   # [t_current]

    def get_initial_state(self, batch_size: int = 1) -> StateList:
        return [tf.zeros([1, 1], dtype=neuraltide.config.get_dtype())]

    def derivatives(self, state, total_synaptic_input) -> list:
        """Нет динамики. NetworkRNNCell не вызывает integrator для InputPopulation."""
        return []

    def get_firing_rate(self, state: StateList) -> TensorType:
        """
        Возвращает выход генератора для текущего времени.

        Args:
            state: [t_current], где t_current shape [1, 1].

        Returns:
            tf.Tensor, shape [1, n_units], в Гц.
        """
        t = state[0]   # [1, 1]
        rate = self.generator(t)
        return tf.reshape(rate, [1, self.n_units])

    def observables(self, state: StateList) -> Dict[str, TensorType]:
        return {'firing_rate': self.get_firing_rate(state)}

    @property
    def parameter_spec(self) -> Dict[str, Dict[str, Any]]:
        return {}
```

---

## 3.5 `core/state.py`

```python
@dataclass
class NetworkState:
    """
    Полное состояние сети в один момент времени.

    Атрибуты:
        population_states: dict[str, StateList]
            Ключи — имена популяций в порядке регистрации в NetworkGraph.
            Включает как динамические популяции, так и InputPopulation.
        synapse_states: dict[str, StateList]
            Ключи — имена синаптических проекций в порядке регистрации.

    Сериализация:
        Плоский список для tf.keras.layers.RNN строится в фиксированном порядке:
        сначала все популяции (в порядке регистрации),
        затем все синапсы (в порядке регистрации),
        затем один скалярный тензор stability_error shape [1].
    """
    population_states: Dict[str, StateList]
    synapse_states:    Dict[str, StateList]

    def to_flat_list(self) -> StateList:
        """
        Сериализует в плоский список тензоров для RNN state.
        Порядок: популяции → синапсы → stability_error.
        """
        flat = []
        for state_list in self.population_states.values():
            flat.extend(state_list)
        for state_list in self.synapse_states.values():
            flat.extend(state_list)
        return flat

    @staticmethod
    def from_flat_list(
        flat: StateList,
        pop_names:    List[str],
        pop_sizes:    Dict[str, int],   # name → число тензоров в state
        syn_names:    List[str],
        syn_sizes:    Dict[str, int],
    ) -> 'NetworkState':
        """Десериализует из плоского списка."""
        idx = 0
        pop_states = {}
        for name in pop_names:
            n = pop_sizes[name]
            pop_states[name] = flat[idx:idx+n]
            idx += n
        syn_states = {}
        for name in syn_names:
            n = syn_sizes[name]
            syn_states[name] = flat[idx:idx+n]
            idx += n
        return NetworkState(
            population_states=pop_states,
            synapse_states=syn_states,
        )
```

---

## 3.6 `core/network.py`

### 3.6.1 `NetworkGraph`

```python
@dataclass
class _SynapseEntry:
    model: SynapseModel
    src:   str
    tgt:   str

class NetworkGraph:
    """
    Описание топологии сети: популяции и синаптические проекции.

    Популяции двух видов:
        - Динамические: IzhikevichMeanField, WilsonCowan и т.д.
        - Входные: InputPopulation (оборачивает BaseInputGenerator).

    Синаптические проекции:
        Любой тип SynapseModel от любой популяции (включая InputPopulation)
        к любой динамической популяции.
        InputPopulation не может быть целью синапса.

    Матрица связей:
        Концептуально [N+K, N], где N — число динамических популяций
        (суммарно по n_units), K — число входных популяций.
        Реализуется как набор матриц [n_pre_i, n_post_j]
        для каждой проекции.

    Args:
        dt (float): шаг интегрирования в мс.
            Используется при создании InputPopulation через add_input_population().

    Пример:
        graph = NetworkGraph(dt=0.5)
        graph.add_input_population('theta', VonMisesGenerator(params))
        graph.add_population('exc', IzhikevichMeanField(n_units=4, dt=0.5, params=...))
        graph.add_population('inh', IzhikevichMeanField(n_units=2, dt=0.5, params=...))
        graph.add_synapse('theta->exc', TsodyksMarkramSynapse(...), src='theta', tgt='exc')
        graph.add_synapse('exc->inh',  TsodyksMarkramSynapse(...), src='exc',   tgt='inh')
        graph.add_synapse('exc->exc',  TsodyksMarkramSynapse(...), src='exc',   tgt='exc')
    """

    def __init__(self, dt: float):
        self.dt = dt
        self._populations: OrderedDict[str, PopulationModel]  = OrderedDict()
        self._synapses:    OrderedDict[str, _SynapseEntry]    = OrderedDict()

    def add_population(self, name: str, model: PopulationModel) -> None:
        """
        Регистрирует популяцию (динамическую или InputPopulation).

        Args:
            name: уникальный идентификатор.
            model: экземпляр PopulationModel.

        Raises:
            ValueError: если имя уже занято.
        """
        if name in self._populations:
            raise ValueError(f"Population '{name}' already registered.")
        self._populations[name] = model

    def add_input_population(
        self,
        name:      str,
        generator: BaseInputGenerator,
    ) -> None:
        """
        Регистрирует входной генератор как псевдо-популяцию.

        Создаёт InputPopulation(generator, dt=self.dt) и вызывает add_population().
        После регистрации к входу подключаются синапсы через обычный add_synapse().

        Args:
            name: уникальный идентификатор входа.
            generator: экземпляр BaseInputGenerator.
                n_units входной популяции = generator.n_outputs.

        Пример:
            graph.add_input_population('theta', VonMisesGenerator(params))
            graph.add_synapse('theta->exc',
                              TsodyksMarkramSynapse(n_pre=1, n_post=4, ...),
                              src='theta', tgt='exc')
        """
        pop = InputPopulation(generator=generator, dt=self.dt,
                              name=name + '_input_pop')
        self.add_population(name, pop)

    def add_synapse(
        self,
        name:  str,
        model: SynapseModel,
        src:   str,
        tgt:   str,
    ) -> None:
        """
        Регистрирует синаптическую проекцию.

        Args:
            name: уникальный идентификатор проекции.
            model: экземпляр SynapseModel.
            src: имя пресинаптической популяции.
            tgt: имя постсинаптической популяции.

        Note:
            Допускается несколько проекций между одной парой популяций
            (например, AMPA и NMDA). Каждая регистрируется под уникальным именем.
            Рекуррентные синапсы (src == tgt) разрешены.

        Raises:
            ValueError: если name уже занято.
            ValueError: если src или tgt не зарегистрированы.
            ValueError: если tgt является InputPopulation.
        """
        if name in self._synapses:
            raise ValueError(f"Synapse '{name}' already registered.")
        if src not in self._populations:
            raise ValueError(f"Synapse '{name}': src '{src}' not registered.")
        if tgt not in self._populations:
            raise ValueError(f"Synapse '{name}': tgt '{tgt}' not registered.")
        if isinstance(self._populations[tgt], InputPopulation):
            raise ValueError(
                f"Synapse '{name}': InputPopulation '{tgt}' "
                f"cannot be a synaptic target."
            )
        self._synapses[name] = _SynapseEntry(model=model, src=src, tgt=tgt)

    def validate(self) -> None:
        """
        Проверяет корректность топологии перед построением NetworkRNN.

        Проверки:
            1. Все ссылки src/tgt в синапсах разрешаются (уже проверено в add_synapse).
            2. model.n_pre совпадает с n_units популяции-источника.
            3. model.n_post совпадает с n_units популяции-цели.
            4. Предупреждение (не ошибка): динамические популяции без входящих синапсов.

        Raises:
            ValueError: при несовместимости размерностей n_pre/n_post.
        """
        for syn_name, entry in self._synapses.items():
            src_pop = self._populations[entry.src]
            tgt_pop = self._populations[entry.tgt]
            if entry.model.n_pre != src_pop.n_units:
                raise ValueError(
                    f"Synapse '{syn_name}': n_pre={entry.model.n_pre} "
                    f"!= src '{entry.src}' n_units={src_pop.n_units}."
                )
            if entry.model.n_post != tgt_pop.n_units:
                raise ValueError(
                    f"Synapse '{syn_name}': n_post={entry.model.n_post} "
                    f"!= tgt '{entry.tgt}' n_units={tgt_pop.n_units}."
                )
        # Предупреждение об изолированных динамических популяциях
        for pop_name, pop in self._populations.items():
            if isinstance(pop, InputPopulation):
                continue
            has_input = any(
                e.tgt == pop_name for e in self._synapses.values()
            )
            if not has_input:
                import warnings
                warnings.warn(
                    f"Population '{pop_name}' has no incoming synapses.",
                    UserWarning
                )

    @property
    def population_names(self) -> List[str]:
        return list(self._populations.keys())

    @property
    def synapse_names(self) -> List[str]:
        return list(self._synapses.keys())

    @property
    def dynamic_population_names(self) -> List[str]:
        """Имена только динамических популяций (без InputPopulation)."""
        return [
            name for name, pop in self._populations.items()
            if not isinstance(pop, InputPopulation)
        ]

    @property
    def input_population_names(self) -> List[str]:
        """Имена только входных популяций."""
        return [
            name for name, pop in self._populations.items()
            if isinstance(pop, InputPopulation)
        ]
```

---

### 3.6.2 `NetworkRNNCell`

```python
class NetworkRNNCell(tf.keras.layers.AbstractRNNCell):
    """
    RNN-ячейка, реализующая один шаг динамики сети.

    Используется внутри NetworkRNN через tf.keras.layers.RNN.
    Пользователь напрямую не создаёт этот класс.

    Входной сигнал:
        inputs: tf.Tensor shape [batch, 1] — текущее время t в мс.
        Временна́я ось подаётся как входная последовательность RNN.

    Логика одного шага:
        1. Распаковать states → NetworkState.
        2. Обновить InputPopulation: new_state = [t].
        3. Для каждой синаптической проекции:
            a. pre_rate  = src_pop.get_firing_rate(src_state)  [1, n_pre]
            b. post_v    = tgt_pop.observables(tgt_state).get('v_mean',
                                               zeros [1, n_post])
            c. (current_dict, new_syn_state) = syn.forward(
                                               pre_rate, post_v, syn_state, dt)
            d. Накопить I_syn и g_syn для целевой популяции.
        4. Для каждой динамической популяции:
            a. total_syn = {'I_syn': accumulated_I, 'g_syn': accumulated_g}
            b. (new_pop_state, local_err) = integrator.step(
                                            pop, pop_state, total_syn)
            c. stability_error += local_err
        5. Упаковать NetworkState → flat list.
        6. output = concat([pop.get_firing_rate(state) for dynamic_pops],
                            axis=-1)  shape [1, total_n_dynamic_units]
           + stability_error как последний элемент output
           → output shape [1, total_n_dynamic_units + 1]

    state_size:
        Плоский список TensorShape всех состояний популяций и синапсов
        в порядке регистрации.

    output_size:
        sum(n_units для динамических популяций) + 1 (stability_error).
    """

    def __init__(self, graph: NetworkGraph,
                 integrator: 'BaseIntegrator', **kwargs):
        super().__init__(**kwargs)
        graph.validate()
        self.graph      = graph
        self.integrator = integrator
        self._build_size_maps()

    def _build_size_maps(self) -> None:
        """Вычисляет state_size и вспомогательные словари размеров."""
        self._pop_state_sizes: Dict[str, int] = {}
        self._syn_state_sizes: Dict[str, int] = {}

        flat_state_size = []
        for name, pop in self.graph._populations.items():
            n = len(pop.state_size)
            self._pop_state_sizes[name] = n
            flat_state_size.extend(pop.state_size)

        for name, entry in self.graph._synapses.items():
            n = len(entry.model.state_size)
            self._syn_state_sizes[name] = n
            flat_state_size.extend(entry.model.state_size)

        self._flat_state_size = flat_state_size

        self._total_dynamic_units = sum(
            pop.n_units
            for name, pop in self.graph._populations.items()
            if not isinstance(pop, InputPopulation)
        )

    @property
    def state_size(self) -> List:
        return self._flat_state_size

    @property
    def output_size(self) -> int:
        # dynamic firing rates + stability_error
        return self._total_dynamic_units + 1

    def get_initial_state(self, inputs=None,
                          batch_size: int = 1,
                          dtype=None) -> StateList:
        """Собирает начальные состояния всех популяций и синапсов."""
        flat = []
        for pop in self.graph._populations.values():
            flat.extend(pop.get_initial_state(batch_size))
        for entry in self.graph._synapses.values():
            flat.extend(entry.model.get_initial_state(batch_size))
        return flat

    def call(
        self,
        inputs: TensorType,
        states: StateList,
    ) -> Tuple[TensorType, StateList]:
        """
        Args:
            inputs: shape [batch, 1] — текущее время t в мс.
            states: flat list тензоров состояния.

        Returns:
            (output, new_states):
                output shape [batch, total_n_dynamic_units + 1].
                new_states: обновлённый flat list.
        """
        dtype = neuraltide.config.get_dtype()
        t     = inputs   # [batch, 1]

        # 1. Распаковать состояние
        net_state = NetworkState.from_flat_list(
            flat      = list(states),
            pop_names = self.graph.population_names,
            pop_sizes = self._pop_state_sizes,
            syn_names = self.graph.synapse_names,
            syn_sizes = self._syn_state_sizes,
        )

        # 2. Обновить InputPopulation (состояние = текущее t)
        for name, pop in self.graph._populations.items():
            if isinstance(pop, InputPopulation):
                net_state.population_states[name] = [t]

        # 3. Накопить синаптические токи
        syn_I: Dict[str, TensorType] = {}
        syn_g: Dict[str, TensorType] = {}
        for name in self.graph.dynamic_population_names:
            n = self.graph._populations[name].n_units
            syn_I[name] = tf.zeros([1, n], dtype=dtype)
            syn_g[name] = tf.zeros([1, n], dtype=dtype)

        for syn_name, entry in self.graph._synapses.items():
            src_pop   = self.graph._populations[entry.src]
            tgt_pop   = self.graph._populations[entry.tgt]
            src_state = net_state.population_states[entry.src]
            tgt_state = net_state.population_states[entry.tgt]
            syn_state = net_state.synapse_states[syn_name]

            pre_rate = src_pop.get_firing_rate(src_state)   # [1, n_pre]

            tgt_obs  = tgt_pop.observables(tgt_state)
            post_v   = tgt_obs.get(
                'v_mean',
                tf.zeros([1, tgt_pop.n_units], dtype=dtype)
            )                                                # [1, n_post]

            current_dict, new_syn_state = entry.model.forward(
                pre_rate, post_v, syn_state, entry.model.dt
            )

            syn_I[entry.tgt] += current_dict['I_syn']
            syn_g[entry.tgt] += current_dict['g_syn']
            net_state.synapse_states[syn_name] = new_syn_state

        # 4. Обновить динамические популяции
        stability_error = tf.zeros([1], dtype=dtype)
        for name in self.graph.dynamic_population_names:
            pop       = self.graph._populations[name]
            pop_state = net_state.population_states[name]
            total_syn = {'I_syn': syn_I[name], 'g_syn': syn_g[name]}

            new_pop_state, local_err = self.integrator.step(
                pop, pop_state, total_syn
            )
            net_state.population_states[name] = new_pop_state
            stability_error += local_err

        # 5. Собрать выход
        rates = tf.concat(
            [
                self.graph._populations[name].get_firing_rate(
                    net_state.population_states[name]
                )
                for name in self.graph.dynamic_population_names
            ],
            axis=-1
        )   # [1, total_n_dynamic_units]

        output = tf.concat(
            [rates, stability_error[tf.newaxis, :]],
            axis=-1
        )   # [1, total_n_dynamic_units + 1]

        # 6. Упаковать новое состояние
        new_states_flat = net_state.to_flat_list()

        return output, new_states_flat
```

---

### 3.6.3 `NetworkRNN` и `NetworkOutput`

```python
@dataclass
class NetworkOutput:
    """
    Результат прогона NetworkRNN.

    Атрибуты:
        firing_rates: dict[str, tf.Tensor]
            Ключи — имена динамических популяций.
            Форма каждого тензора: [batch, T, n_units_i].
            Единицы: Гц.
        hidden_states: dict[str, dict[str, tf.Tensor]] или None.
            Внешний ключ — имя популяции.
            Внутренний ключ — имя наблюдаемой ('firing_rate', 'v_mean', ...).
            None если return_hidden_states=False.
        stability_loss: tf.Tensor, scalar.
            Суммарная оценка численной ошибки интегрирования по всем шагам.
            0.0 если используется EulerIntegrator.
    """
    firing_rates:    Dict[str, TensorType]
    hidden_states:   Optional[Dict[str, Dict[str, TensorType]]]
    stability_loss:  TensorType


class NetworkRNN(tf.keras.layers.Layer):
    """
    Обёртка над tf.keras.layers.RNN для симуляции сети на временно́й оси.

    Args:
        graph (NetworkGraph): топология сети.
        integrator (BaseIntegrator): интегратор ОДУ.
        return_sequences (bool): True — все шаги, False — только последний.
            По умолчанию True.
        return_hidden_states (bool): если True, возвращает полные внутренние
            состояния популяций в NetworkOutput.hidden_states.
            Используется для доступа к v_mean, LFP-прокси и т.д.
            По умолчанию False.
        stability_penalty_weight (float): вес штрафа за численную нестабильность.
            0.0 отключает штраф. По умолчанию 0.0.
            Штраф вычисляется как ||RK4 - Heun|| внутри RK4Integrator
            и ||Heun - Euler|| внутри HeunIntegrator.
            EulerIntegrator всегда возвращает 0.0.

    Метод call:
        Args:
            t_sequence: tf.Tensor, shape [batch, T, 1], время в мс.
            initial_state: list of tf.Tensor или None.
                None → использует NetworkRNNCell.get_initial_state().
        Returns:
            NetworkOutput.

    Пример:
        network = NetworkRNN(graph, integrator=RK4Integrator(),
                             stability_penalty_weight=1e-3)
        t_seq   = tf.constant(np.arange(2000)*0.5, dtype=tf.float32)[None, :, None]
        output  = network(t_seq)
        # output.firing_rates['exc'] shape: [1, 2000, 4]
    """

    def __init__(
        self,
        graph:                    NetworkGraph,
        integrator:               'BaseIntegrator',
        return_sequences:         bool  = True,
        return_hidden_states:     bool  = False,
        stability_penalty_weight: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._cell                    = NetworkRNNCell(graph, integrator)
        self._graph                   = graph
        self._return_hidden_states    = return_hidden_states
        self._stability_penalty_weight = stability_penalty_weight
        self._rnn = tf.keras.layers.RNN(
            self._cell,
            return_sequences=return_sequences,
            return_state=False,
        )

    def call(
        self,
        t_sequence:    TensorType,
        initial_state: Optional[StateList] = None,
        training:      bool = False,
    ) -> NetworkOutput:
        if initial_state is None:
            initial_state = self._cell.get_initial_state(batch_size=1)

        raw = self._rnn(t_sequence, initial_state=initial_state,
                        training=training)
        # raw shape: [batch, T, total_n_dynamic_units + 1]

        total_units = self._cell._total_dynamic_units
        raw_rates   = raw[:, :, :total_units]          # [batch, T, total_units]
        raw_err     = raw[:, :, total_units:]           # [batch, T, 1]

        # Разбить по популяциям
        firing_rates: Dict[str, TensorType] = {}
        offset = 0
        for name in self._graph.dynamic_population_names:
            n = self._graph._populations[name].n_units
            firing_rates[name] = raw_rates[:, :, offset:offset + n]
            offset += n

        stability_loss = self._stability_penalty_weight * tf.reduce_mean(raw_err)

        return NetworkOutput(
            firing_rates   = firing_rates,
            hidden_states  = None,   # TODO: return_hidden_states в v0.2
            stability_loss = stability_loss,
        )
```

---

## 3.7 `integrators/`

### 3.7.1 `BaseIntegrator`

```python
class BaseIntegrator:
    """
    Базовый класс интегратора ОДУ.

    Интегратор отдельно от популяционной модели — пользователь
    может менять схему интегрирования не изменяя модель.

    Замечание о синаптических входах:
        syn_input и ext_input считаются константными в течение
        одного шага dt (стандартное приближение для нейронных
        симуляций с малым dt). Они передаются во все стадии
        многошаговых методов (RK4) без пересчёта.
    """

    @abstractmethod
    def step(
        self,
        population:        PopulationModel,
        state:             StateList,
        total_synaptic_input: Dict[str, TensorType],
    ) -> Tuple[StateList, TensorType]:
        """
        Выполняет один шаг интегрирования.

        Args:
            population: экземпляр PopulationModel.
            state: текущее состояние популяции.
            total_synaptic_input: {'I_syn': [1, n], 'g_syn': [1, n]}.

        Returns:
            (new_state, local_error_estimate):
                new_state: новое состояние популяции.
                local_error_estimate: tf.Tensor shape [1] — оценка локальной ошибки.
                    EulerIntegrator: всегда tf.zeros([1]).
                    HeunIntegrator: ||Heun - Euler||.
                    RK4Integrator:  ||RK4 - Heun||.
        """
        raise NotImplementedError
```

### 3.7.2 Реализации интеграторов

**`EulerIntegrator`:**
```
new_state[i] = state[i] + dt * deriv[i]
local_error  = tf.zeros([1])
```

**`HeunIntegrator`:**
```
k1 = derivatives(state)
k2 = derivatives(state + dt * k1)
new_state[i]  = state[i] + dt/2 * (k1[i] + k2[i])
euler_state[i]= state[i] + dt * k1[i]
local_error   = mean(||new_state - euler_state||²)
```

**`RK4Integrator`:**
```
k1 = derivatives(state)
k2 = derivatives(state + dt/2 * k1)
k3 = derivatives(state + dt/2 * k2)
k4 = derivatives(state + dt   * k3)
new_state[i]  = state[i] + dt/6 * (k1+2k2+2k3+k4)[i]
heun_state[i] = state[i] + dt/2 * (k1+k2)[i]
local_error   = mean(||new_state - heun_state||²)
```

Все три реализации должны быть traceable `tf.function`. Python-цикл по стадиям RK допустим (число стадий фиксировано).

---

## 3.8 Встроенные популяционные модели

### 3.8.1 `IzhikevichMeanField`

Реализует систему Монтбрио–Пазо–Рокселя (next-generation mean-field):

\[
\dot{r} = \frac{\Delta_\eta}{\pi} + 2rv - (\alpha + g_{syn,tot})\,r
\]
\[
\dot{v} = v^2 - \alpha v - w + I_{ext} + I_{syn} - (\pi r)^2
\]
\[
\dot{w} = a(bv - w) + w_{jump}\cdot r
\]

**Состояние:** `[r, v, w]`, каждое shape `[1, n_units]`.

**Параметры конструктора** (все через `_make_param`):

| Параметр | Типичные значения | Trainable | Ограничения |
|---|---|---|---|
| `alpha` | [0.5, ...] | нет | — |
| `a` | [0.02, ...] | нет | — |
| `b` | [0.2, ...] | нет | — |
| `w_jump` | [0.1, ...] | нет | — |
| `Delta_I` | [0.5, ...] | **да** | [min, max] |
| `I_ext` | [1.0, ...] | **да** | — |

`trainable` для каждого параметра задаётся пользователем через `params`.

**`get_firing_rate()`:**
```python
return tf.nn.relu(r)  # dimensionless rate
```

**`observables()`:** `{'firing_rate': ..., 'v_mean': v, 'w_mean': w}`.

**`derivatives(state, total_synaptic_input)`:**
```python
r, v, w    = state
g_syn_tot  = total_synaptic_input['g_syn']   # [1, n_units]
I_syn      = total_synaptic_input['I_syn']   # [1, n_units]
PI         = get_pi()

drdt = Delta_I/PI + 2*r*v - (alpha + g_syn_tot)*r
dvdt = v**2 - alpha*v - w + I_ext + I_syn - (PI*r)**2
dwdt = a*(b*v - w) + w_jump*r
return [drdt, dvdt, dwdt]
```

---

### 3.8.2 `WilsonCowan`

\[
\tau_E \dot{E} = -E + F_E(w_{EE}E - w_{IE}I + I_{ext,E} + I_{syn,E})
\]
\[
\tau_I \dot{I} = -I + F_I(w_{EI}E - w_{II}I + I_{ext,I} + I_{syn,I})
\]

\(F(x) = 1/(1 + e^{-a_{coeff}(x - \theta)})\)

**Состояние:** `[E, I]`, каждое shape `[1, n_units]`.

**Параметры:** `tau_E`, `tau_I`, `a_E`, `a_I`, `theta_E`, `theta_I`, `w_EE`, `w_IE`, `w_EI`, `w_II`, `I_ext_E`, `I_ext_I`, `max_rate`.

**`observables()`:** только `{'firing_rate': E * max_rate}`. Нет `v_mean` — `NMDASynapse` получит нулевой `post_voltage`.

**`get_firing_rate()`:** `tf.nn.relu(E) * max_rate`.

---

### 3.8.3 `FokkerPlanckPopulation`

Базовый класс для популяций, где состояние — дискретизованное распределение \(P(V, t)\).

**Состояние:** `[P]`, shape `[1, grid_size]` — вектор вероятностей на сетке.

**Контракт:** пользователь наследуется и реализует `derivatives()` с дискретизованным оператором Фоккера–Планка. Пакет не предоставляет инфраструктуры дискретизации по пространственным переменным.

**`get_firing_rate()`:** поток через правую границу сетки (пороговый потенциал):
```python
# J_out = -D * dP/dV|_{V=V_threshold}  (аппроксимация конечными разностями)
return J_out * tf.ones([1, 1])  # scalar firing rate
```

**Вспомогательный метод `get_boundary_flux(P, dV)`** предоставляется в базовом классе.

---

## 3.9 Встроенные синаптические модели

### 3.9.1 `TsodyksMarkramSynapse`

Кратковременная пластичность (STP). Аналитическое решение между шагами.

**Состояние:** `[R, U, A]`, каждое shape `[n_pre, n_post]`.

**Параметры** (все через `_make_param`, broadcast до `[n_pre, n_post]`):

| Параметр | Trainable | Ограничение | Описание |
|---|---|---|---|
| `gsyn_max` | да | NonNeg | максимальная проводимость |
| `tau_f` | да | [6, 240] мс | время фасилитации |
| `tau_d` | да | [2, 15] мс | время депрессии |
| `tau_r` | да | [91, 1300] мс | время восстановления |
| `Uinc` | да | [0.04, 0.7] | инкремент использования |
| `pconn` | нет | [0, 1] | матрица топологии |
| `e_r` | нет | — | потенциал реверсии |

**`forward()` логика:**
```python
# Нормализованная пресинаптическая активность
firing_probs   = dt * pre_firing_rate / 1000.0    # [1, n_pre] → вероятность в dt
firing_probs_T = tf.transpose(firing_probs)        # [n_pre, 1]
FRpre_normed   = pconn * firing_probs_T            # [n_pre, n_post]  ← маска топологии

# Аналитическое решение
tau1r   = tf.where(tau_d != tau_r, tau_d/(tau_d - tau_r), 1e-13)
exp_d   = tf.exp(-dt / tau_d)
exp_f   = tf.exp(-dt / tau_f)
exp_r   = tf.exp(-dt / tau_r)

a_   = A * exp_d
r_   = 1 + (R - 1 + tau1r*A)*exp_r - tau1r*A
u_   = U * exp_f

released = U * r_ * FRpre_normed

U_new = u_ + Uinc*(1 - u_)*FRpre_normed
A_new = a_ + released
R_new = r_ - released

# Синаптический ток
g_eff   = gsyn_max * A_new                         # [n_pre, n_post]
post_v_T = tf.transpose(post_voltage)              # [n_post, 1]
I_pair   = g_eff * (e_r - post_v_T)               # [n_pre, n_post]
I_syn    = tf.reduce_sum(I_pair,  axis=0, keepdims=True)  # [1, n_post]
g_syn    = tf.reduce_sum(g_eff,   axis=0, keepdims=True)  # [1, n_post]

return ({'I_syn': I_syn, 'g_syn': g_syn}, [R_new, U_new, A_new])
```

**Замечание о кэшировании экспонент:**

`tau_f`, `tau_d`, `tau_r` — обучаемые параметры. Поэтому `exp(-dt/tau)` пересчитывается в каждом вызове `forward()`. Это одна поэлементная операция на матрицу — незначительно по стоимости.

---

### 3.9.2 `NMDASynapse`

Двойная экспонента + магниевый блок.

**Состояние:** `[gnmda, dgnmda]`, shape `[n_pre, n_post]`.

**Параметры:**

| Параметр | Trainable | Описание |
|---|---|---|
| `gsyn_max_nmda` | да | максимальная проводимость |
| `tau1_nmda` | нет | время нарастания (мс) |
| `tau2_nmda` | нет | время спада (мс) |
| `Mgb` | нет | концентрация Mg²⁺ (мМ) |
| `av_nmda` | нет | крутизна вольт-зависимости |
| `pconn_nmda` | нет | матрица топологии |
| `e_r_nmda` | нет | потенциал реверсии |
| `v_ref` | нет | опорный потенциал для Mg-блока |

**`forward()` логика:**
```python
firing_probs_T = tf.transpose(dt * pre_firing_rate / 1000.0)
s_input        = pconn_nmda * firing_probs_T           # [n_pre, n_post]

# Euler для 2-го порядка ОДУ двойной экспоненты
dgnmda_new = dgnmda + dt*(s_input - gnmda
             - (tau1_nmda + tau2_nmda)*dgnmda) \
             / (tau1_nmda * tau2_nmda)
gnmda_new  = gnmda + dt * dgnmda_new

# Магниевый блок (использует post_voltage)
post_v_T = tf.transpose(post_voltage)                  # [n_post, 1]
mg_block = 1.0 / (1.0 + Mgb*tf.exp(-av_nmda*(post_v_T - v_ref)))

g_eff    = gsyn_max_nmda * gnmda_new * mg_block        # [n_pre, n_post]
I_syn    = tf.reduce_sum(g_eff*(e_r_nmda - post_v_T),
                         axis=0, keepdims=True)        # [1, n_post]
g_syn    = tf.reduce_sum(g_eff, axis=0, keepdims=True) # [1, n_post]

return ({'I_syn': I_syn, 'g_syn': g_syn}, [gnmda_new, dgnmda_new])
```

---

### 3.9.3 `StaticSynapse`

Без пластичности. Состояние: `[]` (пустой список).

```python
def forward(self, pre_firing_rate, post_voltage, state, dt):
    firing_probs_T = tf.transpose(dt * pre_firing_rate / 1000.0)
    FRpre_normed   = pconn * firing_probs_T              # [n_pre, n_post]
    post_v_T       = tf.transpose(post_voltage)
    I_pair = gsyn_max * FRpre_normed * (e_r - post_v_T) # [n_pre, n_post]
    I_syn  = tf.reduce_sum(I_pair, axis=0, keepdims=True)
    g_syn  = tf.reduce_sum(gsyn_max * FRpre_normed,
                           axis=0, keepdims=True)
    return ({'I_syn': I_syn, 'g_syn': g_syn}, [])
```

---

### 3.9.4 `CompositeSynapse`

Объединяет несколько синапсов. Токи суммируются.

```python
syn = CompositeSynapse(
    n_pre=4, n_post=3, dt=0.5,
    components=[
        ('ampa', TsodyksMarkramSynapse(...)),
        ('nmda', NMDASynapse(...)),
    ]
)
```

`state_size` = конкатенация `state_size` всех компонент в порядке объявления.

`forward()` разбивает `state` между компонентами, вызывает каждую, суммирует `I_syn` и `g_syn`.

---

## 3.10 Входные генераторы

### Общий интерфейс

Все генераторы теперь векторизованы аналогично популяциям. Интерфейс:

```python
class BaseInputGenerator(tf.keras.layers.Layer):
    def __init__(self, params: Dict[str, Any], dt: float, name: str, **kwargs):
        # n_units определяется автоматически из размерности параметров:
        # - скаляр → n_units=1
        # - список len=n → n_units=n
        # Все параметры регистрируются через _make_param()
```

**Параметры params:**
- Ключ: имя параметра
- Значение: скаляр, вектор (len=1 или n_units), или словарь `{'value': ..., 'trainable': bool, 'min':, 'max':}`

### `VonMisesGenerator`

Генератор тета-ритмического входа на основе распределения фон Мизаса.

```python
rate(t) = (mean_rate / I0(kappa)) * exp(kappa * cos(2π*freq*t/1000 - phase))

# Параметры:
#   mean_rate: средняя частота (Hz). Скаляр или вектор [n_units].
#   R: R-value (0-1), характеризует концентрированность. Скаляр или вектор.
#   freq: частота тета-ритма (Hz). Скаляр или вектор.
#   phase: начальная фаза (рад). Скаляр или вектор.

# r2kappa аппроксимация:
#   R < 0.53:   kappa = 2*R + R^3 + 5/6*R^5
#   0.53 ≤ R < 0.85: kappa = -0.4 + 1.39*R + 0.43/(1-R)
#   R ≥ 0.85:   kappa = 1/(3*R - 4*R^2 + R^3)

# Пример (один вход, n_units=1):
gen = VonMisesGenerator(
    dt=0.5,
    params={
        'mean_rate': 20.0,
        'R': 0.8,
        'freq': 8.0,
        'phase': 0.0,
    }
)

# Пример (три входа, n_units=3):
gen = VonMisesGenerator(
    dt=0.5,
    params={
        'mean_rate': [20.0, 15.0, 10.0],
        'R': [0.9, 0.7, 0.5],
        'freq': [8.0, 10.0, 12.0],
        'phase': [0.0, np.pi/2, np.pi],
    }
)
```

### `SinusoidalGenerator`

```python
rate(t) = max(0, amplitude * sin(2π*freq*t/1000 + phase) + offset)

# Параметры:
#   amplitude: амплитуда (Hz). Скаляр или вектор [n_units].
#   freq: частота (Hz). Скаляр или вектор.
#   phase: начальная фаза (рад). Скаляр или вектор.
#   offset: смещение (Hz). Скаляр или вектор.

# Пример:
gen = SinusoidalGenerator(
    dt=0.5,
    params={
        'amplitude': 10.0,
        'freq': 8.0,
        'phase': 0.0,
        'offset': 5.0,
    }
)
```

### `ConstantRateGenerator`

```python
rate(t) = rate  (независимо от t)

# Параметры:
#   rate: постоянная частота (Hz). Скаляр или вектор [n_units].

# Пример:
gen = ConstantRateGenerator(
    dt=0.5,
    params={
        'rate': 10.0,
    }
)
```

### Подключение к сети

Генераторы подключаются через `InputPopulation`:

```python
graph = NetworkGraph(dt=0.5)

# Один вход (n_pre=1)
gen1 = VonMisesGenerator(dt=0.5, params={'mean_rate': 20.0, 'R': 0.8, 'freq': 8.0, 'phase': 0.0})
graph.add_input_population('theta', gen1)

# Несколько входов (n_pre=3)
gen2 = VonMisesGenerator(dt=0.5, params={
    'mean_rate': [20.0, 15.0, 10.0],
    'R': [0.9, 0.7, 0.5],
    'freq': [8.0, 10.0, 12.0],
    'phase': [0.0, np.pi/2, np.pi],
})
graph.add_input_population('inputs', gen2)

# Синапс: 3 входа → 4 популяции
syn = TsodyksMarkramSynapse(n_pre=3, n_post=4, dt=0.5, params={...})
graph.add_synapse('inputs->exc', syn, src='inputs', tgt='exc')
```

---

## 3.11 `training/readouts.py`

```python
class BaseReadout(tf.keras.layers.Layer):
    """
    Базовый класс readout-слоя.

    Readout принимает наблюдаемую переменную (firing_rate или другую)
    и преобразует её перед сравнением с target в loss.

    Назначение: экспериментальные данные часто обрывочны. Например,
    для популяций гиппокампа известна тета-модуляция (4-12 Гц),
    но не гамма. BandpassReadout позволяет вычислять loss только
    по частотам в заданной полосе, не штрафуя за динамику вне полосы.

    Градиент проходит через readout к предсказанному сигналу.
    """

    @abstractmethod
    def call(self, x: TensorType) -> TensorType:
        """
        Args:
            x: [batch, T, n_units].
        Returns:
            [batch, T', n_features].
        """
        raise NotImplementedError
```

**Встроенные реализации:**

| Класс | Описание |
|---|---|
| `IdentityReadout` | без преобразования |
| `LinearReadout(n_in, n_out, trainable)` | `y = xW + b`, W обучаемая |
| `BandpassReadout(f_low, f_high, dt, n_taps=51)` | FIR-фильтр через `tf.nn.conv1d` |
| `LFPProxyReadout(weights)` | взвешенная сумма токов из `hidden_states` |
| `HemodynamicReadout(dt)` | свёртка с HRF (двойная гамма, параметры фиксированы) |

**`BandpassReadout` детали:**

```python
def build(self, input_shape):
    from scipy.signal import firwin
    nyq = 0.5 * (1000.0 / self.dt)
    # firwin вычисляет коэффициенты FIR-фильтра
    # Сохраняются как нетренируемый tf.constant
    self._kernel = tf.constant(
        coeffs[None, :, None],
        dtype=neuraltide.config.get_dtype()
    )

def call(self, x):
    # x: [batch, T, n_units]
    # tf.nn.conv1d применяет фильтр к каждому каналу
    return tf.nn.conv1d(x, self._kernel, stride=1, padding='SAME')
```

---

## 3.12 `training/losses.py`

```python
class CompositeLoss:
    """
    Составная функция потерь: L = Σ weight_i * L_i.

    Пример:
        loss_fn = CompositeLoss([
            (1.0,   MSELoss(target_rates)),
            (1e-3,  StabilityPenalty()),
            (1e-4,  L2RegularizationLoss(network.trainable_variables)),
        ])
        loss = loss_fn(output, network)
    """

    def __init__(self, terms: List[Tuple[float, 'BaseLoss']]):
        self.terms = terms

    def __call__(self, predictions: NetworkOutput,
                 model: NetworkRNN) -> TensorType:
        total = tf.zeros([], dtype=neuraltide.config.get_dtype())
        for weight, loss_obj in self.terms:
            total += weight * loss_obj(predictions, model)
        return total


class MSELoss(BaseLoss):
    """
    MSE между предсказанной и целевой активностью.

    Поддерживает частичные наблюдения: target задаётся только
    для наблюдаемых популяций. Популяции без target в словаре
    не штрафуются.

    Args:
        target: dict[str, tf.Tensor] — целевые траектории.
            Ключи — имена популяций. Shape: [batch, T, n_units_i].
        readout: BaseReadout — предобработка перед сравнением.
            Применяется и к pred, и к target.
            По умолчанию IdentityReadout.
        observable_key: str — какую наблюдаемую использовать.
            'firing_rate' (по умолчанию) или любой ключ из observables().
        mask: dict[str, tf.Tensor] или None — маска по времени.
            None → все точки учитываются.
            Shape маски: [batch, T, 1] или [batch, T, n_units_i].
    """


class StabilityPenalty(BaseLoss):
    """
    Штраф за численную нестабильность.
    Возвращает predictions.stability_loss.
    Ненулевой только при RK4Integrator или HeunIntegrator.
    Вес определяется в CompositeLoss.
    """
    def __call__(self, predictions, model):
        return predictions.stability_loss


class L2RegularizationLoss(BaseLoss):
    """L2-регуляризация: Σ ||w||²."""


class ParameterBoundLoss(BaseLoss):
    """
    Мягкие границы на параметры (дифференцируемая альтернатива constraints).
    Штраф нарастает при выходе за [min, max].
    """
```

---

## 3.13 `training/trainer.py`

```python
class Trainer:
    """
    Высокоуровневый API обучения.

    Args:
        network (NetworkRNN): сеть.
        loss_fn (CompositeLoss): функция потерь.
        optimizer: tf.keras.optimizers.Optimizer.
        grad_clip_norm (float): максимальная норма градиента.
            0.0 — без клиппинга. По умолчанию 1.0.
        run_eagerly (bool): отключить tf.function для отладки.

    Методы:
        fit(t_sequence, epochs, callbacks, verbose) → TrainingHistory
        train_step(t_sequence) → dict[str, float]
        predict(t_sequence) → NetworkOutput
        save_experiment(path)
        load_experiment(path) [classmethod]
    """

    @tf.function
    def train_step(self, t_sequence: TensorType) -> Dict[str, float]:
        with tf.GradientTape() as tape:
            output = self.network(t_sequence, training=True)
            loss   = self.loss_fn(output, self.network)

        grads = tape.gradient(loss, self.network.trainable_variables)

        if self.grad_clip_norm > 0:
            grads, _ = tf.clip_by_global_norm(grads, self.grad_clip_norm)

        # Логирование None-градиентов
        for g, v in zip(grads, self.network.trainable_variables):
            if g is None:
                tf.print("WARNING: None gradient for", v.name)

        self.optimizer.apply_gradients(
            zip(grads, self.network.trainable_variables)
        )
        return {'loss': loss}
```

**`save_experiment()` структура директории:**
```
{path}/
├── checkpoint/
│   ├── checkpoint
│   └── ckpt-1.*
├── config.json
├── versions.json
├── training_history.json
└── seeds.json
```

---

## 3.14 `training/callbacks.py`

| Класс | Назначение |
|---|---|
| `DivergenceDetector` | останавливает обучение при NaN/Inf в loss |
| `GradientMonitor` | логирует нормы градиентов по переменным |
| `ExperimentLogger(save_dir, save_every)` | сохраняет checkpoint каждые N эпох |

---

## 3.15 `constraints/param_constraints.py`

| Класс | Действие |
|---|---|
| `MinMaxConstraint(min_val, max_val)` | `clip(w, min, max)` |
| `NonNegConstraint` | `relu(w)` |
| `UnitIntervalConstraint` | `clip(w, 0, 1)` |

Все реализуют `get_config()` / `from_config()` для сериализации.

---

## 3.16 `utils/summary.py`

```
print_summary(network) выводит:

┌─────────────────────────────────────────────────────────────────┐
│ NEURALTIDE MODEL SUMMARY                                        │
├──────────────────────────────────────────────────────────────── │
│ INPUT POPULATIONS                                               │
├────────────┬──────────────────────┬─────────┬────────────────  │
│ Name       │ Generator            │ n_units │                  │
├────────────┼──────────────────────┼─────────┼────────────────  │
│ theta      │ VonMisesGenerator    │ 1       │                  │
├──────────────────────────────────────────────────────────────── │
│ DYNAMIC POPULATIONS                                             │
├────────────┬──────────────────────┬─────────┬────────────────  │
│ Name       │ Model                │ n_units │ Parameters       │
├────────────┼──────────────────────┼─────────┼────────────────  │
│ exc        │ IzhikevichMeanField   │ 4       │ Delta_I    T [4] │
│            │                      │         │ I_ext     T [4] │
│            │                      │         │ alpha     F [4] │
│ inh        │ IzhikevichMeanField   │ 2       │ Delta_I    T [2] │
├──────────────────────────────────────────────────────────────── │
│ PROJECTIONS                                                     │
├──────────────┬────────────────┬──────────┬──────────────────   │
│ Name         │ src → tgt      │ Shape    │ Synapse            │
├──────────────┼────────────────┼──────────┼──────────────────   │
│ theta->exc   │ theta → exc    │ [1, 4]   │ TsodyksMarkram T   │
│ exc->exc     │ exc → exc      │ [4, 4]   │ TsodyksMarkram T   │
│ exc->inh     │ exc → inh      │ [4, 2]   │ Composite T        │
│ inh->exc     │ inh → exc      │ [2, 4]   │ Static             │
├──────────────────────────────────────────────────────────────── │
│ Trainable params: 58 │ Non-trainable: 24                        │
└─────────────────────────────────────────────────────────────────┘
T = trainable, F = frozen
```

Если `rich` недоступен — fallback на ASCII.

---

## 3.17 `config/schema.py`

```python
@dataclass
class PopulationConfig:
    name:        str
    model_class: str            # ключ в POPULATION_REGISTRY
    dt:          float
    params:      Dict[str, Any] # каждый элемент — значение или {'value', 'trainable', ...}

@dataclass
class SynapseConfig:
    name:         str
    synapse_class: str          # ключ в SYNAPSE_REGISTRY
    src:          str
    tgt:          str
    dt:           float
    params:       Dict[str, Any]
    components:   Optional[List['SynapseConfig']] = None  # для Composite

@dataclass
class InputConfig:
    name:            str
    generator_class: str        # ключ в INPUT_REGISTRY
    params:          Dict[str, Any]

@dataclass
class NetworkConfig:
    dt:                       float
    integrator:               str    # 'euler', 'heun', 'rk4'
    populations:              List[PopulationConfig]
    synapses:                 List[SynapseConfig]
    inputs:                   List[InputConfig]
    stability_penalty_weight: float = 0.0
    return_hidden_states:     bool  = False
```

**`build_network_from_config(config: NetworkConfig) → NetworkRNN`:**
- Для каждого `InputConfig`: создаёт генератор из `INPUT_REGISTRY`, вызывает `graph.add_input_population()`.
- Для каждого `PopulationConfig`: создаёт модель из `POPULATION_REGISTRY`, вызывает `graph.add_population()`.
- Для каждого `SynapseConfig`: создаёт синапс из `SYNAPSE_REGISTRY`, вызывает `graph.add_synapse()`.

---

# Часть IV. Требования к тестам

## 4.1 `tests/conftest.py`

```python
import pytest
import tensorflow as tf
import neuraltide

@pytest.fixture(autouse=True)
def reset_dtype():
    neuraltide.config.set_dtype(tf.float32)
    yield

@pytest.fixture
def dt():
    return 0.5

@pytest.fixture
def n_steps():
    return 100

@pytest.fixture
def small_izh_params():
    """Минимальные параметры IzhikevichMeanField для n_units=2."""
    return {
        'alpha':     {'value': [0.5, 0.5],    'trainable': False},
        'a':         {'value': [0.02, 0.02],  'trainable': False},
        'b':         {'value': [0.2, 0.2],    'trainable': False},
        'w_jump':    {'value': [0.1, 0.1],    'trainable': False},
        'Delta_I':   {'value': [0.5, 0.6],    'trainable': True,
                      'min': 0.01, 'max': 2.0},
        'I_ext':     {'value': [1.0, 1.2],    'trainable': True},
    }
```

---

## 4.2 Модульные тесты

### `test_config.py`
```
- get_dtype() возвращает float32 по умолчанию
- set_dtype(float64) меняет dtype
- register_population() добавляет класс в POPULATION_REGISTRY
- register_synapse() добавляет класс в SYNAPSE_REGISTRY
```

### `test_constraints.py`
```
- MinMaxConstraint(0, 1): clip([-1, 0.5, 2]) == [0, 0.5, 1]
- MinMaxConstraint(None, 1): нижняя граница не применяется
- MinMaxConstraint(0, None): верхняя граница не применяется
- NonNegConstraint: relu применяется корректно
- UnitIntervalConstraint: эквивалентна MinMaxConstraint(0, 1)
- get_config() / from_config() для всех constraints
```

### `test_populations.py`
```
Тест 1:  IzhikevichMeanField.get_initial_state() — нули, shapes [1, n_units]
Тест 2:  IzhikevichMeanField.derivatives() — корректные shapes
Тест 3:  IzhikevichMeanField.derivatives() совместим с tf.function
Тест 4:  WilsonCowan — аналогично тестам 1-3
Тест 5:  FokkerPlanckPopulation — базовая дымовая проверка наследования
Тест 6:  observables() содержит 'firing_rate' у всех встроенных моделей
Тест 7:  get_firing_rate() возвращает неотрицательные значения
Тест 8:  _make_param() broadcast скаляра до [n_units]
Тест 9:  _make_param() список правильной длины → shape [n_units]
Тест 10: _make_param() список неправильной длины → ValueError
Тест 11: _make_param() trainable=True → переменная в trainable_variables
Тест 12: _make_param() trainable=False → НЕ в trainable_variables
Тест 13: _make_param() с min/max создаёт MinMaxConstraint
Тест 14: производные батчированной популяции (n_units=4, разные I_ext)
         → каждый элемент результата отличается
Тест 15: IzhikevichMeanField.observables() содержит 'v_mean'
Тест 16: WilsonCowan.observables() НЕ содержит 'v_mean'
```

### `test_synapses.py`
```
Тест 1:  TsodyksMarkramSynapse.get_initial_state() — нули, shapes [n_pre, n_post]
Тест 2:  TsodyksMarkramSynapse инвариант R+A <= 1 за 100 шагов
Тест 3:  NMDASynapse — зависимость тока от post_voltage (магниевый блок)
Тест 4:  StaticSynapse — пропорциональность тока частоте
Тест 5:  CompositeSynapse — сумма токов компонент = суммарный ток
Тест 6:  Все синапсы совместимы с tf.function
Тест 7:  forward() возвращает I_syn [1, n_post], g_syn [1, n_post]
Тест 8:  _broadcast_to_matrix() скаляр → [n_pre, n_post]
Тест 9:  _broadcast_to_matrix() вектор [n_pre] → [n_pre, n_post]
Тест 10: _broadcast_to_matrix() вектор [n_post], n_pre≠n_post → [n_pre, n_post]
Тест 11: _broadcast_to_matrix() вектор неправильной длины → ValueError
Тест 12: _broadcast_to_matrix() матрица [n_pre, n_post] → без изменений
Тест 13: _broadcast_to_matrix() матрица неправильной формы → ValueError
Тест 14: pconn маска: pconn[i,j]=0 → ток I_syn[0,j] не зависит от pre_rate[0,i]
Тест 15: рекуррентный синапс n_pre=n_post=4 — forward() без ошибок
Тест 16: gsyn_max скаляр + pconn матрица → корректное обнуление
```

### `test_integrators.py`
```
Тест 1: Euler — dy/dt = -y, y(0)=1, ошибка < 1% за 1000 шагов dt=0.1
Тест 2: Heun  — то же, ошибка < 0.1%
Тест 3: RK4   — то же, ошибка < 1e-4%
Тест 4: RK4 local_error_estimate ненулевой при нелинейной динамике
Тест 5: Euler local_error_estimate == 0.0
Тест 6: Все интеграторы совместимы с tf.function
```

### `test_inputs.py`
```
Тест 1:  InputPopulation.get_initial_state() → [zeros [1,1]]
Тест 2:  InputPopulation.derivatives() → []
Тест 3:  InputPopulation.get_firing_rate([t]) вызывает generator(t)
Тест 4:  InputPopulation.get_firing_rate() shape [1, n_units]
Тест 5:  VonMisesGenerator — периодичность выхода с правильной частотой
Тест 6:  VonMisesGenerator — выход неотрицателен
Тест 7:  SinusoidalGenerator — амплитуда и частота
Тест 8:  ConstantRateGenerator — постоянный выход
Тест 9:  Все генераторы совместимы с tf.function
```

### `test_network.py`
```
Тест 1:  NetworkGraph.validate() → ValueError при n_pre ≠ n_units источника
Тест 2:  NetworkGraph.validate() → ValueError при n_post ≠ n_units цели
Тест 3:  NetworkGraph.add_synapse() → ValueError если tgt — InputPopulation
Тест 4:  NetworkGraph.add_synapse() → ValueError если src не зарегистрирован
Тест 5:  NetworkRNNCell.state_size = сумма sizes всех популяций и синапсов
Тест 6:  NetworkRNN(t_seq) → firing_rates правильной формы [batch, T, n_units_i]
Тест 7:  NetworkRNN с IzhikevichMeanField + WilsonCowan (mixed model)
         → прогон без ошибок, выходы обеих популяций присутствуют
Тест 8:  NetworkRNN совместим с tf.function
Тест 9:  NetworkRNNCell не вызывает integrator.step() для InputPopulation
         (мокировать integrator через unittest.mock)
Тест 10: Одна InputPopulation → несколько популяций через разные синапсы
         → токи аккумулируются независимо в каждой целевой популяции
Тест 11: Несколько InputPopulation → одна популяция
         → токи суммируются корректно
Тест 12: Рекуррентный синапс (src==tgt) — прогон без ошибок
```

### `test_losses.py`
```
Тест 1: MSELoss(target=pred) == 0.0
Тест 2: BandpassReadout подавляет компоненты вне полосы > 40 дБ
Тест 3: StabilityPenalty == 0.0 при EulerIntegrator
Тест 4: CompositeLoss — взвешенная сумма корректна
Тест 5: MSELoss с частичными наблюдениями — потери только по указанным популяциям
Тест 6: MSELoss с mask — замаскированные точки не влияют на loss
```

### `test_readouts.py`
```
Тест 1: IdentityReadout — не изменяет сигнал
Тест 2: LinearReadout — корректная форма выхода
Тест 3: BandpassReadout — синусоида внутри полосы проходит (< 3 дБ ослабление)
Тест 4: BandpassReadout — синусоида вне полосы подавляется (> 40 дБ)
Тест 5: BandpassReadout совместим с tf.function
Тест 6: Градиент проходит через BandpassReadout (нет None градиентов)
```

### `test_training.py`
```
Тест 1: Один train_step() уменьшает loss (smoke test)
Тест 2: fit() без ошибок на 3 эпохи на toy модели
Тест 3: Ненулевые градиенты по всем trainable_variables
Тест 4: MinMaxConstraint применяется после шага оптимизатора
Тест 5: Градиент течёт через InputPopulation → синапс → популяция
         gsyn_max входного синапса имеет ненулевой градиент
```

---

## 4.3 Интеграционные тесты

### `test_gradient_check.py`
```
Тест 1: Numerical gradient check — IzhikevichMeanField + TsodyksMarkramSynapse
        Параметры: Delta_I, I_ext, gsyn_max
        Критерий: max relative error < 1e-3
        (сравнить tape.gradient с конечными разностями)

Тест 2: Gradient check для NMDASynapse (voltage-dependent nonlinearity)

Тест 3: Gradient check для BandpassReadout

Тест 4: Gradient check для CompositeSynapse (AMPA + NMDA)
```

### `test_analytic_solutions.py`
```
Тест 1: WilsonCowan — нулевое начальное состояние при нулевом входе остаётся нулём
        (если нуль является фиксированной точкой)

Тест 2: WilsonCowan — установившееся значение при постоянном входе
        совпадает с аналитически вычисленной фиксированной точкой (< 1% ошибка)

Тест 3: IzhikevichMeanField — при Delta_I=0, I_ext=0, w_jump=0
        частота монотонно убывает к 0

Тест 4: TsodyksMarkramSynapse — установившееся A_inf при постоянной частоте r0:
        A_inf = Uinc*r0*tau_d / (1 + Uinc*r0*(tau_d + tau_f))
        Допуск: < 1%
```

### `test_full_pipeline.py`
```
Тест 1: Полный цикл обучения
        - 1 популяция IzhikevichMeanField, n_units=2
        - target = синусоида 8 Гц
        - 50 эпох, MSELoss + StabilityPenalty
        - Итоговый loss < начальный loss

Тест 2: Смешанная сеть IzhikevichMeanField + NMDA
        - exc (n_units=4) + inh (n_units=2)
        - AMPA + NMDA синапсы
        - 1 секунда симуляции (T=2000, dt=0.5)
        - Все выходы конечны (нет NaN/Inf)
        - Нет None градиентов

Тест 3: Воспроизводимость
        - seed_everything(42), обучение 5 эпох
        - seed_everything(42), обучение 5 эпох снова
        - Все параметры совпадают до 1e-6

Тест 4: Сохранение и загрузка
        - save_experiment_state(path)
        - Trainer.load_experiment(path)
        - Предсказания до и после сохранения совпадают до 1e-6

Тест 5: BandpassReadout в loss — частотная селективность
        - target = тета 8 Гц
        - pred = target + синусоида 60 Гц
        - BandpassReadout(4, 12, dt=0.5) → loss ≈ 0
        - Без фильтра → loss >> 0

Тест 6: InputPopulation через обучаемый синапс
        - VonMisesGenerator → TsodyksMarkramSynapse → IzhikevichMeanField
        - Обучение 10 эпох
        - gsyn_max изменился от начального значения
```

---

# Часть V. Примеры

## `example_01_single_population.py`

```python
"""
Пример 1: Одна популяция IzhikevichMeanField (n_units=2),
тета-ритмический вход через обучаемый синапс.
Цель: воспроизвести заданную firing rate траекторию.
"""
import numpy as np
import tensorflow as tf
import neuraltide
from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.populations import IzhikevichMeanField
from neuraltide.synapses import TsodyksMarkramSynapse
from neuraltide.inputs import VonMisesGenerator
from neuraltide.integrators import RK4Integrator
from neuraltide.training import Trainer, CompositeLoss, MSELoss, StabilityPenalty
from neuraltide.utils import seed_everything, print_summary

seed_everything(42)

dt = 0.5   # мс
T  = 2000  # шагов = 1 секунда

# --- Популяция ---
pop = IzhikevichMeanField(n_units=2, dt=dt, params={
    'alpha':     {'value': [0.5, 0.5],   'trainable': False},
    'a':         {'value': [0.02, 0.02], 'trainable': False},
    'b':         {'value': [0.2, 0.2],   'trainable': False},
    'w_jump':    {'value': [0.1, 0.1],   'trainable': False},
    'Delta_I':   {'value': [0.5, 0.6],   'trainable': True,
                  'min': 0.01, 'max': 2.0},
    'I_ext':     {'value': [1.0, 1.2],   'trainable': True},
})

# --- Входной генератор ---
gen = VonMisesGenerator(params=[
    {'MeanFiringRate': 20.0, 'R': 0.5, 'ThetaFreq': 8.0, 'ThetaPhase': 0.0},
], name='theta_gen')

# --- Синапс вход → популяция ---
syn_in = TsodyksMarkramSynapse(n_pre=1, n_post=2, dt=dt, params={
    'gsyn_max': {'value': [[0.1, 0.1]], 'trainable': True},
    'tau_f':    {'value': 20.0,  'trainable': True, 'min': 6.0,  'max': 240.0},
    'tau_d':    {'value': 5.0,   'trainable': True, 'min': 2.0,  'max': 15.0},
    'tau_r':    {'value': 200.0, 'trainable': True, 'min': 91.0, 'max': 1300.0},
    'Uinc':     {'value': 0.2,   'trainable': True, 'min': 0.04, 'max': 0.7},
    'pconn':    {'value': [[1.0, 1.0]], 'trainable': False},
    'e_r':      {'value': 0.0,   'trainable': False},
})

# --- Рекуррентный синапс ---
syn_rec = TsodyksMarkramSynapse(n_pre=2, n_post=2, dt=dt, params={
    'gsyn_max': {'value': 0.05,  'trainable': True},
    'tau_f':    {'value': 20.0,  'trainable': True, 'min': 6.0,  'max': 240.0},
    'tau_d':    {'value': 5.0,   'trainable': True, 'min': 2.0,  'max': 15.0},
    'tau_r':    {'value': 200.0, 'trainable': True, 'min': 91.0, 'max': 1300.0},
    'Uinc':     {'value': 0.2,   'trainable': True, 'min': 0.04, 'max': 0.7},
    'pconn':    {'value': [[1, 1], [1, 1]], 'trainable': False},
    'e_r':      {'value': 0.0,   'trainable': False},
})

# --- Топология ---
graph = NetworkGraph(dt=dt)
graph.add_input_population('theta', gen)
graph.add_population('exc', pop)
graph.add_synapse('theta->exc', syn_in,  src='theta', tgt='exc')
graph.add_synapse('exc->exc',   syn_rec, src='exc',   tgt='exc')

network = NetworkRNN(graph, integrator=RK4Integrator(),
                     stability_penalty_weight=1e-3)
print_summary(network)

# --- Временна́я ось ---
t_values = np.arange(T, dtype=np.float32) * dt
t_seq    = tf.constant(t_values[None, :, None])   # [1, T, 1]

# --- Целевая траектория ---
target_0 = 10.0 + 5.0*np.sin(2*np.pi*8.0*t_values/1000.0)
target_1 = 8.0  + 4.0*np.sin(2*np.pi*8.0*t_values/1000.0 + 0.5)
target = {
    'exc': tf.constant(
        np.stack([target_0, target_1], axis=-1)[None, :, :],
        dtype=tf.float32
    )   # [1, T, 2]
}

# --- Обучение ---
loss_fn = CompositeLoss([
    (1.0,  MSELoss(target)),
    (1e-3, StabilityPenalty()),
])
trainer = Trainer(network, loss_fn,
                  optimizer=tf.keras.optimizers.Adam(1e-3))
history = trainer.fit(t_seq, epochs=200, verbose=1)
```

---

## `example_02_exc_inh_nmda.py`

```python
"""
Пример 2: exc (IzhikevichMF, n_units=4) + inh (IzhikevichMF, n_units=2).
Синапсы: AMPA+NMDA (exc→exc), AMPA (exc→inh), GABA_A (inh→exc).
Вход: тета-генератор через TsodyksMarkramSynapse.
Частичные наблюдения: target только для exc.
BandpassReadout: loss считается только в тета-диапазоне.
"""
from neuraltide.synapses import CompositeSynapse, NMDASynapse
from neuraltide.training.readouts import BandpassReadout

# exc→exc: AMPA + NMDA
ampa_ee = TsodyksMarkramSynapse(n_pre=4, n_post=4, dt=dt, params={...})
nmda_ee = NMDASynapse(n_pre=4, n_post=4, dt=dt, params={
    'gsyn_max_nmda': {'value': 0.05,  'trainable': True},
    'tau1_nmda':     {'value': 2.0,   'trainable': False},
    'tau2_nmda':     {'value': 100.0, 'trainable': False},
    'Mgb':           {'value': 1.0,   'trainable': False},
    'av_nmda':       {'value': 0.062, 'trainable': False},
    'pconn_nmda':    {'value': 1.0,   'trainable': False},
    'e_r_nmda':      {'value': 0.0,   'trainable': False},
    'v_ref':         {'value': 1.0,   'trainable': False},
})

graph.add_synapse('exc->exc',
    CompositeSynapse(n_pre=4, n_post=4, dt=dt,
                     components=[('ampa', ampa_ee), ('nmda', nmda_ee)]),
    src='exc', tgt='exc'
)

# Частичные наблюдения + тета-фильтр
theta_readout = BandpassReadout(f_low=4.0, f_high=12.0, dt=dt)
loss_fn = CompositeLoss([
    (1.0,  MSELoss({'exc': target_exc}, readout=theta_readout)),
    (1e-3, StabilityPenalty()),
])
```

---

## `example_03_custom_population.py`

```python
"""
Пример 3: Пользовательская популяционная модель.
Минимальный контракт: derivatives, get_initial_state, get_firing_rate, parameter_spec.
"""
import neuraltide
from neuraltide.core.base import PopulationModel
from neuraltide.config import register_population

class MyRateModel(PopulationModel):
    """tau * dr/dt = -r + tanh(I_ext + I_syn)"""

    def __init__(self, n_units, dt, params, **kwargs):
        super().__init__(n_units=n_units, dt=dt, **kwargs)
        self.tau   = self._make_param(params, 'tau')
        self.I_ext = self._make_param(params, 'I_ext')
        self.state_size = [tf.TensorShape([1, n_units])]

    def get_initial_state(self, batch_size=1):
        return [tf.zeros([1, self.n_units],
                         dtype=neuraltide.config.get_dtype())]

    def derivatives(self, state, total_synaptic_input):
        r     = state[0]
        I_tot = self.I_ext + total_synaptic_input['I_syn']
        drdt  = (-r + tf.nn.tanh(I_tot)) / self.tau
        return [drdt]

    def get_firing_rate(self, state):
        return tf.nn.relu(state[0]) * 100.0   # Гц

    @property
    def parameter_spec(self):
        return {
            'tau':   {'shape': (self.n_units,), 'trainable': False,
                      'constraint': None, 'units': 'ms'},
            'I_ext': {'shape': (self.n_units,), 'trainable': True,
                      'constraint': None, 'units': 'dimensionless'},
        }

register_population('MyRateModel', MyRateModel)
```

---

# Часть VI. Пошаговый план реализации

## Шаг 0: Инфраструктура

**Файлы:**
```
pyproject.toml
README.md
.gitignore
neuraltide/__init__.py
neuraltide/config/__init__.py
tests/__init__.py
tests/conftest.py
```

**`pyproject.toml`:**
```toml
[project]
name = "neuraltide"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "tensorflow>=2.16",
    "numpy>=1.26",
    "scipy>=1.12",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov", "black", "ruff"]
vis = ["rich>=13.0"]
```

**`neuraltide/__init__.py`:**
```python
from neuraltide import config
from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.utils.reproducibility import seed_everything
from neuraltide.utils.summary import print_summary
__version__ = "0.1.0"
```

**Тесты:** `test_config.py` — dtype get/set, register_population.

---

## Шаг 1: `core/types.py` и `constraints/`

**Файлы:**
```
neuraltide/core/types.py
neuraltide/constraints/__init__.py
neuraltide/constraints/param_constraints.py
tests/unit/test_constraints.py
```

**Тесты:** все из `test_constraints.py`.

---

## Шаг 2: `core/base.py`

**Файлы:**
```
neuraltide/core/__init__.py
neuraltide/core/base.py
tests/unit/test_base_contracts.py
```

Реализовать: `PopulationModel` (с `_make_param`), `SynapseModel` (с `_make_param`, `_broadcast_to_matrix`), `BaseInputGenerator`.

**Тесты:**
```
- Нельзя инстанциировать абстрактные классы напрямую
- Минимальный конкретный подкласс проходит isinstance проверку
- _make_param broadcast скаляра (базовый smoke test)
- _broadcast_to_matrix все варианты форм
```

---

## Шаг 3: Интеграторы

**Файлы:**
```
neuraltide/integrators/__init__.py
neuraltide/integrators/base.py
neuraltide/integrators/euler.py
neuraltide/integrators/heun.py
neuraltide/integrators/rk4.py
tests/unit/test_integrators.py
```

**Тесты:** все из `test_integrators.py`.

---

## Шаг 4: `IzhikevichMeanField`

**Файлы:**
```
neuraltide/populations/__init__.py
neuraltide/populations/izhikevich_mf.py
tests/unit/test_populations.py  (тесты 1-3, 6-16)
```

---

## Шаг 5: `WilsonCowan`

**Файлы:**
```
neuraltide/populations/wilson_cowan.py
tests/unit/test_populations.py  (тесты 4, 16)
tests/integration/test_analytic_solutions.py  (тесты 1-2)
```

---

## Шаг 6: `FokkerPlanckPopulation`

**Файлы:**
```
neuraltide/populations/fokker_planck.py
tests/unit/test_populations.py  (тест 5)
```

---

## Шаг 7: `InputPopulation` и входные генераторы

**Файлы:**
```
neuraltide/populations/input_population.py
neuraltide/inputs/__init__.py
neuraltide/inputs/base.py
neuraltide/inputs/von_mises.py
neuraltide/inputs/sinusoidal.py
neuraltide/inputs/constant.py
tests/unit/test_inputs.py
```

**Тесты:** все из `test_inputs.py`.

---

## Шаг 8: `TsodyksMarkramSynapse`

**Файлы:**
```
neuraltide/synapses/__init__.py
neuraltide/synapses/tsodyks_markram.py
tests/unit/test_synapses.py  (тесты 1-2, 6-16)
tests/integration/test_analytic_solutions.py  (тест 4)
```

---

## Шаг 9: `NMDASynapse` и `StaticSynapse`

**Файлы:**
```
neuraltide/synapses/nmda.py
neuraltide/synapses/static.py
tests/unit/test_synapses.py  (тесты 3-4)
```

---

## Шаг 10: `CompositeSynapse`

**Файлы:**
```
neuraltide/synapses/composite.py
tests/unit/test_synapses.py  (тест 5)
```

---

## Шаг 11: `core/state.py` и `NetworkGraph`

**Файлы:**
```
neuraltide/core/state.py
neuraltide/core/network.py  (только NetworkGraph)
tests/unit/test_network.py  (тесты 1-4)
```

---

## Шаг 12: `NetworkRNNCell`

**Файлы:**
```
neuraltide/core/network.py  (добавить NetworkRNNCell)
tests/unit/test_network.py  (тесты 5, 9-12)
```

---

## Шаг 13: `NetworkRNN` и `NetworkOutput`

**Файлы:**
```
neuraltide/core/network.py  (добавить NetworkRNN, NetworkOutput)
tests/unit/test_network.py  (тесты 6-8)
```

---

## Шаг 14: Readout-слои

**Файлы:**
```
neuraltide/training/__init__.py
neuraltide/training/readouts.py
tests/unit/test_readouts.py
```

**Тесты:** все из `test_readouts.py`.

---

## Шаг 15: Функции потерь

**Файлы:**
```
neuraltide/training/losses.py
tests/unit/test_losses.py
```

**Тесты:** все из `test_losses.py`.

---

## Шаг 16: `Trainer` и callbacks

**Файлы:**
```
neuraltide/training/trainer.py
neuraltide/training/callbacks.py
tests/unit/test_training.py
tests/integration/test_full_pipeline.py  (тесты 1, 3, 4, 6)
```

---

## Шаг 17: `utils/`

**Файлы:**
```
neuraltide/utils/__init__.py
neuraltide/utils/reproducibility.py
neuraltide/utils/summary.py
neuraltide/utils/sparse.py  (заглушка для v0.2)
tests/integration/test_full_pipeline.py  (тесты 3, 4)
```

---

## Шаг 18: `config/schema.py`

**Файлы:**
```
neuraltide/config/schema.py
tests/unit/test_config.py  (дополнительные тесты)
```

---

## Шаг 19: Интеграционные тесты

**Файлы:**
```
tests/integration/test_gradient_check.py
tests/integration/test_analytic_solutions.py  (оставшиеся)
tests/integration/test_full_pipeline.py  (оставшиеся)
```

---

## Шаг 20: Примеры

**Файлы:**
```
examples/example_01_single_population.py
examples/example_02_exc_inh_nmda.py
examples/example_03_custom_population.py
examples/example_04_fokker_planck.py
