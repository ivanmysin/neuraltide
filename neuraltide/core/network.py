from collections import namedtuple
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import tensorflow as tf

import neuraltide
import neuraltide.config
from neuraltide.core.base import PopulationModel, SynapseModel
from neuraltide.core.types import TensorType, StateList
from neuraltide.integrators.base import BaseIntegrator


def _with_free_batch(shape: tf.TensorShape) -> tf.TensorShape:
    """Форма состояния популяции с незакреплённой батч-осью: [None, *rest].

    state_size популяций — per-sample формы [1, n] (первая ось — батч);
    заменяем её на None. Батчированные состояния синапсов строятся иначе —
    первая ось дописывается вперёд (см. _build).
    """
    dims = shape.as_list()
    if not dims:
        return tf.TensorShape(shape)
    dims[0] = None
    return tf.TensorShape(dims)


@dataclass
class _SynapseEntry:
    model: SynapseModel
    src: str
    tgt: str


class NetworkGraph:
    """
    Описание топологии сети: популяции, внешние входы и синаптические проекции.

    Популяции:
        Динамические: IzhikevichMeanField, WilsonCowan и т.д.

    Внешние входы:
        Объявляются через declare_input(name, n_units).
        Не являются объектами PopulationModel, не имеют состояния.
        Входные частоты разрядов передаются извне через inputs= в call().

    Синаптические проекции:
        Любой тип SynapseModel от любой популяции или объявленного входа
        к любой динамической популяции.
    """

    def __init__(self, dt: float):
        self.dt = dt
        self._populations: Dict[str, PopulationModel] = {}
        self._synapses: Dict[str, _SynapseEntry] = {}
        # Внешние входы
        self._external_inputs: Dict[str, int] = {}  # name -> n_units
        self._input_names: List[str] = []  # ordered by declaration
        self._input_offsets: Dict[str, int] = {}  # name -> offset in stacked tensor
        self._total_input_units: int = 0
        # Кэшированные списки для ускорения итераций
        self._cached_population_names: Optional[List[str]] = None
        self._cached_synapse_names: Optional[List[str]] = None
        self._cached_dynamic_population_names: Optional[List[str]] = None
        self._pop_info_cache: List[Tuple[str, PopulationModel, int]] = []
        self._syn_info_cache: List[Tuple[str, _SynapseEntry, int]] = []
        self._dynamic_pop_indices: List[int] = []

    def add_population(self, name: str, model: PopulationModel) -> None:
        """Регистрирует популяцию (динамическую или InputPopulation)."""
        if name in self._populations:
            raise ValueError(f"Population '{name}' already registered.")
        self._populations[name] = model
        # Сброс кэшей при изменении структуры
        self._cached_population_names = None
        self._cached_dynamic_population_names = None
        self._pop_info_cache = []
        self._dynamic_pop_indices = []

    def declare_input(self, name: str, n_units: int) -> None:
        """Объявляет внешний входной источник (не.population).

        Args:
            name: уникальное имя входа (используется как src в add_synapse)
            n_units: число входных каналов (должно совпадать с generator.n_units)
        """
        if name in self._external_inputs:
            raise ValueError(f"Input '{name}' already declared.")
        if name in self._populations:
            raise ValueError(f"Input name '{name}' conflicts with population name.")
        self._external_inputs[name] = n_units
        self._input_names.append(name)
        self._input_offsets[name] = self._total_input_units
        self._total_input_units += n_units

    def add_synapse(
        self,
        name: str,
        model: SynapseModel,
        src: str,
        tgt: str,
    ) -> None:
        """Регистрирует синаптическую проекцию.

        src может быть именем популяции или объявленного внешнего входа.
        """
        if name in self._synapses:
            raise ValueError(f"Synapse '{name}' already registered.")
        if src not in self._populations and src not in self._external_inputs:
            raise ValueError(f"Synapse '{name}': src '{src}' not registered.")
        if tgt not in self._populations:
            raise ValueError(f"Synapse '{name}': tgt '{tgt}' not registered.")
        # Validate n_pre matches source
        if src in self._external_inputs:
            expected_n = self._external_inputs[src]
        else:
            expected_n = self._populations[src].n_units
        if model.n_pre != expected_n:
            raise ValueError(
                f"Synapse '{name}': n_pre={model.n_pre} "
                f"!= source '{src}' n_units={expected_n}."
            )
        model._container_name = name
        self._synapses[name] = _SynapseEntry(model=model, src=src, tgt=tgt)
        # Сброс кэшей при изменении структуры
        self._cached_synapse_names = None
        self._syn_info_cache = []

    def _build_caches(self) -> None:
        """Построение кэшированных структур для ускорения итераций."""
        if self._cached_population_names is not None:
            return  # Уже построено

        self._cached_population_names = list(self._populations.keys())
        self._cached_synapse_names = list(self._synapses.keys())

        # Кэш для популяций: (name, model, state_size)
        self._pop_info_cache = []
        self._dynamic_pop_indices = []
        for i, name in enumerate(self._cached_population_names):
            pop = self._populations[name]
            n = len(pop.state_size)
            self._pop_info_cache.append((name, pop, n))
            self._dynamic_pop_indices.append(i)

        self._cached_dynamic_population_names = [
            name for name, _, _ in self._pop_info_cache
        ]

        # Кэш для синапсов: (name, entry, state_size)
        self._syn_info_cache = []
        for name in self._cached_synapse_names:
            entry = self._synapses[name]
            n = len(entry.model.state_size)
            self._syn_info_cache.append((name, entry, n))

    def validate(self) -> None:
        """Проверяет корректность топологии перед построением NetworkRNN."""
        for syn_name, entry in self._synapses.items():
            # Validate n_pre against source (population or external input)
            if entry.src in self._external_inputs:
                src_n = self._external_inputs[entry.src]
            else:
                src_pop = self._populations[entry.src]
                src_n = src_pop.n_units
            if entry.model.n_pre != src_n:
                raise ValueError(
                    f"Synapse '{syn_name}': n_pre={entry.model.n_pre} "
                    f"!= src '{entry.src}' n_units={src_n}."
                )
            tgt_pop = self._populations[entry.tgt]
            if entry.model.n_post != tgt_pop.n_units:
                raise ValueError(
                    f"Synapse '{syn_name}': n_post={entry.model.n_post} "
                    f"!= tgt '{entry.tgt}' n_units={tgt_pop.n_units}."
                )
        for pop_name, pop in self._populations.items():
            has_input = any(e.tgt == pop_name for e in self._synapses.values())
            if not has_input:
                import warnings

                warnings.warn(
                    f"Population '{pop_name}' has no incoming synapses.", UserWarning
                )

        # Построить кэши после валидации
        self._build_caches()

    @property
    def population_names(self) -> List[str]:
        if self._cached_population_names is None:
            self._build_caches()
        return self._cached_population_names  # type: ignore

    @property
    def synapse_names(self) -> List[str]:
        if self._cached_synapse_names is None:
            self._build_caches()
        return self._cached_synapse_names  # type: ignore

    @property
    def dynamic_population_names(self) -> List[str]:
        """Имена динамических популяций."""
        if self._cached_dynamic_population_names is None:
            self._build_caches()
        return self._cached_dynamic_population_names  # type: ignore

    @property
    def input_names(self) -> List[str]:
        """Имена объявленных внешних входов (в порядке объявления)."""
        return list(self._input_names)

    @property
    def input_offsets(self) -> Dict[str, int]:
        """Offsets каждого входа в стекированном тензоре."""
        return dict(self._input_offsets)

    @property
    def total_input_units(self) -> int:
        """Общее число входных каналов."""
        return self._total_input_units

    def pack_inputs(self, input_dict: Dict[str, TensorType]) -> TensorType:
        """Складывает именованные входные тензоры в один.

        Args:
            input_dict: {name: tensor[batch, T, n_units_i]}

        Returns:
            Tensor[batch, T, total_input_units] — сконкатенированный
            в порядке declare_input().
        """
        parts = []
        for name in self._input_names:
            if name not in input_dict:
                raise ValueError(
                    f"Input '{name}' not provided. Expected: {self._input_names}"
                )
            t = input_dict[name]
            expected = self._external_inputs[name]
            if t.shape[-1] != expected:
                raise ValueError(
                    f"Input '{name}': last dim is {t.shape[-1]}, expected {expected}"
                )
            parts.append(t)
        return tf.concat(parts, axis=-1)


def unpack_state(
    graph: NetworkGraph,
    flat_pop_states: StateList,
    flat_syn_states: StateList,
) -> Tuple[Dict[str, StateList], Dict[str, StateList]]:
    """
    Unpack flat state lists into dictionaries keyed by population/synapse name.

    Args:
        graph: NetworkGraph
        flat_pop_states: Flat list of population states
        flat_syn_states: Flat list of synapse states

    Returns:
        (pop_states_dict, syn_states_dict)
    """
    pop_states_dict = {}
    idx = 0
    # Use cached info for efficient iteration
    for name, pop, n in graph._pop_info_cache:
        pop_states_dict[name] = flat_pop_states[idx : idx + n]
        idx += n

    syn_states_dict = {}
    idx = 0
    # Use cached info for efficient iteration
    for name, entry, n in graph._syn_info_cache:
        syn_states_dict[name] = flat_syn_states[idx : idx + n]
        idx += n

    return pop_states_dict, syn_states_dict


def get_firing_rates(
    graph: NetworkGraph,
    pop_states_dict: Dict[str, StateList],
) -> Dict[str, TensorType]:
    """
    Get firing rates for all dynamic populations.

    Args:
        graph: NetworkGraph
        pop_states_dict: Dict of population states

    Returns:
        Dict of firing rates keyed by population name
    """
    all_rates = {}
    for name, pop, _ in graph._pop_info_cache:
        all_rates[name] = pop.get_firing_rate(pop_states_dict[name])
    return all_rates


def _infer_batch_size(states: StateList) -> tf.Tensor:
    """Батч-ось из первого доступного тензора состояния (динамический int)."""
    for s in states:
        if isinstance(s, tf.Tensor):
            return tf.shape(s)[0]
    return tf.constant(1, dtype=tf.int32)


class _JointNetworkRHS:
    """
    Согласованная правая часть (joint RHS) связанной сети.

    Используется в режиме integration_mode='network': интегратор с
    supports_network_step=True вызывает step_network(rhs, pop_states,
    syn_states, dt), а этот объект вычисляет производные всех популяций
    и синапсов по их состояниям на начало шага.

    Семантика (__call__):
        1. Распаковка плоских списков состояний по кэшам graph
           (_pop_info_cache / _syn_info_cache).
        2. Для каждой популяции кэшируются firing rate и observables;
           'v_mean' отсутствует → нулевой fallback формы [batch, n_units].
        3. Для каждого синапса: pre_rate из input_rates, если src —
           внешний вход (значение на текущем шаге, zero-order hold),
           иначе из кэшированных firing rates; post_v — 'v_mean' цели
           (или нулевой fallback). dsyn = model.derivatives(syn_state,
           pre_rate, post_v); current = model.compute_current(syn_state,
           pre_rate, post_v).
        4. Токи всех синапсов аккумулируются в syn_I/syn_g целевых
           популяций.
        5. dpop = pop.derivatives(pop_state, {'I_syn', 'g_syn'}).

    Возвращает плоские списки производных в кэшированном порядке
    (популяции, затем синапсы). Блоки моделей без динамики
    (InputPopulation, StaticSynapse) дают пустые списки производных —
    их состояние интегратор оставляет неизменным. Батч-ось выводится
    из формы состояний (произвольное B, состояния [B, ...]).

    Args:
        graph: NetworkGraph с построенными кэшами (validate()).
        input_rates: {name: tensor[batch, n_units]} — частоты разрядов
            внешних входов на текущем шаге. None эквивалентен {}.
    """

    def __init__(
        self,
        graph: NetworkGraph,
        input_rates: Optional[Dict[str, TensorType]],
    ):
        self._graph = graph
        self._input_rates = input_rates if input_rates is not None else {}

    def __call__(
        self,
        pop_states: StateList,
        syn_states: StateList,
    ) -> Tuple[StateList, StateList]:
        graph = self._graph
        dtype = neuraltide.config.get_dtype()
        input_rates = self._input_rates

        # Распаковка состояний по кэшированным оффсетам
        pop_states_dict = {}
        idx = 0
        for name, pop, n in graph._pop_info_cache:
            pop_states_dict[name] = pop_states[idx : idx + n]
            idx += n

        syn_states_dict = {}
        idx = 0
        for name, entry, n in graph._syn_info_cache:
            syn_states_dict[name] = syn_states[idx : idx + n]
            idx += n

        batch = _infer_batch_size(list(pop_states) + list(syn_states))

        # Инициализация синаптических токов для динамических популяций
        syn_I: Dict[str, TensorType] = {}
        syn_g: Dict[str, TensorType] = {}
        for name in graph.dynamic_population_names:
            n = graph._populations[name].n_units
            syn_I[name] = tf.zeros([batch, n], dtype=dtype)
            syn_g[name] = tf.zeros([batch, n], dtype=dtype)

        # Кэширование firing_rates и observables для каждой популяции
        cached_rates: Dict[str, TensorType] = {}
        cached_obs: Dict[str, Dict[str, TensorType]] = {}
        for pop_name in graph._populations:
            pop = graph._populations[pop_name]
            pop_state = pop_states_dict[pop_name]
            cached_rates[pop_name] = pop.get_firing_rate(pop_state)
            cached_obs[pop_name] = pop.observables(pop_state)

        # Производные синапсов и аккумуляция токов
        dsyn_flat: StateList = []
        for syn_name, entry, _ in graph._syn_info_cache:
            syn_state = syn_states_dict[syn_name]
            tgt_pop = graph._populations[entry.tgt]

            if entry.src in graph._external_inputs:
                pre_rate = input_rates[entry.src]
            else:
                pre_rate = cached_rates[entry.src]

            post_v = cached_obs[entry.tgt].get("v_mean")
            if post_v is None:
                post_v = tf.zeros([batch, tgt_pop.n_units], dtype=dtype)

            dsyn = entry.model.derivatives(syn_state, pre_rate, post_v)
            dsyn_flat.extend(dsyn)

            current_dict = entry.model.compute_current(syn_state, pre_rate, post_v)
            syn_I[entry.tgt] = syn_I[entry.tgt] + current_dict["I_syn"]
            syn_g[entry.tgt] = syn_g[entry.tgt] + current_dict["g_syn"]

        # Производные популяций по суммарному синаптическому входу
        dpop_flat: StateList = []
        for name, pop, _ in graph._pop_info_cache:
            pop_state = pop_states_dict[name]
            total_syn = {"I_syn": syn_I[name], "g_syn": syn_g[name]}
            dpop = pop.derivatives(pop_state, total_syn)
            dpop_flat.extend(dpop)

        return dpop_flat, dsyn_flat


def _step_fn(
    states: Tuple[StateList, StateList, TensorType],
    t: TensorType,
    input_rates: Optional[Dict[str, TensorType]],
    graph: NetworkGraph,
    integrator: BaseIntegrator,
    *,
    integration_mode: str = "split",
) -> Tuple[StateList, StateList, TensorType]:
    """
    Один шаг симуляции сети. Используется с tf.scan.

    Args:
        states: (pop_states, syn_states, stability_error) - текущие состояния
        t: текущее время
        input_rates: {name: tensor[batch, n_units]} — предвычисленные
            частоты разрядов для внешних входов на данном шаге.
        graph: NetworkGraph
        integrator: BaseIntegrator
        integration_mode: 'split' — отдельные шаги синапсов и популяций
            (как раньше); 'network' — единый шаг интегратора по всей
            связанной сети через _JointNetworkRHS + step_network
            (требует supports_network_step=True у интегратора).

    Returns:
        (new_pop_states, new_syn_states, new_stability_error) —
        stability-аккумулятор скалярный (per-step local_err интеграторов
        [1] усредняется до скаляра при накоплении).
    """
    pop_states, syn_states, stability_acc = states
    dtype = neuraltide.config.get_dtype()

    if integration_mode == "network":
        rhs = _JointNetworkRHS(graph, input_rates)
        new_pop_list, new_syn_list = integrator.step_network(
            rhs, pop_states, syn_states, graph.dt
        )
        new_pop_list = list(new_pop_list)
        new_syn_list = list(new_syn_list)
        if neuraltide.config.get_debug_numerics():
            for i, s in enumerate(new_pop_list):
                new_pop_list[i] = tf.debugging.check_numerics(
                    s, f"NaN in population state[{i}]"
                )
            for i, s in enumerate(new_syn_list):
                new_syn_list[i] = tf.debugging.check_numerics(
                    s, f"NaN in synapse state[{i}]"
                )
        return (tuple(new_pop_list), tuple(new_syn_list), stability_acc)

    # Распаковка состояний популяций с использованием кэшированных оффсетов
    pop_states_dict = {}
    idx = 0
    for name, pop, n in graph._pop_info_cache:
        pop_states_dict[name] = pop_states[idx : idx + n]
        idx += n

    # Распаковка состояний синапсов с использованием кэшированных оффсетов
    syn_states_dict = {}
    idx = 0
    for name, entry, n in graph._syn_info_cache:
        syn_states_dict[name] = syn_states[idx : idx + n]
        idx += n

    # Батч-ось из формы состояний (произвольное B; состояния [B, ...])
    batch = _infer_batch_size(list(pop_states) + list(syn_states))

    # Инициализация синаптических токов для динамических популяций
    syn_I: Dict[str, TensorType] = {}
    syn_g: Dict[str, TensorType] = {}
    for name in graph.dynamic_population_names:
        n = graph._populations[name].n_units
        syn_I[name] = tf.zeros([batch, n], dtype=dtype)
        syn_g[name] = tf.zeros([batch, n], dtype=dtype)

    # Кэширование firing_rates и observables для каждой популяции (T1)
    # Вычисляем один раз, а не на каждый синапс
    cached_rates: Dict[str, TensorType] = {}
    cached_obs: Dict[str, Dict[str, TensorType]] = {}
    for pop_name in graph._populations:
        pop = graph._populations[pop_name]
        pop_state = pop_states_dict[pop_name]
        cached_rates[pop_name] = pop.get_firing_rate(pop_state)
        cached_obs[pop_name] = pop.observables(pop_state)

    # Обработка всех синапсов
    if input_rates is None:
        input_rates = {}
    for syn_name, entry, _ in graph._syn_info_cache:
        syn_state = syn_states_dict[syn_name]

        tgt_pop = graph._populations[entry.tgt]

        # Получение пресинаптической частоты разрядов из кэша
        if entry.src in graph._external_inputs:
            pre_rate = input_rates[entry.src]
        else:
            pre_rate = cached_rates[entry.src]

        post_v = cached_obs[entry.tgt].get("v_mean")
        if post_v is None:
            post_v = tf.zeros([batch, tgt_pop.n_units], dtype=dtype)

        new_syn_state, local_err = integrator.step_synapse(
            entry.model, syn_state, pre_rate, post_v, entry.model.dt
        )

        current_dict = entry.model.compute_current(new_syn_state, pre_rate, post_v)

        syn_I[entry.tgt] = syn_I[entry.tgt] + current_dict["I_syn"]
        syn_g[entry.tgt] = syn_g[entry.tgt] + current_dict["g_syn"]
        syn_states_dict[syn_name] = new_syn_state

    # Интегрирование популяций
    stability_error = stability_acc
    new_pop_states_list = []
    for name, pop, n in graph._pop_info_cache:
        pop_state = pop_states_dict[name]

        total_syn = {"I_syn": syn_I[name], "g_syn": syn_g[name]}
        new_pop_state, local_err = integrator.step(pop, pop_state, total_syn)
        new_pop_states_list.extend(new_pop_state)
        # local_err интеграторов — [1] (конвенция RK); аккумулятор скалярный.
        stability_error = stability_error + tf.reduce_mean(local_err)

    # Сборка списка состояний синапсов
    new_syn_states_list = []
    for name, _, _ in graph._syn_info_cache:
        new_syn_states_list.extend(syn_states_dict[name])

    if neuraltide.config.get_debug_numerics():
        for i, s in enumerate(new_pop_states_list):
            new_pop_states_list[i] = tf.debugging.check_numerics(
                s, f"NaN in population state[{i}]"
            )
        for i, s in enumerate(new_syn_states_list):
            new_syn_states_list[i] = tf.debugging.check_numerics(
                s, f"NaN in synapse state[{i}]"
            )

    return (tuple(new_pop_states_list), tuple(new_syn_states_list), stability_error)


class NetworkOutput(
    namedtuple(
        "_NetworkOutputBase",
        ["firing_rates", "stability_loss", "final_state"],
    )
):
    """
    Результат прогона NetworkRNN.

    Атрибуты:
        firing_rates: dict[str, tf.Tensor]
            Ключи — имена динамических популяций.
            Форма каждого тензора: [batch, T, n_units_i].
        stability_loss: tf.Tensor, scalar.
        final_state: tuple[StateList, StateList]
            Конечное состояние (pop_states, syn_states)
            для продолжения симуляции.
    """

    __slots__ = ()


class NetworkRNN(tf.keras.layers.Layer):
    """
    Симуляция сети на временно́й оси через tf.scan.

    Использует пошаговое интегрирование для каждого временного шага,
    что позволяет избежать проблем с variable scoping в Keras RNN.

    Режимы интегрирования (integration_mode):
        'split' (по умолчанию): синапсы и популяции интегрируются
            отдельно через integrator.step_synapse()/step(); токи
            вычисляются по уже обновлённым состояниям синапсов.
        'network': интегратор получает согласованную правую часть всей
            связанной сети (_JointNetworkRHS) через step_network(rhs,
            pop_states, syn_states, graph.dt) — требует интегратор с
            supports_network_step=True. Токи и производные вычисляются
            по состояниям на начало шага.

    Графовое ядро скана закреплено сигнатурами: self._scan_compiled
    (tf.function с input_signature) строится в _build() и не
    ретрейсится при смене длины T и батча B (батч-оси сигнатур —
    динамические). Публичный call() — eager-обёртка: нормализует ранг
    t_sequence, материализует inputs=None и initial_state=None
    (per-sample состояние broadcast'ится на B) до входа в скомпилированное
    ядро. Поддержан произвольный batch B ≥ 1: состояния [B, n], токи
    [B, n_post]; батч-прогон эквивалентен B индивидуальным прогонам с
    одинаковым протоколом.

    Args:
        jit_compile: если True, скан компилируется через XLA
            (tf.function(jit_compile=True)). По умолчанию выключено.
            Проверено на TF 2.21 (CPU): прямой прогон компилируется для
            всех встроенных интеграторов; train step с градиентами
            (Trainer(jit_compile=True)) компилируется для всех, КРОМЕ
            AdaptiveRKIntegrator (вложенный while_loop адаптивного шага),
            ExponentialEulerIntegrator (tf.linalg.expm) и split-шага
            RosenbrockIntegrator (динамические батч-оси скана —
            BroadcastGradientArgs; на статических формах компилируется):
            их градиент не компилируется под XLA и падает при первом
            вызове с InvalidArgumentError. Не сочетайте jit_compile=True у
            сети с Trainer по умолчанию (графовый train step без XLA):
            TensorArray-градиент не пересекает границу XLA/TF
            (UnimplementedError «TensorList crossing the XLA/TF boundary»)
            — используйте jit_compile у обоих или ни у кого.
    """

    def __init__(
        self,
        graph: NetworkGraph,
        integrator: BaseIntegrator,
        stability_penalty_weight: float = 0.0,
        *,
        integration_mode: str = "split",
        jit_compile: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Не даём Keras autocast приводить входы к float32: симуляция идёт
        # в глобальном neuraltide.config.get_dtype() (аналогично
        # BaseInputGenerator), чтобы float64-путь работал без потери точности.
        self.autocast = False
        if integration_mode not in ("split", "network"):
            raise ValueError(
                f"Unknown integration_mode '{integration_mode}'; "
                f"expected 'split' or 'network'."
            )
        if integration_mode == "network" and not getattr(
            integrator, "supports_network_step", False
        ):
            raise ValueError(
                f"integration_mode='network' requires an integrator with "
                f"supports_network_step=True; got "
                f"{type(integrator).__name__}."
            )
        graph.validate()
        self._graph = graph
        self._integrator = integrator
        self._stability_penalty_weight = stability_penalty_weight
        self._integration_mode = integration_mode
        self._jit_compile = bool(jit_compile)

        self._build()

    def _build(self) -> None:
        """Собирает кэшированные (per-sample, batch=1) начальные состояния
        и вычисляет размерности.

        Кэш хранит состояния формы [1, ...] (per-sample); при вызове с
        batch B>1 call() broadcast'ит их до [B, ...]. Батч-агностичные
        формы ([None, ...] — первая ось состояний) используются в
        сигнатуре скана и element_shape TensorArray.
        """
        batch_size = 1

        self._init_pop_states: StateList = []
        self._pop_state_offsets: Dict[str, int] = {}
        self._pop_state_shapes: List[tf.TensorShape] = []
        for name in self._graph.population_names:
            pop = self._graph._populations[name]
            self._pop_state_offsets[name] = len(self._init_pop_states)
            self._init_pop_states.extend(pop.get_initial_state(batch_size))
            self._pop_state_shapes.extend(pop.state_size)

        self._init_syn_states: StateList = []
        self._syn_state_offsets: Dict[str, int] = {}
        self._syn_state_shapes: List[tf.TensorShape] = []
        for name in self._graph.synapse_names:
            entry = self._graph._synapses[name]
            self._syn_state_offsets[name] = len(self._init_syn_states)
            self._init_syn_states.extend(entry.model.get_initial_state(batch_size))
            # Per-sample формы из публичного state_size ([n_pre, n_post] —
            # без батч-оси; кэшированные состояния — [1, n_pre, n_post]).
            self._syn_state_shapes.extend(entry.model.state_size)

        # Батч-агностичные формы состояний — для input_signature скана и
        # element_shape TensorArray: одна сигнатура на любой B, ретрейсинга
        # при смене батча нет. state_size популяций уже содержит батч-ось
        # ([1, n] → [None, n]); у синапсов per-sample форма [n_pre, n_post]
        # без батч-оси — она дописывается вперёд ([None, n_pre, n_post]).
        self._pop_state_elem_shapes: List[tf.TensorShape] = [
            _with_free_batch(s) for s in self._pop_state_shapes
        ]
        self._syn_state_elem_shapes: List[tf.TensorShape] = [
            tf.TensorShape([None]).concatenate(s) for s in self._syn_state_shapes
        ]

        self._total_dynamic_units = sum(
            pop.n_units for name, pop in self._graph._populations.items()
        )

        self._pop_state_count = sum(
            len(pop.state_size) for pop in self._graph._populations.values()
        )
        self._syn_state_count = sum(
            len(entry.model.state_size) for entry in self._graph._synapses.values()
        )

        self._cached_trainable_vars = self._collect_trainable_vars()
        self._scan_compiled = self._make_scan_function()

    def _make_scan_function(self):
        """Строит скомпилированное графовое ядро скана.

        input_signature фиксирует аргументы: смена длины T, смена батча B,
        чередование initial_state=None/тензоры и ранг t_sequence больше не
        порождают ретрейсинг (нормализация выполняется в eager-обёртке
        call()). Батч-ось всех тензоров — None (произвольное B).
        dtype фиксируется на момент построения сети: при смене global
        dtype нужен новый экземпляр сети (существующий контракт).
        """
        dtype = neuraltide.config.get_dtype()
        self._scan_dtype = dtype
        # Порядок элементов — как у параметров _scan_forward_states
        # (t_sequence, init_pop, init_syn, inputs).
        input_signature = [
            tf.TensorSpec([None, None, 1], dtype=dtype),
            tuple(tf.TensorSpec(s, dtype=dtype) for s in self._pop_state_elem_shapes),
            tuple(tf.TensorSpec(s, dtype=dtype) for s in self._syn_state_elem_shapes),
            tf.TensorSpec([None, None, self._graph.total_input_units], dtype=dtype),
        ]
        return tf.function(
            self._scan_forward_impl,
            input_signature=input_signature,
            jit_compile=self._jit_compile,
        )

    def _collect_trainable_vars(self) -> List[tf.Variable]:
        vars_ = list(super().trainable_variables)
        for pop in self._graph._populations.values():
            vars_.extend(pop.trainable_variables)
        for entry in self._graph._synapses.values():
            vars_.extend(entry.model.trainable_variables)
        return vars_

    def _scan_forward_states(
        self,
        t_sequence: TensorType,
        init_pop: Tuple[TensorType, ...],
        init_syn: Tuple[TensorType, ...],
        inputs: Optional[TensorType] = None,
    ) -> Tuple[
        Dict[str, TensorType],
        TensorType,
        Tuple[TensorType, ...],
        Tuple[TensorType, ...],
        Tuple[TensorType, ...],
        Tuple[TensorType, ...],
    ]:
        """Скан с возвратом stacked-тензоров состояний.

        Единая точка входа в скан для прямых вызовов (в т.ч.
        AdjointSolver.forward_pass): делегирует в _scan_compiled, поэтому
        значения совпадают с network(...) побитово. Чисто eager-исполнение
        тела скана (для сверок и отладки) — _scan_forward_impl.

        Внутренний контракт: входы должны быть нормализованы публичным
        call — t формы [B, T, 1] в dtype сети (батч-ось динамическая);
        inputs либо None (== нулевая ширина), либо тензор [B, T, total]
        точной ширины; чужой dtype теперь даёт TypeError на биндинге
        сигнатуры.
        """
        if inputs is None:
            inputs = tf.zeros(
                [tf.shape(t_sequence)[0], tf.shape(t_sequence)[1], 0],
                dtype=self._scan_dtype,
            )
        return self._scan_compiled(t_sequence, init_pop, init_syn, inputs)

    def _scan_forward_impl(
        self,
        t_sequence: TensorType,
        init_pop: Tuple[TensorType, ...],
        init_syn: Tuple[TensorType, ...],
        inputs: TensorType,
    ) -> Tuple[
        Dict[str, TensorType],
        TensorType,
        Tuple[TensorType, ...],
        Tuple[TensorType, ...],
        Tuple[TensorType, ...],
        Tuple[TensorType, ...],
    ]:
        """Тело скана (tf.while_loop + TensorArray). Трассируется внутри
        self._scan_compiled (см. _make_scan_function); при прямом вызове
        исполняется в eager-режиме. inputs здесь всегда материализованный
        тензор [batch, T, total_input_units] (нулевой ширины при отсутствии
        объявленных входов) — ветвления по None нет, ширина валидируется
        сигнатурой.
        """
        dtype = neuraltide.config.get_dtype()
        T = tf.shape(t_sequence)[1]
        elems = tf.transpose(t_sequence, [1, 0, 2])

        # Pre-transpose inputs: [batch, T, total] -> [T, batch, total]
        inputs_T = tf.transpose(inputs, [1, 0, 2])

        # Build dict of transposed slices per input name
        input_seqs = {}
        for name in self._graph._input_names:
            offset = self._graph._input_offsets[name]
            size = self._graph._external_inputs[name]
            input_seqs[name] = inputs_T[:, :, offset : offset + size]

        init_stability = tf.zeros([], dtype=dtype)

        pop_arrays = [
            tf.TensorArray(
                dtype,
                size=T,
                clear_after_read=False,
                element_shape=self._pop_state_elem_shapes[i],
            )
            for i in range(self._pop_state_count)
        ]
        syn_arrays = [
            tf.TensorArray(
                dtype,
                size=T,
                clear_after_read=False,
                element_shape=self._syn_state_elem_shapes[i],
            )
            for i in range(self._syn_state_count)
        ]
        stab_array = tf.TensorArray(
            dtype, size=T, clear_after_read=False, element_shape=[]
        )

        i0 = tf.constant(0)

        def _cond(i, pop_carry, syn_carry, stab_carry, pop_arrs, syn_arrs, stab_arr):
            return i < T

        def _body(i, pop_carry, syn_carry, stab_carry, pop_arrs, syn_arrs, stab_arr):
            t = elems[i]
            # Slice inputs at this time step
            input_step = {name: seq[i] for name, seq in input_seqs.items()}
            new_pop, new_syn, new_stab = _step_fn(
                (pop_carry, syn_carry, stab_carry),
                t,
                input_step,
                self._graph,
                self._integrator,
                integration_mode=self._integration_mode,
            )
            for j in range(self._pop_state_count):
                pop_arrs[j] = pop_arrs[j].write(i, new_pop[j])
            for j in range(self._syn_state_count):
                syn_arrs[j] = syn_arrs[j].write(i, new_syn[j])
            stab_arr = stab_arr.write(i, new_stab)
            return (i + 1, new_pop, new_syn, new_stab, pop_arrs, syn_arrs, stab_arr)

        loop_vars = (
            i0,
            init_pop,
            init_syn,
            init_stability,
            pop_arrays,
            syn_arrays,
            stab_array,
        )

        pop_state_inv = tuple(
            tf.TensorShape(s) if not isinstance(s, tf.TensorShape) else s
            for s in self._pop_state_elem_shapes
        )
        syn_state_inv = tuple(
            tf.TensorShape(s) if not isinstance(s, tf.TensorShape) else s
            for s in self._syn_state_elem_shapes
        )
        shape_inv = (
            tf.TensorShape([]),
            pop_state_inv,
            syn_state_inv,
            tf.TensorShape([]),
            [tf.TensorShape(None)] * len(pop_arrays),
            [tf.TensorShape(None)] * len(syn_arrays),
            tf.TensorShape(None),
        )

        # maximum_iterations=T нужен XLA-градиенту: обратный проход while_loop
        # с TensorArray иначе не компилируется (требуется фиксированный
        # размер tensor list). На семантику не влияет: _cond всё равно
        # останавливается на i == T.
        (*_, final_pop, final_syn, _, pop_arrs, syn_arrs, stab_arr) = tf.while_loop(
            _cond,
            _body,
            loop_vars,
            parallel_iterations=1,
            maximum_iterations=T,
            shape_invariants=shape_inv,
        )

        all_pop_stacked = tuple(ta.stack() for ta in pop_arrs)
        all_syn_stacked = tuple(ta.stack() for ta in syn_arrs)
        all_stability = stab_arr.stack()

        all_rates = {}
        for name in self._graph.dynamic_population_names:
            idx = self._pop_state_offsets[name]
            stacked_r = all_pop_stacked[idx]
            pop = self._graph._populations[name]
            rate = pop.get_firing_rate([stacked_r])
            all_rates[name] = tf.transpose(rate, [1, 0, 2])

        stability_loss = self._stability_penalty_weight * tf.reduce_mean(
            all_stability[-1]
        )

        final_pop_states = tuple([s[-1] for s in all_pop_stacked])
        final_syn_states = tuple([s[-1] for s in all_syn_stacked])

        return (
            all_rates,
            stability_loss,
            final_pop_states,
            final_syn_states,
            all_pop_stacked,
            all_syn_stacked,
        )

    def call(
        self,
        t_sequence: TensorType,
        inputs: Optional[TensorType] = None,
        initial_state: Optional[Tuple[StateList, StateList]] = None,
        training: Optional[bool] = None,
    ) -> NetworkOutput:
        """
        Симулирует сеть на протяжении временно́й последовательности.

        Eager-обёртка над скомпилированным графовым ядром (self._scan_compiled
        с закреплёнными input_signature): здесь нормализуются ранги и
        материализуются None-аргументы, поэтому смена длины T, смена батча,
        чередование initial_state=None/заданным состоянием и ранг 2 vs 3 у
        t_sequence не вызывают ретрейсинга. При вызове изнутри tf.function эта
        нормализация трассируется как обычные ops внешнего графа.

        Поддерживается произвольный batch B ≥ 1 (состояния [B, n], токи
        [B, n]): один батч-прогон эквивалентен B индивидуальным прогонам
        с одинаковым протоколом; initial_state=None broadcast'ит
        per-sample начальное состояние сети на все слоты батча. Все слоты
        батча исполняют один и тот же протокол (t_sequence/inputs одни
        на всех). InputPopulation внутри полного скана не поддержан.

        Args:
            t_sequence: tf.Tensor shape [batch, T, 1] или [batch, T]
                Временная последовательность в мс.
            inputs: tf.Tensor shape [batch, T, total_input_units]
                Предвычисленные частоты разрядов внешних входов,
                сложенные через pack_inputs(). None допустим только при
                отсутствии объявленных входов (материализуется тензор
                нулевой ширины). Батч должен совпадать с t_sequence.
            initial_state: Optional[Tuple[pop_states, syn_states]]
                Начальное состояние: тензоры формы [B, *state_shape] с
                единым B по всем слотам (хвосты форм — как у state_size
                моделей). Если None, используется per-sample начальное
                состояние популяций/синапсов сети, broadcast'нутое на B.
            training: игнорируется (симуляция не зависит от фазы обучения).

        Returns:
            NetworkOutput с firing_rates, final_state и stability_loss.
        """
        if t_sequence.shape.rank == 2:
            t_sequence = t_sequence[:, :, tf.newaxis]
        if t_sequence.dtype != self._scan_dtype:
            # До закрепления сигнатур чужой dtype у t_sequence проходил
            # незамеченным (t не участвует в шаге скана); сохраняем
            # совместимость явным приведением к dtype сети.
            t_sequence = tf.cast(t_sequence, self._scan_dtype)

        batch = tf.shape(t_sequence)[0]
        T = tf.shape(t_sequence)[1]
        if inputs is None:
            if self._graph.total_input_units > 0:
                raise ValueError(
                    f"Network has {len(self._graph._external_inputs)} "
                    f"declared inputs ({self._graph._input_names}) "
                    f"but no inputs provided."
                )
            inputs = tf.zeros([batch, T, 0], dtype=self._scan_dtype)
        else:
            expected_last = self._graph.total_input_units
            if inputs.shape[-1] != expected_last:
                raise ValueError(
                    f"inputs last dim is {inputs.shape[-1]}, "
                    f"expected {expected_last} "
                    f"(total_input_units from declare_input calls)"
                )
            # Дешёвая статическая сверка батчей (динамическое несовпадение
            # отловит broadcast внутри графа).
            if (
                t_sequence.shape[0] is not None
                and inputs.shape[0] is not None
                and int(t_sequence.shape[0]) != int(inputs.shape[0])
            ):
                raise ValueError(
                    f"t_sequence batch {int(t_sequence.shape[0])} != "
                    f"inputs batch {int(inputs.shape[0])}."
                )

        if initial_state is None:
            # Per-sample кэш [1, ...] -> broadcast на [B, ...]. При
            # статическом B=1 broadcast не нужен (и вреден под XLA:
            # градиент BroadcastTo требует константных аргументов формы).
            static_batch = t_sequence.shape[0]
            if static_batch == 1:
                init_pop = tuple(self._init_pop_states)
                init_syn = tuple(self._init_syn_states)
            else:
                init_pop = tuple(
                    tf.broadcast_to(t, [batch] + t.shape[1:].as_list())
                    for t in self._init_pop_states
                )
                init_syn = tuple(
                    tf.broadcast_to(t, [batch] + t.shape[1:].as_list())
                    for t in self._init_syn_states
                )
        else:
            init_pop, init_syn = initial_state
            if len(init_pop) != self._pop_state_count:
                raise ValueError(
                    f"initial_state[0] has {len(init_pop)} population state "
                    f"tensors, expected {self._pop_state_count}."
                )
            if len(init_syn) != self._syn_state_count:
                raise ValueError(
                    f"initial_state[1] has {len(init_syn)} synapse state "
                    f"tensors, expected {self._syn_state_count}."
                )
            # Хвосты форм должны совпадать с per-sample формами слотов;
            # батч — единый по всем тензорам (статически, где известен).
            user_batch = None
            for t in list(init_pop) + list(init_syn):
                b = t.shape[0] if isinstance(t, tf.Tensor) else None
                if b is not None:
                    user_batch = int(b) if user_batch is None else user_batch
                    if int(b) != user_batch:
                        raise ValueError(
                            "initial_state tensors must share one batch "
                            f"dimension, got {user_batch} and {int(b)}."
                        )
            if (
                user_batch is not None
                and t_sequence.shape[0] is not None
                and user_batch != int(t_sequence.shape[0])
            ):
                raise ValueError(
                    f"initial_state batch {user_batch} != t_sequence batch "
                    f"{int(t_sequence.shape[0])}."
                )
            for t, ref in zip(init_pop, self._pop_state_shapes):
                if t.shape.rank is not None and t.shape.rank != len(ref):
                    raise ValueError(
                        f"initial_state pop tensor rank {t.shape.rank} != "
                        f"expected {len(ref)} (per-sample shape {ref} with "
                        "batch axis)."
                    )
            for t, ref in zip(init_syn, self._syn_state_shapes):
                if t.shape.rank is not None and t.shape.rank != len(ref) + 1:
                    raise ValueError(
                        f"initial_state synapse tensor rank {t.shape.rank} != "
                        f"expected {len(ref) + 1} (per-sample shape {ref} "
                        "with batch axis)."
                    )
            init_pop = tuple(init_pop)
            init_syn = tuple(init_syn)

        all_rates, stability_loss, final_pop, final_syn, _, _ = self._scan_compiled(
            t_sequence, init_pop, init_syn, inputs
        )

        return NetworkOutput(
            firing_rates=all_rates,
            stability_loss=stability_loss,
            final_state=(final_pop, final_syn),
        )

    @property
    def integration_mode(self) -> str:
        """
        Режим интегрирования связанной сети: 'split' (по умолчанию)
        или 'network' (единый шаг интегратора по всей сети через
        step_network, требует supports_network_step=True у интегратора).
        """
        return self._integration_mode

    @property
    def jit_compile(self) -> bool:
        """True, если скан сети компилируется через XLA (jit_compile=True)."""
        return self._jit_compile

    @property
    def pop_state_offsets(self) -> Dict[str, Tuple[int, int]]:
        """
        Отображение имя популяции -> (start, count) в плоском списке
        начальных состояний популяций (первый элемент кортежа
        начального состояния, см. get_initial_state()).

        start — индекс первого тензора популяции, count — число тензоров
        (len(pop.state_size)). Возвращается копия; используется слоем
        калибровки для адресации обучаемых компонент начального состояния.
        """
        return {
            name: (start, len(self._graph._populations[name].state_size))
            for name, start in self._pop_state_offsets.items()
        }

    @property
    def trainable_variables(self) -> List[tf.Variable]:
        return self._cached_trainable_vars

    def get_initial_state(self, batch_size: int = 1) -> Tuple[StateList, StateList]:
        """Возвращает начальное состояние сети для батча batch_size.

        Популяции — тензоры [batch_size, n_i], синапсы —
        [batch_size, n_pre, n_post]. batch_size=1 — per-sample формы
        прежнего поведения.
        """
        init_pop = []
        for pop in self._graph._populations.values():
            init_pop.extend(pop.get_initial_state(batch_size))

        init_syn = []
        for entry in self._graph._synapses.values():
            init_syn.extend(entry.model.get_initial_state(batch_size))

        return init_pop, init_syn
