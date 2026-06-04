from collections import namedtuple
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import tensorflow as tf

import neuraltide
import neuraltide.config
from neuraltide.core.base import PopulationModel, SynapseModel
from neuraltide.core.types import TensorType, StateList
from neuraltide.integrators.base import BaseIntegrator


def _tensorarr_shape(t: TensorType) -> List[Optional[int]]:
    return [d if d is not None else None for d in t.shape.as_list()]


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
            raise ValueError(
                f"Input name '{name}' conflicts with population name."
            )
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
                    f"Population '{pop_name}' has no incoming synapses.",
                    UserWarning
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
                    f"Input '{name}' not provided. "
                    f"Expected: {self._input_names}"
                )
            t = input_dict[name]
            expected = self._external_inputs[name]
            if t.shape[-1] != expected:
                raise ValueError(
                    f"Input '{name}': last dim is {t.shape[-1]}, "
                    f"expected {expected}"
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
        pop_states_dict[name] = flat_pop_states[idx:idx + n]
        idx += n
    
    syn_states_dict = {}
    idx = 0
    # Use cached info for efficient iteration
    for name, entry, n in graph._syn_info_cache:
        syn_states_dict[name] = flat_syn_states[idx:idx + n]
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


def _step_fn(
    states: Tuple[StateList, StateList, TensorType],
    t: TensorType,
    input_rates: Optional[Dict[str, TensorType]],
    graph: NetworkGraph,
    integrator: BaseIntegrator,
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
    
    Returns:
        (new_pop_states, new_syn_states, new_stability_error)
    """
    pop_states, syn_states, stability_acc = states
    dtype = neuraltide.config.get_dtype()
    
    # Распаковка состояний популяций с использованием кэшированных оффсетов
    pop_states_dict = {}
    idx = 0
    for name, pop, n in graph._pop_info_cache:
        pop_states_dict[name] = pop_states[idx:idx + n]
        idx += n

    # Распаковка состояний синапсов с использованием кэшированных оффсетов
    syn_states_dict = {}
    idx = 0
    for name, entry, n in graph._syn_info_cache:
        syn_states_dict[name] = syn_states[idx:idx + n]
        idx += n

    # Инициализация синаптических токов для динамических популяций
    syn_I: Dict[str, TensorType] = {}
    syn_g: Dict[str, TensorType] = {}
    for name in graph.dynamic_population_names:
        n = graph._populations[name].n_units
        syn_I[name] = tf.zeros([1, n], dtype=dtype)
        syn_g[name] = tf.zeros([1, n], dtype=dtype)

    # Обработка всех синапсов
    if input_rates is None:
        input_rates = {}
    for syn_name, entry, _ in graph._syn_info_cache:
        tgt_state = pop_states_dict[entry.tgt]
        syn_state = syn_states_dict[syn_name]

        tgt_pop = graph._populations[entry.tgt]

        # Получение пресинаптической частоты разрядов
        if entry.src in graph._external_inputs:
            pre_rate = input_rates[entry.src]
        else:
            src_pop = graph._populations[entry.src]
            src_state = pop_states_dict[entry.src]
            pre_rate = src_pop.get_firing_rate(src_state)

        tgt_obs = tgt_pop.observables(tgt_state)
        post_v = tgt_obs.get('v_mean')
        if post_v is None:
            post_v = tf.zeros([1, tgt_pop.n_units], dtype=dtype)

        new_syn_state, local_err = integrator.step_synapse(
            entry.model, syn_state, pre_rate, post_v, entry.model.dt
        )

        current_dict = entry.model.compute_current(
            new_syn_state, pre_rate, post_v
        )

        syn_I[entry.tgt] = syn_I[entry.tgt] + current_dict['I_syn']
        syn_g[entry.tgt] = syn_g[entry.tgt] + current_dict['g_syn']
        syn_states_dict[syn_name] = new_syn_state

    # Интегрирование популяций
    stability_error = stability_acc
    new_pop_states_list = []
    for name, pop, n in graph._pop_info_cache:
        pop_state = pop_states_dict[name]

        total_syn = {'I_syn': syn_I[name], 'g_syn': syn_g[name]}
        new_pop_state, local_err = integrator.step(pop, pop_state, total_syn)
        new_pop_states_list.extend(new_pop_state)
        stability_error = stability_error + local_err

    # Сборка списка состояний синапсов
    new_syn_states_list = []
    for name, _, _ in graph._syn_info_cache:
        new_syn_states_list.extend(syn_states_dict[name])

    if neuraltide.config.get_debug_numerics():
        for i, s in enumerate(new_pop_states_list):
            new_pop_states_list[i] = tf.debugging.check_numerics(s, f'NaN in population state[{i}]')
        for i, s in enumerate(new_syn_states_list):
            new_syn_states_list[i] = tf.debugging.check_numerics(s, f'NaN in synapse state[{i}]')

    return (tuple(new_pop_states_list), tuple(new_syn_states_list), stability_error)


class NetworkOutput(namedtuple('_NetworkOutputBase',
    ['firing_rates', 'hidden_states', 'stability_loss', 'final_state'])):
    """
    Результат прогона NetworkRNN.

    Атрибуты:
        firing_rates: dict[str, tf.Tensor]
            Ключи — имена динамических популяций.
            Форма каждого тензора: [batch, T, n_units_i].
        hidden_states: dict[str, dict[str, tf.Tensor]] или None.
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
    """

    def __init__(
        self,
        graph: NetworkGraph,
        integrator: BaseIntegrator,
        return_hidden_states: bool = False,
        stability_penalty_weight: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        graph.validate()
        self._graph = graph
        self._integrator = integrator
        self._return_hidden_states = return_hidden_states
        self._stability_penalty_weight = stability_penalty_weight

        self._build()

    def _build(self) -> None:
        """Собирает начальные состояния и вычисляет размерности."""
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
        for name in self._graph.synapse_names:
            entry = self._graph._synapses[name]
            self._syn_state_offsets[name] = len(self._init_syn_states)
            self._init_syn_states.extend(entry.model.get_initial_state(batch_size))
        
        self._total_dynamic_units = sum(
            pop.n_units
            for name, pop in self._graph._populations.items()
        )
        
        self._pop_state_count = sum(
            len(pop.state_size)
            for pop in self._graph._populations.values()
        )
        self._syn_state_count = sum(
            len(entry.model.state_size)
            for entry in self._graph._synapses.values()
        )

        self._cached_trainable_vars = self._collect_trainable_vars()

    def _collect_trainable_vars(self) -> List[tf.Variable]:
        vars_ = list(super().trainable_variables)
        for pop in self._graph._populations.values():
            vars_.extend(pop.trainable_variables)
        for entry in self._graph._synapses.values():
            vars_.extend(entry.model.trainable_variables)
        return vars_

    @tf.function
    def _scan_forward(
        self,
        t_sequence: TensorType,
        inputs: TensorType,
        init_pop: Tuple[TensorType, ...],
        init_syn: Tuple[TensorType, ...],
    ) -> Tuple[Dict[str, TensorType], TensorType, StateList, StateList]:
        """Сканирует временну́ю ось через tf.while_loop + TensorArray."""
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
            input_seqs[name] = inputs_T[:, :, offset:offset + size]

        init_stability = tf.zeros([1], dtype=dtype)

        pop_arrays = [
            tf.TensorArray(dtype, size=T, clear_after_read=False,
                           element_shape=self._pop_state_shapes[i])
            for i in range(self._pop_state_count)
        ]
        syn_arrays = [
            tf.TensorArray(dtype, size=T, clear_after_read=False,
                           element_shape=_tensorarr_shape(init_syn[i]))
            for i in range(self._syn_state_count)
        ]
        stab_array = tf.TensorArray(dtype, size=T, clear_after_read=False,
                                    element_shape=[1])

        i0 = tf.constant(0)

        def _cond(i, pop_carry, syn_carry, stab_carry,
                  pop_arrs, syn_arrs, stab_arr):
            return i < T

        def _body(i, pop_carry, syn_carry, stab_carry,
                  pop_arrs, syn_arrs, stab_arr):
            t = elems[i]
            # Slice inputs at this time step
            input_step = {name: seq[i] for name, seq in input_seqs.items()}
            new_pop, new_syn, new_stab = _step_fn(
                (pop_carry, syn_carry, stab_carry),
                t, input_step, self._graph, self._integrator,
            )
            for j in range(self._pop_state_count):
                pop_arrs[j] = pop_arrs[j].write(i, new_pop[j])
            for j in range(self._syn_state_count):
                syn_arrs[j] = syn_arrs[j].write(i, new_syn[j])
            stab_arr = stab_arr.write(i, new_stab)
            return (i + 1, new_pop, new_syn, new_stab,
                    pop_arrs, syn_arrs, stab_arr)

        loop_vars = (i0, init_pop, init_syn, init_stability,
                     pop_arrays, syn_arrays, stab_array)

        pop_state_inv = tuple(
            tf.TensorShape(s) if not isinstance(s, tf.TensorShape) else s
            for s in self._pop_state_shapes
        )
        shape_inv = (
            tf.TensorShape([]),
            pop_state_inv,
            tuple(t.shape for t in init_syn),
            tf.TensorShape([1]),
            [tf.TensorShape(None)] * len(pop_arrays),
            [tf.TensorShape(None)] * len(syn_arrays),
            tf.TensorShape(None),
        )

        (*_, final_pop, final_syn, _,
         pop_arrs, syn_arrs, stab_arr) = tf.while_loop(
            _cond, _body, loop_vars, parallel_iterations=1,
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
            all_stability[-1])

        final_pop_states = [s[-1] for s in all_pop_stacked]
        final_syn_states = [s[-1] for s in all_syn_stacked]

        return all_rates, stability_loss, final_pop_states, final_syn_states

    @tf.function
    def _scan_forward_states(
        self,
        t_sequence: TensorType,
        init_pop: Tuple[TensorType, ...],
        init_syn: Tuple[TensorType, ...],
        *,
        inputs: Optional[TensorType] = None,
    ) -> Tuple[
        Dict[str, TensorType],
        TensorType,
        Tuple[TensorType, ...],
        Tuple[TensorType, ...],
        List[TensorType],
        List[TensorType],
    ]:
        """Same as _scan_forward but also returns stacked state tensors."""
        dtype = neuraltide.config.get_dtype()
        T = tf.shape(t_sequence)[1]
        elems = tf.transpose(t_sequence, [1, 0, 2])

        if inputs is None:
            inputs = tf.zeros(
                [tf.shape(t_sequence)[0], tf.shape(t_sequence)[1], 0],
                dtype=dtype)

        # Pre-transpose inputs: [batch, T, total] -> [T, batch, total]
        inputs_T = tf.transpose(inputs, [1, 0, 2])

        # Build dict of transposed slices per input name
        input_seqs = {}
        for name in self._graph._input_names:
            offset = self._graph._input_offsets[name]
            size = self._graph._external_inputs[name]
            input_seqs[name] = inputs_T[:, :, offset:offset + size]

        init_stability = tf.zeros([1], dtype=dtype)

        pop_arrays = [
            tf.TensorArray(dtype, size=T, clear_after_read=False,
                           element_shape=self._pop_state_shapes[i])
            for i in range(self._pop_state_count)
        ]
        syn_arrays = [
            tf.TensorArray(dtype, size=T, clear_after_read=False,
                           element_shape=_tensorarr_shape(init_syn[i]))
            for i in range(self._syn_state_count)
        ]
        stab_array = tf.TensorArray(dtype, size=T, clear_after_read=False,
                                    element_shape=[1])

        i0 = tf.constant(0)

        def _cond(i, pop_carry, syn_carry, stab_carry,
                  pop_arrs, syn_arrs, stab_arr):
            return i < T

        def _body(i, pop_carry, syn_carry, stab_carry,
                  pop_arrs, syn_arrs, stab_arr):
            t = elems[i]
            # Slice inputs at this time step
            input_step = {name: seq[i] for name, seq in input_seqs.items()}
            new_pop, new_syn, new_stab = _step_fn(
                (pop_carry, syn_carry, stab_carry),
                t, input_step, self._graph, self._integrator,
            )
            for j in range(self._pop_state_count):
                pop_arrs[j] = pop_arrs[j].write(i, new_pop[j])
            for j in range(self._syn_state_count):
                syn_arrs[j] = syn_arrs[j].write(i, new_syn[j])
            stab_arr = stab_arr.write(i, new_stab)
            return (i + 1, new_pop, new_syn, new_stab,
                    pop_arrs, syn_arrs, stab_arr)

        loop_vars = (i0, init_pop, init_syn, init_stability,
                      pop_arrays, syn_arrays, stab_array)

        pop_state_inv = tuple(
            tf.TensorShape(s) if not isinstance(s, tf.TensorShape) else s
            for s in self._pop_state_shapes
        )
        shape_inv = (
            tf.TensorShape([]),
            pop_state_inv,
            tuple(t.shape for t in init_syn),
            tf.TensorShape([1]),
            [tf.TensorShape(None)] * len(pop_arrays),
            [tf.TensorShape(None)] * len(syn_arrays),
            tf.TensorShape(None),
        )

        (*_, final_pop, final_syn, _,
          pop_arrs, syn_arrs, stab_arr) = tf.while_loop(
            _cond, _body, loop_vars, parallel_iterations=1,
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
            all_stability[-1])

        final_pop_states = tuple([s[-1] for s in all_pop_stacked])
        final_syn_states = tuple([s[-1] for s in all_syn_stacked])

        return (all_rates, stability_loss,
                final_pop_states, final_syn_states,
                all_pop_stacked, all_syn_stacked)

    @tf.function
    def call(
        self,
        t_sequence: TensorType,
        inputs: Optional[TensorType] = None,
        initial_state: Optional[Tuple[StateList, StateList]] = None,
    ) -> NetworkOutput:
        """
        Симулирует сеть на протяжении временно́й последовательности.

        Args:
            t_sequence: tf.Tensor shape [batch, T, 1] или [batch, T]
                Временная последовательность в мс.
            inputs: tf.Tensor shape [batch, T, total_input_units]
                Предвычисленные частоты разрядов внешних входов,
                сложенные через pack_inputs().
            initial_state: Optional[Tuple[pop_states, syn_states]]
                Начальное состояние. Если None, используется нулевое.

        Returns:
            NetworkOutput с firing_rates, final_state и stability_loss.
        """
        if t_sequence.shape.rank == 2:
            t_sequence = t_sequence[:, :, tf.newaxis]

        # Validate inputs
        if self._graph.total_input_units > 0:
            if inputs is None:
                raise ValueError(
                    f"Network has {len(self._graph._external_inputs)} "
                    f"declared inputs ({self._graph._input_names}) "
                    f"but no inputs provided."
                )
            expected_last = self._graph.total_input_units
            if inputs.shape[-1] != expected_last:
                raise ValueError(
                    f"inputs last dim is {inputs.shape[-1]}, "
                    f"expected {expected_last} "
                    f"(total_input_units from declare_input calls)"
                )
        else:
            if inputs is None:
                inputs = tf.zeros(
                    [tf.shape(t_sequence)[0], tf.shape(t_sequence)[1], 0],
                    dtype=neuraltide.config.get_dtype())

        if initial_state is not None:
            init_pop, init_syn = initial_state
        else:
            init_pop = list(self._init_pop_states)
            init_syn = list(self._init_syn_states)

        all_rates, stability_loss, final_pop, final_syn = \
            self._scan_forward(t_sequence, inputs, tuple(init_pop), tuple(init_syn))

        return NetworkOutput(
            firing_rates=all_rates,
            hidden_states=None,
            stability_loss=stability_loss,
            final_state=(final_pop, final_syn),
        )
    
    @property
    def trainable_variables(self) -> List[tf.Variable]:
        return self._cached_trainable_vars

    def get_initial_state(self, batch_size: int = 1) -> Tuple[StateList, StateList]:
        """Возвращает начальное состояние сети (по умолчанию нулевое)."""
        init_pop = []
        for pop in self._graph._populations.values():
            init_pop.extend(pop.get_initial_state(batch_size))
        
        init_syn = []
        for entry in self._graph._synapses.values():
            init_syn.extend(entry.model.get_initial_state(batch_size))
        
        return init_pop, init_syn
