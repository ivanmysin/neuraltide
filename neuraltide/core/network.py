from collections import namedtuple
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, OrderedDict

import tensorflow as tf

import neuraltide
import neuraltide.config
from neuraltide.core.base import PopulationModel, SynapseModel, BaseInputGenerator
from neuraltide.core.state import NetworkState
from neuraltide.core.types import TensorType, StateList
from neuraltide.integrators.base import BaseIntegrator
from neuraltide.populations.input_population import InputPopulation


@dataclass
class _SynapseEntry:
    model: SynapseModel
    src: str
    tgt: str


def compute_synapse_current(
    synapse: SynapseModel,
    state: StateList,
    pre_firing_rate: TensorType,
    post_voltage: TensorType,
) -> Dict[str, TensorType]:
    """
    Compute synaptic current and conductance from synapse state.
    """
    return synapse.compute_current(state, pre_firing_rate, post_voltage)


class NetworkGraph:
    """
    Описание топологии сети: популяции и синаптические проекции.

    Популяции двух видов:
        - Динамические: IzhikevichMeanField, WilsonCowan и т.д.
        - Входные: InputPopulation (оборачивает BaseInputGenerator).

    Синаптические проекции:
        Любой тип SynapseModel от любой популяции (включая InputPopulation)
        к любой динамической популяции.
    """

    def __init__(self, dt: float):
        self.dt = dt
        self._populations: OrderedDict[str, PopulationModel] = OrderedDict()
        self._synapses: OrderedDict[str, _SynapseEntry] = OrderedDict()

    def add_population(self, name: str, model: PopulationModel) -> None:
        """Регистрирует популяцию (динамическую или InputPopulation)."""
        if name in self._populations:
            raise ValueError(f"Population '{name}' already registered.")
        self._populations[name] = model

    def add_input_population(
        self,
        name: str,
        generator: BaseInputGenerator,
    ) -> None:
        """Регистрирует входной генератор как псевдо-популяцию."""
        pop = InputPopulation(generator=generator, name=name + '_input_pop')
        self.add_population(name, pop)

    def add_synapse(
        self,
        name: str,
        model: SynapseModel,
        src: str,
        tgt: str,
    ) -> None:
        """Регистрирует синаптическую проекцию."""
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
        model._container_name = name
        self._synapses[name] = _SynapseEntry(model=model, src=src, tgt=tgt)

    def validate(self) -> None:
        """Проверяет корректность топологии перед построением NetworkRNN."""
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
        for pop_name, pop in self._populations.items():
            if isinstance(pop, InputPopulation):
                continue
            has_input = any(e.tgt == pop_name for e in self._synapses.values())
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
    for name in graph.population_names:
        pop = graph._populations[name]
        n = len(pop.state_size)
        pop_states_dict[name] = flat_pop_states[idx:idx + n]
        idx += n
    
    syn_states_dict = {}
    idx = 0
    for name in graph.synapse_names:
        entry = graph._synapses[name]
        n = len(entry.model.state_size)
        syn_states_dict[name] = flat_syn_states[idx:idx + n]
        idx += n
    
    return pop_states_dict, syn_states_dict


def pack_state(
    pop_states_dict: Dict[str, StateList],
    syn_states_dict: Dict[str, StateList],
) -> Tuple[StateList, StateList]:
    """
    Pack state dictionaries into flat lists.
    
    Args:
        pop_states_dict: Dict of population states
        syn_states_dict: Dict of synapse states
    
    Returns:
        (flat_pop_states, flat_syn_states)
    """
    flat_pop = []
    for name in pop_states_dict:
        flat_pop.extend(pop_states_dict[name])
    
    flat_syn = []
    for name in syn_states_dict:
        flat_syn.extend(syn_states_dict[name])
    
    return flat_pop, flat_syn


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
    for name in graph.dynamic_population_names:
        pop = graph._populations[name]
        all_rates[name] = pop.get_firing_rate(pop_states_dict[name])
    return all_rates


def _step_fn(
    states: Tuple[StateList, StateList, TensorType],
    t: TensorType,
    graph: NetworkGraph,
    integrator: BaseIntegrator,
) -> Tuple[StateList, StateList, TensorType]:
    """
    Один шаг симуляции сети. Используется с tf.scan.
    
    Args:
        states: (pop_states, syn_states, stability_error) - текущие состояния
        t: текущее время
        graph: NetworkGraph
        integrator: BaseIntegrator
    
    Returns:
        (new_pop_states, new_syn_states, new_stability_error)
    """
    pop_states, syn_states, stability_acc = states
    dtype = neuraltide.config.get_dtype()
    
    pop_states_dict = {}
    idx = 0
    for name in graph.population_names:
        pop = graph._populations[name]
        n = len(pop.state_size)
        pop_states_dict[name] = pop_states[idx:idx + n]
        idx += n
    
    syn_states_dict = {}
    idx = 0
    for name in graph.synapse_names:
        entry = graph._synapses[name]
        n = len(entry.model.state_size)
        syn_states_dict[name] = syn_states[idx:idx + n]
        idx += n
    
    for name in graph.population_names:
        pop = graph._populations[name]
        if isinstance(pop, InputPopulation):
            pop_states_dict[name] = [t]
    
    syn_I: Dict[str, TensorType] = {}
    syn_g: Dict[str, TensorType] = {}
    for name in graph.dynamic_population_names:
        n = graph._populations[name].n_units
        syn_I[name] = tf.zeros([1, n], dtype=dtype)
        syn_g[name] = tf.zeros([1, n], dtype=dtype)
    
    for syn_name, entry in graph._synapses.items():
        src_pop = graph._populations[entry.src]
        tgt_pop = graph._populations[entry.tgt]
        src_state = pop_states_dict[entry.src]
        tgt_state = pop_states_dict[entry.tgt]
        syn_state = syn_states_dict[syn_name]
        
        pre_rate = src_pop.get_firing_rate(src_state)

        tgt_obs = tgt_pop.observables(tgt_state)
        post_v = tgt_obs.get(
            'v_mean',
            tf.zeros([1, tgt_pop.n_units], dtype=dtype)
        )

        new_syn_state, local_err = integrator.step_synapse(
            entry.model, syn_state, pre_rate, post_v, entry.model.dt
        )

        current_dict = compute_synapse_current(
            entry.model, new_syn_state, pre_rate, post_v
        )

        syn_I[entry.tgt] = syn_I[entry.tgt] + current_dict['I_syn']
        syn_g[entry.tgt] = syn_g[entry.tgt] + current_dict['g_syn']
        syn_states_dict[syn_name] = new_syn_state
    
    stability_error = stability_acc
    new_pop_states_list = []
    for name in graph.population_names:
        pop = graph._populations[name]
        pop_state = pop_states_dict[name]
        
        if isinstance(pop, InputPopulation):
            new_pop_states_list.extend(pop_state)
        else:
            total_syn = {'I_syn': syn_I[name], 'g_syn': syn_g[name]}
            new_pop_state, local_err = integrator.step(pop, pop_state, total_syn)
            new_pop_states_list.extend(new_pop_state)
            stability_error = stability_error + local_err
    
    new_syn_states_list = []
    for name in graph.synapse_names:
        new_syn_states_list.extend(syn_states_dict[name])

    for i, s in enumerate(new_pop_states_list):
        new_pop_states_list[i] = neuraltide.config.maybe_check_numerics(s, f'NaN in population state[{i}]')
    for i, s in enumerate(new_syn_states_list):
        new_syn_states_list[i] = neuraltide.config.maybe_check_numerics(s, f'NaN in synapse state[{i}]')

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
        for name in self._graph.population_names:
            pop = self._graph._populations[name]
            self._pop_state_offsets[name] = len(self._init_pop_states)
            self._init_pop_states.extend(pop.get_initial_state(batch_size))
        
        self._init_syn_states: StateList = []
        self._syn_state_offsets: Dict[str, int] = {}
        for name in self._graph.synapse_names:
            entry = self._graph._synapses[name]
            self._syn_state_offsets[name] = len(self._init_syn_states)
            self._init_syn_states.extend(entry.model.get_initial_state(batch_size))
        
        self._total_dynamic_units = sum(
            pop.n_units
            for name, pop in self._graph._populations.items()
            if not isinstance(pop, InputPopulation)
        )
        
        self._pop_state_count = sum(
            len(pop.state_size)
            for pop in self._graph._populations.values()
        )
        self._syn_state_count = sum(
            len(entry.model.state_size)
            for entry in self._graph._synapses.values()
        )

    @tf.function
    def _scan_forward(
        self,
        t_sequence: TensorType,
        init_pop: Tuple[TensorType, ...],
        init_syn: Tuple[TensorType, ...],
    ) -> Tuple[Dict[str, TensorType], TensorType, StateList, StateList]:
        """Сканирует временну́ю ось через tf.scan. Возвращает rates, stability, final states."""
        init_stability = tf.zeros([1], dtype=neuraltide.config.get_dtype())

        elems = tf.transpose(t_sequence, [1, 0, 2])

        scan_all = tf.scan(
            lambda carry, t: _step_fn(carry, t, self._graph, self._integrator),
            elems=elems,
            initializer=(init_pop, init_syn, init_stability),
            parallel_iterations=1,
        )

        all_pop_stacked, all_syn_stacked, all_stability = scan_all

        all_rates = {}
        for name in self._graph.dynamic_population_names:
            idx = self._pop_state_offsets[name]
            stacked_r = all_pop_stacked[idx]
            pop = self._graph._populations[name]
            rate = pop.get_firing_rate([stacked_r])
            all_rates[name] = tf.transpose(rate, [1, 0, 2])

        stability_loss = self._stability_penalty_weight * tf.reduce_mean(all_stability[-1])

        final_pop_states = [s[-1] for s in all_pop_stacked]
        final_syn_states = [s[-1] for s in all_syn_stacked]

        return all_rates, stability_loss, final_pop_states, final_syn_states

    @tf.function
    def call(
        self,
        t_sequence: TensorType,
        initial_state: Optional[Tuple[StateList, StateList]] = None,
        training: bool = False,
    ) -> NetworkOutput:
        """
        Симулирует сеть на протяжении временно́й последовательности.

        Args:
            t_sequence: tf.Tensor shape [batch, T, 1] или [batch, T]
                Временная последовательность в мс.
            initial_state: Optional[Tuple[pop_states, syn_states]]
                Начальное состояние. Если None, используется нулевое.
            training: bool — режим обучения.

        Returns:
            NetworkOutput с firing_rates, final_state и stability_loss.
        """
        if t_sequence.shape.rank == 2:
            t_sequence = t_sequence[:, :, tf.newaxis]

        if initial_state is not None:
            init_pop, init_syn = initial_state
        else:
            init_pop = list(self._init_pop_states)
            init_syn = list(self._init_syn_states)

        all_rates, stability_loss, final_pop, final_syn = \
            self._scan_forward(t_sequence, tuple(init_pop), tuple(init_syn))

        return NetworkOutput(
            firing_rates=all_rates,
            hidden_states=None,
            stability_loss=stability_loss,
            final_state=(final_pop, final_syn),
        )
    
    @property
    def trainable_variables(self) -> List[tf.Variable]:
        """Агрегирует trainable_variables из всех популяций и синапсов графа."""
        vars_ = list(super().trainable_variables)
        for pop in self._graph._populations.values():
            vars_.extend(pop.trainable_variables)
        for entry in self._graph._synapses.values():
            vars_.extend(entry.model.trainable_variables)
        return vars_

    def get_initial_state(self, batch_size: int = 1) -> Tuple[StateList, StateList]:
        """Возвращает начальное состояние сети (по умолчанию нулевое)."""
        init_pop = []
        for pop in self._graph._populations.values():
            init_pop.extend(pop.get_initial_state(batch_size))
        
        init_syn = []
        for entry in self._graph._synapses.values():
            init_syn.extend(entry.model.get_initial_state(batch_size))
        
        return init_pop, init_syn
