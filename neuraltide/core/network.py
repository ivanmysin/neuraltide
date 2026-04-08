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
        
        current_dict, new_syn_state = entry.model.forward(
            pre_rate, post_v, syn_state, entry.model.dt
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
    
    return (tuple(new_pop_states_list), tuple(new_syn_states_list), stability_error)


@dataclass
class NetworkOutput:
    """
    Результат прогона NetworkRNN.

    Атрибуты:
        firing_rates: dict[str, tf.Tensor]
            Ключи — имена динамических популяций.
            Форма каждого тензора: [batch, T, n_units_i].
        hidden_states: dict[str, dict[str, tf.Tensor]] или None.
        stability_loss: tf.Tensor, scalar.
    """

    firing_rates: Dict[str, TensorType]
    hidden_states: Optional[Dict[str, Dict[str, TensorType]]]
    stability_loss: TensorType


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
        return_sequences: bool = True,
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
        for pop in self._graph._populations.values():
            self._init_pop_states.extend(pop.get_initial_state(batch_size))
        
        self._init_syn_states: StateList = []
        for entry in self._graph._synapses.values():
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

    def call(
        self,
        t_sequence: TensorType,
        initial_state: Optional[Tuple[StateList, StateList]] = None,
        training: bool = False,
    ) -> NetworkOutput:
        """
        Симулирует сеть на протяжении временной последовательности.
        
        Args:
            t_sequence: tf.Tensor shape [batch, T, 1] или [batch, T]
                Временная последовательность в мс.
            initial_state: Optional[Tuple[pop_states, syn_states]]
                Начальное состояние. Если None, используется нулевое.
        
        Returns:
            NetworkOutput с firing_rates для каждой динамической популяции.
        """
        if t_sequence.shape.rank == 2:
            t_sequence = t_sequence[:, :, tf.newaxis]
        
        n_steps = int(t_sequence.shape[1])
        
        if initial_state is None:
            init_pop = list(self._init_pop_states)
            init_syn = list(self._init_syn_states)
        else:
            init_pop, init_syn = initial_state
        
        pop_states = list(init_pop)
        syn_states = list(init_syn)
        stability_acc = tf.zeros([1], dtype=neuraltide.config.get_dtype())
        
        pop_states_dict: Dict[str, StateList] = {}
        idx = 0
        for name in self._graph.population_names:
            pop = self._graph._populations[name]
            n = len(pop.state_size)
            pop_states_dict[name] = pop_states[idx:idx + n]
            idx += n
        
        dyn_names = self._graph.dynamic_population_names
        all_rates: Dict[str, List[TensorType]] = {name: [] for name in dyn_names}
        
        for step in range(n_steps):
            t = t_sequence[:, step:step+1, 0]
            
            pop_states_tuple = tuple(pop_states)
            syn_states_tuple = tuple(syn_states)
            
            new_pop, new_syn, stability_acc = _step_fn(
                (pop_states_tuple, syn_states_tuple, stability_acc),
                t,
                self._graph,
                self._integrator
            )
            
            pop_states = list(new_pop)
            syn_states = list(new_syn)
            
            idx = 0
            for name in self._graph.population_names:
                pop = self._graph._populations[name]
                n = len(pop.state_size)
                pop_states_dict[name] = pop_states[idx:idx + n]
                idx += n
            
            for name in dyn_names:
                pop = self._graph._populations[name]
                rate = pop.get_firing_rate(pop_states_dict[name])
                all_rates[name].append(rate)
        
        for name in dyn_names:
            all_rates[name] = tf.stack(all_rates[name], axis=1)
        
        stability_loss = self._stability_penalty_weight * tf.reduce_mean(stability_acc)
        
        return NetworkOutput(
            firing_rates=all_rates,
            hidden_states=None,
            stability_loss=stability_loss,
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
        """Возвращает начальное состояние сети."""
        init_pop = []
        for pop in self._graph._populations.values():
            init_pop.extend(pop.get_initial_state(batch_size))
        
        init_syn = []
        for entry in self._graph._synapses.values():
            init_syn.extend(entry.model.get_initial_state(batch_size))
        
        return init_pop, init_syn
