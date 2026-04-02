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
        pop = InputPopulation(generator=generator, dt=self.dt, name=name + '_input_pop')
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


class NetworkRNNCell(tf.keras.layers.Layer):
    """
    RNN-ячейка, реализующая один шаг динамики сети.

    Входной сигнал:
        inputs: tf.Tensor shape [batch, 1] — текущее время t в мс.

    Логика одного шага:
        1. Распаковать states → NetworkState.
        2. Обновить InputPopulation: new_state = [t].
        3. Для каждой синаптической проекции: вычислить I_syn, g_syn.
        4. Для каждой динамической популяции: интегрировать.
        5. Упаковать NetworkState → flat list.
        6. output = concat(firing rates) + stability_error.
    """

    def __init__(self, graph: NetworkGraph, integrator: BaseIntegrator, **kwargs):
        super().__init__(**kwargs)
        graph.validate()
        self.graph = graph
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
        return self._total_dynamic_units + 1

    def get_initial_state(self, inputs=None, batch_size: int = 1, dtype=None) -> StateList:
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
        dtype = neuraltide.config.get_dtype()
        t = inputs

        net_state = NetworkState.from_flat_list(
            flat=list(states),
            pop_names=self.graph.population_names,
            pop_sizes=self._pop_state_sizes,
            syn_names=self.graph.synapse_names,
            syn_sizes=self._syn_state_sizes,
        )

        for name, pop in self.graph._populations.items():
            if isinstance(pop, InputPopulation):
                net_state.population_states[name] = [t]

        syn_I: Dict[str, TensorType] = {}
        syn_g: Dict[str, TensorType] = {}
        for name in self.graph.dynamic_population_names:
            n = self.graph._populations[name].n_units
            syn_I[name] = tf.zeros([1, n], dtype=dtype)
            syn_g[name] = tf.zeros([1, n], dtype=dtype)

        for syn_name, entry in self.graph._synapses.items():
            src_pop = self.graph._populations[entry.src]
            tgt_pop = self.graph._populations[entry.tgt]
            src_state = net_state.population_states[entry.src]
            tgt_state = net_state.population_states[entry.tgt]
            syn_state = net_state.synapse_states[syn_name]

            pre_rate = src_pop.get_firing_rate(src_state)

            tgt_obs = tgt_pop.observables(tgt_state)
            post_v = tgt_obs.get(
                'v_mean',
                tf.zeros([1, tgt_pop.n_units], dtype=dtype)
            )

            current_dict, new_syn_state = entry.model.forward(
                pre_rate, post_v, syn_state, entry.model.dt
            )

            syn_I[entry.tgt] += current_dict['I_syn']
            syn_g[entry.tgt] += current_dict['g_syn']
            net_state.synapse_states[syn_name] = new_syn_state

        stability_error = tf.zeros([1], dtype=dtype)
        for name in self.graph.dynamic_population_names:
            pop = self.graph._populations[name]
            pop_state = net_state.population_states[name]
            total_syn = {'I_syn': syn_I[name], 'g_syn': syn_g[name]}

            new_pop_state, local_err = self.integrator.step(
                pop, pop_state, total_syn
            )
            net_state.population_states[name] = new_pop_state
            stability_error += local_err

        rates = tf.concat(
            [
                self.graph._populations[name].get_firing_rate(
                    net_state.population_states[name]
                )
                for name in self.graph.dynamic_population_names
            ],
            axis=-1
        )

        output = tf.concat(
            [rates, stability_error[tf.newaxis, :]],
            axis=-1
        )

        new_states_flat = net_state.to_flat_list()

        return output, new_states_flat


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
    Обёртка над tf.keras.layers.RNN для симуляции сети на временно́й оси.
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
        self._cell = NetworkRNNCell(graph, integrator)
        self._graph = graph
        self._return_hidden_states = return_hidden_states
        self._stability_penalty_weight = stability_penalty_weight
        self._rnn = tf.keras.layers.RNN(
            self._cell,
            return_sequences=return_sequences,
            return_state=False,
        )

    def call(
        self,
        t_sequence: TensorType,
        initial_state: Optional[StateList] = None,
        training: bool = False,
    ) -> NetworkOutput:
        if initial_state is None:
            initial_state = self._cell.get_initial_state(batch_size=1)

        raw = self._rnn(t_sequence, initial_state=initial_state, training=training)

        total_units = self._cell._total_dynamic_units
        raw_rates = raw[:, :, :total_units]
        raw_err = raw[:, :, total_units:]

        firing_rates: Dict[str, TensorType] = {}
        offset = 0
        for name in self._graph.dynamic_population_names:
            n = self._graph._populations[name].n_units
            firing_rates[name] = raw_rates[:, :, offset:offset + n]
            offset += n

        stability_loss = self._stability_penalty_weight * tf.reduce_mean(raw_err)

        return NetworkOutput(
            firing_rates=firing_rates,
            hidden_states=None,
            stability_loss=stability_loss,
        )
