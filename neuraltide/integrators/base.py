from abc import ABC, abstractmethod
from typing import Dict, Tuple


from neuraltide.core.base import PopulationModel, SynapseModel
from neuraltide.core.types import TensorType, StateList


class BaseIntegrator(ABC):
    """
    Базовый класс интегратора ОДУ.

    Интегратор отдельно от популяционной модели — пользователь
    может менять схему интегрирования не изменяя модель.

    Network-wide интегрирование: NetworkRNN с integration_mode='network'
    вызывает step_network(rhs, pop_states, syn_states, dt) — единый
    согласованный шаг по объединённым плоским спискам состояний всех
    популяций и синапсов (rhs — _JointNetworkRHS). Поддержка метода
    флагируется supports_network_step.
    """

    supports_network_step: bool = False

    @abstractmethod
    def step(
        self,
        population: PopulationModel,
        state: StateList,
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
        """
        raise NotImplementedError

    @abstractmethod
    def step_synapse(
        self,
        synapse: SynapseModel,
        state: StateList,
        pre_firing_rate: TensorType,
        post_voltage: TensorType,
        dt: float,
    ) -> Tuple[StateList, TensorType]:
        """
        Выполняет один шаг интегрирования синапса.

        Args:
            synapse: экземпляр SynapseModel.
            state: текущее состояние синапса.
            pre_firing_rate: частота пресинаптических популяций, shape = [1, n_pre], в Гц.
            post_voltage: средний потенциал постсинаптических популяций, shape = [1, n_post].
            dt: шаг интегрирования в мс.

        Returns:
            (new_state, local_error_estimate):
                new_state: новое состояние синапса.
                local_error_estimate: tf.Tensor shape [1] — оценка локальной ошибки.
        """
        raise NotImplementedError

    def step_network(
        self,
        rhs,
        pop_states: StateList,
        syn_states: StateList,
        dt: float,
    ) -> Tuple[StateList, StateList]:
        """
        Один согласованный шаг по связанной сети (integration_mode='network').

        Args:
            rhs: согласованная правая часть связанной сети:
                rhs(pop_states, syn_states) -> (dpop_flat, dsyn_flat) —
                плоские списки производных в кэшированном порядке графа
                (_JointNetworkRHS из neuraltide.core.network).
            pop_states: плоский список тензоров состояний популяций
                (shape [1, n] каждый).
            syn_states: плоский список тензоров состояний синапсов.
            dt: шаг интегрирования в мс.

        Returns:
            (new_pop_states, new_syn_states) — полные плоские списки той же
            длины и структуры; блоки моделей с пустыми derivatives
            (InputPopulation, StaticSynapse) возвращаются неизменными.
            Оценка локальной ошибки в network-режиме не возвращается:
            NetworkRNN пока не аккумулирует stability_loss в этом режиме.

        Raises:
            NotImplementedError: если интегратор не поддерживает
                network-wide шаг (supports_network_step=False). Используйте
                EulerIntegrator, HeunIntegrator, RK4Integrator или
                RosenbrockIntegrator, либо integration_mode='split'.
        """
        raise NotImplementedError(
            f"{type(self).__name__} не поддерживает network-wide "
            "интегрирование через step_network() "
            "(integration_mode='network'). Используйте EulerIntegrator, "
            "HeunIntegrator, RK4Integrator или RosenbrockIntegrator, либо "
            "integration_mode='split'."
        )
