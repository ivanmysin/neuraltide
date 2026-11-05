from typing import Dict, Tuple

import tensorflow as tf

import neuraltide.config
from neuraltide.core.base import PopulationModel, SynapseModel
from neuraltide.core.types import TensorType, StateList
from neuraltide.integrators.base import BaseIntegrator
from neuraltide.integrators.stiffness import joint_dynamic_indices


class EulerIntegrator(BaseIntegrator):
    """
    Явный метод Эйлера.

    new_state[i] = state[i] + dt * deriv[i]
    local_error  = tf.zeros([1])

    Поддерживает network-wide интегрирование (integration_mode='network'):
    step_network() делает один согласованный шаг Эйлера по объединённым
    плоским спискам состояний популяций и синапсов; слоты моделей без
    динамики (пустые derivatives) остаются неизменными.
    """

    supports_network_step = True

    def step(
        self,
        population: PopulationModel,
        state: StateList,
        total_synaptic_input: Dict[str, TensorType],
    ) -> Tuple[StateList, TensorType]:
        derivs = population.derivatives(state, total_synaptic_input)
        dt = population.dt

        new_state = [state[i] + dt * derivs[i] for i in range(len(state))]

        local_error = tf.zeros([1], dtype=neuraltide.config.get_dtype())

        return new_state, local_error

    def step_network(
        self,
        rhs,
        pop_states: StateList,
        syn_states: StateList,
        dt: float,
    ) -> Tuple[StateList, StateList]:
        """
        Один шаг Эйлера по связанной сети (integration_mode='network').

        Совпадает с совместным Эйлером split-режима, пока синапсы не имеют
        собственной динамики (split обновляет синапсы раньше популяций, но
        у статических синапсов состояние не меняется, а токи совпадают).

        Args:
            rhs: _JointNetworkRHS — rhs(pop_states, syn_states) ->
                (dpop_flat, dsyn_flat).
            pop_states: плоский список состояний популяций.
            syn_states: плоский список состояний синапсов.
            dt: шаг интегрирования в мс.

        Returns:
            (new_pop_states, new_syn_states) — полные плоские списки той
            же структуры; слоты без производных неизменны.
        """
        dpop, dsyn = rhs(pop_states, syn_states)
        if not dpop and not dsyn:
            return list(pop_states), list(syn_states)

        pop_idx, syn_idx = joint_dynamic_indices(rhs, len(dpop), len(dsyn))

        new_pop = list(pop_states)
        for j, i in enumerate(pop_idx):
            new_pop[i] = new_pop[i] + dt * dpop[j]

        new_syn = list(syn_states)
        for j, i in enumerate(syn_idx):
            new_syn[i] = new_syn[i] + dt * dsyn[j]

        return new_pop, new_syn

    def step_synapse(
        self,
        synapse: SynapseModel,
        state: StateList,
        pre_firing_rate: TensorType,
        post_voltage: TensorType,
        dt: float,
    ) -> Tuple[StateList, TensorType]:
        derivs = synapse.derivatives(state, pre_firing_rate, post_voltage)

        if len(derivs) == 0:
            return state, tf.zeros([1], dtype=neuraltide.config.get_dtype())

        new_state = [state[i] + dt * derivs[i] for i in range(len(state))]

        local_error = tf.zeros([1], dtype=neuraltide.config.get_dtype())

        return new_state, local_error
