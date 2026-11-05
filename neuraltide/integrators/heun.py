from typing import Dict, Tuple

import tensorflow as tf

import neuraltide.config
from neuraltide.core.base import PopulationModel, SynapseModel
from neuraltide.core.types import TensorType, StateList
from neuraltide.integrators.base import BaseIntegrator
from neuraltide.integrators.stiffness import joint_dynamic_indices


class HeunIntegrator(BaseIntegrator):
    """
    Метод Хьюна (предсказывающий-корректирующий, RK2).

    k1 = derivatives(state)
    k2 = derivatives(state + dt * k1)
    new_state[i]  = state[i] + dt/2 * (k1[i] + k2[i])
    local_error   = mean(||new_state - (state + dt*k1)||²)

    Поддерживает network-wide интегрирование (integration_mode='network'):
    стадии строятся по объединённым плоским спискам состояний популяций
    и синапсов через rhs; слоты моделей без динамики неизменны.
    """

    supports_network_step = True

    def step(
        self,
        population: PopulationModel,
        state: StateList,
        total_synaptic_input: Dict[str, TensorType],
    ) -> Tuple[StateList, TensorType]:
        dt = population.dt

        k1 = population.derivatives(state, total_synaptic_input)

        state_plus_k1 = [state[i] + dt * k1[i] for i in range(len(state))]
        k2 = population.derivatives(state_plus_k1, total_synaptic_input)

        new_state = [state[i] + dt * 0.5 * (k1[i] + k2[i]) for i in range(len(state))]

        errors = [
            tf.reduce_sum((new_state[i] - state_plus_k1[i]) ** 2)
            for i in range(len(state))
        ]
        local_error = tf.reduce_mean(errors)[tf.newaxis]

        return new_state, local_error

    def step_synapse(
        self,
        synapse: SynapseModel,
        state: StateList,
        pre_firing_rate: TensorType,
        post_voltage: TensorType,
        dt: float,
    ) -> Tuple[StateList, TensorType]:
        dtype = neuraltide.config.get_dtype()

        k1 = synapse.derivatives(state, pre_firing_rate, post_voltage)

        if len(k1) == 0:
            return state, tf.zeros([1], dtype=dtype)

        state_plus_k1 = [state[i] + dt * k1[i] for i in range(len(k1))]
        k2 = synapse.derivatives(state_plus_k1, pre_firing_rate, post_voltage)

        new_state = [state[i] + dt * 0.5 * (k1[i] + k2[i]) for i in range(len(k1))]

        errors = [
            tf.reduce_sum((new_state[i] - state_plus_k1[i]) ** 2)
            for i in range(len(k1))
        ]
        local_error = tf.reduce_mean(errors)[tf.newaxis]

        return new_state, local_error

    def step_network(
        self,
        rhs,
        pop_states: StateList,
        syn_states: StateList,
        dt: float,
    ) -> Tuple[StateList, StateList]:
        """
        Один шаг Хьюна (RK2) по связанной сети.

        Предиктор — совместный шаг Эйлера по объединённым состояниям,
        корректор — среднее производных на начало шага и в предсказанной
        точке. Даёт согласованный 2-й порядок по всей связанной сети
        (включая динамику синапсов), в отличие от split-режима.

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
        k1p, k1s = rhs(pop_states, syn_states)
        if not k1p and not k1s:
            return list(pop_states), list(syn_states)

        pop_idx, syn_idx = joint_dynamic_indices(rhs, len(k1p), len(k1s))

        pred_pop = list(pop_states)
        pred_syn = list(syn_states)
        for j, i in enumerate(pop_idx):
            pred_pop[i] = pred_pop[i] + dt * k1p[j]
        for j, i in enumerate(syn_idx):
            pred_syn[i] = pred_syn[i] + dt * k1s[j]

        k2p, k2s = rhs(pred_pop, pred_syn)

        half_dt = 0.5 * dt
        new_pop = list(pop_states)
        new_syn = list(syn_states)
        for j, i in enumerate(pop_idx):
            new_pop[i] = new_pop[i] + half_dt * (k1p[j] + k2p[j])
        for j, i in enumerate(syn_idx):
            new_syn[i] = new_syn[i] + half_dt * (k1s[j] + k2s[j])

        return new_pop, new_syn
