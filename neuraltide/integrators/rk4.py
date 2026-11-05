from typing import Dict, Tuple

import tensorflow as tf

import neuraltide.config
from neuraltide.core.base import PopulationModel, SynapseModel
from neuraltide.core.types import TensorType, StateList
from neuraltide.integrators.base import BaseIntegrator
from neuraltide.integrators.stiffness import joint_dynamic_indices


class RK4Integrator(BaseIntegrator):
    """
    Метод Рунге-Кутты 4-го порядка.

    k1 = derivatives(state)
    k2 = derivatives(state + dt/2 * k1)
    k3 = derivatives(state + dt/2 * k2)
    k4 = derivatives(state + dt   * k3)
    new_state[i] = state[i] + dt/6 * (k1+2k2+2k3+k4)[i]
    heun_state[i]= state[i] + dt/2 * (k1+k2)[i]   (только если compute_error=True)
    local_error  = mean(||new_state - heun_state||²)

    Поддерживает network-wide интегрирование (integration_mode='network'):
    4 стадии по объединённым плоским спискам состояний популяций и
    синапсов через rhs — согласованный 4-й порядок по всей связанной сети
    (split-режим с динамическими синапсами падает до ~2). Слоты моделей
    без динамики неизменны.
    """

    supports_network_step = True

    def __init__(self, compute_error: bool = True):
        self.compute_error = compute_error

    def step(
        self,
        population: PopulationModel,
        state: StateList,
        total_synaptic_input: Dict[str, TensorType],
    ) -> Tuple[StateList, TensorType]:
        dt = population.dt

        k1 = population.derivatives(state, total_synaptic_input)

        state_plus_k1_half = [state[i] + dt * 0.5 * k1[i] for i in range(len(state))]
        k2 = population.derivatives(state_plus_k1_half, total_synaptic_input)

        state_plus_k2_half = [state[i] + dt * 0.5 * k2[i] for i in range(len(state))]
        k3 = population.derivatives(state_plus_k2_half, total_synaptic_input)

        state_plus_k3 = [state[i] + dt * k3[i] for i in range(len(state))]
        k4 = population.derivatives(state_plus_k3, total_synaptic_input)

        new_state = [
            state[i] + dt * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6.0
            for i in range(len(state))
        ]

        if self.compute_error:
            heun_state = [
                state[i] + dt * 0.5 * (k1[i] + k2[i]) for i in range(len(state))
            ]
            errors = [
                tf.reduce_sum((new_state[i] - heun_state[i]) ** 2)
                for i in range(len(state))
            ]
            local_error = tf.reduce_mean(errors)[tf.newaxis]
        else:
            local_error = tf.zeros([1], dtype=neuraltide.config.get_dtype())

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

        state_plus_k1_half = [state[i] + dt * 0.5 * k1[i] for i in range(len(k1))]
        k2 = synapse.derivatives(state_plus_k1_half, pre_firing_rate, post_voltage)

        state_plus_k2_half = [state[i] + dt * 0.5 * k2[i] for i in range(len(k1))]
        k3 = synapse.derivatives(state_plus_k2_half, pre_firing_rate, post_voltage)

        state_plus_k3 = [state[i] + dt * k3[i] for i in range(len(k1))]
        k4 = synapse.derivatives(state_plus_k3, pre_firing_rate, post_voltage)

        new_state = [
            state[i] + dt * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6.0
            for i in range(len(k1))
        ]

        if self.compute_error:
            heun_state = [state[i] + dt * 0.5 * (k1[i] + k2[i]) for i in range(len(k1))]
            errors = [
                tf.reduce_sum((new_state[i] - heun_state[i]) ** 2)
                for i in range(len(k1))
            ]
            local_error = tf.reduce_mean(errors)[tf.newaxis]
        else:
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
        Один шаг RK4 по связанной сети (integration_mode='network').

        Каждая из 4 стадий вызывает rhs(pop_states, syn_states) на
        промежуточном объединённом состоянии; обновляются и список
        популяций, и список синапсов. Токи и частоты внутри шага
        согласованы (все стадии — по состояниям на начало субшага),
        поэтому порядок 4 сохраняется и при динамических синапсах.

        Args:
            rhs: _JointNetworkRHS — rhs(pop_states, syn_states) ->
                (dpop_flat, dsyn_flat).
            pop_states: плоский список состояний популяций.
            syn_states: плоский список состояний синапсов.
            dt: шаг интегрирования в мс.

        Returns:
            (new_pop_states, new_syn_states) — полные плоские списки той
            же структуры; слоты без производных неизменны. Оценка локальной
            ошибки (Heun-пара) в network-режиме не возвращается.
        """
        k1p, k1s = rhs(pop_states, syn_states)
        if not k1p and not k1s:
            return list(pop_states), list(syn_states)

        pop_idx, syn_idx = joint_dynamic_indices(rhs, len(k1p), len(k1s))

        def stage(kp: StateList, ks: StateList) -> Tuple[StateList, StateList]:
            """Промежуточное состояние: y + (приращение) по динамическим слотам."""
            stage_pop = list(pop_states)
            stage_syn = list(syn_states)
            for j, i in enumerate(pop_idx):
                stage_pop[i] = stage_pop[i] + kp[j]
            for j, i in enumerate(syn_idx):
                stage_syn[i] = stage_syn[i] + ks[j]
            return stage_pop, stage_syn

        half_dt = 0.5 * dt
        p2, s2 = stage([half_dt * k for k in k1p], [half_dt * k for k in k1s])
        k2p, k2s = rhs(p2, s2)
        p3, s3 = stage([half_dt * k for k in k2p], [half_dt * k for k in k2s])
        k3p, k3s = rhs(p3, s3)
        p4, s4 = stage([dt * k for k in k3p], [dt * k for k in k3s])
        k4p, k4s = rhs(p4, s4)

        sixth_dt = dt / 6.0
        new_pop = list(pop_states)
        new_syn = list(syn_states)
        for j, i in enumerate(pop_idx):
            new_pop[i] = new_pop[i] + sixth_dt * (
                k1p[j] + 2.0 * k2p[j] + 2.0 * k3p[j] + k4p[j]
            )
        for j, i in enumerate(syn_idx):
            new_syn[i] = new_syn[i] + sixth_dt * (
                k1s[j] + 2.0 * k2s[j] + 2.0 * k3s[j] + k4s[j]
            )

        return new_pop, new_syn
