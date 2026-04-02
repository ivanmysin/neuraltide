from typing import Dict, Tuple

import tensorflow as tf

import neuraltide.config
from neuraltide.core.base import PopulationModel
from neuraltide.core.types import TensorType, StateList
from neuraltide.integrators.base import BaseIntegrator


class RK4Integrator(BaseIntegrator):
    """
    Метод Рунге-Кутты 4-го порядка.

    k1 = derivatives(state)
    k2 = derivatives(state + dt/2 * k1)
    k3 = derivatives(state + dt/2 * k2)
    k4 = derivatives(state + dt   * k3)
    new_state[i]  = state[i] + dt/6 * (k1+2k2+2k3+k4)[i]
    heun_state[i] = state[i] + dt/2 * (k1+k2)[i]
    local_error   = mean(||new_state - heun_state||²)
    """

    def step(
        self,
        population: PopulationModel,
        state: StateList,
        total_synaptic_input: Dict[str, TensorType],
    ) -> Tuple[StateList, TensorType]:
        dt = population.dt

        k1 = population.derivatives(state, total_synaptic_input)

        state_plus_k1_half = [
            state[i] + dt * 0.5 * k1[i]
            for i in range(len(state))
        ]
        k2 = population.derivatives(state_plus_k1_half, total_synaptic_input)

        state_plus_k2_half = [
            state[i] + dt * 0.5 * k2[i]
            for i in range(len(state))
        ]
        k3 = population.derivatives(state_plus_k2_half, total_synaptic_input)

        state_plus_k3 = [
            state[i] + dt * k3[i]
            for i in range(len(state))
        ]
        k4 = population.derivatives(state_plus_k3, total_synaptic_input)

        new_state = [
            state[i] + dt * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) / 6.0
            for i in range(len(state))
        ]

        heun_state = [
            state[i] + dt * 0.5 * (k1[i] + k2[i])
            for i in range(len(state))
        ]

        errors = [
            tf.reduce_sum((new_state[i] - heun_state[i]) ** 2)
            for i in range(len(state))
        ]
        local_error = tf.reduce_mean(errors)[tf.newaxis]

        return new_state, local_error
