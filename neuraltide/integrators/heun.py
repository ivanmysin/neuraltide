from typing import Dict, Tuple

import tensorflow as tf

import neuraltide.config
from neuraltide.core.base import PopulationModel, SynapseModel
from neuraltide.core.types import TensorType, StateList
from neuraltide.integrators.base import BaseIntegrator


class HeunIntegrator(BaseIntegrator):
    """
    Метод Хьюна (предсказывающий-корректирующий).

    k1 = derivatives(state)
    k2 = derivatives(state + dt * k1)
    new_state[i]  = state[i] + dt/2 * (k1[i] + k2[i])
    local_error   = mean(||new_state - (state + dt*k1)||²)
    """

    def step(
        self,
        population: PopulationModel,
        state: StateList,
        total_synaptic_input: Dict[str, TensorType],
    ) -> Tuple[StateList, TensorType]:
        dt = population.dt
        dtype = neuraltide.config.get_dtype()

        k1 = population.derivatives(state, total_synaptic_input)

        state_plus_k1 = [
            state[i] + dt * k1[i]
            for i in range(len(state))
        ]
        k2 = population.derivatives(state_plus_k1, total_synaptic_input)

        new_state = [
            state[i] + dt * 0.5 * (k1[i] + k2[i])
            for i in range(len(state))
        ]

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

        state_plus_k1 = [
            state[i] + dt * k1[i]
            for i in range(len(k1))
        ]
        k2 = synapse.derivatives(state_plus_k1, pre_firing_rate, post_voltage)

        new_state = [
            state[i] + dt * 0.5 * (k1[i] + k2[i])
            for i in range(len(k1))
        ]

        errors = [
            tf.reduce_sum((new_state[i] - state_plus_k1[i]) ** 2)
            for i in range(len(k1))
        ]
        local_error = tf.reduce_mean(errors)[tf.newaxis]

        return new_state, local_error
