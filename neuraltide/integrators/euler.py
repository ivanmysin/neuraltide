from typing import Dict, Tuple

import tensorflow as tf

import neuraltide.config
from neuraltide.core.base import PopulationModel
from neuraltide.core.types import TensorType, StateList
from neuraltide.integrators.base import BaseIntegrator


class EulerIntegrator(BaseIntegrator):
    """
    Явный метод Эйлера.

    new_state[i] = state[i] + dt * deriv[i]
    local_error  = tf.zeros([1])
    """

    def step(
        self,
        population: PopulationModel,
        state: StateList,
        total_synaptic_input: Dict[str, TensorType],
    ) -> Tuple[StateList, TensorType]:
        derivs = population.derivatives(state, total_synaptic_input)
        dt = population.dt

        new_state = [
            state[i] + dt * derivs[i]
            for i in range(len(state))
        ]

        local_error = tf.zeros([1], dtype=neuraltide.config.get_dtype())

        return new_state, local_error
