from abc import ABC, abstractmethod
from typing import Dict, Tuple

import tensorflow as tf

from neuraltide.core.base import PopulationModel
from neuraltide.core.types import TensorType, StateList


class BaseIntegrator(ABC):
    """
    Базовый класс интегратора ОДУ.

    Интегратор отдельно от популяционной модели — пользователь
    может менять схему интегрирования не изменяя модель.
    """

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
