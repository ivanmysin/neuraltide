import tensorflow as tf

import neuraltide
import neuraltide.config
from neuraltide.core.base import PopulationModel, BaseInputGenerator
from neuraltide.core.types import TensorType, StateList, Dict


class InputPopulation(PopulationModel):
    """
    Псевдо-популяция без динамики. Оборачивает BaseInputGenerator.

    Назначение:
        Позволяет подключать внешние входные сигналы к сети через
        полноценные обучаемые синапсы, единообразно с рекуррентными
        проекциями между динамическими популяциями.

    Семантика:
        - Не имеет уравнений динамики (derivatives() возвращает []).
        - Не обновляется интегратором.
        - Состояние: [t_current], shape [1, 1] — текущее время в мс.
        - NetworkRNNCell обновляет состояние напрямую: new_state = [t].
        - get_firing_rate(state) вызывает generator(state[0]).
        - Не может быть целью синапса (только источником).
    """

    def __init__(self, generator: BaseInputGenerator, **kwargs):
        super().__init__(
            n_units=generator.n_units,
            dt=generator.dt,
            **kwargs
        )
        self.generator = generator
        self.state_size = [tf.TensorShape([1, 1])]

    def get_initial_state(self, batch_size: int = 1) -> StateList:
        return [tf.zeros([1, 1], dtype=neuraltide.config.get_dtype())]

    def derivatives(self, state, total_synaptic_input) -> list:
        return []

    def get_firing_rate(self, state: StateList) -> TensorType:
        t = state[0]
        rate = self.generator(t)
        return tf.reshape(rate, [1, self.n_units])

    def observables(self, state: StateList) -> Dict[str, TensorType]:
        return {'firing_rate': self.get_firing_rate(state)}

    @property
    def parameter_spec(self) -> Dict[str, Dict[str, any]]:
        return {}
