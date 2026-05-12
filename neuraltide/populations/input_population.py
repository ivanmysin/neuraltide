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
        - Состояние: [t_current, extra_inputs], shape [1, 1] и [1, n_cols].
        - NetworkRNN обновляет состояние напрямую: new_state = [t, extra].
        - get_firing_rate(state) вызывает generator(state[0], state[1]).
        - Не может быть целью синапса (только источником).
    """

    def __init__(self, generator: BaseInputGenerator, **kwargs):
        super().__init__(
            n_units=generator.n_units,
            dt=generator.dt,
            **kwargs
        )
        self.generator = generator
        self._has_extra = True
        self.state_size = [
            tf.TensorShape([1, 1]),
            tf.TensorShape([1, None]),  # extra inputs, variable cols
        ]

    def get_initial_state(self, batch_size: int = 1) -> StateList:
        dtype = neuraltide.config.get_dtype()
        return [
            tf.zeros([1, 1], dtype=dtype),
            tf.zeros([1, 0], dtype=dtype),
        ]

    def derivatives(self, state, total_synaptic_input) -> list:
        return []

    def get_firing_rate(self, state: StateList) -> TensorType:
        t = state[0]
        extra = state[1] if len(state) > 1 else None
        rate = self.generator(t, extra_inputs=extra)
        return tf.reshape(rate, [1, self.n_units])

    def observables(self, state: StateList) -> Dict[str, TensorType]:
        return {'firing_rate': self.get_firing_rate(state)}

    @property
    def parameter_spec(self) -> Dict[str, Dict[str, any]]:
        return {}
