import tensorflow as tf
from typing import Any, Dict

import neuraltide.config
from neuraltide.core.base import BaseInputGenerator
from neuraltide.core.types import TensorType


class ConstantRateGenerator(BaseInputGenerator):
    """
    Генератор постоянной частоты.

    rate(t) = constant_rate (независимо от t)

    Параметры (params):
        rate: постоянная частота (Hz). Скаляр или вектор [n_units].

    Пример:
        gen = ConstantRateGenerator(
            dt=0.5,
            params={
                'rate': 10.0,
            }
        )
        # n_units=1

        gen = ConstantRateGenerator(
            dt=0.5,
            params={
                'rate': [10.0, 15.0, 20.0],
            }
        )
        # n_units=3
    """

    def __init__(
        self,
        dt: float,
        params: Dict[str, Any],
        name: str = "constant_rate_generator",
        **kwargs
    ):
        super().__init__(params=params, dt=dt, name=name, **kwargs)

        self.rate = self._make_param(self._params, 'rate')

    def call(self, t: TensorType) -> TensorType:
        """
        Args:
            t: текущее время в мс. shape = [batch, 1].

        Returns:
            tf.Tensor, shape = [batch, n_units], в Гц.
        """
        rate = tf.reshape(self.rate, [1, self.n_units])
        batch_size = tf.shape(t)[0]
        return tf.broadcast_to(rate, [batch_size, self.n_units])

    @property
    def parameter_spec(self) -> Dict[str, Dict[str, Any]]:
        return {
            'rate': {
                'shape': (self.n_units,),
                'trainable': self.rate.trainable,
                'constraint': self._get_constraint_name(self.rate),
                'units': 'Hz',
            },
        }

    def _get_constraint_name(self, var: tf.Variable) -> str:
        if var.constraint is not None:
            return var.constraint.__class__.__name__
        return None
