import tensorflow as tf

import neuraltide
import neuraltide.config
from neuraltide.core.base import BaseInputGenerator
from neuraltide.core.types import TensorType


class ConstantRateGenerator(BaseInputGenerator):
    """
    Генератор постоянной частоты.

    rate(t) = constant_rate (независимо от t)
    """

    def __init__(self, constant_rate: float, name: str = "constant_rate_generator", **kwargs):
        super().__init__(n_outputs=1, name=name, **kwargs)
        self.constant_rate = tf.constant(constant_rate, dtype=neuraltide.config.get_dtype())

    def call(self, t: TensorType) -> TensorType:
        return tf.ones_like(t) * self.constant_rate
