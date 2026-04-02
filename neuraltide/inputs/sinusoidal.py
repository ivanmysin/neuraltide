import tensorflow as tf

import neuraltide
import neuraltide.config
from neuraltide.core.base import BaseInputGenerator
from neuraltide.core.types import TensorType


class SinusoidalGenerator(BaseInputGenerator):
    """
    Синусоидальный генератор.

    rate(t) = max(0, amplitude * sin(2π*freq*t/1000 + phase) + offset)
    """

    def __init__(self, amplitude: float, freq: float, phase: float = 0.0,
                 offset: float = 0.0, name: str = "sinusoidal_generator", **kwargs):
        super().__init__(n_outputs=1, name=name, **kwargs)
        self.amplitude = tf.constant(amplitude, dtype=neuraltide.config.get_dtype())
        self.freq = tf.constant(freq, dtype=neuraltide.config.get_dtype())
        self.phase = tf.constant(phase, dtype=neuraltide.config.get_dtype())
        self.offset = tf.constant(offset, dtype=neuraltide.config.get_dtype())

    def call(self, t: TensorType) -> TensorType:
        two_pi = tf.constant(2.0 * 3.141592653589793, dtype=neuraltide.config.get_dtype())
        rate = (self.amplitude * tf.sin(two_pi * self.freq * t / 1000.0 + self.phase)
                + self.offset)
        return tf.nn.relu(rate)
