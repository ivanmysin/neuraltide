import tensorflow as tf
from typing import Any, Dict

import neuraltide.config
from neuraltide.core.base import BaseInputGenerator
from neuraltide.core.types import TensorType


class SinusoidalGenerator(BaseInputGenerator):
    """
    Синусоидальный генератор.

    rate(t) = max(0, amplitude * sin(2π*freq*t/1000 + phase) + offset)

    Параметры (params):
        amplitude: амплитуда (Hz). Скаляр или вектор [n_units].
        freq: частота (Hz). Скаляр или вектор [n_units].
        phase: начальная фаза (рад). Скаляр или вектор [n_units].
        offset: смещение (Hz). Скаляр или вектор [n_units].

    Пример:
        gen = SinusoidalGenerator(
            dt=0.5,
            params={
                'amplitude': 10.0,
                'freq': 8.0,
                'phase': 0.0,
                'offset': 5.0,
            }
        )
        # n_units=1

        gen = SinusoidalGenerator(
            dt=0.5,
            params={
                'amplitude': [10.0, 15.0],
                'freq': [8.0, 10.0],
                'phase': [0.0, 1.5],
                'offset': [5.0, 7.0],
            }
        )
        # n_units=2
    """

    def __init__(
        self,
        dt: float,
        params: Dict[str, Any],
        name: str = "sinusoidal_generator",
        **kwargs
    ):
        super().__init__(params=params, dt=dt, name=name, **kwargs)

        self.amplitude = self._make_param(self._params, 'amplitude')
        self.freq = self._make_param(self._params, 'freq')
        self.phase = self._make_param(self._params, 'phase')
        self.offset = self._make_param(self._params, 'offset')

    def call(self, t: TensorType) -> TensorType:
        """
        Args:
            t: текущее время в мс. shape = [batch, 1].

        Returns:
            tf.Tensor, shape = [batch, n_units], в Гц.
        """
        two_pi = tf.constant(2.0 * 3.141592653589793, dtype=neuraltide.config.get_dtype())

        t_expanded = t
        amplitude = tf.reshape(self.amplitude, [1, self.n_units])
        freq = tf.reshape(self.freq, [1, self.n_units])
        phase = tf.reshape(self.phase, [1, self.n_units])
        offset = tf.reshape(self.offset, [1, self.n_units])

        rate = (amplitude * tf.sin(two_pi * freq * t_expanded / 1000.0 + phase)
                + offset)
        return tf.nn.relu(rate)

    @property
    def parameter_spec(self) -> Dict[str, Dict[str, Any]]:
        return {
            'amplitude': {
                'shape': (self.n_units,),
                'trainable': self.amplitude.trainable,
                'constraint': self._get_constraint_name(self.amplitude),
                'units': 'Hz',
            },
            'freq': {
                'shape': (self.n_units,),
                'trainable': self.freq.trainable,
                'constraint': self._get_constraint_name(self.freq),
                'units': 'Hz',
            },
            'phase': {
                'shape': (self.n_units,),
                'trainable': self.phase.trainable,
                'constraint': self._get_constraint_name(self.phase),
                'units': 'rad',
            },
            'offset': {
                'shape': (self.n_units,),
                'trainable': self.offset.trainable,
                'constraint': self._get_constraint_name(self.offset),
                'units': 'Hz',
            },
        }

    def _get_constraint_name(self, var: tf.Variable) -> str:
        if var.constraint is not None:
            return var.constraint.__class__.__name__
        return None
