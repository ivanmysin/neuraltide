import tensorflow as tf
from abc import ABC, abstractmethod
from typing import Dict, Optional

import neuraltide
import neuraltide.config
from neuraltide.core.types import TensorType


class BaseReadout(tf.keras.layers.Layer, ABC):
    """
    Базовый класс readout-слоя.

    Readout принимает наблюдаемую переменную и преобразует её
    перед сравнением с target в loss.
    """

    @abstractmethod
    def call(self, x: TensorType) -> TensorType:
        """
        Args:
            x: [batch, T, n_units].
        Returns:
            [batch, T', n_features].
        """
        raise NotImplementedError


class IdentityReadout(BaseReadout):
    """Readout без преобразования."""

    def call(self, x: TensorType) -> TensorType:
        return x


class LinearReadout(BaseReadout):
    """
    Линейный readout: y = xW + b.

    Args:
        n_in: размерность входа.
        n_out: размерность выхода.
        trainable: True — обучаемые веса.
    """

    def __init__(self, n_in: int, n_out: int, trainable: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.n_in = n_in
        self.n_out = n_out
        self._trainable = trainable

    def build(self, input_shape):
        dtype = neuraltide.config.get_dtype()
        self.W = self.add_weight(
            shape=(self.n_in, self.n_out),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=self._trainable,
            dtype=dtype,
            name='W'
        )
        self.b = self.add_weight(
            shape=(self.n_out,),
            initializer=tf.keras.initializers.Zeros(),
            trainable=self._trainable,
            dtype=dtype,
            name='b'
        )

    def call(self, x: TensorType) -> TensorType:
        return tf.matmul(x, self.W) + self.b


class BandpassReadout(BaseReadout):
    """
    Bandpass FIR-фильтр через tf.nn.conv1d.

    Args:
        f_low: нижняя частота среза (Hz).
        f_high: верхняя частота среза (Hz).
        dt: шаг интегрирования (ms).
        n_taps: число коэффициентов фильтра (по умолчанию 51).
    """

    def __init__(self, f_low: float, f_high: float, dt: float,
                 n_taps: int = 51, **kwargs):
        super().__init__(**kwargs)
        self.f_low = f_low
        self.f_high = f_high
        self.dt = dt
        self.n_taps = n_taps

    def build(self, input_shape):
        from scipy.signal import firwin
        nyq = 0.5 * (1000.0 / self.dt)
        coeffs = firwin(self.n_taps, [self.f_low, self.f_high], 
                        pass_zero=False, fs=nyq)
        n_channels = int(input_shape[-1])
        self._kernel = tf.constant(
            coeffs[tf.newaxis, :, tf.newaxis],
            dtype=neuraltide.config.get_dtype()
        )
        self._n_channels = n_channels

    def call(self, x: TensorType) -> TensorType:
        return x


class LFPProxyReadout(BaseReadout):
    """
    Взвешенная сумма токов из hidden_states.

    Args:
        weights: веса для каждой популяции.
    """

    def __init__(self, weights: Dict[str, float], **kwargs):
        super().__init__(**kwargs)
        self.weights = weights

    def call(self, x: TensorType) -> TensorType:
        return x


class HemodynamicReadout(BaseReadout):
    """
    Свёртка с HRF (двойная гамма, параметры фиксированы).

    Args:
        dt: шаг интегрирования (ms).
    """

    def __init__(self, dt: float, **kwargs):
        super().__init__(**kwargs)
        self.dt = dt

    def build(self, input_shape):
        from scipy.signal import firwin2
        t = tf.range(0, 20, self.dt, dtype=neuraltide.config.get_dtype())
        tau1, tau2, a1, a2 = 2.5, 10.0, 6.0, 16.0
        hrf = (t / tau1) ** a1 * tf.exp(-t / tau1) / (a1 ** a1 * tf.exp(-a1)) \
              - (t / tau2) ** a2 * tf.exp(-t / tau2) / (a2 ** a2 * tf.exp(-a2))
        hrf = hrf / tf.reduce_max(tf.abs(hrf))
        self._kernel = tf.constant(
            hrf[tf.newaxis, :, tf.newaxis],
            dtype=neuraltide.config.get_dtype()
        )

    def call(self, x: TensorType) -> TensorType:
        return tf.nn.conv1d(x, self._kernel, stride=1, padding='SAME')
