import tensorflow as tf
from abc import ABC, abstractmethod

import neuraltide
import neuraltide.config
from neuraltide.core.types import TensorType, cast_if_needed


class BaseReadout(tf.keras.layers.Layer, ABC):
    """
    Базовый класс readout-слоя.

    Readout принимает наблюдаемую переменную и преобразует её
    перед сравнением с target в loss.

    autocast отключён: слой работает в глобальном neuraltide.config.get_dtype()
    (аналогично BaseInputGenerator), чтобы float64-траектории не приводились
    Keras к float32.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.autocast = False

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
            name="W",
        )
        self.b = self.add_weight(
            shape=(self.n_out,),
            initializer=tf.keras.initializers.Zeros(),
            trainable=self._trainable,
            dtype=dtype,
            name="b",
        )

    def call(self, x: TensorType) -> TensorType:
        return tf.matmul(x, self.W) + self.b


def _depthwise_time_conv(x: TensorType, kernel: TensorType) -> TensorType:
    """
    Свёртка по времени ядром [width, 1, 1] без смешивания каналов.

    [b, T, C] → [b*C, T, 1] → conv1d(SAME) → [b, T, C].
    """
    b = tf.shape(x)[0]
    n = tf.shape(x)[1]
    c = tf.shape(x)[2]
    x_t = tf.transpose(x, [0, 2, 1])  # [b, C, T]
    x_t = tf.reshape(x_t, [b * c, n, 1])  # [b*C, T, 1]
    y = tf.nn.conv1d(x_t, kernel, stride=1, padding="SAME")
    y = tf.reshape(y, [b, c, n])
    return tf.transpose(y, [0, 2, 1])


def _depthwise_causal_conv(x: TensorType, kernel: TensorType) -> TensorType:
    """
    Каузальная свёртка по времени (HRF): y[t] = Σ_j kernel[j]·x[t−j].

    tf.nn.conv1d вычисляет cross-correlation, поэтому вход дополняется
    K−1 нулями слева и сворачивается с развёрнутым ядром (padding=VALID).
    Каналы не смешиваются.
    """
    b = tf.shape(x)[0]
    n = tf.shape(x)[1]
    c = tf.shape(x)[2]
    k_len = tf.shape(kernel)[0]
    x_t = tf.transpose(x, [0, 2, 1])  # [b, C, T]
    x_t = tf.reshape(x_t, [b * c, n, 1])  # [b*C, T, 1]
    x_t = tf.pad(x_t, [[0, 0], [k_len - 1, 0], [0, 0]])
    y = tf.nn.conv1d(x_t, tf.reverse(kernel, axis=[0]), stride=1, padding="VALID")
    y = tf.reshape(y, [b, c, n])
    return tf.transpose(y, [0, 2, 1])


class BandpassReadout(BaseReadout):
    """
    Bandpass FIR-фильтр через tf.nn.conv1d.

    Каналы фильтруются независимо (без смешивания). Частоты задаются в Гц,
    шаг dt — в мс.

    Args:
        f_low: нижняя частота среза (Hz).
        f_high: верхняя частота среза (Hz).
        dt: шаг интегрирования (ms).
        n_taps: число коэффициентов фильтра (по умолчанию 51).
    """

    def __init__(
        self, f_low: float, f_high: float, dt: float, n_taps: int = 51, **kwargs
    ):
        super().__init__(**kwargs)
        self.f_low = f_low
        self.f_high = f_high
        self.dt = dt
        self.n_taps = n_taps

    def build(self, input_shape):
        from scipy.signal import firwin

        fs = 1000.0 / self.dt  # dt в мс → частота дискретизации в Гц
        coeffs = firwin(self.n_taps, [self.f_low, self.f_high], pass_zero=False, fs=fs)
        self._kernel = tf.constant(
            coeffs[:, None, None], dtype=neuraltide.config.get_dtype()
        )

    def call(self, x: TensorType) -> TensorType:
        kernel = cast_if_needed(self._kernel, x.dtype)
        return _depthwise_time_conv(x, kernel)


class HemodynamicReadout(BaseReadout):
    """
    Каузальная свёртка с HRF (двойная гамма, параметры фиксированы).

    Каналы фильтруются независимо (без смешивания).

    Args:
        dt: шаг интегрирования (ms).
    """

    def __init__(self, dt: float, **kwargs):
        super().__init__(**kwargs)
        self.dt = dt

    def build(self, input_shape):
        dtype = neuraltide.config.get_dtype()
        t = tf.range(0, 20, self.dt, dtype=dtype)
        tau1, tau2 = tf.constant(2.5, dtype=dtype), tf.constant(10.0, dtype=dtype)
        a1, a2 = tf.constant(6.0, dtype=dtype), tf.constant(16.0, dtype=dtype)
        hrf = (t / tau1) ** a1 * tf.exp(-t / tau1) / (a1**a1 * tf.exp(-a1)) - (
            t / tau2
        ) ** a2 * tf.exp(-t / tau2) / (a2**a2 * tf.exp(-a2))
        hrf = hrf / tf.reduce_max(tf.abs(hrf))
        self._kernel = tf.reshape(hrf, [-1, 1, 1])

    def call(self, x: TensorType) -> TensorType:
        kernel = cast_if_needed(self._kernel, x.dtype)
        return _depthwise_causal_conv(x, kernel)
