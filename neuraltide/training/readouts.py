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


def _fft_causal_conv(x: TensorType, kernel: TensorType) -> TensorType:
    """
    Каузальная свёртка y[t] = Σ_j kernel[j]·x[t−j] через FFT.

    То же, что _depthwise_causal_conv, но за O((T+K) log(T+K)) — для
    длинных ядер (HRF длиной десятки секунд при шаге в доли мс). Каналы
    не смешиваются.
    """
    n = tf.shape(x)[1]
    k_len = tf.shape(kernel)[0]
    length = n + k_len - 1
    x_t = tf.transpose(x, [0, 2, 1])  # [b, C, T]
    h = tf.reshape(kernel[:, 0, 0], [1, 1, -1])
    fft_len = tf.reshape(length, [1])
    y = tf.signal.irfft(
        tf.signal.rfft(x_t, fft_length=fft_len) * tf.signal.rfft(h, fft_length=fft_len),
        fft_length=fft_len,
    )
    y = tf.cast(y[:, :, :n], x.dtype)
    return tf.transpose(y, [0, 2, 1])


class BandpassReadout(BaseReadout):
    """
    Bandpass FIR-фильтр через tf.nn.conv1d.

    Каналы фильтруются независимо (без смешивания). Частоты задаются в Гц,
    шаг dt — в мс.

    build() вызывает scipy.signal.firwin (Python/NumPy), поэтому слой
    обязан быть построен в eager-режиме до попадания внутрь трассируемого
    графа: __call__ форсирует build() при первом eager-вызове, а попытка
    трассировки непостроенного слоя даёт понятную ошибку.

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

    def __call__(self, x: TensorType, *args, **kwargs) -> TensorType:
        # scipy-вызов в build() не трейсуем: построение только в eager.
        if not self.built:
            if not tf.executing_eagerly():
                raise RuntimeError(
                    "BandpassReadout попал внутрь трассируемого графа до "
                    "построения (build() не вызывался). build() использует "
                    "scipy.signal.firwin и должен выполниться в eager-режиме: "
                    "вызовите слой один раз снаружи tf.function (например, при "
                    "подготовке target/predict) до компиляции обучения."
                )
            self.build(x.shape)
        return super().__call__(x, *args, **kwargs)

    def build(self, input_shape):
        from scipy.signal import firwin

        fs = 1000.0 / self.dt  # dt в мс → частота дискретизации в Гц
        coeffs = firwin(self.n_taps, [self.f_low, self.f_high], pass_zero=False, fs=fs)
        self._kernel = tf.constant(
            coeffs[:, None, None], dtype=neuraltide.config.get_dtype()
        )
        super().build(input_shape)

    def call(self, x: TensorType) -> TensorType:
        kernel = cast_if_needed(self._kernel, x.dtype)
        return _depthwise_time_conv(x, kernel)


class HemodynamicReadout(BaseReadout):
    """
    Каузальная свёртка с канонической HRF (двойная гамма SPM, Glover 1999).

        h(t) = Γ(t; 6) − Γ(t; 16)/6,   t — в секундах,

    где Γ(t; a) — плотность гамма-распределения с формой a и масштабом 1 с:
    пик отклика около 5 с, минимум постстимульного провала около 15 с.
    Ядро длиной duration_s (по умолчанию 32 с) строится на сетке шага dt
    (мс) и нормируется на max|h| = 1. Свёртка выполняется через FFT:
    при dt порядка 0.1–1 мс ядро содержит десятки тысяч отсчётов.

    Каналы фильтруются независимо (без смешивания).

    Args:
        dt: шаг интегрирования (ms).
        duration_s: длина ядра в секундах (по умолчанию 32).
    """

    def __init__(self, dt: float, duration_s: float = 32.0, **kwargs):
        super().__init__(**kwargs)
        if dt <= 0.0:
            raise ValueError(f"HemodynamicReadout: dt must be positive, got {dt}.")
        if duration_s <= 0.0:
            raise ValueError(
                f"HemodynamicReadout: duration_s must be positive, got {duration_s}."
            )
        self.dt = dt
        self.duration_s = duration_s

    @staticmethod
    def _gamma_pdf(t_s: tf.Tensor, shape: float) -> tf.Tensor:
        a = tf.constant(shape, dtype=t_s.dtype)
        safe_t = tf.where(t_s > 0.0, t_s, tf.ones_like(t_s))
        log_pdf = (a - 1.0) * tf.math.log(safe_t) - safe_t - tf.math.lgamma(a)
        return tf.where(t_s > 0.0, tf.exp(log_pdf), tf.zeros_like(t_s))

    def build(self, input_shape):
        dtype = neuraltide.config.get_dtype()
        n_taps = max(int(round(self.duration_s * 1000.0 / self.dt)), 1)
        t_s = tf.range(n_taps, dtype=dtype) * tf.constant(self.dt / 1000.0, dtype=dtype)
        hrf = self._gamma_pdf(t_s, 6.0) - self._gamma_pdf(t_s, 16.0) / 6.0
        hrf = hrf / tf.reduce_max(tf.abs(hrf))
        self._kernel = tf.reshape(hrf, [-1, 1, 1])

    def call(self, x: TensorType) -> TensorType:
        kernel = cast_if_needed(self._kernel, x.dtype)
        return _fft_causal_conv(x, kernel)
