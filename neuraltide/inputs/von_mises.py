import tensorflow as tf
from typing import Any, Dict, Optional

from neuraltide.core.base import BaseInputGenerator, _get_constraint_name
from neuraltide.core.types import TensorType, get_pi


class VonMisesGenerator(BaseInputGenerator):
    """
    Генератор тета-ритмического входа на основе распределения фон Мизаса.

    rate(t) = (MeanFiringRate / I0(kappa)) * exp(kappa * cos(2π*freq*t/1000 - phase))

    kappa = kappa(R) — концентрация, пересчитывается из R при каждом вызове,
    поэтому R может быть обучаемым параметром.

    Параметры (params):
        mean_rate: средняя частота (Hz). Скаляр или вектор [n_units].
        R: R-value (0-1), характеризует концентрированность. Скаляр или вектор [n_units].
        freq: частота тета-ритма (Hz). Скаляр или вектор [n_units].
        phase: начальная фаза (рад). Скаляр или вектор [n_units].

    Пример:
        gen = VonMisesGenerator(
            dt=0.5,
            params={
                'mean_rate': 10.0,
                'R': 0.8,
                'freq': 8.0,
                'phase': 0.0,
            }
        )
        # n_units=1

        gen = VonMisesGenerator(
            dt=0.5,
            params={
                'mean_rate': [10.0, 15.0],
                'R': [0.8, 0.5],
                'freq': [8.0, 10.0],
                'phase': [0.0, 1.5],
            }
        )
        # n_units=2
    """

    def __init__(
        self,
        dt: float,
        params: Dict[str, Any],
        name: str = "von_mises_generator",
        **kwargs,
    ):
        super().__init__(params=params, dt=dt, name=name, **kwargs)

        self.mean_rate = self._make_param(self._params, "mean_rate")
        self.R = self._make_param(self._params, "R")
        self.freq = self._make_param(self._params, "freq")
        self.phase = self._make_param(self._params, "phase")

    @staticmethod
    def _r2kappa_tf(R: tf.Tensor) -> tf.Tensor:
        """Векторизованная аппроксимация kappa(R) (Banerjee et al. 2005)."""
        kappa = tf.where(R < 0.53, 2 * R + R**3 + 5.0 / 6.0 * R**5, tf.zeros_like(R))
        kappa = tf.where(
            tf.logical_and(R >= 0.53, R < 0.85), -0.4 + 1.39 * R + 0.43 / (1 - R), kappa
        )
        kappa = tf.where(R >= 0.85, 1.0 / (3 * R - 4 * R**2 + R**3), kappa)
        return kappa

    def call(
        self, t: TensorType, extra_inputs: Optional[TensorType] = None
    ) -> TensorType:
        """
        Args:
            t: текущее время в мс. shape = [batch, 1].
            extra_inputs: дополнительные входные данные (игнорируется).

        Returns:
            tf.Tensor, shape = [batch, n_units], в Гц.
        """
        two_pi = 2.0 * get_pi()

        mean_rate = tf.reshape(self.mean_rate, [1, self.n_units])
        # Защита от полюса аппроксимации kappa(R) при R -> 1
        R = tf.clip_by_value(tf.reshape(self.R, [1, self.n_units]), 0.0, 0.9999)
        kappa = self._r2kappa_tf(R)
        i0_kappa = tf.math.bessel_i0(kappa)
        freq = tf.reshape(self.freq, [1, self.n_units])
        phase = tf.reshape(self.phase, [1, self.n_units])

        norm_factor = mean_rate / i0_kappa

        argument = kappa * tf.cos(two_pi * freq * t / 1000.0 - phase)
        rate = norm_factor * tf.exp(argument)
        return tf.nn.relu(rate)

    @property
    def parameter_spec(self) -> Dict[str, Dict[str, Any]]:
        return {
            "mean_rate": {
                "shape": (self.n_units,),
                "trainable": self.mean_rate.trainable,
                "constraint": self._get_constraint_name(self.mean_rate),
                "units": "Hz",
            },
            "R": {
                "shape": (self.n_units,),
                "trainable": self.R.trainable,
                "constraint": self._get_constraint_name(self.R),
                "units": "dimensionless",
            },
            "freq": {
                "shape": (self.n_units,),
                "trainable": self.freq.trainable,
                "constraint": self._get_constraint_name(self.freq),
                "units": "Hz",
            },
            "phase": {
                "shape": (self.n_units,),
                "trainable": self.phase.trainable,
                "constraint": self._get_constraint_name(self.phase),
                "units": "rad",
            },
        }

    def _get_constraint_name(self, var: tf.Variable) -> Optional[str]:
        return _get_constraint_name(var)
