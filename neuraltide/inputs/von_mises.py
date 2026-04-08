import tensorflow as tf
import numpy as np
from typing import Any, Dict

import neuraltide.config
from neuraltide.core.base import BaseInputGenerator
from neuraltide.core.types import TensorType


class VonMisesGenerator(BaseInputGenerator):
    """
    Генератор тета-ритмического входа на основе распределения фон Мизаса.

    rate(t) = (MeanFiringRate / I0(kappa)) * exp(kappa * cos(2π*freq*t/1000 - phase))

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
        **kwargs
    ):
        super().__init__(params=params, dt=dt, name=name, **kwargs)

        self.mean_rate = self._make_param(self._params, 'mean_rate')
        raw_R = self._params.get('R')
        if isinstance(raw_R, dict):
            raw_R = raw_R.get('value', raw_R)
        R_array = np.array(raw_R)
        if R_array.ndim == 0:
            R_array = np.array([R_array.item()])
        R_array = np.broadcast_to(R_array, [self.n_units])
        
        kappa_np = self._r2kappa_np(R_array)
        i0_kappa_np = self._i0_np(kappa_np)
        
        self.kappa = self.add_weight(
            shape=(self.n_units,),
            initializer=tf.keras.initializers.Constant(kappa_np),
            trainable=False,
            dtype=neuraltide.config.get_dtype(),
            name='kappa',
        )
        self.i0_kappa = self.add_weight(
            shape=(self.n_units,),
            initializer=tf.keras.initializers.Constant(i0_kappa_np),
            trainable=False,
            dtype=neuraltide.config.get_dtype(),
            name='i0_kappa',
        )
        self.freq = self._make_param(self._params, 'freq')
        self.phase = self._make_param(self._params, 'phase')

    def _i0_np(self, kappa: np.ndarray) -> np.ndarray:
        """Compute I0(kappa) using scipy."""
        from scipy.special import i0
        return i0(kappa).astype(np.float32)

    def _r2kappa_np(self, R: np.ndarray) -> np.ndarray:
        """Векторизованная версия r2kappa для numpy."""
        result = np.zeros_like(R, dtype=np.float64)
        mask1 = R < 0.53
        mask2 = (R >= 0.53) & (R < 0.85)
        mask3 = R >= 0.85
        
        result[mask1] = 2 * R[mask1] + R[mask1]**3 + (5/6) * R[mask1]**5
        result[mask2] = -0.4 + 1.39 * R[mask2] + 0.43 / (1 - R[mask2])
        result[mask3] = 1.0 / (3 * R[mask3] - 4 * R[mask3]**2 + R[mask3]**3)
        
        return result.astype(np.float32)

    def _r2kappa_tf(self, R: tf.Tensor) -> tf.Tensor:
        """Векторизованная версия r2kappa для TensorFlow."""
        kappa = tf.where(R < 0.53, 2 * R + R**3 + 5/6 * R**5, tf.zeros_like(R))
        kappa = tf.where(tf.logical_and(R >= 0.53, R < 0.85), 
                         -0.4 + 1.39 * R + 0.43 / (1 - R), kappa)
        kappa = tf.where(R >= 0.85, 
                         1.0 / (3 * R - 4 * R**2 + R**3), kappa)
        return kappa

    def call(self, t: TensorType) -> TensorType:
        """
        Args:
            t: текущее время в мс. shape = [batch, 1].

        Returns:
            tf.Tensor, shape = [batch, n_units], в Гц.
        """
        two_pi = tf.constant(2.0 * 3.141592653589793, dtype=neuraltide.config.get_dtype())

        mean_rate = tf.reshape(self.mean_rate, [1, self.n_units])
        kappa = tf.reshape(self.kappa, [1, self.n_units])
        i0_kappa = tf.reshape(self.i0_kappa, [1, self.n_units])
        freq = tf.reshape(self.freq, [1, self.n_units])
        phase = tf.reshape(self.phase, [1, self.n_units])

        norm_factor = mean_rate / i0_kappa

        argument = kappa * tf.cos(two_pi * freq * t / 1000.0 - phase)
        rate = norm_factor * tf.exp(argument)
        return tf.nn.relu(rate)

    @property
    def parameter_spec(self) -> Dict[str, Dict[str, Any]]:
        return {
            'mean_rate': {
                'shape': (self.n_units,),
                'trainable': self.mean_rate.trainable,
                'constraint': self._get_constraint_name(self.mean_rate),
                'units': 'Hz',
            },
            'kappa': {
                'shape': (self.n_units,),
                'trainable': self.kappa.trainable,
                'constraint': self._get_constraint_name(self.kappa),
                'units': 'dimensionless',
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
        }

    def _get_constraint_name(self, var: tf.Variable) -> str:
        if var.constraint is not None:
            return var.constraint.__class__.__name__
        return None
