import tensorflow as tf
from scipy.special import i0
import numpy as np

import neuraltide
import neuraltide.config
from neuraltide.core.base import BaseInputGenerator
from neuraltide.core.types import TensorType


def _r2kappa(r: float) -> float:
    """Аппроксимация r2kappa по формуле."""
    if r <= 0:
        return 0.0
    if r >= 1:
        return 1000.0
    return r * (2.0 + r**2) / (1.0 - r**2)


class VonMisesGenerator(BaseInputGenerator):
    """
    Генератор тета-ритмического входа на основе распределения фон Мизаса.

    rate(t) = (MeanFiringRate / I0(kappa)) * exp(kappa * cos(2π*freq*t/1000 - phase))

    Параметры (params — список словарей с ключами):
        - MeanFiringRate: средняя частота (Hz)
        - R: R-value (0-1), характеризует концентрированность
        - ThetaFreq: частота тета-ритма (Hz)
        - ThetaPhase: начальная фаза (рад)
    """

    def __init__(self, params, name: str = "von_mises_generator", **kwargs):
        n_outputs = len(params)
        super().__init__(n_outputs=n_outputs, name=name, **kwargs)
        self.params = []
        for p in params:
            mean_rate = tf.constant(p['MeanFiringRate'], dtype=neuraltide.config.get_dtype())
            kappa = tf.constant(_r2kappa(p['R']), dtype=neuraltide.config.get_dtype())
            freq = tf.constant(p['ThetaFreq'], dtype=neuraltide.config.get_dtype())
            phase = tf.constant(p['ThetaPhase'], dtype=neuraltide.config.get_dtype())
            self.params.append({
                'mean_rate': mean_rate,
                'kappa': kappa,
                'freq': freq,
                'phase': phase,
            })

    def call(self, t: TensorType) -> TensorType:
        result = []
        for p in self.params:
            mean_rate = p['mean_rate']
            kappa = p['kappa']
            freq = p['freq']
            phase = p['phase']

            kappa_float = kappa.numpy()
            i0_kappa = i0(kappa_float) if kappa_float > 0 else 1.0
            norm_factor = mean_rate / i0_kappa

            two_pi = tf.constant(2.0 * 3.141592653589793, dtype=neuraltide.config.get_dtype())
            argument = kappa * tf.cos(two_pi * freq * t / 1000.0 - phase)
            rate = norm_factor * tf.exp(argument)
            result.append(rate)

        return tf.concat(result, axis=-1)
