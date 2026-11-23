import numpy as np

import neuraltide.config
from neuraltide.core.base import BaseInputGenerator
from neuraltide.inputs.constant import ConstantRateGenerator
from neuraltide.inputs.place_field import PlaceFieldGenerator
from neuraltide.inputs.sinusoidal import SinusoidalGenerator
from neuraltide.inputs.von_mises import VonMisesGenerator

neuraltide.config.register_input("VonMisesGenerator", VonMisesGenerator)
neuraltide.config.register_input("SinusoidalGenerator", SinusoidalGenerator)
neuraltide.config.register_input("ConstantRateGenerator", ConstantRateGenerator)
neuraltide.config.register_input("PlaceFieldGenerator", PlaceFieldGenerator)


def peak_time_to_phases(t_peak_ms: float, freq_hz: float) -> dict:
    """Фазы встроенных тета-генераторов для максимума в момент t_peak_ms.

    Конвенции знака фазы у генераторов разные: VonMises — cos(ωt − φ),
    Sinusoidal — sin(ωt + φ), PlaceField — cos(ωt + φ + dφ). Хелпер
    переводит желаемый момент максимума в аргумент phase каждого
    генератора в его собственных единицах (полный разбор —
    docs/inputs.md, раздел «Конвенции фазы»).

    Args:
        t_peak_ms: момент максимума, мс от t=0.
        freq_hz: частота, Гц.

    Returns:
        dict с ключами:
            'von_mises'  — фаза phase [рад];
            'sinusoidal' — фаза phase [рад];
            'place_field' — фаза precession_init_phase [град] (та же
              конвенция у phase_outside, тоже в градусах).
    """
    omega_t = 2.0 * np.pi * freq_hz * t_peak_ms / 1000.0
    return {
        "von_mises": omega_t,
        "sinusoidal": np.pi / 2.0 - omega_t,
        "place_field": float(np.degrees(-omega_t)),
    }


__all__ = [
    "BaseInputGenerator",
    "ConstantRateGenerator",
    "PlaceFieldGenerator",
    "SinusoidalGenerator",
    "VonMisesGenerator",
    "peak_time_to_phases",
]
