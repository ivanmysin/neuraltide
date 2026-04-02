from neuraltide.inputs.base import BaseInputGenerator
from neuraltide.inputs.von_mises import VonMisesGenerator
from neuraltide.inputs.sinusoidal import SinusoidalGenerator
from neuraltide.inputs.constant import ConstantRateGenerator

import neuraltide.config

neuraltide.config.register_input('VonMisesGenerator', VonMisesGenerator)
neuraltide.config.register_input('SinusoidalGenerator', SinusoidalGenerator)
neuraltide.config.register_input('ConstantRateGenerator', ConstantRateGenerator)

__all__ = [
    "BaseInputGenerator",
    "VonMisesGenerator",
    "SinusoidalGenerator",
    "ConstantRateGenerator",
]
