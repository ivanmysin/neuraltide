from neuraltide.inputs.base import BaseInputGenerator
from neuraltide.inputs.von_mises import VonMisesGenerator
from neuraltide.inputs.sinusoidal import SinusoidalGenerator
from neuraltide.inputs.constant import ConstantRateGenerator
from neuraltide.inputs.place_field import PlaceFieldGenerator

import neuraltide.config

neuraltide.config.register_input('VonMisesGenerator', VonMisesGenerator)
neuraltide.config.register_input('SinusoidalGenerator', SinusoidalGenerator)
neuraltide.config.register_input('ConstantRateGenerator', ConstantRateGenerator)
neuraltide.config.register_input('PlaceFieldGenerator', PlaceFieldGenerator)

__all__ = [
    "BaseInputGenerator",
    "VonMisesGenerator",
    "SinusoidalGenerator",
    "ConstantRateGenerator",
    "PlaceFieldGenerator",
]
