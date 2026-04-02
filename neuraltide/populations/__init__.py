from neuraltide.populations.izhikevich_mf import IzhikevichMeanField
from neuraltide.populations.wilson_cowan import WilsonCowan
from neuraltide.populations.fokker_planck import FokkerPlanckPopulation
from neuraltide.populations.input_population import InputPopulation

import neuraltide.config

neuraltide.config.register_population('IzhikevichMeanField', IzhikevichMeanField)
neuraltide.config.register_population('WilsonCowan', WilsonCowan)
neuraltide.config.register_population('FokkerPlanckPopulation', FokkerPlanckPopulation)
neuraltide.config.register_population('InputPopulation', InputPopulation)

__all__ = [
    "IzhikevichMeanField",
    "WilsonCowan",
    "FokkerPlanckPopulation",
    "InputPopulation",
]
