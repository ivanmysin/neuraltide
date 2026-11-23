import neuraltide.config
from neuraltide.populations.brunel_fp import BrunelFokkerPlanck
from neuraltide.populations.fokker_planck import FokkerPlanckPopulation
from neuraltide.populations.input_population import InputPopulation
from neuraltide.populations.izhikevich_mf import IzhikevichMeanField
from neuraltide.populations.wilson_cowan import WilsonCowan

neuraltide.config.register_population("IzhikevichMeanField", IzhikevichMeanField)
neuraltide.config.register_population("WilsonCowan", WilsonCowan)
neuraltide.config.register_population("FokkerPlanckPopulation", FokkerPlanckPopulation)
neuraltide.config.register_population("BrunelFokkerPlanck", BrunelFokkerPlanck)
neuraltide.config.register_population("InputPopulation", InputPopulation)

__all__ = [
    "BrunelFokkerPlanck",
    "FokkerPlanckPopulation",
    "InputPopulation",
    "IzhikevichMeanField",
    "WilsonCowan",
]
