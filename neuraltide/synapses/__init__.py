import neuraltide.config
from neuraltide.synapses.composite import CompositeSynapse
from neuraltide.synapses.nmda import NMDASynapse
from neuraltide.synapses.second_order import SecondOrderSynapse
from neuraltide.synapses.static import StaticSynapse
from neuraltide.synapses.tsodyks_markram import TsodyksMarkramSynapse

neuraltide.config.register_synapse("TsodyksMarkramSynapse", TsodyksMarkramSynapse)
neuraltide.config.register_synapse("NMDASynapse", NMDASynapse)
neuraltide.config.register_synapse("StaticSynapse", StaticSynapse)
neuraltide.config.register_synapse("SecondOrderSynapse", SecondOrderSynapse)
neuraltide.config.register_synapse("CompositeSynapse", CompositeSynapse)

__all__ = [
    "CompositeSynapse",
    "NMDASynapse",
    "SecondOrderSynapse",
    "StaticSynapse",
    "TsodyksMarkramSynapse",
]
