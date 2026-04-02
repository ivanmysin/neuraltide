from neuraltide.synapses.tsodyks_markram import TsodyksMarkramSynapse
from neuraltide.synapses.nmda import NMDASynapse
from neuraltide.synapses.static import StaticSynapse
from neuraltide.synapses.composite import CompositeSynapse

import neuraltide.config

neuraltide.config.register_synapse('TsodyksMarkramSynapse', TsodyksMarkramSynapse)
neuraltide.config.register_synapse('NMDASynapse', NMDASynapse)
neuraltide.config.register_synapse('StaticSynapse', StaticSynapse)
neuraltide.config.register_synapse('CompositeSynapse', CompositeSynapse)

__all__ = [
    "TsodyksMarkramSynapse",
    "NMDASynapse",
    "StaticSynapse",
    "CompositeSynapse",
]
