from dataclasses import dataclass
from typing import Dict, List, Any, Optional

import neuraltide.config
from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.integrators import EulerIntegrator, HeunIntegrator, RK4Integrator


@dataclass
class PopulationConfig:
    """Конфигурация популяции."""
    name: str
    model_class: str
    dt: float
    params: Dict[str, Any]


@dataclass
class SynapseConfig:
    """Конфигурация синапса."""
    name: str
    synapse_class: str
    src: str
    tgt: str
    dt: float
    params: Dict[str, Any]
    components: Optional[List['SynapseConfig']] = None


@dataclass
class InputConfig:
    """Конфигурация входного генератора."""
    name: str
    generator_class: str
    params: Dict[str, Any]


@dataclass
class NetworkConfig:
    """Полная конфигурация сети."""
    dt: float
    integrator: str
    populations: List[PopulationConfig]
    synapses: List[SynapseConfig]
    inputs: List[InputConfig]
    stability_penalty_weight: float = 0.0
    return_hidden_states: bool = False


def _get_integrator(name: str) -> Any:
    """Возвращает класс интегратора по имени."""
    integrators = {
        'euler': EulerIntegrator,
        'heun': HeunIntegrator,
        'rk4': RK4Integrator,
    }
    if name not in integrators:
        raise ValueError(f"Unknown integrator: {name}. Available: {list(integrators.keys())}")
    return integrators[name]()


def build_network_from_config(config: NetworkConfig) -> NetworkRNN:
    """
    Строит NetworkRNN из конфигурации.

    Args:
        config: NetworkConfig с описанием сети.

    Returns:
        NetworkRNN.
    """
    graph = NetworkGraph(dt=config.dt)

    for input_config in config.inputs:
        gen_class = neuraltide.config.INPUT_REGISTRY.get(input_config.generator_class)
        if gen_class is None:
            raise ValueError(f"Unknown input generator: {input_config.generator_class}")
        generator = gen_class(**input_config.params)
        graph.add_input_population(input_config.name, generator)

    for pop_config in config.populations:
        pop_class = neuraltide.config.POPULATION_REGISTRY.get(pop_config.model_class)
        if pop_class is None:
            raise ValueError(f"Unknown population model: {pop_config.model_class}")
        population = pop_class(
            n_units=pop_config.params.get('n_units', 1),
            dt=pop_config.dt,
            params=pop_config.params,
            name=pop_config.name,
        )
        graph.add_population(pop_config.name, population)

    for syn_config in config.synapses:
        syn_class = neuraltide.config.SYNAPSE_REGISTRY.get(syn_config.synapse_class)
        if syn_class is None:
            raise ValueError(f"Unknown synapse: {syn_config.synapse_class}")

        if syn_config.components is not None:
            components = []
            for comp_config in syn_config.components:
                comp_syn_class = neuraltide.config.SYNAPSE_REGISTRY.get(comp_config.synapse_class)
                if comp_syn_class is None:
                    raise ValueError(f"Unknown synapse component: {comp_config.synapse_class}")
                comp_syn = comp_syn_class(
                    n_pre=syn_config.params.get('n_pre'),
                    n_post=syn_config.params.get('n_post'),
                    dt=syn_config.dt,
                    params=comp_config.params,
                    name=comp_config.name,
                )
                components.append((comp_config.name, comp_syn))
            synapse = neuraltide.config.SYNAPSE_REGISTRY['CompositeSynapse'](
                n_pre=syn_config.params.get('n_pre'),
                n_post=syn_config.params.get('n_post'),
                dt=syn_config.dt,
                components=components,
                name=syn_config.name,
            )
        else:
            synapse = syn_class(
                n_pre=syn_config.params.get('n_pre'),
                n_post=syn_config.params.get('n_post'),
                dt=syn_config.dt,
                params=syn_config.params,
                name=syn_config.name,
            )
        graph.add_synapse(syn_config.name, synapse, src=syn_config.src, tgt=syn_config.tgt)

    integrator = _get_integrator(config.integrator)

    network = NetworkRNN(
        graph=graph,
        integrator=integrator,
        stability_penalty_weight=config.stability_penalty_weight,
        return_hidden_states=config.return_hidden_states,
    )

    return network
