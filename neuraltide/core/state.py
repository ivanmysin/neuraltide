from dataclasses import dataclass
from typing import Dict, List

from neuraltide.core.types import StateList


@dataclass
class NetworkState:
    """
    Полное состояние сети в один момент времени.

    Атрибуты:
        population_states: dict[str, StateList]
            Ключи — имена популяций в порядке регистрации в NetworkGraph.
            Включает как динамические популяции, так и InputPopulation.
        synapse_states: dict[str, StateList]
            Ключи — имена синаптических проекций в порядке регистрации.
    """

    population_states: Dict[str, StateList]
    synapse_states: Dict[str, StateList]

    def to_flat_list(self) -> StateList:
        """
        Сериализует в плоский список тензоров для RNN state.
        Порядок: популяции → синапсы → stability_error.
        """
        flat = []
        for state_list in self.population_states.values():
            flat.extend(state_list)
        for state_list in self.synapse_states.values():
            flat.extend(state_list)
        return flat

    @staticmethod
    def from_flat_list(
        flat: StateList,
        pop_names: List[str],
        pop_sizes: Dict[str, int],
        syn_names: List[str],
        syn_sizes: Dict[str, int],
    ) -> 'NetworkState':
        """Десериализует из плоского списка."""
        idx = 0
        pop_states = {}
        for name in pop_names:
            n = pop_sizes[name]
            pop_states[name] = flat[idx:idx + n]
            idx += n
        syn_states = {}
        for name in syn_names:
            n = syn_sizes[name]
            syn_states[name] = flat[idx:idx + n]
            idx += n
        return NetworkState(
            population_states=pop_states,
            synapse_states=syn_states,
        )
