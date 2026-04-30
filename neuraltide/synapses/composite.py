import tensorflow as tf
from typing import Dict, Tuple, List

import neuraltide
import neuraltide.config
from neuraltide.core.base import SynapseModel
from neuraltide.core.types import TensorType, StateList


class CompositeSynapse(SynapseModel):
    """
    Композитный синапс, объединяющий несколько синаптических компонент.

    Токи суммируются:
        I_syn_total = sum(I_syn_i for i in components)
        g_syn_total = sum(g_syn_i for i in components)

    Args:
        components: список кортежей (name, SynapseModel) для каждой компоненты.
    """

    def __init__(
        self,
        n_pre: int,
        n_post: int,
        dt: float,
        components: List[Tuple[str, SynapseModel]],
        **kwargs
    ):
        super().__init__(n_pre=n_pre, n_post=n_post, dt=dt, **kwargs)

        self.components: List[Tuple[str, SynapseModel]] = components

        self.state_size = []
        for name, syn in components:
            self.state_size.extend(syn.state_size)

    def get_initial_state(self, batch_size: int = 1) -> StateList:
        states = []
        for name, syn in self.components:
            states.extend(syn.get_initial_state(batch_size))
        return states

    def forward(
        self,
        pre_firing_rate: TensorType,
        post_voltage: TensorType,
        state: StateList,
        dt: float,
    ) -> Tuple[Dict[str, TensorType], StateList]:
        dtype = neuraltide.config.get_dtype()
        total_I_syn = tf.zeros([1, self.n_post], dtype=dtype)
        total_g_syn = tf.zeros([1, self.n_post], dtype=dtype)

        new_states = []
        state_idx = 0

        for name, syn in self.components:
            n_syn_states = len(syn.state_size)
            syn_state = state[state_idx:state_idx + n_syn_states]
            state_idx += n_syn_states

            current_dict, new_syn_state = syn.forward(
                pre_firing_rate, post_voltage, syn_state, dt
            )

            total_I_syn += current_dict['I_syn']
            total_g_syn += current_dict['g_syn']
            new_states.extend(new_syn_state)

        return ({'I_syn': total_I_syn, 'g_syn': total_g_syn}, new_states)

    def derivatives(
        self,
        state: StateList,
        pre_firing_rate: TensorType,
        post_voltage: TensorType,
    ) -> StateList:
        """
        Compute derivatives as the concatenation of all component derivatives.
        """
        derivs = []
        state_idx = 0

        for name, syn in self.components:
            n_syn_states = len(syn.state_size)
            syn_state = state[state_idx:state_idx + n_syn_states]
            state_idx += n_syn_states

            comp_derivs = syn.derivatives(syn_state, pre_firing_rate, post_voltage)
            derivs.extend(comp_derivs)

        return derivs

    def compute_current(
        self,
        state: StateList,
        pre_firing_rate: TensorType,
        post_voltage: TensorType,
    ) -> Dict[str, TensorType]:
        """
        Compute total synaptic current as sum of all components.
        """
        dtype = neuraltide.config.get_dtype()
        total_I_syn = tf.zeros([1, self.n_post], dtype=dtype)
        total_g_syn = tf.zeros([1, self.n_post], dtype=dtype)

        state_idx = 0

        for name, syn in self.components:
            n_syn_states = len(syn.state_size)
            syn_state = state[state_idx:state_idx + n_syn_states]
            state_idx += n_syn_states

            comp_current = syn.compute_current(syn_state, pre_firing_rate, post_voltage)
            total_I_syn += comp_current['I_syn']
            total_g_syn += comp_current['g_syn']

        return {'I_syn': total_I_syn, 'g_syn': total_g_syn}

    @property
    def parameter_spec(self) -> Dict[str, Dict[str, any]]:
        spec = {}
        for name, syn in self.components:
            syn_spec = syn.parameter_spec
            for param_name, param_info in syn_spec.items():
                spec[f"{name}_{param_name}"] = param_info
        return spec
