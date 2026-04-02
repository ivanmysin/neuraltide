import tensorflow as tf
from typing import Dict, Tuple

import neuraltide
import neuraltide.config
from neuraltide.core.base import SynapseModel
from neuraltide.core.types import TensorType, StateList


class NMDASynapse(SynapseModel):
    """
    NMDA синапс с двойной экспонентой и магниевым блоком.

    Состояние: [gnmda, dgnmda], shape [n_pre, n_post].

    Уравнения:
        dgnmda_new = dgnmda + dt*(s_input - gnmda - (tau1 + tau2)*dgnmda) / (tau1*tau2)
        gnmda_new  = gnmda + dt * dgnmda_new

    Магниевый блок:
        mg_block = 1 / (1 + Mgb * exp(-av*(post_v - v_ref)))
    """

    def __init__(self, n_pre: int, n_post: int, dt: float, params: dict, **kwargs):
        super().__init__(n_pre=n_pre, n_post=n_post, dt=dt, **kwargs)

        self.gsyn_max_nmda = self._make_param(params, 'gsyn_max_nmda')
        self.tau1_nmda = self._make_param(params, 'tau1_nmda')
        self.tau2_nmda = self._make_param(params, 'tau2_nmda')
        self.Mgb = self._make_param(params, 'Mgb')
        self.av_nmda = self._make_param(params, 'av_nmda')
        self.pconn_nmda = self._make_param(params, 'pconn_nmda')
        self.e_r_nmda = self._make_param(params, 'e_r_nmda')
        self.v_ref = self._make_param(params, 'v_ref')

        self.state_size = [
            tf.TensorShape([n_pre, n_post]),
            tf.TensorShape([n_pre, n_post]),
        ]

    def get_initial_state(self, batch_size: int = 1) -> StateList:
        dtype = neuraltide.config.get_dtype()
        return [
            tf.zeros([self.n_pre, self.n_post], dtype=dtype),
            tf.zeros([self.n_pre, self.n_post], dtype=dtype),
        ]

    def forward(
        self,
        pre_firing_rate: TensorType,
        post_voltage: TensorType,
        state: StateList,
        dt: float,
    ) -> Tuple[Dict[str, TensorType], StateList]:
        gnmda, dgnmda = state

        dtype = neuraltide.config.get_dtype()
        gsyn_max_nmda = tf.cast(self.gsyn_max_nmda, dtype)
        tau1_nmda = tf.cast(self.tau1_nmda, dtype)
        tau2_nmda = tf.cast(self.tau2_nmda, dtype)
        Mgb = tf.cast(self.Mgb, dtype)
        av_nmda = tf.cast(self.av_nmda, dtype)
        pconn_nmda = tf.cast(self.pconn_nmda, dtype)
        e_r_nmda = tf.cast(self.e_r_nmda, dtype)
        v_ref = tf.cast(self.v_ref, dtype)

        firing_probs_T = tf.transpose(dt * pre_firing_rate / 1000.0)
        s_input = pconn_nmda * firing_probs_T

        dgnmda_new = (dgnmda + dt * (s_input - gnmda - (tau1_nmda + tau2_nmda) * dgnmda)
                       / (tau1_nmda * tau2_nmda))
        gnmda_new = gnmda + dt * dgnmda_new

        post_v_flat = tf.reshape(post_voltage, [self.n_post])
        mg_block = 1.0 / (1.0 + Mgb * tf.exp(-av_nmda * (post_v_flat - v_ref)))

        g_eff = gsyn_max_nmda * gnmda_new * mg_block
        I_pair = g_eff * (e_r_nmda - post_v_flat)
        I_syn = tf.reduce_sum(I_pair, axis=0, keepdims=True)
        g_syn = tf.reduce_sum(g_eff, axis=0, keepdims=True)

        return ({'I_syn': I_syn, 'g_syn': g_syn}, [gnmda_new, dgnmda_new])

    @property
    def parameter_spec(self) -> Dict[str, Dict[str, any]]:
        return {
            'gsyn_max_nmda': {
                'shape': (self.n_pre, self.n_post),
                'trainable': self.gsyn_max_nmda.trainable,
                'constraint': 'NonNegConstraint',
                'units': 'mS/cm^2',
            },
            'tau1_nmda': {
                'shape': (self.n_pre, self.n_post),
                'trainable': False,
                'constraint': None,
                'units': 'ms',
            },
            'tau2_nmda': {
                'shape': (self.n_pre, self.n_post),
                'trainable': False,
                'constraint': None,
                'units': 'ms',
            },
            'Mgb': {
                'shape': (self.n_pre, self.n_post),
                'trainable': False,
                'constraint': None,
                'units': 'mM',
            },
            'av_nmda': {
                'shape': (self.n_pre, self.n_post),
                'trainable': False,
                'constraint': None,
                'units': 'mV^-1',
            },
            'pconn_nmda': {
                'shape': (self.n_pre, self.n_post),
                'trainable': False,
                'constraint': 'UnitIntervalConstraint',
                'units': 'dimensionless',
            },
            'e_r_nmda': {
                'shape': (self.n_pre, self.n_post),
                'trainable': False,
                'constraint': None,
                'units': 'mV',
            },
            'v_ref': {
                'shape': (self.n_pre, self.n_post),
                'trainable': False,
                'constraint': None,
                'units': 'mV',
            },
        }
