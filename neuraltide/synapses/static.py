import tensorflow as tf
from typing import Dict, Tuple

import neuraltide
import neuraltide.config
from neuraltide.core.base import SynapseModel
from neuraltide.core.types import TensorType, StateList


class StaticSynapse(SynapseModel):
    """
    Статический синапс без пластичности.

    Состояние: [] (пустой список).

    I_syn = gsyn_max * FRpre_normed * (e_r - post_v)
    g_syn = gsyn_max * FRpre_normed
    """

    def __init__(self, n_pre: int, n_post: int, dt: float, params: dict, **kwargs):
        super().__init__(n_pre=n_pre, n_post=n_post, dt=dt, **kwargs)

        self.gsyn_max = self._make_param(params, 'gsyn_max')
        self.pconn = self._make_param(params, 'pconn')
        self.e_r = self._make_param(params, 'e_r')

        self.state_size = []

    def get_initial_state(self, batch_size: int = 1) -> StateList:
        return []

    def forward(
        self,
        pre_firing_rate: TensorType,
        post_voltage: TensorType,
        state: StateList,
        dt: float,
    ) -> Tuple[Dict[str, TensorType], StateList]:
        dtype = neuraltide.config.get_dtype()
        gsyn_max = tf.cast(self.gsyn_max, dtype)
        pconn = tf.cast(self.pconn, dtype)
        e_r = tf.cast(self.e_r, dtype)

        firing_probs_T = tf.transpose(dt * pre_firing_rate / 1000.0)
        FRpre_normed = pconn * firing_probs_T

        post_v_T = tf.transpose(post_voltage)
        I_pair = gsyn_max * FRpre_normed * (e_r - post_v_T)
        I_syn = tf.reduce_sum(I_pair, axis=0, keepdims=True)
        g_syn = tf.reduce_sum(gsyn_max * FRpre_normed, axis=0, keepdims=True)

        return ({'I_syn': I_syn, 'g_syn': g_syn}, [])

    @property
    def parameter_spec(self) -> Dict[str, Dict[str, any]]:
        return {
            'gsyn_max': {
                'shape': (self.n_pre, self.n_post),
                'trainable': self.gsyn_max.trainable,
                'constraint': 'NonNegConstraint',
                'units': 'mS/cm^2',
            },
            'pconn': {
                'shape': (self.n_pre, self.n_post),
                'trainable': False,
                'constraint': 'UnitIntervalConstraint',
                'units': 'dimensionless',
            },
            'e_r': {
                'shape': (self.n_pre, self.n_post),
                'trainable': False,
                'constraint': None,
                'units': 'mV',
            },
        }
