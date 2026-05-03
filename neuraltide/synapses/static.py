import tensorflow as tf
from typing import Dict, Tuple, List

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

    def derivatives(
        self,
        state: StateList,
        pre_firing_rate: TensorType,
        post_voltage: TensorType,
    ) -> StateList:
        """
        Compute derivatives for the static synapse.

        Static synapse has no state dynamics, so derivatives is empty.
        """
        return []

    def compute_current(
        self,
        state: StateList,
        pre_firing_rate: TensorType,
        post_voltage: TensorType,
    ) -> Dict[str, TensorType]:
        """
        Compute synaptic current and conductance.

        Used after numerical integration to compute currents.
        """
        dtype = neuraltide.config.get_dtype()
        gsyn_max = tf.cast(self.gsyn_max, dtype)
        pconn = tf.cast(self.pconn, dtype)
        e_r = tf.cast(self.e_r, dtype)

        firing_probs_T = tf.transpose(pre_firing_rate / 1000.0)
        FRpre_normed = pconn * firing_probs_T

        post_v_T = tf.transpose(post_voltage)
        I_pair = gsyn_max * FRpre_normed * (e_r - post_v_T)
        I_syn = tf.reduce_sum(I_pair, axis=0, keepdims=True)
        g_syn = tf.reduce_sum(gsyn_max * FRpre_normed, axis=0, keepdims=True)

        return {'I_syn': I_syn, 'g_syn': g_syn}

    def adjoint_derivatives(
        self,
        adjoint_state: StateList,
        state: StateList,
        pre_firing_rate: TensorType,
        post_voltage: TensorType,
    ) -> StateList:
        return []

    def parameter_jacobian(
        self,
        param_name: str,
        state: StateList,
        pre_firing_rate: TensorType,
        post_voltage: TensorType,
    ) -> TensorType:
        dtype = neuraltide.config.get_dtype()
        return tf.zeros([self.n_pre, self.n_post], dtype=dtype)

    def compute_current_state_vjp(
        self,
        lam_I: TensorType,
        lam_g: TensorType,
        state: StateList,
        pre_firing_rate: TensorType,
        post_voltage: TensorType,
    ) -> StateList:
        return []

    def compute_current_param_grad(
        self,
        lam_I: TensorType,
        lam_g: TensorType,
        state: StateList,
        pre_firing_rate: TensorType,
        post_voltage: TensorType,
    ) -> Dict[str, TensorType]:
        """
        Static synapse: gsyn_max only.

        I_syn = sum_j(gsyn_max * FRpre_normed * (e_r - post_v))
        ∂I_syn/∂gsyn_max[i,j] = FRpre_normed[i,j] * (e_r[i,j] - post_v[j])
        """
        dtype = neuraltide.config.get_dtype()
        gsyn_max = tf.cast(self.gsyn_max, dtype)
        pconn = tf.cast(self.pconn, dtype)
        e_r = tf.cast(self.e_r, dtype)

        firing_probs_T = tf.transpose(pre_firing_rate / 1000.0)
        FRpre_normed = pconn * firing_probs_T

        lam_I_sum = tf.reduce_sum(lam_I, axis=0)
        lam_g_sum = tf.reduce_sum(lam_g, axis=0)

        if self.gsyn_max.trainable:
            dgsyn_max = FRpre_normed * (
                lam_I_sum * (e_r - post_voltage) + lam_g_sum)
        else:
            dgsyn_max = tf.zeros_like(gsyn_max)

        return {'gsyn_max': dgsyn_max}

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
