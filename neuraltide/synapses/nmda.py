import tensorflow as tf
from typing import Dict, Tuple, List

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

    def derivatives(
        self,
        state: StateList,
        pre_firing_rate: TensorType,
        post_voltage: TensorType,
    ) -> StateList:
        """
        Compute derivatives for the NMDA synapse dynamics.

        Differential equations:
            d(gnmda)/dt = dgnmda
            d(dgnmda)/dt = (s_input - gnmda - (tau1 + tau2)*dgnmda) / (tau1*tau2)

        where s_input = pconn * firing_probability (converted to rate)
        """
        gnmda, dgnmda = state

        dtype = neuraltide.config.get_dtype()
        pconn = tf.cast(self.pconn_nmda, dtype)
        tau1 = tf.maximum(tf.cast(self.tau1_nmda, dtype), 1e-6)
        tau2 = tf.maximum(tf.cast(self.tau2_nmda, dtype), 1e-6)

        firing_probs_T = tf.transpose(pre_firing_rate / 1000.0)
        s_input = pconn * firing_probs_T

        d_gnmda_dt = dgnmda

        d_dgnmda_dt = (s_input - gnmda - (tau1 + tau2) * dgnmda) / (tau1 * tau2)

        d_gnmda_dt = neuraltide.config.maybe_check_numerics(d_gnmda_dt, 'NMDA d(gnmda)/dt NaN')
        d_dgnmda_dt = neuraltide.config.maybe_check_numerics(d_dgnmda_dt, 'NMDA d(dgnmda)/dt NaN')

        return [d_gnmda_dt, d_dgnmda_dt]

    def compute_current(
        self,
        state: StateList,
        pre_firing_rate: TensorType,
        post_voltage: TensorType,
    ) -> Dict[str, TensorType]:
        """
        Compute synaptic current and conductance from state.

        Used after numerical integration to compute currents.
        Includes magnesium block.
        """
        dtype = neuraltide.config.get_dtype()
        gsyn_max_nmda = tf.cast(self.gsyn_max_nmda, dtype)
        Mgb = tf.cast(self.Mgb, dtype)
        av_nmda = tf.cast(self.av_nmda, dtype)
        e_r_nmda = tf.cast(self.e_r_nmda, dtype)
        v_ref = tf.cast(self.v_ref, dtype)

        gnmda = state[0]

        post_v_flat = tf.reshape(post_voltage, [-1])
        mg_block = 1.0 / (1.0 + Mgb * tf.exp(-av_nmda * (post_v_flat - v_ref)))

        g_eff = gsyn_max_nmda * gnmda * mg_block
        I_pair = g_eff * (e_r_nmda - post_v_flat)
        I_syn = tf.reduce_sum(I_pair, axis=0, keepdims=True)
        g_syn = tf.reduce_sum(g_eff, axis=0, keepdims=True)

        return {'I_syn': I_syn, 'g_syn': g_syn}

    def adjoint_derivatives(
        self,
        adjoint_state: StateList,
        state: StateList,
        pre_firing_rate: TensorType,
        post_voltage: TensorType,
    ) -> StateList:
        """
        Compute adjoint derivatives for NMDA synapse.

        dλ_gnmda/dt = λ_dgnmda / (tau1 * tau2)
        dλ_dgnmda/dt = -λ_gnmda + (tau1+tau2)/(tau1*tau2) * λ_dgnmda
        """
        lam_gnmda, lam_dgnmda = adjoint_state

        dtype = neuraltide.config.get_dtype()
        tau1 = tf.maximum(tf.cast(self.tau1_nmda, dtype), 1e-6)
        tau2 = tf.maximum(tf.cast(self.tau2_nmda, dtype), 1e-6)
        t12 = tau1 * tau2

        dlam_gnmda = lam_dgnmda / t12
        dlam_dgnmda = -lam_gnmda + (tau1 + tau2) / t12 * lam_dgnmda

        dlam_gnmda = neuraltide.config.maybe_check_numerics(dlam_gnmda, 'NMDA adjoint dlam_gnmda NaN')
        dlam_dgnmda = neuraltide.config.maybe_check_numerics(dlam_dgnmda, 'NMDA adjoint dlam_dgnmda NaN')

        return [dlam_gnmda, dlam_dgnmda]

    def parameter_jacobian(
        self,
        param_name: str,
        state: StateList,
        pre_firing_rate: TensorType,
        post_voltage: TensorType,
    ) -> TensorType:
        """
        ∂derivatives/∂param for NMDA synapse.

        Only gsyn_max_nmda is trainable and does not appear in derivatives → zero.
        Non-trainable params (tau1, tau2) also not implemented.
        """
        gnmda, dgnmda = state
        return tf.zeros_like(gnmda)

    def compute_current_state_vjp(
        self,
        lam_I: TensorType,
        lam_g: TensorType,
        state: StateList,
        pre_firing_rate: TensorType,
        post_voltage: TensorType,
    ) -> StateList:
        """
        VJP from I_syn/g_syn back to synapse state.

        I_syn[b,j] = sum_i gsyn_max[i,j] * gnmda[i,j] * mg_block[b,j] * (e_r - post_v[b,j])
        g_syn[b,j] = sum_i gsyn_max[i,j] * gnmda[i,j] * mg_block[b,j]

        Only gnmda contributes (dgnmda doesn't appear):
            dgnmda[i,j] += sum_b lam_I[b,j] * gsyn_max[i,j] * mg_block[b,j] * (e_r - post_v[b,j])
                          + sum_b lam_g[b,j] * gsyn_max[i,j] * mg_block[b,j]
        """
        gnmda, dgnmda = state

        dtype = neuraltide.config.get_dtype()
        gsyn_max = tf.cast(self.gsyn_max_nmda, dtype)
        Mgb = tf.cast(self.Mgb, dtype)
        av_nmda = tf.cast(self.av_nmda, dtype)
        e_r_nmda = tf.cast(self.e_r_nmda, dtype)
        v_ref = tf.cast(self.v_ref, dtype)

        mg_block = 1.0 / (1.0 + Mgb * tf.exp(-av_nmda * (post_voltage - v_ref)))

        lam_I_sum = tf.reduce_sum(lam_I, axis=0)
        lam_g_sum = tf.reduce_sum(lam_g, axis=0)

        dgnmda_contrib = gsyn_max * mg_block * (lam_I_sum * (e_r_nmda - post_voltage) + lam_g_sum)

        return [dgnmda_contrib, tf.zeros_like(dgnmda)]

    def compute_current_param_grad(
        self,
        lam_I: TensorType,
        lam_g: TensorType,
        state: StateList,
        pre_firing_rate: TensorType,
        post_voltage: TensorType,
    ) -> Dict[str, TensorType]:
        """
        Gradients for current-level parameters (only gsyn_max_nmda).
        """
        gnmda, dgnmda = state

        dtype = neuraltide.config.get_dtype()
        gsyn_max = tf.cast(self.gsyn_max_nmda, dtype)
        Mgb = tf.cast(self.Mgb, dtype)
        av_nmda = tf.cast(self.av_nmda, dtype)
        e_r_nmda = tf.cast(self.e_r_nmda, dtype)
        v_ref = tf.cast(self.v_ref, dtype)

        mg_block = 1.0 / (1.0 + Mgb * tf.exp(-av_nmda * (post_voltage - v_ref)))
        lam_I_sum = tf.reduce_sum(lam_I, axis=0)
        lam_g_sum = tf.reduce_sum(lam_g, axis=0)

        if self.gsyn_max_nmda.trainable:
            dgsyn_max = gnmda * mg_block * (
                lam_I_sum * (e_r_nmda - post_voltage) + lam_g_sum)
        else:
            dgsyn_max = tf.zeros_like(gnmda)

        return {'gsyn_max_nmda': dgsyn_max}

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
