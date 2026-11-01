import tensorflow as tf
from typing import Dict, Tuple

import neuraltide
import neuraltide.config
from neuraltide.core.base import SynapseModel
from neuraltide.core.types import TensorType, StateList


class SecondOrderSynapse(SynapseModel):
    """
    Синапс второго порядка без пластичности.

    Состояние: [g_s, dg_s], каждое shape [n_pre, n_post].

    Уравнение:
        tau1 * tau2 * d²g_s/dt² + (tau1 + tau2) * dg_s/dt + g_s = nu_pre

    Стационарное состояние: g_s,eq = nu_pre
    Ток: I_syn = gsyn_max * g_s * (e_r - V)
    """

    def __init__(self, n_pre: int, n_post: int, dt: float, params: dict, **kwargs):
        super().__init__(n_pre=n_pre, n_post=n_post, dt=dt, **kwargs)

        self.gsyn_max = self._make_param(params, 'gsyn_max')
        self.tau1 = self._make_param(params, 'tau1')
        self.tau2 = self._make_param(params, 'tau2')
        self.pconn = self._make_param(params, 'pconn')
        self.e_r = self._make_param(params, 'e_r')

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
        g_s, dg_s = state

        dtype = neuraltide.config.get_dtype()
        gsyn_max = tf.cast(self.gsyn_max, dtype)
        tau1 = tf.cast(self.tau1, dtype)
        tau2 = tf.cast(self.tau2, dtype)
        pconn = tf.cast(self.pconn, dtype)
        e_r = tf.cast(self.e_r, dtype)

        firing_probs_T = tf.transpose(dt * pre_firing_rate / 1000.0)
        s_input = pconn * firing_probs_T

        tau12 = tau1 * tau2
        t12_sum = tau1 + tau2

        dg_s_new = dg_s + dt * (s_input - t12_sum * dg_s - g_s) / tau12
        g_s_new = g_s + dt * dg_s_new

        post_v_flat = tf.reshape(post_voltage, [self.n_post])
        I_pair = gsyn_max * g_s_new * (e_r - post_v_flat)
        I_syn = tf.reduce_sum(I_pair, axis=0, keepdims=True)
        g_syn = tf.reduce_sum(gsyn_max * g_s_new, axis=0, keepdims=True)

        return ({'I_syn': I_syn, 'g_syn': g_syn}, [g_s_new, dg_s_new])

    def derivatives(
        self,
        state: StateList,
        pre_firing_rate: TensorType,
        post_voltage: TensorType,
    ) -> StateList:
        g_s, dg_s = state

        dtype = neuraltide.config.get_dtype()
        pconn = tf.cast(self.pconn, dtype)
        tau1 = tf.maximum(tf.cast(self.tau1, dtype), 1e-6)
        tau2 = tf.maximum(tf.cast(self.tau2, dtype), 1e-6)

        firing_probs_T = tf.transpose(pre_firing_rate / 1000.0)
        s_input = pconn * firing_probs_T

        tau12 = tau1 * tau2
        t12_sum = tau1 + tau2

        d_g_s_dt = dg_s
        d_dg_s_dt = (s_input - t12_sum * dg_s - g_s) / tau12

        d_g_s_dt = neuraltide.config.maybe_check_numerics(d_g_s_dt, 'SecondOrder d(g_s)/dt NaN')
        d_dg_s_dt = neuraltide.config.maybe_check_numerics(d_dg_s_dt, 'SecondOrder d(dg_s)/dt NaN')

        return [d_g_s_dt, d_dg_s_dt]

    def compute_current(
        self,
        state: StateList,
        pre_firing_rate: TensorType,
        post_voltage: TensorType,
    ) -> Dict[str, TensorType]:
        dtype = neuraltide.config.get_dtype()
        gsyn_max = tf.cast(self.gsyn_max, dtype)
        e_r = tf.cast(self.e_r, dtype)

        g_s = state[0]

        post_v_flat = tf.reshape(post_voltage, [-1])
        I_pair = gsyn_max * g_s * (e_r - post_v_flat)
        I_syn = tf.reduce_sum(I_pair, axis=0, keepdims=True)
        g_syn = tf.reduce_sum(gsyn_max * g_s, axis=0, keepdims=True)

        return {'I_syn': I_syn, 'g_syn': g_syn}

    def adjoint_derivatives(
        self,
        adjoint_state: StateList,
        state: StateList,
        pre_firing_rate: TensorType,
        post_voltage: TensorType,
    ) -> StateList:
        lam_g, lam_dg = adjoint_state

        dtype = neuraltide.config.get_dtype()
        tau1 = tf.maximum(tf.cast(self.tau1, dtype), 1e-6)
        tau2 = tf.maximum(tf.cast(self.tau2, dtype), 1e-6)
        tau12 = tau1 * tau2
        t12_sum = tau1 + tau2

        dlam_g = lam_dg / tau12
        dlam_dg = -lam_g + t12_sum / tau12 * lam_dg

        dlam_g = neuraltide.config.maybe_check_numerics(dlam_g, 'SecondOrder adjoint dlam_g NaN')
        dlam_dg = neuraltide.config.maybe_check_numerics(dlam_dg, 'SecondOrder adjoint dlam_dg NaN')

        return [dlam_g, dlam_dg]

    def parameter_jacobian(
        self,
        param_name: str,
        state: StateList,
        pre_firing_rate: TensorType,
        post_voltage: TensorType,
    ) -> TensorType:
        """
        ∂derivatives/∂param для синапса второго порядка.

        Forward: f₂ = (s - (τ₁+τ₂)·dg_s - g_s) / (τ₁·τ₂)

        ∂f₂/∂τ₁ = -f₂/τ₁,  ∂f₂/∂τ₂ = -f₂/τ₂

        Adjoint solver: grad = λ₁·∂f₁/∂τ + λ₂·∂f₂/∂τ = -λ₂·f₂/τ_k

        Возвращается [n_pre, n_post] — вклад от dg_s (f₁ не зависит от τ).
        """
        g_s, dg_s = state

        if param_name in ('tau1', 'tau2'):
            dtype = neuraltide.config.get_dtype()
            tau1 = tf.maximum(tf.cast(self.tau1, dtype), 1e-6)
            tau2 = tf.maximum(tf.cast(self.tau2, dtype), 1e-6)
            firing_probs_T = tf.transpose(pre_firing_rate / 1000.0)
            s_input = tf.cast(self.pconn, dtype) * firing_probs_T
            tau12 = tau1 * tau2
            t12_sum = tau1 + tau2

            dg_s_ddot = (s_input - t12_sum * dg_s - g_s) / tau12

            if param_name == 'tau1':
                dtau = -dg_s_ddot / tau1
            else:
                dtau = -dg_s_ddot / tau2

            return neuraltide.config.maybe_check_numerics(
                dtau, f'SecondOrder parameter_jacobian {param_name} NaN')
        else:
            return tf.zeros_like(g_s)

    def compute_current_state_vjp(
        self,
        lam_I: TensorType,
        lam_g: TensorType,
        state: StateList,
        pre_firing_rate: TensorType,
        post_voltage: TensorType,
    ) -> StateList:
        g_s, dg_s = state

        dtype = neuraltide.config.get_dtype()
        gsyn_max = tf.cast(self.gsyn_max, dtype)
        e_r = tf.cast(self.e_r, dtype)

        lam_I_sum = tf.reduce_sum(lam_I, axis=0)
        lam_g_sum = tf.reduce_sum(lam_g, axis=0)

        dg_s_contrib = gsyn_max * (lam_I_sum * (e_r - post_voltage) + lam_g_sum)

        return [dg_s_contrib, tf.zeros_like(dg_s)]

    def compute_current_param_grad(
        self,
        lam_I: TensorType,
        lam_g: TensorType,
        state: StateList,
        pre_firing_rate: TensorType,
        post_voltage: TensorType,
    ) -> Dict[str, TensorType]:
        g_s, dg_s = state

        dtype = neuraltide.config.get_dtype()
        e_r = tf.cast(self.e_r, dtype)

        lam_I_sum = tf.reduce_sum(lam_I, axis=0)
        lam_g_sum = tf.reduce_sum(lam_g, axis=0)

        if self.gsyn_max.trainable:
            dgsyn_max = g_s * (lam_I_sum * (e_r - post_voltage) + lam_g_sum)
        else:
            dgsyn_max = tf.zeros_like(g_s)

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
            'tau1': {
                'shape': (self.n_pre, self.n_post),
                'trainable': self.tau1.trainable,
                'constraint': 'MinMaxConstraint',
                'units': 'ms',
            },
            'tau2': {
                'shape': (self.n_pre, self.n_post),
                'trainable': self.tau2.trainable,
                'constraint': 'MinMaxConstraint',
                'units': 'ms',
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
