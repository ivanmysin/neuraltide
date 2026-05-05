import tensorflow as tf
from typing import Dict, Tuple, List

import neuraltide
import neuraltide.config
from neuraltide.core.base import SynapseModel
from neuraltide.core.types import TensorType, StateList


class TsodyksMarkramSynapse(SynapseModel):
    """
    Синапс Цойдокса-Маркрама с кратковременной пластичностью (STP).

    Состояние: [R, U, A], каждое shape [n_pre, n_post].

    Уравнения (аналитическое решение между шагами):
        a_  = A * exp(-dt/tau_d)
        r_  = 1 + (R - 1 + tau_d/(tau_d - tau_r)*A) * exp(-dt/tau_r)
              - tau_d/(tau_d - tau_r)*A
        u_  = U * exp(-dt/tau_f)
        R_new = r_ - U * r_ * FRpre_normed
        U_new = u_ + Uinc*(1 - u_)*FRpre_normed
        A_new = a_ + U * r_ * FRpre_normed
    """

    def __init__(self, n_pre: int, n_post: int, dt: float, params: dict, **kwargs):
        super().__init__(n_pre=n_pre, n_post=n_post, dt=dt, **kwargs)

        self.gsyn_max = self._make_param(params, 'gsyn_max')
        self.tau_f = self._make_param(params, 'tau_f')
        self.tau_d = self._make_param(params, 'tau_d')
        self.tau_r = self._make_param(params, 'tau_r')
        self.Uinc = self._make_param(params, 'Uinc')
        self.pconn = self._make_param(params, 'pconn')
        self.e_r = self._make_param(params, 'e_r')

        self.state_size = [
            tf.TensorShape([n_pre, n_post]),
            tf.TensorShape([n_pre, n_post]),
            tf.TensorShape([n_pre, n_post]),
        ]

    def get_initial_state(self, batch_size: int = 1) -> StateList:
        dtype = neuraltide.config.get_dtype()
        return [
            tf.ones([self.n_pre, self.n_post], dtype=dtype),
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
        R, U, A = state



        firing_probs = dt * pre_firing_rate / 1000.0
        firing_probs_T = tf.transpose(firing_probs)
        FRpre_normed = self.pconn * firing_probs_T

        tau1r = tf.math.divide_no_nan(self.tau_d, self.tau_d - self.tau_r)


        exp_d = tf.exp(-dt / self.tau_d)
        exp_f = tf.exp(-dt / self.tau_f)
        exp_r = tf.exp(-dt / self.tau_r)

        a_ = A * exp_d
        r_ = 1.0 + (R - 1.0 + tau1r * A) * exp_r - tau1r * A
        u_ = U * exp_f

        released = U * r_ * FRpre_normed

        U_new = u_ + self.Uinc * (1.0 - u_) * FRpre_normed
        A_new = a_ + released
        R_new = r_ - released

        g_eff = self.gsyn_max * A_new
        post_v_flat = tf.reshape(post_voltage, [self.n_post])
        I_pair = g_eff * (self.e_r - post_v_flat)
        I_syn = tf.reduce_sum(I_pair, axis=0, keepdims=True)
        g_syn = tf.reduce_sum(g_eff, axis=0, keepdims=True)

        return ({'I_syn': I_syn, 'g_syn': g_syn}, [R_new, U_new, A_new])

    def derivatives(
        self,
        state: StateList,
        pre_firing_rate: TensorType,
        post_voltage: TensorType,
    ) -> StateList:
        """
        Compute derivatives for the Tsodyks-Markram short-term plasticity dynamics.

        Differential equations:
            dA/dt = -A/tau_d + U * R * s_input
            dU/dt = -U/tau_f + Uinc * (1 - U) * s_input
            dR/dt = (1 - R)/tau_r - U * R * s_input

        where s_input = pconn * firing_probability (converted to rate)
        """
        R, U, A = state

        # dtype = neuraltide.config.get_dtype()
        # pconn = tf.cast(self.pconn, dtype)
        # tau_d = tf.maximum(tf.cast(self.tau_d, dtype), 1e-6)
        # tau_f = tf.maximum(tf.cast(self.tau_f, dtype), 1e-6)
        # tau_r = tf.maximum(tf.cast(self.tau_r, dtype), 1e-6)
        # Uinc = tf.cast(self.Uinc, dtype)

        firing_probs_T = tf.transpose(pre_firing_rate / 1000.0)
        s_input = self.pconn * firing_probs_T

        dA_dt = -A / self.tau_d + U * R * s_input

        dU_dt = -U / self.tau_f + self.Uinc * (1.0 - U) * s_input

        dR_dt = (1.0 - R - A) / self.tau_r - U * R * s_input

        # dR_dt = neuraltide.config.maybe_check_numerics(dR_dt, 'TsodyksMarkram dR/dt NaN')
        # dU_dt = neuraltide.config.maybe_check_numerics(dU_dt, 'TsodyksMarkram dU/dt NaN')
        # dA_dt = neuraltide.config.maybe_check_numerics(dA_dt, 'TsodyksMarkram dA/dt NaN')

        return [dR_dt, dU_dt, dA_dt]

    def compute_current(
        self,
        state: StateList,
        pre_firing_rate: TensorType,
        post_voltage: TensorType,
    ) -> Dict[str, TensorType]:
        """
        Compute synaptic current and conductance from state.

        Used after numerical integration to compute currents.
        """


        if len(state) > 0:
            R_new, U_new, A_new = state
            g_eff = self.gsyn_max * A_new
        else:
            firing_probs_T = tf.transpose(pre_firing_rate / 1000.0)
            FRpre_normed = self.pconn * firing_probs_T
            g_eff = self.gsyn_max * FRpre_normed

        post_v_flat = tf.reshape(post_voltage, [-1])
        I_pair = g_eff * (self.e_r - post_v_flat)
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
        Compute adjoint derivatives for Tsodyks-Markram synapse.

        dλ_R/dt = (1/tau_r + U*s) * λ_R - U*s * λ_A
        dλ_U/dt = R*s * λ_R + (1/tau_f + Uinc*s) * λ_U - R*s * λ_A
        dλ_A/dt = λ_R / tau_r + λ_A / tau_d

        where s = pconn @ (pre_firing_rate_T / 1000.0)
        """
        R, U, A = state
        lam_R, lam_U, lam_A = adjoint_state
        Uinc = self.Uinc

        firing_probs_T = tf.transpose(pre_firing_rate / 1000.0)
        s = self.pconn * firing_probs_T

        dlam_R = (1.0 / self.tau_r + U * s) * lam_R - U * s * lam_A
        dlam_U = R * s * lam_R + (1.0 / self.tau_f + Uinc * s) * lam_U - R * s * lam_A
        dlam_A = lam_R / self.tau_r + lam_A / self.tau_d

        # dlam_R = neuraltide.config.maybe_check_numerics(dlam_R, 'TM adjoint dlam_R NaN')
        # dlam_U = neuraltide.config.maybe_check_numerics(dlam_U, 'TM adjoint dlam_U NaN')
        # dlam_A = neuraltide.config.maybe_check_numerics(dlam_A, 'TM adjoint dlam_A NaN')

        return [dlam_R, dlam_U, dlam_A]

    def parameter_jacobian(
        self,
        param_name: str,
        state: StateList,
        pre_firing_rate: TensorType,
        post_voltage: TensorType,
    ) -> TensorType:
        """
        ∂derivatives/∂param for Tsodyks-Markram.

        Only parameters appearing in derivatives() contribute:
            tau_f:  ∂(dU/dt)/∂tau_f = U / tau_f^2
            tau_d:  ∂(dA/dt)/∂tau_d = A / tau_d^2
            tau_r:  ∂(dR/dt)/∂tau_r = -(1-R-A) / tau_r^2
            Uinc:   ∂(dU/dt)/∂Uinc = (1-U) * s
            gsyn_max, pconn, e_r: not in derivatives → zero
        """
        R, U, A = state

        dtype = neuraltide.config.get_dtype()
        firing_probs_T = tf.transpose(pre_firing_rate / 1000.0)
        s = self.pconn * firing_probs_T

        if param_name == 'tau_f':
            return U / tf.square(tf.maximum(tf.cast(self.tau_f, dtype), 1e-6))
        elif param_name == 'tau_d':
            return A / tf.square(tf.maximum(tf.cast(self.tau_d, dtype), 1e-6))
        elif param_name == 'tau_r':
            return -(1.0 - R - A) / tf.square(tf.maximum(tf.cast(self.tau_r, dtype), 1e-6))
        elif param_name == 'Uinc':
            return (1.0 - U) * s
        else:
            return tf.zeros_like(R)

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

        I_syn[b,j] = sum_i gsyn_max[i,j] * A[i,j] * (e_r[i,j] - post_v[b,j])
        g_syn[b,j] = sum_i gsyn_max[i,j] * A[i,j]

        Only A contributes (R, U don't appear in current):
            ∂I/∂A[i,j] = gsyn_max[i,j] * (e_r[i,j] - post_v[b,j])
            ∂g/∂A[i,j] = gsyn_max[i,j]

        VJP: lam_A[i,j] += sum_b lam_I[b,j]*gsyn_max[i,j]*(e_r[i,j]-post_v[b,j])
                           + sum_b lam_g[b,j]*gsyn_max[i,j]
        """
        dtype = neuraltide.config.get_dtype()
        gsyn_max = tf.cast(self.gsyn_max, dtype)
        e_r = tf.cast(self.e_r, dtype)

        lam_I_sum = tf.reduce_sum(lam_I, axis=0)
        lam_g_sum = tf.reduce_sum(lam_g, axis=0)

        dA = gsyn_max * (lam_I_sum * (e_r - post_voltage) + lam_g_sum)

        return [tf.zeros_like(state[0]), tf.zeros_like(state[1]), dA]

    def compute_current_param_grad(
        self,
        lam_I: TensorType,
        lam_g: TensorType,
        state: StateList,
        pre_firing_rate: TensorType,
        post_voltage: TensorType,
    ) -> Dict[str, TensorType]:
        """
        Gradients for current-level parameters.

        dgsyn_max = A * (lam_I @ (e_r - post_v) + lam_g)
        (sum over batch dimension)
        """
        dtype = neuraltide.config.get_dtype()
        gsyn_max = tf.cast(self.gsyn_max, dtype)
        e_r = tf.cast(self.e_r, dtype)
        R, U, A = state

        lam_I_sum = tf.reduce_sum(lam_I, axis=0)
        lam_g_sum = tf.reduce_sum(lam_g, axis=0)

        if self.gsyn_max.trainable:
            dgsyn_max = A * (lam_I_sum * (e_r - post_voltage) + lam_g_sum)
        else:
            dgsyn_max = tf.zeros_like(A)

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
            'tau_f': {
                'shape': (self.n_pre, self.n_post),
                'trainable': self.tau_f.trainable,
                'constraint': 'MinMaxConstraint',
                'units': 'ms',
            },
            'tau_d': {
                'shape': (self.n_pre, self.n_post),
                'trainable': self.tau_d.trainable,
                'constraint': 'MinMaxConstraint',
                'units': 'ms',
            },
            'tau_r': {
                'shape': (self.n_pre, self.n_post),
                'trainable': self.tau_r.trainable,
                'constraint': 'MinMaxConstraint',
                'units': 'ms',
            },
            'Uinc': {
                'shape': (self.n_pre, self.n_post),
                'trainable': self.Uinc.trainable,
                'constraint': 'MinMaxConstraint',
                'units': 'dimensionless',
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
