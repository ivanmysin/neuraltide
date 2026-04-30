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

        dtype = neuraltide.config.get_dtype()
        pconn = tf.cast(self.pconn, dtype)
        tau_d = tf.maximum(tf.cast(self.tau_d, dtype), 1e-6)
        tau_f = tf.maximum(tf.cast(self.tau_f, dtype), 1e-6)
        tau_r = tf.maximum(tf.cast(self.tau_r, dtype), 1e-6)
        Uinc = tf.cast(self.Uinc, dtype)

        firing_probs_T = tf.transpose(pre_firing_rate / 1000.0)
        s_input = pconn * firing_probs_T

        dA_dt = -A / tau_d + U * R * s_input

        dU_dt = -U / tau_f + Uinc * (1.0 - U) * s_input

        dR_dt = (1.0 - R - A) / tau_r - U * R * s_input

        dR_dt = tf.debugging.check_numerics(dR_dt, 'TsodyksMarkram dR/dt NaN')
        dU_dt = tf.debugging.check_numerics(dU_dt, 'TsodyksMarkram dU/dt NaN')
        dA_dt = tf.debugging.check_numerics(dA_dt, 'TsodyksMarkram dA/dt NaN')

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
        dtype = neuraltide.config.get_dtype()
        gsyn_max = tf.cast(self.gsyn_max, dtype)
        pconn = tf.cast(self.pconn, dtype)
        e_r = tf.cast(self.e_r, dtype)

        if len(state) > 0:
            R_new, U_new, A_new = state
            g_eff = gsyn_max * A_new
        else:
            firing_probs_T = tf.transpose(pre_firing_rate / 1000.0)
            FRpre_normed = pconn * firing_probs_T
            g_eff = gsyn_max * FRpre_normed

        post_v_flat = tf.reshape(post_voltage, [-1])
        I_pair = g_eff * (e_r - post_v_flat)
        I_syn = tf.reduce_sum(I_pair, axis=0, keepdims=True)
        g_syn = tf.reduce_sum(g_eff, axis=0, keepdims=True)

        return {'I_syn': I_syn, 'g_syn': g_syn}

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
