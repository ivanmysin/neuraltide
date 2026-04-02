import tensorflow as tf
from typing import Dict, Tuple

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

        dtype = neuraltide.config.get_dtype()
        gsyn_max = tf.cast(self.gsyn_max, dtype)
        tau_f = tf.cast(self.tau_f, dtype)
        tau_d = tf.cast(self.tau_d, dtype)
        tau_r = tf.cast(self.tau_r, dtype)
        Uinc = tf.cast(self.Uinc, dtype)
        pconn = tf.cast(self.pconn, dtype)
        e_r = tf.cast(self.e_r, dtype)

        firing_probs = dt * pre_firing_rate / 1000.0
        firing_probs_T = tf.transpose(firing_probs)
        FRpre_normed = pconn * firing_probs_T

        tau1r = tf.where(
            tf.math.abs(tau_d - tau_r) > 1e-13,
            tau_d / (tau_d - tau_r),
            tf.constant(1e-13, dtype=dtype)
        )

        exp_d = tf.exp(-dt / tau_d)
        exp_f = tf.exp(-dt / tau_f)
        exp_r = tf.exp(-dt / tau_r)

        a_ = A * exp_d
        r_ = 1.0 + (R - 1.0 + tau1r * A) * exp_r - tau1r * A
        u_ = U * exp_f

        released = U * r_ * FRpre_normed

        U_new = u_ + Uinc * (1.0 - u_) * FRpre_normed
        A_new = a_ + released
        R_new = r_ - released

        g_eff = gsyn_max * A_new
        post_v_flat = tf.reshape(post_voltage, [self.n_post])
        I_pair = g_eff * (e_r - post_v_flat)
        I_syn = tf.reduce_sum(I_pair, axis=0, keepdims=True)
        g_syn = tf.reduce_sum(g_eff, axis=0, keepdims=True)

        return ({'I_syn': I_syn, 'g_syn': g_syn}, [R_new, U_new, A_new])

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
