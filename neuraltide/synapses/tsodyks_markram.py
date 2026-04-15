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

        dtype = neuraltide.config.get_dtype()
        self.gsyn_max = tf.cast(self.gsyn_max, dtype)
        self.tau_f = tf.cast(self.tau_f, dtype)
        self.tau_d = tf.cast(self.tau_d, dtype)
        self.tau_r = tf.cast(self.tau_r, dtype)
        self.Uinc = tf.cast(self.Uinc, dtype)
        self.pconn = tf.cast(self.pconn, dtype)
        self.e_r = tf.cast(self.e_r, dtype)

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

        tau1r = tf.where(
            tf.math.abs(self.tau_d - self.tau_r) > 1e-13,
            self.tau_d / (self.tau_d - self.tau_r),
            1e-13)


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

    def adjoint_forward(
        self,
        adjoint_current: Dict[str, TensorType],
        pre_firing_rate: TensorType,
        post_voltage: TensorType,
        state: StateList,
    ) -> Tuple[Dict[str, TensorType], StateList]:
        """Улучшенная adjoint propagation для Tsodyks-Markram synapse.

        Учитывает:
        - Вклад λ_I и λ_g в adjoint по A (conductance)
        - Приближённое влияние на presynaptic rate (через released fraction)
        - Adjoint по состоянию [R, U, A] для следующего backward шага
        """
        R, U, A = state
        dtype = neuraltide.config.get_dtype()

        lambda_I = adjoint_current.get('I_syn', tf.zeros([1, self.n_post], dtype=dtype))
        lambda_g = adjoint_current.get('g_syn', tf.zeros([1, self.n_post], dtype=dtype))

        post_v = post_voltage if post_voltage is not None else tf.zeros_like(lambda_I)

        # 1. Adjoint по A (основной вклад в проводимость)
        dI_dA = self.gsyn_max * (self.e_r - tf.transpose(post_v))
        dg_dA = self.gsyn_max

        lambda_A = tf.transpose(
            tf.transpose(lambda_I) * dI_dA + tf.transpose(lambda_g) * dg_dA
        )

        # 2. Приближённый adjoint по presynaptic firing rate
        firing_probs_T = tf.transpose(pre_firing_rate * self.dt / 1000.0)
        FR_normed = self.pconn * firing_probs_T
        released = U * R * FR_normed

        lambda_pre = tf.transpose(
            tf.transpose(lambda_I) * released * (self.e_r - tf.transpose(post_v)) +
            tf.transpose(lambda_g) * released
        )

        # 3. Adjoint по состоянию [R, U, A]
        # Approximate λ_R and λ_U from released transmitter dynamics
        lambda_R = U * lambda_pre * self.pconn * (pre_firing_rate * self.dt / 1000.0)
        lambda_U = R * lambda_pre * self.pconn * (pre_firing_rate * self.dt / 1000.0)

        new_adjoint_state = [
            lambda_R,           # λ_R
            lambda_U,           # λ_U
            lambda_A            # λ_A — основной вклад
        ]

        # Return adjoint for presynaptic rate (to propagate to previous population)
        # and for synaptic state
        return {
            'I_syn': tf.zeros_like(lambda_I),
            'g_syn': tf.zeros_like(lambda_g),
            'pre_rate': lambda_pre,
        }, new_adjoint_state

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
