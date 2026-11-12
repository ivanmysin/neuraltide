import tensorflow as tf
from typing import Dict, Tuple

import neuraltide
import neuraltide.config
from neuraltide.core.base import SynapseModel
from neuraltide.core.types import TensorType, StateList, cast_if_needed


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

        self.gsyn_max_nmda = self._make_param(params, "gsyn_max_nmda")
        self.tau1_nmda = self._make_param(params, "tau1_nmda")
        self.tau2_nmda = self._make_param(params, "tau2_nmda")
        self.Mgb = self._make_param(params, "Mgb")
        self.av_nmda = self._make_param(params, "av_nmda")
        self.pconn_nmda = self._make_param(params, "pconn_nmda")
        self.e_r_nmda = self._make_param(params, "e_r_nmda")
        self.v_ref = self._make_param(params, "v_ref")

        self.state_size = [
            tf.TensorShape([n_pre, n_post]),
            tf.TensorShape([n_pre, n_post]),
        ]

    def get_initial_state(self, batch_size: int = 1) -> StateList:
        dtype = neuraltide.config.get_dtype()
        return [
            tf.zeros([batch_size, self.n_pre, self.n_post], dtype=dtype),
            tf.zeros([batch_size, self.n_pre, self.n_post], dtype=dtype),
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
        _c = cast_if_needed
        gsyn_max_nmda = _c(self.gsyn_max_nmda, dtype)
        tau1_nmda = _c(self.tau1_nmda, dtype)
        tau2_nmda = _c(self.tau2_nmda, dtype)
        Mgb = _c(self.Mgb, dtype)
        av_nmda = _c(self.av_nmda, dtype)
        pconn_nmda = _c(self.pconn_nmda, dtype)
        e_r_nmda = _c(self.e_r_nmda, dtype)
        v_ref = _c(self.v_ref, dtype)

        firing_probs = dt * pre_firing_rate / 1000.0
        s_input = pconn_nmda * firing_probs[:, :, tf.newaxis]  # [B, n_pre, n_post]

        dgnmda_new = dgnmda + dt * (
            s_input - gnmda - (tau1_nmda + tau2_nmda) * dgnmda
        ) / (tau1_nmda * tau2_nmda)
        gnmda_new = gnmda + dt * dgnmda_new

        post_v = post_voltage[:, tf.newaxis, :]  # [B, 1, n_post]
        mg_block = 1.0 / (1.0 + Mgb * tf.exp(-av_nmda * (post_v - v_ref)))

        g_eff = gsyn_max_nmda * gnmda_new * mg_block
        I_pair = g_eff * (e_r_nmda - post_v)
        I_syn = tf.reduce_sum(I_pair, axis=-2)  # [B, n_post]
        g_syn = tf.reduce_sum(g_eff, axis=-2)

        return ({"I_syn": I_syn, "g_syn": g_syn}, [gnmda_new, dgnmda_new])

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
        _c = cast_if_needed
        pconn = _c(self.pconn_nmda, dtype)
        tau1 = tf.maximum(_c(self.tau1_nmda, dtype), 1e-6)
        tau2 = tf.maximum(_c(self.tau2_nmda, dtype), 1e-6)

        firing_probs = pre_firing_rate / 1000.0
        s_input = pconn * firing_probs[:, :, tf.newaxis]  # [B, n_pre, n_post]

        d_gnmda_dt = dgnmda

        d_dgnmda_dt = (s_input - gnmda - (tau1 + tau2) * dgnmda) / (tau1 * tau2)

        d_gnmda_dt = neuraltide.config.maybe_check_numerics(
            d_gnmda_dt, "NMDA d(gnmda)/dt NaN"
        )
        d_dgnmda_dt = neuraltide.config.maybe_check_numerics(
            d_dgnmda_dt, "NMDA d(dgnmda)/dt NaN"
        )

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
        _c = cast_if_needed
        gsyn_max_nmda = _c(self.gsyn_max_nmda, dtype)
        Mgb = _c(self.Mgb, dtype)
        av_nmda = _c(self.av_nmda, dtype)
        e_r_nmda = _c(self.e_r_nmda, dtype)
        v_ref = _c(self.v_ref, dtype)

        gnmda = state[0]

        post_v = post_voltage[:, tf.newaxis, :]  # [B, 1, n_post]
        mg_block = 1.0 / (1.0 + Mgb * tf.exp(-av_nmda * (post_v - v_ref)))

        g_eff = gsyn_max_nmda * gnmda * mg_block
        I_pair = g_eff * (e_r_nmda - post_v)
        I_syn = tf.reduce_sum(I_pair, axis=-2)  # [B, n_post]
        g_syn = tf.reduce_sum(g_eff, axis=-2)

        return {"I_syn": I_syn, "g_syn": g_syn}

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
        _c = cast_if_needed
        tau1 = tf.maximum(_c(self.tau1_nmda, dtype), 1e-6)
        tau2 = tf.maximum(_c(self.tau2_nmda, dtype), 1e-6)
        t12 = tau1 * tau2

        dlam_gnmda = lam_dgnmda / t12
        dlam_dgnmda = -lam_gnmda + (tau1 + tau2) / t12 * lam_dgnmda

        dlam_gnmda = neuraltide.config.maybe_check_numerics(
            dlam_gnmda, "NMDA adjoint dlam_gnmda NaN"
        )
        dlam_dgnmda = neuraltide.config.maybe_check_numerics(
            dlam_dgnmda, "NMDA adjoint dlam_dgnmda NaN"
        )

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
        VJP from I_syn/g_syn back to synapse state (per-batch-slot).

        I_syn[b,j] = sum_i gsyn_max[i,j] * gnmda[b,i,j] * mg_block[b,j] * (e_r - post_v[b,j])
        g_syn[b,j] = sum_i gsyn_max[i,j] * gnmda[b,i,j] * mg_block[b,j]

        Only gnmda contributes (dgnmda doesn't appear):
            dgnmda[b,i,j] = lam_I[b,j] * gsyn_max[i,j] * mg_block[b,j] * (e_r - post_v[b,j])
                          + lam_g[b,j] * gsyn_max[i,j] * mg_block[b,j]
        """
        gnmda, dgnmda = state

        dtype = neuraltide.config.get_dtype()
        _c = cast_if_needed
        gsyn_max = _c(self.gsyn_max_nmda, dtype)
        Mgb = _c(self.Mgb, dtype)
        av_nmda = _c(self.av_nmda, dtype)
        e_r_nmda = _c(self.e_r_nmda, dtype)
        v_ref = _c(self.v_ref, dtype)

        post_v = post_voltage[:, tf.newaxis, :]  # [B, 1, n_post]
        mg_block = 1.0 / (1.0 + Mgb * tf.exp(-av_nmda * (post_v - v_ref)))

        lam_I_b = lam_I[:, tf.newaxis, :]
        lam_g_b = lam_g[:, tf.newaxis, :]

        dgnmda_contrib = gsyn_max * mg_block * (lam_I_b * (e_r_nmda - post_v) + lam_g_b)

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
        Gradients for current-level parameters (only gsyn_max_nmda;
        per-batch-slot).
        """
        gnmda, dgnmda = state

        dtype = neuraltide.config.get_dtype()
        _c = cast_if_needed
        Mgb = _c(self.Mgb, dtype)
        av_nmda = _c(self.av_nmda, dtype)
        e_r_nmda = _c(self.e_r_nmda, dtype)
        v_ref = _c(self.v_ref, dtype)

        post_v = post_voltage[:, tf.newaxis, :]  # [B, 1, n_post]
        mg_block = 1.0 / (1.0 + Mgb * tf.exp(-av_nmda * (post_v - v_ref)))
        lam_I_b = lam_I[:, tf.newaxis, :]
        lam_g_b = lam_g[:, tf.newaxis, :]

        if self.gsyn_max_nmda.trainable:
            dgsyn_max = gnmda * mg_block * (lam_I_b * (e_r_nmda - post_v) + lam_g_b)
        else:
            dgsyn_max = tf.zeros_like(gnmda)

        return {"gsyn_max_nmda": dgsyn_max}

    @property
    def parameter_spec(self) -> Dict[str, Dict[str, any]]:
        return {
            "gsyn_max_nmda": {
                "shape": (self.n_pre, self.n_post),
                "trainable": self.gsyn_max_nmda.trainable,
                "constraint": "NonNegConstraint",
                "units": "mS/cm^2",
            },
            "tau1_nmda": {
                "shape": (self.n_pre, self.n_post),
                "trainable": False,
                "constraint": None,
                "units": "ms",
            },
            "tau2_nmda": {
                "shape": (self.n_pre, self.n_post),
                "trainable": False,
                "constraint": None,
                "units": "ms",
            },
            "Mgb": {
                "shape": (self.n_pre, self.n_post),
                "trainable": False,
                "constraint": None,
                "units": "mM",
            },
            "av_nmda": {
                "shape": (self.n_pre, self.n_post),
                "trainable": False,
                "constraint": None,
                "units": "mV^-1",
            },
            "pconn_nmda": {
                "shape": (self.n_pre, self.n_post),
                "trainable": False,
                "constraint": "UnitIntervalConstraint",
                "units": "dimensionless",
            },
            "e_r_nmda": {
                "shape": (self.n_pre, self.n_post),
                "trainable": False,
                "constraint": None,
                "units": "mV",
            },
            "v_ref": {
                "shape": (self.n_pre, self.n_post),
                "trainable": False,
                "constraint": None,
                "units": "mV",
            },
        }
