import numpy as np
import tensorflow as tf

import neuraltide
import neuraltide.config
from neuraltide.core.types import TensorType, StateList, Dict
from neuraltide.populations.fokker_planck import FokkerPlanckPopulation


class BrunelFokkerPlanck(FokkerPlanckPopulation):
    """
    Популяция ЛИФ-нейронов в приближении Фоккера–Планка (Brunel, 2000,
    модель A: одинаковые нейроны).

    Уравнение (6) из статьи:

        τ_m · ∂P/∂t = (σ²(t)/2) · ∂²P/∂V²
                      + ∂/∂V [ (V − μ(t)) · P ]

    Граничные условия:
        - порог v_theta: поглощающая граница, P(θ, t) = 0;
          частота разрядов ν(t) — поток через порог:
              ν = −(σ²/(2·τ_m)) · ∂P/∂V |_{V=θ}   (уравнения (9), (12) из FP.md)
        - v_reset: реинжекция потока ν(t − τ_rp) в точку сброса
          (уравнение (10));
        - рефрактерная масса p_r(t) = ∫_{t−τ_rp}^{t} ν(u) du
          (уравнение (12));
        - нормировка: ∫P(V,t)dV + p_r(t) = 1.

    Связь с сетью (по умолчанию):
        μ(t) = mu_0 + I_syn(t)   — синаптический ток (мВ) добавляется
                                   к дрейфу;
        σ²(t) = sigma_ext²       — параметр.

    Дискретизация: консервативная конечно-объёмная схема на равномерной
    сетке [v_min, v_theta]; диффузия — центральные разности, дрейф —
    против потока (upwind). На последней грани (у порога) адвективный
    член отключён, что соответствует P(θ) = 0. Задержка рефрактерности
    ν(t − τ_rp) хранится в кольцевом буфере из n_delay = τ_rp/dt шагов;
    поэтому производная буфера корректна для метода Эйлера
    (рекомендуется EulerIntegrator или HeunIntegrator).

    Состояние: [P, p_r, nu_buf]:
        P:      [1, grid_size] — плотность вероятности на сетке V;
        p_r:    [1, 1]         — рефрактерная масса;
        nu_buf: [1, n_delay]   — история ν (1/мс), буфер задержки.

    Параметры (params):
        tau_m:      мембранная постоянная времени (мс);
        v_theta:    пороговый потенциал (мВ);
        v_reset:    потенциал сброса (мВ);
        tau_rp:     рефрактерный период (мс);
        mu_0:       средний дрейф в покое (мВ);
        sigma_ext:  стандартное отклонение шума (мВ).

    Пример:
        pop = BrunelFokkerPlanck(
            dt=0.5,
            params={
                'tau_m':      {'value': 20.0, 'trainable': False},
                'v_theta':    {'value': 20.0, 'trainable': False},
                'v_reset':    {'value': 10.0, 'trainable': False},
                'tau_rp':     {'value': 2.0,  'trainable': False},
                'mu_0':       {'value': 20.0, 'trainable': True,
                               'min': 0.0, 'max': 40.0},
                'sigma_ext':  {'value': 2.0,  'trainable': True,
                               'min': 0.1, 'max': 6.0},
            },
            grid_size=300,
            v_min=-80.0,
        )
    """

    def __init__(
        self,
        dt: float,
        params: dict,
        grid_size: int = 300,
        v_min: float = -100.0,
        name: str = "brunel_fokker_planck",
        **kwargs,
    ):
        self._params = params
        self._validate_params(params)

        # Порог задаётся параметром, поэтому сетка строится до super().__init__
        v_theta = float(
            params["v_theta"]["value"]
            if isinstance(params["v_theta"], dict)
            else params["v_theta"]
        )
        super().__init__(
            n_units=1,
            dt=dt,
            grid_size=grid_size,
            v_min=v_min,
            v_max=v_theta,
            name=name,
            **kwargs,
        )

        self.tau_m = self._make_param(self._params, "tau_m")
        self.v_theta = self._make_param(self._params, "v_theta")
        self.v_reset = self._make_param(self._params, "v_reset")
        self.tau_rp = self._make_param(self._params, "tau_rp")
        self.mu_0 = self._make_param(self._params, "mu_0")
        self.sigma_ext = self._make_param(self._params, "sigma_ext")

        # Индекс узла сетки, ближайшего к v_reset
        reset_v = self._params["v_reset"]
        reset_v = reset_v["value"] if isinstance(reset_v, dict) else reset_v
        self._reset_index = int(round((float(reset_v) - v_min) / self.dV))
        self._reset_index = min(max(self._reset_index, 0), grid_size - 2)

        # Длина буфера задержки рефрактерности (в шагах dt)
        rp = self._params["tau_rp"]
        rp = rp["value"] if isinstance(rp, dict) else rp
        self._n_delay = max(1, int(round(float(rp) / dt)))

        # Внутреннее субшагивение для устойчивости явной конечно-объёмной
        # схемы: ограничения Куранта по диффузии и по дрейфу (худший случай
        # |v − μ| ≤ span при μ внутри сетки).
        tau_v = float(
            self._params["tau_m"]["value"]
            if isinstance(self._params["tau_m"], dict)
            else self._params["tau_m"]
        )
        sig_v = float(
            self._params["sigma_ext"]["value"]
            if isinstance(self._params["sigma_ext"], dict)
            else self._params["sigma_ext"]
        )
        span = v_theta - v_min
        dt_diff = tau_v * self.dV * self.dV / (sig_v * sig_v) if sig_v > 0 else dt
        dt_adv = tau_v * self.dV / span
        dt_sub = 0.9 * min(dt, dt_diff, dt_adv)
        self._n_sub = max(1, int(np.ceil(dt / dt_sub)))
        self._dt_sub = dt / self._n_sub

        self.state_size = [
            tf.TensorShape([1, self.grid_size]),
            tf.TensorShape([1, 1]),
            tf.TensorShape([1, self._n_delay]),
        ]

        dtype = neuraltide.config.get_dtype()
        self._v_grid = self._v_grid_or_linspace(dtype)
        self._inject_mask = tf.one_hot(self._reset_index, self.grid_size, dtype=dtype)[
            tf.newaxis, :
        ]

    @staticmethod
    def _validate_params(params: dict) -> None:
        required = ["tau_m", "v_theta", "v_reset", "tau_rp", "mu_0", "sigma_ext"]
        for name in required:
            if name not in params:
                raise ValueError(
                    f"BrunelFokkerPlanck: required parameter '{name}' "
                    f"not found in params."
                )

    def _sigma2_over_2tau(self, dtype) -> TensorType:
        sigma = tf.cast(self.sigma_ext[0], dtype)
        tau = tf.cast(self.tau_m[0], dtype)
        return sigma * sigma / (2.0 * tau)

    def get_initial_state(self, batch_size: int = 1) -> StateList:
        dtype = neuraltide.config.get_dtype()
        mu0 = tf.cast(self.mu_0[0], dtype)
        sigma = tf.cast(self.sigma_ext[0], dtype)
        # Центр начального распределения — внутри сетки, даже если
        # mu_0 > v_theta (режим выше порога)
        center = tf.minimum(mu0, tf.cast(self.v_theta[0], dtype) - 3.0 * sigma)
        # Per-sample P [1, grid] -> broadcast на батч.
        P = self._gaussian_initial_state(center, sigma)
        P = tf.broadcast_to(P, [batch_size, self.grid_size])
        p_r = tf.zeros([batch_size, 1], dtype=dtype)
        nu_buf = tf.zeros([batch_size, self._n_delay], dtype=dtype)
        return [P, p_r, nu_buf]

    def _threshold_flux_ms(self, P: TensorType, dtype: tf.DType) -> TensorType:
        """Поток через порог ν в 1/мс: (σ²/2τ)·P(θ−dV)/dV, P(θ)=0."""
        P_b = tf.concat([P[..., :-1], tf.zeros_like(P[..., -1:])], axis=-1)
        return self._sigma2_over_2tau(dtype) * P_b[..., -2:-1] / self.dV

    def derivatives(
        self,
        state: StateList,
        total_synaptic_input: Dict[str, TensorType],
    ) -> StateList:
        P, p_r, nu_buf = state
        I_syn = total_synaptic_input["I_syn"]

        dtype = neuraltide.config.get_dtype()
        dV = tf.constant(self.dV, dtype=dtype)
        dt = tf.constant(self.dt, dtype=dtype)
        h = tf.constant(self._dt_sub, dtype=dtype)
        tau = tf.cast(self.tau_m[0], dtype)
        # Синаптический вход per-batch: [B, 1] -> [B]
        mu = tf.cast(self.mu_0[0], dtype) + tf.cast(I_syn[:, 0], dtype)

        nu_delayed = nu_buf[:, 0:1]  # ν(t − τ_rp), 1/мс

        # Внутренние субшаги явной схемы (устойчивость по Куранту).
        # Клип P ≥ 0 после каждого субшага: без этого центральная
        # диффузия даёт маленькие отрицательные просадки у порога,
        # а unclipped поток при max(flux, 0) создаёт массу.
        # Удаление за шаг считается точно — суммой потоков через
        # порог по субшагам — для точной бухгалтерии массы в p_r.
        P_work = P
        nu_acc = tf.zeros_like(nu_delayed)
        for _ in range(self._n_sub):
            P_work, nu_sub = self._euler_substep(P_work, mu, tau, nu_delayed, dV, h)
            P_work = tf.maximum(P_work, 0.0)
            nu_acc += nu_sub * h
        nu_now = nu_acc / dt

        dP = (P_work - P) / dt

        dp_r = nu_now - nu_delayed

        # Сдвиг буфера задержки на один шаг (точно для Эйлера)
        buf_target = tf.concat([nu_buf[:, 1:], nu_now], axis=1)
        dbuf = (buf_target - nu_buf) / dt

        return [dP, dp_r, dbuf]

    def _euler_substep(
        self,
        P: TensorType,
        mu: TensorType,
        tau: TensorType,
        nu_delayed: TensorType,
        dV: TensorType,
        h: TensorType,
    ):
        """Один явный субшаг консервативной конечно-объёмной схемы.

        Returns:
            (P_new, nu_sub): новая плотность и поток через порог
                за этот субшаг (1/мс).
        """
        dtype = neuraltide.config.get_dtype()

        # Поглощающая граница на пороге
        P_b = tf.concat([P[:, :-1], tf.zeros_like(P[:, -1:])], axis=1)

        # Потоки на гранях: диффузия (центральные) + дрейф (upwind).
        # Физическая скорость адвекции −(V−μ)/τ (поток к μ), поэтому
        # знак дрейфового потока отрицательный. μ per-batch: [B] -> [B, 1].
        dP_face = (P_b[:, 1:] - P_b[:, :-1]) / dV  # [B, M-1]
        v_face = 0.5 * (self._v_grid[:, 1:] + self._v_grid[:, :-1])  # [1, M-1]
        diff_flux = -self._sigma2_over_2tau(dtype) * dP_face
        drift = (v_face - mu[:, tf.newaxis]) / tau  # [B, M-1]
        upwind_P = tf.where(drift >= 0.0, P_b[:, 1:], P_b[:, :-1])
        adv_flux = -drift * upwind_P
        # На грани у порога адвекция отключена (P(θ) = 0)
        adv_flux = tf.concat(
            [adv_flux[:, :-1], tf.zeros_like(adv_flux[:, -1:])], axis=1
        )
        J = diff_flux + adv_flux  # [B, M-1]
        nu_sub = J[:, -1:]  # 1/мс

        dP = tf.concat(
            [
                -J[:, 0:1] / dV,
                -(J[:, 1:] - J[:, :-1]) / dV,
                -P[:, -1:] / h,  # поглощающий узел
            ],
            axis=1,
        )
        dP = dP + nu_delayed / dV * self._inject_mask

        return P + h * dP, nu_sub

    def get_firing_rate(self, state: StateList) -> TensorType:
        dtype = neuraltide.config.get_dtype()
        nu_ms = self._threshold_flux_ms(state[0], dtype)
        return 1000.0 * nu_ms  # 1/мс → Гц, shape [..., 1]

    def observables(self, state: StateList) -> Dict[str, TensorType]:
        dtype = neuraltide.config.get_dtype()
        P = state[0]
        dV = tf.constant(self.dV, dtype=dtype)
        # Масса и средний V — per-batch (по последней оси сетки).
        mass = tf.reduce_sum(P, axis=-1) * dV + state[1][:, 0]  # [B]
        v_mean = tf.reduce_sum(self._v_grid * P, axis=-1) * dV / tf.maximum(mass, 1e-9)
        return {
            "firing_rate": self.get_firing_rate(state),
            "v_mean": tf.reshape(v_mean, [-1, 1]),
            "p_refractory": state[1],
        }

    @property
    def parameter_spec(self) -> Dict[str, Dict[str, any]]:
        return {
            "tau_m": {
                "shape": (1,),
                "trainable": self.tau_m.trainable,
                "constraint": self._constraint_name(self.tau_m),
                "units": "ms",
            },
            "v_theta": {
                "shape": (1,),
                "trainable": self.v_theta.trainable,
                "constraint": self._constraint_name(self.v_theta),
                "units": "mV",
            },
            "v_reset": {
                "shape": (1,),
                "trainable": self.v_reset.trainable,
                "constraint": self._constraint_name(self.v_reset),
                "units": "mV",
            },
            "tau_rp": {
                "shape": (1,),
                "trainable": self.tau_rp.trainable,
                "constraint": self._constraint_name(self.tau_rp),
                "units": "ms",
            },
            "mu_0": {
                "shape": (1,),
                "trainable": self.mu_0.trainable,
                "constraint": self._constraint_name(self.mu_0),
                "units": "mV",
            },
            "sigma_ext": {
                "shape": (1,),
                "trainable": self.sigma_ext.trainable,
                "constraint": self._constraint_name(self.sigma_ext),
                "units": "mV",
            },
        }

    def _constraint_name(self, var: tf.Variable) -> str:
        if var.constraint is not None:
            return var.constraint.__class__.__name__
        return None
