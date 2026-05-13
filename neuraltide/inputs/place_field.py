import numpy as np
import tensorflow as tf
from typing import Any, Dict, Optional, Tuple

import neuraltide.config
from neuraltide.core.base import BaseInputGenerator
from neuraltide.core.types import TensorType


class PlaceFieldGenerator(BaseInputGenerator):
    """
    Генератор клеток места с эффектом фазовой прецессии в 2D пространстве.

    Все пространственные координаты — в сантиметрах (см).
    Все временные величины — в миллисекундах (мс).
    Частоты — в герцах (Гц).

    Каждая клетка места имеет Гауссову tuning-кривую (place field) в 2D.
    Скорость спайков:
        rate_i(t) = peak_rate * spatial * theta_inside   (внутри поля)
                   + bg_rate * (1 - spatial) * (1 - tmf + tmf * theta_outside)  (вне поля)
    где:
        spatial = exp(-0.5 * ((x-cx)/r)^2 + ((y-cy)/r)^2)
        theta_inside = exp(kappa * cos(2π·freq·t/1000 + ph0 + dphi)) / I0(kappa)
        theta_outside = exp(kappa * cos(2π·freq·t/1000 + ph_out)) / I0(kappa)
        dphi = -slope_rad * (x - cx)  — фазовая прецессия
        tmf — theta_modulation_factor (0→нет модуляции вне поля, 1→полная)

    Параметры (params):
        center_x:        центр place field по X (см). Скаляр или [n_units].
        center_y:        центр place field по Y (см). Скаляр или [n_units].
        radius:          ширина place field (σ гауссианы, см). Скаляр или [n_units].
        peak_rate:       пиковая частота в центре поля (Гц). Скаляр или [n_units].
        background_rate: фоновая частота вне поля (Гц). По умолчанию 0.0.
        theta_modulation_factor: сила тета-модуляции вне поля (0–1).
                               0 — нет модуляции, 1 — полная. По умолчанию 0.0.
        precession_slope: наклон фазовой прецессии (град/см).
                         Положительный → классическая прецессия.
        precession_init_phase: фаза тета-ритма в центре поля (градусы). По умолчанию 0.0.
        phase_outside:   фаза тета-ритма вне поля (градусы). По умолчанию 0.0.
        R:               R-value (0–1) для концентрации Von Mises тета-модуляции.
        freq:            частота тета-ритма (Гц).

    Параметры окружения:
        arena_size:  ((x_min, x_max), (y_min, y_max)) — границы арены (см).
                     По умолчанию ((-100, 100), (-100, 100)).
        arena_radius: радиус круглой арены (см). По умолчанию 100.0.

    Вход (extra_inputs):
        extra_inputs shape = [batch, T, n_cols], n_cols >= 2.
        Колонка 0: координата X (см).
        Колонка 1: координата Y (см).

    Пример:
        gen = PlaceFieldGenerator(
            dt=0.5,
            params={
                'center_x': [20.0, -30.0, 50.0],
                'center_y': [30.0, 10.0, -40.0],
                'radius': [30.0, 40.0, 35.0],
                'peak_rate': [20.0, 25.0, 15.0],
                'background_rate': [2.0, 3.0, 1.5],
                'precession_slope': [30.0, 25.0, 35.0],
                'precession_init_phase': [0.0, 90.0, 45.0],
                'phase_outside': 0.0,
                'theta_modulation_factor': 0.0,
                'R': 0.8,
                'freq': 8.0,
            },
        )
    """

    def __init__(
        self,
        dt: float,
        params: Dict[str, Any],
        arena_size: Tuple[Tuple[float, float], Tuple[float, float]] = (
            (-100.0, 100.0), (-100.0, 100.0)),
        arena_radius: float = 100.0,
        name: str = "place_field_generator",
        **kwargs,
    ):
        self._arena_size = (
            (float(arena_size[0][0]), float(arena_size[0][1])),
            (float(arena_size[1][0]), float(arena_size[1][1])),
        )
        self._arena_radius = float(arena_radius)
        self._dt = dt
        super().__init__(params=params, dt=dt, name=name, **kwargs)

        self.center_x = self._make_param(self._params, 'center_x')
        self.center_y = self._make_param(self._params, 'center_y')
        self.radius = self._make_param(self._params, 'radius')

        peak_rate_raw = self._params.get('peak_rate', 20.0)
        if isinstance(peak_rate_raw, dict):
            peak_rate_raw = peak_rate_raw['value']
        pk_np = np.atleast_1d(np.array(peak_rate_raw, dtype=np.float64))
        if pk_np.shape == ():
            pk_np = np.array([pk_np.item()], dtype=np.float64)

        self.peak_rate = self.add_weight(
            shape=(self.n_units,),
            initializer=tf.keras.initializers.Constant(pk_np),
            trainable=False,
            dtype=neuraltide.config.get_dtype(),
            name='peak_rate',
        )

        bg_raw = self._params.get('background_rate', 0.0)
        if isinstance(bg_raw, dict):
            bg_raw = bg_raw['value']
        bg_np = np.atleast_1d(np.array(bg_raw, dtype=np.float64))
        if bg_np.shape == ():
            bg_np = np.array([bg_np.item()], dtype=np.float64)
        self.background_rate = self.add_weight(
            shape=(self.n_units,),
            initializer=tf.keras.initializers.Constant(bg_np),
            trainable=False,
            dtype=neuraltide.config.get_dtype(),
            name='background_rate',
        )

        tmf_raw = self._params.get('theta_modulation_factor', 0.0)
        if isinstance(tmf_raw, dict):
            tmf_raw = tmf_raw['value']
        tmf_np = np.atleast_1d(np.array(tmf_raw, dtype=np.float64))
        if tmf_np.shape == ():
            tmf_np = np.array([tmf_np.item()], dtype=np.float64)
        self.theta_modulation_factor = self.add_weight(
            shape=(self.n_units,),
            initializer=tf.keras.initializers.Constant(tmf_np),
            trainable=False,
            dtype=neuraltide.config.get_dtype(),
            name='theta_modulation_factor',
        )

        slp_raw = self._params.get('precession_slope', 0.0)
        if isinstance(slp_raw, dict):
            slp_raw = slp_raw['value']
        slp_np = np.atleast_1d(np.array(slp_raw, dtype=np.float64))
        if slp_np.shape == ():
            slp_np = np.array([slp_np.item()], dtype=np.float64)
        slp_rad_per_cm = slp_np * (np.pi / 180.0)
        self.precession_slope_rad = self.add_weight(
            shape=(self.n_units,),
            initializer=tf.keras.initializers.Constant(slp_rad_per_cm),
            trainable=False,
            dtype=neuraltide.config.get_dtype(),
            name='precession_slope_rad',
        )

        ph_raw = self._params.get('precession_init_phase', 0.0)
        if isinstance(ph_raw, dict):
            ph_raw = ph_raw['value']
        ph_np = np.atleast_1d(np.array(ph_raw, dtype=np.float64))
        if ph_np.shape == ():
            ph_np = np.array([ph_np.item()], dtype=np.float64)
        ph_rad = ph_np * np.pi / 180.0
        self.precession_init_phase = self.add_weight(
            shape=(self.n_units,),
            initializer=tf.keras.initializers.Constant(ph_rad),
            trainable=False,
            dtype=neuraltide.config.get_dtype(),
            name='precession_init_phase',
        )

        pout_raw = self._params.get('phase_outside', 0.0)
        if isinstance(pout_raw, dict):
            pout_raw = pout_raw['value']
        pout_np = np.atleast_1d(np.array(pout_raw, dtype=np.float64))
        if pout_np.shape == ():
            pout_np = np.array([pout_np.item()], dtype=np.float64)
        pout_rad = pout_np * np.pi / 180.0
        self.phase_outside = self.add_weight(
            shape=(self.n_units,),
            initializer=tf.keras.initializers.Constant(pout_rad),
            trainable=False,
            dtype=neuraltide.config.get_dtype(),
            name='phase_outside',
        )

        raw_R = self._params.get('R')
        if isinstance(raw_R, dict):
            raw_R = raw_R.get('value', raw_R)
        R_array = np.array(raw_R)
        if R_array.ndim == 0:
            R_array = np.array([R_array.item()])
        R_array = np.broadcast_to(R_array, [self.n_units])
        kappa_np = self._r2kappa_np(R_array)
        self.kappa = tf.constant(kappa_np, dtype=neuraltide.config.get_dtype())
        self.i0_kappa = tf.constant(self._i0_np(kappa_np), dtype=neuraltide.config.get_dtype())
        self.freq = self._make_param(self._params, 'freq')


    @property
    def arena_size(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return self._arena_size

    @property
    def arena_radius(self) -> float:
        return self._arena_radius

    @property
    def R_value(self) -> float:
        return self._R_mean

    @property
    def kappa_value(self) -> float:
        return self._kappa_mean

    @staticmethod
    def _i0_np(kappa):
        return tf.math.bessel_i0(kappa)

    @staticmethod
    def _r2kappa_np(R):
        result = np.zeros_like(R, dtype=np.float64)
        mask1 = R < 0.53
        mask2 = (R >= 0.53) & (R < 0.85)
        mask3 = R >= 0.85
        result[mask1] = 2 * R[mask1] + R[mask1]**3 + (5/6) * R[mask1]**5
        result[mask2] = -0.4 + 1.39 * R[mask2] + 0.43 / (1 - R[mask2])
        result[mask3] = 1.0 / (3 * R[mask3] - 4 * R[mask3]**2 + R[mask3]**3)
        return result.astype(np.float32)

    def build(self, input_shape):
        super().build(input_shape)
        self._two_pi = 2.0 * np.pi
        self._ms_per_s = 1000.0

    def call(self, t: TensorType, extra_inputs: Optional[TensorType] = None) -> TensorType:
        """
        Args:
            t: время в мс, shape [batch, T, 1], [batch, T] или [T].
            extra_inputs: координаты (x, y) в см, shape [batch, T, n_cols]
                          или [batch, n_cols]. Колонка 0 = x, 1 = y.
                          Если None или n_cols < 2 — spatial=0 (нет позиции).

        Returns:
            tf.Tensor, shape [batch, T, n_units], в Гц.
        """
        t_rank = tf.rank(t)
        t = tf.cond(tf.equal(t_rank, 1),
                    lambda: tf.expand_dims(t, 0),
                    lambda: t)
        t = tf.cond(tf.equal(tf.rank(t), 2),
                    lambda: tf.expand_dims(t, -1),
                    lambda: t)

        batch_n = tf.shape(t)[0]
        T_n = tf.shape(t)[1]
        n_units = self.n_units

        def _bcast(p):
            return tf.broadcast_to(tf.reshape(p, [1, 1, n_units]),
                                   [batch_n, T_n, n_units])

        cx = _bcast(self.center_x)
        cy = _bcast(self.center_y)
        r = _bcast(self.radius)
        rate_peak = _bcast(self.peak_rate)
        bg_rate = _bcast(self.background_rate)
        slp_rad = _bcast(self.precession_slope_rad)
        ph = _bcast(self.precession_init_phase)
        ph_out = _bcast(self.phase_outside)
        kap = _bcast(self.kappa)
        i0k = _bcast(self.i0_kappa)
        fq = _bcast(self.freq)

        two_pi = self._two_pi
        ms_per_s = self._ms_per_s

        if extra_inputs is not None:
            extra_rank = tf.rank(extra_inputs)
            extra = tf.cond(tf.equal(extra_rank, 2),
                            lambda: tf.expand_dims(extra_inputs, 1),
                            lambda: extra_inputs)
            extra_cols = tf.shape(extra)[-1]
        else:
            extra_cols = tf.constant(0, dtype=tf.int32)

        def _has_position():
            e = extra
            e = tf.cond(tf.not_equal(tf.shape(e)[1], T_n),
                        lambda: tf.tile(e, [1, T_n, 1]),
                        lambda: e)
            px = tf.broadcast_to(e[:, :, 0:1], [batch_n, T_n, 1])
            py = tf.broadcast_to(e[:, :, 1:2], [batch_n, T_n, 1])
            dx = (px - cx) / r
            dy = (py - cy) / r
            spatial = tf.exp(-0.5 * (dx ** 2 + dy ** 2))
            dphi = -slp_rad * (px - cx)  # приращение фазы зависит только от x !!!
            return px, py, spatial, dphi

        def _no_position():
            spatial = tf.zeros([batch_n, T_n, n_units], dtype=tf.float32)
            dphi = tf.zeros([batch_n, T_n, n_units], dtype=tf.float32)
            px = tf.zeros([batch_n, T_n, 1], dtype=tf.float32)
            py = tf.zeros([batch_n, T_n, 1], dtype=tf.float32)
            return px, py, spatial, dphi

        pos_x, pos_y, spatial, dphi = tf.cond(
            tf.greater_equal(extra_cols, 2), _has_position, _no_position)

        theta_inside_arg = two_pi * fq * t / ms_per_s + ph + dphi
        theta_mod_inside = tf.exp(kap * tf.cos(theta_inside_arg)) / tf.maximum(i0k, 1e-5)

        theta_outside_arg = two_pi * fq * t / ms_per_s + ph_out
        theta_mod_outside = tf.exp(kap * tf.cos(theta_outside_arg)) / tf.maximum(i0k, 1e-5)

        inside_mean_rate = (rate_peak - bg_rate) / (bg_rate + 1)

        rate = ( inside_mean_rate * spatial * theta_mod_inside
                + bg_rate * (1.0 - spatial) * theta_mod_outside)

        rate = tf.nn.relu(rate)
        return rate

    @property
    def parameter_spec(self) -> Dict[str, Dict[str, Any]]:
        return {
            'center_x': {
                'shape': (self.n_units,),
                'trainable': self.center_x.trainable,
                'constraint': self._get_constraint_name(self.center_x),
                'units': 'cm',
            },
            'center_y': {
                'shape': (self.n_units,),
                'trainable': self.center_y.trainable,
                'constraint': self._get_constraint_name(self.center_y),
                'units': 'cm',
            },
            'radius': {
                'shape': (self.n_units,),
                'trainable': self.radius.trainable,
                'constraint': self._get_constraint_name(self.radius),
                'units': 'cm',
            },
            'peak_rate': {
                'shape': (self.n_units,),
                'trainable': False,
                'constraint': None,
                'units': 'Hz',
            },
            'background_rate': {
                'shape': (self.n_units,),
                'trainable': False,
                'constraint': None,
                'units': 'Hz',
            },
            'theta_modulation_factor': {
                'shape': (self.n_units,),
                'trainable': False,
                'constraint': None,
                'units': 'dimensionless',
            },
            'precession_slope': {
                'shape': (self.n_units,),
                'trainable': False,
                'constraint': None,
                'units': 'deg/cm',
            },
            'precession_init_phase': {
                'shape': (self.n_units,),
                'trainable': False,
                'constraint': None,
                'units': 'deg',
            },
            'phase_outside': {
                'shape': (self.n_units,),
                'trainable': False,
                'constraint': None,
                'units': 'deg',
            },
            'kappa': {
                'shape': (self.n_units,),
                'trainable': self.kappa.trainable,
                'constraint': self._get_constraint_name(self.kappa),
                'units': 'dimensionless',
            },
            'freq': {
                'shape': (self.n_units,),
                'trainable': self.freq.trainable,
                'constraint': self._get_constraint_name(self.freq),
                'units': 'Hz',
            },
        }

    def _get_constraint_name(self, var: tf.Variable) -> Optional[str]:
        if var.constraint is not None:
            return var.constraint.__class__.__name__
        return None
