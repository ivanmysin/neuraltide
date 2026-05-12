import numpy as np
import tensorflow as tf
from typing import Any, Dict, Optional, Tuple

import neuraltide.config
from neuraltide.core.base import BaseInputGenerator
from neuraltide.core.types import TensorType


class PlaceFieldGenerator(BaseInputGenerator):
    """
    Генератор клеток места с эффектом фазовой прецессии в 2D пространстве.

    Каждая клетка места имеет Гауссову tuning-кривую (place field) в 2D.
    Скорость спайков состоит из трёх частей:
        rate_i(t) = bg_rate + mod_outside * VM(t, phi_outside)   — вне поля
        rate_i(t) = (bg_rate + peak_rate * spatial * mod_inside) * VM(t, phi_inside) — внутри поля
    где VM — Von Mises тета-модуляция, фаза phi сдвигается для фазовой прецессии.

    Параметры (params):
        center_x: центр place field по X (arena units). Скаляр или [n_units].
        center_y: центр place field по Y (arena units). Скаляр или [n_units].
        radius: ширина place field (scale param, arena units). Скаляр или [n_units].
        peak_rate: пиковая частота над фоном (Hz). Скаляр или [n_units].
        background_rate: фоновая частота (Hz). Скаляр или [n_units]. По умолчанию 0.0.
        theta_modulation_factor: множитель тета-модуляции вне поля.
            Если 0, тета-модуляции вне поля нет. Если 1, полная модуляция.
            Скаляр или [n_units]. По умолчанию 0.0.
        precession_slope: наклон фазовой прецессии
            (градусы / см). Положительный => классическая прецессия
            (фаза уходит вперёд при движении в place field).
            Преобразуется внутрь: slope_internal = precession_slope * dt_cm.
            Скаляр или [n_units].
        precession_init_phase: начальная фаза прецессии (градусы).
            Скаляр или [n_units]. По умолчанию 0.0. Преобразуется в радианы.
        R: R-value (0-1) для концентрации тета-модуляции.
            R=0 => равномерный фон, R=1 => пик в одной фазе. Скаляр или [n_units].
        freq: частота тета-ритма (Hz). Скаляр или [n_units].

    Входная последовательность (extra_inputs):
        extra_inputs shape = [batch, T, n_cols], n_cols >= 2.
        Дополнительный столбец 0: координата X животного (arena units).
        Дополнительный столбец 1: координата Y животного (arena units).

    Параметры окружения:
        arena_size: ((x_min, x_max), (y_min, y_max)) — границы арены
            в условных единицах (1 arena_unit = 1 м).
            По умолчанию ((-1, 1), (-1, 1)).
        arena_radius: радиус круглой арены в arena_units. По умолчанию 1.0.

    Пример:
        gen = PlaceFieldGenerator(
            dt=0.5,
            params={
                'center_x': [0.2, -0.3, 0.5],
                'center_y': [0.3, 0.1, -0.4],
                'radius': [0.3, 0.4, 0.35],
                'peak_rate': [20.0, 25.0, 15.0],
                'background_rate': [2.0, 3.0, 1.5],
                'precession_slope': [30.0, 25.0, 35.0],  # deg/cm
                'precession_init_phase': [0.0, 90.0, 45.0],  # degrees
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
        arena_size: Tuple[Tuple[float, float], Tuple[float, float]] = ((-1.0, 1.0), (-1.0, 1.0)),
        arena_radius: float = 1.0,
        name: str = "place_field_generator",
        **kwargs,
    ):
        # Convert arena_size units to cm assuming 1 arena_unit = 1 m = 100 cm
        # This way slope in deg/cm works with arena coordinates
        self._cm_per_arena_unit = 100.0
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

        # background_rate: default 0.0
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

        # theta_modulation_factor: outside field modulation strength
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

        # precession_slope in rad/arena_unit.
        # Input: deg/cm.  Internal: rad/m = slope_deg_per_cm * (π/180) * 100 cm/m.
        slp_raw = self._params.get('precession_slope', 0.0)
        if isinstance(slp_raw, dict):
            slp_raw = slp_raw['value']
        slp_np = np.atleast_1d(np.array(slp_raw, dtype=np.float64))
        if slp_np.shape == ():
            slp_np = np.array([slp_np.item()], dtype=np.float64)
        # Convert deg/cm → rad/arena_unit  (1 arena_unit = 1 m = 100 cm)
        slp_rad = slp_np * (np.pi / 180.0) * self._cm_per_arena_unit
        self.precession_slope_rad = self.add_weight(
            shape=(self.n_units,),
            initializer=tf.keras.initializers.Constant(slp_rad),
            trainable=False,
            dtype=neuraltide.config.get_dtype(),
            name='precession_slope_rad',
        )

        # precession_init_phase in radians
        ph_raw = self._params.get('precession_init_phase', 0.0)
        if isinstance(ph_raw, dict):
            ph_raw = ph_raw['value']
        ph_np = np.atleast_1d(np.array(ph_raw, dtype=np.float64))
        if ph_np.shape == ():
            ph_np = np.array([ph_np.item()], dtype=np.float64)
        # Input is in degrees, convert to radians
        ph_rad = ph_np * np.pi / 180.0
        self.precession_init_phase = self.add_weight(
            shape=(self.n_units,),
            initializer=tf.keras.initializers.Constant(ph_rad),
            trainable=False,
            dtype=neuraltide.config.get_dtype(),
            name='precession_init_phase',
        )

        raw_R = self._params.get('R')
        if isinstance(raw_R, dict):
            raw_R = raw_R.get('value', raw_R)
        R_array = np.array(raw_R)
        if R_array.ndim == 0:
            R_array = np.array([R_array.item()])
        R_array = np.broadcast_to(R_array, [self.n_units])

        kappa_np = self._r2kappa_np(R_array)
        i0_kappa_np = self._i0_np(kappa_np)

        self.kappa = self.add_weight(
            shape=(self.n_units,),
            initializer=tf.keras.initializers.Constant(kappa_np),
            trainable=False,
            dtype=neuraltide.config.get_dtype(),
            name='kappa',
        )
        self.i0_kappa = self.add_weight(
            shape=(self.n_units,),
            initializer=tf.keras.initializers.Constant(i0_kappa_np),
            trainable=False,
            dtype=neuraltide.config.get_dtype(),
            name='i0_kappa',
        )
        self.freq = self._make_param(self._params, 'freq')

    @property
    def arena_size(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return self._arena_size

    @property
    def arena_radius(self) -> float:
        return self._arena_radius

    @staticmethod
    def _i0_np(kappa):
        from scipy.special import i0
        return i0(kappa).astype(np.float64)

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

    def call(self, t: TensorType, extra_inputs: Optional[TensorType] = None) -> TensorType:
        """
        Args:
            t: current time in ms. shape = [batch, T, 1] or [batch, T] or [T].
            extra_inputs: extra data with shape [batch, T, n_extra_cols]
                          or [batch, n_extra_cols] (per-step from InputPopulation).
                          Column 0 = x, Column 1 = y. If n_extra_cols < 2,
                          use default circular trajectory.
                          If None, use default circular trajectory.

        Returns:
            tf.Tensor, shape = [batch, T, n_units], in Hz.
        """
        dtype = neuraltide.config.get_dtype()
        two_pi = tf.constant(2.0 * np.pi, dtype=dtype)

        t = tf.cast(t, dtype)
        # -- Normalize t to [batch, T, 1] --
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
        slp_rad = _bcast(self.precession_slope_rad)
        ph = _bcast(self.precession_init_phase)
        kap = _bcast(self.kappa)
        i0k = _bcast(self.i0_kappa)
        fq = _bcast(self.freq)
        bg_rate = _bcast(self.background_rate)
        tmf = _bcast(self.theta_modulation_factor)

        # -- Position from extra_inputs or default circular trajectory --
        if extra_inputs is None:
            extra_inputs = tf.zeros([batch_n, T_n, 0], dtype=dtype)

        extra = tf.cast(extra_inputs, dtype)
        extra = tf.cond(tf.equal(tf.rank(extra), 2),
                        lambda: tf.expand_dims(extra, 1),
                        lambda: extra)
        extra_cols = tf.shape(extra)[-1]

        def _from_extra():
            e = extra
            e = tf.cond(tf.not_equal(tf.shape(e)[1], T_n),
                        lambda: tf.tile(e, [1, T_n, 1]),
                        lambda: e)
            px = tf.broadcast_to(e[:, :, 0:1], [batch_n, T_n, 1])
            py = tf.broadcast_to(e[:, :, 1:2], [batch_n, T_n, 1])
            return px, py

        def _from_default():
            x_mid = tf.constant(
                (self._arena_size[0][0] + self._arena_size[0][1]) / 2.0, dtype=dtype)
            y_mid = tf.constant(
                (self._arena_size[1][0] + self._arena_size[1][1]) / 2.0, dtype=dtype)
            R_traj = tf.constant(0.8 * self._arena_radius, dtype=dtype)
            theta = (two_pi * t / tf.constant(1000.0, dtype=dtype)
                     * tf.constant(16.0, dtype=dtype))
            px = x_mid + R_traj * tf.cos(theta)
            py = y_mid + R_traj * tf.sin(theta)
            return px, py

        pos_x, pos_y = tf.cond(
            tf.greater_equal(extra_cols, 2), _from_extra, _from_default)

        # -- Gaussian place field --
        dx = (pos_x - cx) / r
        dy = (pos_y - cy) / r
        spatial = tf.exp(-0.5 * (dx ** 2 + dy ** 2))

        # -- Phase precession shift (rad) --
        # slope is rad/arena_unit, pos_x-cx is arena_units → product is radians
        dphi = -slp_rad * (pos_x - cx)

        # -- Theta modulation inside field (with precession) --
        theta_inside = (two_pi * fq * t
                        / tf.constant(1000.0, dtype=dtype) + ph + dphi)
        theta_mod_inside = (tf.exp(kap * tf.cos(theta_inside))
                            / tf.maximum(i0k, 1e-8))

        # -- Theta modulation outside field --
        theta_outside = (two_pi * fq * t
                         / tf.constant(1000.0, dtype=dtype) + ph)
        theta_mod_outside = (tf.exp(kap * tf.cos(theta_outside))
                              / tf.maximum(i0k, 1e-8))

        # -- Composite rate --
        rate_outside = tmf * theta_mod_outside
        rate = (bg_rate
                + spatial * (rate_peak * theta_mod_inside
                             + (1.0 - spatial) * rate_outside))

        rate = tf.nn.relu(rate)
        rate = tf.cast(rate, dtype)
        return rate

    @property
    def parameter_spec(self) -> Dict[str, Dict[str, Any]]:
        return {
            'center_x': {
                'shape': (self.n_units,),
                'trainable': self.center_x.trainable,
                'constraint': self._get_constraint_name(self.center_x),
                'units': 'arena_units',
            },
            'center_y': {
                'shape': (self.n_units,),
                'trainable': self.center_y.trainable,
                'constraint': self._get_constraint_name(self.center_y),
                'units': 'arena_units',
            },
            'radius': {
                'shape': (self.n_units,),
                'trainable': self.radius.trainable,
                'constraint': self._get_constraint_name(self.radius),
                'units': 'arena_units',
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

    def _get_constraint_name(self, var: tf.Variable) -> str:
        if var.constraint is not None:
            return var.constraint.__class__.__name__
        return None
