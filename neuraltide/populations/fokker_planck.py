import tensorflow as tf

import neuraltide
import neuraltide.config
from neuraltide.core.base import PopulationModel
from neuraltide.core.types import TensorType, StateList, Dict


class FokkerPlanckPopulation(PopulationModel):
    """
    Базовый класс для популяций с дискретизованным распределением P(V, t).

    Состояние: [P], shape [1, grid_size] — вектор вероятностей на сетке.

    Контракт: пользователь наследуется и реализует derivatives() с
    дискретизованным оператором Фоккера-Планка.

    get_firing_rate() возвращает поток через правую границу сетки
    (пороговый потенциал) по умолчанию.
    """

    def __init__(
        self,
        n_units: int,
        dt: float,
        grid_size: int = 100,
        v_min: float = -100.0,
        v_max: float = 50.0,
        **kwargs,
    ):
        super().__init__(n_units=n_units, dt=dt, **kwargs)
        self.grid_size = grid_size
        self.v_min = v_min
        self.v_max = v_max
        self.dV = (v_max - v_min) / (grid_size - 1)
        self.state_size = [tf.TensorShape([1, grid_size])]

    def get_boundary_flux(self, P: TensorType, dV: float) -> TensorType:
        """
        Вычисляет поток через правую границу (пороговый потенциал).

        J_out = -D * dP/dV|_{V=V_threshold}

        Args:
            P: вектор вероятностей shape [1, grid_size].
            dV: шаг сетки.

        Returns:
            scalar firing rate.
        """
        dP_dV = (P[:, -1] - P[:, -2]) / dV
        D = tf.constant(1.0, dtype=neuraltide.config.get_dtype())
        J_out = -D * dP_dV
        return J_out[:, tf.newaxis]

    def _gaussian_initial_state(
        self,
        center: TensorType,
        sigma: TensorType,
    ) -> TensorType:
        """
        Нормированное гауссово распределение на сетке V — валидное
        начальное состояние P (норма ∫P dV = 1, узел порога обнулён).

        Args:
            center: центр распределения (мВ), скалярный тензор.
            sigma: стандартное отклонение (мВ), скалярный тензор.

        Returns:
            tf.Tensor [1, grid_size].
        """
        dtype = neuraltide.config.get_dtype()
        v = tf.cast(self._v_grid_or_linspace(dtype), dtype)
        center = tf.cast(center, dtype)
        sigma = tf.cast(sigma, dtype)
        var = tf.maximum(sigma * sigma, 1e-6)
        P = tf.exp(-0.5 * (v - center) ** 2 / var)
        P = tf.concat([P[:, :-1], tf.zeros_like(P[:, -1:])], axis=1)
        return P / (tf.reduce_sum(P) * self.dV)

    def _v_grid_or_linspace(self, dtype) -> TensorType:
        """Сетка потенциалов [1, grid_size] (линейная от v_min до v_max)."""
        v_grid = tf.linspace(
            tf.constant(self.v_min, dtype=dtype),
            tf.constant(self.v_max, dtype=dtype),
            self.grid_size,
        )
        return tf.reshape(v_grid, [1, self.grid_size])

    def get_initial_state(self, batch_size: int = 1) -> StateList:
        raise NotImplementedError(
            "FokkerPlanckPopulation: базовый класс не задаёт начальное "
            "распределение. Используйте self._gaussian_initial_state(center, "
            "sigma) в подклассе или переопределите get_initial_state()."
        )

    def get_firing_rate(self, state: StateList) -> TensorType:
        P = state[0]
        return self.get_boundary_flux(P, self.dV)

    @property
    def parameter_spec(self) -> Dict[str, Dict[str, any]]:
        return {
            "grid_size": {
                "shape": (),
                "trainable": False,
                "constraint": None,
                "units": "dimensionless",
            },
        }
