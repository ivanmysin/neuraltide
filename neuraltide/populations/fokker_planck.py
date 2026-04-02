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

    def __init__(self, n_units: int, dt: float, grid_size: int = 100,
                 v_min: float = -100.0, v_max: float = 50.0, **kwargs):
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

    def get_initial_state(self, batch_size: int = 1) -> StateList:
        dtype = neuraltide.config.get_dtype()
        return [tf.zeros([1, self.grid_size], dtype=dtype)]

    def get_firing_rate(self, state: StateList) -> TensorType:
        P = state[0]
        return self.get_boundary_flux(P, self.dV)

    @property
    def parameter_spec(self) -> Dict[str, Dict[str, any]]:
        return {
            'grid_size': {
                'shape': (),
                'trainable': False,
                'constraint': None,
                'units': 'dimensionless',
            },
        }
