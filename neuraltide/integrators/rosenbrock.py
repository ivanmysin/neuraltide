"""
Линейно-неявный метод Розенброка 2-го порядка (ROS2) — этап P2.

Для y' = f(y) с якобианом J = ∂f/∂y, замороженным на субшаге:

    (I − γhJ)·g1 = f(y)
    (I − γhJ)·g2 = f(y + h·g1) − 2γh·J·g1
    y_new = y + h/2·(g1 + g2)

с γ = 1 − √2/2 (L-устойчивый выбор). Якобиан вычисляется блочно по
юнитам ([n_units, d, d]) через d вызовов ForwardAccumulator (JVP) по
правой части популяции — внутри юнитов модель батчирована, поэтому один
JVP даёт столбец якобиана сразу для всех юнитов. Решения линейных систем
— батчевые tf.linalg.solve.

Якобиан берётся в точке начала субшага и заморожен на субшаг (классика
ROS2). Под tf.function (NetworkRNN) всё трассируется в граф; ForwardAccumulator
в tf.function поддержан. Градиенты — через развёртку шагов (BPTT/AD);
аналитический while_loop-adjoint не поддерживается.

Синапсы в P2 остаются явными (явный Эйлер через step_synapse) — метод
линеаризует только внутрипопуляционную динамику.
"""

from typing import Dict, Optional, Tuple

import tensorflow as tf

import neuraltide.config
from neuraltide.core.base import PopulationModel, SynapseModel
from neuraltide.core.types import TensorType, StateList
from neuraltide.integrators.base import BaseIntegrator
from neuraltide.integrators.euler import EulerIntegrator
from neuraltide.integrators.stiffness import (
    flatten_state,
    state_vector_field_jacobian,
    unflatten_state,
)

_GAMMA = 1.0 - 0.5**0.5  # 1 − √2/2 ≈ 0.2929, L-устойчивость ROS2


class RosenbrockIntegrator(BaseIntegrator):
    """
    Двухстадийный метод Розенброка ROS2 (порядок 2, L-устойчивый).

    Устойчив на жёстких задачах при шагах, где явные RK расходятся
    (см. benchmarks/l2_stiff_integrators.py). Цена субшага: 2 вызова
    derivatives + d JVP + 2 батчевых решения системы [d, d] на юнит.

    Args:
        n_substeps: число субшагов внутри шага сети (dt = n_substeps·h).
        jacobian_method: 'jvp' (ForwardAccumulator, по умолчанию) или 'fd'
            (центральные разности, дороже и менее точно).
        fd_eps: шаг разности для method='fd'.
    """

    def __init__(
        self,
        n_substeps: int = 1,
        jacobian_method: str = "jvp",
        fd_eps: Optional[float] = None,
    ):
        self.n_substeps = max(1, int(n_substeps))
        self.jacobian_method = jacobian_method
        self.fd_eps = fd_eps
        self._syn_euler = EulerIntegrator()

    # ------------------------------------------------------------------ #

    def step(
        self,
        population: PopulationModel,
        state: StateList,
        total_synaptic_input: Dict[str, TensorType],
    ) -> Tuple[StateList, TensorType]:
        dtype = neuraltide.config.get_dtype()
        dt = float(population.dt)
        h = dt / self.n_substeps
        h_t = tf.constant(h, dtype=dtype)

        n = int(state[0].shape[1])
        d = len(state)
        eye = tf.eye(d, batch_shape=[n], dtype=dtype)

        y = state
        err_acc = tf.zeros([], dtype=dtype)
        for _ in range(self.n_substeps):
            f = population.derivatives(y, total_synaptic_input)
            J = state_vector_field_jacobian(
                population, y, total_synaptic_input, self.jacobian_method, self.fd_eps
            )
            M = eye - _GAMMA * h_t * J  # [n, d, d]

            f_flat = flatten_state(f)
            g1 = tf.linalg.solve(M, f_flat[..., tf.newaxis])[..., 0]

            y_flat = flatten_state(y)
            y_mid = unflatten_state(y_flat + h_t * g1, y)
            f_mid = population.derivatives(y_mid, total_synaptic_input)

            Jg1 = tf.linalg.matvec(J, g1)
            rhs2 = flatten_state(f_mid) - 2.0 * _GAMMA * h_t * Jg1
            g2 = tf.linalg.solve(M, rhs2[..., tf.newaxis])[..., 0]

            y_new_flat = y_flat + 0.5 * h_t * (g1 + g2)
            # Индикатор: отклонение от явного Эйлера (конвенция RK4 —
            # mean по переменным суммы квадратов).
            euler_flat = y_flat + h_t * f_flat
            err_acc += tf.reduce_sum((y_new_flat - euler_flat) ** 2) / d
            y = unflatten_state(y_new_flat, y)

        return y, err_acc[tf.newaxis]

    def step_synapse(
        self,
        synapse: SynapseModel,
        state: StateList,
        pre_firing_rate: TensorType,
        post_voltage: TensorType,
        dt: float,
    ) -> Tuple[StateList, TensorType]:
        return self._syn_euler.step_synapse(
            synapse, state, pre_firing_rate, post_voltage, dt
        )
