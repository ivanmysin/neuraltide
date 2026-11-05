"""
Экспоненциальный метод Эйлера (exponential Euler) для жёсткой динамики.

Правая часть на субшаге линеаризуется в текущей точке:

    f(y) ≈ f(y0) + J·(y − y0),  J = ∂f/∂y

и субшаг решается точно для линеаризованной системы:

    y_new = y0 + h·φ1(hJ)·f(y0),  φ1(z) = (e^z − 1)/z

φ1(hJ)·f вычисляется без обращения матрицы и без потери точности при
малых hJ через матричную экспоненту расширенного блока (scaling-and-
squaring внутри tf.linalg.expm):

    exp([[hJ, h·f],    = [[e^{hJ}, h·φ1(hJ)·f],
         [0,   0 ]])       [0,      1           ]]

(bottom-right обязан быть 0: тогда (1,2)-блок экспоненты — интеграл
Дюамеля ∫₀¹ e^{hJ(1−s)}·h·f·e^{0·s} ds = h·φ1(hJ)·f.)

Блоки малые и блочные по юнитам: [n_units, d+1, d+1]. Для линейной
динамики популяции метод точен на любом субшаге (L-устойчив, A-устойчив).

Fallback при недопустимом размере блока (d+1 > max_block_size): режим
on_large_block='raise' (по умолчанию) бросает ValueError; 'euler' —
честный fallback на явный Эйлер с warning. Проверка размера — статическая
(Python int d), поэтому работает и под tf.function.

Градиенты — через развёртку шагов (BPTT/AD, tf.linalg.expm дифференцируем);
аналитический while_loop-adjoint не поддерживается. Синапсы остаются явными.

Network-wide шаг (integration_mode='network') не поддерживается
(supports_network_step=False): NetworkRNN в этом режиме бросит ValueError.
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


class ExponentialEulerIntegrator(BaseIntegrator):
    """
    Экспоненциальный Эйлер с блочной JVP-линеаризацией по юнитам.

    Args:
        n_substeps: число субшагов внутри шага сети.
        jacobian_method: 'jvp' (ForwardAccumulator, по умолчанию) или 'fd'.
        fd_eps: шаг разности для method='fd'.
        max_block_size: предел размера расширенного блока d+1 (d — число
            переменных состояния на юнит). Превышение — по on_large_block.
        on_large_block: 'raise' (ValueError) или 'euler' (fallback на явный
            Эйлер с warning).
    """

    def __init__(
        self,
        n_substeps: int = 1,
        jacobian_method: str = "jvp",
        fd_eps: Optional[float] = None,
        max_block_size: int = 8,
        on_large_block: str = "raise",
    ):
        self.n_substeps = max(1, int(n_substeps))
        self.jacobian_method = jacobian_method
        self.fd_eps = fd_eps
        self.max_block_size = int(max_block_size)
        if on_large_block not in ("raise", "euler"):
            raise ValueError("on_large_block must be 'raise' or 'euler'.")
        self.on_large_block = on_large_block
        self._syn_euler = EulerIntegrator()
        self._warned_fallback = False

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

        d = len(state)
        if d + 1 > self.max_block_size:
            msg = (
                f"ExponentialEulerIntegrator: augmented block size d+1="
                f"{d + 1} exceeds max_block_size={self.max_block_size}. "
                "Increase max_block_size or use on_large_block='euler'."
            )
            if self.on_large_block == "raise":
                raise ValueError(msg)
            if not self._warned_fallback and tf.executing_eagerly():
                import warnings

                warnings.warn(msg + " Falling back to explicit Euler.", RuntimeWarning)
                self._warned_fallback = True
            return self._syn_euler.step(population, state, total_synaptic_input)

        n = int(state[0].shape[1])
        zeros_1d = tf.zeros([n, 1, d], dtype=dtype)
        zeros_11 = tf.zeros([n, 1, 1], dtype=dtype)

        y = state
        err_acc = tf.zeros([], dtype=dtype)
        for _ in range(self.n_substeps):
            f = population.derivatives(y, total_synaptic_input)
            J = state_vector_field_jacobian(
                population, y, total_synaptic_input, self.jacobian_method, self.fd_eps
            )
            f_flat = flatten_state(f)
            # Расширенный блок: [[hJ, h f], [0, 0]] -> экспонента.
            top = tf.concat([h_t * J, (h_t * f_flat)[..., tf.newaxis]], axis=2)
            bottom = tf.concat([zeros_1d, zeros_11], axis=2)
            aug = tf.concat([top, bottom], axis=1)  # [n, d+1, d+1]
            exp_aug = tf.linalg.expm(aug)
            delta = exp_aug[:, :d, d]  # h·φ1(hJ)·f

            y_flat = flatten_state(y)
            y_new_flat = y_flat + delta
            # Индикатор: отклонение от явного Эйлера (конвенция RK4).
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
