"""
Линейно-неявный метод Розенброка 2-го порядка (ROS2) для жёсткой динамики.

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

Синапсы остаются явными (явный Эйлер через step_synapse) — метод
линеаризует только внутрипопуляционную динамику.

В network-режиме (integration_mode='network', step_network) линеаризуется
уже полная связанная правая часть: синаптическая связность входит в плотный
якобиан [D, D] по объединённому состоянию сети (см. stiffness.py).
"""

from typing import Dict, Optional, Tuple

import tensorflow as tf

import neuraltide.config
from neuraltide.core.base import PopulationModel, SynapseModel
from neuraltide.core.types import TensorType, StateList
from neuraltide.integrators.base import BaseIntegrator
from neuraltide.integrators.euler import EulerIntegrator
from neuraltide.integrators.stiffness import (
    flat_vector_field_jacobian,
    flatten_joint_state,
    flatten_state,
    joint_dynamic_indices,
    state_vector_field_jacobian,
    unflatten_joint_state,
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

    Поддерживает network-wide интегрирование (integration_mode='network'):
    step_network() объединяет состояния всех популяций и синапсов в плоский
    вектор (flatten_joint_state), линеаризует полную связанную правую часть
    плотным якобианом [D, D] (flat_vector_field_jacobian) и делает ROS2
    субшаги через tf.linalg.solve. Слоты моделей без динамики неизменны.
    """

    supports_network_step = True

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
        # Пошаговый якобиан строится per-unit ([n_units, d, d]) — батч-ось
        # должна быть 1. Проверка через assert_equal (а не статическую
        # форму): внутри графа скана батч-оси динамические, статическая
        # проверка shape[0] их пропускала бы. Для batch>1 используйте
        # Euler/Heun/RK4 (split) или integration_mode='network'
        # (step_network — батч-агностичен).
        for s in state:
            tf.debugging.assert_equal(
                tf.shape(s)[0],
                1,
                message=(
                    "RosenbrockIntegrator.step supports batch=1 only "
                    "(population jacobian is per-unit); got state batch "
                    "!= 1. Use Euler/Heun/RK4 or integration_mode="
                    "'network' for batch > 1."
                ),
            )
        dtype = neuraltide.config.get_dtype()
        dt = float(population.dt)
        h = dt / self.n_substeps
        h_t = tf.constant(h, dtype=dtype)

        d = len(state)
        # Динамическая батч-ось: при статическом [1, d, d] solve вставил
        # бы BroadcastTo с неизвестной формой (градиент под XLA падает).
        batch = tf.shape(state[0])[0]
        eye = tf.eye(d, batch_shape=[batch], dtype=dtype)

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

    def step_network(
        self,
        rhs,
        pop_states: StateList,
        syn_states: StateList,
        dt: float,
    ) -> Tuple[StateList, StateList]:
        """
        Один ROS2-шаг по связанной сети (integration_mode='network').

        Объединённое состояние сети уплощается в вектор
        (flatten_joint_state); динамические слоты (получающие производные
        от rhs) интегрируются методом ROS2 с плотным якобианом [D, D]
        (D — число динамических компонент), замороженным на субшаг.
        Линейные системы (I − γhJ)·g = rhs решаются tf.linalg.solve.
        Синаптическая связность входит в J, поэтому метод устойчив на
        жёстких связанных задачах, где явные RK расходятся. Слоты моделей
        без динамики возвращаются неизменными.

        Args:
            rhs: _JointNetworkRHS — rhs(pop_states, syn_states) ->
                (dpop_flat, dsyn_flat).
            pop_states: плоский список состояний популяций.
            syn_states: плоский список состояний синапсов.
            dt: шаг интегрирования в мс.

        Returns:
            (new_pop_states, new_syn_states) — полные плоские списки той
            же структуры. Оценка локальной ошибки в network-режиме не
            возвращается (индикатор отклонения от совместного Эйлера —
            конвенция split-версии step()).
        """
        dpop, dsyn = rhs(pop_states, syn_states)
        if not dpop and not dsyn:
            return list(pop_states), list(syn_states)

        dtype = neuraltide.config.get_dtype()
        pop_idx, syn_idx = joint_dynamic_indices(rhs, len(dpop), len(dsyn))
        n_pop = len(pop_states)
        dyn_idx = list(pop_idx) + [n_pop + i for i in syn_idx]

        flat0 = flatten_joint_state(pop_states, syn_states)
        # Динамические sizes: батч-оси состояний внутри скана неизвестны
        # статически (tf.split принимает 1-D тензор размеров).
        all_states = list(pop_states) + list(syn_states)
        sizes_t = tf.stack([tf.size(s) for s in all_states])
        dyn_sizes_t = tf.stack([sizes_t[i] for i in dyn_idx])
        flat_pieces = tf.split(flat0, sizes_t)
        z0 = tf.concat([flat_pieces[i] for i in dyn_idx], axis=0)

        def scatter_z(z: TensorType) -> TensorType:
            """Вставляет динамические компоненты z в полный плоский вектор."""
            upd = list(flat_pieces)
            z_pieces = tf.split(z, dyn_sizes_t)
            for k, i in enumerate(dyn_idx):
                upd[i] = z_pieces[k]
            return tf.concat(upd, axis=0)

        def rhs_flat(z: TensorType) -> TensorType:
            """Плоская правая часть по динамическим компонентам [D]."""
            new_pop, new_syn = unflatten_joint_state(
                scatter_z(z), pop_states, syn_states
            )
            fp, fs = rhs(new_pop, new_syn)
            return tf.concat([tf.reshape(t, [-1]) for t in list(fp) + list(fs)], axis=0)

        h = float(dt) / self.n_substeps
        h_t = tf.constant(h, dtype=dtype)
        # Единичная матрица ПЛОСКОГО вектора [D, D]: D — суммарная
        # размерность динамических слотов с батч-осью (B·per-slot), а не
        # число слотов. tf.eye по числу слотов бродкастился бы до матрицы
        # из единиц (корректно только при B=1, где D == len(dyn_idx)).
        eye = tf.eye(tf.size(z0), dtype=dtype)

        z = z0
        for _ in range(self.n_substeps):
            f = rhs_flat(z)
            J = flat_vector_field_jacobian(
                rhs_flat, z, self.jacobian_method, self.fd_eps
            )
            M = eye - _GAMMA * h_t * J  # [D, D]

            g1 = tf.linalg.solve(M, f[:, tf.newaxis])[:, 0]

            f_mid = rhs_flat(z + h_t * g1)
            Jg1 = tf.linalg.matvec(J, g1)
            rhs2 = f_mid - 2.0 * _GAMMA * h_t * Jg1
            g2 = tf.linalg.solve(M, rhs2[:, tf.newaxis])[:, 0]

            z = z + 0.5 * h_t * (g1 + g2)

        return unflatten_joint_state(scatter_z(z), pop_states, syn_states)
