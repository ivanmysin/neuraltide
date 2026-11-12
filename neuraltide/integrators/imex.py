"""
IMEX-Euler (Implicit-Explicit) интегратор с аддитивным разделением
правой части.

Шаг сети разбивается на n_substeps равных субшагов h = dt/n_substeps.
На каждом субшаге нежёсткая компонента обрабатывается явно, жёсткая —
неявно через линейный оператор популяции:

    y*    = y + h·f_nonstiff(y)
    (I − h·A)·y_new = y* + h·b        (блочно по юнитам, tf.linalg.solve)

Контракт популяции (оба метода — аддитивные хуки в PopulationModel):

    derivatives_split(state, inp) -> (f_nonstiff, f_stiff)
    linear_operator(state, inp)   -> (A [n_units, d, d], b [n_units, d])
                                   или None

Честные fallback'ы (без молчаливых изменений семантики):

- derivatives_split не переопределён или f_stiff ≡ 0, и linear_operator()
  даёт None — схема тождественно совпадает с явным Эйлером (формулы
  схлопываются), в eager выдаётся one-time warning;
- f_stiff ≢ 0, но linear_operator() даёт None — неявная часть невозможна,
  f_stiff обрабатывается явно (тот же порядок, что и явный Эйлер), в eager
  выдаётся one-time warning.

Под tf.function (NetworkRNN трассирует шаги) проверки значений тензоров
недоступны — warnings пропускаются, вычислительный путь идентичен.

Синапсы остаются явными: step_synapse делегирует явному Эйлеру.

Network-wide шаг (integration_mode='network') не поддерживается
(supports_network_step=False): NetworkRNN в этом режиме бросит ValueError.
"""

from typing import Dict, Tuple

import tensorflow as tf

import neuraltide.config
from neuraltide.core.base import PopulationModel, SynapseModel
from neuraltide.core.types import TensorType, StateList, cast_if_needed
from neuraltide.integrators.base import BaseIntegrator
from neuraltide.integrators.euler import EulerIntegrator
from neuraltide.integrators.stiffness import flatten_state, unflatten_state


class IMEXEulerIntegrator(BaseIntegrator):
    """
    IMEX-Euler по разделению derivatives_split/linear_operator.

    Порядок 1 (явный Эйлер по нежёсткой части, неявный Эйлер по жёсткой).
    Жёсткая часть предполагается линейной и постоянной на шаге сети:
    f_stiff(y) = A·y_flat + b (опционально с ненулевым b).

    Args:
        n_substeps: число равных субшагов внутри шага сети.
        warn_on_fallback: выдавать one-time warnings о fallback в eager.
    """

    def __init__(self, n_substeps: int = 1, warn_on_fallback: bool = True):
        self.n_substeps = max(1, int(n_substeps))
        self.warn_on_fallback = bool(warn_on_fallback)
        self._syn_euler = EulerIntegrator()
        self._warned_zero_split = False
        self._warned_no_operator = False

    # ------------------------------------------------------------------ #

    def _maybe_warn_fallback(
        self,
        has_operator: bool,
        f_nonstiff: StateList,
        f_stiff: StateList,
    ) -> None:
        """Eager-диагностика fallback; в графе молча пропускается."""
        if not self.warn_on_fallback or not tf.executing_eagerly():
            return
        stiff_norm = sum(float(tf.reduce_sum(tf.abs(x))) for x in f_stiff)
        if has_operator:
            return
        if stiff_norm == 0.0 and not self._warned_zero_split:
            import warnings

            warnings.warn(
                "IMEXEulerIntegrator: derivatives_split gives zero stiff "
                "part and linear_operator() is None — stepping reduces to "
                "explicit Euler of the full dynamics.",
                RuntimeWarning,
                stacklevel=3,
            )
            self._warned_zero_split = True
        elif stiff_norm > 0.0 and not self._warned_no_operator:
            import warnings

            warnings.warn(
                "IMEXEulerIntegrator: stiff part is non-zero but "
                "linear_operator() returns None — cannot solve the "
                "implicit part; stiff component is treated explicitly "
                "(explicit-Euler accuracy). Override linear_operator in "
                "the population for a true IMEX step.",
                RuntimeWarning,
                stacklevel=3,
            )
            self._warned_no_operator = True

    def step(
        self,
        population: PopulationModel,
        state: StateList,
        total_synaptic_input: Dict[str, TensorType],
    ) -> Tuple[StateList, TensorType]:
        # Линеаризация строится per-unit ([n_units, d, d]) — батч-ось должна
        # быть 1. assert_equal (не статическая форма): внутри графа скана
        # батч-оси динамические. Для batch>1 используйте Euler/Heun/RK4
        # (split) или integration_mode='network'.
        for s in state:
            tf.debugging.assert_equal(
                tf.shape(s)[0],
                1,
                message=(
                    "IMEXEulerIntegrator.step supports batch=1 only "
                    "(population jacobian is per-unit); got state batch "
                    "!= 1. Use Euler/Heun/RK4 or integration_mode="
                    "'network' for batch > 1."
                ),
            )
        dtype = neuraltide.config.get_dtype()
        dt = float(population.dt)
        h = dt / self.n_substeps
        h_t = tf.constant(h, dtype=dtype)

        op = population.linear_operator(state, total_synaptic_input)
        f_nonstiff0, f_stiff0 = population.derivatives_split(
            state, total_synaptic_input
        )
        self._maybe_warn_fallback(op is not None, f_nonstiff0, f_stiff0)

        n = int(state[0].shape[1])
        d = len(state)
        if op is not None:
            A = cast_if_needed(op[0], dtype)
            b = cast_if_needed(op[1], dtype)
            A = tf.reshape(A, [n, d, d])
            b = tf.reshape(b, [n, d])
            eye = tf.eye(d, batch_shape=[n], dtype=dtype)
        else:
            A = None
            b = None

        y = state
        err_acc = tf.zeros([], dtype=dtype)
        for _ in range(self.n_substeps):
            f_nonstiff, f_stiff = population.derivatives_split(y, total_synaptic_input)
            y_flat = flatten_state(y)
            y_ex = y_flat + h_t * flatten_state(f_nonstiff)
            # Индикатор уровня явного Эйлера той же сетки (конвенция RK4:
            # mean по переменным суммы квадратов отклонения).
            ref = y_flat + h_t * (flatten_state(f_nonstiff) + flatten_state(f_stiff))
            if A is not None:
                rhs = y_ex + h_t * b
                M = eye - h_t * A
                y_new_flat = tf.linalg.solve(M, rhs[..., tf.newaxis])[..., 0]
            else:
                y_new_flat = y_ex + h_t * flatten_state(f_stiff)
            err_acc += tf.reduce_mean(
                tf.stack(
                    [
                        tf.reduce_sum((y_new_flat[:, i] - ref[:, i]) ** 2)
                        for i in range(d)
                    ]
                )
            )
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
