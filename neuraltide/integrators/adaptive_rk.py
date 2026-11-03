"""
Адаптивный интегратор Рунге–Кутты 3(2) Богаки–Шампина с PI-контроллером
шага (этап P2).

Внутри одного шага сети выполняется адаптивный substepping: пара
Богаки–Шампина даёт встроенную оценку локальной ошибки, PI-контроллер
(Gustafsson) управляет длиной субшага, неудачные пробы отклоняются.
Наружу отдаётся состояние на границе шага сети и накопленная оценка
локальной ошибки.

Контур управления шагом обёрнут в tf.stop_gradient и выполнен через
tf.while_loop: метод работает и в eager-режиме, и под tf.function
(NetworkRNN трассирует шаги в графе). Градиенты (BPTT/AD) проходят
через математику стадий Богаки–Шампина; аналитический while_loop-adjoint
не поддерживается (см. docs/integrators.md).

Синапсы в P2 остаются явными: step_synapse делегирует явному Эйлеру.
"""

from typing import Dict, Tuple

import tensorflow as tf

import neuraltide.config
from neuraltide.core.base import PopulationModel, SynapseModel
from neuraltide.core.types import TensorType, StateList
from neuraltide.integrators.base import BaseIntegrator
from neuraltide.integrators.euler import EulerIntegrator
from neuraltide.integrators.stiffness import flatten_state, unflatten_state

# Показатели PI-контроллера Gustafsson (порядок пары 3(2): p = 3).
_PI_B1 = 0.7 / 3.0
_PI_B2 = 0.4 / 3.0
_REJECT_EXP = 1.0 / 3.0


class AdaptiveRKIntegrator(BaseIntegrator):
    """
    Встроенная схема RK 3(2) Богаки–Шампина с адаптивным субшагом.

    Стадии (f заморожена по синаптическому входу внутри шага сети):

        k1 = f(y)
        k2 = f(y + h/2·k1)
        k3 = f(y + 3h/4·k2)
        y_new = y + h·(2/9·k1 + 1/3·k2 + 4/9·k3)          (порядок 3)
        k4 = f(y_new)
        y_low = y + h·(7/24·k1 + 1/4·k2 + 1/3·k3 + 1/8·k4) (порядок 2)

    Взвешенная норма ошибки: err = RMS((y_new − y_low) / (atol + rtol·|y|)),
    проба принимается при err <= 1. PI-контроллер:

        accept:  fac = safety · err^(−b1) · err_prev^(b2),  clip [fac_min, fac_max]
        reject:  fac = safety · err^(−1/3),                  clip [fac_min, 1]

    При невозможности удовлетворить допуск на минимальном субшаге min_h
    проба принудительно принимается (состояние не вылетает в бесконечный
    цикл), фиксируется флаг failed и (в eager) выдаётся warning.

    Args:
        rtol, atol: допуск взвешенной нормы ошибки.
        h0: начальный субшаг (по умолчанию весь шаг сети dt).
        min_h_factor: минимальный субшаг = min_h_factor · dt.
        max_substeps: аварийный лимит числа проб на шаг сети.
        safety: запас безопасности контроллера.
        adapt: False — фиксированный субшаг h0 без отклонений (для
            сходимости по порядку и отладки); оценка ошибки всё равно
            накапливается в local_error.
        fac_min, fac_max: пределы изменения субшага.
    """

    def __init__(
        self,
        rtol: float = 1e-4,
        atol: float = 1e-6,
        h0: float | None = None,
        min_h_factor: float = 1e-6,
        max_substeps: int = 100_000,
        safety: float = 0.9,
        adapt: bool = True,
        fac_min: float = 0.2,
        fac_max: float = 10.0,
    ):
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.h0 = None if h0 is None else float(h0)
        self.min_h_factor = float(min_h_factor)
        self.max_substeps = int(max_substeps)
        self.safety = float(safety)
        self.adapt = bool(adapt)
        self.fac_min = float(fac_min)
        self.fac_max = float(fac_max)
        self._syn_euler = EulerIntegrator()
        # Статистика последнего step() (заполняется только в eager).
        self.last_n_substeps: int | None = None
        self.last_mean_substep: float | None = None
        self.last_local_error: float | None = None
        self.last_failed: bool | None = None
        self._n_warnings = 0

    # ------------------------------------------------------------------ #

    def step(
        self,
        population: PopulationModel,
        state: StateList,
        total_synaptic_input: Dict[str, TensorType],
    ) -> Tuple[StateList, TensorType]:
        dtype = neuraltide.config.get_dtype()
        dt = float(population.dt)
        dt_t = tf.constant(dt, dtype=dtype)
        h_init = self.h0 if self.h0 is not None else dt
        h0_t = tf.constant(min(h_init, dt), dtype=dtype)
        min_h_t = tf.constant(max(self.min_h_factor * dt, 1e-30), dtype=dtype)
        max_sub_t = tf.constant(self.max_substeps, dtype=tf.int32)

        y0_flat = flatten_state(state)
        n_elems = tf.constant(
            sum(int(s.shape.num_elements()) for s in state), dtype=dtype
        )

        def _bs23(y_flat, h_eff):
            y = unflatten_state(y_flat, state)
            f1 = population.derivatives(y, total_synaptic_input)
            y2 = [y[i] + h_eff * 0.5 * f1[i] for i in range(len(y))]
            f2 = population.derivatives(y2, total_synaptic_input)
            y3 = [y[i] + h_eff * 0.75 * f2[i] for i in range(len(y))]
            f3 = population.derivatives(y3, total_synaptic_input)
            y_new = [
                y[i]
                + h_eff * (2.0 / 9.0 * f1[i] + 1.0 / 3.0 * f2[i] + 4.0 / 9.0 * f3[i])
                for i in range(len(y))
            ]
            f4 = population.derivatives(y_new, total_synaptic_input)
            y_low = [
                y[i]
                + h_eff
                * (
                    7.0 / 24.0 * f1[i]
                    + 1.0 / 4.0 * f2[i]
                    + 1.0 / 3.0 * f3[i]
                    + 1.0 / 8.0 * f4[i]
                )
                for i in range(len(y))
            ]
            return y_new, y_low

        def _err_norms(y, y_new, y_low):
            sq_w = None
            sq_raw = None
            for i in range(len(y)):
                scale = self.atol + self.rtol * tf.maximum(
                    tf.abs(y_new[i]), tf.abs(y[i])
                )
                d = y_new[i] - y_low[i]
                term_w = tf.reduce_sum((d / scale) ** 2)
                term_raw = tf.reduce_sum(d**2)
                sq_w = term_w if sq_w is None else sq_w + term_w
                sq_raw = term_raw if sq_raw is None else sq_raw + term_raw
            err_w = tf.sqrt(sq_w / n_elems)
            err_raw = sq_raw / len(y)  # конвенция RK4: mean по переменным
            return err_w, err_raw

        if self.adapt:

            def _cond(t_done, h, y_flat, err_prev, rejected, err_acc, n_sub, failed):
                return (t_done < dt_t) & (n_sub < max_sub_t)

            def _body(t_done, h, y_flat, err_prev, rejected, err_acc, n_sub, failed):
                h_eff = tf.minimum(h, dt_t - t_done)
                y = unflatten_state(y_flat, state)
                y_new, y_low = _bs23(y_flat, h_eff)
                err_w, err_raw = _err_norms(y, y_new, y_low)
                forced = h_eff <= min_h_t
                accept = (err_w <= 1.0) | forced

                # --- PI-контроллер (контур управления без градиентов) ---
                fac_acc = (
                    self.safety
                    * tf.math.pow(err_w, -_PI_B1)
                    * tf.math.pow(err_prev, _PI_B2)
                )
                fac_rej = self.safety * tf.math.pow(err_w, -_REJECT_EXP)
                fac_acc = tf.clip_by_value(fac_acc, self.fac_min, self.fac_max)
                fac_rej = tf.clip_by_value(fac_rej, self.fac_min, 1.0)
                was_rejected = rejected > 0.5
                fac = tf.where(was_rejected, fac_rej, fac_acc)
                fac = tf.stop_gradient(fac)

                h_next = tf.maximum(h_eff * fac, min_h_t)

                new_flat = flatten_state(y_new)
                y_flat_next = tf.where(accept, new_flat, y_flat)
                t_next = t_done + tf.where(accept, h_eff, tf.zeros_like(h_eff))
                err_prev_next = tf.where(accept, err_w, err_prev)
                zero = tf.zeros([], dtype=dtype)
                one = tf.ones([], dtype=dtype)
                rejected_next = tf.where(accept, zero, one)
                err_acc_next = err_acc + tf.where(accept, err_raw, zero)
                failed_next = failed | (forced & (err_w > 1.0))
                return (
                    t_next,
                    h_next,
                    y_flat_next,
                    err_prev_next,
                    rejected_next,
                    err_acc_next,
                    n_sub + 1,
                    failed_next,
                )

            init = (
                tf.zeros([], dtype=dtype),
                h0_t,
                y0_flat,
                tf.ones([], dtype=dtype),
                tf.zeros([], dtype=dtype),
                tf.zeros([], dtype=dtype),
                tf.constant(0, dtype=tf.int32),
                tf.constant(False),
            )
            t_done, h_last, y_flat, _, _, err_acc, n_sub, failed = tf.while_loop(
                _cond, _body, init
            )
        else:
            # Фиксированный субшаг: честная развёртка (дифференцируема).
            n_substeps = max(1, int(round(dt / h_init)))
            y_flat = y0_flat
            err_acc = tf.zeros([], dtype=dtype)
            t_done = tf.zeros([], dtype=dtype)
            failed = tf.constant(False)
            n_sub = tf.constant(0, dtype=tf.int32)
            h_eff = tf.constant(dt / n_substeps, dtype=dtype)
            for _ in range(n_substeps):
                y = unflatten_state(y_flat, state)
                y_new, y_low = _bs23(y_flat, h_eff)
                _, err_raw = _err_norms(y, y_new, y_low)
                y_flat = flatten_state(y_new)
                err_acc += err_raw
                t_done += h_eff
                n_sub += 1

        if tf.executing_eagerly():
            self.last_n_substeps = int(n_sub)
            self.last_mean_substep = float(t_done) / max(int(n_sub), 1)
            self.last_local_error = float(err_acc)
            self.last_failed = bool(failed)
            if bool(failed) and self._n_warnings < 3:
                import warnings

                warnings.warn(
                    "AdaptiveRKIntegrator: tolerance not met even at min_h; "
                    "step accepted forcibly. Consider smaller dt or an "
                    "implicit/exponential integrator (stiff problem?).",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._n_warnings += 1
            unfinished = bool(t_done < dt_t - 1e-12)
            if unfinished:
                raise RuntimeError(
                    f"AdaptiveRKIntegrator: max_substeps={self.max_substeps} "
                    f"exhausted after advancing {float(t_done):.6g} of dt={dt}. "
                    "Increase max_substeps or rtol/atol."
                )

        new_state = unflatten_state(y_flat, state)
        return new_state, err_acc[tf.newaxis]

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
