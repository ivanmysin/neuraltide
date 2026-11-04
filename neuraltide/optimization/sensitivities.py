"""
Чувствительности и производные длинных горизонтов (этап P1).

Поверх CalibrationProblem (этап P0/P1) реализованы:

- residual_jvp(problem, phi, v) — одна forward-mode JVP-оценка невязок:
    r = problem.residual(phi), J·v через tf.autodiff.ForwardAccumulator.
    Сид переменных модели = dθ/dφ · v (поэлементный transform.jacobian),
    tensor-backed записи ('initial_state'/'shooting_state') материализуются
    внутри аккумулятора из φ, поэтому их касательная распространяется сама;
- residual_jacobian(problem, phi, *, force, max_params) — полный якобиан
    невязок по столбцам через 1 общую оценку невязок и n JVP-прогонов
    без повторного счёта невязок (DTO JacobianResult, n+1 прямых прогонов);
- value_and_gradient_checkpointed(problem, phi, segment_length) — сегментный
    прогон с переносом final_state и gradient checkpointing: сначала
    пробуется tf.recompute_grad, при недоступности/некорректном градиенте
    на текущем TF — ручной fallback через tf.custom_gradient с пересчётом
    сегмента. Критерий приёмки — совпадение с plain BPTT;
- truncated_bptt(problem, phi0, optimizer, *, horizon, epochs, warm_steps) —
    явно приближённый режим: градиент обрывается tf.stop_gradient на границе
    окна, carry — финальное состояние предыдущего окна, после обучения всегда
    выполняется полный free-run.

Зафиксированные зависимости от приватного API (не изменять без синхронизации):
  * neuraltide.optimization.problem: _materialize_initial_state,
    _chain_gradient, _phi_gradient, а также методы/атрибуты CalibrationProblem
    ._require_objective(), ._apply_phi(), ._counters (счётчики, общие с
    view-задачами);
  * Keras 3 backend Variable._value: ForwardAccumulator требует primal с
    .dtype=tf.DType, а keras Variable.dtype возвращает строку; underlying
    ResourceVariable (._value) обёрнут в helper _tf_variable.

Ограничения: eager-режим; batch=1 (контракт NetworkRNN); objective для
residual-операций обязан предоставлять residual(predictions) (иначе TypeError
от problem.residual); все numpy-выходы — float64.
"""

import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

import neuraltide.config
from neuraltide.core.types import cast_if_needed
from neuraltide.optimization.problem import (
    _chain_gradient,
    _materialize_initial_state,
    _phi_gradient,
    CalibrationProblem,
)
from neuraltide.optimization.results import JacobianResult
from neuraltide.optimization.objectives import likelihood_row_scale

__all__ = [
    "CheckpointedGradient",
    "TruncatedResult",
    "residual_jvp",
    "residual_jacobian",
    "value_and_gradient_checkpointed",
    "truncated_bptt",
]


# ---------------------------------------------------------------------------
# DTO
# ---------------------------------------------------------------------------


@dataclass
class CheckpointedGradient:
    """
    Результат value_and_gradient_checkpointed.

    Атрибуты:
        loss: скалярная потеря полного прогона (сумма сегментных вкладов);
        grad: градиент по φ (numpy float64, форма [size]);
        segments: число сегментов, на которые разбит горизонт;
        wall_time: длительность вызова, секунды;
        peak_memory: best-effort оценка пиковой памяти (bytes, tracemalloc;
            None, если измерение недоступно). В eager-режиме Python-аллокации
            не отражают реальный пик GPU/CPU тензоров — трактовать только как
            нижнюю оценку.
    """

    loss: float
    grad: np.ndarray
    segments: int
    wall_time: float
    peak_memory: Optional[float] = None

    def __post_init__(self) -> None:
        self.loss = float(self.loss)
        self.grad = np.asarray(self.grad, dtype=np.float64)
        self.segments = int(self.segments)
        self.wall_time = float(self.wall_time)
        if self.peak_memory is not None:
            self.peak_memory = float(self.peak_memory)


@dataclass
class TruncatedResult:
    """
    Результат truncated_bptt.

    Атрибуты:
        history: список записей по эпохам
            [{"epoch": int, "loss": float}];
        final_loss: потеря последнего обучающего окна (приближённая оценка);
        free_run_loss: потеря полного свободного прогона после обучения;
        notes: документация приближённости режима.
    """

    history: List[Dict[str, float]] = field(default_factory=list)
    final_loss: float = 0.0
    free_run_loss: float = 0.0
    notes: str = ""

    def __post_init__(self) -> None:
        self.history = [dict(h) for h in self.history]
        self.final_loss = float(self.final_loss)
        self.free_run_loss = float(self.free_run_loss)
        self.notes = str(self.notes)


# ---------------------------------------------------------------------------
# Внутренние helpers
# ---------------------------------------------------------------------------


def _tf_variable(var: tf.Variable) -> tf.Variable:
    """
    Underlying tf.Variable для Keras 3 backend Variable.

    tf.autodiff.ForwardAccumulator требует primal с .dtype=tf.DType, а
    keras Variable.dtype возвращает строку; ._value (ResourceVariable)
    удовлетворяет контракту аккумулятора (dtype + handle).
    """
    inner = getattr(var, "_value", None)
    return inner if isinstance(inner, tf.Variable) else var


def _entry_offsets(space) -> Dict[str, Tuple[int, int]]:
    """Смещения записей ParameterSpace в плоском векторе φ."""
    offsets: Dict[str, Tuple[int, int]] = {}
    start = 0
    for name, entry in space.entries.items():
        offsets[name] = (start, entry.numel)
        start += entry.numel
    return offsets


def _flat_state(
    state: Tuple[Sequence[tf.Tensor], Sequence[tf.Tensor]],
) -> Tuple[tf.Tensor, ...]:
    """(pop, syn) -> плоский кортеж тензоров."""
    return tuple(state[0]) + tuple(state[1])


def _split_state(flat: Sequence[tf.Tensor], n_pop: int, n_syn: int):
    return list(flat[:n_pop]), list(flat[n_pop : n_pop + n_syn])


def _segment_loss(
    problem: CalibrationProblem,
    objective,
    output,
    s: int,
    e: int,
) -> tf.Tensor:
    """
    Вклад временного среза [s, e) в полную потерю.

    Для потерь с аддитивным per-step интерфейсом (ResidualLoss: per_step_loss
    + target) нормировка N берётся от полной траектории наблюдения, поэтому
    сумма срезов воспроизводит полную потерю точно. Для прочих objective
    срез допускается только на всём горизонте (s=0, e=T), иначе TypeError.
    """
    per_step_fn = getattr(objective, "per_step_loss", None)
    target_map = getattr(objective, "target", None)
    if not callable(per_step_fn) or not isinstance(target_map, dict):
        T = int(problem.t_sequence.shape[1])
        if (s, e) != (0, T):
            raise TypeError(
                f"sensitivities: objective "
                f"'{type(objective).__name__}' не предоставляет аддитивный "
                "per_step_loss/target-интерфейс, поэтому сегментный/оконный "
                "режим поддерживается только на полном горизонте "
                f"(s=0, e=T); получен срез [{s}, {e}), T={T}."
            )
        return objective(output, problem.network)

    rates = {pop: rates_k for pop, rates_k in output.firing_rates.items()}
    target = {name: t[:, s:e] for name, t in target_map.items()}
    mask_map = getattr(objective, "mask", None)
    mask = None
    if isinstance(mask_map, dict) and mask_map:
        mask = {}
        for name, m in mask_map.items():
            t_full = target_map[name].shape[1] if name in target_map else None
            has_time = m.shape.rank == 3 and t_full is not None and m.shape[1] == t_full
            # Маску без полной оси времени ([batch, n], [1, 1, n]) не режем:
            # срез [:, s:e] резал бы её по оси юнитов или давал пустой тензор.
            mask[name] = m[:, s:e] if has_time else m
    return per_step_fn(rates, target, mask)


# ---------------------------------------------------------------------------
# Forward sensitivities: JVP и якобиан невязок
# ---------------------------------------------------------------------------


def _forward_jvp_only(
    problem: CalibrationProblem,
    phi_t: tf.Tensor,
    v_t: tf.Tensor,
) -> np.ndarray:
    """
    Forward-mode JVP J·v для уже упакованного φ (без оценки невязок).

    Тело forward-JVP из residual_jvp без вызова problem.residual: один
    прогон под tf.autodiff.ForwardAccumulator, где primals — trainable-
    переменные записей kind 'model_var' и сам вектор φ (ожидается, что
    переменные модели уже переведены в θ = unpack(phi_t) вызывающей
    стороной, как это делает problem.residual). Сиды переменных =
    dθ/dφ · v (поэлементный transform.jacobian), касательная φ равна v,
    поэтому tensor-backed записи ('initial_state'/'shooting_state')
    получают свою касательную автоматически при материализации из φ.

    Args:
        problem: CalibrationProblem с objective, предоставляющим
            residual(predictions); иначе TypeError (от problem.residual).
        phi_t: упакованный вектор φ (problem.pack), глобальный dtype.
        v_t: направление той же размерности и dtype, что и phi_t.

    Returns:
        J·v — numpy float64 массив формы [n_residuals].

    Счётчики: n_forward увеличивается на 1 (JVP-прогон); wall_time
        дополняется длительностью прогона.
    """
    residual_fn = getattr(problem.objective, "residual", None)
    if residual_fn is None:
        # Отражаем контракт problem.residual до любого прогона/мутации.
        problem.residual(None)  # noqa: B018 — вызов ради понятной ошибки

    space = problem.parameter_space
    offsets = _entry_offsets(space)
    var_primals: List[tf.Tensor] = []
    var_tangents: List[tf.Tensor] = []
    for name, entry in space.entries.items():
        if entry.kind != "model_var" or entry.variable is None:
            continue
        start, numel = offsets[name]
        piece = tf.reshape(phi_t[start : start + numel], entry.shape)
        v_piece = tf.reshape(v_t[start : start + numel], entry.shape)
        jac = entry.transform.jacobian(piece)
        seed = tf.reshape(jac * v_piece, entry.variable.shape)
        var_primals.append(_tf_variable(entry.variable))
        var_tangents.append(cast_if_needed(seed, entry.variable.dtype))

    t0 = time.perf_counter()
    with tf.autodiff.ForwardAccumulator(
        var_primals + [phi_t], var_tangents + [v_t]
    ) as acc:
        theta = space.unpack(phi_t)
        initial_state, _ = _materialize_initial_state(
            problem.network, space, theta, problem.initial_state
        )
        output = problem.network(
            problem.t_sequence,
            inputs=problem.inputs,
            initial_state=initial_state,
        )
        r_t = residual_fn(output)
    jv_t = acc.jvp(r_t)

    problem._counters["n_forward"] += 1
    problem._counters["wall_time"] += time.perf_counter() - t0
    return np.asarray(jv_t, dtype=np.float64)


def residual_jvp(
    problem: CalibrationProblem,
    phi: Optional[Any],
    v: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Невязки и forward-mode JVP J·v в точке φ.

    r = problem.residual(phi); J — якобиан невязок по φ. J·v вычисляется
    одним прямым прогоном под tf.autodiff.ForwardAccumulator (см.
    _forward_jvp_only): primals — trainable-переменные записей kind
    'model_var' и сам вектор φ; сиды переменных = dθ/dφ · v (поэлементный
    transform.jacobian), касательная φ равна v, поэтому tensor-backed
    записи ('initial_state'/'shooting_state') получают свою касательную
    автоматически при материализации из φ.

    Args:
        problem: CalibrationProblem с objective, предоставляющим
            residual(predictions); иначе TypeError (от problem.residual).
        phi: вектор φ (None — текущие значения модели, тогда tensor-backed
            записи запрещены контрактом pack).
        v: направление той же размерности, что и φ.

    Returns:
        (r, J·v) — numpy float64 массивы формы [n_residuals].

    Счётчики: n_forward увеличивается на 2 (оценка невязок + JVP-прогон);
    wall_time дополняется длительностью вызова.
    """
    residual_fn = getattr(problem.objective, "residual", None)
    if residual_fn is None:
        # Отражаем контракт problem.residual до любого прогона/мутации.
        problem.residual(phi)  # noqa: B018 — вызов ради понятной ошибки

    dtype = neuraltide.config.get_dtype()
    v_t = cast_if_needed(tf.convert_to_tensor(v), dtype)
    v_t = tf.reshape(v_t, [-1])
    if v_t.shape[0] is not None and int(v_t.shape[0]) != problem.size:
        raise ValueError(
            f"residual_jvp: v has {int(v_t.shape[0])} elements, "
            f"expected {problem.size}."
        )
    tf.debugging.assert_equal(
        tf.size(v_t), problem.size, message="residual_jvp: bad v size"
    )

    # Невязки в точке (при phi != None переводит модель в θ = unpack(phi)).
    t0 = time.perf_counter()
    r = np.asarray(problem.residual(phi), dtype=np.float64)
    problem._counters["wall_time"] += time.perf_counter() - t0
    phi_t = problem.pack(phi)

    jv = _forward_jvp_only(problem, phi_t, v_t)
    return r, jv


def residual_jacobian(
    problem: CalibrationProblem,
    phi: Optional[Any],
    *,
    force: bool = False,
    max_params: int = 200,
) -> JacobianResult:
    """
    Полный якобиан невязок J = ∂r/∂φ по столбцам через n unit-JVP.

    Столбец J[:, i] = residual_jvp(problem, phi, e_i)[1]. Невязки r в точке
    оцениваются один раз и переиспользуются для всех столбцов, поэтому
    требуется 1 оценка невязок + n JVP-прогонов (n+1 прямых прогонов
    сети). Для большого n это всё равно невыгодно: при n > max_params и не
    force поднимается ValueError с честной оценкой стоимости и
    альтернативами.

    Args:
        problem: CalibrationProblem с residual-objective.
        phi: точка оценки (None — текущие значения модели).
        force: явно разрешить n > max_params.
        max_params: порог числа параметров без force.

    Returns:
        JacobianResult с J [n_residuals, size], r, loss = 0.5·||r||²,
        phi, n_forward (дельта счётчика задачи), wall_time, method='forward'
        и likelihood_scale — множители строк для перевода J в единицы
        полного NLL (для ResidualLoss с normalization='mean' это sqrt(N);
        compute_fim и laplace_approximation применяют их сами).
    """
    n = problem.size
    if n > max_params and not force:
        raise ValueError(
            f"residual_jacobian: n={n} параметров превышает "
            f"max_params={max_params}. Построение требует 1 оценку невязок "
            f"плюс {n} JVP-прогонов (≈{n + 1} прямых прогонов сети), "
            f"т.е. ~O({n}) стоимости одного прогона горизонта T. "
            "Альтернативы: конечно-разностный якобиан ('2-point' в "
            "scipy.optimize.least_squares, ≈2n прогонов без forwardprop), "
            "adjoint-градиент для скалярной потери, или force=True для "
            "явного запуска."
        )

    dtype = neuraltide.config.get_dtype()
    phi_np = np.asarray(problem.pack(phi), dtype=np.float64)
    t0 = time.perf_counter()
    n_forward0 = problem.n_forward

    # Невязки в точке — один общий прогон для всех столбцов (+1 n_forward;
    # при phi != None переводит модель в θ = unpack(phi)).
    t_r = time.perf_counter()
    r = np.asarray(problem.residual(phi), dtype=np.float64)
    problem._counters["wall_time"] += time.perf_counter() - t_r
    phi_t = problem.pack(phi)

    J = np.zeros((r.shape[0], n), dtype=np.float64)
    for i in range(n):
        e = np.zeros(n, dtype=np.float64)
        e[i] = 1.0
        v_t = cast_if_needed(tf.convert_to_tensor(e), dtype)
        J[:, i] = _forward_jvp_only(problem, phi_t, v_t)

    elapsed = time.perf_counter() - t0
    loss = 0.5 * float(r @ r)
    scale = likelihood_row_scale(problem)
    if scale is not None and scale.shape[0] != r.shape[0]:
        scale = None  # предсказания другой формы, чем target: множители неизвестны
    return JacobianResult(
        J=J,
        r=r,
        loss=loss,
        phi=phi_np,
        n_forward=problem.n_forward - n_forward0,
        wall_time=elapsed,
        method="forward",
        likelihood_scale=scale,
    )


# ---------------------------------------------------------------------------
# Checkpointed BPTT
# ---------------------------------------------------------------------------

_RECOMPUTE_PROBE: Optional[bool] = None


def _probe_recompute_grad() -> bool:
    """
    Проверяет, что tf.recompute_grad на текущем TF даёт корректный eager-
    градиент через assign переменной и её чтение внутри @tf.function.

    Именно эта комбинация нужна для полного градиента по φ (θ = unpack(φ)
    записывается в переменные). На TF, где assign не дифференцируем через
    recompute (градиент теряет вклад dθ/dφ), вернёт False — тогда
    используется ручной fallback tf.custom_gradient.
    """
    global _RECOMPUTE_PROBE
    if _RECOMPUTE_PROBE is not None:
        return _RECOMPUTE_PROBE
    ok = False
    try:
        if hasattr(tf, "recompute_grad"):
            dtype = neuraltide.config.get_dtype()
            probe_var = tf.Variable([1.0, 1.0], dtype=dtype)

            @tf.function
            def _read(x):
                return tf.reduce_sum(x * probe_var)

            def _fn(a):
                probe_var.assign(a * 2.0)
                return _read(a)

            a = tf.constant([2.0, 3.0], dtype=dtype)
            with tf.GradientTape() as tape:
                tape.watch(a)
                y = tf.recompute_grad(_fn)(a)
            g = tape.gradient(y, a)
            # y = sum(a * 2a) = 2·sum(a²)  ⇒  dy/da = 4a = [8, 12].
            ok = g is not None and bool(
                np.allclose(np.asarray(g, dtype=np.float64), [8.0, 12.0], rtol=1e-4)
            )
    except Exception:
        ok = False
    _RECOMPUTE_PROBE = ok
    return ok


def _select_checkpoint_backend() -> str:
    """Выбор backend checkpointing: 'recompute_grad' или 'custom_gradient'."""
    if _probe_recompute_grad():
        return "recompute_grad"
    return "custom_gradient"


def _make_checkpoint_segment(
    problem: CalibrationProblem,
    t_k: tf.Tensor,
    inputs_k: Optional[tf.Tensor],
    n_pop: int,
    n_syn: int,
    s: int,
    e: int,
):
    """
    Сегментная функция с ручным checkpointing (tf.custom_gradient).

    Входы: плоские состояния carry + φ. Выходы: (loss, *новые состояния).
    В прямом проходе — один прогон сегмента; градиент — пересчёт сегмента
    под внутренней лентой с взвешенным скаляром-заместителем
    surr = Σ_k dy_k · out_k (dy как константы), что воспроизводит
    output_gradients-семантику точно. dφ собирается в chain-режиме
    (переменные × jacobian, x0-вклад приходит через carry-входы).
    """
    network = problem.network
    objective = problem.objective
    space = problem.parameter_space
    dtype = neuraltide.config.get_dtype()
    counters = problem._counters

    @tf.custom_gradient
    def _segment(*flat_inputs):
        phi = flat_inputs[-1]
        init_pop, init_syn = _split_state(flat_inputs[:-1], n_pop, n_syn)
        output = network(t_k, inputs=inputs_k, initial_state=(init_pop, init_syn))
        loss_t = _segment_loss(problem, objective, output, s, e)
        counters["n_forward"] += 1
        new_pop = tuple(output.final_state[0])
        new_syn = tuple(output.final_state[1])

        def _grad_fn(*dys, variables=None):
            dloss = dys[0]
            dnew = dys[1:]
            with tf.GradientTape(persistent=True) as inner:
                inner.watch(flat_inputs)
                out2 = network(t_k, inputs=inputs_k, initial_state=(init_pop, init_syn))
                loss2 = _segment_loss(problem, objective, out2, s, e)
                flat_out = (loss2,) + _flat_state(out2.final_state)
                surr = tf.zeros((), dtype=dtype)
                for t_i, dy_i in zip(flat_out, (dloss,) + dnew):
                    if dy_i is None:
                        continue
                    dy_c = tf.stop_gradient(cast_if_needed(dy_i, dtype))
                    surr = surr + tf.reduce_sum(cast_if_needed(t_i, dtype) * dy_c)
            dphi = _chain_gradient(space, inner, surr, phi, {})
            dinputs = inner.gradient(surr, list(flat_inputs[:-1]))
            del inner
            counters["n_forward"] += 1
            var_grads = [tf.zeros_like(_tf_variable(var)) for var in (variables or [])]
            input_grads = tuple(dinputs) + (dphi,)
            return input_grads, var_grads

        return (loss_t,) + new_pop + new_syn, _grad_fn

    return _segment


def _make_recompute_segment(
    problem: CalibrationProblem,
    t_k: tf.Tensor,
    inputs_k: Optional[tf.Tensor],
    n_pop: int,
    n_syn: int,
    s: int,
    e: int,
):
    """
    Сегментная функция через tf.recompute_grad (backend для TF, где probe
    подтвердил корректный градиент через assign + чтение переменных).
    """

    network = problem.network
    objective = problem.objective
    space = problem.parameter_space
    counters = problem._counters

    def _seg(*flat):
        phi = flat[-1]
        theta = space.unpack(phi)
        space.apply_to_model(theta)
        init_pop, init_syn = _split_state(flat[:-1], n_pop, n_syn)
        init, _ = _materialize_initial_state(
            network, space, theta, (init_pop, init_syn)
        )
        output = network(t_k, inputs=inputs_k, initial_state=init)
        counters["n_forward"] += 1
        return (_segment_loss(problem, objective, output, s, e),) + _flat_state(
            output.final_state
        )

    return tf.recompute_grad(_seg)


def value_and_gradient_checkpointed(
    problem: CalibrationProblem,
    phi: Optional[Any],
    segment_length: int,
) -> CheckpointedGradient:
    """
    Точный градиент дискретной модели сегментным прогоном + checkpointing.

    Горизонт T разбивается на ceil(T / segment_length) сегментов; начальное
    состояние каждого следующего сегмента — final_state предыдущего (carry).
    Каждый сегмент обёрнут в checkpointing-механизм: tf.recompute_grad, если
    на текущем TF подтверждён корректный градиент (probe), иначе ручной
    fallback tf.custom_gradient с пересчётом сегмента. Градиент совпадает с
    plain problem.value_and_gradient (одна и та же дискретная карта), что и
    является критерием приёмки; экономия памяти в eager ограничена и не
    гарантируется (peak_memory — best-effort).

    Loss — сумма сегментных вкладов objective; для аддитивных потерь
    (ResidualLoss) совпадает с полной потерей точно.

    Args:
        problem: CalibrationProblem.
        phi: вектор φ (None — текущие значения модели).
        segment_length: число временных шагов в сегменте (>= 1).

    Returns:
        CheckpointedGradient(loss, grad, segments, wall_time, peak_memory).

    Счётчики: n_forward увеличивается на 2·segments (прямые + пересчёт при
    обратном проходе), n_gradient на 1; wall_time дополняется.
    """
    problem._require_objective()
    segment_length = int(segment_length)
    if segment_length < 1:
        raise ValueError(
            f"value_and_gradient_checkpointed: segment_length must be >= 1, "
            f"got {segment_length}."
        )
    dtype = neuraltide.config.get_dtype()
    t_start = time.perf_counter()
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    else:
        _, mem_peak0 = tracemalloc.get_traced_memory()

    phi_t = problem.pack(phi)
    if phi is not None:
        problem._apply_phi(phi_t)

    T = int(problem.t_sequence.shape[1])
    starts = list(range(0, T, segment_length))
    segments = len(starts)
    backend = _select_checkpoint_backend()

    try:
        with tf.GradientTape() as tape:
            tape.watch(phi_t)
            # Материализация начального состояния — внутри ленты: вклад
            # tensor-backed записей идёт через carry-входы первого сегмента.
            theta = problem.parameter_space.unpack(phi_t)
            base_init, _ = _materialize_initial_state(
                problem.network,
                problem.parameter_space,
                theta,
                problem.initial_state,
            )
            if base_init is None:
                pop0, syn0 = problem.network.get_initial_state(batch_size=1)
                base_init = (list(pop0), list(syn0))
            n_pop = len(base_init[0])
            n_syn = len(base_init[1])
            carry = (list(base_init[0]), list(base_init[1]))
            losses = []
            for s in starts:
                e = min(s + segment_length, T)
                t_k = problem.t_sequence[:, s:e]
                in_k = None if problem.inputs is None else problem.inputs[:, s:e]
                flat_in = _flat_state(carry) + (phi_t,)
                if backend == "recompute_grad":
                    seg_fn = _make_recompute_segment(
                        problem, t_k, in_k, n_pop, n_syn, s, e
                    )
                else:
                    seg_fn = _make_checkpoint_segment(
                        problem, t_k, in_k, n_pop, n_syn, s, e
                    )
                outs = seg_fn(*flat_in)
                losses.append(outs[0])
                carry = _split_state(outs[1:], n_pop, n_syn)
            total = tf.add_n(losses)
        grad_t = tape.gradient(total, phi_t)
    finally:
        if not was_tracing:
            _, mem_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_memory: Optional[float] = float(mem_peak)
        else:
            _, mem_peak = tracemalloc.get_traced_memory()
            peak_memory = float(max(0, mem_peak - mem_peak0))

    if grad_t is None:
        grad_t = tf.zeros([problem.size], dtype=dtype)
    problem._counters["n_gradient"] += 1
    elapsed = time.perf_counter() - t_start
    problem._counters["wall_time"] += elapsed
    return CheckpointedGradient(
        loss=float(total),
        grad=np.asarray(grad_t, dtype=np.float64),
        segments=segments,
        wall_time=elapsed,
        peak_memory=peak_memory,
    )


# ---------------------------------------------------------------------------
# Truncated BPTT
# ---------------------------------------------------------------------------


def truncated_bptt(
    problem: CalibrationProblem,
    phi0: Any,
    optimizer,
    *,
    horizon: int,
    epochs: int,
    warm_steps: int = 0,
) -> TruncatedResult:
    """
    Явно приближённый truncated BPTT по окнам длины horizon.

    Окна: [s, s + horizon), s = k·(horizon − warm_steps), пока s < T. Между
    окнами carry передаётся через tf.stop_gradient — скрытого градиентного
    потока через границу окна нет, поэтому режим приближён: градиент окна не
    учитывает чувствительность стартового состояния. При warm_steps > 0 окна
    перекрываются на warm_steps шагов; перенос состояния time-consistent: окно
    прогоняется в двух частях [s, s+step) и [s+step, s+horizon), и carry для
    следующего окна — финальное состояние первой части (состояние ровно в
    начале следующего окна). Градиент внутри окна проходит обе части целиком
    (truncation только на границах окон); перекрывающиеся участки получают
    независимые градиентные пути от своего carry.

    Параметры хранятся в tf.Variable, инициализированной из phi0; модель
    переводится в θ(φ) перед каждым окном и остаётся в финальной точке после
    возврата (общий контракт мутации CalibrationProblem). Keras-оптимизатор
    используется штатно (apply_gradients), его состояние между окнами и
    эпохами сохраняется.

    После обучения всегда выполняется полный свободный прогон
    (problem.evaluate) — free_run_loss; final_loss — потеря последнего
    обучающего окна.

    Args:
        problem: CalibrationProblem.
        phi0: стартовый вектор φ (обязателен, задаёт начальную точку).
        optimizer: Keras-оптимизатор с apply_gradients.
        horizon: длина окна в шагах (>= 1).
        epochs: число проходов по набору окон (>= 1).
        warm_steps: перекрытие окон, 0 <= warm_steps < horizon.

    Returns:
        TruncatedResult(history, final_loss, free_run_loss, notes).

    Счётчики: на каждое окно +1 n_forward и +1 n_gradient, +1 n_forward за
    финальный free-run; wall_time дополняется.
    """
    problem._require_objective()
    if not callable(getattr(optimizer, "apply_gradients", None)):
        raise TypeError(
            "truncated_bptt: optimizer must provide apply_gradients "
            f"(Keras optimizer), got {type(optimizer).__name__}."
        )
    horizon = int(horizon)
    epochs = int(epochs)
    warm_steps = int(warm_steps)
    if horizon < 1:
        raise ValueError(f"truncated_bptt: horizon must be >= 1, got {horizon}.")
    if epochs < 1:
        raise ValueError(f"truncated_bptt: epochs must be >= 1, got {epochs}.")
    if not 0 <= warm_steps < horizon:
        raise ValueError(
            f"truncated_bptt: warm_steps must be in [0, horizon), got "
            f"{warm_steps} (horizon={horizon})."
        )

    dtype = neuraltide.config.get_dtype()
    phi_var = tf.Variable(problem.pack(phi0), dtype=dtype, trainable=True)
    T = int(problem.t_sequence.shape[1])
    step = horizon - warm_steps
    starts = list(range(0, T, step))

    history: List[Dict[str, float]] = []
    t_begin = time.perf_counter()
    last_window_loss = float("nan")
    for epoch in range(epochs):
        carry: Optional[Tuple[List[tf.Tensor], List[tf.Tensor]]] = None
        window_losses: List[float] = []
        for s in starts:
            e = min(s + horizon, T)
            split = min(s + step, e)
            t_a = problem.t_sequence[:, s:split]
            in_a = None if problem.inputs is None else problem.inputs[:, s:split]
            t_b = problem.t_sequence[:, split:e]
            in_b = None if problem.inputs is None else problem.inputs[:, split:e]
            problem._apply_phi(phi_var)
            with tf.GradientTape() as tape:
                theta = problem.parameter_space.unpack(phi_var)
                base = carry if carry is not None else problem.initial_state
                init, x0_map = _materialize_initial_state(
                    problem.network, problem.parameter_space, theta, base
                )
                # Часть A: [s, split) — несёт carry для следующего окна.
                out_a = problem.network(t_a, inputs=in_a, initial_state=init)
                loss_a = _segment_loss(problem, problem.objective, out_a, s, split)
                problem._counters["n_forward"] += 1
                # Часть B: [split, e) — хвост окна (пуст при warm_steps=0
                # или укороченном последнем окне).
                if split < e:
                    out_b = problem.network(
                        t_b, inputs=in_b, initial_state=out_a.final_state
                    )
                    loss_b = _segment_loss(problem, problem.objective, out_b, split, e)
                    problem._counters["n_forward"] += 1
                    loss_t = loss_a + loss_b
                else:
                    loss_t = loss_a
            grad_t = _phi_gradient(
                problem.parameter_space,
                problem.grad_mode,
                tape,
                loss_t,
                phi_var,
                x0_map,
            )
            problem._counters["n_gradient"] += 1
            if grad_t is None:
                grad_t = tf.zeros([problem.size], dtype=dtype)
            optimizer.apply_gradients([(grad_t, phi_var)])
            # Carry с оборванным градиентом — truncation boundary.
            carry = (
                [
                    tf.stop_gradient(cast_if_needed(t, dtype))
                    for t in out_a.final_state[0]
                ],
                [
                    tf.stop_gradient(cast_if_needed(t, dtype))
                    for t in out_a.final_state[1]
                ],
            )
            window_losses.append(float(loss_t))
        last_window_loss = float(window_losses[-1])
        history.append({"epoch": float(epoch), "loss": float(np.mean(window_losses))})

    # Финальный свободный прогон без обучения — обязательная проверка режима.
    free_eval = problem.evaluate(phi_var)
    problem._counters["wall_time"] += time.perf_counter() - t_begin

    notes = (
        "Truncated BPTT — приближённый режим: tf.stop_gradient на границе "
        f"окна (горизонт градиента ≤ {horizon} шагов); чувствительность "
        "стартового состояния окна в градиент не входит. "
        f"warm_steps={warm_steps} (шаг окна {step}): при warm_steps>0 окна "
        "перекрываются; перенос состояния time-consistent — carry берётся из "
        "снапшота середины окна (прогон в двух частях), перекрывающиеся шаги "
        "пересимулируются с независимым градиентным путём от своего carry. "
        "final_loss — последнее обучающее окно, free_run_loss — полный "
        "прогон после обучения; значения сопоставлять через free_run_loss."
    )
    return TruncatedResult(
        history=history,
        final_loss=last_window_loss,
        free_run_loss=float(free_eval.loss),
        notes=notes,
    )
