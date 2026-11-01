"""
Численная верификация градиентов и постановки оптимизации.

Пользовательский набор проверок этапа P0 дорожной карты:

- finite_difference_scan — центральные конечные разности со сканированием h;
- taylor_test — направленный Taylor test (порядки 1 и 2 до численного пола);
- dot_product_test — согласованность прямого и сопряжённого операторов;
- dt_convergence — сходимость потери/градиента/выходов при уточнении dt;
- dtype_consistency_check — согласованность float32/float64 через config;
- compare_bptt_adjoint — равенство потери и градиентов BPTT и adjoint.

Отчёты — numpy-only (без pandas), в eager-режиме. Все функции возвращают
dict/list[dict] с float-значениями, пригодными для сериализации и assert'ов.
"""

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import tensorflow as tf

import neuraltide.config
from neuraltide.core.network import NetworkRNN
from neuraltide.core.types import TensorType
from neuraltide.training.adjoint import AdjointSolver
from neuraltide.training.losses import BaseLoss, CompositeLoss

__all__ = [
    "finite_difference_scan",
    "taylor_test",
    "dot_product_test",
    "dt_convergence",
    "dtype_consistency_check",
    "compare_bptt_adjoint",
]


# ── Внутренние helpers ───────────────────────────────────────────────────────


def _arr_stats(a: np.ndarray) -> Dict[str, float]:
    """Краткая сводка массива: mean/std/max|.|."""
    a = np.asarray(a, dtype=np.float64)
    return {
        "mean": float(a.mean()),
        "std": float(a.std()),
        "max_abs": float(np.max(np.abs(a))),
    }


def _summarize_outputs(outputs) -> Dict[str, Dict[str, float]]:
    """Сводка выходов: dict имя → stats либо stats единственного массива."""
    if isinstance(outputs, dict):
        return {str(k): _arr_stats(np.asarray(v)) for k, v in outputs.items()}
    return {"outputs": _arr_stats(np.asarray(outputs))}


def _default_h_values() -> np.ndarray:
    """Стандартная геометрическая сетка шагов h."""
    return np.logspace(-1, -6, 12)


def _loglog_slope(r0: float, r1: float, h0: float, h1: float) -> float:
    """Наклон между двумя точками в log-log шкале; nan при нулевых значениях."""
    if r0 <= 0.0 or r1 <= 0.0 or h0 <= 0.0 or h1 <= 0.0:
        return float("nan")
    return float(np.log(r0 / r1) / np.log(h0 / h1))


# ── 1. Конечные разности ─────────────────────────────────────────────────────


def finite_difference_scan(
    func: Callable[[np.ndarray], float],
    phi0: Sequence[float],
    h_values: Optional[Sequence[float]] = None,
) -> Dict[str, object]:
    """
    Центральные конечные разности градиента скалярной функции.

    Выполняет сканирование по шагу h (а не одну произвольную проверку):
    для каждого h градиент оценивается как (f(φ+h·eᵢ) − f(φ−h·eᵢ)) / (2h)
    по всем координатам.

    Args:
        func: f(φ) → scalar; принимает np.ndarray той же формы, что phi0.
        phi0: точка оценки градиента.
        h_values: шаг(и) h. По умолчанию 12 значений от 1e-1 до 1e-6.

    Returns:
        dict с ключами:
            phi0, f0, h_values, gradients (list[np.ndarray] — градиент для
            каждого h), best/best_h — оценка на плато перед численным полом
            (h, где соседние оценки максимально близки).
    """
    phi0 = np.asarray(phi0, dtype=np.float64)
    if h_values is None:
        h_values = _default_h_values()
    h_arr = np.atleast_1d(np.asarray(h_values, dtype=np.float64)).reshape(-1)

    f0 = float(func(phi0))
    gradients: List[np.ndarray] = []
    for h in h_arr:
        grad = np.zeros_like(phi0)
        it = np.nditer(phi0, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            step = np.zeros_like(phi0)
            step[idx] = h
            grad[idx] = (float(func(phi0 + step)) - float(func(phi0 - step))) / (
                2.0 * h
            )
        gradients.append(grad)

    if len(gradients) > 1:
        diffs = [
            float(np.max(np.abs(gradients[k + 1] - gradients[k])))
            for k in range(len(gradients) - 1)
        ]
        k_best = int(np.argmin(diffs))
    else:
        k_best = 0

    return {
        "phi0": phi0,
        "f0": f0,
        "h_values": h_arr,
        "gradients": gradients,
        "best": gradients[k_best],
        "best_h": float(h_arr[k_best]),
    }


# ── 2. Направленный Taylor test ──────────────────────────────────────────────


def taylor_test(
    value_and_grad: Callable[[np.ndarray], tuple],
    phi0: Sequence[float],
    v: Optional[Sequence[float]] = None,
    h_values: Optional[Sequence[float]] = None,
) -> Dict[str, object]:
    """
    Направленный Taylor test для value_and_grad.

    Для R0(h) = f(φ + h·v) − f(φ) ожидается O(h) (slope_R0 ≈ 1);
    для R1(h) = R0(h) − h·∇f(φ)ᵀv ожидается O(h²) (slope_R1 ≈ 2)
    до численного предала округления.

    Args:
        value_and_grad: f(φ) → (scalar, gradient).
        phi0: точка разложения.
        v: направление; по умолчанию фиксированный случайный unit-вектор.
        h_values: сетка h (по умолчанию от 1e-1 до 1e-6).

    Returns:
        dict с ключами phi0, v, f0, grad_dot_v, rows (список dict:
        h, R0, R1, slope_R0, slope_R1 для соседних пар), а также
        slope_R0_median, slope_R1_median — медианные наклоны.
    """
    phi0 = np.asarray(phi0, dtype=np.float64)
    if v is None:
        rng = np.random.default_rng(0)
        v = rng.standard_normal(phi0.shape)
        v = v / (np.linalg.norm(v) + 1e-300)
    v = np.asarray(v, dtype=np.float64)
    if h_values is None:
        h_values = _default_h_values()
    h_arr = np.atleast_1d(np.asarray(h_values, dtype=np.float64)).reshape(-1)

    f0, g0 = value_and_grad(phi0)
    f0 = float(f0)
    g0 = np.asarray(g0, dtype=np.float64)
    grad_dot_v = float(np.sum(g0 * v))

    rows: List[Dict[str, float]] = []
    for h in h_arr:
        fh = float(value_and_grad(phi0 + h * v)[0])
        R0 = fh - f0
        R1 = R0 - h * grad_dot_v
        rows.append({"h": float(h), "R0": R0, "R1": R1})

    for k in range(len(rows) - 1):
        r_a, r_b = rows[k], rows[k + 1]
        r_a["slope_R0"] = _loglog_slope(
            abs(r_a["R0"]), abs(r_b["R0"]), r_a["h"], r_b["h"]
        )
        r_a["slope_R1"] = _loglog_slope(
            abs(r_a["R1"]), abs(r_b["R1"]), r_a["h"], r_b["h"]
        )

    slopes0 = [r["slope_R0"] for r in rows[:-1] if np.isfinite(r["slope_R0"])]
    slopes1 = [r["slope_R1"] for r in rows[:-1] if np.isfinite(r["slope_R1"])]

    return {
        "phi0": phi0,
        "v": v,
        "f0": f0,
        "grad_dot_v": grad_dot_v,
        "rows": rows,
        "slope_R0_median": float(np.median(slopes0)) if slopes0 else float("nan"),
        "slope_R1_median": float(np.median(slopes1)) if slopes1 else float("nan"),
    }


# ── 3. Dot-product test ──────────────────────────────────────────────────────


def dot_product_test(
    jvp_fn: Callable[[np.ndarray], np.ndarray],
    vjp_fn: Callable[[np.ndarray], np.ndarray],
    x: Sequence[float],
    y: Sequence[float],
    tol: float = 1e-5,
) -> Dict[str, object]:
    """
    Проверка сопряжённости пар операторов: <Jx, y> против <x, Jᵀy>.

    Args:
        jvp_fn: x → J·x.
        vjp_fn: y → Jᵀ·y.
        x, y: аргументы проверки.
        tol: допуск относительной ошибки для флага passed.

    Returns:
        dict с lhs=<Jx,y>, rhs=<x,Jᵀy>, abs_diff, rel_error, passed.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    Jx = np.asarray(jvp_fn(x), dtype=np.float64)
    JTy = np.asarray(vjp_fn(y), dtype=np.float64)

    lhs = float(np.sum(Jx * y))
    rhs = float(np.sum(x * JTy))
    denom = max(abs(lhs), abs(rhs), 1e-300)
    rel_error = abs(lhs - rhs) / denom

    return {
        "lhs": lhs,
        "rhs": rhs,
        "abs_diff": abs(lhs - rhs),
        "rel_error": float(rel_error),
        "passed": bool(rel_error < tol),
    }


# ── 4. Сходимость по dt ──────────────────────────────────────────────────────


def dt_convergence(
    build_and_run: Callable[[float], Dict[str, object]],
    dts: Sequence[float],
) -> List[Dict[str, object]]:
    """
    Сводка потери, градиента и выходов по уточняющейся сетке dt.

    Args:
        build_and_run: callable(dt) → dict с опциональными ключами:
            "loss" (scalar, обязательный), "grads" (array-like),
            "outputs" (array-like или dict имя → array).
        dts: сетка шагов, например [dt, dt/2, dt/4].

    Returns:
        list[dict]: по строке на dt; каждая содержит dt, loss, опционально
        grad_norm, grad, output_summary, а со второй строки — loss_diff,
        loss_rel_diff и grad_diff относительно предыдущей строки.

    Note:
        Только для обычных интеграторов (Euler/Heun/RK4). У
        BrunelFokkerPlanck внутреннее CFL-субшагивание делает показатели
        по внешнему dt неинформативными — используйте внешнюю сетку
        субшагов самой модели, а не эту функцию.
    """
    rows: List[Dict[str, object]] = []
    prev: Optional[Dict[str, object]] = None
    for dt in dts:
        res = build_and_run(float(dt))
        row: Dict[str, object] = {"dt": float(dt), "loss": float(res["loss"])}
        if res.get("grads") is not None:
            g = np.asarray(res["grads"], dtype=np.float64).ravel()
            row["grad_norm"] = float(np.linalg.norm(g))
            row["grad"] = g
        if res.get("outputs") is not None:
            row["output_summary"] = _summarize_outputs(res["outputs"])
        if prev is not None:
            loss_diff = abs(row["loss"] - prev["loss"])
            row["loss_diff"] = loss_diff
            row["loss_rel_diff"] = loss_diff / max(abs(prev["loss"]), 1e-300)
            if "grad" in row and "grad" in prev:
                row["grad_diff"] = float(np.linalg.norm(row["grad"] - prev["grad"]))
        rows.append(row)
        prev = row
    return rows


# ── 5. Согласованность dtype ─────────────────────────────────────────────────


def dtype_consistency_check(
    build_and_run: Callable[[tf.DType], Dict[str, object]],
    dtypes: Sequence[tf.DType] = (tf.float32, tf.float64),
) -> List[Dict[str, object]]:
    """
    Запуск build_and_run при разных глобальных dtype через config.set_dtype.

    Args:
        build_and_run: callable(dtype) → dict с ключами как у dt_convergence
            ("loss" обязателен, "grads"/"outputs" опциональны). Объекты
            (популяции, синапсы, сеть) должны создаваться внутри вызова,
            после установки dtype.
        dtypes: проверяемые dtype (по умолчанию float32 и float64).

    Returns:
        list[dict]: по строке на dtype; в последней строке дополнительно
        loss_rel_diff_vs_first и grad_rel_diff_vs_first — различие
        относительно первого dtype. Прежний глобальный dtype восстанавливается.

    Note:
        conftest проекта сбрасывает dtype после каждого теста, поэтому
        восстановление здесь — защита от загрязнения состояния вне тестов.
    """
    prev_dtype = neuraltide.config.get_dtype()
    rows: List[Dict[str, object]] = []
    try:
        for dtype in dtypes:
            neuraltide.config.set_dtype(dtype)
            res = build_and_run(dtype)
            row: Dict[str, object] = {
                "dtype": str(dtype.name),
                "loss": float(res["loss"]),
            }
            if res.get("grads") is not None:
                g = np.asarray(res["grads"], dtype=np.float64).ravel()
                row["grad_norm"] = float(np.linalg.norm(g))
                row["grad"] = g
            if res.get("outputs") is not None:
                row["output_summary"] = _summarize_outputs(res["outputs"])
            rows.append(row)
    finally:
        neuraltide.config.set_dtype(prev_dtype)

    if len(rows) >= 2 and "grad" in rows[0] and "grad" in rows[-1]:
        first, last = rows[0], rows[-1]
        last["loss_rel_diff_vs_first"] = abs(last["loss"] - first["loss"]) / max(
            abs(first["loss"]), 1e-300
        )
        last["grad_rel_diff_vs_first"] = float(
            np.linalg.norm(last["grad"] - first["grad"])
            / max(np.linalg.norm(first["grad"]), 1e-300)
        )
    return rows


# ── 6. BPTT ↔ adjoint ────────────────────────────────────────────────────────


def _extract_targets_from_loss(loss_fn) -> Dict[str, TensorType]:
    """Собирает target из термов потери (атрибут target), как Trainer."""
    targets: Dict[str, TensorType] = {}
    terms = loss_fn.terms if isinstance(loss_fn, CompositeLoss) else [(1.0, loss_fn)]
    for _, sub in terms:
        tgt = getattr(sub, "target", None)
        if isinstance(tgt, dict):
            targets.update(tgt)
    return targets


def compare_bptt_adjoint(
    network: NetworkRNN,
    loss_fn: BaseLoss,
    t_seq: TensorType,
    inputs: Optional[TensorType] = None,
    target: Optional[Dict[str, TensorType]] = None,
    initial_state: Optional[tuple] = None,
) -> Dict[str, object]:
    """
    Сравнивает скалярную потерю и градиенты BPTT и adjoint на одной сети.

    Args:
        network: NetworkRNN (один и тот же объект для обоих методов).
        loss_fn: потеря с per_step_loss.
        t_seq: временная последовательность [batch, T, 1].
        inputs: упакованные внешние входы или None (если входов нет).
        target: целевые траектории; если None, собираются из атрибута
            target термов loss_fn.
        initial_state: опциональное начальное состояние (pop, syn).

    Returns:
        dict с ключами bptt_loss, adjoint_loss, loss_abs_diff,
        max_grad_abs_diff, max_grad_rel_diff, cosine_similarity и
        per_variable (list[dict] с именами и различиями по переменным).
    """
    if target is None:
        target = _extract_targets_from_loss(loss_fn)

    # BPTT
    with tf.GradientTape() as tape:
        output = network(
            t_seq, inputs=inputs, training=False, initial_state=initial_state
        )
        bptt_loss = loss_fn(output, network)
    bptt_grads = tape.gradient(bptt_loss, network.trainable_variables)

    # Adjoint
    solver = AdjointSolver(network, network._integrator)
    adj_grads, variables, adj_output = solver.compute_gradients(
        t_seq,
        inputs,
        target,
        loss_fn,
        initial_state=initial_state,
    )
    adjoint_loss = loss_fn(adj_output, network)

    per_variable: List[Dict[str, object]] = []
    max_abs = 0.0
    max_rel = 0.0
    dots = 0.0
    norm_b = 0.0
    norm_a = 0.0
    for var, g_b, g_a in zip(variables, bptt_grads, adj_grads):
        gb = (
            np.zeros(var.shape, dtype=np.float64)
            if g_b is None
            else np.asarray(g_b, dtype=np.float64)
        )
        ga = (
            np.zeros(var.shape, dtype=np.float64)
            if g_a is None
            else np.asarray(g_a, dtype=np.float64)
        )
        abs_diff = float(np.max(np.abs(gb - ga))) if gb.size else 0.0
        denom = np.max(np.abs(gb))
        rel_diff = abs_diff / max(float(denom), 1e-300)
        max_abs = max(max_abs, abs_diff)
        max_rel = max(max_rel, rel_diff)
        dots += float(np.sum(gb * ga))
        norm_b += float(np.sum(gb**2))
        norm_a += float(np.sum(ga**2))
        per_variable.append(
            {
                "name": var.name,
                "max_abs_diff": abs_diff,
                "max_rel_diff": rel_diff,
            }
        )

    cos = dots / np.sqrt(max(norm_b * norm_a, 1e-300))

    return {
        "bptt_loss": float(bptt_loss),
        "adjoint_loss": float(adjoint_loss),
        "loss_abs_diff": abs(float(bptt_loss) - float(adjoint_loss)),
        "max_grad_abs_diff": max_abs,
        "max_grad_rel_diff": max_rel,
        "cosine_similarity": float(cos),
        "per_variable": per_variable,
    }
