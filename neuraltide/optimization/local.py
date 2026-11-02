"""
Локальные оптимизаторы поверх CalibrationProblem (P1, ранг 5 дорожной карты).

Два SciPy-адаптера с единым контрактом:

- minimize_lbfgsb — scipy.optimize.minimize(method='L-BFGS-B', jac=True)
  поверх problem.value_and_gradient: скалярная потеря и градиент одним
  прогоном; bounds берутся из problem.bounds_phi() (None, если ни одна
  координата не ограничена);
- least_squares — scipy.optimize.least_squares поверх problem.residual:
  'trf' как основной метод (поддерживает bounds), 'lm' только без bounds
  (при конечных границах честный ValueError с советом перейти на 'trf');
  jacobian: None | '2-point' | '3-point' | callable(x) -> [m, n].
  Callable-режим лениво связывается с
  neuraltide.optimization.sensitivities.residual_jacobian (модуль
  появляется параллельно; его отсутствие не мешает остальным режимам,
  а пользовательский callable используется как есть).

Оба адаптера:

- принимают phi0=None как problem.pack() (текущее состояние модели);
- проверяют финитность стартовой точки/loss/невязок ДО запуска
  решателя (ValueError вместо безмолвных больших штрафов);
- возвращают LocalOptimizerResult (results.py): счётчики
  n_forward/n_gradient — дельта счётчиков problem за запуск, wall_time,
  bounds_active — имена координат ParameterSpace, прижатых к границе в
  оптимуме (допуск по bounds), validation = problem.validate(phi_best)
  при validate=True и непустом held_out, иначе пустой dict;
- заполняют base=OptimizationResult (phi_best/theta_best/loss_best/
  history/seed=None/config/versions/dtype).

Детерминированность: при фиксированном phi0 и options повторный запуск
даёт тот же результат (SciPy-решатели детерминированы; seed не нужен).

Модель после solve остаётся в точке последней оценённой θ согласно
контракту мутации CalibrationProblem; адаптеры не вносят дополнительных
мутаций (theta_best строится чистым unpack).
"""

import time
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
from scipy import optimize as sp_optimize

import neuraltide.config
from neuraltide.optimization.results import (
    LocalOptimizerResult,
    OptimizationResult,
)

__all__ = ["minimize_lbfgsb", "least_squares"]

# Допуск определения «координата на границе»: |phi - bound| <= atol + rtol|b|.
_BOUNDS_RTOL = 1e-6
_BOUNDS_ATOL = 1e-8


def _jsonable(value: Any) -> Any:
    """Рекурсивное приведение config к JSON-сериализуемому виду."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _counters(problem) -> Tuple[int, int]:
    return (
        int(getattr(problem, "n_forward", 0)),
        int(getattr(problem, "n_gradient", 0)),
    )


def _initial_phi(problem, phi0) -> np.ndarray:
    if phi0 is None:
        phi_t = problem.pack()
    else:
        phi_t = problem.pack(phi0)
    return np.asarray(phi_t, dtype=np.float64).reshape(-1)


def _problem_bounds(problem, size: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Границы (lb, ub) из problem.bounds_phi(); None, если все бесконечны."""
    bounds_fn = getattr(problem, "bounds_phi", None)
    if bounds_fn is None:
        return None
    lb, ub = bounds_fn()
    lb = np.asarray(lb, dtype=np.float64).reshape(-1)
    ub = np.asarray(ub, dtype=np.float64).reshape(-1)
    if lb.shape[0] != size or ub.shape[0] != size:
        raise ValueError(
            "local: problem.bounds_phi() вернул границы размера "
            f"({lb.shape[0]}, {ub.shape[0]}), ожидается {size}."
        )
    if not (np.any(np.isfinite(lb)) or np.any(np.isfinite(ub))):
        return None
    return lb, ub


def _prepare_start(
    problem, phi0, size: int
) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, np.ndarray]]]:
    """Стартовый вектор φ + bounds; нефинитный старт отклоняется ValueError."""
    x0 = _initial_phi(problem, phi0)
    if not np.all(np.isfinite(x0)):
        raise ValueError(
            "local: стартовый phi0 содержит нефинитные значения (NaN/inf); "
            "запуск решателя отклонён."
        )
    bounds = _problem_bounds(problem, size)
    if bounds is not None:
        # Проекция недопустимой стартовой точки в допустимую область
        # (L-BFGS-B делает это внутри себя, TRF честно падает).
        x0 = np.clip(x0, bounds[0], bounds[1])
    return x0, bounds


def _coordinate_names(problem, size: int) -> List[str]:
    """Имена координат φ: имена записей ParameterSpace (для bounds_active)."""
    space = getattr(problem, "parameter_space", None)
    if space is None:
        return [f"phi[{i}]" for i in range(size)]

    def _entry_names(names: List[str]) -> List[str]:
        expanded: List[str] = []
        for name in names:
            entry = space.entries[name]
            if entry.numel == 1:
                expanded.append(str(name))
            else:
                expanded.extend(f"{name}[{i}]" for i in range(entry.numel))
        return expanded

    names = _entry_names(list(space.entries.keys()))
    if len(names) != size:
        # Reduced φ (FrozenProblem): координаты — свободные записи
        # в их порядке.
        free = getattr(problem, "free_names", None)
        if free is None:
            return [f"phi[{i}]" for i in range(size)]
        names = _entry_names(list(free))
    return names


def _bounds_active(
    problem,
    phi: np.ndarray,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]],
    size: int,
) -> List[str]:
    """Имена координат, прижатых к конечной границе в оптимуме."""
    if bounds is None:
        return []
    lb, ub = bounds
    names = _coordinate_names(problem, size)
    active = np.zeros(size, dtype=bool)
    finite_lo = np.isfinite(lb)
    if np.any(finite_lo):
        margin_lo = _BOUNDS_ATOL + _BOUNDS_RTOL * np.abs(lb[finite_lo])
        active[finite_lo] |= phi[finite_lo] <= lb[finite_lo] + margin_lo
    finite_hi = np.isfinite(ub)
    if np.any(finite_hi):
        margin_hi = _BOUNDS_ATOL + _BOUNDS_RTOL * np.abs(ub[finite_hi])
        active[finite_hi] |= phi[finite_hi] >= ub[finite_hi] - margin_hi
    return [names[int(i)] for i in np.nonzero(active)[0]]


class _LossGradientFn:
    """numpy-float64 обёртка problem.value_and_gradient для scipy (jac=True)."""

    def __init__(self, problem):
        self._problem = problem
        self.last_loss: Optional[float] = None
        self.last_grad: Optional[np.ndarray] = None

    def __call__(self, x) -> Tuple[float, np.ndarray]:
        eval_result, grad = self._problem.value_and_gradient(
            np.asarray(x, dtype=np.float64)
        )
        loss = float(eval_result.loss)
        grad_np = np.asarray(grad, dtype=np.float64).reshape(-1)
        self.last_loss = loss
        self.last_grad = grad_np
        return loss, grad_np


class _ResidualFn:
    """numpy-float64 обёртка problem.residual для scipy least_squares."""

    def __init__(self, problem):
        self._problem = problem
        self.last_residuals: Optional[np.ndarray] = None

    def __call__(self, x) -> np.ndarray:
        residuals = np.asarray(
            self._problem.residual(np.asarray(x, dtype=np.float64)),
            dtype=np.float64,
        ).reshape(-1)
        self.last_residuals = residuals
        return residuals


def _callable_jacobian(problem, jacobian):
    """
    scipy-обёртка для callable jacobian.

    Если передан именно sensitivities.residual_jacobian (импортируется
    лениво), он связывается с problem: jac(x) = rj(problem, x). Иначе
    пользовательский callable используется как есть: jac(x) -> [m, n].
    """
    try:
        from neuraltide.optimization import sensitivities
    except ImportError:
        sensitivities = None
    residual_jacobian = None
    if sensitivities is not None:
        residual_jacobian = getattr(sensitivities, "residual_jacobian", None)

    def _user_jac(x) -> np.ndarray:
        return np.asarray(jacobian(np.asarray(x, dtype=np.float64)), dtype=np.float64)

    if residual_jacobian is None or jacobian is not residual_jacobian:
        return _user_jac

    def _sens_jac(x) -> np.ndarray:
        # residual_jacobian возвращает JacobianResult (DTO) — scipy ждёт
        # матрицу [m, n], поэтому берём поле .J.
        return np.asarray(
            residual_jacobian(problem, np.asarray(x, dtype=np.float64)).J,
            dtype=np.float64,
        )

    return _sens_jac


def _resolve_jacobian(problem, jacobian):
    if jacobian is None:
        return None
    if isinstance(jacobian, str):
        if jacobian in ("2-point", "3-point"):
            return jacobian
        raise ValueError(
            "least_squares: jacobian строка должна быть '2-point' или "
            f"'3-point', получена {jacobian!r}."
        )
    if callable(jacobian):
        return _callable_jacobian(problem, jacobian)
    raise TypeError(
        "least_squares: jacobian должен быть None, '2-point', '3-point' "
        f"или callable(x) -> [m, n]; получен {type(jacobian).__name__}."
    )


def _build_result(
    *,
    problem,
    method: str,
    phi_best: np.ndarray,
    loss_best: float,
    history: List[Dict[str, Any]],
    config: Dict[str, Any],
    start_counters: Tuple[int, int],
    t_start: float,
    validate: bool,
    success: bool,
    status: int,
    message: str,
    nit: int,
    nfev: int,
    njev: int,
    grad_norm: float,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]],
) -> LocalOptimizerResult:
    """Сборка LocalOptimizerResult (+ base=OptimizationResult)."""
    size = int(problem.size)
    phi_best = np.asarray(phi_best, dtype=np.float64).reshape(-1)
    unpack = getattr(problem, "unpack", None)
    if callable(unpack):
        theta = unpack(phi_best)
    else:
        theta = problem.parameter_space.unpack(phi_best)
    theta_best = {str(k): np.asarray(v) for k, v in theta.items()}

    validation: Dict[str, float] = {}
    if validate:
        held_out = getattr(problem, "held_out", None)
        if held_out:
            validation = {
                str(k): float(v) for k, v in problem.validate(phi_best).items()
            }

    fwd0, grad0 = start_counters
    base = OptimizationResult(
        phi_best=phi_best,
        theta_best=theta_best,
        loss_best=float(loss_best),
        history=history,
        seed=None,
        config=config,
        dtype=str(neuraltide.config.get_dtype().name),
    )
    return LocalOptimizerResult(
        base=base,
        method=method,
        success=bool(success),
        status=int(status),
        message=str(message),
        nit=int(nit),
        nfev=int(nfev),
        njev=int(njev),
        grad_norm=float(grad_norm),
        bounds_active=_bounds_active(problem, phi_best, bounds, size),
        n_forward=_counters(problem)[0] - fwd0,
        n_gradient=_counters(problem)[1] - grad0,
        wall_time=time.perf_counter() - t_start,
        validation=validation,
    )


def minimize_lbfgsb(
    problem,
    phi0=None,
    *,
    maxiter: int = 200,
    gtol: float = 1e-6,
    ftol: float = 2.2e-9,
    validate: bool = True,
    options: Optional[Mapping[str, Any]] = None,
) -> LocalOptimizerResult:
    """
    Локальная минимизация скалярной потери методом L-BFGS-B.

    Args:
        problem: CalibrationProblem (или duck-typed аналог с
            value_and_gradient/pack/bounds_phi/counters);
        phi0: стартовый вектор φ; None — текущее значение параметров
            модели (problem.pack());
        maxiter / gtol / ftol: критерии остановки L-BFGS-B (передаются
            в options решателя);
        validate: вызывать problem.validate(phi_best) для held-out
            протоколов (результат — в validation; без held_out — {});
        options: дополнительные ключи options scipy (переопределяют
            maxiter/gtol/ftol при совпадении).

    Returns:
        LocalOptimizerResult с base=OptimizationResult
        (seed=None, dtype — текущий глобальный).

    Raises:
        ValueError: нефинитный стартовый phi0/loss/градиент или
            нефинитный результат решателя — до/после запуска с
            диагностикой, без больших штрафов.
    """
    t_start = time.perf_counter()
    start_counters = _counters(problem)
    size = int(problem.size)
    x0, bounds = _prepare_start(problem, phi0, size)

    fun = _LossGradientFn(problem)
    eval0, grad0_t = problem.value_and_gradient(x0)
    f0 = float(eval0.loss)
    g0 = np.asarray(grad0_t, dtype=np.float64).reshape(-1)
    if not np.isfinite(f0) or not np.all(np.isfinite(g0)):
        raise ValueError(
            "minimize_lbfgsb: стартовая точка нефинитна "
            f"(loss={f0}); запуск решателя отклонён."
        )
    history: List[Dict[str, Any]] = [
        {"nit": 0, "loss": f0, "grad_norm": float(np.linalg.norm(g0))}
    ]

    def _callback(arg) -> None:
        nit = getattr(arg, "nit", None)
        loss = getattr(arg, "fun", fun.last_loss)
        jac = getattr(arg, "jac", fun.last_grad)
        history.append(
            {
                "nit": int(nit) if nit is not None else len(history),
                "loss": float(loss),
                "grad_norm": (
                    None
                    if jac is None
                    else float(np.linalg.norm(np.asarray(jac, dtype=np.float64)))
                ),
            }
        )

    scipy_options: Dict[str, Any] = {
        "maxiter": int(maxiter),
        "gtol": float(gtol),
        "ftol": float(ftol),
    }
    if options:
        scipy_options.update(dict(options))
    bounds_arg = None if bounds is None else list(zip(bounds[0], bounds[1]))
    result = sp_optimize.minimize(
        fun,
        x0,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds_arg,
        options=scipy_options,
        callback=_callback,
    )

    phi_best = np.asarray(result.x, dtype=np.float64).reshape(-1)
    loss_best = float(result.fun)
    if not np.all(np.isfinite(phi_best)) or not np.isfinite(loss_best):
        raise ValueError(
            "minimize_lbfgsb: решатель вернул нефинитную точку/потерю "
            f"(loss={loss_best}); message: {result.message}"
        )
    grad_norm = float(np.linalg.norm(np.asarray(result.jac, dtype=np.float64)))
    config = {"method": "L-BFGS-B", "options": _jsonable(scipy_options)}
    return _build_result(
        problem=problem,
        method="L-BFGS-B",
        phi_best=phi_best,
        loss_best=loss_best,
        history=history,
        config=config,
        start_counters=start_counters,
        t_start=t_start,
        validate=validate,
        success=bool(result.success),
        status=int(result.status),
        message=str(result.message),
        nit=int(getattr(result, "nit", 0) or 0),
        nfev=int(result.nfev),
        njev=int(getattr(result, "njev", 0) or 0),
        grad_norm=grad_norm,
        bounds=bounds,
    )


def least_squares(
    problem,
    phi0=None,
    *,
    method: str = "trf",
    jacobian=None,
    x_scale="jac",
    max_nfev: Optional[int] = None,
    validate: bool = True,
    **kwargs,
) -> LocalOptimizerResult:
    """
    Нелинейные наименьшие квадраты поверх problem.residual (TRF/LM).

    Невязки — честный residual-интерфейс задачи (0.5*||r||^2 == loss,
    например ResidualLoss). Если problem/objective residual не даёт —
    TypeError с подсказкой использовать minimize_lbfgsb.

    Args:
        problem: CalibrationProblem (или duck-typed аналог с
            residual/pack/bounds_phi/counters);
        phi0: стартовый вектор φ; None — текущее значение параметров
            модели (problem.pack());
        method: 'trf' (основной, поддерживает bounds) или 'lm' (только
            без bounds; при конечных границах — ValueError с советом
            перейти на 'trf');
        jacobian: None | '2-point' | '3-point' | callable(x) -> [m, n].
            Callable-режим лениво связывается с
            sensitivities.residual_jacobian (см. _callable_jacobian);
        x_scale: масштаб координат (по умолчанию 'jac'; для 'lm'
            игнорируется scipy);
        max_nfev: бюджет оценок невязок (None — без ограничения);
        validate: вызывать problem.validate(phi_best) для held-out
            протоколов (результат — в validation; без held_out — {});
        **kwargs: прочие параметры scipy.optimize.least_squares
            (ftol/xtol/gtol, tr_solver, diff_step, ...).

    Returns:
        LocalOptimizerResult с base=OptimizationResult (seed=None,
        dtype — текущий глобальный); loss_best == 0.5*||r*||^2.

    Raises:
        TypeError: problem не предоставляет residual(phi);
        ValueError: нефинитный стартовый phi0/невязка, 'lm' с bounds
            или неподдержанные method/jacobian.
    """
    t_start = time.perf_counter()
    start_counters = _counters(problem)
    size = int(problem.size)
    method_key = str(method).lower()
    if method_key not in ("trf", "lm"):
        raise ValueError(
            f"least_squares: неподдержанный method={method!r}; доступны 'trf' и 'lm'."
        )
    x0, bounds = _prepare_start(problem, phi0, size)
    if method_key == "lm" and bounds is not None:
        raise ValueError(
            "least_squares: method='lm' не поддерживает bounds (у задачи "
            "заданы конечные границы). Используйте method='trf' — он "
            "работает с bounds, либо уберите ограничения параметров."
        )

    if not callable(getattr(problem, "residual", None)):
        raise TypeError(
            "least_squares: problem не предоставляет residual(phi) — "
            "least-squares постановка невозможна. Используйте "
            "minimize_lbfgsb для скалярной постановки, либо передайте "
            "objective с residual-интерфейсом (ResidualLoss)."
        )
    fun = _ResidualFn(problem)
    try:
        r0 = fun(x0)
    except TypeError as exc:
        raise TypeError(
            "least_squares: problem не предоставляет residual(phi) — "
            "least-squares постановка невозможна. Используйте "
            "minimize_lbfgsb для скалярной постановки, либо передайте "
            "objective с residual-интерфейсом (ResidualLoss)."
        ) from exc
    if not np.all(np.isfinite(r0)):
        raise ValueError(
            "least_squares: стартовая невязка нефинитна "
            f"(residual norm={float(np.linalg.norm(r0))}); "
            "запуск решателя отклонён."
        )
    jac = _resolve_jacobian(problem, jacobian)
    history: List[Dict[str, Any]] = [
        {
            "loss": float(0.5 * np.dot(r0, r0)),
            "residual_norm": float(np.linalg.norm(r0)),
        }
    ]

    def _callback(arg) -> None:
        del arg  # scipy передаёт x; loss берём из последнего вызова fun
        r = fun.last_residuals
        if r is not None:
            history.append(
                {
                    "loss": float(0.5 * np.dot(r, r)),
                    "residual_norm": float(np.linalg.norm(r)),
                }
            )

    call_kwargs: Dict[str, Any] = dict(kwargs)
    call_kwargs["x_scale"] = x_scale
    if max_nfev is not None:
        call_kwargs["max_nfev"] = int(max_nfev)
    if bounds is not None:
        call_kwargs["bounds"] = (bounds[0], bounds[1])
    result = sp_optimize.least_squares(
        fun,
        x0,
        jac="2-point" if jac is None else jac,
        method=method_key,
        **call_kwargs,
    )

    phi_best = np.asarray(result.x, dtype=np.float64).reshape(-1)
    loss_best = float(result.cost)  # 0.5 * ||r*||^2 == loss по контракту
    if not np.all(np.isfinite(phi_best)) or not np.isfinite(loss_best):
        raise ValueError(
            "least_squares: решатель вернул нефинитную точку/потерю "
            f"(loss={loss_best}); message: {result.message}"
        )
    grad = np.asarray(result.grad, dtype=np.float64).reshape(-1)
    grad_norm = float(np.linalg.norm(grad)) if grad.size == size else 0.0
    config = {
        "method": method_key,
        "jacobian": "callable" if callable(jacobian) else str(jacobian),
        "x_scale": _jsonable(x_scale),
        "max_nfev": None if max_nfev is None else int(max_nfev),
        "kwargs": _jsonable(kwargs),
    }
    return _build_result(
        problem=problem,
        method=method_key.upper(),
        phi_best=phi_best,
        loss_best=loss_best,
        history=history,
        config=config,
        start_counters=start_counters,
        t_start=t_start,
        validate=validate,
        success=bool(result.success),
        status=int(result.status),
        message=str(result.message),
        nit=int(result.get("nit") or 0),
        nfev=int(result.nfev),
        njev=int(result.get("njev") or 0),
        grad_norm=grad_norm,
        bounds=bounds,
    )
