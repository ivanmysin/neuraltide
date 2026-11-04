"""
Локальные оптимизаторы поверх CalibrationProblem (P1, ранг 5 дорожной карты).

Два SciPy-адаптера с единым контрактом:

- minimize_lbfgsb — scipy.optimize.minimize(method='L-BFGS-B', jac=True)
  поверх problem.value_and_gradient: скалярная потеря и градиент одним
  прогоном; bounds берутся из problem.bounds_phi() (None, если ни одна
  координата не ограничена); ftol=None — dtype-aware auto
  (float64 → 2.2e-9 — прежнее поведение; float32 → ≈eps32, 1.19e-7),
  явный ftol/options['ftol'] всегда чтится (options['ftol']=None — как
  «не задано»), resolved ftol зафиксирован в config['options']['ftol'],
  None-значения options в scipy не передаются; float32-φ с эффективным
  ftol < 1e-7 — одноразовый UserWarning (риск
  ABNORMAL_TERMINATION_IN_LNSRCH), значение не переопределяем;
- least_squares — scipy.optimize.least_squares поверх problem.residual:
  'trf' как основной метод (поддерживает bounds), 'lm' только без bounds
  (при конечных границах честный ValueError с советом перейти на 'trf');
  jacobian: None | '2-point' | '3-point' | callable(x) -> [m, n].
  При jacobian=None по умолчанию используется forward-якобиан через
  neuraltide.optimization.sensitivities.residual_jacobian, если
  problem.size <= jacobian_max_params (по умолчанию 200) и задача
  реализует его приватный контракт (полный CalibrationProblem); иначе, а
  также при недоступности sensitivities, — честный fallback на '2-point'.
  Forward-режим несовместим с предшествующими gradient-прогонами в
  пределах одного NetworkRNN (TF forward-AD state): если
  problem.n_gradient > 0, поднимается UserWarning и выполняется честный
  fallback на '2-point' (фактический режим — в config['jacobian']).
  Пользовательский callable лениво связывается с
  sensitivities.residual_jacobian; фактический режим фиксируется в
  config['jacobian'] ('forward' | '2-point' | '3-point' | 'callable').
  maxiter — унифицированный бюджет итераций (как у minimize_lbfgsb):
  при заданном maxiter и max_nfev=None бюджет оценок невязок маппируется
  как max_nfev = maxiter * (problem.size + 2) (~n+1 оценка невязок на
  итерацию: n на 2-point якобиан + пробные шаги); одновременная передача
  maxiter и max_nfev отклоняется ValueError.
  callback(intermediate_result) — пользовательский callback итераций
  (поддержка scipy >= 1.16; на 1.15 не передаётся в scipy, история тогда
  стартовая + финальная; у method='lm' не поддержан scipy — UserWarning);
  scipy-defaults ftol/xtol/gtol не меняем, но явный kwargs['ftol'] с
  float32-φ < 1e-7 — одноразовый UserWarning (как у minimize_lbfgsb).

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
import warnings
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import scipy
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

# Дефолты dtype-aware ftol для minimize_lbfgsb (ftol=None — auto):
# float64 — прежнее поведение scipy (2.2e-9), float32 — ~eps32: жёстче
# порога float32-потеря не сходится, L-BFGS-B уходит в
# ABNORMAL_TERMINATION_IN_LNSRCH.
_FTOL_AUTO_FLOAT64 = 2.2e-9
_FTOL_AUTO_FLOAT32 = float(np.finfo(np.float32).eps)  # ≈ 1.19e-7
# Эффективный ftol ниже порога при float32-φ — повод для одноразового
# UserWarning (значение не переопределяем).
_FTOL_FLOAT32_WARN_BELOW = 1e-7


def _phi_dtype_is_float32(phi_dtype) -> bool:
    """True, если dtype φ-тензора — float32 (tf.DType/np.dtype/str)."""
    return str(phi_dtype) == "float32"


def _warn_ftol_float32(phi_dtype, ftol, *, where: str, stacklevel: int = 3) -> None:
    """
    Одноразовый UserWarning: эффективный ftol < 1e-7 при float32-координатах φ.

    Из-за точности float32 потеря не способна сойтись до такого порога:
    L-BFGS-B рискует завершиться ABNORMAL_TERMINATION_IN_LNSRCH, а TRF/LM
    будут итероваться впустую. Явный ftol чтится — значение НЕ
    переопределяем, только предупреждаем.
    """
    if not _phi_dtype_is_float32(phi_dtype):
        return
    if float(ftol) >= _FTOL_FLOAT32_WARN_BELOW:
        return
    warnings.warn(
        f"{where}: эффективный ftol={float(ftol):g} ниже порога "
        f"{_FTOL_FLOAT32_WARN_BELOW:g} при float32-координатах φ: потеря не "
        "сойдётся до такого ftol из-за точности float32 — L-BFGS-B рискует "
        "завершиться ABNORMAL_TERMINATION_IN_LNSRCH (TRF/LM будут "
        "итероваться впустую). Значение ftol оставлено без изменений; "
        "рекомендуется ftol >= 1e-7 для float32 или dtype float64.",
        UserWarning,
        stacklevel=stacklevel,
    )


def _resolve_lbfgsb_ftol(ftol, options, phi_dtype) -> float:
    """
    Эффективный ftol L-BFGS-B.

    Приоритет: options['ftol'] > явный ftol > dtype-aware auto
    (float64 → 2.2e-9 — прежнее поведение; float32 → ≈eps32 1.19e-7).
    Ключ options['ftol']=None трактуется как «не задано» — fallback на
    явный ftol/auto. Значение фиксируется в config['options']['ftol'];
    при float32 со значением < 1e-7 — одноразовый UserWarning (см.
    _warn_ftol_float32).
    """
    if options and options.get("ftol") is not None:
        effective = float(options["ftol"])
    elif ftol is not None:
        effective = float(ftol)
    elif _phi_dtype_is_float32(phi_dtype):
        effective = _FTOL_AUTO_FLOAT32
    else:
        effective = _FTOL_AUTO_FLOAT64
    _warn_ftol_float32(phi_dtype, effective, where="minimize_lbfgsb")
    return effective


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


def _initial_phi(problem, phi0) -> Tuple[np.ndarray, Any]:
    """
    Стартовый вектор φ; pack() вызывается ровно один раз.

    Returns:
        (phi_np, phi_dtype): numpy-float64 вектор и dtype исходного
        φ-тензора (нужен для dtype-aware дефолтов ftol, см.
        _resolve_lbfgsb_ftol).
    """
    phi_t = problem.pack() if phi0 is None else problem.pack(phi0)
    phi_arr = np.asarray(phi_t)
    phi_np = np.asarray(phi_arr, dtype=np.float64).reshape(-1)
    return phi_np, phi_arr.dtype


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
) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, np.ndarray]], Any]:
    """
    Стартовый вектор φ + bounds + dtype φ.

    Нефинитный старт отклоняется ValueError. Bounds приводятся к
    trainable-only размерности задачи: non-trainable записи ParameterSpace
    вне φ не должны ломать адаптеры.
    """
    x0, phi_dtype = _initial_phi(problem, phi0)
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
    return x0, bounds, phi_dtype


def _coordinate_names(problem, size: int) -> List[str]:
    """
    Имена координат φ: trainable-имена ParameterSpace (для bounds_active).

    Имена строятся по ``space.trainable_names`` (координаты φ) — non-trainable
    записи реестра в вектор φ не входят и bounds/history не ломают.
    """
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

    trainable = getattr(space, "trainable_names", None)
    if trainable is None:
        trainable = list(space.entries.keys())
    names = _entry_names(list(trainable))
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


def _sensitivities_residual_jacobian():
    """Ленивый доступ к sensitivities.residual_jacobian; None при недоступности."""
    try:
        from neuraltide.optimization import sensitivities
    except ImportError:
        return None
    return getattr(sensitivities, "residual_jacobian", None)


def _callable_jacobian(problem, jacobian):
    """
    scipy-обёртка для callable jacobian.

    Если передан именно sensitivities.residual_jacobian (импортируется
    лениво), он связывается с problem: jac(x) = rj(problem, x). Иначе
    пользовательский callable используется как есть: jac(x) -> [m, n].
    """
    residual_jacobian = _sensitivities_residual_jacobian()

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


def _default_forward_jacobian(problem, residual_jacobian, max_params: int):
    """Связка sensitivities.residual_jacobian с problem для default-режима."""

    def _jac(x) -> np.ndarray:
        return np.asarray(
            residual_jacobian(
                problem, np.asarray(x, dtype=np.float64), max_params=max_params
            ).J,
            dtype=np.float64,
        )

    return _jac


# Атрибуты полного CalibrationProblem, без которых sensitivities.
# residual_jacobian/_forward_jvp_only работать не могут (приватный контракт
# модуля sensitivities; view/duck-typed задачи его не реализуют).
_FORWARD_JACOBIAN_REQUIRED_ATTRS = (
    "_counters",
    "parameter_space",
    "network",
    "t_sequence",
    "inputs",
    "objective",
    "pack",
    "residual",
    "n_forward",
)


def _supports_forward_jacobian(problem) -> bool:
    """
    True, если problem совместима с sensitivities.residual_jacobian.

    Требуется (а) набор атрибутов полного CalibrationProblem и (б) честный
    residual-контракт CalibrationProblem: sensitivities дифференцируют
    objective.residual(problem.network(...)), что совпадает с якобианом
    problem.residual только для самого CalibrationProblem — обёртки
    (ConstrainedProblem с блоками ограничений, FrozenProblem и пр.)
    дополняют/редуцируют невязки, forward-якобиан для них несогласован.
    """
    if not all(hasattr(problem, attr) for attr in _FORWARD_JACOBIAN_REQUIRED_ATTRS):
        return False
    try:
        from neuraltide.optimization.problem import CalibrationProblem
    except ImportError:
        return False
    return getattr(type(problem), "residual", None) is CalibrationProblem.residual


def _resolve_jacobian(problem, jacobian, jacobian_max_params: int):
    """
    Выбор фактического jacobian для scipy и строки режима для config.

    Возвращает (jac, mode), где mode — 'forward' | '2-point' | '3-point' |
    'callable'. jacobian=None при size <= jacobian_max_params лениво
    связывается с sensitivities.residual_jacobian (режим 'forward'), если
    problem реализует его приватный контракт (полный CalibrationProblem);
    иначе (превышение порога, недоступность sensitivities, view/duck-typed
    задача без _counters/objective/network/...) — честный fallback на
    '2-point'. Дополнительный fallback на '2-point' с UserWarning —
    при problem.n_gradient > 0 (forward-AD несовместим с предшествующими
    gradient-прогонами на том же NetworkRNN). Пользовательские строки
    '2-point'/'3-point' и callable сохраняют приоритет и поведение.
    """
    if jacobian is None:
        size = int(getattr(problem, "size", 0) or 0)
        if size <= int(jacobian_max_params) and _supports_forward_jacobian(problem):
            residual_jacobian = _sensitivities_residual_jacobian()
            if residual_jacobian is not None:
                if int(getattr(problem, "n_gradient", 0) or 0) > 0:
                    # TF forward-AD несовместим с предшествующими
                    # gradient-прогонами (BPTT) в пределах одного NetworkRNN:
                    # JVP-прогон падал бы с загадочной ошибкой — честный
                    # fallback на конечные разности с понятным предупреждением.
                    warnings.warn(
                        "least_squares: default forward-якобиан несовместим "
                        "с предшествующими gradient-прогонами на этом "
                        "экземпляре problem (TF forward-AD state одного "
                        "NetworkRNN); используется fallback на '2-point'. "
                        "Для forward-режима передайте jacobian=None на свежем "
                        "экземпляре problem. Фактический режим зафиксирован "
                        "в config['jacobian'].",
                        UserWarning,
                        stacklevel=3,
                    )
                    return "2-point", "2-point"
                return (
                    _default_forward_jacobian(
                        problem, residual_jacobian, int(jacobian_max_params)
                    ),
                    "forward",
                )
        return "2-point", "2-point"
    if isinstance(jacobian, str):
        if jacobian in ("2-point", "3-point"):
            return jacobian, jacobian
        raise ValueError(
            "least_squares: jacobian строка должна быть '2-point' или "
            f"'3-point', получена {jacobian!r}."
        )
    if callable(jacobian):
        return _callable_jacobian(problem, jacobian), "callable"
    raise TypeError(
        "least_squares: jacobian должен быть None, '2-point', '3-point' "
        f"или callable(x) -> [m, n]; получен {type(jacobian).__name__}."
    )


def _least_squares_callback_supported() -> bool:
    """scipy >= 1.16 поддерживает callback у optimize.least_squares."""
    try:
        major, minor = scipy.__version__.split(".")[:2]
        return (int(major), int(minor)) >= (1, 16)
    except (AttributeError, TypeError, ValueError):
        # Неизвестный формат версии: безопасный путь — не передавать
        # callback (scipy без поддержки упал бы с TypeError).
        return False


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
    ftol: Optional[float] = None,
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
        maxiter / gtol: критерии остановки L-BFGS-B (передаются в options
            решателя);
        ftol: критерий остановки L-BFGS-B; None — dtype-aware auto
            (float64 → 2.2e-9 — прежнее поведение; float32 → ≈eps32,
            1.19e-7). Явный ftol или options['ftol'] всегда чтится
            (приоритет у options; options['ftol']=None трактуется как
            «не задано» — fallback на явный ftol/auto). Если dtype φ
            float32 и эффективный ftol < 1e-7 — одноразовый UserWarning
            о риске ABNORMAL_TERMINATION_IN_LNSRCH; значение не
            переопределяем;
        validate: вызывать problem.validate(phi_best) для held-out
            протоколов (результат — в validation; без held_out — {});
        options: дополнительные ключи options scipy (переопределяют
            maxiter/gtol/ftol при совпадении). Ключи со значением None
            игнорируются как «не задано» (options={'ftol': None} — то же,
            что отсутствие ключа: resolved-значение сохраняется).

    Returns:
        LocalOptimizerResult с base=OptimizationResult
        (seed=None, dtype — текущий глобальный); resolved ftol зафиксирован
        в config['options']['ftol'].

    Raises:
        ValueError: нефинитный стартовый phi0/loss/градиент или
            нефинитный результат решателя — до/после запуска с
            диагностикой, без больших штрафов.
    """
    t_start = time.perf_counter()
    start_counters = _counters(problem)
    size = int(problem.size)
    x0, bounds, phi_dtype = _prepare_start(problem, phi0, size)

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
        "ftol": _resolve_lbfgsb_ftol(ftol, options, phi_dtype),
    }
    if options:
        # Ключи options со значением None трактуем как «не задано»
        # (например, options={'ftol': None}): resolved-значения не
        # перезатираем и None в scipy не передаём (scipy падает с
        # TypeError при float(None)).
        user_options = {k: v for k, v in dict(options).items() if v is not None}
        scipy_options.update(user_options)
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
    jacobian_max_params: int = 200,
    callback=None,
    x_scale="jac",
    max_nfev: Optional[int] = None,
    maxiter: Optional[int] = None,
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
            None (по умолчанию) — forward-якобиан через
            sensitivities.residual_jacobian при problem.size <=
            jacobian_max_params и полном контракте CalibrationProblem
            (иначе, как и при недоступности sensitivities, — честный
            fallback на '2-point'). Callable-режим лениво связывается с
            sensitivities.residual_jacobian (см. _callable_jacobian);
        jacobian_max_params: порог размерности для default forward-режима;
        callback: None | callable(intermediate_result) — пользовательский
            callback итераций (поддержка scipy >= 1.16; на 1.15 не
            передаётся в scipy, история тогда стартовая + финальная);
        x_scale: масштаб координат (по умолчанию 'jac'; для 'lm'
            игнорируется scipy);
        max_nfev: бюджет оценок невязок (None — без ограничения);
        maxiter: унифицированный бюджет итераций (как у minimize_lbfgsb;
            keyword-only). При заданном maxiter и max_nfev=None бюджет
            оценок невязок маппируется как
            max_nfev = maxiter * (problem.size + 2) (~n+1 оценка невязок
            на итерацию: n на конечно-разностный якобиан + пробные шаги;
            scipy у least_squares maxiter не поддерживает). maxiter и
            max_nfev одновременно не принимаются (ValueError);
        validate: вызывать problem.validate(phi_best) для held-out
            протоколов (результат — в validation; без held_out — {});
        **kwargs: прочие параметры scipy.optimize.least_squares
            (ftol/xtol/gtol, tr_solver, diff_step, ...). scipy-defaults не
            меняем; если kwargs содержит ftol и dtype φ float32 со
            значением < 1e-7 — одноразовый UserWarning (риск итераций
            впустую), значение не переопределяем.

    Returns:
        LocalOptimizerResult с base=OptimizationResult (seed=None,
        dtype — текущий глобальный); loss_best == 0.5*||r*||^2;
        config['jacobian'] фиксирует фактически использованный режим
        ('forward' | '2-point' | '3-point' | 'callable'), config['maxiter']
        и config['max_nfev'] — запрошенный/эффективный бюджеты.

    Raises:
        TypeError: problem не предоставляет residual(phi);
        ValueError: нефинитный стартовый phi0/невязка, 'lm' с bounds,
            одновременно заданные maxiter/max_nfev, maxiter < 1 или
            неподдержанные method/jacobian.

    Предупреждения (UserWarning): forward-режим после gradient-прогонов
    на том же экземпляре problem (problem.n_gradient > 0) — честный
    fallback на '2-point'; callback на scipy < 1.16 или с method='lm' —
    callback не передаётся в scipy.
    """
    t_start = time.perf_counter()
    start_counters = _counters(problem)
    size = int(problem.size)
    method_key = str(method).lower()
    if method_key not in ("trf", "lm"):
        raise ValueError(
            f"least_squares: неподдержанный method={method!r}; доступны 'trf' и 'lm'."
        )
    if maxiter is not None:
        maxiter = int(maxiter)
        if maxiter < 1:
            raise ValueError(
                f"least_squares: maxiter должен быть >= 1, получен {maxiter}."
            )
        if max_nfev is not None:
            raise ValueError(
                "least_squares: maxiter и max_nfev заданы одновременно — "
                "бюджет неоднозначен. Укажите только один из двух "
                "(maxiter маппируется в max_nfev, см. документацию)."
            )
    x0, bounds, phi_dtype = _prepare_start(problem, phi0, size)
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
    jac, jac_mode = _resolve_jacobian(problem, jacobian, jacobian_max_params)
    history: List[Dict[str, Any]] = [
        {
            "loss": float(0.5 * np.dot(r0, r0)),
            "residual_norm": float(np.linalg.norm(r0)),
        }
    ]

    def _history_callback(intermediate_result) -> None:
        # Имя параметра значимо: scipy >= 1.16 передаёт OptimizeResult
        # только при сигнатуре callback(intermediate_result); иначе — x.
        r = getattr(intermediate_result, "fun", None)
        if r is None:
            r = fun.last_residuals
        if r is not None:
            r_np = np.asarray(r, dtype=np.float64)
            history.append(
                {
                    "loss": float(0.5 * np.dot(r_np, r_np)),
                    "residual_norm": float(np.linalg.norm(r_np)),
                }
            )
        if callback is not None:
            callback(intermediate_result)

    call_kwargs: Dict[str, Any] = dict(kwargs)
    call_kwargs["x_scale"] = x_scale
    # maxiter — унифицированный контракт бюджета (strategies передаёт его
    # screen_local оптимизаторам наравне с minimize_lbfgsb); scipy у
    # least_squares maxiter не поддерживает, маппируем в max_nfev.
    max_nfev_effective = max_nfev
    if maxiter is not None:
        max_nfev_effective = int(maxiter) * (size + 2)
    if max_nfev_effective is not None:
        call_kwargs["max_nfev"] = int(max_nfev_effective)
    if bounds is not None:
        call_kwargs["bounds"] = (bounds[0], bounds[1])
    # callback у scipy.optimize.least_squares появился в 1.16; у 'lm' он
    # по-прежнему не поддержан (scipy сам предупреждает и игнорирует) —
    # предупреждаем явно, чтобы молчаливая потеря callback не была сюрпризом.
    callback_supported = _least_squares_callback_supported()
    if callback_supported and method_key == "trf":
        call_kwargs["callback"] = _history_callback
    elif callback is not None:
        if not callback_supported:
            warnings.warn(
                "least_squares: callback требует scipy>=1.16 "
                f"(текущая версия {scipy.__version__}); callback игнорируется, "
                "история будет содержать стартовую и финальную записи.",
                UserWarning,
                stacklevel=2,
            )
        else:
            warnings.warn(
                "least_squares: callback не поддержан методом 'lm' "
                "(scipy.optimize.least_squares не вызывает callback у lm); "
                "callback пропущен, история будет содержать стартовую и "
                "финальную записи. Используйте method='trf' для callback.",
                UserWarning,
                stacklevel=2,
            )
    if kwargs.get("ftol") is not None:
        # Явный ftol чтим (scipy-defaults не меняем); при float32-φ со
        # значением < 1e-7 — тот же одноразовый warning, что и в
        # minimize_lbfgsb (риск бесконечных итераций впустую). Порядок
        # важен: warning идёт после всех ранних валидаций (method/bounds,
        # старт/невязка, jacobian), чтобы не всплывал там, где запуск
        # всё равно завершится ValueError.
        _warn_ftol_float32(phi_dtype, kwargs["ftol"], where="least_squares")
    result = sp_optimize.least_squares(
        fun,
        x0,
        jac=jac,
        method=method_key,
        **call_kwargs,
    )
    if len(history) == 1:
        # Per-iteration callback недоступен (scipy < 1.16 или method='lm'):
        # дописываем финальную точку, чтобы история не вырождалась в старт.
        r_fin = np.asarray(result.fun, dtype=np.float64)
        history.append(
            {
                "loss": float(result.cost),
                "residual_norm": float(np.linalg.norm(r_fin)),
            }
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
        "jacobian": jac_mode,
        "jacobian_max_params": int(jacobian_max_params),
        "callback": callback is not None,
        "x_scale": _jsonable(x_scale),
        "maxiter": None if maxiter is None else int(maxiter),
        "max_nfev": (None if max_nfev_effective is None else int(max_nfev_effective)),
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
