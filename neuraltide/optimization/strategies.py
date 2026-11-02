"""
Глобальные и гибридные стратегии калибровки (этап P1, ранг 6 дорожной карты).

- sample_phi0s — детерминированная (по seed) выборка начальных точек φ
  методами Sobol / Latin Hypercube / random поверх ParameterSpace:
  ограниченные записи семплируются равномерно (или лог-равномерно) в
  физических границах с последующим transform.inverse, неограниченные —
  гауссово вокруг текущего phi с предупреждением;
- multistart — screening всех стартов одним problem.evaluate на старт,
  полная локальная доводка top-k кандидатов через адаптеры local.py
  (или пользовательский callable), сохранение всех стартов со статусами,
  greedy-кластеризация решений, checkpoint/resume через RunCheckpoint;
- dt_continuation — цепочка «coarse dt → fine dt»: перенос theta между
  ParameterSpace соседних задач через unpack/pack;
- subset_continuation — цепочка FrozenProblem «reduced subset → joint» с
  автоматическим добавлением финальной совместной доводки по всем
  записям модели.

ContinuationResult — локальный DTO этапов continuation: stage_results,
final_result, final_dt, wall_time, history; сериализация через
to_dict/from_dict/save/load (стадии кодируются их собственным to_dict,
если есть — иначе repr).
"""

import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
from scipy.special import ndtri
from scipy.stats import qmc

import neuraltide.config
from neuraltide.core.types import cast_if_needed
from neuraltide.optimization.parameters import ParameterSpace
from neuraltide.optimization.problem import FrozenProblem
from neuraltide.optimization.results import (
    LocalOptimizerResult,
    MultistartResult,
    OptimizationResult,
    RunCheckpoint,
    StartResult,
    _JsonDTO,
)

__all__ = [
    "sample_phi0s",
    "multistart",
    "dt_continuation",
    "subset_continuation",
    "ContinuationResult",
    "resolve_local_optimizer",
]

_SAMPLING_METHODS = ("sobol", "lhs", "random")
_LOCAL_METHODS = ("lbfgsb", "least_squares")

# Защита ndtri от вырожденных квантилей 0/1 (Sobol/LHS теоретически могут
# дать точную 0 в нулевой разрядной плоскости).
_UNIT_EPS = 1e-12


# ---------------------------------------------------------------------------
# Локальные оптимизаторы: строка -> адаптер (ленивый импорт)
# ---------------------------------------------------------------------------


def resolve_local_optimizer(
    optimizer,
) -> Tuple[Callable, str]:
    """
    Приводит строковое имя ('lbfgsb' | 'least_squares') или callable к
    (fn, имя). Адаптеры импортируются лениво из neuraltide.optimization.local;
    callable возвращается как есть с именем __name__/repr.
    """
    if callable(optimizer):
        return optimizer, str(getattr(optimizer, "__name__", repr(optimizer)))
    key = str(optimizer).lower()
    if key == "lbfgsb":
        from neuraltide.optimization.local import minimize_lbfgsb

        return minimize_lbfgsb, "lbfgsb"
    if key == "least_squares":
        from neuraltide.optimization.local import least_squares

        return least_squares, "least_squares"
    raise ValueError(
        f"strategies: неизвестный локальный оптимизатор {optimizer!r}; "
        f"доступны {list(_LOCAL_METHODS)} или callable(problem, phi0, **kwargs)."
    )


# ---------------------------------------------------------------------------
# Выборка начальных точек
# ---------------------------------------------------------------------------


def _unit_cube(n: int, size: int, method: str, seed: Optional[int]) -> np.ndarray:
    """n точек единичного гиперкуба [0,1)^(n,size) выбранным методом."""
    rng = np.random.default_rng(seed)
    if method == "sobol":
        sampler = qmc.Sobol(d=size, scramble=True, seed=rng)
        # scipy предупреждает о балансе при n не степени двойки — дизайн
        # при этом остаётся валидным и детерминированным, скрываем шум.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            points = sampler.random(n)
        return np.asarray(points, dtype=np.float64)
    if method == "lhs":
        sampler = qmc.LatinHypercube(d=size, scramble=True, seed=rng)
        return np.asarray(sampler.random(n), dtype=np.float64)
    return np.asarray(rng.random((n, size)), dtype=np.float64)


def _entry_offsets(space: ParameterSpace) -> Dict[str, Tuple[int, int]]:
    offsets: Dict[str, Tuple[int, int]] = {}
    start = 0
    for name, entry in space.entries.items():
        offsets[name] = (start, entry.numel)
        start += entry.numel
    return offsets


def _current_phi(space: ParameterSpace) -> np.ndarray:
    """
    Текущий phi ParameterSpace: переменные модели, где они есть, и нули
    для записей без переменной (тензорные/ручные записи).
    """
    values = {}
    for name, entry in space.entries.items():
        if entry.variable is not None:
            values[name] = entry.variable
        else:
            values[name] = np.zeros(entry.shape, dtype=np.float64)
    return np.asarray(space.pack(values), dtype=np.float64).reshape(-1)


def _inverse_numpy(entry, theta: np.ndarray) -> np.ndarray:
    """transform.inverse для массива theta -> phi (numpy, форма theta)."""
    dtype = neuraltide.config.get_dtype()
    theta_t = cast_if_needed(tf.convert_to_tensor(theta), dtype)
    phi_t = entry.transform.inverse(theta_t)
    return np.asarray(phi_t, dtype=np.float64).reshape(theta.shape)


def _unbounded_scale(center: np.ndarray) -> np.ndarray:
    """Характерный масштаб гауссова сэмплирования неограниченной записи."""
    return 0.5 * np.maximum(np.abs(center), 1.0)


def sample_phi0s(
    space: ParameterSpace,
    n: int,
    *,
    method: str = "sobol",
    seed: Optional[int] = None,
    log_uniform: Optional[Sequence[str]] = None,
    clip_to_bounds: bool = True,
) -> np.ndarray:
    """
    Детерминированная (по seed) выборка n начальных векторов phi размера
    space.size.

    Семантика по записям ParameterSpace:

    - ограниченные записи (min и max заданы): равномерный сэмплинг theta в
      физических границах, затем transform.inverse -> phi. Записи из
      log_uniform семплируются лог-равномерно внутри положительных границ
      (требуются конечные min > 0 и max);
    - неограниченные (в т.ч. частично ограниченные) записи: гауссов сэмплинг
      phi ~ N(current_phi, scale), scale = 0.5 * max(|current_phi|, 1) —
      с предупреждением UserWarning, перечисляющим такие записи.

    Args:
        space: ParameterSpace (порядок координат — порядок записей).
        n: число начальных точек (>= 1).
        method: 'sobol' | 'lhs' | 'random' — scipy.stats.qmc Sobol/LHS со
            scramble=True (низкорасхожденные дизайны) или numpy RNG.
        seed: seed воспроизводимости (None — недетерминированно).
        log_uniform: опциональный список full_names для лог-равномерного
            сэмплирования; неизвестные имена -> ValueError, неположительные
            или неограниченные границы -> ValueError.
        clip_to_bounds: применить space.project к каждой точке (физические
            theta гарантированно в [min, max]).

    Returns:
        np.ndarray формы [n, size] (float64).

    Raises:
        ValueError: неизвестный method, n < 1, ошибки в log_uniform.
    """
    if int(n) < 1:
        raise ValueError(f"sample_phi0s: n должен быть >= 1, получен {n}.")
    n = int(n)
    method = str(method).lower()
    if method not in _SAMPLING_METHODS:
        raise ValueError(
            f"sample_phi0s: неизвестный method={method!r}; "
            f"доступны {list(_SAMPLING_METHODS)}."
        )
    log_set = {str(name) for name in (log_uniform or [])}
    unknown = sorted(log_set - set(space.entries))
    if unknown:
        raise ValueError(
            f"sample_phi0s: log_uniform содержит неизвестные имена {unknown}."
        )

    size = int(space.size)
    if size == 0:
        raise ValueError("sample_phi0s: пустой ParameterSpace (size == 0).")

    offsets = _entry_offsets(space)
    phi_current = _current_phi(space)
    unit = _unit_cube(n, size, method, seed)

    phi = np.empty((n, size), dtype=np.float64)
    unbounded_names: List[str] = []
    for name, entry in space.entries.items():
        start, numel = offsets[name]
        sl = slice(start, start + numel)
        u = np.clip(unit[:, sl], _UNIT_EPS, 1.0 - _UNIT_EPS)
        lo, hi = entry.min, entry.max
        if name in log_set:
            if lo is None or hi is None or not np.isfinite(lo) or not np.isfinite(hi):
                raise ValueError(
                    f"sample_phi0s: log_uniform для '{name}' требует конечных "
                    "границ min/max."
                )
            if lo <= 0.0:
                raise ValueError(
                    f"sample_phi0s: log_uniform для '{name}' требует min > 0, "
                    f"получен min={lo}."
                )
            theta = np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo)))
            phi[:, sl] = _inverse_numpy(entry, theta)
        elif lo is not None and hi is not None:
            theta = lo + u * (hi - lo)
            phi[:, sl] = _inverse_numpy(entry, theta)
        else:
            unbounded_names.append(name)
            center = phi_current[sl]
            scale = _unbounded_scale(center)
            phi[:, sl] = center + ndtri(u) * scale

    if unbounded_names:
        warnings.warn(
            "sample_phi0s: записи без конечных min/max семплируются "
            f"гауссово вокруг текущего phi: {sorted(unbounded_names)}.",
            UserWarning,
            stacklevel=2,
        )

    if clip_to_bounds:
        projected = [np.asarray(space.project(row), dtype=np.float64) for row in phi]
        phi = np.stack(projected, axis=0)
    return phi


# ---------------------------------------------------------------------------
# Дизайн стартов для duck-typed проблем (без ParameterSpace)
# ---------------------------------------------------------------------------


def _design_for_problem(
    problem, n: int, method: str, seed: Optional[int]
) -> np.ndarray:
    """
    Начальные phi для проблемы: полный sample_phi0s для ParameterSpace,
    иначе fallback по bounds_phi()/pack() для duck-typed проблем.
    """
    space = getattr(problem, "parameter_space", None)
    if isinstance(space, ParameterSpace) and space.size > 0:
        return sample_phi0s(space, n, method=method, seed=seed)

    size = int(problem.size)
    lb, ub = np.full(size, -np.inf), np.full(size, np.inf)
    bounds_fn = getattr(problem, "bounds_phi", None)
    if callable(bounds_fn):
        lb, ub = bounds_fn()
        lb = np.asarray(lb, dtype=np.float64).reshape(-1)
        ub = np.asarray(ub, dtype=np.float64).reshape(-1)
    unit = np.clip(_unit_cube(n, size, method, seed), _UNIT_EPS, 1.0 - _UNIT_EPS)
    try:
        center = np.asarray(problem.pack(), dtype=np.float64).reshape(-1)
    except Exception:
        center = np.zeros(size, dtype=np.float64)
    bounded = np.isfinite(lb) & np.isfinite(ub)
    phi = np.empty((n, size), dtype=np.float64)
    if np.any(bounded):
        phi[:, bounded] = lb[bounded] + unit[:, bounded] * (ub - lb)[bounded]
    if np.any(~bounded):
        scale = _unbounded_scale(center[~bounded])
        phi[:, ~bounded] = center[~bounded] + ndtri(unit[:, ~bounded]) * scale
    if np.any(np.isfinite(lb)) or np.any(np.isfinite(ub)):
        clip_lo = np.where(np.isfinite(lb), lb, -np.inf)
        clip_hi = np.where(np.isfinite(ub), ub, np.inf)
        np.clip(phi, clip_lo, clip_hi, out=phi)
    return phi


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def _multistart_config(
    n_starts: int,
    method: str,
    screen_evals: int,
    refine_top_k: int,
    local_name: str,
    dedup_rtol: float,
) -> Dict[str, Any]:
    return {
        "n_starts": int(n_starts),
        "method": str(method),
        "screen_evals": int(screen_evals),
        "refine_top_k": int(refine_top_k),
        "local": str(local_name),
        "dedup_rtol": float(dedup_rtol),
    }


def _write_checkpoint(path, seed, config, completed: Mapping[int, StartResult]) -> None:
    indices = sorted(completed)
    best_ok = [
        i
        for i in indices
        if completed[i].status == "ok" and np.isfinite(completed[i].loss)
    ]
    best_index = min(best_ok, key=lambda i: (completed[i].loss, i)) if best_ok else -1
    pending = [i for i in sorted(config["n_starts_list"]) if i not in completed]
    checkpoint = RunCheckpoint(
        seed=seed,
        config=config,
        completed=[completed[i].to_dict() for i in indices],
        pending=pending,
        best_index=best_index,
    )
    checkpoint.save(os.fspath(path))


def _load_completed(
    path,
) -> Tuple[Optional[int], Dict[str, Any], Dict[int, StartResult]]:
    checkpoint = RunCheckpoint.load(os.fspath(path))
    completed: Dict[int, StartResult] = {}
    for item in checkpoint.completed:
        if not isinstance(item, Mapping):
            raise ValueError(
                "multistart: checkpoint.completed должен содержать словари "
                "StartResult (получен не-словарный элемент)."
            )
        start = StartResult.from_dict(item)
        completed[start.index] = start
    return checkpoint.seed, checkpoint.config, completed


# ---------------------------------------------------------------------------
# Кластеризация решений
# ---------------------------------------------------------------------------


def _phi_close(a: np.ndarray, b: np.ndarray, rtol: float) -> bool:
    """Близость двух phi: |a-b| <= rtol * max(|b|, 1) покомпонентно."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return bool(np.all(np.abs(a - b) <= rtol * np.maximum(np.abs(b), 1.0)))


def _cluster_solutions(solutions: Sequence[StartResult], rtol: float) -> Dict[str, Any]:
    """
    Greedy-кластеризация успешных (status 'ok') стартов по возрастанию
    loss: старт присоединяется к кластеру, если его phi близок к phi
    представителя кластера (rtol-норма), иначе открывает новый кластер.
    """
    labels = [-1] * len(solutions)
    representatives: List[int] = []
    order = sorted(
        (
            i
            for i, s in enumerate(solutions)
            if s.status == "ok" and np.isfinite(s.loss)
        ),
        key=lambda i: (solutions[i].loss, i),
    )
    for i in order:
        for cluster_id, rep in enumerate(representatives):
            if _phi_close(solutions[i].phi, solutions[rep].phi, rtol):
                labels[i] = cluster_id
                break
        else:
            labels[i] = len(representatives)
            representatives.append(i)
    return {
        "n_clusters": len(representatives),
        "labels": labels,
        "representatives": representatives,
    }


# ---------------------------------------------------------------------------
# multistart
# ---------------------------------------------------------------------------


def _screen(problem, phi0: np.ndarray, screen_evals: int) -> float:
    """Скрининг старта: screen_evals оценок, среднее по ним."""
    losses = [float(problem.evaluate(phi0).loss) for _ in range(int(screen_evals))]
    return float(np.mean(losses))


def _run_multistart(
    problem,
    *,
    n_starts: int,
    method: str,
    screen_evals: int,
    refine_top_k: int,
    local,
    seed: Optional[int],
    checkpoint_path,
    resume: bool,
    dedup_rtol: float,
) -> MultistartResult:
    t_begin = time.perf_counter()
    local_fn, local_name = resolve_local_optimizer(local)
    config = _multistart_config(
        n_starts, method, screen_evals, refine_top_k, local_name, dedup_rtol
    )
    config["n_starts_list"] = list(range(n_starts))

    completed: Dict[int, StartResult] = {}
    if checkpoint_path is not None and resume and os.path.exists(checkpoint_path):
        ckpt_seed, ckpt_config, completed = _load_completed(checkpoint_path)
        if ckpt_seed != seed:
            raise ValueError(
                f"multistart: seed чекпоинта ({ckpt_seed}) не совпадает с "
                f"запрошенным ({seed}); resume невозможен."
            )
        stored = {k: v for k, v in ckpt_config.items() if k != "n_starts_list"}
        requested = {k: v for k, v in config.items() if k != "n_starts_list"}
        if stored != requested:
            raise ValueError(
                "multistart: конфигурация чекпоинта не совпадает с "
                f"запрошенной: checkpoint={stored}, requested={requested}."
            )

    phi0s = _design_for_problem(problem, n_starts, method, seed)

    def _save() -> None:
        if checkpoint_path is not None:
            _write_checkpoint(checkpoint_path, seed, config, completed)

    # --- Фаза 1: screening всех недоделанных стартов ---------------------
    for i in range(n_starts):
        if i in completed:
            continue
        t0 = time.perf_counter()
        phi0 = np.asarray(phi0s[i], dtype=np.float64).reshape(-1)
        start = StartResult(
            index=i,
            x0_origin=str(method),
            phi0=phi0.copy(),
            phi=phi0.copy(),
        )
        try:
            screened = _screen(problem, phi0, screen_evals)
            if not np.isfinite(screened):
                raise FloatingPointError("нефинитная потеря при screening")
            start.screened_loss = screened
            start.loss = screened
            start.status = "screen_rejected"
            start.message = "отклонён на этапе screening"
        except Exception as exc:  # noqa: BLE001 - старт не должен ронять прогон
            start.status = "failed"
            start.message = f"screening: {exc}"
            start.loss = float("inf")
        start.wall_time = time.perf_counter() - t0
        completed[i] = start
        _save()

    # --- Фаза 2: доводка top-k по screened loss (детерминированный ------
    # --- порядок: сортировка по (screened_loss, index)) ------------------
    screened_ok = sorted(
        (
            i
            for i, s in completed.items()
            if s.screened_loss is not None and np.isfinite(s.screened_loss)
        ),
        key=lambda i: (completed[i].screened_loss, i),
    )
    k = max(0, min(int(refine_top_k), n_starts))
    for i in screened_ok[:k]:
        start = completed[i]
        if start.refined or start.status == "failed":
            continue
        t0 = time.perf_counter()
        phi0 = np.asarray(start.phi0, dtype=np.float64).reshape(-1)
        try:
            # Явный phi0: адаптер сам переводит модель в theta=unpack(phi0)
            # перед первым прогоном (контракт CalibrationProblem).
            result = local_fn(problem, phi0)
            loss = float(result.base.loss_best)
            if not np.isfinite(loss):
                raise FloatingPointError("нефинитная потеря после доводки")
            start.phi = np.asarray(result.base.phi_best, dtype=np.float64).reshape(-1)
            start.loss = loss
            start.status = "ok"
            start.message = str(getattr(result, "message", ""))
            start.refined = True
            start.local = result
        except Exception as exc:  # noqa: BLE001 - старт не должен ронять прогон
            start.status = "failed"
            start.message = f"refinement: {exc}"
            start.refined = False
        start.wall_time += time.perf_counter() - t0
        _save()

    # --- Сборка результата ----------------------------------------------
    solutions = [completed[i] for i in sorted(completed)]
    clustering = _cluster_solutions(solutions, float(dedup_rtol))
    ok_indices = [
        i for i, s in enumerate(solutions) if s.status == "ok" and np.isfinite(s.loss)
    ]
    if ok_indices:
        best_index = min(ok_indices, key=lambda i: (solutions[i].loss, i))
        best = solutions[best_index]
        phi_best, loss_best = best.phi, best.loss
    else:
        best_index = -1
        finite = [s for s in solutions if np.isfinite(s.loss)]
        fallback = min(finite, key=lambda s: s.loss) if finite else solutions[0]
        phi_best, loss_best = fallback.phi, fallback.loss

    unpack = getattr(problem, "unpack", None)
    if callable(unpack):
        theta = unpack(phi_best)
    else:
        theta = problem.parameter_space.unpack(phi_best)
    theta_best = {str(name): np.asarray(value) for name, value in theta.items()}

    history = [
        {
            "index": s.index,
            "status": s.status,
            "screened_loss": s.screened_loss,
            "loss": s.loss,
            "refined": s.refined,
            "wall_time": s.wall_time,
        }
        for s in solutions
    ]
    base = OptimizationResult(
        phi_best=np.asarray(phi_best, dtype=np.float64),
        theta_best=theta_best,
        loss_best=float(loss_best),
        history=history,
        seed=seed,
        config={k: v for k, v in config.items() if k != "n_starts_list"},
        dtype=str(neuraltide.config.get_dtype().name),
    )
    return MultistartResult(
        base=base,
        solutions=solutions,
        best_index=best_index,
        n_starts=len(solutions),
        n_refined=sum(1 for s in solutions if s.refined),
        n_failed=sum(1 for s in solutions if s.status == "failed"),
        seed=seed,
        clustering=clustering,
        wall_time=time.perf_counter() - t_begin,
    )


def multistart(
    problem,
    *,
    n_starts: int = 20,
    method: str = "sobol",
    screen_evals: int = 1,
    refine_top_k: int = 5,
    local="lbfgsb",
    seed: Optional[int] = None,
    checkpoint_path=None,
    resume: bool = False,
    dedup_rtol: float = 1e-4,
) -> MultistartResult:
    """
    Мультистарт «глобальный дизайн -> screening -> локальная доводка top-k».

    1. Генерируется детерминированный по (method, seed) дизайн из
       n_starts начальных phi (sample_phi0s для ParameterSpace проблемы,
       fallback по bounds_phi для duck-typed).
    2. Screening: ровно screen_evals вызовов problem.evaluate на старт;
       старты с нефинитной потерей/исключением получают status 'failed'.
    3. Доводка: refine_top_k лучших по screened_loss стартов прогоняются
       через локальный оптимизатор ('lbfgsb' | 'least_squares' — ленивый
       импорт адаптеров local.py, либо callable(problem, phi0)).
       Исключение адаптера для старта -> status 'failed' с message, прогон
       продолжается. Перед каждым стартом модель явно переводится в
       theta = unpack(phi0) (контракт CalibrationProblem), скрытое
       состояние между стартами не используется.
    4. В solutions сохраняются ВСЕ старты со статусами 'ok' /
       'screen_rejected' / 'failed'; удачные старты кластеризуются greedy
       по (loss, близость phi с rtol=dedup_rtol).

    Checkpoint/resume: при checkpoint_path после каждого старта пишется
    RunCheckpoint (seed/config/completed/pending/best_index; без состояния
    RNG). Повторный запуск с resume=True и теми же (seed, n_starts,
    конфигурацией) досчитывает только недоделанные старты и даёт тот же
    результат, что и непрерывный прогон.

    Args:
        problem: CalibrationProblem (или duck-typed с evaluate/pack/
            bounds_phi и, для доводки, value_and_gradient/residual).
        n_starts: число стартов (>= 1).
        method: 'sobol' | 'lhs' | 'random' — дизайн начальных точек.
        screen_evals: число evaluate на старт при screening (>= 1,
            среднее по прогонам).
        refine_top_k: сколько лучших стартов доводить локально.
        local: 'lbfgsb' | 'least_squares' | callable(problem, phi0).
        seed: seed дизайна и чекпоинта.
        checkpoint_path: путь к JSON-чекпоинту (None — без чекпоинта).
        resume: дочитать существующий checkpoint_path.
        dedup_rtol: rtol близости phi при кластеризации.

    Returns:
        MultistartResult (base = OptimizationResult лучшего старта).
    """
    n_starts = int(n_starts)
    screen_evals = int(screen_evals)
    refine_top_k = int(refine_top_k)
    if n_starts < 1:
        raise ValueError(f"multistart: n_starts должен быть >= 1, {n_starts}.")
    if screen_evals < 1:
        raise ValueError(f"multistart: screen_evals должен быть >= 1, {screen_evals}.")
    if refine_top_k < 0:
        raise ValueError(f"multistart: refine_top_k должен быть >= 0, {refine_top_k}.")
    if float(dedup_rtol) <= 0.0:
        raise ValueError(f"multistart: dedup_rtol должен быть > 0, {dedup_rtol}.")
    return _run_multistart(
        problem,
        n_starts=n_starts,
        method=str(method).lower(),
        screen_evals=screen_evals,
        refine_top_k=refine_top_k,
        local=local,
        seed=seed,
        checkpoint_path=checkpoint_path,
        resume=bool(resume),
        dedup_rtol=float(dedup_rtol),
    )


# ---------------------------------------------------------------------------
# Continuation
# ---------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    """Рекурсивное приведение к JSON-сериализуемому виду."""
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


def _stage_to_dict(result: Any) -> Dict[str, Any]:
    to_dict = getattr(result, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    return {"repr": repr(result)}


def _stage_from_dict(data: Any) -> Any:
    if isinstance(data, Mapping) and "base" in data:
        return LocalOptimizerResult.from_dict(data)
    return data


@dataclass
class ContinuationResult(_JsonDTO):
    """
    Итог continuation-цепочки (dt_continuation / subset_continuation,
    единый формат с ContinuationScheduler в schedules.py).

    Атрибуты:
        stage_results: результаты локальных оптимизаторов по стадиям
            (LocalOptimizerResult или пользовательский объект);
        final_result: результат последней стадии;
        final_dt: шаг интегрирования последней задачи (0.0, если шаг не
            определён задачей);
        wall_time: суммарное время, секунды;
        history: краткая история по стадиям (dt/loss/success/...).
    """

    stage_results: List[Any] = field(default_factory=list)
    final_result: Any = None
    final_dt: float = 0.0
    wall_time: float = 0.0
    history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_results": [_stage_to_dict(r) for r in self.stage_results],
            "final_result": _stage_to_dict(self.final_result),
            "final_dt": float(self.final_dt),
            "wall_time": float(self.wall_time),
            "history": _jsonable(self.history),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ContinuationResult":
        return cls(
            stage_results=[_stage_from_dict(d) for d in data.get("stage_results", [])],
            final_result=_stage_from_dict(data.get("final_result")),
            final_dt=float(data.get("final_dt", 0.0)),
            wall_time=float(data.get("wall_time", 0.0)),
            history=list(data.get("history", [])),
        )


def _resolve_optimizer_call(optimizer, problem, phi: np.ndarray, kwargs) -> Any:
    """Запуск строкового адаптера или callable для одной стадии."""
    if callable(optimizer):
        return optimizer(problem, phi, **kwargs)
    fn, _ = resolve_local_optimizer(optimizer)
    return fn(problem, phi, **kwargs)


def _history_entry(stage: int, dt: float, result: Any) -> Dict[str, Any]:
    base = getattr(result, "base", None)
    return {
        "stage": int(stage),
        "dt": float(dt),
        "loss": None if base is None else float(base.loss_best),
        "success": bool(getattr(result, "success", False)),
        "nit": int(getattr(result, "nit", 0) or 0),
        "wall_time": float(getattr(result, "wall_time", 0.0)),
    }


def dt_continuation(
    build_problem,
    dts: Sequence[float],
    *,
    optimizer="lbfgsb",
    **kwargs,
) -> ContinuationResult:
    """
    Continuation «coarse dt -> fine dt».

    Args:
        build_problem: callable(dt) -> CalibrationProblem; задачи на
            разных dt — разные объекты со своими ParameterSpace.
        dts: непустая последовательность шагов интегрирования, мс; старт
            первой стадии — pack() свежесозданной задачи
            build_problem(dts[0]) (значения параметров её конструктора, а
            не «текущее состояние модели» какой-либо ранее существовавшей
            задачи — build_problem строит новые объекты сети); далее theta
            лучшей точки переносится между соседними ParameterSpace через
            unpack -> pack.
        optimizer: 'lbfgsb' | 'least_squares' | callable(problem, phi0,
            **kwargs).
        **kwargs: параметры локального оптимизатора (maxiter и т.п.).

    Returns:
        ContinuationResult: stage_results по стадиям, final_result —
        результат последней (fine dt) стадии.
    """
    dts = [float(dt) for dt in dts]
    if not dts:
        raise ValueError("dt_continuation: непустая последовательность dts.")
    t_begin = time.perf_counter()
    problems = [build_problem(dt) for dt in dts]
    stage_results: List[Any] = []
    history: List[Dict[str, Any]] = []
    phi = np.asarray(problems[0].pack(), dtype=np.float64).reshape(-1)
    for stage, (dt, problem) in enumerate(zip(dts, problems)):
        result = _resolve_optimizer_call(optimizer, problem, phi, kwargs)
        stage_results.append(result)
        history.append(_history_entry(stage, dt, result))
        phi = np.asarray(result.base.phi_best, dtype=np.float64).reshape(-1)
        if stage + 1 < len(problems):
            nxt = problems[stage + 1]
            unpack = getattr(problem, "unpack", None)
            if callable(unpack):
                theta = unpack(phi)
            else:
                theta = problem.parameter_space.unpack(phi)
            try:
                phi = np.asarray(
                    nxt.parameter_space.pack(theta), dtype=np.float64
                ).reshape(-1)
            except ValueError as exc:
                raise ValueError(
                    f"dt_continuation: перенос theta между ParameterSpace "
                    f"стадий {stage}->{stage + 1} невозможен: {exc}"
                ) from exc
    return ContinuationResult(
        stage_results=stage_results,
        final_result=stage_results[-1],
        final_dt=dts[-1],
        wall_time=time.perf_counter() - t_begin,
        history=history,
    )


def subset_continuation(
    problem,
    stages_free_names: Sequence[Sequence[str]],
    *,
    local="lbfgsb",
    **kwargs,
) -> ContinuationResult:
    """
    Continuation «reduced parameter subset -> joint refinement».

    Каждая стадия — FrozenProblem над той же базовой задачей со своим
    набором свободных записей. Если последний переданный набор не
    покрывает все записи ParameterSpace, автоматически добавляется
    финальная совместная доводка по всем записям. Точка переносится между
    стадиями явно: phi_best стадии применяется к модели через
    frozen.evaluate(phi_best) (полный вектор, замороженные координаты —
    на снимке конструирования FrozenProblem), далее pack() следующего
    FrozenProblem берёт свободные координаты текущего состояния модели.

    Args:
        problem: базовая CalibrationProblem (CalibrationProblem/
            MultiExperimentProblem).
        stages_free_names: список списков full_names, свободных на каждой
            стадии; непустой, имена проверяются на известность.
        local: 'lbfgsb' | 'least_squares' | callable(frozen, phi0, **kwargs).
        **kwargs: параметры локального оптимизатора.

    Returns:
        ContinuationResult: stage_results (включая автоматическую
        финальную стадию), final_result — совместная доводка.
    """
    stages = [list(map(str, names)) for names in stages_free_names]
    if not stages:
        raise ValueError("subset_continuation: непустой stages_free_names обязателен.")
    space = problem.parameter_space
    known = set(space.entries)
    for stage_id, names in enumerate(stages):
        unknown = sorted(set(names) - known)
        if unknown:
            raise ValueError(
                f"subset_continuation: неизвестные имена на стадии "
                f"{stage_id}: {unknown}."
            )
    all_names = list(space.entries.keys())
    if set(stages[-1]) != set(all_names):
        stages = stages + [all_names]

    dt = 0.0
    network = getattr(problem, "network", None)
    graph = getattr(network, "_graph", None)
    if graph is not None:
        dt = float(getattr(graph, "dt", 0.0))

    t_begin = time.perf_counter()
    stage_results: List[Any] = []
    history: List[Dict[str, Any]] = []
    for stage_id, names in enumerate(stages):
        # FrozenProblem строится ПОСЛЕ применения предыдущей стадии:
        # замороженные координаты снимаются с текущего состояния модели,
        # свободные координаты старта читаются через pack() (None).
        frozen = FrozenProblem(problem, free_names=names)
        phi0 = np.asarray(frozen.pack(), dtype=np.float64).reshape(-1)
        if phi0.shape[0] != frozen.size:
            raise ValueError(
                f"subset_continuation: размер phi0 ({phi0.shape[0]}) не "
                f"совпадает со стадией {stage_id} (size={frozen.size})."
            )
        result = _resolve_optimizer_call(local, frozen, phi0, kwargs)
        stage_results.append(result)
        history.append(_history_entry(stage_id, dt, result))
        # Явный перенос лучшей точки в модель перед следующей стадией.
        frozen.evaluate(np.asarray(result.base.phi_best, dtype=np.float64))
    return ContinuationResult(
        stage_results=stage_results,
        final_result=stage_results[-1],
        final_dt=dt,
        wall_time=time.perf_counter() - t_begin,
        history=history,
    )
