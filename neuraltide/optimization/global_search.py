"""
Глобальный безградиентный поиск (CMA-ES/DE/SPSA/basinhopping/NSGA-II,
база для параллелизма по кандидатам).

Единая точка входа global_search поверх CalibrationProblem (или
duck-typed проблемы с evaluate/pack/bounds_phi/unpack):

- de — Differential Evolution через scipy.optimize.differential_evolution.
  Требуются конечные границы (иначе ValueError с советом использовать
  transforms/sample_phi0s). Параллелизм SciPy (workers=) не используется;
  внешний параллелизм — через evaluator.
- cma — CMA-ES через опциональный пакет cma (extra 'global',
  ``pip install 'neuraltide[global]'``). Ask/tell-цикл, checkpoint/resume.
- spsa — компактная numpy-реализация SPSA без внешних зависимостей.
  Ask/tell-цикл, checkpoint/resume.
- basinhopping — scipy.optimize.basinhopping; локальный минимизатор
  L-BFGS-B, случайное смещение шага зажато в границы.
- nsga2 — NSGA-II через опциональный pymoo (extra 'pareto',
  ``pip install 'neuraltide[pareto]'``). Целевой вектор берётся из
  Evaluation.components; результат — ParetoResult (фронт).

Единый бюджет max_evals ограничивает число вычислений problem.evaluate
(при внешнем evaluator — число переданных кандидатов). NaN/исключение
кандидата дают loss=+inf и увеличивают счётчик отказов n_failed (записи
history имеют ``"failed": True``). Детерминизм при фиксированных
(seed, phi0, max_evals): вся случайность проходит через _Oracle.

refine_top_k лучших кандидатов глобальной фазы доводятся локальным
оптимизатором (ленивый импорт strategies.resolve_local_optimizer и
адаптеров local); записываются как StartResult в candidates. Доводка
имеет собственный бюджет локального метода (как в multistart) и в
max_evals не входит.

Checkpoint/resume (cma/spsa): JSON в стиле RunCheckpoint
(seed/config/completed/pending/best_index). completed хранит поколения/
итерации с потерями; pending для открытых ask/tell-прогонов пуст —
достаточно completed и бюджета. resume восстанавливает дизайн из seed
(документированная аппроксимация: требуются те же версии пакетов и те же
опции) и досчитывает бюджет ``max_evals - sum(completed.n_eval)``,
поэтому результат resume совпадает с непрерывным прогоном. Для scipy DE
и basinhopping resume не поддерживается: популяция/цепочка состояний
живут внутри решателя и из seed не восстанавливаются.

Результаты: GlobalOptimizerResult (base OptimizationResult + method/
success/message/n_evals/n_forward/n_gradient/wall_time/history/
candidates/validation) и ParetoResult (base-like OptimizationResult
опорной точки фронта, frontier dict, objective_names, wall_time,
history) — JSON-сериализация через _JsonDTO (to_dict/from_dict/save/load).
"""

import math
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import optimize as sp_optimize

import neuraltide.config
from neuraltide.optimization.results import (
    OptimizationResult,
    RunCheckpoint,
    StartResult,
    _JsonDTO,
)

__all__ = [
    "OptionalDependencyError",
    "GlobalOptimizerResult",
    "ParetoResult",
    "global_search",
]

_GLOBAL_METHODS = ("de", "cma", "spsa", "basinhopping", "nsga2")
_ASK_TELL_METHODS = ("cma", "spsa")
_INSTALL_GLOBAL = "pip install 'neuraltide[global]'"
_INSTALL_PARETO = "pip install 'neuraltide[pareto]'"
# Верхняя крышка конечной потери, подаваемой в SciPy-решатели: +inf в
# history/oracle сохраняется, а решатели получают конечное значение
# (L-BFGS-B не переносит inf в стартовой точке).
_LOSS_CAP = 1e20

_OPTION_KEYS = {
    "de": {"popsize", "tol", "polish", "mutation", "recombination", "init", "maxiter"},
    "cma": {"sigma0", "popsize", "maxiter", "cma_options"},
    "spsa": {"a", "c", "A", "alpha", "gamma", "max_iter", "a_mult", "c_mult"},
    "basinhopping": {"niter", "stepsize", "T", "inner_maxiter"},
    "nsga2": {"pop_size", "n_gen"},
}


class OptionalDependencyError(ImportError):
    """Отсутствует опциональная зависимость (extra 'global'/'pareto')."""


class _BudgetExhausted(Exception):
    """Внутренний сигнал исчерпания бюджета max_evals."""


# ---------------------------------------------------------------------------
# DTO результата
# ---------------------------------------------------------------------------


@dataclass
class GlobalOptimizerResult(_JsonDTO):
    """
    Результат глобального оптимизатора поверх CalibrationProblem.

    Атрибуты:
        base: базовый OptimizationResult (phi_best/theta_best/loss_best/
            история оценок/seed/config);
        method: имя метода ('de' | 'cma' | 'spsa' | 'basinhopping');
        success: признак успешного завершения (конечная лучшая потеря);
        message: сообщение решателя / причина остановки;
        n_evals: число оценённых кандидатов глобальной фазы (<= max_evals);
        n_forward / n_gradient: дельта счётчиков problem за весь запуск
            (включая финальную явную оценку и доводку);
        wall_time: общее время, секунды;
        history: записи оценок {"n_evals", "loss", "failed"};
        candidates: результаты доводки refine_top_k (StartResult);
        validation: метрики validate() на held-out протоколах.
    """

    base: OptimizationResult
    method: str = "de"
    success: bool = False
    message: str = ""
    n_evals: int = 0
    n_forward: int = 0
    n_gradient: int = 0
    wall_time: float = 0.0
    history: List[Dict[str, Any]] = field(default_factory=list)
    candidates: List[StartResult] = field(default_factory=list)
    validation: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base": self.base.to_dict(),
            "method": str(self.method),
            "success": bool(self.success),
            "message": str(self.message),
            "n_evals": int(self.n_evals),
            "n_forward": int(self.n_forward),
            "n_gradient": int(self.n_gradient),
            "wall_time": float(self.wall_time),
            "history": _jsonable(self.history),
            "candidates": [c.to_dict() for c in self.candidates],
            "validation": {str(k): float(v) for k, v in self.validation.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GlobalOptimizerResult":
        return cls(
            base=OptimizationResult.from_dict(data["base"]),
            method=str(data.get("method", "de")),
            success=bool(data.get("success", False)),
            message=str(data.get("message", "")),
            n_evals=int(data.get("n_evals", 0)),
            n_forward=int(data.get("n_forward", 0)),
            n_gradient=int(data.get("n_gradient", 0)),
            wall_time=float(data.get("wall_time", 0.0)),
            history=list(data.get("history", [])),
            candidates=[StartResult.from_dict(d) for d in data.get("candidates", [])],
            validation={
                str(k): float(v) for k, v in data.get("validation", {}).items()
            },
        )


@dataclass
class ParetoResult(_JsonDTO):
    """
    Результат NSGA-II: Pareto-фронт по компонентам Evaluation.components.

    Атрибуты:
        base: base-like OptimizationResult опорной (knee) точки фронта —
            phi_best/theta_best/loss_best (loss_best — сумма целей точки);
        frontier: dict {objective_name: ndarray значений на фронте};
        objective_names: имена целей (порядок столбцов frontier);
        method: имя метода ('nsga2');
        success: признак успешного завершения (непустой фронт);
        message: сообщение решателя / причина остановки;
        n_evals: число реально оценённых кандидатов (<= max_evals);
        n_forward: дельта счётчиков problem за запуск;
        wall_time: общее время, секунды;
        history: записи оценок {"n_evals", "loss", "failed"};
        seed: seed запуска (или None);
        config: конфигурация запуска (dict).
    """

    base: OptimizationResult
    frontier: Dict[str, np.ndarray] = field(default_factory=dict)
    objective_names: List[str] = field(default_factory=list)
    method: str = "nsga2"
    success: bool = False
    message: str = ""
    n_evals: int = 0
    n_forward: int = 0
    wall_time: float = 0.0
    history: List[Dict[str, Any]] = field(default_factory=list)
    seed: Optional[int] = None
    config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.frontier = {
            str(k): np.asarray(v, dtype=np.float64) for k, v in self.frontier.items()
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base": self.base.to_dict(),
            "frontier": {k: v.tolist() for k, v in self.frontier.items()},
            "objective_names": [str(n) for n in self.objective_names],
            "method": str(self.method),
            "success": bool(self.success),
            "message": str(self.message),
            "n_evals": int(self.n_evals),
            "n_forward": int(self.n_forward),
            "wall_time": float(self.wall_time),
            "history": _jsonable(self.history),
            "seed": self.seed,
            "config": _jsonable(self.config),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ParetoResult":
        return cls(
            base=OptimizationResult.from_dict(data["base"]),
            frontier={
                str(k): np.asarray(v, dtype=np.float64)
                for k, v in data.get("frontier", {}).items()
            },
            objective_names=[str(n) for n in data.get("objective_names", [])],
            method=str(data.get("method", "nsga2")),
            success=bool(data.get("success", False)),
            message=str(data.get("message", "")),
            n_evals=int(data.get("n_evals", 0)),
            n_forward=int(data.get("n_forward", 0)),
            wall_time=float(data.get("wall_time", 0.0)),
            history=list(data.get("history", [])),
            seed=data.get("seed"),
            config=dict(data.get("config", {})),
        )


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    """Рекурсивное приведение config/историй к JSON-сериализуемому виду."""
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


def _cap_loss(loss: float) -> float:
    """+inf -> _LOSS_CAP для подачи в SciPy-решатели."""
    loss = float(loss)
    if not np.isfinite(loss):
        return _LOSS_CAP
    return min(loss, _LOSS_CAP)


def _check_options(method: str, options: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    opts = dict(options or {})
    allowed = _OPTION_KEYS[method]
    unknown = sorted(set(opts) - allowed)
    if unknown:
        raise ValueError(
            f"global_search({method}): неизвестные options {unknown}; "
            f"допустимы {sorted(allowed)}."
        )
    return opts


def _resolve_phi0(problem, phi0, size: int) -> np.ndarray:
    if phi0 is None:
        phi0 = problem.pack()
    arr = np.asarray(phi0, dtype=np.float64).reshape(-1)
    if arr.shape[0] != size:
        raise ValueError(
            f"global_search: phi0 имеет {arr.shape[0]} элементов, ожидается "
            f"{size} (problem.size)."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("global_search: phi0 содержит нефинитные значения.")
    return arr


def _resolve_bounds(
    problem, size: int, bounds
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Границы: явный аргумент > problem.search_box_phi() > problem.bounds_phi()
    > None.

    search_box_phi — бокс по θ без 1 % у краёв физического диапазона; в
    bounds_phi у sigmoid-образных преобразований стоят границы машинной
    точности, и равномерная выборка DE/CMA ложилась бы к краям диапазона.
    """
    if bounds is None:
        fn = getattr(problem, "search_box_phi", None)
        if not callable(fn):
            fn = getattr(problem, "bounds_phi", None)
        if not callable(fn):
            return None
        lb, ub = fn()
    else:
        lb, ub = bounds
    lb = np.asarray(lb, dtype=np.float64).reshape(-1)
    ub = np.asarray(ub, dtype=np.float64).reshape(-1)
    if lb.shape[0] != size or ub.shape[0] != size:
        raise ValueError(
            f"global_search: bounds размера ({lb.shape[0]}, {ub.shape[0]}), "
            f"ожидается ({size}, {size})."
        )
    if np.any(lb > ub):
        raise ValueError("global_search: нижняя граница больше верхней.")
    return lb, ub


def _require_finite_bounds(bounds, method: str) -> Tuple[np.ndarray, np.ndarray]:
    """Конечные границы или ValueError с советом (transforms/sample_phi0s)."""
    if bounds is None or not (
        np.all(np.isfinite(bounds[0])) and np.all(np.isfinite(bounds[1]))
    ):
        raise ValueError(
            f"global_search({method}): метод требует конечных границ "
            "bounds_phi() по всем координатам. Задайте min/max в params "
            "(перевод θ→φ выполняют transforms, например "
            "ScaledSigmoidTransform), либо сэмплируйте стартовые точки "
            "sample_phi0s(...) и передайте bounds/phi0 явно."
        )
    return bounds


def _clip_to_bounds(theta: np.ndarray, bounds) -> np.ndarray:
    if bounds is None:
        return theta
    return np.clip(theta, bounds[0], bounds[1])


def _theta_best(problem, phi_best: np.ndarray) -> Dict[str, np.ndarray]:
    unpack = getattr(problem, "unpack", None)
    if callable(unpack):
        theta = unpack(phi_best)
    else:
        theta = problem.parameter_space.unpack(phi_best)
    return {str(k): np.asarray(v) for k, v in theta.items()}


def _non_dominated_mask(F: np.ndarray) -> np.ndarray:
    """Булева маска недоминируемых строк матрицы целей [n, m] (минимум)."""
    n = F.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        better = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        if np.any(better):
            keep[i] = False
    return keep


# ---------------------------------------------------------------------------
# Oracle: единый бюджет, отказы, история
# ---------------------------------------------------------------------------


class _Oracle:
    """
    Обертка problem.evaluate с единым бюджетом max_evals и учётом отказов.

    Каждый кандидат — одно вычисление problem.evaluate (при внешнем
    evaluator — одна передача пакета кандидатов). NaN/исключение кандидата
    дают loss=+inf и увеличивают n_failed; запись history помечается
    ``"failed": True``. Кандидаты, не дооценённые из-за бюджета (хвост
    ask-пакета), получают +inf БЕЗ учёта в n_evals/n_failed/history.
    """

    def __init__(self, problem, max_evals: Optional[int], evaluator) -> None:
        if max_evals is not None:
            max_evals = int(max_evals)
            if max_evals < 1:
                raise ValueError(
                    f"global_search: max_evals должен быть >= 1, {max_evals}."
                )
        self._problem = problem
        self._max_evals = max_evals
        self._evaluator = evaluator
        self.n_evals = 0
        self.n_failed = 0
        self.history: List[Dict[str, Any]] = []
        self._scored: List[Tuple[float, np.ndarray]] = []
        # (phi, F-строка) для nsga2:
        self.F_history: List[Tuple[np.ndarray, np.ndarray]] = []

    @property
    def max_evals(self) -> Optional[int]:
        return self._max_evals

    def remaining(self) -> float:
        if self._max_evals is None:
            return math.inf
        return max(0, self._max_evals - self.n_evals)

    def _n_take(self, n: int) -> int:
        remaining = self.remaining()
        if not np.isfinite(remaining):
            return n
        return max(0, min(n, int(remaining)))

    def _commit(self, loss: float, phi: Optional[np.ndarray]) -> float:
        """Учёт одной реальной оценки; возвращает записанный loss (+inf)."""
        loss = float(loss)
        failed = not np.isfinite(loss)
        self.n_evals += 1
        if failed:
            self.n_failed += 1
        self.history.append({"n_evals": self.n_evals, "loss": loss, "failed": failed})
        if not failed and phi is not None:
            self._scored.append((loss, np.asarray(phi, dtype=np.float64).copy()))
        return loss

    def _replay(self, losses: Sequence[float], phis: Sequence[np.ndarray]) -> None:
        """Восстановление истории из checkpoint (resume) без вычислений."""
        for loss, phi in zip(losses, phis):
            self._commit(loss, phi)

    def _eval_real(self, phis: Sequence[np.ndarray]) -> List[float]:
        """Реальная оценка пакета; отказы -> +inf (счётчики в _commit)."""
        phis = [np.asarray(p, dtype=np.float64).reshape(-1) for p in phis]
        if not phis:
            return []
        if self._evaluator is not None:
            values = [float(v) for v in self._evaluator(list(phis))]
            if len(values) != len(phis):
                raise ValueError(
                    "global_search: evaluator должен вернуть по значению "
                    f"на кандидата (получено {len(values)}, ожидалось "
                    f"{len(phis)})."
                )
            return [v if np.isfinite(v) else float("inf") for v in values]
        out = []
        for p in phis:
            try:
                loss = float(self._problem.evaluate(p).loss)
            except Exception:  # noqa: BLE001 - отказ кандидата не роняет прогон
                loss = float("inf")
            out.append(loss if np.isfinite(loss) else float("inf"))
        return out

    def evaluate_one(self, phi: np.ndarray) -> float:
        """
        Оценка одного кандидата; бросает _BudgetExhausted, если бюджет
        исчерпан (для SciPy-обёрток, где +inf не остановил бы решатель).
        """
        if self.remaining() <= 0:
            raise _BudgetExhausted(f"max_evals={self._max_evals} исчерпан")
        phi = np.asarray(phi, dtype=np.float64).reshape(-1)
        (loss,) = self._eval_real([phi])
        self._commit(loss, phi)
        return loss

    def evaluate_many(self, phis: Sequence[np.ndarray]) -> np.ndarray:
        """
        Пакетная оценка с мягким бюджетом: хвост пакета за пределами
        бюджета получает +inf и НЕ учитывается в счётчиках.
        """
        phis = [np.asarray(p, dtype=np.float64).reshape(-1) for p in phis]
        n_take = self._n_take(len(phis))
        losses = np.full(len(phis), np.inf, dtype=np.float64)
        for i, loss in enumerate(self._eval_real(phis[:n_take])):
            losses[i] = self._commit(loss, phis[i])
        return losses

    def evaluate_one_components(
        self, phi: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        """
        Оценка одного кандидата с возвратом (loss, components); бросает
        _BudgetExhausted, если бюджет исчерпан.
        """
        if self.remaining() <= 0:
            raise _BudgetExhausted(f"max_evals={self._max_evals} исчерпан")
        phi = np.asarray(phi, dtype=np.float64).reshape(-1)
        components: Dict[str, float] = {}
        try:
            ev = self._problem.evaluate(phi)
            components = {
                str(k): float(v)
                for k, v in (getattr(ev, "components", None) or {}).items()
            }
            loss = float(ev.loss)
        except Exception:  # noqa: BLE001 - отказ кандидата не роняет прогон
            loss, components = float("inf"), {}
        loss = self._commit(loss, phi)
        if not np.isfinite(loss):
            components = {}
        return loss, components

    def evaluate_components(
        self, phis: Sequence[np.ndarray], objective_names: Sequence[str]
    ) -> np.ndarray:
        """
        Пакетная оценка векторов целей из Evaluation.components.
        Отказ/недостающая компонента -> строка +inf (учёт как отказ).
        """
        names = list(objective_names)
        phis = [np.asarray(p, dtype=np.float64).reshape(-1) for p in phis]
        n_take = self._n_take(len(phis))
        F = np.full((len(phis), len(names)), np.inf, dtype=np.float64)
        for i, phi in enumerate(phis[:n_take]):
            row: Optional[np.ndarray] = None
            try:
                ev = self._problem.evaluate(phi)
                comp = getattr(ev, "components", None) or {}
                row = np.array([float(comp.get(nm, np.inf)) for nm in names])
            except Exception:  # noqa: BLE001 - отказ кандидата не роняет прогон
                row = None
            if row is None or not np.all(np.isfinite(row)):
                self._commit(float("inf"), None)
            else:
                # _commit уже добавляет (loss, phi) в _scored; здесь —
                # только история векторов целей для Pareto-фронта.
                self._commit(float(row.sum()), phi)
                self.F_history.append((phi.copy(), row.copy()))
            F[i] = float("inf") if row is None else row
        return F

    def best(self, k: int = 1) -> List[Tuple[float, np.ndarray]]:
        """Топ-k уникальных конечных кандидатов: [(loss, phi), ...]."""
        if not self._scored:
            return []
        order = sorted(range(len(self._scored)), key=lambda i: (self._scored[i][0], i))
        out: List[Tuple[float, np.ndarray]] = []
        seen = set()
        for i in order:
            loss, phi = self._scored[i]
            key = phi.tobytes()
            if key in seen:
                continue
            seen.add(key)
            out.append((loss, phi))
            if len(out) >= k:
                break
        return out


# ---------------------------------------------------------------------------
# Checkpoint helpers (ask/tell методы)
# ---------------------------------------------------------------------------


def _save_ask_tell_checkpoint(
    path,
    seed: Optional[int],
    config: Mapping[str, Any],
    completed: Sequence[Mapping[str, Any]],
) -> None:
    best_index = -1
    best_value = math.inf
    for i, gen in enumerate(completed):
        values = [float(v) for v in gen.get("y", [])[: int(gen.get("n_eval", 0))]]
        if values:
            value = min(values)
            if value < best_value:
                best_value = value
                best_index = i
    RunCheckpoint(
        seed=seed,
        config=dict(config),
        completed=[_jsonable(g) for g in completed],
        pending=[],  # открытый ask/tell-прогон: достаточно completed+бюджета
        best_index=best_index,
    ).save(os.fspath(path))


def _load_ask_tell_checkpoint(
    path, seed: Optional[int], config: Mapping[str, Any]
) -> List[Dict[str, Any]]:
    """Загрузка и валидация чекпоинта; resume восстанавливает дизайн из seed."""
    checkpoint = RunCheckpoint.load(os.fspath(path))
    if checkpoint.seed != seed:
        raise ValueError(
            f"global_search: seed чекпоинта ({checkpoint.seed}) не совпадает "
            f"с запрошенным ({seed}); resume невозможен."
        )
    stored = {k: v for k, v in checkpoint.config.items()}
    requested = {k: v for k, v in config.items()}
    if stored != requested:
        raise ValueError(
            "global_search: конфигурация чекпоинта не совпадает с "
            f"запрошенной: checkpoint={stored}, requested={requested}."
        )
    completed = []
    for item in checkpoint.completed:
        if not isinstance(item, Mapping):
            raise ValueError(
                "global_search: checkpoint.completed должен содержать "
                "словари поколений/итераций (получен не-словарный элемент)."
            )
        completed.append(dict(item))
    return completed


# ---------------------------------------------------------------------------
# Метод: Differential Evolution (SciPy)
# ---------------------------------------------------------------------------


def _run_de(oracle, phi0, bounds, seed, opts) -> Tuple[np.ndarray, float, bool, str]:
    lb, ub = _require_finite_bounds(bounds, "de")

    def fun(x) -> float:
        return _cap_loss(oracle.evaluate_one(np.asarray(x, dtype=np.float64)))

    kwargs = {
        "maxiter": int(opts.get("maxiter", 1000)),
        "popsize": int(opts.get("popsize", 15)),
        "tol": float(opts.get("tol", 0.01)),
        "mutation": opts.get("mutation", (0.5, 1.0)),
        "recombination": float(opts.get("recombination", 0.7)),
        "polish": bool(opts.get("polish", False)),
        "init": opts.get("init", "latinhypercube"),
        "updating": "immediate",
        # workers= (параллелизм SciPy) намеренно не используется;
        # внешний параллелизм — через evaluator.
    }
    x0 = np.clip(phi0, lb, ub)
    try:
        res = sp_optimize.differential_evolution(
            fun, sp_optimize.Bounds(lb, ub), x0=x0, rng=seed, **kwargs
        )
    except _BudgetExhausted:
        best = oracle.best(1)
        if not best:
            return (
                x0,
                float("inf"),
                False,
                (
                    f"бюджет исчерпан (max_evals={oracle.max_evals}) до первой "
                    "успешной оценки"
                ),
            )
        ((loss, phi),) = best
        return (
            phi,
            loss,
            True,
            (f"бюджет исчерпан (max_evals={oracle.max_evals}); лучшая оценённая точка"),
        )
    phi_best = np.asarray(res.x, dtype=np.float64).reshape(-1)
    loss = float(res.fun)
    if not np.isfinite(loss):
        best = oracle.best(1)
        if best:
            loss, phi_best = best[0]
    return phi_best, loss, bool(res.success), str(res.message)


# ---------------------------------------------------------------------------
# Метод: CMA-ES (опциональный пакет cma, ask/tell)
# ---------------------------------------------------------------------------


def _import_cma():
    try:
        import cma as cma_mod
    except ImportError as exc:
        raise OptionalDependencyError(
            "global_search(method='cma'): требуется пакет 'cma'. "
            f"Установите extra: {_INSTALL_GLOBAL}."
        ) from exc
    return cma_mod


def _run_cma(
    oracle, phi0, bounds, seed, opts, checkpoint_path, resume, config
) -> Tuple[np.ndarray, float, bool, str]:
    cma_mod = _import_cma()
    if bounds is not None and not (
        np.all(np.isfinite(bounds[0])) and np.all(np.isfinite(bounds[1]))
    ):
        raise ValueError(
            "global_search(cma): CMA-ES принимает либо полные конечные "
            "границы, либо неограниченную задачу; частичные границы не "
            "поддерживаются (см. transforms для параметризации)."
        )
    sigma0 = opts.get("sigma0")
    if sigma0 is None:
        if bounds is not None:
            sigma0 = 0.25 * float(np.median(bounds[1] - bounds[0]))
        else:
            sigma0 = 1.0
        sigma0 = max(float(sigma0), 1e-12)
    cma_opts: Dict[str, Any] = {
        "maxiter": int(opts.get("maxiter", 10000)),
        "verbose": -9,
    }
    if opts.get("popsize") is not None:
        cma_opts["popsize"] = int(opts["popsize"])
    if seed is not None:
        cma_opts["seed"] = int(seed)
    if bounds is not None:
        cma_opts["bounds"] = [bounds[0].tolist(), bounds[1].tolist()]
    cma_opts.update(dict(opts.get("cma_options") or {}))

    es = cma_mod.CMAEvolutionStrategy(phi0.tolist(), float(sigma0), cma_opts)

    completed: List[Dict[str, Any]] = []
    if checkpoint_path is not None and resume and os.path.exists(checkpoint_path):
        completed = _load_ask_tell_checkpoint(checkpoint_path, seed, config)
        for gen in completed:
            # Дизайн восстанавливается из seed: ask повторяет спрос. В tell
            # передаются те же массивы, что вернул ask (как в живом цикле):
            # копии из JSON cma считает внешними (injected) решениями и
            # обрабатывает иначе, из-за чего resume расходился с непрерывным
            # прогоном на ~1e-10.
            asked = [np.asarray(x, dtype=np.float64).reshape(-1) for x in es.ask()]
            stored = [np.asarray(x, dtype=np.float64).reshape(-1) for x in gen["x"]]
            same = len(asked) == len(stored) and all(
                np.array_equal(a, b) for a, b in zip(asked, stored)
            )
            if not same:
                warnings.warn(
                    "global_search(cma): resume — ask() после восстановления из "
                    "seed не совпал с сохранённым поколением (другая версия "
                    "cma или опции); используются сохранённые точки.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            es.tell(asked if same else stored, gen["y"])
            n_eval = int(gen.get("n_eval", len(gen["y"])))
            oracle._replay(
                [float(v) for v in gen["y"][:n_eval]],
                [np.asarray(x, dtype=np.float64) for x in gen["x"][:n_eval]],
            )

    while not es.stop() and oracle.remaining() > 0:
        xs = [np.asarray(x, dtype=np.float64).reshape(-1) for x in es.ask()]
        n_take = oracle._n_take(len(xs))
        losses = oracle.evaluate_many(xs)
        ys = [float(v) for v in losses]
        es.tell(xs, ys)  # +inf у недооценённого хвоста допустимы
        completed.append(
            {
                "x": [x.tolist() for x in xs],
                "y": ys,
                "n_eval": int(n_take),
            }
        )
        if checkpoint_path is not None:
            _save_ask_tell_checkpoint(checkpoint_path, seed, config, completed)

    best = oracle.best(1)
    if best:
        loss_best, phi_best = best[0]
    else:  # все оценки отказные: берём лучшее, что знает CMA
        phi_best = np.asarray(es.best.x, dtype=np.float64).reshape(-1)
        loss_best = float(es.best.f)
    stop = es.stop()
    if stop:
        message = "cma stop: " + ", ".join(sorted(stop))
    elif oracle.remaining() <= 0:
        message = f"бюджет исчерпан (max_evals={oracle.max_evals})"
    else:
        message = "cma: цикл завершён без критериев остановки"
    return phi_best, loss_best, bool(np.isfinite(loss_best)), message


# ---------------------------------------------------------------------------
# Метод: SPSA (numpy, ask/tell)
# ---------------------------------------------------------------------------


def _spsa_scale(phi0: np.ndarray, bounds) -> np.ndarray:
    if bounds is None:
        return np.maximum(np.abs(phi0), 1.0)
    span = bounds[1] - bounds[0]
    fallback = np.maximum(np.abs(phi0), 1.0)
    return np.where(np.isfinite(span), span, fallback)


def _run_spsa(
    oracle, phi0, bounds, seed, opts, checkpoint_path, resume, config
) -> Tuple[np.ndarray, float, bool, str]:
    size = int(phi0.shape[0])
    scale = _spsa_scale(phi0, bounds)
    alpha = float(opts.get("alpha", 0.602))
    gamma = float(opts.get("gamma", 0.101))
    max_evals = oracle.max_evals
    max_iter = opts.get("max_iter")
    if max_iter is None:
        max_iter = 200 if max_evals is None else max(1, (int(max_evals) - 1) // 2)
    max_iter = int(max_iter)
    A = float(opts.get("A", 0.1 * max_iter))
    a = opts.get("a")
    a_vec = np.broadcast_to(
        float(opts.get("a_mult", 0.05)) * scale if a is None else float(a), (size,)
    ).astype(np.float64)
    c = opts.get("c")
    c_vec = np.broadcast_to(
        float(opts.get("c_mult", 0.02)) * scale if c is None else float(c), (size,)
    ).astype(np.float64)

    rng = np.random.default_rng(seed)

    def _draw_delta() -> np.ndarray:
        return (rng.random(size) >= 0.5) * 2.0 - 1.0

    def _spsa_update(
        theta: np.ndarray, k: int, delta: np.ndarray, f_plus: float, f_minus: float
    ) -> np.ndarray:
        """Один шаг SPSA; идентичен в live-цикле и при replay resume."""
        if not (np.isfinite(f_plus) and np.isfinite(f_minus)):
            return theta
        c_k = c_vec / k**gamma
        g_hat = (f_plus - f_minus) / (2.0 * c_k * delta)
        theta_next = theta - (a_vec / (k + A) ** alpha) * g_hat
        if not np.all(np.isfinite(theta_next)):
            return theta
        return _clip_to_bounds(theta_next, bounds)

    theta = phi0.copy()
    completed: List[Dict[str, Any]] = []
    if checkpoint_path is not None and resume and os.path.exists(checkpoint_path):
        completed = _load_ask_tell_checkpoint(checkpoint_path, seed, config)
        gen0 = completed[0]
        theta = np.asarray(gen0["theta"], dtype=np.float64).reshape(-1)
        oracle._replay(
            [float(gen0["y"][0])], [theta]
        )  # запись k=0: начальная точка phi0
        for gen in completed[1:]:
            k = int(gen["k"])
            delta = _draw_delta()  # последовательность повторяет прогон
            c_k = c_vec / k**gamma
            plus = _clip_to_bounds(theta + c_k * delta, bounds)
            minus = _clip_to_bounds(theta - c_k * delta, bounds)
            n_eval = int(gen.get("n_eval", 2))
            oracle._replay(
                [float(v) for v in gen["y"][:n_eval]],
                [plus, minus][:n_eval],
            )
            theta = _spsa_update(
                theta, k, delta, float(gen["y"][0]), float(gen["y"][1])
            )
    else:
        f0 = float(oracle.evaluate_many([theta])[0])
        completed.append({"k": 0, "theta": theta.tolist(), "y": [f0], "n_eval": 1})
        if checkpoint_path is not None:
            _save_ask_tell_checkpoint(checkpoint_path, seed, config, completed)

    k = int(completed[-1]["k"]) if completed else 0
    while oracle.remaining() > 0 and k < max_iter:
        k += 1
        delta = _draw_delta()
        c_k = c_vec / k**gamma
        plus = _clip_to_bounds(theta + c_k * delta, bounds)
        minus = _clip_to_bounds(theta - c_k * delta, bounds)
        n_take = oracle._n_take(2)
        losses = oracle.evaluate_many([plus, minus])
        f_plus, f_minus = float(losses[0]), float(losses[1])
        completed.append(
            {
                "k": k,
                "theta": theta.tolist(),
                "y": [f_plus, f_minus],
                "n_eval": int(n_take),
            }
        )
        if checkpoint_path is not None:
            _save_ask_tell_checkpoint(checkpoint_path, seed, config, completed)
        theta = _spsa_update(theta, k, delta, f_plus, f_minus)

    best = oracle.best(1)
    if best:
        loss_best, phi_best = best[0]
    else:
        phi_best, loss_best = theta, float("inf")
    if oracle.remaining() <= 0:
        message = f"бюджет исчерпан (max_evals={oracle.max_evals})"
    elif k >= max_iter:
        message = f"достигнут max_iter={max_iter}"
    else:
        message = "spsa: цикл завершён"
    return phi_best, loss_best, bool(np.isfinite(loss_best)), message


# ---------------------------------------------------------------------------
# Метод: basin hopping (SciPy)
# ---------------------------------------------------------------------------


class _BoundedDisplacement:
    """Смещение шага basin hopping, зажатое в границы (детерминировано)."""

    def __init__(self, stepsize: float, bounds, rng: np.random.Generator) -> None:
        self._stepsize = float(stepsize)
        self._bounds = bounds
        self._rng = rng

    def __call__(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        step = self._rng.normal(0.0, self._stepsize, size=x.shape)
        return _clip_to_bounds(x + step, self._bounds)


def _run_basinhopping(
    oracle, phi0, bounds, seed, opts
) -> Tuple[np.ndarray, float, bool, str]:
    niter = int(opts.get("niter", 100))
    stepsize = float(opts.get("stepsize", 0.5))
    T = float(opts.get("T", 1.0))
    inner_maxiter = int(opts.get("inner_maxiter", 50))
    x0 = _clip_to_bounds(phi0, bounds)

    def fun(x) -> float:
        return _cap_loss(oracle.evaluate_one(np.asarray(x, dtype=np.float64)))

    rng = np.random.default_rng(seed)
    stepper = _BoundedDisplacement(stepsize, bounds, rng)
    minimizer_kwargs: Dict[str, Any] = {
        "method": "L-BFGS-B",
        "options": {"maxiter": inner_maxiter},
    }
    if bounds is not None:
        minimizer_kwargs["bounds"] = sp_optimize.Bounds(bounds[0], bounds[1])
    try:
        res = sp_optimize.basinhopping(
            fun,
            x0,
            minimizer_kwargs=minimizer_kwargs,
            niter=niter,
            T=T,
            take_step=stepper,
            rng=seed,
        )
    except _BudgetExhausted:
        best = oracle.best(1)
        if not best:
            return (
                x0,
                float("inf"),
                False,
                (
                    f"бюджет исчерпан (max_evals={oracle.max_evals}) до первой "
                    "успешной оценки"
                ),
            )
        ((loss, phi),) = best
        return (
            phi,
            loss,
            True,
            (f"бюджет исчерпан (max_evals={oracle.max_evals}); лучшая оценённая точка"),
        )
    low = getattr(res, "lowest_optimization_result", res)
    phi_best = np.asarray(low.x, dtype=np.float64).reshape(-1)
    loss = float(low.fun)
    if not np.isfinite(loss):
        best = oracle.best(1)
        if best:
            loss, phi_best = best[0]
    success = bool(getattr(res, "success", np.isfinite(loss)))
    message = str(getattr(res, "message", getattr(low, "message", "")))
    return phi_best, loss, success, message


# ---------------------------------------------------------------------------
# Метод: NSGA-II (опциональный pymoo)
# ---------------------------------------------------------------------------


@dataclass
class _NSGA2Output:
    frontier_phi: np.ndarray  # [m, size]
    frontier_F: np.ndarray  # [m, n_obj]
    objective_names: List[str]
    success: bool
    message: str


def _run_nsga2(oracle, phi0, bounds, seed, opts, objective_names) -> _NSGA2Output:
    # Проверки, не требующие pymoo, — до импорта.
    lb, ub = _require_finite_bounds(bounds, "nsga2")
    if objective_names is None:
        _, components = oracle.evaluate_one_components(phi0)
        names = sorted(components)
    else:
        names = [str(n) for n in objective_names]
    if len(names) < 2:
        raise ValueError(
            "global_search(nsga2): нужно >= 2 целевых компонент в "
            f"Evaluation.components, получены {names}."
        )
    try:
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.core.problem import Problem as _PymooProblem
        from pymoo.optimize import minimize as _pymoo_minimize
        from pymoo.termination import get_termination
    except ImportError as exc:
        raise OptionalDependencyError(
            "global_search(method='nsga2'): требуется пакет 'pymoo'. "
            f"Установите extra: {_INSTALL_PARETO}."
        ) from exc

    size = int(phi0.shape[0])
    pop_size = int(opts.get("pop_size", 40))
    oracle_ref = oracle
    names_ref = names

    class _VectorProblem(_PymooProblem):
        def __init__(self) -> None:
            super().__init__(n_var=size, n_obj=len(names_ref), xl=lb, xu=ub)

        def _evaluate(self, x, out, *args, **kwargs) -> None:
            rows = [np.asarray(r, dtype=np.float64).reshape(-1) for r in x]
            if oracle_ref.remaining() <= 0:
                raise _BudgetExhausted(f"max_evals={oracle_ref.max_evals} исчерпан")
            out["F"] = oracle_ref.evaluate_components(rows, names_ref)

    problem = _VectorProblem()
    algorithm = NSGA2(pop_size=pop_size)
    if oracle.max_evals is not None:
        termination = get_termination("n_eval", int(oracle.max_evals))
    else:
        termination = get_termination("n_gen", int(opts.get("n_gen", 50)))

    message = "nsga2: завершено"
    try:
        _pymoo_minimize(
            problem,
            algorithm,
            termination,
            seed=seed,
            verbose=False,
        )
    except _BudgetExhausted:
        message = f"бюджет исчерпан (max_evals={oracle.max_evals})"

    if not oracle.F_history:
        raise RuntimeError(
            "global_search(nsga2): не удалось оценить ни одного кандидата."
        )
    phis = np.stack([p for p, _ in oracle.F_history])
    F = np.stack([f for _, f in oracle.F_history])
    mask = _non_dominated_mask(F)
    success = bool(np.any(mask))
    if not success:
        message = "nsga2: все оценки отказные"
    return _NSGA2Output(
        frontier_phi=phis[mask],
        frontier_F=F[mask],
        objective_names=names,
        success=success,
        message=message,
    )


# ---------------------------------------------------------------------------
# Доводка top-k (ленивый импорт strategies/local)
# ---------------------------------------------------------------------------


def _refine_candidates(problem, scored, k: int, local) -> List[StartResult]:
    """Локальная доводка top-k кандидатов; ленивый импорт адаптеров."""
    from neuraltide.optimization.strategies import resolve_local_optimizer

    local_fn, _ = resolve_local_optimizer(local)
    candidates: List[StartResult] = []
    for index, (loss0, phi0) in enumerate(scored[:k]):
        start = StartResult(
            index=index,
            x0_origin="global",
            phi0=np.asarray(phi0, dtype=np.float64).copy(),
            phi=np.asarray(phi0, dtype=np.float64).copy(),
            loss=float(loss0),
            status="pending",
        )
        try:
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
        except Exception as exc:  # noqa: BLE001 - доводка не роняет прогон
            start.status = "failed"
            start.message = f"refinement: {exc}"
        candidates.append(start)
    return candidates


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------


def global_search(
    problem,
    *,
    method: str = "de",
    max_evals: Optional[int] = None,
    bounds=None,
    phi0=None,
    seed: Optional[int] = None,
    refine_top_k: int = 0,
    local="lbfgsb",
    checkpoint_path=None,
    resume: bool = False,
    evaluator: Optional[Callable[[Sequence[np.ndarray]], Sequence[float]]] = None,
    options: Optional[Mapping[str, Any]] = None,
    objective_names: Optional[Sequence[str]] = None,
    validate: bool = True,
):
    """
    Глобальный поиск поверх CalibrationProblem.

    Args:
        problem: CalibrationProblem (или duck-typed: size/pack/evaluate/
            bounds_phi/unpack, опционально value_and_gradient/validate).
        method: 'de' | 'cma' | 'spsa' | 'basinhopping' | 'nsga2'.
        max_evals: единый бюджет оценок problem.evaluate глобальной фазы
            (None — ограничения только критериями метода). Доводка
            refine_top_k имеет собственный бюджет локального метода.
        bounds: опциональные (lb, ub); иначе problem.bounds_phi().
        phi0: стартовая точка (None — problem.pack()); для 'de' входит
            в начальную популяцию.
        seed: seed всей случайности метода (None — недетерминированно).
        refine_top_k: сколько лучших кандидатов глобальной фазы довести
            локально ('lbfgsb' | 'least_squares' | callable).
        local: локальный оптимизатор для доводки.
        checkpoint_path: JSON-чекпоинт поколений/итераций (только
            'cma'/'spsa'; после каждого поколения/итерации).
        resume: дочитать checkpoint_path; дизайн восстанавливается из
            seed (документированная аппроксимация), досчитывается бюджет
            max_evals - sum(completed.n_eval), результат совпадает с
            непрерывным прогоном. Для 'de'/'basinhopping' не поддержан.
        evaluator: callable([[phi, ...]]) -> [loss, ...] — внешний
            пакетный параллелизм без зависимости от parallel.py;
            несовместим с method='nsga2' (нужны компоненты целей).
        options: метод-специфичные опции (см. _OPTION_KEYS).
        objective_names: имена компонент целей для 'nsga2' (None —
            сорти­рованные ключи Evaluation.components первой оценки).
        validate: выполнить problem.validate(phi_best) при непустом
            held_out.

    Returns:
        GlobalOptimizerResult (методы с одной целью) или ParetoResult
        (method='nsga2').
    """
    t_begin = time.perf_counter()
    method_key = str(method).lower()
    if method_key not in _GLOBAL_METHODS:
        raise ValueError(
            f"global_search: неизвестный method={method!r}; доступны "
            f"{list(_GLOBAL_METHODS)}."
        )
    if evaluator is not None and not callable(evaluator):
        raise TypeError(
            f"global_search: evaluator должен быть callable, получен "
            f"{type(evaluator).__name__}."
        )
    if method_key == "nsga2" and evaluator is not None:
        raise ValueError(
            "global_search(nsga2): evaluator несовместим — целевые "
            "векторы берутся из problem.evaluate(...).components."
        )
    if method_key == "nsga2" and int(refine_top_k) > 0:
        raise ValueError(
            "global_search(nsga2): refine_top_k не поддержан "
            "(результат — ParetoResult)."
        )
    if checkpoint_path is not None and method_key not in _ASK_TELL_METHODS:
        raise ValueError(
            f"global_search({method_key}): checkpoint_path/resume "
            f"поддержаны только для методов {list(_ASK_TELL_METHODS)} "
            "(scipy DE и basinhopping не восстанавливаются из seed)."
        )
    opts = _check_options(method_key, options)
    size = int(problem.size)
    if size < 1:
        raise ValueError(f"global_search: пустая задача (size={size}).")
    phi0_arr = _resolve_phi0(problem, phi0, size)
    bounds_arr = _resolve_bounds(problem, size, bounds)

    fwd0 = int(getattr(problem, "n_forward", 0))
    grad0 = int(getattr(problem, "n_gradient", 0))
    oracle = _Oracle(problem, max_evals, evaluator)
    config: Dict[str, Any] = {
        "method": method_key,
        "max_evals": oracle.max_evals,
        "size": size,
        "phi0": phi0_arr.tolist(),
        "bounds": (
            None
            if bounds_arr is None
            else [bounds_arr[0].tolist(), bounds_arr[1].tolist()]
        ),
        "options": _jsonable(opts),
        "refine_top_k": int(refine_top_k),
        "local": getattr(local, "__name__", str(local)),
    }

    if method_key == "nsga2":
        out = _run_nsga2(oracle, phi0_arr, bounds_arr, seed, opts, objective_names)
        return _assemble_pareto(
            problem,
            oracle,
            out,
            fwd0=fwd0,
            t_begin=t_begin,
            seed=seed,
            config=config,
        )

    if method_key == "de":
        phi_best, loss_best, success, message = _run_de(
            oracle, phi0_arr, bounds_arr, seed, opts
        )
    elif method_key == "cma":
        phi_best, loss_best, success, message = _run_cma(
            oracle,
            phi0_arr,
            bounds_arr,
            seed,
            opts,
            checkpoint_path,
            bool(resume),
            config,
        )
    elif method_key == "spsa":
        phi_best, loss_best, success, message = _run_spsa(
            oracle,
            phi0_arr,
            bounds_arr,
            seed,
            opts,
            checkpoint_path,
            bool(resume),
            config,
        )
    else:  # basinhopping
        phi_best, loss_best, success, message = _run_basinhopping(
            oracle, phi0_arr, bounds_arr, seed, opts
        )

    candidates: List[StartResult] = []
    if int(refine_top_k) > 0:
        candidates = _refine_candidates(
            problem, oracle.best(int(refine_top_k)), int(refine_top_k), local
        )
        ok = [c for c in candidates if c.status == "ok" and np.isfinite(c.loss)]
        if ok:
            best_refined = min(ok, key=lambda c: (c.loss, c.index))
            if best_refined.loss <= loss_best or not np.isfinite(loss_best):
                phi_best = np.asarray(best_refined.phi, dtype=np.float64)
                loss_best = float(best_refined.loss)

    # Финальная явная оценка лучшей точки (честный loss_best/theta_best).
    final_loss = float("inf")
    try:
        final_eval = problem.evaluate(np.asarray(phi_best, dtype=np.float64))
        final_loss = float(final_eval.loss)
    except Exception as exc:  # noqa: BLE001 - отчитываемся даже при отказе
        message = f"{message}; финальная оценка не удалась: {exc}"
    if np.isfinite(final_loss):
        loss_best = final_loss

    history = list(oracle.history)
    dtype_name = str(neuraltide.config.get_dtype().name)
    base = OptimizationResult(
        phi_best=np.asarray(phi_best, dtype=np.float64),
        theta_best=_theta_best(problem, np.asarray(phi_best, dtype=np.float64)),
        loss_best=float(loss_best),
        history=history,
        seed=seed,
        config=config,
        dtype=dtype_name,
    )
    validation: Dict[str, float] = {}
    if validate and getattr(problem, "held_out", None):
        validation = {
            str(k): float(v)
            for k, v in problem.validate(np.asarray(phi_best, dtype=np.float64)).items()
        }
    return GlobalOptimizerResult(
        base=base,
        method=method_key,
        success=bool(success and np.isfinite(loss_best)),
        message=str(message),
        n_evals=int(oracle.n_evals),
        n_forward=int(getattr(problem, "n_forward", 0)) - fwd0,
        n_gradient=int(getattr(problem, "n_gradient", 0)) - grad0,
        wall_time=time.perf_counter() - t_begin,
        history=history,
        candidates=candidates,
        validation=validation,
    )


def _assemble_pareto(
    problem, oracle, out: _NSGA2Output, *, fwd0, t_begin, seed, config
) -> ParetoResult:
    F = np.asarray(out.frontier_F, dtype=np.float64)
    phis = np.asarray(out.frontier_phi, dtype=np.float64)
    span = np.ptp(F, axis=0)
    scale = np.where(span > 0, span, 1.0)
    norm = (F - F.min(axis=0)) / scale
    knee = int(np.argmin(norm.sum(axis=1)))
    phi_best = np.asarray(phis[knee], dtype=np.float64).reshape(-1)
    dtype_name = str(neuraltide.config.get_dtype().name)
    base = OptimizationResult(
        phi_best=phi_best,
        theta_best=_theta_best(problem, phi_best),
        loss_best=float(F[knee].sum()),
        history=list(oracle.history),
        seed=seed,
        config=config,
        dtype=dtype_name,
    )
    return ParetoResult(
        base=base,
        frontier={name: F[:, j].copy() for j, name in enumerate(out.objective_names)},
        objective_names=list(out.objective_names),
        method="nsga2",
        success=bool(out.success),
        message=str(out.message),
        n_evals=int(oracle.n_evals),
        n_forward=int(getattr(problem, "n_forward", 0)) - fwd0,
        wall_time=time.perf_counter() - t_begin,
        history=list(oracle.history),
        seed=seed,
        config=config,
    )
