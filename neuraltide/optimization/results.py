"""
Форматы результатов калибровки.

Evaluation — результат одной оценки задачи CalibrationProblem: скалярная
потеря, компоненты, невязки, градиент, счётчики прямых прогонов и
градиентов, wall time, dtype и шаг интегрирования.

OptimizationResult — единый формат результата оптимизации: лучшие φ и
физические θ, история, seed, конфигурация, версии пакетов и dtype.
Сериализация — JSON с преобразованием numpy -> list; сериализация модели
сознательно не входит в контракт (модель восстанавливается через
ParameterSpace.apply_to_model(theta_best)).

Производные DTO строятся композицией (каждый содержит base:
OptimizationResult) и используют единый механизм сериализации
(_JsonDTO): to_dict/from_dict/save/load, numpy -> list:

    LocalOptimizerResult — результат локального оптимизатора (SciPy
        L-BFGS-B / least_squares TRF/LM, Keras-адаптеры): статус,
        число итераций/вычислений, норма градиента, активные границы,
        счётчики прогонов, validation-метрики;
    StartResult — один старт мультистарта ( screening + local );
    MultistartResult — итог мультистарта: все старты, лучший индекс,
        кластеризация, seed;
    RunCheckpoint — чекпоинт прерванного запуска (completed/pending);
    JacobianResult — якобиан невязок в точке φ (forward sensitivities).
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Union

import numpy as np
import tensorflow as tf

import neuraltide

__all__ = [
    "Evaluation",
    "OptimizationResult",
    "LocalOptimizerResult",
    "StartResult",
    "MultistartResult",
    "RunCheckpoint",
    "JacobianResult",
]


def _default_versions() -> Dict[str, str]:
    """Версии пакетов, влияющих на воспроизводимость вычислений."""
    return {
        "neuraltide": getattr(neuraltide, "__version__", "unknown"),
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
    }


@dataclass
class Evaluation:
    """
    Результат одного вычисления CalibrationProblem.

    Атрибуты:
        loss: скалярная потеря (float);
        components: взвешенные компоненты потери {имя: float};
        residuals: вектор невязок (numpy) либо массив нулевой длины,
            если objective не предоставляет residual-интерфейс;
        grad: градиент по φ (numpy) либо None (в evaluate/residual);
        status: 'ok' | 'no_residual_interface';
        n_forward: суммарное число прямых прогонов сети на момент оценки;
        n_gradient: суммарное число вычислений градиента;
        wall_time: длительность этого вызова, секунды;
        dtype: имя глобального вычислительного dtype;
        dt: шаг интегрирования сети, мс.
    """

    loss: float
    components: Dict[str, float] = field(default_factory=dict)
    residuals: np.ndarray = field(default_factory=lambda: np.zeros(0))
    grad: Optional[np.ndarray] = None
    status: str = "ok"
    n_forward: int = 0
    n_gradient: int = 0
    wall_time: float = 0.0
    dtype: str = "float32"
    dt: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Словарь с JSON-сериализуемыми значениями (numpy -> list)."""
        return {
            "loss": float(self.loss),
            "components": {str(k): float(v) for k, v in self.components.items()},
            "residuals": np.asarray(self.residuals).tolist(),
            "grad": (
                None
                if self.grad is None
                else np.asarray(self.grad, dtype=np.float64).tolist()
            ),
            "status": str(self.status),
            "n_forward": int(self.n_forward),
            "n_gradient": int(self.n_gradient),
            "wall_time": float(self.wall_time),
            "dtype": str(self.dtype),
            "dt": float(self.dt),
        }


@dataclass
class OptimizationResult:
    """
    Единый формат результата оптимизации.

    Атрибуты:
        phi_best: лучший вектор φ оптимизатора;
        theta_best: физические значения {full_name: ndarray в исходной форме};
        loss_best: лучшая достигнутая потеря;
        history: список записей итераций (словарей), например
            [{"loss": ..., "wall_time": ...}, ...];
        seed: seed генератора случайных чисел (или None);
        config: пользовательская конфигурация запуска (dict);
        versions: версии пакетов (по умолчанию заполняются автоматически);
        dtype: имя вычислительного dtype.
    """

    phi_best: np.ndarray
    theta_best: Mapping[str, np.ndarray]
    loss_best: float
    history: List[Dict[str, Any]] = field(default_factory=list)
    seed: Optional[int] = None
    config: Dict[str, Any] = field(default_factory=dict)
    versions: Dict[str, str] = field(default_factory=dict)
    dtype: str = "float32"

    def __post_init__(self) -> None:
        self.phi_best = np.asarray(self.phi_best, dtype=np.float64)
        self.theta_best = {
            str(name): np.asarray(value) for name, value in self.theta_best.items()
        }
        self.loss_best = float(self.loss_best)
        if not self.versions:
            self.versions = _default_versions()

    def to_dict(self) -> Dict[str, Any]:
        """Словарь с JSON-сериализуемыми значениями (numpy -> list)."""
        return {
            "phi_best": self.phi_best.tolist(),
            "theta_best": {
                name: np.asarray(value).tolist()
                for name, value in self.theta_best.items()
            },
            "loss_best": float(self.loss_best),
            "history": list(self.history),
            "seed": self.seed,
            "config": dict(self.config),
            "versions": dict(self.versions),
            "dtype": str(self.dtype),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OptimizationResult":
        """Обратное преобразование to_dict; списки -> numpy массивы."""
        return cls(
            phi_best=np.asarray(data["phi_best"], dtype=np.float64),
            theta_best={
                str(name): np.asarray(value)
                for name, value in data["theta_best"].items()
            },
            loss_best=float(data["loss_best"]),
            history=list(data.get("history", [])),
            seed=data.get("seed"),
            config=dict(data.get("config", {})),
            versions=dict(data.get("versions", {})),
            dtype=str(data.get("dtype", "float32")),
        )

    def save(self, path: Union[str, os.PathLike]) -> None:
        """Сохраняет результат в JSON-файл."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Union[str, os.PathLike]) -> "OptimizationResult":
        """Загружает результат из JSON-файла, записанного save()."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


# ---------------------------------------------------------------------------
# Производные DTO
# ---------------------------------------------------------------------------


def _encode(value: Any) -> Any:
    """Рекурсивное приведение к JSON-сериализуемым значениям (numpy -> list)."""
    if isinstance(value, OptimizationResult):
        return value.to_dict()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(v) for v in value]
    return value


class _JsonDTO:
    """Единый механизм сериализации DTO: to_dict/from_dict + save/load."""

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - интерфейс
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]):  # pragma: no cover - интерфейс
        raise NotImplementedError

    def save(self, path: Union[str, os.PathLike]) -> None:
        """Сохраняет DTO в JSON-файл."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Union[str, os.PathLike]):
        """Загружает DTO из JSON-файла, записанного save()."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class LocalOptimizerResult(_JsonDTO):
    """
    Результат локального оптимизатора поверх CalibrationProblem.

    Атрибуты:
        base: базовый OptimizationResult (phi_best/theta_best/loss_best/
            история/seed/config);
        method: имя метода ('L-BFGS-B', 'TRF', 'LM', ...);
        success: признак успешной сходимости;
        status: числовой/строковый статус решателя;
        message: сообщение решателя;
        nit: число итераций;
        nfev / njev: число оценок функции / градиента (якобиана);
        grad_norm: норма градиента на выходе;
        bounds_active: имена записей, прижатых к границе;
        n_forward / n_gradient: счётчики прямых прогонов и градиентов;
        wall_time: накопленное время, секунды;
        validation: метрики validate() на held-out протоколах.
    """

    base: OptimizationResult
    method: str = "L-BFGS-B"
    success: bool = False
    status: int = 0
    message: str = ""
    nit: int = 0
    nfev: int = 0
    njev: int = 0
    grad_norm: float = 0.0
    bounds_active: List[str] = field(default_factory=list)
    n_forward: int = 0
    n_gradient: int = 0
    wall_time: float = 0.0
    validation: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base": self.base.to_dict(),
            "method": str(self.method),
            "success": bool(self.success),
            "status": int(self.status),
            "message": str(self.message),
            "nit": int(self.nit),
            "nfev": int(self.nfev),
            "njev": int(self.njev),
            "grad_norm": float(self.grad_norm),
            "bounds_active": [str(x) for x in self.bounds_active],
            "n_forward": int(self.n_forward),
            "n_gradient": int(self.n_gradient),
            "wall_time": float(self.wall_time),
            "validation": {str(k): float(v) for k, v in self.validation.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LocalOptimizerResult":
        return cls(
            base=OptimizationResult.from_dict(data["base"]),
            method=str(data.get("method", "L-BFGS-B")),
            success=bool(data.get("success", False)),
            status=int(data.get("status", 0)),
            message=str(data.get("message", "")),
            nit=int(data.get("nit", 0)),
            nfev=int(data.get("nfev", 0)),
            njev=int(data.get("njev", 0)),
            grad_norm=float(data.get("grad_norm", 0.0)),
            bounds_active=[str(x) for x in data.get("bounds_active", [])],
            n_forward=int(data.get("n_forward", 0)),
            n_gradient=int(data.get("n_gradient", 0)),
            wall_time=float(data.get("wall_time", 0.0)),
            validation={
                str(k): float(v) for k, v in data.get("validation", {}).items()
            },
        )


@dataclass
class StartResult(_JsonDTO):
    """
    Один старт мультистарта.

    Атрибуты:
        index: порядковый номер старта;
        x0_origin: происхождение начальной точки ('lhs', 'sobol',
            'manual', 'checkpoint', ...);
        phi0: стартовый вектор φ;
        phi: финальный вектор φ (после доводки);
        loss: финальная потеря;
        status: строковый статус ('ok', 'failed', 'screened', ...);
        message: сообщение/причина отказа;
        screened_loss: потеря после короткого screening (или None);
        refined: была ли выполнена полная локальная доводка;
        screen_refined: был ли выполнен короткий screen-refine при
            screening (улучшение screened_loss; см. multistart
            screen_refine_steps);
        local: результат локального оптимизатора (или None);
        wall_time: время старта, секунды.
    """

    index: int = 0
    x0_origin: str = "manual"
    phi0: np.ndarray = field(default_factory=lambda: np.zeros(0))
    phi: np.ndarray = field(default_factory=lambda: np.zeros(0))
    loss: float = 0.0
    status: str = "pending"
    message: str = ""
    screened_loss: Optional[float] = None
    refined: bool = False
    screen_refined: bool = False
    local: Optional[LocalOptimizerResult] = None
    wall_time: float = 0.0

    def __post_init__(self) -> None:
        self.phi0 = np.asarray(self.phi0, dtype=np.float64)
        self.phi = np.asarray(self.phi, dtype=np.float64)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": int(self.index),
            "x0_origin": str(self.x0_origin),
            "phi0": self.phi0.tolist(),
            "phi": self.phi.tolist(),
            "loss": float(self.loss),
            "status": str(self.status),
            "message": str(self.message),
            "screened_loss": (
                None if self.screened_loss is None else float(self.screened_loss)
            ),
            "refined": bool(self.refined),
            "screen_refined": bool(self.screen_refined),
            "local": None if self.local is None else self.local.to_dict(),
            "wall_time": float(self.wall_time),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StartResult":
        local = data.get("local")
        screened = data.get("screened_loss")
        return cls(
            index=int(data.get("index", 0)),
            x0_origin=str(data.get("x0_origin", "manual")),
            phi0=np.asarray(data.get("phi0", []), dtype=np.float64),
            phi=np.asarray(data.get("phi", []), dtype=np.float64),
            loss=float(data.get("loss", 0.0)),
            status=str(data.get("status", "pending")),
            message=str(data.get("message", "")),
            screened_loss=None if screened is None else float(screened),
            refined=bool(data.get("refined", False)),
            screen_refined=bool(data.get("screen_refined", False)),
            local=None if local is None else LocalOptimizerResult.from_dict(local),
            wall_time=float(data.get("wall_time", 0.0)),
        )


@dataclass
class MultistartResult(_JsonDTO):
    """
    Итог мультистарта.

    Атрибуты:
        base: базовый OptimizationResult лучшего старта;
        solutions: все старты, включая неудачные (StartResult);
        best_index: индекс лучшего старта (-1, если нет);
        n_starts: общее число стартов;
        n_refined: число дошедших до полной доводки;
        n_failed: число неудачных стартов;
        seed: seed генератора начальных точек (или None);
        clustering: словарь результатов кластеризации решений
            (например {"n_clusters": int, "labels": [...]});
        wall_time: общее время, секунды.
    """

    base: OptimizationResult
    solutions: List[StartResult] = field(default_factory=list)
    best_index: int = -1
    n_starts: int = 0
    n_refined: int = 0
    n_failed: int = 0
    seed: Optional[int] = None
    clustering: Dict[str, Any] = field(default_factory=dict)
    wall_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base": self.base.to_dict(),
            "solutions": [s.to_dict() for s in self.solutions],
            "best_index": int(self.best_index),
            "n_starts": int(self.n_starts),
            "n_refined": int(self.n_refined),
            "n_failed": int(self.n_failed),
            "seed": self.seed,
            "clustering": _encode(self.clustering),
            "wall_time": float(self.wall_time),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MultistartResult":
        return cls(
            base=OptimizationResult.from_dict(data["base"]),
            solutions=[StartResult.from_dict(d) for d in data.get("solutions", [])],
            best_index=int(data.get("best_index", -1)),
            n_starts=int(data.get("n_starts", 0)),
            n_refined=int(data.get("n_refined", 0)),
            n_failed=int(data.get("n_failed", 0)),
            seed=data.get("seed"),
            clustering=dict(data.get("clustering", {})),
            wall_time=float(data.get("wall_time", 0.0)),
        )


@dataclass
class RunCheckpoint(_JsonDTO):
    """
    Чекпоинт прерванного запуска (мультистарт/гибрид).

    Атрибуты:
        seed: seed генератора начальных точек (или None);
        config: пользовательская конфигурация запуска (dict);
        completed: завершённые старты (индексы или записи);
        pending: незавершённые старты (индексы или записи);
        best_index: индекс лучшего завершённого старта (-1, если нет).
    """

    seed: Optional[int] = None
    config: Dict[str, Any] = field(default_factory=dict)
    completed: List[Any] = field(default_factory=list)
    pending: List[Any] = field(default_factory=list)
    best_index: int = -1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "config": _encode(self.config),
            "completed": _encode(self.completed),
            "pending": _encode(self.pending),
            "best_index": int(self.best_index),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RunCheckpoint":
        return cls(
            seed=data.get("seed"),
            config=dict(data.get("config", {})),
            completed=list(data.get("completed", [])),
            pending=list(data.get("pending", [])),
            best_index=int(data.get("best_index", -1)),
        )


@dataclass
class JacobianResult(_JsonDTO):
    """
    Якобиан невязок в точке φ (forward sensitivities).

    Атрибуты:
        J: матрица ∂r/∂φ (numpy, [n_residuals, size]);
        r: вектор невязок в точке;
        loss: скалярная потеря 0.5 * ||r||^2 (или целевая loss точки);
        phi: точка оценки;
        n_forward: число прямых прогонов, потребованное оценкой;
        wall_time: время оценки, секунды;
        method: способ построения ('forward' по умолчанию);
        likelihood_scale: множители строк для перевода невязок в единицы
            полного NLL (r_nll = s·r, J_nll = s[:, None]·J; см.
            objectives.likelihood_row_scale) или None — невязки уже в этих
            единицах. Используется compute_fim / laplace_approximation.
    """

    J: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    r: np.ndarray = field(default_factory=lambda: np.zeros(0))
    loss: float = 0.0
    phi: np.ndarray = field(default_factory=lambda: np.zeros(0))
    n_forward: int = 0
    wall_time: float = 0.0
    method: str = "forward"
    likelihood_scale: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.J = np.asarray(self.J, dtype=np.float64)
        self.r = np.asarray(self.r, dtype=np.float64)
        self.phi = np.asarray(self.phi, dtype=np.float64)
        if self.likelihood_scale is not None:
            self.likelihood_scale = np.asarray(
                self.likelihood_scale, dtype=np.float64
            ).reshape(-1)

    def likelihood_jacobian(self) -> np.ndarray:
        """J в единицах полного NLL: s[:, None]·J (или J, если s нет)."""
        if self.likelihood_scale is None:
            return self.J
        if self.likelihood_scale.shape[0] != self.J.shape[0]:
            raise ValueError(
                f"JacobianResult: likelihood_scale has "
                f"{self.likelihood_scale.shape[0]} rows, J has {self.J.shape[0]}."
            )
        return self.likelihood_scale[:, None] * self.J

    def to_dict(self) -> Dict[str, Any]:
        return {
            "J": self.J.tolist(),
            "r": self.r.tolist(),
            "loss": float(self.loss),
            "phi": self.phi.tolist(),
            "n_forward": int(self.n_forward),
            "wall_time": float(self.wall_time),
            "method": str(self.method),
            "likelihood_scale": (
                None
                if self.likelihood_scale is None
                else self.likelihood_scale.tolist()
            ),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "JacobianResult":
        scale = data.get("likelihood_scale")
        return cls(
            J=np.asarray(data.get("J", []), dtype=np.float64),
            r=np.asarray(data.get("r", []), dtype=np.float64),
            loss=float(data.get("loss", 0.0)),
            phi=np.asarray(data.get("phi", []), dtype=np.float64),
            n_forward=int(data.get("n_forward", 0)),
            wall_time=float(data.get("wall_time", 0.0)),
            method=str(data.get("method", "forward")),
            likelihood_scale=None
            if scale is None
            else np.asarray(scale, dtype=np.float64),
        )
