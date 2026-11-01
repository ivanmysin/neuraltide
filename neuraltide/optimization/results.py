"""
Форматы результатов калибровки (этап P0).

Evaluation — результат одной оценки задачи CalibrationProblem: скалярная
потеря, компоненты, невязки, градиент, счётчики прямых прогонов и
градиентов, wall time, dtype и шаг интегрирования.

OptimizationResult — единый формат результата оптимизации: лучшие φ и
физические θ, история, seed, конфигурация, версии пакетов и dtype.
Сериализация — JSON с преобразованием numpy -> list; сериализация модели
сознательно не входит в контракт (модель восстанавливается через
ParameterSpace.apply_to_model(theta_best)).
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Union

import numpy as np
import tensorflow as tf

import neuraltide

__all__ = ["Evaluation", "OptimizationResult"]


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
    Единый формат результата оптимизации (P0).

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
