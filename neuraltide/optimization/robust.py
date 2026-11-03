"""
Робастные и адаптивные многокомпонентные потери (этап P2).

Дорожная карта (docs/optimization_roadmap.md, раздел 7.3) требует поверх
нормированных невязок (objectives, этап P0) добавить:

- робастные виды штрафа ObservationSpec kind 'cauchy' / 'tukey' с масштабом
  scale c (см. neuraltide.optimization.objectives) — реализовано там же;
- стратегии адаптивных весов компонент (WeightStrategy): FixedWeights,
  UncertaintyWeighting, GradNormWeights, SelfAdaptiveWeights;
- проекторы конфликтующих градиентов: pcgrad_project (PCGrad, Yu et al.
  2020) и cagrad_project (CAGrad, Liu et al. 2021, в min-max постановке);
- balanced_gradient — один прямой прогон с per-component GradientTape;
- adaptive_window_sampler — выбор временного окна с большим остатком и
  построение focused view задачи с усиленной маской окна;
- balanced_train — драйвер обучения с логированием весов и обязательным
  baseline-сравнением с фиксированными весами;
- BalancedResult — DTO результата (base OptimizationResult, история весов,
  baseline, заметки) с JSON-сериализацией.

Честные ограничения
-------------------
- Все стратегии весов — ДЕТЕРМИНИРОВАННЫЕ суррогаты соответствующих методов:
  оригинальные Uncertainty Weighting / Self-Adaptive / GradNorm обучают
  веса (или log-sigma) градиентным спуском совместно с параметрами; здесь
  веса пересчитываются аналитически по наблюдённым losses/grads каждого
  шага. Фиксированные точки совпадают по постановке, динамика — нет.
- PCGrad/CAGrad и весь модуль работают ТОЛЬКО с BPTT-режимом
  CalibrationProblem (per-component градиенты через tf.GradientTape).
  Применение к adjoint-режиму (problem с grad_method == 'adjoint') или к
  training.Trainer поднимает NotImplementedError.
- CAGrad здесь — min-max постановка conflict-averse проекции: минимизируется
  максимальное направленное произведение max_i <g_i, d> (верхняя граница
  косинусного конфликта по Коши–Шварцу) в шаре ||d - g0|| <= c ||g0|| вокруг
  среднего градиента g0; гарантия прогресса <d, g0> >= (1-c) ||g0||^2.
  Точное решение малого QP — через SLSQP (scipy); при сходимости решателя
  возвращается g0 (документированный fallback).
- adaptive_window_sampler строит НОВЫЙ view задачи (with_objective); исходные
  ObservationSpec не мутируются. Один вызов = один прямой прогон
  (problem.simulate); phi != None переводит модель по общему контракту
  CalibrationProblem.
- balanced_train прогоняет обучение дважды (основной прогон + baseline с
  фиксированными весами из той же phi0) — честное сравнение дороже ровно в
  2 раза. Адаптивные веса не гарантируют улучшение над baseline; notes
  содержат прямое сравнение.
- eager-режим, batch=1 (контракт NetworkRNN); все numpy-выходы float64.

Зафиксированные зависимости от приватного API (не изменять без
синхронизации, прецедент — neuraltide.optimization.sensitivities):
  * neuraltide.optimization.problem: _chain_gradient (per-component
    chain-градиент dloss/dφ по переменным модели и x0-тензорам),
    _materialize_initial_state, а также методы CalibrationProblem
    ._require_objective(), ._apply_phi(), .pack() и ._counters;
  * neuraltide.optimization.objectives: ResidualLoss.components/.observations
    и приватные _error/_psi (только внутри adaptive_window_sampler).
"""

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf
from scipy import optimize as sp_optimize

import neuraltide.config
from neuraltide.core.types import cast_if_needed
from neuraltide.optimization.objectives import ObservationSpec, ResidualLoss
from neuraltide.optimization.problem import _chain_gradient, _materialize_initial_state
from neuraltide.optimization.results import OptimizationResult
from neuraltide.training.losses import _MASK_DENOM_EPS

__all__ = [
    "WeightStrategy",
    "FixedWeights",
    "UncertaintyWeighting",
    "GradNormWeights",
    "SelfAdaptiveWeights",
    "pcgrad_project",
    "cagrad_project",
    "balanced_gradient",
    "adaptive_window_sampler",
    "AdaptiveWindowFocus",
    "balanced_train",
    "BalancedResult",
]

PathLike = Union[str, os.PathLike]
GradMap = Union[Mapping[str, np.ndarray], Sequence[np.ndarray]]


# ---------------------------------------------------------------------------
# Внутренние helpers
# ---------------------------------------------------------------------------


def _grad_list(grads: GradMap) -> List[np.ndarray]:
    """Mapping/Sequence градиентов -> список плоских float64 numpy массивов."""
    values = list(grads.values()) if isinstance(grads, Mapping) else list(grads)
    if not values:
        raise ValueError("robust: at least one gradient is required.")
    out = [np.asarray(g, dtype=np.float64).reshape(-1) for g in values]
    shape = out[0].shape
    for g in out[1:]:
        if g.shape != shape:
            raise ValueError(
                f"robust: gradient shapes must match, got {shape} and {g.shape}."
            )
    for g in out:
        if not np.all(np.isfinite(g)):
            raise ValueError("robust: gradients must be finite.")
    return out


def _weights_vector(
    weights: Union[Mapping[str, float], Sequence[float], np.ndarray],
    names: Sequence[str],
) -> np.ndarray:
    """Веса компонент (Mapping по именам или вектор) -> float64 вектор."""
    if isinstance(weights, Mapping):
        missing = [n for n in names if n not in weights]
        if missing:
            raise KeyError(f"robust: weights missing components: {missing}.")
        vec = np.asarray([float(weights[n]) for n in names], dtype=np.float64)
    else:
        vec = np.asarray(weights, dtype=np.float64).reshape(-1)
    if vec.shape[0] != len(names):
        raise ValueError(
            f"robust: weights has {vec.shape[0]} entries, "
            f"expected {len(names)} ({list(names)})."
        )
    if not np.all(np.isfinite(vec)) or np.any(vec < 0.0):
        raise ValueError("robust: weights must be finite and non-negative.")
    return vec


def _json_safe(value: Any) -> Any:
    """Рекурсивное приведение к JSON-сериализуемым значениям."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _require_bptt_problem(problem: Any, projector: Any = None) -> None:
    """
    Guard: модуль работает только с BPTT CalibrationProblem.

    PCGrad/CAGrad (projector != None) и per-component градиенты требуют
    tf.GradientTape; adjoint-режим Trainer/задачи не поддерживается —
    поднимаем NotImplementedError до любых прогонов/мутаций.
    """
    what = "PCGrad/CAGrad и " if projector is not None else ""
    if type(problem).__module__.startswith("neuraltide.training"):
        raise NotImplementedError(
            f"robust: {what}balanced-gradient API работает с "
            "CalibrationProblem (BPTT через tf.GradientTape), а не с "
            f"training.{type(problem).__name__}. Для adjoint-режима Trainer "
            "per-component градиенты недоступны."
        )
    if getattr(problem, "grad_method", None) == "adjoint":
        raise NotImplementedError(
            f"robust: {what}per-component градиенты требуют BPTT; "
            "problem.grad_method == 'adjoint' не поддерживается. "
            "Используйте BPTT-задачу (CalibrationProblem) или projector=None."
        )
    required = (
        "parameter_space",
        "network",
        "t_sequence",
        "initial_state",
        "pack",
        "_apply_phi",
        "_counters",
        "_require_objective",
    )
    missing = [a for a in required if not hasattr(problem, a)]
    if missing:
        raise TypeError(
            "robust: ожидается CalibrationProblem (BPTT); у объекта "
            f"{type(problem).__name__} отсутствуют атрибуты {missing}."
        )


# ---------------------------------------------------------------------------
# Стратегии адаптивных весов
# ---------------------------------------------------------------------------


class WeightStrategy:
    """
    Базовый протокол стратегии весов компонент (детерминированный суррогат).

    Контракт:
        initial_weights(names) -> np.ndarray [n] — стартовые веса;
        update(losses, grads, step) -> np.ndarray [n] — веса после шага;
        state_dict()/load_state_dict() — JSON-сериализуемое состояние.

    Аргументы update:
        losses: {имя компоненты: float} — наблюдённые потери шага;
        grads: {имя: np.ndarray} per-component градиенты или None;
        step: номер шага (0-based).

    Реализации здесь — не обучаемые параметры, а аналитические пересчёты
    по наблюдённым значениям; см. честные ограничения в документации модуля.
    """

    def __init__(self) -> None:
        self._names: Optional[List[str]] = None
        self._step = 0

    # ------------------------------------------------------------------
    def initial_weights(self, names: Sequence[str]) -> np.ndarray:
        names = [str(n) for n in names]
        if not names:
            raise ValueError("WeightStrategy: 'names' must be non-empty.")
        self._names = names
        w = self._initial(names)
        return self._validate(w)

    def update(
        self,
        losses: Mapping[str, float],
        grads: Optional[Mapping[str, np.ndarray]],
        step: int,
    ) -> np.ndarray:
        if self._names is None:
            raise RuntimeError(
                "WeightStrategy: initial_weights() must be called before update()."
            )
        missing = [n for n in self._names if n not in losses]
        if missing:
            raise KeyError(f"WeightStrategy: losses missing components: {missing}.")
        w = self._update({n: float(losses[n]) for n in self._names}, grads, int(step))
        self._step = int(step) + 1
        return self._validate(w)

    # ------------------------------------------------------------------
    def _initial(self, names: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def _update(
        self,
        losses: Mapping[str, float],
        grads: Optional[Mapping[str, np.ndarray]],
        step: int,
    ) -> np.ndarray:
        raise NotImplementedError

    def _validate(self, w: np.ndarray) -> np.ndarray:
        w = np.asarray(w, dtype=np.float64).reshape(-1)
        if self._names is not None and w.shape[0] != len(self._names):
            raise ValueError(
                f"WeightStrategy: weights has {w.shape[0]} entries, expected "
                f"{len(self._names)}."
            )
        if not np.all(np.isfinite(w)) or np.any(w < 0.0):
            raise ValueError("WeightStrategy: weights must be finite, >= 0.")
        return w

    # ------------------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        """JSON-сериализуемое состояние (тип, шаг, веса, гиперпараметры)."""
        return {
            "type": type(self).__name__,
            "step": int(self._step),
            "names": list(self._names) if self._names is not None else None,
            "weights": (
                None
                if self._names is None
                else np.asarray(self._current(), dtype=np.float64).tolist()
            ),
            "params": _json_safe(self._params()),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Восстановление из state_dict(); тип должен совпадать."""
        stype = str(state.get("type", ""))
        if stype and stype != type(self).__name__:
            raise ValueError(
                f"WeightStrategy: state type '{stype}' != '{type(self).__name__}'."
            )
        self._step = int(state.get("step", 0))
        names = state.get("names")
        self._names = None if names is None else [str(n) for n in names]
        self._load_params(dict(state.get("params", {})))
        weights = state.get("weights")
        if weights is not None and self._names is not None:
            self._set_current(np.asarray(weights, dtype=np.float64))

    # Хуки подклассов.
    def _params(self) -> Dict[str, Any]:
        return {}

    def _load_params(self, params: Mapping[str, Any]) -> None:
        del params

    def _current(self) -> np.ndarray:
        raise NotImplementedError

    def _set_current(self, w: np.ndarray) -> None:
        del w
        raise NotImplementedError


class FixedWeights(WeightStrategy):
    """
    Фиксированные веса (baseline «нормировка + фиксированные веса»).

    Args:
        weights: float (все компоненты равны), Mapping {имя: вес} или
            вектор, выровненный по names initial_weights().
    """

    def __init__(self, weights: Union[float, Mapping[str, float], Sequence[float]]):
        super().__init__()
        if isinstance(weights, Mapping):
            self._fixed_map = {str(k): float(v) for k, v in weights.items()}
            self._fixed_vec: Optional[np.ndarray] = None
        else:
            arr = np.asarray(weights, dtype=np.float64).reshape(-1)
            if arr.shape[0] == 1:
                self._fixed_vec = arr
            else:
                self._fixed_vec = arr
            self._fixed_map = None

    def _initial(self, names: Sequence[str]) -> np.ndarray:
        if self._fixed_map is not None:
            missing = [n for n in names if n not in self._fixed_map]
            if missing:
                raise KeyError(f"FixedWeights: missing components: {missing}.")
            return np.asarray([self._fixed_map[n] for n in names])
        vec = np.asarray(self._fixed_vec, dtype=np.float64)
        if vec.shape[0] == 1:
            return np.full(len(names), vec[0])
        if vec.shape[0] != len(names):
            raise ValueError(
                f"FixedWeights: vector has {vec.shape[0]} entries, "
                f"expected {len(names)}."
            )
        return vec

    def _update(self, losses, grads, step) -> np.ndarray:
        del losses, grads, step
        return self._current()

    def _params(self) -> Dict[str, Any]:
        return {
            "fixed_map": self._fixed_map,
            "fixed_vec": (
                None if self._fixed_vec is None else self._fixed_vec.tolist()
            ),
        }

    def _load_params(self, params: Mapping[str, Any]) -> None:
        fm = params.get("fixed_map")
        self._fixed_map = (
            None if fm is None else {str(k): float(v) for k, v in fm.items()}
        )
        fv = params.get("fixed_vec")
        self._fixed_vec = None if fv is None else np.asarray(fv, dtype=np.float64)

    def _current(self) -> np.ndarray:
        if self._names is None:
            raise RuntimeError("FixedWeights: initial_weights() not called.")
        return self._initial(self._names)

    def _set_current(self, w: np.ndarray) -> None:
        del w  # фиксированные веса не меняются состоянием


class _LossPoweredWeights(WeightStrategy):
    """
    Общая основа суррогатов вида w_i ∝ (loss_i + eps)^(-exponent).

    Mean-нормировка (средний вес = 1) сохраняет масштаб суммарной потери.
    """

    def __init__(
        self,
        exponent: float,
        eps: float = 1e-8,
        min_weight: float = 1e-3,
        max_weight: float = 1e3,
    ):
        super().__init__()
        if eps <= 0.0:
            raise ValueError(f"{type(self).__name__}: eps must be positive.")
        if min_weight <= 0.0 or max_weight < min_weight:
            raise ValueError(
                f"{type(self).__name__}: require 0 < min <= max for weights."
            )
        self.exponent = float(exponent)
        self.eps = float(eps)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self._w: Optional[np.ndarray] = None

    def _initial(self, names: Sequence[str]) -> np.ndarray:
        self._w = np.ones(len(names), dtype=np.float64)
        return self._w

    def _update(self, losses, grads, step) -> np.ndarray:
        del grads, step
        losses_v = np.asarray([losses[n] for n in self._names], dtype=np.float64)
        w = (losses_v + self.eps) ** (-self.exponent)
        w_mean = np.mean(w)
        if not np.isfinite(w_mean) or w_mean <= 0.0:
            w = np.ones_like(w)
        else:
            w = w / w_mean
        w = np.clip(w, self.min_weight, self.max_weight)
        self._w = w
        return w

    def _params(self) -> Dict[str, Any]:
        return {
            "exponent": self.exponent,
            "eps": self.eps,
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
        }

    def _load_params(self, params: Mapping[str, Any]) -> None:
        for key in ("exponent", "eps", "min_weight", "max_weight"):
            if key in params:
                setattr(self, key, float(params[key]))

    def _current(self) -> np.ndarray:
        if self._w is None:
            raise RuntimeError(f"{type(self).__name__}: initial_weights() not called.")
        return self._w

    def _set_current(self, w: np.ndarray) -> None:
        self._w = np.asarray(w, dtype=np.float64)


class UncertaintyWeighting(_LossPoweredWeights):
    """
    Детерминированный суррогат Uncertainty Weighting (Kendall et al., 2018).

    w_i ∝ 1 / sqrt(loss_i + eps), mean-нормировано. Компонента с большим
    наблюдённым шумом (потерей) получает меньший вес.

    Ограничение: оригинал обучает log-sigma совместно с моделью через
    -log p(y|x) = 0.5 w L + log sigma; здесь w пересчитывается аналитически
    из наблюдённых losses каждого шага (гиперпараметр, не переменная модели).
    """

    def __init__(
        self,
        eps: float = 1e-8,
        min_weight: float = 1e-3,
        max_weight: float = 1e3,
    ):
        super().__init__(
            exponent=0.5, eps=eps, min_weight=min_weight, max_weight=max_weight
        )


class SelfAdaptiveWeights(_LossPoweredWeights):
    """
    Детерминированный суррогат self-adaptive loss weights.

    Классический вид (Duan et al., 2022 и след.): w_i = 1 / (2 sigma_i^2)
    с обучаемым sigma — приравнивание к наблюдённой потере даёт
    w_i ∝ 1 / loss_i; здесь w_i ∝ (loss_i + eps)^(-exponent),
    по умолчанию exponent=1.

    Ограничение: как и UncertaintyWeighting — аналитический пересчёт вместо
    совместного обучения; exponent=0.5 формально совпадает с
    UncertaintyWeighting.
    """

    def __init__(
        self,
        exponent: float = 1.0,
        eps: float = 1e-8,
        min_weight: float = 1e-3,
        max_weight: float = 1e3,
    ):
        super().__init__(
            exponent=exponent, eps=eps, min_weight=min_weight, max_weight=max_weight
        )


class GradNormWeights(WeightStrategy):
    """
    Упрощённая детерминированная форма GradNorm (Chen et al., 2018).

    Целевые нормы G̃_i = mean_j(n_j) * (L_i / mean_j(L_j))**alpha, где
    n_i = ||g_i|| — норма per-component градиента. Мультипликативная
    коррекция w_i *= (G̃_i / (n_i + eps))**step_size; при step_size=1 взвешенные
    нормы w_i n_i выравниваются к целевым за один шаг (локально, пока n_i
    зависит от w_i почти линейно). После коррекции веса mean-нормируются
    (средний вес = 1) и клипуются в [min_weight, max_weight].

    Ограничения: требует per-component градиенты (grads != None и полные по
    именам); при их отсутствии веса не меняются. Оригинальный GradNorm
    обновляет веса градиентным спуском по |n_i - G̃_i|; здесь — прямая
    мультипликативная коррекция к тому же неподвижному состоянию.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        step_size: float = 1.0,
        eps: float = 1e-8,
        min_weight: float = 1e-3,
        max_weight: float = 1e3,
    ):
        super().__init__()
        if eps <= 0.0:
            raise ValueError("GradNormWeights: eps must be positive.")
        if step_size <= 0.0:
            raise ValueError("GradNormWeights: step_size must be positive.")
        if min_weight <= 0.0 or max_weight < min_weight:
            raise ValueError("GradNormWeights: require 0 < min <= max.")
        self.alpha = float(alpha)
        self.step_size = float(step_size)
        self.eps = float(eps)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self._w: Optional[np.ndarray] = None

    def _initial(self, names: Sequence[str]) -> np.ndarray:
        self._w = np.ones(len(names), dtype=np.float64)
        return self._w

    def _update(self, losses, grads, step) -> np.ndarray:
        del step
        w = self._current().copy()
        if grads is None:
            return w
        norms = np.zeros(len(self._names), dtype=np.float64)
        for i, n in enumerate(self._names):
            g = grads.get(n)
            if g is None:
                return w  # неполные градиенты — веса не меняем
            norms[i] = float(np.linalg.norm(np.asarray(g, dtype=np.float64)))
        losses_v = np.asarray([losses[n] for n in self._names], dtype=np.float64)
        l_mean = np.mean(losses_v)
        n_mean = np.mean(norms)
        target = n_mean * ((losses_v + self.eps) / (l_mean + self.eps)) ** self.alpha
        ratio = target / (norms + self.eps)
        w = w * ratio**self.step_size
        w_mean = np.mean(w)
        if not np.isfinite(w_mean) or w_mean <= 0.0:
            return self._current()
        w = np.clip(w / w_mean, self.min_weight, self.max_weight)
        self._w = w
        return w

    def _params(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "step_size": self.step_size,
            "eps": self.eps,
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
        }

    def _load_params(self, params: Mapping[str, Any]) -> None:
        for key in ("alpha", "step_size", "eps", "min_weight", "max_weight"):
            if key in params:
                setattr(self, key, float(params[key]))

    def _current(self) -> np.ndarray:
        if self._w is None:
            raise RuntimeError("GradNormWeights: initial_weights() not called.")
        return self._w

    def _set_current(self, w: np.ndarray) -> None:
        self._w = np.asarray(w, dtype=np.float64)


# ---------------------------------------------------------------------------
# Проекторы конфликтующих градиентов
# ---------------------------------------------------------------------------


def pcgrad_project(grads: GradMap) -> np.ndarray:
    """
    PCGrad (Yu et al., 2020): проекция каждого градиента на нормальное
    пространство каждого конфликтующего (g_i·g_j < 0) градиента, усреднение.

    Для каждой пары i≠j с g_i·g_j < 0:
        g_i := g_i - (g_i·g_j) / ||g_j||^2 * g_j.
    Возвращает mean_i g_i (numpy float64, та же форма, что входы).

    Args:
        grads: Mapping {имя: grad} или Sequence градиентов (одной формы).

    Границы: стоимость O(n^2) пар; градиенты обязаны быть конечными.
    Только для BPTT-режима (через balanced_gradient/balanced_train).
    """
    gs = _grad_list(grads)
    n = len(gs)
    shapes = gs[0].shape
    projected = [g.copy() for g in gs]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dot = float(np.dot(gs[i], gs[j]))
            if dot < 0.0:
                norm_sq = float(np.dot(gs[j], gs[j]))
                if norm_sq > 0.0:
                    projected[i] = projected[i] - (dot / norm_sq) * gs[j]
    out = np.mean(projected, axis=0)
    return out.reshape(shapes)


def cagrad_project(grads: GradMap, c: float = 0.5) -> np.ndarray:
    """
    CAGrad-проекция (conflict-averse) в min-max постановке.

    g0 = mean_i g_i. Решается малое QP:
        min_d  max_i <g_i, d>
        s.t.   ||d - g0|| <= c ||g0||.
    Гарантия прогресса: <d, g0> >= (1-c) ||g0||^2 (из ограничения шара);
    c=0 возвращает d=g0 точно; c ближе к 1 — больше свободы ухода от g0
    ради уменьшения max_i <g_i, d> (конфликта по Коши–Шварцу:
    cos(g_i, d) <= <g_i, d> / (||g_i|| ||d||)).

    Решение — точное (SLSQP, scipy, детерминированно); при неудаче решателя
    возвращается g0 с документированным fallback.

    Args:
        grads: Mapping {имя: grad} или Sequence градиентов (одной формы).
        c: радиус шара в долях ||g0||, 0 <= c < 1.

    Только для BPTT-режима (через balanced_gradient/balanced_train).
    """
    if not 0.0 <= c < 1.0:
        raise ValueError(f"cagrad_project: c must be in [0, 1), got {c}.")
    gs = _grad_list(grads)
    shapes = gs[0].shape
    g0 = np.mean(gs, axis=0)
    g0_norm = float(np.linalg.norm(g0))
    if g0_norm == 0.0:
        return np.zeros(shapes)
    if c == 0.0:
        return g0.reshape(shapes)

    G = np.stack(gs, axis=0)  # [n, dim]
    n = G.shape[0]
    radius = c * g0_norm

    def _objective(x: np.ndarray) -> float:
        return float(x[-1])

    # t >= <g_i, d>  <=>  <g_i, d> - t <= 0 (линейные ограничения).
    a_mat = np.hstack([G, -np.ones((n, 1))])
    linear = sp_optimize.LinearConstraint(a_mat, -np.inf, 0.0)
    # ||d - g0||^2 <= radius^2.
    ball = sp_optimize.NonlinearConstraint(
        lambda x: float(np.sum((x[:-1] - g0) ** 2)),
        -np.inf,
        radius**2,
    )
    t0 = float(np.max(G @ g0))
    x0 = np.concatenate([g0, [t0]])
    res = sp_optimize.minimize(
        _objective,
        x0,
        method="SLSQP",
        constraints=[linear, ball],
        tol=1e-12,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not res.success or not np.all(np.isfinite(res.x)):
        return g0.reshape(shapes)
    d = res.x[:-1]
    return d.reshape(shapes)


# ---------------------------------------------------------------------------
# Balanced gradient: один прогон, per-component градиенты
# ---------------------------------------------------------------------------


@dataclass
class _GradientBreakdown:
    """Внутренний результат _balanced_gradient_full."""

    losses: Dict[str, float]
    grads: Dict[str, np.ndarray]
    weighted_grad: np.ndarray
    names: List[str]
    weights: np.ndarray
    weighted_total: float


def _balanced_gradient_full(
    problem: Any,
    phi: Any,
    weights: Union[Mapping[str, float], Sequence[float], np.ndarray],
    projector: Any = None,
) -> _GradientBreakdown:
    """
    Один прямой прогон: per-component losses и градиенты, взвешенный суммарный
    градиент (или проекция projector по per-component градиентам).

    Использует приватные хелперы problem.py (прецедент sensitivities):
    _materialize_initial_state и _chain_gradient (model_var-записи через
    переменные × jacobian, tensor-backed x0 — через подставленные тензоры).
    """
    _require_bptt_problem(problem, projector)
    problem._require_objective()
    components_fn = getattr(problem.objective, "components", None)
    if not callable(components_fn):
        raise TypeError(
            "balanced_gradient: objective "
            f"'{type(problem.objective).__name__}' не предоставляет "
            "components(predictions, network). Нужен компонентный objective "
            "(например ResidualLoss)."
        )
    dtype = neuraltide.config.get_dtype()
    t0 = time.perf_counter()
    phi_t = problem.pack(phi)
    if phi is not None:
        problem._apply_phi(phi_t)
    space = problem.parameter_space
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(phi_t)
        theta = space.unpack(phi_t)
        initial_state, x0_map = _materialize_initial_state(
            problem.network, space, theta, problem.initial_state
        )
        output = problem.network(
            problem.t_sequence,
            inputs=problem.inputs,
            initial_state=initial_state,
        )
        comps = components_fn(output, problem.network)
    if not comps:
        del tape
        raise TypeError(
            "balanced_gradient: objective.components() вернул пустой "
            "словарь — нечего балансировать."
        )
    names = [str(k) for k in comps.keys()]
    w = _weights_vector(weights, names)

    losses: Dict[str, float] = {}
    grads: Dict[str, np.ndarray] = {}
    combined = np.zeros(problem.size, dtype=np.float64)
    for name in names:
        loss_i = cast_if_needed(tf.convert_to_tensor(comps[name]), dtype)
        g_t = _chain_gradient(space, tape, loss_i, phi_t, x0_map)
        g = np.asarray(g_t, dtype=np.float64).reshape(-1)
        grads[name] = g
        losses[name] = float(loss_i)
        combined += w[names.index(name)] * g
    del tape
    if projector is not None:
        combined = np.asarray(projector(grads), dtype=np.float64).reshape(-1)
    problem._counters["n_forward"] += 1
    problem._counters["n_gradient"] += 1
    problem._counters["wall_time"] += time.perf_counter() - t0
    weighted_total = float(np.dot(w, np.asarray([losses[n] for n in names])))
    return _GradientBreakdown(
        losses=losses,
        grads=grads,
        weighted_grad=combined,
        names=names,
        weights=w,
        weighted_total=weighted_total,
    )


def balanced_gradient(
    problem: Any,
    phi: Any,
    weights: Union[Mapping[str, float], Sequence[float], np.ndarray],
    projector: Any = None,
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Один прямой прогон сети с per-component GradientTape.

    Args:
        problem: CalibrationProblem (BPTT; adjoint/Trainer — NotImplementedError).
        phi: вектор φ (None — текущие значения модели).
        weights: веса компонент {имя: вес} или вектор по порядку components().
        projector: опционально callable(grads_map) -> объединённый градиент
            (например pcgrad_project / cagrad_project); при None — взвешенная
            сумма Σ w_i g_i.

    Returns:
        (losses, weighted_grad): dict {имя: float} и numpy float64 [size].

    Счётчики: +1 n_forward, +1 n_gradient, wall_time дополняется.
    """
    full = _balanced_gradient_full(problem, phi, weights, projector)
    return full.losses, full.weighted_grad


# ---------------------------------------------------------------------------
# Adaptive window sampler
# ---------------------------------------------------------------------------


class AdaptiveWindowFocus:
    """
    Выбор временного окна с большим остатком и focused view задачи.

    Один вызов select()/__call__() = один прямой прогон (problem.simulate;
    phi != None переводит модель по общему контракту CalibrationProblem).
    Энергия окна w — сумма по наблюдениям masked-mean остатка psi в окне:

        E_w = Σ_spec sum_{t∈w}(m psi) / max(sum_{t∈w} m, eps).

    focus='large_residual': выбирается окно с максимальной энергией.
    __call__() возвращает НОВЫЙ view задачи (with_objective) с маской
    spec.mask * multiplier, где multiplier = boost внутри окна и 1 вне его
    (masked-mean нормировка подстраивается автоматически). Исходные
    ObservationSpec НЕ мутируются — view можно безопасно выбрасывать.
    """

    _FOCI = ("large_residual",)

    def __init__(
        self,
        problem: Any,
        window_size: int,
        focus: str = "large_residual",
        boost: float = 3.0,
    ):
        _require_bptt_problem(problem)
        problem._require_objective()
        if focus not in self._FOCI:
            raise ValueError(
                f"adaptive_window_sampler: unknown focus '{focus}'. "
                f"Available: {list(self._FOCI)}."
            )
        window_size = int(window_size)
        if window_size < 1:
            raise ValueError(
                f"adaptive_window_sampler: window_size must be >= 1, got {window_size}."
            )
        if boost < 1.0:
            raise ValueError(
                f"adaptive_window_sampler: boost must be >= 1, got {boost}."
            )
        observations = getattr(problem.objective, "observations", None)
        if not observations:
            raise TypeError(
                "adaptive_window_sampler: objective "
                f"'{type(problem.objective).__name__}' не предоставляет "
                "список observations (нужен ResidualLoss)."
            )
        self._problem = problem
        self.window_size = window_size
        self.focus = focus
        self.boost = float(boost)
        self.last_window: Optional[Tuple[int, int]] = None
        self.last_energies: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    @property
    def problem(self) -> Any:
        return self._problem

    @property
    def n_windows(self) -> int:
        T = int(self._problem.t_sequence.shape[1])
        return len(range(0, T, self.window_size))

    def window_bounds(self) -> List[Tuple[int, int]]:
        T = int(self._problem.t_sequence.shape[1])
        return [
            (s, min(s + self.window_size, T)) for s in range(0, T, self.window_size)
        ]

    def window_multiplier(self, start: int, stop: int) -> tf.Tensor:
        """[1, T, 1] множитель маски: boost в [start, stop), 1 вне."""
        T = int(self._problem.t_sequence.shape[1])
        dtype = neuraltide.config.get_dtype()
        idx = tf.range(T, dtype=dtype)
        mult = tf.where(
            (idx >= float(start)) & (idx < float(stop)),
            tf.constant(self.boost, dtype=dtype),
            tf.constant(1.0, dtype=dtype),
        )
        return mult[None, :, None]

    # ------------------------------------------------------------------
    def compute_energies(self, phi: Any = None) -> np.ndarray:
        """Одна симуляция -> пер-оконные энергии остатка (numpy [n_windows])."""
        objective = self._problem.objective
        output = self._problem.simulate(phi)  # phi != None мутирует модель
        dtype = neuraltide.config.get_dtype()
        eps_const = tf.constant(_MASK_DENOM_EPS, dtype=dtype)
        bounds = self.window_bounds()
        energies = np.zeros(len(bounds), dtype=np.float64)
        for spec in objective.observations:
            rates = output.firing_rates[spec.population]
            e = objective._error(spec, rates, spec.target)
            psi = objective._psi(spec, e)
            if spec.mask is None:
                m = tf.ones_like(psi)
            else:
                m = cast_if_needed(spec.mask, psi.dtype)
            mb = m * tf.ones_like(psi)
            per_t = tf.reduce_sum(m * psi, axis=[0, 2])  # [T]
            mask_t = tf.reduce_sum(mb, axis=[0, 2])  # [T]
            per_t = tf.reshape(per_t, [-1])
            mask_t = tf.reshape(mask_t, [-1])
            for w_idx, (s, stop) in enumerate(bounds):
                num = tf.reduce_sum(per_t[s:stop])
                den = tf.maximum(tf.reduce_sum(mask_t[s:stop]), eps_const)
                energies[w_idx] += float(num / den)
        self.last_energies = energies
        return energies

    def select(self, phi: Any = None) -> Tuple[int, int]:
        """Выбрать окно с максимальной энергией; вернуть (start, stop)."""
        energies = self.compute_energies(phi)
        w_idx = int(np.argmax(energies))
        self.last_window = self.window_bounds()[w_idx]
        return self.last_window

    def __call__(self, phi: Any = None) -> Any:
        """
        Focused view задачи: маски усилены в выбранном окне (boost-кратно).

        Исходные ObservationSpec не трогаются; возвращается view
        (with_objective) с общими счётчиками задачи. Выбранное окно и
        энергии доступны как last_window / last_energies.
        """
        start, stop = self.select(phi)
        objective = self._problem.objective
        mult = self.window_multiplier(start, stop)
        new_specs: List[ObservationSpec] = []
        for spec in objective.observations:
            base = spec.mask
            new_mask = mult if base is None else base * mult
            new_specs.append(
                ObservationSpec(
                    name=spec.name,
                    population=spec.population,
                    target=spec.target,
                    sigma=spec.sigma,
                    mask=new_mask,
                    weight=spec.weight,
                    kind=spec.kind,
                    readout=spec.readout,
                    eps=spec.eps,
                    delta=spec.delta,
                    scale=spec.scale,
                )
            )
        return self._problem.with_objective(ResidualLoss(new_specs))


def adaptive_window_sampler(
    problem: Any,
    window_size: int,
    focus: str = "large_residual",
    boost: float = 3.0,
) -> AdaptiveWindowFocus:
    """
    Фабрика AdaptiveWindowFocus (см. класс): окно с большим остатком ->
    focused view задачи с усиленной маской окна, безопасно и явно.
    """
    return AdaptiveWindowFocus(problem, window_size, focus=focus, boost=boost)


# ---------------------------------------------------------------------------
# BalancedResult DTO
# ---------------------------------------------------------------------------


@dataclass
class BalancedResult:
    """
    Результат balanced_train.

    Атрибуты:
        base: базовый OptimizationResult (phi_best/theta_best/loss_best/
            история шагов/seed/config);
        weight_history: записи по шагам; каждая содержит 'step', 'loss'
            (взвешенная сумма), 'grad_norm' и ключи 'w:<имя>' / 'l:<имя>' —
            веса и потери компонент (все веса логируются);
        notes: честные заметки — описание стратегии/проектора, baseline-
            сравнение и ограничения;
        baseline: итог baseline-прогона с фиксированными весами
            {'method', 'best_loss', 'final_loss', 'history_len', 'weights'}.
    """

    base: OptimizationResult
    weight_history: List[Dict[str, Any]] = field(default_factory=list)
    notes: str = ""
    baseline: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Словарь с JSON-сериализуемыми значениями."""
        return {
            "base": self.base.to_dict(),
            "weight_history": _json_safe(self.weight_history),
            "notes": str(self.notes),
            "baseline": _json_safe(self.baseline),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BalancedResult":
        return cls(
            base=OptimizationResult.from_dict(data["base"]),
            weight_history=[dict(h) for h in data.get("weight_history", [])],
            notes=str(data.get("notes", "")),
            baseline=dict(data.get("baseline", {})),
        )

    def save(self, path: PathLike) -> None:
        """Сохраняет результат в JSON-файл."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: PathLike) -> "BalancedResult":
        """Загружает результат из JSON-файла, записанного save()."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


# ---------------------------------------------------------------------------
# balanced_train
# ---------------------------------------------------------------------------


def _resolve_optimizer(optimizer: Any) -> Any:
    """None/'sgd'/'adam'/Keras optimizer -> Keras optimizer с apply_gradients."""
    if optimizer is None:
        return tf.keras.optimizers.SGD(learning_rate=0.05)
    if isinstance(optimizer, str):
        name = optimizer.lower()
        if name == "sgd":
            return tf.keras.optimizers.SGD(learning_rate=0.05)
        if name == "adam":
            return tf.keras.optimizers.Adam(learning_rate=0.01)
        raise ValueError(
            f"balanced_train: unknown optimizer '{optimizer}'. "
            "Use None, 'sgd', 'adam' or a Keras optimizer instance."
        )
    if callable(getattr(optimizer, "apply_gradients", None)):
        return optimizer
    raise TypeError(
        "balanced_train: optimizer must be None, 'sgd', 'adam' or a Keras "
        f"optimizer with apply_gradients, got {type(optimizer).__name__}."
    )


def _clone_optimizer(optimizer: Any) -> Any:
    """Свежая копия Keras-оптимизатора для baseline-прогона."""
    try:
        return type(optimizer).from_config(optimizer.get_config())
    except Exception:
        return tf.keras.optimizers.SGD(learning_rate=0.05)


def _loop(
    problem: Any,
    phi0: Any,
    names: Sequence[str],
    weights: np.ndarray,
    optimizer: Any,
    max_steps: int,
    projector: Any,
    strategy: Optional[WeightStrategy],
    history_path: Optional[PathLike],
) -> Tuple[List[Dict[str, Any]], np.ndarray, float, np.ndarray]:
    """
    Основной цикл: по шагу на итерацию — один balanced-gradient прогон,
    применение объединённого градиента, (опционально) update стратегии,
    логирование. Возвращает (history, phi_best, best_loss, final_weights).
    """
    dtype = neuraltide.config.get_dtype()
    phi_var = tf.Variable(problem.pack(phi0), dtype=dtype, trainable=True)
    history: List[Dict[str, Any]] = []
    w = np.asarray(weights, dtype=np.float64)
    best_loss = float("inf")
    best_phi = np.asarray(phi_var, dtype=np.float64).reshape(-1)
    grad_norm = 0.0
    for step in range(int(max_steps)):
        full = _balanced_gradient_full(problem, phi_var, w, projector)
        if full.weighted_total < best_loss:
            best_loss = full.weighted_total
            best_phi = phi_var.numpy().copy().reshape(-1)
        grad_t = tf.convert_to_tensor(
            np.ascontiguousarray(full.weighted_grad), dtype=dtype
        )
        grad_norm = float(np.linalg.norm(full.weighted_grad))
        optimizer.apply_gradients([(grad_t, phi_var)])
        if strategy is not None:
            w = np.asarray(strategy.update(full.losses, full.grads, step))
        entry: Dict[str, Any] = {
            "step": int(step),
            "loss": float(full.weighted_total),
            "grad_norm": grad_norm,
        }
        for i, n in enumerate(names):
            entry[f"w:{n}"] = float(w[i])
            entry[f"l:{n}"] = float(full.losses[n])
        history.append(entry)
        if history_path is not None:
            with open(history_path, "w") as f:
                json.dump(_json_safe(history), f, indent=2)
    return history, best_phi, best_loss, w


def balanced_train(
    problem: Any,
    phi0: Any,
    strategy: WeightStrategy,
    *,
    optimizer: Any = None,
    max_steps: int,
    projector: Any = None,
    seed: Optional[int] = None,
    history_path: Optional[PathLike] = None,
) -> BalancedResult:
    """
    Обучение с адаптивным балансом компонент и обязательным baseline.

    Каждый шаг: один прогон _balanced_gradient_full (per-component losses и
    градиенты), объединённый градиент (взвешенная сумма или projector), шаг
    Keras-оптимизатора, update стратегии, логирование losses/весов.

    После основного прогона из той же phi0 с тем же числом шагов прогоняется
    baseline со стратегией FixedWeights(1.0) — сравнение включается в notes
    и baseline; адаптивные веса не гарантируют победу над baseline.

    Args:
        problem: CalibrationProblem (BPTT; adjoint/Trainer —
            NotImplementedError через _require_bptt_problem).
        phi0: стартовый вектор φ (обязателен).
        strategy: WeightStrategy (напр. UncertaintyWeighting()).
        optimizer: None (SGD lr=0.05), 'sgd', 'adam' или Keras optimizer.
        max_steps: число шагов обучения (>= 1).
        projector: опционально pcgrad_project / cagrad_project /
            callable(grads_map) -> grad (только BPTT).
        seed: записывается в результат; текущие стратегии детерминированы.
        history_path: опциональный JSON-файл; история перезаписывается
            после каждого шага (файл всегда актуален).

    Returns:
        BalancedResult(base, weight_history, notes, baseline).

    Счётчики задачи пополняются каждым прогоном (основной + baseline).
    """
    _require_bptt_problem(problem, projector)
    problem._require_objective()
    if not isinstance(strategy, WeightStrategy):
        raise TypeError(
            "balanced_train: strategy must be a WeightStrategy, got "
            f"{type(strategy).__name__}."
        )
    max_steps = int(max_steps)
    if max_steps < 1:
        raise ValueError(f"balanced_train: max_steps must be >= 1, got {max_steps}.")
    opt = _resolve_optimizer(optimizer)

    probe = problem.evaluate(phi0)
    if not probe.components:
        raise TypeError(
            "balanced_train: objective "
            f"'{type(problem.objective).__name__}' не предоставляет "
            "компоненты (components) — нечего балансировать; используйте "
            "ResidualLoss или другой компонентный objective."
        )
    names = [str(k) for k in probe.components.keys()]
    w0 = strategy.initial_weights(names)

    history, phi_best, best_loss, w_final = _loop(
        problem,
        phi0,
        names,
        w0,
        opt,
        max_steps,
        projector,
        strategy,
        history_path,
    )

    # Baseline: фиксированные единичные веса, та же phi0/число шагов.
    base_opt = _clone_optimizer(opt)
    base_strategy = FixedWeights(np.ones(len(names)))
    base_w0 = base_strategy.initial_weights(names)
    base_history, _, base_best, _ = _loop(
        problem,
        phi0,
        names,
        base_w0,
        base_opt,
        max_steps,
        projector,
        None,
        None,
    )
    baseline = {
        "method": "FixedWeights(1.0)",
        "best_loss": float(base_best),
        "final_loss": float(base_history[-1]["loss"]),
        "history_len": len(base_history),
        "weights": {n: 1.0 for n in names},
    }

    theta = problem.parameter_space.unpack(
        tf.constant(phi_best, dtype=neuraltide.config.get_dtype())
    )
    theta_best = {n: np.asarray(v) for n, v in theta.items()}
    projector_name = (
        None
        if projector is None
        else getattr(projector, "__name__", type(projector).__name__)
    )
    config = {
        "max_steps": int(max_steps),
        "optimizer": type(opt).__name__,
        "projector": projector_name,
        "strategy": type(strategy).__name__,
        "component_names": list(names),
        "initial_weights": {n: float(w0[i]) for i, n in enumerate(names)},
        "final_weights": {n: float(w_final[i]) for i, n in enumerate(names)},
        "strategy_state": strategy.state_dict(),
        "baseline_best_loss": float(base_best),
    }
    base = OptimizationResult(
        phi_best=np.asarray(phi_best, dtype=np.float64),
        theta_best=theta_best,
        loss_best=float(best_loss),
        history=history,
        seed=seed,
        config=config,
        dtype=str(neuraltide.config.get_dtype().name),
    )
    notes = (
        f"Стратегия: {type(strategy).__name__} — детерминированный суррогат "
        "(веса пересчитываются из наблюдённых losses/grads, а не обучаются "
        "совместно с моделью; см. neuraltide.optimization.robust). "
        f"Проектор: {projector_name or 'нет (взвешенная сумма)'}"
        + (
            " — PCGrad/CAGrad применяются к per-component градиентам BPTT; "
            "adjoint-режим не поддерживается."
            if projector is not None
            else "."
        )
        + f" Веса логируются в weight_history ('w:<имя>'). "
        f"Baseline {baseline['method']}: best_loss={baseline['best_loss']:.6g} "
        f"vs balanced best_loss={best_loss:.6g}. "
        "Адаптивные веса не гарантируют улучшение над фиксированными; "
        "сравнивайте best_loss при одинаковом phi0 и числе шагов."
    )
    return BalancedResult(
        base=base,
        weight_history=history,
        notes=notes,
        baseline=baseline,
    )
