"""
Ограничения на прогнозы и метод допустимой области (этап P2).

Поверх CalibrationProblem (этапы P0/P1) реализованы:

- ConstraintSpec — декларативное ограничение на прогнозы сети вида
  c(x) = 0 ('eq') или g(x) <= 0 ('ineq'): fn(predictions, network) ->
  Tensor; jacobian: None | 'fd' | callable(phi) -> [m, n] (нужен только
  для duck-typed задач без единой GradientTape);
- ConstrainedProblem — duck-typed обёртка над CalibrationProblem (или
  duck-аналогом): size / pack / unpack / bounds_phi / evaluate /
  value_and_gradient / residual / validate + augmented_terms(phi) ->
  {name: (value, violation)}. Потеря данных и AL-член вычисляются в
  ОДНОМ прогоне/ленте: для CalibrationProblem повторяется схема
  sensitivities.py через приватные _forward / _materialize_initial_state /
  _phi_gradient; для прочих задач data-часть берётся из
  problem.value_and_gradient, а ограничения — из problem.simulate(phi)
  с якобианом 'fd' или callable (один прогон на значение ограничения,
  без лишних data-прогонов);
- AugmentedLagrangianState — множители λ (поимённо, в форме значений
  ограничений), штраф μ и стандартное правило обновления (Nocedal &
  Wright, гл. 17): eq: λ <- λ − μ·c; ineq: λ <- max(0, λ + μ·g);
  μ <- mu_factor·μ, если нарушение не сократилось в eta раз;
- augmented_lagrangian — внешние итерации AL поверх локальных адаптеров
  local.py ('lbfgsb' минимизирует скалярную AL-потерю с точным
  градиентом; 'least_squares' — data-невязки плюс блоки ограничений,
  см. оговорку в docstring) или пользовательского callable;
  checkpoint JSON и resume;
- epsilon_constraint — превращение выбранного компонента objective в
  ограничение <= eps; целевая функция — взвешенная сумма остальных
  компонентов или явный objective;
- ALResult — DTO результата (base=OptimizationResult, state,
  violation_history, n_outer, validation) с сериализацией;
- fd_jacobian — центральные конечные разности с фиксированным или
  сканируемым шагом h (бюджет оценок см. в docstring).

Оговорка для local='least_squares': AL-цель содержит линейный по
ограничениям член -λ·c, который не представим как сумма квадратов.
В residual добавляются блоки sqrt(2·max(ψ_j, 0)) (ψ_j — AL-вклад
ограничения), поэтому 0.5·||r_aug||² = data loss + Σ max(ψ_j, 0)
совпадает с AL-потерей, пока вклады неотрицательны, и остаётся
согласованной квадратичной штрафной аппроксимацией иначе; λ при этом
между внешними итерациями обновляется по общему правилу.

Зафиксированные зависимости от приватного API (не изменять без
синхронизации): neuraltide.optimization.problem._forward (через вызов
метода), _materialize_initial_state, _phi_gradient, CalibrationProblem
._prepare / ._require_objective / ._build_evaluation / ._counters (как
в sensitivities.py); neuraltide.optimization.results._JsonDTO (как в
strategies.py).

Ограничения: eager-режим; batch=1; для CalibrationProblem градиент
AL-потери точный (одна лента); для duck-typed задач нужны simulate(phi)
-> predictions и jacobian ('fd' или callable) у каждого ConstraintSpec.
"""

import json
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

import neuraltide.config
from neuraltide.core.types import cast_if_needed
from neuraltide.optimization.problem import (
    CalibrationProblem,
    _materialize_initial_state,
    _phi_gradient,
)
from neuraltide.optimization.results import (
    Evaluation,
    OptimizationResult,
    _JsonDTO,
)
from neuraltide.optimization.strategies import resolve_local_optimizer

__all__ = [
    "ConstraintSpec",
    "AugmentedLagrangianState",
    "ConstrainedProblem",
    "ALResult",
    "epsilon_constraint",
    "augmented_lagrangian",
    "fd_jacobian",
]

_KINDS = ("eq", "ineq")


# ---------------------------------------------------------------------------
# ConstraintSpec
# ---------------------------------------------------------------------------


@dataclass
class ConstraintSpec:
    """
    Ограничение на прогнозы сети.

    Атрибуты:
        name: уникальное имя ограничения (ключ в augmented_terms,
            violation_history и state.lambdas);
        kind: 'eq' — равенство c(x) = 0; 'ineq' — неравенство g(x) <= 0;
        fn: callable(predictions, network) -> Tensor; predictions —
            NetworkOutput (CalibrationProblem) либо результат simulate(phi)
            duck-typed задачи;
        jacobian: None | 'fd' | callable(phi) -> array [m, n] (m — размер
            значения ограничения, n — problem.size). Используется только на
            generic-пути (не-CalibrationProblem); для CalibrationProblem
            градиент ограничений получается автоматически в общей ленте.
    """

    name: str
    kind: str
    fn: Callable[[Any, Any], Any]
    jacobian: Optional[Any] = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("ConstraintSpec: 'name' must be a non-empty string.")
        if self.kind not in _KINDS:
            raise ValueError(
                f"ConstraintSpec: unknown kind '{self.kind}'. "
                f"Available kinds: {list(_KINDS)} ('ineq' означает g <= 0)."
            )
        if not callable(self.fn):
            raise TypeError(
                f"ConstraintSpec: 'fn' must be callable "
                f"(predictions, network) -> Tensor, got {type(self.fn).__name__}."
            )
        if self.jacobian is not None:
            if isinstance(self.jacobian, str):
                if self.jacobian != "fd":
                    raise ValueError(
                        f"ConstraintSpec: jacobian-строка должна быть 'fd', "
                        f"получена {self.jacobian!r}."
                    )
            elif not callable(self.jacobian):
                raise TypeError(
                    "ConstraintSpec: 'jacobian' должен быть None, 'fd' или "
                    f"callable(phi) -> [m, n]; получен {type(self.jacobian).__name__}."
                )


# ---------------------------------------------------------------------------
# AugmentedLagrangianState
# ---------------------------------------------------------------------------


@dataclass
class AugmentedLagrangianState:
    """
    Состояние метода аугментированных множителей.

    Атрибуты:
        lambdas: {constraint_name: ndarray} — множители в форме значений
            ограничений (той же размерности, что и fn); для 'eq' любого
            знака, для 'ineq' неотрицательные;
        mu: штрафной параметр (> 0);
        mu_factor: множитель роста mu (> 1), когда нарушение не
            сокращается достаточно быстро;
        prev_violation: норма нарушения на предыдущем внешнем шаге
            (для правила роста mu; None до первого update).

    Правило обновления update() (Nocedal & Wright, гл. 17):
        eq:  λ <- λ − μ·c(x_k);
        ineq: λ <- max(0, λ + μ·g(x_k));
        v = max по всем ограничениям max|violation|; если
        v > eta · prev_violation, то μ <- mu_factor·μ.
    """

    lambdas: Dict[str, np.ndarray]
    mu: float = 10.0
    mu_factor: float = 10.0
    prev_violation: Optional[float] = None

    def __post_init__(self) -> None:
        self.lambdas = {
            str(k): np.asarray(v, dtype=np.float64).reshape(-1)
            for k, v in self.lambdas.items()
        }
        self.mu = float(self.mu)
        self.mu_factor = float(self.mu_factor)
        if self.mu <= 0.0:
            raise ValueError(
                f"AugmentedLagrangianState: mu must be > 0, got {self.mu}."
            )
        if self.mu_factor <= 1.0:
            raise ValueError(
                "AugmentedLagrangianState: mu_factor must be > 1, "
                f"got {self.mu_factor}."
            )
        if self.prev_violation is not None:
            self.prev_violation = float(self.prev_violation)

    def lambda_for(self, name: str, size: int) -> np.ndarray:
        """
        Множитель ограничения name нужной длины; лениво создаёт нули
        при первом обращении (размерность значения известна только
        после первого прогона).
        """
        arr = self.lambdas.get(name)
        if arr is None:
            arr = np.zeros(int(size), dtype=np.float64)
            self.lambdas[name] = arr
        elif arr.shape[0] != int(size):
            raise ValueError(
                f"AugmentedLagrangianState: constraint '{name}' дал значение "
                f"размера {int(size)}, а множитель имеет размер {arr.shape[0]}."
            )
        return arr

    def violation_array(self, name: str, kind: str, value: np.ndarray) -> np.ndarray:
        """Поэлементное нарушение: |c| для 'eq', max(g, 0) для 'ineq'."""
        value = np.asarray(value, dtype=np.float64).reshape(-1)
        self.lambda_for(name, value.size)
        if kind == "eq":
            return np.abs(value)
        return np.maximum(value, 0.0)

    def update(
        self,
        values: Mapping[str, np.ndarray],
        kinds: Mapping[str, str],
        *,
        eta: float = 0.25,
    ) -> float:
        """
        Обновить множители (и при необходимости mu) по значениям
        ограничений в точке x_k.

        Args:
            values: {name: значение ограничения} в точке x_k;
            kinds: {name: 'eq' | 'ineq'};
            eta: требуемое сокращение нарушения (0 < eta < 1); при
                невыполнении mu увеличивается в mu_factor раз.

        Returns:
            v — max-норма нарушения по всем ограничениям после x_k.
        """
        eta = float(eta)
        if not 0.0 < eta < 1.0:
            raise ValueError(
                f"AugmentedLagrangianState.update: eta must be in (0, 1), got {eta}."
            )
        v = 0.0
        for name, value in values.items():
            value = np.asarray(value, dtype=np.float64).reshape(-1)
            lam = self.lambda_for(name, value.size)
            kind = kinds[name]
            if kind == "eq":
                lam -= self.mu * value
                viol = np.abs(value)
            else:
                np.maximum(lam + self.mu * value, 0.0, out=lam)
                viol = np.maximum(value, 0.0)
            if viol.size:
                v = max(v, float(np.max(viol)))
        if self.prev_violation is not None and v > eta * self.prev_violation:
            self.mu *= self.mu_factor
        self.prev_violation = v
        return v

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lambdas": {k: v.tolist() for k, v in self.lambdas.items()},
            "mu": float(self.mu),
            "mu_factor": float(self.mu_factor),
            "prev_violation": (
                None if self.prev_violation is None else float(self.prev_violation)
            ),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AugmentedLagrangianState":
        prev = data.get("prev_violation")
        return cls(
            lambdas={
                str(k): np.asarray(v, dtype=np.float64)
                for k, v in data.get("lambdas", {}).items()
            },
            mu=float(data.get("mu", 10.0)),
            mu_factor=float(data.get("mu_factor", 10.0)),
            prev_violation=None if prev is None else float(prev),
        )


# ---------------------------------------------------------------------------
# fd_jacobian
# ---------------------------------------------------------------------------


def fd_jacobian(
    value_fn: Callable[[np.ndarray], np.ndarray],
    phi: Sequence[float],
    *,
    h: Optional[float] = None,
    h_values: Optional[Sequence[float]] = None,
    base_value: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """
    Центральные конечные разности якобиана вектор-функции ограничений.

    J[:, j] = (v(φ + h·e_j) − v(φ − h·e_j)) / (2h), v = value_fn(φ).

    Бюджет оценок value_fn (каждая оценка — один прогон на значение
    ограничения, data-loss не вычисляется):
        фиксированный h:   2·n (+1, если не передан base_value —
                           значение v(φ) в базовой точке);
        сканирование h_values: 2·n·len(h_values) (+1 без base_value).
    При сканировании возвращается оценка с плато (минимальное попарное
    расхождение соседних оценок), как в diagnostics.finite_difference_scan.

    Args:
        value_fn: v(φ) -> array [m];
        phi: точка оценки;
        h: фиксированный шаг (по умолчанию eps^(1/3)·max(1, |φ_j|)
            покоординатно — оптимальный масштаб для центральных разностей);
        h_values: сетка шагов для сканирования (переопределяет h);
        base_value: уже вычисленное v(φ) — базовая точка не оценивается
            повторно (в ConstrainedProblem значение ограничения известно
            из общего прогона).

    Returns:
        J — ndarray [m, n].
    """
    phi_np = np.asarray(phi, dtype=np.float64).reshape(-1)
    n = phi_np.shape[0]
    if base_value is None:
        base = np.asarray(value_fn(phi_np), dtype=np.float64).reshape(-1)
    else:
        base = np.asarray(base_value, dtype=np.float64).reshape(-1)

    def _column(step: np.ndarray, step_h: float) -> np.ndarray:
        jp = np.asarray(value_fn(phi_np + step), dtype=np.float64).reshape(-1)
        jm = np.asarray(value_fn(phi_np - step), dtype=np.float64).reshape(-1)
        if jp.shape != base.shape or jm.shape != base.shape:
            raise ValueError(
                "fd_jacobian: value_fn вернула разные формы на соседних "
                "точках — якобиан не определён."
            )
        return (jp - jm) / (2.0 * step_h)

    def _jacobian(step_vec: np.ndarray) -> np.ndarray:
        jac = np.empty((base.shape[0], n), dtype=np.float64)
        for j in range(n):
            step = np.zeros(n, dtype=np.float64)
            step[j] = step_vec[j]
            jac[:, j] = _column(step, step_vec[j])
        return jac

    if h_values is not None:
        hs = np.atleast_1d(np.asarray(h_values, dtype=np.float64)).reshape(-1)
        if hs.size == 0:
            raise ValueError("fd_jacobian: непустой h_values обязателен.")
        jacobians = [_jacobian(np.full(n, float(hh))) for hh in hs]
        if len(jacobians) > 1:
            diffs = [
                float(np.max(np.abs(jacobians[k + 1] - jacobians[k])))
                for k in range(len(jacobians) - 1)
            ]
            k_best = int(np.argmin(diffs))
        else:
            k_best = 0
        return jacobians[k_best]

    if h is None:
        step_vec = np.cbrt(np.finfo(np.float64).eps) * np.maximum(1.0, np.abs(phi_np))
    else:
        step_vec = np.full(n, float(h))
    if np.any(step_vec <= 0.0):
        raise ValueError("fd_jacobian: шаг h должен быть положительным.")
    return _jacobian(step_vec)


# ---------------------------------------------------------------------------
# ConstrainedProblem
# ---------------------------------------------------------------------------


def _rebuild_evaluation(
    eval0: Any,
    *,
    loss: float,
    components: Mapping[str, float],
    grad: Optional[np.ndarray] = None,
) -> Evaluation:
    """Evaluation с заменённой потерей/компонентами (сохраняя счётчики)."""
    grad_np = None if grad is None else np.asarray(grad, dtype=np.float64)
    if isinstance(eval0, Evaluation):
        return Evaluation(
            loss=float(loss),
            components=dict(components),
            residuals=np.asarray(eval0.residuals, dtype=np.float64),
            grad=grad_np,
            status=str(eval0.status),
            n_forward=int(eval0.n_forward),
            n_gradient=int(eval0.n_gradient),
            wall_time=float(eval0.wall_time),
            dtype=str(eval0.dtype),
            dt=float(eval0.dt),
        )
    return Evaluation(loss=float(loss), components=dict(components), grad=grad_np)


class ConstrainedProblem:
    """
    Duck-typed обёртка CalibrationProblem с ограничениями на прогнозы.

    Потеря данных и AL-член вычисляются в ОДНОМ прогоне/ленте:

    - fast-путь (обёрнутый problem — CalibrationProblem): точная схема
      sensitivities.py — pack/_prepare, материализация начального
      состояния и прогон внутри GradientTape, objective + AL-штраф одной
      лентой, градиент через _phi_gradient (учитывает model_var и
      tensor-backed x0-записи); n_forward += 1, n_gradient += 1 на вызов;
    - generic-путь (duck-typed problem): data-часть — из
      problem.value_and_gradient(phi), значения ограничений — из
      problem.simulate(phi) -> predictions (один прогон на значение,
      data-loss повторно не вычисляется), градиент штрафа — через
      jacobian каждого ConstraintSpec ('fd' или callable).

    evaluate(...) возвращает Evaluation, у которого loss = data loss +
    AL-штраф, а components дополнены ключами 'al/penalty' и
    'constraint/<name>' (max-норма нарушения). Контракт мутации модели
    наследуется от обёрнутой задачи.

    Args:
        problem: CalibrationProblem или duck-аналог (size / pack /
            unpack / bounds_phi / evaluate / value_and_gradient /
            residual / validate / simulate для generic-пути);
        constraints: непустая последовательность ConstraintSpec с
            уникальными именами;
        state: AugmentedLagrangianState (по умолчанию нулевые множители,
            mu=10); обёртка читает его значения на каждом вызове, поэтому
            общий state можно менять между внешними итерациями AL;
        augmented_residual: если True, residual() дополняет data-невязки
            блоками ограничений sqrt(2·max(ψ_j, 0)) для least-squares
            постановки (см. оговорку в документации модуля).
    """

    def __init__(
        self,
        problem,
        constraints: Sequence[ConstraintSpec],
        state: Optional[AugmentedLagrangianState] = None,
        *,
        augmented_residual: bool = False,
    ):
        self._problem = problem
        specs = list(constraints)
        if not specs:
            raise ValueError(
                "ConstrainedProblem: непустая последовательность "
                "ConstraintSpec обязательна."
            )
        for spec in specs:
            if not isinstance(spec, ConstraintSpec):
                raise TypeError(
                    "ConstrainedProblem: ожидаются ConstraintSpec, получен "
                    f"{type(spec).__name__}."
                )
        names = [s.name for s in specs]
        if len(set(names)) != len(names):
            raise ValueError(
                f"ConstrainedProblem: constraint names must be unique, got {names}."
            )
        self._constraints = specs
        self._state = (
            state
            if state is not None
            else AugmentedLagrangianState(lambdas={}, mu=10.0)
        )
        self._augmented_residual = bool(augmented_residual)
        self._fast = isinstance(problem, CalibrationProblem)

    # ------------------------------------------------------------------
    # Пасс-through атрибуты обёрнутой задачи
    # ------------------------------------------------------------------
    def __getattr__(self, item: str) -> Any:
        # Вызывается только для атрибутов, не найденных обычным путём.
        return getattr(self._problem, item)

    # ------------------------------------------------------------------
    # Свойства
    # ------------------------------------------------------------------
    @property
    def problem(self):
        """Обёрнутая задача."""
        return self._problem

    @property
    def constraints(self) -> List[ConstraintSpec]:
        return list(self._constraints)

    @property
    def state(self) -> AugmentedLagrangianState:
        return self._state

    @property
    def size(self) -> int:
        return int(self._problem.size)

    @property
    def n_forward(self) -> int:
        return int(getattr(self._problem, "n_forward", 0))

    @property
    def n_gradient(self) -> int:
        return int(getattr(self._problem, "n_gradient", 0))

    @property
    def wall_time(self) -> float:
        return float(getattr(self._problem, "wall_time", 0.0))

    # ------------------------------------------------------------------
    # pack / unpack / bounds_phi
    # ------------------------------------------------------------------
    def pack(self, phi: Optional[Any] = None) -> Any:
        return self._problem.pack(phi)

    def unpack(self, phi: Any) -> Dict[str, Any]:
        unpack = getattr(self._problem, "unpack", None)
        if callable(unpack):
            return unpack(phi)
        space = getattr(self._problem, "parameter_space", None)
        if space is not None and hasattr(space, "unpack"):
            return space.unpack(self.pack(phi))
        raise TypeError(
            "ConstrainedProblem: обёрнутая задача не предоставляет "
            "unpack(phi); добавьте метод unpack к duck-typed задаче."
        )

    def bounds_phi(self) -> Tuple[np.ndarray, np.ndarray]:
        bounds_fn = getattr(self._problem, "bounds_phi", None)
        if callable(bounds_fn):
            lb, ub = bounds_fn()
            return (
                np.asarray(lb, dtype=np.float64).reshape(-1),
                np.asarray(ub, dtype=np.float64).reshape(-1),
            )
        return (
            np.full(self.size, -np.inf, dtype=np.float64),
            np.full(self.size, np.inf, dtype=np.float64),
        )

    # ------------------------------------------------------------------
    # Ограничения: значения, нарушения, AL-штраф
    # ------------------------------------------------------------------
    def _predictions_generic(self, phi: Optional[Any]) -> Any:
        simulate = getattr(self._problem, "simulate", None)
        if not callable(simulate):
            raise TypeError(
                "ConstrainedProblem: duck-typed задача не предоставляет "
                "simulate(phi) -> predictions; ограничения "
                "fn(predictions, network) требуют прогнозов. Для "
                "CalibrationProblem прогнозы берутся из общего прогона."
            )
        return simulate(self.pack(phi) if phi is None else phi)

    def _constraint_values_np(self, predictions: Any) -> Dict[str, np.ndarray]:
        network = getattr(self._problem, "network", None)
        values: Dict[str, np.ndarray] = {}
        for spec in self._constraints:
            raw = spec.fn(predictions, network)
            values[spec.name] = np.asarray(raw, dtype=np.float64).reshape(-1)
        return values

    def _al_penalty_np(
        self, cvals: Mapping[str, np.ndarray]
    ) -> Tuple[float, Dict[str, np.ndarray]]:
        """(значение AL-штрафа, d(штраф)/d(значение ограничения))."""
        mu = self._state.mu
        total = 0.0
        dpsi: Dict[str, np.ndarray] = {}
        for spec in self._constraints:
            c = np.asarray(cvals[spec.name], dtype=np.float64).reshape(-1)
            lam = self._state.lambda_for(spec.name, c.size)
            if spec.kind == "eq":
                psi = -lam * c + 0.5 * mu * c * c
                d = mu * c - lam
            else:
                t = lam + mu * c
                active = t > 0.0
                psi = np.where(
                    active, lam * c + 0.5 * mu * c * c, -lam * lam / (2.0 * mu)
                )
                d = np.where(active, t, 0.0)
            total += float(np.sum(psi))
            dpsi[spec.name] = d
        return total, dpsi

    def _al_penalty_tf(self, output: Any) -> Tuple[Any, Dict[str, np.ndarray]]:
        """AL-штраф в ленте (дифференцируем) + значения ограничений."""
        dtype = neuraltide.config.get_dtype()
        mu = self._state.mu
        network = getattr(self._problem, "network", None)
        total = None
        cvals: Dict[str, np.ndarray] = {}
        for spec in self._constraints:
            raw = spec.fn(output, network)
            c_t = tf.reshape(cast_if_needed(tf.convert_to_tensor(raw), dtype), [-1])
            cvals[spec.name] = np.asarray(c_t, dtype=np.float64).reshape(-1)
            lam = self._state.lambda_for(spec.name, int(c_t.shape[0]))
            lam_t = tf.constant(lam, dtype=dtype)
            if spec.kind == "eq":
                psi = -lam_t * c_t + 0.5 * mu * c_t * c_t
            else:
                t = lam_t + mu * c_t
                psi = (tf.maximum(t, tf.zeros_like(t)) ** 2 - lam_t**2) / (2.0 * mu)
            s = tf.reduce_sum(psi)
            total = s if total is None else total + s
        if total is None:  # pragma: no cover - constraints непусты
            total = tf.constant(0.0, dtype=dtype)
        return total, cvals

    def _component_entries(
        self, cvals: Mapping[str, np.ndarray], penalty: float
    ) -> Dict[str, float]:
        entries = {"al/penalty": float(penalty)}
        for spec in self._constraints:
            viol = self._state.violation_array(spec.name, spec.kind, cvals[spec.name])
            entries[f"constraint/{spec.name}"] = (
                float(np.max(viol)) if viol.size else 0.0
            )
        return entries

    def augmented_terms(
        self, phi: Optional[Any] = None
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Значения и нарушения ограничений в точке phi одним прогоном.

        Returns:
            {name: (value, violation)} — value — сырое значение fn,
            violation — |c| ('eq') или max(g, 0) ('ineq'), numpy float64.
        """
        if self._fast:
            _, output = self._problem._forward(phi)
            cvals = self._constraint_values_np(output)
        else:
            cvals = self._constraint_values_np(self._predictions_generic(phi))
        return {
            spec.name: (
                cvals[spec.name],
                self._state.violation_array(spec.name, spec.kind, cvals[spec.name]),
            )
            for spec in self._constraints
        }

    # ------------------------------------------------------------------
    # Операции задачи
    # ------------------------------------------------------------------
    def evaluate(self, phi: Optional[Any] = None) -> Evaluation:
        """data loss + AL-штраф, одним прогоном (счётчик +1 n_forward)."""
        if self._fast:
            problem = self._problem
            problem._require_objective()
            t0 = time.perf_counter()
            _, output = problem._forward(phi)
            loss_data = problem.objective(output, problem.network)
            penalty, cvals = self._al_penalty_tf(output)
            total = loss_data + penalty
            eval_result = problem._build_evaluation(
                output, grad=None, t0=t0, loss_override=total
            )
            eval_result.components.update(
                self._component_entries(cvals, float(penalty))
            )
            return eval_result
        eval0 = self._problem.evaluate(phi)
        cvals = self._constraint_values_np(self._predictions_generic(phi))
        penalty, _ = self._al_penalty_np(cvals)
        components = dict(getattr(eval0, "components", {}) or {})
        components.update(self._component_entries(cvals, penalty))
        return _rebuild_evaluation(
            eval0, loss=float(eval0.loss) + penalty, components=components
        )

    def _constraint_jacobians(
        self, phi: Optional[Any], cvals: Mapping[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Якобианы ограничений по phi для generic-пути."""
        phi_np = np.asarray(
            self.pack(phi) if phi is None else phi, dtype=np.float64
        ).reshape(-1)
        jacobians: Dict[str, np.ndarray] = {}
        for spec in self._constraints:
            expected = (int(cvals[spec.name].size), phi_np.shape[0])
            if callable(spec.jacobian):
                jac = np.atleast_2d(
                    np.asarray(spec.jacobian(phi_np.copy()), dtype=np.float64)
                )
            elif isinstance(spec.jacobian, str) and spec.jacobian == "fd":

                def _value(p: np.ndarray, name: str = spec.name) -> np.ndarray:
                    return self._constraint_values_np(self._predictions_generic(p))[
                        name
                    ]

                jac = fd_jacobian(_value, phi_np, base_value=cvals[spec.name])
            else:
                raise TypeError(
                    f"ConstrainedProblem: constraint '{spec.name}' требует "
                    "jacobian ('fd' или callable(phi) -> [m, n]): у "
                    "duck-typed задачи нет единой ленты, и градиент "
                    "ограничения недоступен автоматически."
                )
            if jac.shape != expected:
                raise ValueError(
                    f"ConstrainedProblem: jacobian constraint '{spec.name}' "
                    f"имеет форму {jac.shape}, ожидается {expected}."
                )
            jacobians[spec.name] = jac
        return jacobians

    def value_and_gradient(self, phi: Optional[Any] = None):
        """
        (Evaluation, grad) для data loss + AL-штрафа.

        Fast-путь: один прогон в одной GradientTape, точный градиент по φ
        (включая tensor-backed x0-записи). Generic-путь: data-градиент из
        problem.value_and_gradient плюс градиент штрафа через якобианы
        ограничений.
        """
        if self._fast:
            problem = self._problem
            problem._require_objective()
            t0 = time.perf_counter()
            phi_t = problem._prepare(phi)
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(phi_t)
                theta = problem.parameter_space.unpack(phi_t)
                initial_state, x0_map = _materialize_initial_state(
                    problem.network,
                    problem.parameter_space,
                    theta,
                    problem.initial_state,
                )
                output = problem.network(
                    problem.t_sequence,
                    inputs=problem.inputs,
                    initial_state=initial_state,
                )
                loss_data = problem.objective(output, problem.network)
                penalty, cvals = self._al_penalty_tf(output)
                total = loss_data + penalty
            grad_t = _phi_gradient(
                problem.parameter_space,
                problem.grad_mode,
                tape,
                total,
                phi_t,
                x0_map,
            )
            del tape
            problem._counters["n_gradient"] += 1
            eval_result = problem._build_evaluation(
                output, grad=grad_t, t0=t0, loss_override=total
            )
            eval_result.components.update(
                self._component_entries(cvals, float(penalty))
            )
            return eval_result, grad_t

        eval0, grad0 = self._problem.value_and_gradient(phi)
        grad0 = np.asarray(grad0, dtype=np.float64).reshape(-1)
        predictions = self._predictions_generic(phi)
        cvals = self._constraint_values_np(predictions)
        penalty, dpsi = self._al_penalty_np(cvals)
        jacobians = self._constraint_jacobians(phi, cvals)
        grad_penalty = np.zeros_like(grad0)
        for spec in self._constraints:
            grad_penalty = grad_penalty + dpsi[spec.name] @ jacobians[spec.name]
        grad = grad0 + grad_penalty
        components = dict(getattr(eval0, "components", {}) or {})
        components.update(self._component_entries(cvals, penalty))
        eval_result = _rebuild_evaluation(
            eval0,
            loss=float(eval0.loss) + penalty,
            components=components,
            grad=grad,
        )
        return eval_result, grad

    def _residual_blocks(self, cvals: Mapping[str, np.ndarray]) -> List[np.ndarray]:
        """Блоки sqrt(2·max(ψ_j, 0)) для augmented residual."""
        mu = self._state.mu
        blocks: List[np.ndarray] = []
        for spec in self._constraints:
            c = np.asarray(cvals[spec.name], dtype=np.float64).reshape(-1)
            lam = self._state.lambda_for(spec.name, c.size)
            if spec.kind == "eq":
                psi = -lam * c + 0.5 * mu * c * c
            else:
                t = lam + mu * c
                psi = np.where(
                    t > 0.0, lam * c + 0.5 * mu * c * c, -lam * lam / (2.0 * mu)
                )
            blocks.append(np.sqrt(2.0 * np.maximum(psi, 0.0)))
        return blocks

    def residual(self, phi: Optional[Any] = None) -> np.ndarray:
        """
        Невязки задачи; при augmented_residual=True дополнены блоками
        ограничений (0.5·||r_aug||² = data loss + Σ max(ψ_j, 0)).
        """
        if not self._augmented_residual:
            return self._problem.residual(phi)
        if self._fast:
            problem = self._problem
            problem._require_objective()
            residual_fn = getattr(problem.objective, "residual", None)
            if residual_fn is None:
                raise TypeError(
                    "ConstrainedProblem: objective "
                    f"'{type(problem.objective).__name__}' не предоставляет "
                    "residual(predictions). Для least-squares постановки "
                    "используйте ResidualLoss, либо local='lbfgsb'."
                )
            _, output = problem._forward(phi)
            r_data = np.asarray(residual_fn(output), dtype=np.float64).reshape(-1)
            cvals = self._constraint_values_np(output)
        else:
            r_data = np.asarray(self._problem.residual(phi), dtype=np.float64).reshape(
                -1
            )
            cvals = self._constraint_values_np(self._predictions_generic(phi))
        return np.concatenate([r_data] + self._residual_blocks(cvals))

    def simulate(self, phi: Optional[Any] = None) -> Any:
        return self._problem.simulate(phi)

    def validate(self, phi: Optional[Any] = None) -> Dict[str, float]:
        return self._problem.validate(phi)


# ---------------------------------------------------------------------------
# epsilon_constraint
# ---------------------------------------------------------------------------


class _WeightedComponentObjective:
    """Взвешенная сумма выбранных компонентов исходного objective."""

    def __init__(
        self, base_objective: Any, names: Sequence[str], weights: Mapping[str, float]
    ):
        self._base = base_objective
        self._names = [str(n) for n in names]
        self._weights = {str(n): float(weights[n]) for n in self._names}

    def __call__(self, predictions: Any, network: Any) -> Any:
        comps = self._base.components(predictions, network)
        total = None
        for name in self._names:
            if name not in comps:
                raise KeyError(
                    f"_WeightedComponentObjective: component '{name}' not "
                    f"found in {sorted(comps)}."
                )
            term = self._weights[name] * comps[name]
            total = term if total is None else total + term
        return total

    def components(self, predictions: Any, network: Any) -> Dict[str, Any]:
        comps = self._base.components(predictions, network)
        return {name: comps[name] for name in self._names}


def epsilon_constraint(
    problem,
    objectives: Sequence[str],
    index: int,
    eps: float,
    *,
    objective: Optional[Any] = None,
    weights: Optional[Mapping[str, float]] = None,
) -> ConstrainedProblem:
    """
    Эпсилон-ограничение: компонент objectives[index] становится
    ограничением value <= eps, целевая функция — взвешенная сумма
    остальных компонентов (или явный objective).

    Args:
        problem: CalibrationProblem (или duck-аналог с
            with_objective либо изменяемым атрибутом objective);
        objectives: имена компонентов objective.components
            (порядок задаёт index);
        index: индекс выбранного компонента (допускаются
            отрицательные индексы);
        eps: правую часть неравенства (компонент <= eps);
        objective: явная целевая функция callable(predictions, network)
            -> scalar; None — взвешенная сумма остальных компонентов;
        weights: веса {имя: float} для суммы остальных (по умолчанию
            единичные); неизвестные имена → ValueError.

    Returns:
        ConstrainedProblem с одним ограничением kind 'ineq'
        (g = components[selected] - eps <= 0).
    """
    objectives = [str(o) for o in objectives]
    if not objectives:
        raise ValueError("epsilon_constraint: непустой objectives обязателен.")
    index = int(index)
    if index < 0:
        index += len(objectives)
    if not 0 <= index < len(objectives):
        raise IndexError(
            f"epsilon_constraint: index {index} вне [0, {len(objectives)})."
        )
    eps = float(eps)
    base_objective = getattr(problem, "objective", None)
    components_fn = getattr(base_objective, "components", None)
    if not callable(components_fn):
        raise TypeError(
            "epsilon_constraint: objective задачи должен предоставлять "
            "components(predictions, network); иначе выбор компонента "
            "невозможен."
        )
    selected = objectives[index]
    remaining = [name for i, name in enumerate(objectives) if i != index]

    if objective is None:
        if not remaining:
            raise ValueError(
                "epsilon_constraint: не осталось компонентов для целевой "
                "функции; передайте явный objective=...."
            )
        if weights is None:
            weight_map = {name: 1.0 for name in remaining}
        else:
            weight_map = {str(k): float(v) for k, v in weights.items()}
            unknown = sorted(set(weight_map) - set(remaining))
            if unknown:
                raise ValueError(
                    f"epsilon_constraint: weights содержат неизвестные "
                    f"имена {unknown} (остались {remaining})."
                )
            for name in remaining:
                weight_map.setdefault(name, 1.0)
        new_objective = _WeightedComponentObjective(
            base_objective, remaining, weight_map
        )
    else:
        if not callable(objective):
            raise TypeError(
                "epsilon_constraint: 'objective' должен быть callable "
                f"(predictions, network) -> scalar, получен "
                f"{type(objective).__name__}."
            )
        new_objective = objective

    def _component_minus_eps(predictions, network, _c=components_fn, _n=selected):
        return _c(predictions, network)[_n] - eps

    spec = ConstraintSpec(name=f"eps/{selected}", kind="ineq", fn=_component_minus_eps)
    with_objective = getattr(problem, "with_objective", None)
    if callable(with_objective):
        base = with_objective(new_objective)
    else:
        try:
            problem.objective = new_objective
        except AttributeError as exc:
            raise TypeError(
                "epsilon_constraint: duck-typed задача не предоставляет "
                "with_objective и не позволяет присвоить problem.objective."
            ) from exc
        base = problem
    return ConstrainedProblem(base, [spec])


# ---------------------------------------------------------------------------
# ALResult
# ---------------------------------------------------------------------------


@dataclass
class ALResult(_JsonDTO):
    """
    Результат augmented_lagrangian.

    Атрибуты:
        base: базовый OptimizationResult (phi_best/theta_best/loss_best —
            data loss исходной задачи в phi_best, история внутренних
            итераций с тегом 'outer', seed, config);
        state: финальное состояние множителей (lambdas/mu);
        violation_history: записи по внешним итерациям
            [{'outer', 'violation', 'mu_after', 'inner_loss',
            'constraint/<name>'}, ...];
        n_outer: число выполненных внешних итераций;
        validation: метрики validate() на held-out протоколах;
        success: все внутренние решатели сошлись и нарушение убывает; при
            resume уже завершённого прогона (внутренних запусков нет) —
            по итоговой violation_history;
        wall_time: суммарное время, секунды.
    """

    base: OptimizationResult
    state: AugmentedLagrangianState
    violation_history: List[Dict[str, Any]] = field(default_factory=list)
    n_outer: int = 0
    validation: Dict[str, float] = field(default_factory=dict)
    success: bool = False
    wall_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base": self.base.to_dict(),
            "state": self.state.to_dict(),
            "violation_history": [dict(r) for r in self.violation_history],
            "n_outer": int(self.n_outer),
            "validation": {str(k): float(v) for k, v in self.validation.items()},
            "success": bool(self.success),
            "wall_time": float(self.wall_time),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ALResult":
        return cls(
            base=OptimizationResult.from_dict(data["base"]),
            state=AugmentedLagrangianState.from_dict(data["state"]),
            violation_history=[dict(r) for r in data.get("violation_history", [])],
            n_outer=int(data.get("n_outer", 0)),
            validation={
                str(k): float(v) for k, v in data.get("validation", {}).items()
            },
            success=bool(data.get("success", False)),
            wall_time=float(data.get("wall_time", 0.0)),
        )


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def _write_al_checkpoint(
    path,
    *,
    seed: Optional[int],
    config: Mapping[str, Any],
    constraints: Sequence[ConstraintSpec],
    state: AugmentedLagrangianState,
    phi: np.ndarray,
    outer_next: int,
    violation_history: Sequence[Mapping[str, Any]],
) -> None:
    payload = {
        "seed": seed,
        "config": dict(config),
        "constraints": [{"name": s.name, "kind": s.kind} for s in constraints],
        "state": state.to_dict(),
        "phi": np.asarray(phi, dtype=np.float64).reshape(-1).tolist(),
        "outer_next": int(outer_next),
        "violation_history": [dict(r) for r in violation_history],
    }
    with open(os.fspath(path), "w") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# augmented_lagrangian
# ---------------------------------------------------------------------------


def augmented_lagrangian(
    problem,
    constraints: Sequence[ConstraintSpec],
    *,
    local: Any = "lbfgsb",
    max_outer: int = 10,
    mu0: float = 10.0,
    eta: float = 0.25,
    seed: Optional[int] = None,
    checkpoint_path=None,
    resume: bool = False,
) -> ALResult:
    """
    Метод аугментированных множителей для ограничений на прогнозы.

    Внешние итерации (стандартные, Nocedal & Wright гл. 17):

    1. стартовая точка — problem.pack() (недопустимая по bounds_phi
       точка спроектируется клиппингом с предупреждением UserWarning —
       поведение зафиксировано в config['projected_start'], а не
       маскируется);
    2. внутренняя минимизация AL-потери ConstrainedProblem из текущей
       точки: local='lbfgsb' (точный градиент, один прогон на оценку)
       или local='least_squares' (data-невязки + блоки ограничений,
       см. оговорку в документации модуля), либо callable(problem, phi0)
       -> результат с .base.(phi_best, loss_best, history) и .success;
    3. обновление λ (и при недостаточном сокращении нарушения — μ в
       mu_factor раз, eta — требуемое сокращение);
    4. checkpoint JSON после каждой внешней итерации; повторный запуск
       с resume=True и совпадающими (seed, config, constraints)
       продолжает с сохранённой точки и состояния.

    Args:
        problem: CalibrationProblem (fast-путь) или duck-typed аналог
            (generic-путь: simulate + jacobian у каждого ConstraintSpec);
        constraints: непустая последовательность ConstraintSpec;
        local: 'lbfgsb' | 'least_squares' | callable(problem, phi0);
        max_outer: число внешних итераций (>= 1);
        mu0: начальный штрафной параметр (> 0);
        eta: требуемое сокращение нарушения между внешними итерациями
            (0 < eta < 1);
        seed: seed воспроизводимости (фиксируется в config/чекпоинте);
        checkpoint_path: путь к JSON-чекпоинту (None — без чекпоинта);
        resume: продолжить с существующего checkpoint_path.

    Returns:
        ALResult с финальными phi_best/theta_best (data loss исходной
        задачи в loss_best), violation_history и состоянием множителей.
    """
    specs = list(constraints)
    if not specs:
        raise ValueError(
            "augmented_lagrangian: непустая последовательность "
            "ConstraintSpec обязательна."
        )
    for spec in specs:
        if not isinstance(spec, ConstraintSpec):
            raise TypeError(
                "augmented_lagrangian: ожидаются ConstraintSpec, получен "
                f"{type(spec).__name__}."
            )
    max_outer = int(max_outer)
    if max_outer < 1:
        raise ValueError(
            f"augmented_lagrangian: max_outer must be >= 1, got {max_outer}."
        )
    mu0 = float(mu0)
    if mu0 <= 0.0:
        raise ValueError(f"augmented_lagrangian: mu0 must be > 0, got {mu0}.")
    eta = float(eta)
    if not 0.0 < eta < 1.0:
        raise ValueError(f"augmented_lagrangian: eta must be in (0, 1), got {eta}.")
    local_fn, local_name = resolve_local_optimizer(local)
    kinds = {s.name: s.kind for s in specs}
    t_begin = time.perf_counter()

    # --- Стартовая точка: нефинитность — ошибка, нарушение bounds ---
    # --- — явная проекция с предупреждением (не маскировка) ----------
    phi = np.asarray(problem.pack(), dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(phi)):
        raise ValueError(
            "augmented_lagrangian: стартовый phi (problem.pack()) "
            "содержит нефинитные значения; запуск отклонён."
        )
    projected = False
    bounds_fn = getattr(problem, "bounds_phi", None)
    if callable(bounds_fn):
        lb, ub = bounds_fn()
        lb = np.asarray(lb, dtype=np.float64).reshape(-1)
        ub = np.asarray(ub, dtype=np.float64).reshape(-1)
        clipped = np.clip(phi, lb, ub)
        projected = not np.array_equal(clipped, phi)
        if projected:
            warnings.warn(
                "augmented_lagrangian: стартовый phi вне bounds_phi; "
                "точка спроецирована клиппингом перед первой внешней "
                "итерацией.",
                UserWarning,
                stacklevel=2,
            )
            phi = clipped

    config: Dict[str, Any] = {
        "local": local_name,
        "max_outer": int(max_outer),
        "mu0": float(mu0),
        "eta": float(eta),
        "constraints": [{"name": s.name, "kind": s.kind} for s in specs],
        "projected_start": bool(projected),
    }
    state = AugmentedLagrangianState(lambdas={}, mu=mu0)
    violation_history: List[Dict[str, Any]] = []
    start_outer = 0

    if (
        checkpoint_path is not None
        and resume
        and os.path.exists(os.fspath(checkpoint_path))
    ):
        with open(os.fspath(checkpoint_path), "r") as f:
            data = json.load(f)
        if data.get("seed") != seed:
            raise ValueError(
                f"augmented_lagrangian: seed чекпоинта ({data.get('seed')}) "
                f"не совпадает с запрошенным ({seed}); resume невозможен."
            )
        stored = {k: v for k, v in data.get("config", {}).items() if k != "max_outer"}
        requested = {k: v for k, v in config.items() if k != "max_outer"}
        if stored != requested:
            raise ValueError(
                "augmented_lagrangian: конфигурация чекпоинта не "
                f"совпадает с запрошенной (max_outer может отличаться "
                f"— resume продолжает до нового бюджета): "
                f"checkpoint={stored}, requested={requested}."
            )
        stored_c = data.get("constraints", [])
        requested_c = config["constraints"]
        if stored_c != requested_c:
            raise ValueError(
                "augmented_lagrangian: ограничения чекпоинта не "
                f"совпадают с запрошенными: checkpoint={stored_c}, "
                f"requested={requested_c}."
            )
        state = AugmentedLagrangianState.from_dict(data["state"])
        phi = np.asarray(data["phi"], dtype=np.float64).reshape(-1)
        start_outer = int(data["outer_next"])
        violation_history = [dict(r) for r in data.get("violation_history", [])]

    # --- Внешние итерации ----------------------------------------------
    inners: List[Any] = []
    for k in range(start_outer, max_outer):
        cp = ConstrainedProblem(
            problem,
            specs,
            state=state,
            augmented_residual=(local_name == "least_squares"),
        )
        inner = local_fn(cp, phi)
        inners.append(inner)
        inner_base = getattr(inner, "base", None)
        phi = np.asarray(
            getattr(inner_base, "phi_best", phi), dtype=np.float64
        ).reshape(-1)

        terms = cp.augmented_terms(phi)
        values = {name: term[0] for name, term in terms.items()}
        v = state.update(values, kinds, eta=eta)
        rec: Dict[str, Any] = {
            "outer": k + 1,
            "violation": float(v),
            "mu_after": float(state.mu),
            "inner_loss": float(getattr(inner_base, "loss_best", float("nan"))),
        }
        for name, term in terms.items():
            viol = term[1]
            rec[f"constraint/{name}"] = (
                float(np.max(np.abs(viol))) if viol.size else 0.0
            )
        violation_history.append(rec)
        if checkpoint_path is not None:
            _write_al_checkpoint(
                checkpoint_path,
                seed=seed,
                config=config,
                constraints=specs,
                state=state,
                phi=phi,
                outer_next=k + 1,
                violation_history=violation_history,
            )

    # --- Сборка результата ---------------------------------------------
    data_eval = problem.evaluate(phi)
    unpack = getattr(problem, "unpack", None)
    if callable(unpack):
        theta = unpack(phi)
    else:
        theta = problem.parameter_space.unpack(phi)
    theta_best = {str(k): np.asarray(v) for k, v in theta.items()}
    validation: Dict[str, float] = {}
    held_out = getattr(problem, "held_out", None)
    if held_out:
        validation = {str(k): float(v) for k, v in problem.validate(phi).items()}

    history: List[Dict[str, Any]] = []
    for offset, inner in enumerate(inners):
        inner_history = getattr(getattr(inner, "base", None), "history", None) or []
        for entry in inner_history:
            history.append({"outer": start_outer + offset + 1, **dict(entry)})
    base = OptimizationResult(
        phi_best=phi,
        theta_best=theta_best,
        loss_best=float(data_eval.loss),
        history=history,
        seed=seed,
        config=config,
        dtype=str(neuraltide.config.get_dtype().name),
    )

    if inners:
        success = all(bool(getattr(inner, "success", False)) for inner in inners)
    else:
        # resume завершённого прогона (start_outer == max_outer): внутренних
        # запусков нет — успех оценивается по итоговой violation_history /
        # финальному состоянию (см. проверку убывания ниже); без истории
        # нарушений успеха нет.
        success = bool(violation_history)
    if violation_history:
        first_v = float(violation_history[0]["violation"])
        last_v = float(violation_history[-1]["violation"])
        improved = last_v <= max(1e-12, 0.5 * first_v) or last_v < 1e-10
        success = success and improved

    return ALResult(
        base=base,
        state=state,
        violation_history=violation_history,
        n_outer=len(violation_history),
        validation=validation,
        success=success,
        wall_time=time.perf_counter() - t_begin,
    )
