"""
Идентифицируемость: FIM/SVD, профили правдоподобия, выбор подмножества (P1).

Модуль ранга 11 дорожной карты (docs/optimization_roadmap.md, раздел 6.6)
поверх CalibrationProblem / ParameterSpace / FrozenProblem (P0/P1):

- compute_fim(jac, ...) — FIM F = JᵀJ по ГОТОВОМУ якобиану невязок
    (JacobianResult, см. sensitivities.residual_jacobian): спектр, число
    обусловленности, effective rank, жёсткие/мягкие направления с вкладом
    параметров, коллинеарность столбцов J и нормы чувствительности;
- likelihood_profile(problem, name, ...) — одномерный профиль правдоподобия
    по физическому θ фиксированной сеткой (не адаптивной в P1) с доводкой
    остальных координат через локальный оптимизатор (warm start от соседнего
    успешного узла) и интервалами по χ²(1);
- select_subset(problem, jac, keep, ...) — совместная доводка выбранного
    подмножества параметров (FrozenProblem) с честной оценкой смещения:
    изменение замороженных θ относительно точки якобиана, потерь и прогноза.

Допущения и ограничения (P1):
  * якобиан здесь НЕ вычисляется: compute_fim/select_subset принимают готовый
    JacobianResult (например, из sensitivities.residual_jacobian). Способ
    получения J (forward sensitivities, конечные разности) — вне контракта;
  * невязки считаются уже нормированными на sigma (контракт ResidualLoss:
    0.5·||r||² == loss), поэтому F = JᵀJ — FIM для гауссовой модели шума
    с известной диагональной Σ; дополнительный множитель 1/σ² вносить не нужно;
  * выбор спектра: effective rank считается по накопленной энергии
    СОБСТВЕННЫХ значений F (λ = s²) — кривизна логарифма правдоподобия
    определяется спектром F, а не J; condition_number — κ(J) = s_max/s_min
    по ненулевым сингулярным значениям (κ(F) = κ(J)² — отдельное поле
    condition_number_fim);
  * интервалы профиля: loss = NLL(θ) − const = 0.5·||r||², статистика
    2·ΔNLL ~ χ²(1), поэтому порог на rel_loss равен χ²_{0.95}(1)/2 ≈ 1.9207
    (χ²-порог 3.8414588 задан в поле chi2_threshold). Допущение: гауссов шум
    с ИЗВЕСТНОЙ sigma (нормировка уже в residual), один параметр, профиль
    правдоподобия; для huber/robust-видов штрафов интервалы — эвристика;
  * узлы неуспешных запусков не скрываются (losses = NaN, successes = False,
    phi_path = None) и не участвуют в интервалах; немонотонность, несвязность
    допустимой области и минимум на границе сетки фиксируются в warnings;
  * P1-профили — только по скалярным записям ParameterSpace (numel == 1).

Eager API; numpy-выходы — float64. Глобальный dtype (neuraltide.config)
используется только при взаимодействии с ParameterSpace/трансформами.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
from scipy import stats as sp_stats

import neuraltide.config
from neuraltide.optimization.local import least_squares, minimize_lbfgsb
from neuraltide.optimization.problem import FrozenProblem
from neuraltide.optimization.results import (
    JacobianResult,
    LocalOptimizerResult,
    _JsonDTO,
)

__all__ = [
    "FIMDirection",
    "CollinearityPair",
    "FIMResult",
    "ProfileResult",
    "SubsetResult",
    "compute_fim",
    "likelihood_profile",
    "select_subset",
]

# χ²-квантиль для 95% и df=1 (порог на 2·ΔNLL); порог на Δloss — вдвое меньше.
_CHI2_DF1_95 = float(sp_stats.chi2.ppf(0.95, df=1))
_CONFIDENCE_KEY = "0.95"

# Защита от деления на ноль в относительных нормах прогноза.
_NORM_EPS = 1e-12


# ---------------------------------------------------------------------------
# Внутренние helpers
# ---------------------------------------------------------------------------


def _entry_offsets(space) -> Dict[str, Tuple[int, int]]:
    """Смещения записей ParameterSpace в плоском векторе φ."""
    offsets: Dict[str, Tuple[int, int]] = {}
    start = 0
    for name, entry in space.entries.items():
        offsets[str(name)] = (start, entry.numel)
        start += entry.numel
    return offsets


def _flat_names(space, size: int) -> List[str]:
    """
    Имена φ-координат: full_name для скалярных записей, 'name[i]' иначе.
    """
    names: List[str] = []
    for name, entry in space.entries.items():
        if entry.numel == 1:
            names.append(str(name))
        else:
            names.extend(f"{name}[{i}]" for i in range(entry.numel))
    if len(names) != size:
        raise ValueError(
            f"identifiability: ParameterSpace даёт {len(names)} координат, "
            f"ожидается {size} (несовпадение размерности φ)."
        )
    return names


def _counters(problem) -> Tuple[int, int]:
    return (
        int(getattr(problem, "n_forward", 0)),
        int(getattr(problem, "n_gradient", 0)),
    )


def _free_index(offsets: Mapping[str, Tuple[int, int]], names: Sequence[str]):
    """Индексы φ-координат записей names в полном векторе φ."""
    if not names:
        return np.zeros(0, dtype=int)
    return np.concatenate(
        [np.arange(offsets[n][0], offsets[n][0] + offsets[n][1]) for n in names]
    )


# ---------------------------------------------------------------------------
# DTO: направления и пары коллинеарности
# ---------------------------------------------------------------------------


@dataclass
class FIMDirection:
    """
    Одно направление в спектре FIM (правый сингулярный вектор J).

    Атрибуты:
        index: номер в отсортированном по убыванию спектре (0 — самое
            жёсткое направление);
        singular_value: s_i = sqrt(λ_i);
        eigenvalue: λ_i — собственное значение F = JᵀJ;
        energy_fraction: доля энергии спектра λ_i / Σλ;
        cumulative_energy: накопленная доля энергии до i-го направления
            включительно;
        top_params: [(name, loading), ...] — до 5 крупнейших по модулю
            компонент собственного вектора (loading — значение компоненты,
            знак сохранён) с именами φ-координат.
    """

    index: int = 0
    singular_value: float = 0.0
    eigenvalue: float = 0.0
    energy_fraction: float = 0.0
    cumulative_energy: float = 0.0
    top_params: List[Tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": int(self.index),
            "singular_value": float(self.singular_value),
            "eigenvalue": float(self.eigenvalue),
            "energy_fraction": float(self.energy_fraction),
            "cumulative_energy": float(self.cumulative_energy),
            "top_params": [(str(n), float(v)) for n, v in self.top_params],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FIMDirection":
        return cls(
            index=int(data.get("index", 0)),
            singular_value=float(data.get("singular_value", 0.0)),
            eigenvalue=float(data.get("eigenvalue", 0.0)),
            energy_fraction=float(data.get("energy_fraction", 0.0)),
            cumulative_energy=float(data.get("cumulative_energy", 0.0)),
            top_params=[(str(n), float(v)) for n, v in data.get("top_params", [])],
        )


@dataclass
class CollinearityPair:
    """
    Пара φ-координат с высокой коллинеарностью столбцов J.

    Атрибуты:
        first / second: имена координат;
        cosine: косинус угла между столбцами J (по модулю).
    """

    first: str = ""
    second: str = ""
    cosine: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "first": str(self.first),
            "second": str(self.second),
            "cosine": float(self.cosine),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CollinearityPair":
        return cls(
            first=str(data.get("first", "")),
            second=str(data.get("second", "")),
            cosine=float(data.get("cosine", 0.0)),
        )


# ---------------------------------------------------------------------------
# DTO: FIMResult
# ---------------------------------------------------------------------------


@dataclass
class FIMResult(_JsonDTO):
    """
    Результат FIM-анализа в точке φ (спектр F = JᵀJ).

    Атрибуты:
        fim: матрица F [n, n] (float64);
        eigenvalues / eigenvectors: спектр F, отсортированный по убыванию
            (столбцы eigenvectors — направления в φ-пространстве);
        singular_values: s_i = sqrt(λ_i) невозрастающие;
        condition_number: κ(J) = s_max/s_min по ненулевым сингулярным
            значениям (np.inf, если ненулевых нет);
        condition_number_fim: κ(F) = λ_max/λ_min по ненулевым собственным
            значениям (= κ(J)² в пределе машинной точности);
        effective_rank: число ведущих собственных значений, накапливающих
            долю energy общей энергии спектра (выбор — по энергии λ, см.
            модульную документацию);
        energy: использованный порог накопленной энергии;
        stiff_dirs: ведущие effective_rank направлений (кап max_dirs);
        soft_dirs: остальные (мягкие) направления спектра (кап max_dirs);
        collinearity: пары координат с |cos| между столбцами J выше порога;
        param_sensitivity: [(name, ||J[:, j]||), ...] по убыванию нормы;
        param_names: имена φ-координат, соответствующих столбцам J;
        phi: точка оценки (из jac.phi);
        loss: потеря в точке оценки (из jac.loss);
        J_shape: форма (n_residuals, n_params);
        collinearity_threshold: использованный порог коллинеарности;
        max_dirs: кап числа возвращаемых направлений в stiff/soft_dirs.
    """

    fim: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    eigenvalues: np.ndarray = field(default_factory=lambda: np.zeros(0))
    eigenvectors: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    singular_values: np.ndarray = field(default_factory=lambda: np.zeros(0))
    condition_number: float = np.inf
    condition_number_fim: float = np.inf
    effective_rank: int = 0
    energy: float = 0.99
    stiff_dirs: List[FIMDirection] = field(default_factory=list)
    soft_dirs: List[FIMDirection] = field(default_factory=list)
    collinearity: List[CollinearityPair] = field(default_factory=list)
    param_sensitivity: List[Tuple[str, float]] = field(default_factory=list)
    param_names: List[str] = field(default_factory=list)
    phi: np.ndarray = field(default_factory=lambda: np.zeros(0))
    loss: float = 0.0
    J_shape: Tuple[int, int] = (0, 0)
    collinearity_threshold: float = 0.95
    max_dirs: int = 10

    def __post_init__(self) -> None:
        self.fim = np.asarray(self.fim, dtype=np.float64)
        self.eigenvalues = np.asarray(self.eigenvalues, dtype=np.float64)
        self.eigenvectors = np.asarray(self.eigenvectors, dtype=np.float64)
        self.singular_values = np.asarray(self.singular_values, dtype=np.float64)
        self.phi = np.asarray(self.phi, dtype=np.float64)
        self.condition_number = float(self.condition_number)
        self.condition_number_fim = float(self.condition_number_fim)
        self.effective_rank = int(self.effective_rank)
        self.energy = float(self.energy)
        self.J_shape = (int(self.J_shape[0]), int(self.J_shape[1]))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fim": self.fim.tolist(),
            "eigenvalues": self.eigenvalues.tolist(),
            "eigenvectors": self.eigenvectors.tolist(),
            "singular_values": self.singular_values.tolist(),
            "condition_number": float(self.condition_number),
            "condition_number_fim": float(self.condition_number_fim),
            "effective_rank": int(self.effective_rank),
            "energy": float(self.energy),
            "stiff_dirs": [d.to_dict() for d in self.stiff_dirs],
            "soft_dirs": [d.to_dict() for d in self.soft_dirs],
            "collinearity": [p.to_dict() for p in self.collinearity],
            "param_sensitivity": [
                (str(n), float(v)) for n, v in self.param_sensitivity
            ],
            "param_names": [str(n) for n in self.param_names],
            "phi": self.phi.tolist(),
            "loss": float(self.loss),
            "J_shape": [int(self.J_shape[0]), int(self.J_shape[1])],
            "collinearity_threshold": float(self.collinearity_threshold),
            "max_dirs": int(self.max_dirs),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FIMResult":
        j_shape = data.get("J_shape", [0, 0])
        return cls(
            fim=np.asarray(data.get("fim", []), dtype=np.float64),
            eigenvalues=np.asarray(data.get("eigenvalues", []), dtype=np.float64),
            eigenvectors=np.asarray(data.get("eigenvectors", []), dtype=np.float64),
            singular_values=np.asarray(
                data.get("singular_values", []), dtype=np.float64
            ),
            condition_number=float(data.get("condition_number", np.inf)),
            condition_number_fim=float(data.get("condition_number_fim", np.inf)),
            effective_rank=int(data.get("effective_rank", 0)),
            energy=float(data.get("energy", 0.99)),
            stiff_dirs=[FIMDirection.from_dict(d) for d in data.get("stiff_dirs", [])],
            soft_dirs=[FIMDirection.from_dict(d) for d in data.get("soft_dirs", [])],
            collinearity=[
                CollinearityPair.from_dict(d) for d in data.get("collinearity", [])
            ],
            param_sensitivity=[
                (str(n), float(v)) for n, v in data.get("param_sensitivity", [])
            ],
            param_names=[str(n) for n in data.get("param_names", [])],
            phi=np.asarray(data.get("phi", []), dtype=np.float64),
            loss=float(data.get("loss", 0.0)),
            J_shape=(int(j_shape[0]), int(j_shape[1])),
            collinearity_threshold=float(data.get("collinearity_threshold", 0.95)),
            max_dirs=int(data.get("max_dirs", 10)),
        )


def compute_fim(
    jac: JacobianResult,
    *,
    energy: float = 0.99,
    param_names: Optional[Sequence[str]] = None,
    collinearity_threshold: float = 0.95,
    max_dirs: int = 10,
) -> FIMResult:
    """
    FIM-анализ по готовому якобиану невязок: F = JᵀJ.

    Якобиан НЕ вычисляется здесь — передайте готовый JacobianResult
    (например, из neuraltide.optimization.sensitivities.residual_jacobian).
    Невязки должны быть уже нормированы на sigma (тогда F — FIM гауссовой
    модели с известной Σ; множитель 1/σ² вносить не нужно).

    Args:
        jac: JacobianResult с J [n_residuals, n_params] (float64).
        energy: порог накопленной энергии спектра для effective_rank и
            разбиения stiff/soft (доля кривизны логарифма правдоподобия).
        param_names: имена φ-координат столбцов J (для top_params и
            param_sensitivity); None — 'phi[i]'.
        collinearity_threshold: порог |cos| между столбцами J (пары
            параметров с косинусом выше порога попадают в collinearity).
        max_dirs: кап числа направлений в stiff_dirs/soft_dirs
            (спектр может быть длинным; -1 — без капа).

    Returns:
        FIMResult: спектр, обусловленность, effective_rank, направления
        с вкладом параметров, коллинеарные пары, нормы столбцов J.
    """
    J = np.asarray(jac.J, dtype=np.float64)
    if J.ndim != 2:
        raise ValueError(
            f"compute_fim: ожидается двумерный J [n_residuals, n_params], "
            f"получена форма {J.shape}."
        )
    m, n = J.shape
    if not 0.0 < float(energy) < 1.0:
        raise ValueError(f"compute_fim: energy должен быть в (0, 1), got {energy}.")
    if float(collinearity_threshold) < 0.0 or float(collinearity_threshold) >= 1.0:
        raise ValueError(
            "compute_fim: collinearity_threshold должен быть в [0, 1), "
            f"got {collinearity_threshold}."
        )

    if param_names is None:
        names = [f"phi[{i}]" for i in range(n)]
    else:
        names = [str(x) for x in param_names]
        if len(names) != n:
            raise ValueError(
                f"compute_fim: param_names даёт {len(names)} имён, в J {n} столбцов."
            )

    # Спектр F: eigenvalues убывают, eigenvectors — правые сингулярные векторы.
    F = J.T @ J
    eigvals, eigvecs = np.linalg.eigh(F)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.clip(eigvals[order], 0.0, None)
    eigvecs = eigvecs[:, order]
    singular = np.sqrt(eigvals)

    total_energy = float(eigvals.sum())
    if total_energy > 0.0:
        fractions = eigvals / total_energy
        cumulative = np.cumsum(fractions)
        effective_rank = int(min(n, np.searchsorted(cumulative, energy) + 1))
    else:
        fractions = np.zeros(n, dtype=np.float64)
        cumulative = np.zeros(n, dtype=np.float64)
        effective_rank = 0

    # Числа обусловленности по ненулевым частям спектра.
    spec_tol = max(m, n) * np.finfo(np.float64).eps * (singular[0] if n else 0.0)
    s_nonzero = singular[singular > spec_tol]
    condition = float(singular[0] / s_nonzero[-1]) if s_nonzero.size else np.inf
    lam_nonzero = eigvals[eigvals > spec_tol**2]
    condition_fim = float(eigvals[0] / lam_nonzero[-1]) if lam_nonzero.size else np.inf

    def _make_direction(i: int) -> FIMDirection:
        v = eigvecs[:, i]
        top_idx = np.argsort(-np.abs(v))[:5]
        top = [(names[int(j)], float(v[int(j)])) for j in top_idx]
        return FIMDirection(
            index=i,
            singular_value=float(singular[i]),
            eigenvalue=float(eigvals[i]),
            energy_fraction=float(fractions[i]),
            cumulative_energy=float(cumulative[i]),
            top_params=top,
        )

    stiff = [_make_direction(i) for i in range(effective_rank)]
    soft = [_make_direction(i) for i in range(effective_rank, n)]
    cap = None if int(max_dirs) < 0 else int(max_dirs)
    if cap is not None:
        stiff = stiff[:cap]
        soft = soft[:cap]

    # Коллинеарность: |cos| между столбцами J (нулевые столбцы пропускаются).
    col_norms = np.linalg.norm(J, axis=0)
    valid = col_norms > 0.0
    pairs: List[CollinearityPair] = []
    if np.count_nonzero(valid) > 1:
        unit = np.zeros_like(J)
        unit[:, valid] = J[:, valid] / col_norms[valid]
        gram = unit.T @ unit
        for i in range(n):
            if not valid[i]:
                continue
            for j in range(i + 1, n):
                if not valid[j]:
                    continue
                cosine = float(gram[i, j])
                if abs(cosine) >= float(collinearity_threshold):
                    pairs.append(CollinearityPair(names[i], names[j], cosine))

    # Чувствительность: нормы столбцов J по убыванию.
    sensitivity = sorted(
        ((names[j], float(col_norms[j])) for j in range(n)),
        key=lambda item: -item[1],
    )

    return FIMResult(
        fim=F,
        eigenvalues=eigvals,
        eigenvectors=eigvecs,
        singular_values=singular,
        condition_number=condition,
        condition_number_fim=condition_fim,
        effective_rank=effective_rank,
        energy=float(energy),
        stiff_dirs=stiff,
        soft_dirs=soft,
        collinearity=pairs,
        param_sensitivity=sensitivity,
        param_names=names,
        phi=np.asarray(jac.phi, dtype=np.float64).reshape(-1),
        loss=float(jac.loss),
        J_shape=(int(m), int(n)),
        collinearity_threshold=float(collinearity_threshold),
        max_dirs=int(max_dirs),
    )


# ---------------------------------------------------------------------------
# DTO: ProfileResult
# ---------------------------------------------------------------------------


@dataclass
class ProfileResult(_JsonDTO):
    """
    Одномерный профиль правдоподобия по физическому θ параметра.

    Атрибуты:
        name: профилируемая запись ParameterSpace (full_name);
        theta_values: узлы сетки в физических θ (фиксированная сетка);
        losses: потеря после доводки остальных координат (NaN для
            неуспешных узлов);
        rel_loss: losses − min(losses[successes]) (NaN для неуспешных);
        successes: успешность доводки в узле;
        phi_path: полный вектор φ в узле (None для неуспешных);
        nfev: число оценок целевой функции в узле (0 для неуспешных);
        messages: сообщения решателя / причины отказа в узле;
        intervals: {_CONFIDENCE_KEY: (lo, hi) или None} — интервал по
            успешным узлам с порогом chi2_threshold/2 на rel_loss;
        chi2_threshold: χ²_{0.95}(1) ≈ 3.8414588 (порог на 2·ΔNLL);
        rel_loss_threshold: chi2_threshold/2 ≈ 1.9207 (порог на Δloss);
        warnings: диагностика профиля (немонотонность, несвязность
            допустимой области, минимум на границе сетки, ...);
        optimizer: использованный оптимизатор ('lbfgsb', 'least_squares',
            'trf', 'lm' или 'callable');
        warm_start: использован ли warm start от соседнего успешного узла;
        n_forward / n_gradient: дельта счётчиков задачи за профиль;
        wall_time: время построения профиля, секунды;
        notes: допущения (гауссов шум с известной sigma, профиль
            правдоподобия, фиксированная сетка).
    """

    name: str = ""
    theta_values: List[float] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    rel_loss: List[float] = field(default_factory=list)
    successes: List[bool] = field(default_factory=list)
    phi_path: List[Optional[np.ndarray]] = field(default_factory=list)
    nfev: List[int] = field(default_factory=list)
    messages: List[str] = field(default_factory=list)
    intervals: Dict[str, Optional[Tuple[float, float]]] = field(default_factory=dict)
    chi2_threshold: float = _CHI2_DF1_95
    rel_loss_threshold: float = _CHI2_DF1_95 / 2.0
    warnings: List[str] = field(default_factory=list)
    optimizer: str = "lbfgsb"
    warm_start: bool = True
    n_forward: int = 0
    n_gradient: int = 0
    wall_time: float = 0.0
    notes: str = ""

    def __post_init__(self) -> None:
        self.theta_values = [float(v) for v in self.theta_values]
        self.losses = [float(v) for v in self.losses]
        self.rel_loss = [float(v) for v in self.rel_loss]
        self.successes = [bool(v) for v in self.successes]
        self.phi_path = [
            None if p is None else np.asarray(p, dtype=np.float64)
            for p in self.phi_path
        ]
        self.nfev = [int(v) for v in self.nfev]
        self.messages = [str(v) for v in self.messages]
        self.intervals = {
            str(k): (None if v is None else (float(v[0]), float(v[1])))
            for k, v in self.intervals.items()
        }
        self.chi2_threshold = float(self.chi2_threshold)
        self.rel_loss_threshold = float(self.rel_loss_threshold)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": str(self.name),
            "theta_values": list(self.theta_values),
            "losses": list(self.losses),
            "rel_loss": list(self.rel_loss),
            "successes": list(self.successes),
            "phi_path": [None if p is None else p.tolist() for p in self.phi_path],
            "nfev": list(self.nfev),
            "messages": list(self.messages),
            "intervals": {
                k: (None if v is None else [float(v[0]), float(v[1])])
                for k, v in self.intervals.items()
            },
            "chi2_threshold": float(self.chi2_threshold),
            "rel_loss_threshold": float(self.rel_loss_threshold),
            "warnings": list(self.warnings),
            "optimizer": str(self.optimizer),
            "warm_start": bool(self.warm_start),
            "n_forward": int(self.n_forward),
            "n_gradient": int(self.n_gradient),
            "wall_time": float(self.wall_time),
            "notes": str(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProfileResult":
        return cls(
            name=str(data.get("name", "")),
            theta_values=list(data.get("theta_values", [])),
            losses=list(data.get("losses", [])),
            rel_loss=list(data.get("rel_loss", [])),
            successes=list(data.get("successes", [])),
            phi_path=[
                None if p is None else np.asarray(p, dtype=np.float64)
                for p in data.get("phi_path", [])
            ],
            nfev=list(data.get("nfev", [])),
            messages=list(data.get("messages", [])),
            intervals={
                str(k): (None if v is None else (float(v[0]), float(v[1])))
                for k, v in data.get("intervals", {}).items()
            },
            chi2_threshold=float(data.get("chi2_threshold", _CHI2_DF1_95)),
            rel_loss_threshold=float(
                data.get("rel_loss_threshold", _CHI2_DF1_95 / 2.0)
            ),
            warnings=list(data.get("warnings", [])),
            optimizer=str(data.get("optimizer", "lbfgsb")),
            warm_start=bool(data.get("warm_start", True)),
            n_forward=int(data.get("n_forward", 0)),
            n_gradient=int(data.get("n_gradient", 0)),
            wall_time=float(data.get("wall_time", 0.0)),
            notes=str(data.get("notes", "")),
        )


def _profile_interval(
    theta_values: np.ndarray,
    rel_loss: np.ndarray,
    successes: np.ndarray,
    threshold: float,
) -> Tuple[Optional[Tuple[float, float]], List[str]]:
    """
    Интервал по успешным узлам и предупреждения о форме профиля.

    Интервал — (min, max) узлов связной допустимой области (rel_loss <=
    threshold), содержащей глобальный минимум профиля. Немонотонность,
    несвязность допустимой области и минимум на границе сетки фиксируются
    в предупреждениях и не скрываются.
    """
    warnings: List[str] = []
    if not np.any(successes):
        return None, ["нет успешных узлов — интервал не определён"]
    x = np.asarray(theta_values, dtype=np.float64)[successes]
    y = np.asarray(rel_loss, dtype=np.float64)[successes]
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]

    i_min = int(np.argmin(y))
    qualifying = y <= float(threshold)
    if not np.any(qualifying):
        warnings.append(
            "ни один успешный узел не проходит порог χ²/2 — интервал не "
            "определён (сетка не покрывает допустимую область?)"
        )
        return None, warnings

    lo = i_min
    while lo - 1 >= 0 and qualifying[lo - 1]:
        lo -= 1
    hi = i_min
    while hi + 1 < qualifying.size and qualifying[hi + 1]:
        hi += 1
    in_run = np.zeros_like(qualifying)
    in_run[lo : hi + 1] = True
    if np.any(qualifying & ~in_run):
        warnings.append(
            "допустимая область профиля несвязна — возможен ложный подъём "
            "или мультимодальность; интервал построен по связной компоненте "
            "глобального минимума"
        )

    extra_minima = 0
    for i in range(1, y.size - 1):
        if i == i_min:
            continue
        if y[i] < y[i - 1] and y[i] <= y[i + 1]:
            extra_minima += 1
    if extra_minima > 0:
        warnings.append(
            f"профиль немонотонен ({extra_minima} доп. лок. мин.) — возможна "
            "локальная несходимость оптимизатора; узлы не отбрасываются"
        )
    if i_min == 0 or i_min == y.size - 1:
        warnings.append(
            "минимум профиля на границе сетки — расширьте grid, интервал "
            "может быть усечён"
        )
    return (float(x[lo]), float(x[hi])), warnings


def _resolve_profile_optimizer(optimizer: Any) -> Tuple[Any, str]:
    """Строковый идентификатор/ callable → callable(frozen, phi0) -> result."""
    if callable(optimizer):
        return optimizer, "callable"
    key = str(optimizer).lower()
    if key in ("lbfgsb", "l-bfgs-b"):

        def _run_lbfgsb(frozen, phi0):
            return minimize_lbfgsb(frozen, phi0, validate=False)

        return _run_lbfgsb, "lbfgsb"
    if key in ("least_squares", "trf", "lm"):
        method = "lm" if key == "lm" else "trf"

        def _run_ls(frozen, phi0, method=method):
            return least_squares(frozen, phi0, method=method, validate=False)

        return _run_ls, key
    raise ValueError(
        f"likelihood_profile: неизвестный optimizer={optimizer!r}; доступны "
        "'lbfgsb', 'least_squares', 'trf', 'lm' или callable(frozen, phi0) "
        "-> результат с полями success/base.phi_best/base.loss_best."
    )


def _default_profile_grid(entry, theta_current: float, n_points: int) -> np.ndarray:
    """
    Сетка по умолчанию (эвристика): линия по [min, max] записи, если обе
    границы заданы, иначе центр в текущем θ с полушириной max(|θ|, 1).
    Для реального анализа grid задаётся явно.
    """
    n_points = int(n_points)
    if n_points < 2:
        raise ValueError(
            f"likelihood_profile: n_points должен быть >= 2, got {n_points}."
        )
    if entry.min is not None and entry.max is not None:
        return np.linspace(float(entry.min), float(entry.max), n_points)
    span = max(abs(float(theta_current)), 1.0)
    return float(theta_current) + span * np.linspace(-1.0, 1.0, n_points)


def likelihood_profile(
    problem,
    name: str,
    *,
    phi: Optional[Any] = None,
    values: Optional[Sequence[float]] = None,
    n_points: int = 11,
    grid: Optional[Sequence[float]] = None,
    optimizer: Any = "lbfgsb",
    warm_start: bool = True,
) -> ProfileResult:
    """
    Одномерный профиль правдоподобия параметра name (full_name ParameterSpace).

    В каждом узле фиксированной сетки (в физических θ) координата name
    замораживается через FrozenProblem (φ-значение = transform.inverse(θ)),
    остальные координаты доводятся локальным оптимизатором. Сетка
    фиксированная, не адаптивная (P1).

    Args:
        problem: CalibrationProblem (или duck-typed аналог с
            parameter_space/pack/bounds_phi/counters).
        name: full_name скалярной записи ParameterSpace (numel == 1 в P1).
        phi: стартовый вектор φ — точка отсчёта профиля: cold start для
            свободных координат (при warm_start=False) и значения
            непрофилируемых координат в phi_path. None — problem.pack()
            (текущее состояние модели). Если pack() невозможен (в
            ParameterSpace есть tensor-backed записи без переменных
            модели, например начальное состояние), передайте phi явно —
            иначе ValueError с пояснением.
        values: узлы сетки в физических θ (псевдоним grid; оба сразу —
            ValueError). None — grid, иначе эвристическая сетка (см.
            _default_profile_grid).
        n_points: число узлов эвристической сетки (игнорируется при
            явном grid/values).
        grid: узлы сетки в физических θ (то же, что values).
        optimizer: 'lbfgsb' (minimize_lbfgsb), 'least_squares'/'trf',
            'lm' (least_squares) или callable(frozen, phi0) -> результат
            с полями success/base.phi_best/base.loss_best (как
            LocalOptimizerResult).
        warm_start: стартовать узел от решения предыдущего успешного узла
            (reduced φ); иначе — от исходной точки профиля (cold start).

    Returns:
        ProfileResult с theta_values/losses/rel_loss/successes/phi_path,
        intervals['0.95'] по успешным узлам (χ²-порог 3.8414588 на 2·ΔNLL,
        т.е. 1.9207 на Δloss; допущение — гауссов шум с известной sigma,
        нормировка уже в residual) и warnings о форме профиля.

    Raises:
        ValueError: неизвестная запись/сетка/optimizer, а также когда
            phi=None, но problem.pack() невозможен (tensor-backed записи) —
            сообщение предлагает передать phi явно.

    Счётчики n_forward/n_gradient — дельта счётчиков задачи за построение
    профиля. Контракт мутации модели наследуется от CalibrationProblem:
    модель остаётся в точке последней оценённой θ.
    """
    space = problem.parameter_space
    entries = space.entries
    if name not in entries:
        raise ValueError(
            f"likelihood_profile: неизвестная запись '{name}'; доступны "
            f"{sorted(entries)}."
        )
    entry = entries[name]
    if entry.numel != 1:
        raise ValueError(
            f"likelihood_profile: P1-профили поддерживают только скалярные "
            f"записи (numel == 1); '{name}' имеет numel={entry.numel}. "
            "Задайте одну φ-координату явно."
        )
    if values is not None and grid is not None:
        raise ValueError(
            "likelihood_profile: заданы одновременно values и grid — "
            "оставьте один (это псевдонимы)."
        )
    nodes = grid if grid is not None else values
    if nodes is None:
        if entry.variable is None:
            raise ValueError(
                f"likelihood_profile: запись '{name}' без переменной модели — "
                "эвристическая сетка невозможна, задайте grid/values явно."
            )
        theta_current = float(
            np.asarray(entry.variable, dtype=np.float64).reshape(-1)[0]
        )
        nodes = _default_profile_grid(entry, theta_current, n_points)
    nodes = np.asarray(nodes, dtype=np.float64).reshape(-1)
    if nodes.size < 2:
        raise ValueError(
            f"likelihood_profile: сетка должна содержать >= 2 узлов, "
            f"получено {nodes.size}."
        )
    if not np.all(np.isfinite(nodes)):
        raise ValueError("likelihood_profile: сетка содержит нефинитные узлы.")

    run, optimizer_tag = _resolve_profile_optimizer(optimizer)
    if optimizer_tag == "lm":
        lb, ub = problem.bounds_phi()
        if np.any(np.isfinite(lb)) or np.any(np.isfinite(ub)):
            raise ValueError(
                "likelihood_profile: optimizer='lm' не поддерживает bounds "
                "— используйте 'trf'/'least_squares' или уберите ограничения."
            )

    dtype = neuraltide.config.get_dtype()
    offsets = _entry_offsets(space)
    free_names = [n for n in entries if n != name]
    free_index = _free_index(offsets, free_names)
    fix_slice = offsets[name][0]

    if phi is None:
        try:
            phi_base = np.asarray(problem.pack(), dtype=np.float64).reshape(-1)
        except ValueError as exc:
            raise ValueError(
                "likelihood_profile: problem.pack() невозможен — в "
                "ParameterSpace есть записи без переменных модели "
                "(tensor-backed, например начальное состояние), и "
                "стартовую точку профиля не из чего собрать. Передайте её "
                "явно через phi=... (полный вектор φ длины problem.size)."
            ) from exc
    else:
        phi_base = np.asarray(problem.pack(phi), dtype=np.float64).reshape(-1)
    cold_start = phi_base[free_index]

    t_start = time.perf_counter()
    fwd0, grad0 = _counters(problem)

    theta_values: List[float] = []
    losses: List[float] = []
    successes: List[bool] = []
    phi_path: List[Optional[np.ndarray]] = []
    nfev: List[int] = []
    messages: List[str] = []
    warm_ref: Optional[np.ndarray] = None

    for node in nodes:
        theta_values.append(float(node))
        # φ-значение замороженной координаты; нефинитный transform (например
        # log от θ <= 0) — честный отказ узла, а не исключение профиля.
        phi_fixed = float(
            np.asarray(
                entry.transform.inverse(tf.constant(float(node), dtype)),
                dtype=np.float64,
            ).reshape(-1)[0]
        )
        if not np.isfinite(phi_fixed):
            losses.append(float("nan"))
            successes.append(False)
            phi_path.append(None)
            nfev.append(0)
            messages.append(f"transform.inverse({node}) нефинитен — узел отклонён")
            continue
        frozen = FrozenProblem(
            problem, free_names=free_names, fixed={name: float(node)}
        )
        start = warm_ref if (warm_start and warm_ref is not None) else cold_start
        try:
            res = run(frozen, start)
            loss_node = float(res.base.loss_best)
            ok = bool(res.success) and np.isfinite(loss_node)
            message = str(getattr(res, "message", ""))
            nfev.append(int(getattr(res, "nfev", 0) or 0))
        except Exception as exc:  # честная фиксация отказа узла
            loss_node = float("nan")
            ok = False
            message = f"{type(exc).__name__}: {exc}"
            nfev.append(0)
        losses.append(loss_node)
        successes.append(ok)
        messages.append(message)
        if ok:
            phi_full = phi_base.copy()
            phi_full[free_index] = np.asarray(
                res.base.phi_best, dtype=np.float64
            ).reshape(-1)
            phi_full[fix_slice] = phi_fixed
            phi_path.append(phi_full)
            warm_ref = np.asarray(res.base.phi_best, dtype=np.float64).reshape(-1)
        else:
            phi_path.append(None)

    losses_arr = np.asarray(losses, dtype=np.float64)
    ok_arr = np.asarray(successes, dtype=bool)
    if np.any(ok_arr):
        loss_min = float(np.min(losses_arr[ok_arr]))
        rel_loss = [
            (losses_arr[i] - loss_min) if ok_arr[i] else float("nan")
            for i in range(losses_arr.size)
        ]
    else:
        rel_loss = [float("nan")] * losses_arr.size

    interval, interval_warnings = _profile_interval(
        np.asarray(theta_values, dtype=np.float64),
        np.asarray(rel_loss, dtype=np.float64),
        ok_arr,
        _CHI2_DF1_95 / 2.0,
    )
    fwd1, grad1 = _counters(problem)
    notes = (
        "Профиль правдоподобия по физическому θ на фиксированной сетке; "
        "loss = NLL − const = 0.5·||r||² с известной sigma (нормировка уже "
        f"в residual). Порог интервала: 2·ΔNLL ~ χ²(1), χ²_0.95 = "
        f"{_CHI2_DF1_95:.6f}, допуск на rel_loss = χ²/2 = "
        f"{_CHI2_DF1_95 / 2.0:.6f}; df = 1 (один параметр). Интервалы — по "
        "связной области успешных узлов, содержащей минимум; для "
        "huber/robust-видов штрафа интервалы эвристичны. Неуспешные узлы: "
        "losses=NaN, successes=False, phi_path=None."
    )
    return ProfileResult(
        name=str(name),
        theta_values=theta_values,
        losses=losses,
        rel_loss=rel_loss,
        successes=successes,
        phi_path=phi_path,
        nfev=nfev,
        messages=messages,
        intervals={_CONFIDENCE_KEY: interval},
        chi2_threshold=_CHI2_DF1_95,
        rel_loss_threshold=_CHI2_DF1_95 / 2.0,
        warnings=interval_warnings,
        optimizer=optimizer_tag,
        warm_start=bool(warm_start),
        n_forward=fwd1 - fwd0,
        n_gradient=grad1 - grad0,
        wall_time=time.perf_counter() - t_start,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# DTO: SubsetResult и select_subset
# ---------------------------------------------------------------------------


@dataclass
class SubsetResult(_JsonDTO):
    """
    Результат выбора подмножества параметров с совместной доводкой.

    Атрибуты:
        kept_names: доводимые записи (full_name);
        fixed_names: замороженные записи;
        refit_result: результат minimize_lbfgsb над FrozenProblem (None при
            refit=False или при отказе оптимизатора до возврата результата);
        loss_before: потеря в точке якобиана (jac.loss);
        loss_after: потеря после доводки (None при refit=False);
        theta_delta_kept: {name: θ_refit − θ_jac} для доводимых записей
            (массивы в исходной форме записи; None при refit=False);
        theta_delta_fixed: {name: θ_refit − θ_jac} для замороженных записей —
            структурно 0, т.к. FrozenProblem фиксирует их на значениях
            точки якобиана (честно вычислено из phi_full, не hardcoded);
        prediction_delta: {population: ||r_after − r_before|| / (||r_before||
            + eps)} по firing_rates обучающего протокола; None, если
            prediction-прогон недоступен (нет simulate) или завершился
            ошибкой — причина в notes;
        validation_delta: {experiment: loss_after − loss_before} на held-out
            протоколах; None при их отсутствии/недоступности — в notes;
        notes: честные оговорки (structural zero фиксированных, доступность
            prediction/validation, статус refit);
        n_forward / n_gradient: дельта счётчиков задачи за refit;
        wall_time: время refit, секунды.
    """

    kept_names: List[str] = field(default_factory=list)
    fixed_names: List[str] = field(default_factory=list)
    refit_result: Optional[LocalOptimizerResult] = None
    loss_before: float = 0.0
    loss_after: Optional[float] = None
    theta_delta_kept: Optional[Dict[str, np.ndarray]] = None
    theta_delta_fixed: Optional[Dict[str, np.ndarray]] = None
    prediction_delta: Optional[Dict[str, float]] = None
    validation_delta: Optional[Dict[str, float]] = None
    notes: str = ""
    n_forward: int = 0
    n_gradient: int = 0
    wall_time: float = 0.0

    def __post_init__(self) -> None:
        self.kept_names = [str(n) for n in self.kept_names]
        self.fixed_names = [str(n) for n in self.fixed_names]
        self.loss_before = float(self.loss_before)
        self.loss_after = None if self.loss_after is None else float(self.loss_after)
        for attr in ("theta_delta_kept", "theta_delta_fixed"):
            value = getattr(self, attr)
            if value is not None:
                setattr(
                    self,
                    attr,
                    {str(k): np.asarray(v, dtype=np.float64) for k, v in value.items()},
                )
        self.prediction_delta = (
            None
            if self.prediction_delta is None
            else {str(k): float(v) for k, v in self.prediction_delta.items()}
        )
        self.validation_delta = (
            None
            if self.validation_delta is None
            else {str(k): float(v) for k, v in self.validation_delta.items()}
        )

    def to_dict(self) -> Dict[str, Any]:
        def _encode_delta(delta):
            if delta is None:
                return None
            return {
                str(k): np.asarray(v, dtype=np.float64).tolist()
                for k, v in delta.items()
            }

        return {
            "kept_names": list(self.kept_names),
            "fixed_names": list(self.fixed_names),
            "refit_result": (
                None if self.refit_result is None else self.refit_result.to_dict()
            ),
            "loss_before": float(self.loss_before),
            "loss_after": self.loss_after,
            "theta_delta_kept": _encode_delta(self.theta_delta_kept),
            "theta_delta_fixed": _encode_delta(self.theta_delta_fixed),
            "prediction_delta": (
                None
                if self.prediction_delta is None
                else {str(k): float(v) for k, v in self.prediction_delta.items()}
            ),
            "validation_delta": (
                None
                if self.validation_delta is None
                else {str(k): float(v) for k, v in self.validation_delta.items()}
            ),
            "notes": str(self.notes),
            "n_forward": int(self.n_forward),
            "n_gradient": int(self.n_gradient),
            "wall_time": float(self.wall_time),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SubsetResult":
        def _decode_delta(delta):
            if delta is None:
                return None
            return {str(k): np.asarray(v, dtype=np.float64) for k, v in delta.items()}

        refit = data.get("refit_result")
        return cls(
            kept_names=list(data.get("kept_names", [])),
            fixed_names=list(data.get("fixed_names", [])),
            refit_result=(
                None if refit is None else LocalOptimizerResult.from_dict(refit)
            ),
            loss_before=float(data.get("loss_before", 0.0)),
            loss_after=data.get("loss_after"),
            theta_delta_kept=_decode_delta(data.get("theta_delta_kept")),
            theta_delta_fixed=_decode_delta(data.get("theta_delta_fixed")),
            prediction_delta=(
                None
                if data.get("prediction_delta") is None
                else {str(k): float(v) for k, v in data["prediction_delta"].items()}
            ),
            validation_delta=(
                None
                if data.get("validation_delta") is None
                else {str(k): float(v) for k, v in data["validation_delta"].items()}
            ),
            notes=str(data.get("notes", "")),
            n_forward=int(data.get("n_forward", 0)),
            n_gradient=int(data.get("n_gradient", 0)),
            wall_time=float(data.get("wall_time", 0.0)),
        )


def select_subset(
    problem,
    jac: JacobianResult,
    keep: Sequence[str],
    *,
    refit: bool = True,
) -> SubsetResult:
    """
    Совместная доводка подмножества параметров с оценкой смещения.

    Доводимые координаты keep доводятся minimize_lbfgsb поверх FrozenProblem
    (остальные записи заморожены на физических θ из точки якобиана jac.phi).
    Якобиан здесь не вычисляется — передайте готовый JacobianResult
    (например, из sensitivities.residual_jacobian).

    Args:
        problem: CalibrationProblem (или duck-typed аналог с
            parameter_space/pack/simulate/validate/counters).
        jac: JacobianResult в исходной точке (phi/loss/J); phi задаёт точку,
            относительно которой измеряется смещение.
        keep: полные имена доводимых записей ParameterSpace (непусто).
        refit: выполнять доводку; False — только разбиение kept/fixed, поля
            смещения None с пояснением в notes.

    Returns:
        SubsetResult: kept/fixed имена, refit_result, theta_delta_kept/
        theta_delta_fixed (последние структурно нулевые — FrozenProblem),
        loss_before/loss_after, prediction_delta по firing_rates обучающего
        протокола и validation_delta по held-out протоколам (None при
        недоступности — причина честно указана в notes).

    Счётчики — дельта счётчиков задачи за refit. Контракт мутации модели
    наследуется: модель остаётся в точке последней оценённой θ (включая
    prediction/validation прогоны).
    """
    space = problem.parameter_space
    entries = space.entries
    kept = [str(k) for k in keep]
    if not kept:
        raise ValueError("select_subset: keep должен содержать хотя бы одно имя.")
    unknown = [k for k in kept if k not in entries]
    if unknown:
        raise ValueError(
            f"select_subset: неизвестные записи {unknown}; доступны {sorted(entries)}."
        )
    fixed_names = [n for n in entries if n not in kept]

    phi_jac = np.asarray(jac.phi, dtype=np.float64).reshape(-1)
    if phi_jac.size != space.size:
        raise ValueError(
            f"select_subset: jac.phi имеет {phi_jac.size} элементов, "
            f"ParameterSpace — {space.size}."
        )
    loss_before = float(jac.loss)
    theta_jac = {
        n: np.asarray(space.unpack(phi_jac)[n], dtype=np.float64) for n in entries
    }

    if not refit:
        notes = (
            "refit=False: доводка не выполнялась — kept/fixed зафиксированы, "
            "loss_after/theta_delta_*/prediction_delta/validation_delta = None. "
            "Точка отсчёта смещения — jac.phi (loss_before = jac.loss)."
        )
        return SubsetResult(
            kept_names=kept,
            fixed_names=fixed_names,
            refit_result=None,
            loss_before=loss_before,
            loss_after=None,
            notes=notes,
        )

    offsets = _entry_offsets(space)
    free_index = _free_index(offsets, kept)
    phi0_free = phi_jac[free_index]

    t_start = time.perf_counter()
    fwd0, grad0 = _counters(problem)
    frozen = FrozenProblem(
        problem,
        free_names=kept,
        fixed={n: theta_jac[n] for n in fixed_names},
    )
    notes_parts = [
        "Замороженные записи зафиксированы FrozenProblem на физических θ из "
        "jac.phi, поэтому theta_delta_fixed структурно нулевой (это "
        "свойство постановки, значения вычислены, не hardcoded)."
    ]
    try:
        refit_result = minimize_lbfgsb(frozen, phi0_free, validate=False)
        phi_full = phi_jac.copy()
        phi_full[free_index] = np.asarray(
            refit_result.base.phi_best, dtype=np.float64
        ).reshape(-1)
        loss_after = float(refit_result.base.loss_best)
        theta_after = {
            n: np.asarray(space.unpack(phi_full)[n], dtype=np.float64) for n in entries
        }
        theta_delta_kept = {n: theta_after[n] - theta_jac[n] for n in kept}
        theta_delta_fixed = {n: theta_after[n] - theta_jac[n] for n in fixed_names}
        notes_parts.append(
            f"refit: minimize_lbfgsb success={bool(refit_result.success)} "
            f"({refit_result.message}); loss {loss_before:.6g} -> "
            f"{loss_after:.6g}."
        )
    except Exception as exc:
        refit_result = None
        phi_full = None
        loss_after = None
        theta_delta_kept = None
        theta_delta_fixed = None
        notes_parts.append(
            f"refit отклонён до возврата результата: {type(exc).__name__}: "
            f"{exc}; bias-поля None."
        )

    prediction_delta: Optional[Dict[str, float]] = None
    if phi_full is not None:
        simulate = getattr(problem, "simulate", None)
        if not callable(simulate):
            notes_parts.append(
                "prediction_delta=None: problem не предоставляет simulate(phi) "
                "— прогнозный сдвиг не измерен."
            )
        else:
            try:
                rates_before = simulate(phi_jac).firing_rates
                rates_after = simulate(phi_full).firing_rates
                prediction_delta = {}
                for pop in rates_before:
                    if pop not in rates_after:
                        continue
                    before = np.asarray(rates_before[pop], dtype=np.float64)
                    after = np.asarray(rates_after[pop], dtype=np.float64)
                    denom = max(float(np.linalg.norm(before)), _NORM_EPS)
                    prediction_delta[str(pop)] = float(
                        np.linalg.norm(after - before) / denom
                    )
                if not prediction_delta:
                    prediction_delta = None
                    notes_parts.append(
                        "prediction_delta=None: firing_rates пусты или "
                        "несогласованы между прогонами."
                    )
            except Exception as exc:
                prediction_delta = None
                notes_parts.append(
                    f"prediction_delta=None: прогон прогноза завершился "
                    f"ошибкой ({type(exc).__name__}: {exc})."
                )
    else:
        notes_parts.append(
            "prediction_delta=None: refit не дал точки для прогнозного прогона."
        )

    validation_delta: Optional[Dict[str, float]] = None
    if phi_full is None:
        notes_parts.append("validation_delta=None: refit не дал точки для validate().")
    else:
        held_out = getattr(problem, "held_out", None)
        validate = getattr(problem, "validate", None)
        if not held_out or not callable(validate):
            notes_parts.append(
                "validation_delta=None: held-out протоколы отсутствуют "
                "(problem.held_out пуст) — validation сдвиг не измерен; "
                "bias оценен только на обучающем протоколе."
            )
        else:
            try:
                val_before = validate(phi_jac)
                val_after = validate(phi_full)
                validation_delta = {
                    str(k): float(val_after[k]) - float(val_before[k])
                    for k in val_before
                }
            except Exception as exc:
                validation_delta = None
                notes_parts.append(
                    f"validation_delta=None: validate() завершился ошибкой "
                    f"({type(exc).__name__}: {exc})."
                )

    fwd1, grad1 = _counters(problem)
    return SubsetResult(
        kept_names=kept,
        fixed_names=fixed_names,
        refit_result=refit_result,
        loss_before=loss_before,
        loss_after=loss_after,
        theta_delta_kept=theta_delta_kept,
        theta_delta_fixed=theta_delta_fixed,
        prediction_delta=prediction_delta,
        validation_delta=validation_delta,
        notes=" ".join(notes_parts),
        n_forward=fwd1 - fwd0,
        n_gradient=grad1 - grad0,
        wall_time=time.perf_counter() - t_start,
    )
