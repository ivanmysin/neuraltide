"""
Uncertainty quantification этапа P2 (дорожная карта, раздел 7.4).

Поверх CalibrationProblem / ParameterSpace / JacobianResult (P0/P1):

- residual_diagnostics — разложение вектора невязок по ObservationSpec
  ResidualLoss: per-spec mean/std/max|.|, окна времени, автокорреляция
  лагов 1..k по временному ряду (среднее по юнитам/бэтчу), гетероскедастич-
  ность по временным бинам (Levene/Brown–Forsythe); маскированные элементы
  исключаются из пулов; без метаданных observations — fallback на один
  псевдо-spec по плоскому вектору;
- laplace_approximation — гауссово приближение постериора в точке MAP.
  Всё считается в единицах полного NLL (ResidualLoss по умолчанию делит
  вклад на N, это учитывается через JacobianResult.likelihood_scale и
  objectives.likelihood_normalization).
  method='gn': covariance = (JᵀJ + P)⁻¹, J — готовый jac или результат
  sensitivities.residual_jacobian (max_params=200, ленивый импорт);
  method='fd-hessian' — только как явно документированный sanity-check:
  центральные конечные разности градиента потери (2n прогонов с градиентом,
  O(h²) погрешность) с жёстким капом max_fd_params=50. PD-проверка через
  Cholesky с нарастающим jitter и флагом pd; eigvals/std; маргинальные
  интервалы по именам ParameterSpace (в физических θ и в φ-координатах);
- bootstrap_experiments — для MultiExperimentProblem: пары-bootstrap на
  уровне экспериментов (подвыборка с повторением, counts весов в draw;
  копии draw получают уникальные служебные имена
  'boot_{draw}_{position}_{original_name}', исходные Experiment не
  мутируют; пространства с exp-specific x0-записями отклоняются — такие
  записи на служебные имена не переносятся); для одиночной
  CalibrationProblem: байесовский bootstrap случайных весов
  (w ~ Exp(1) на элементах невязок, loss_draw = 0.5·Σ wᵢ rᵢ²). Каждая draw
  стартует с явно захваченного phi_ref — мутация модели чужими рефитами
  не используется; evaluator — callable(problem, phi0) или строковое имя
  локального оптимизатора (ленивая резолюция, без import-зависимости);
  seeds через np.random.SeedSequence (детерминированность по seed);
- residual_bootstrap — target* = pred − e* с ресемплированными
  центрированными ошибками модели; инверсия residual-нормировки точна для
  kind 'l2'/'relative'/'msle'/'huber'/'pseudo_huber' без readout; честные
  оговорки — в notes;
- parameter_ensemble — phi/theta-выборки из готовых результатов
  оптимизации, mean/std, cov при n_samples >= 2;
- predictive_intervals — прогоны по phi_samples, per-(population, time)
  квантили уровня level и центральная (медианная) траектория; квантили
  монотонны по построению; нефинитные phi пропускаются с записью;
- noise_sensitivity — пересчёт ковариации Laplace при σ → f·σ:
  covariance(f) = (G/f² + P)⁻¹, G = JᵀJ (precision_gn); для плоского
  приора std ∝ 1/σ точно, с приором масштабируется только GN-часть.

DTO JSON-сериализуемы (to_dict/from_dict/save/load, нефинитные значения
кодируются null). Eager-режим, numpy-выходы float64, batch=1 (контракт
NetworkRNN). Глобальный dtype (neuraltide.config) используется только при
взаимодействии с ParameterSpace/трансформами и Objective. Координаты φ,
имена параметров и интервалы строятся только по trainable-записям
ParameterSpace (``phi_entries``/``trainable_names``): ручные non-trainable
записи реестра вне вектора φ и в статистику не входят.
"""

import time
import warnings
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
from scipy import stats as sp_stats
from scipy.linalg import cho_solve

import neuraltide.config
from neuraltide.core.types import cast_if_needed
from neuraltide.optimization.objectives import likelihood_normalization
from neuraltide.optimization.results import JacobianResult, _JsonDTO

__all__ = [
    "SpecResidualDiagnostics",
    "WindowDiagnostics",
    "ResidualDiagnostics",
    "residual_diagnostics",
    "LaplaceResult",
    "laplace_approximation",
    "BootstrapResult",
    "bootstrap_experiments",
    "residual_bootstrap",
    "EnsembleResult",
    "parameter_ensemble",
    "PredictiveResult",
    "predictive_intervals",
    "NoiseFactorResult",
    "SensitivityResult",
    "noise_sensitivity",
]

# z-квантиль для двустороннего интервала уровня 0.95.
_Z95 = float(sp_stats.norm.ppf(0.975))

# Нарастающий множитель jitter для PD-проверки precision-матрицы.
_JITTER_STEPS = (0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4)

# Жёсткий кап размерности для режима 'fd-hessian'.
_FD_HESSIAN_MAX_PARAMS = 50

# Допуск симметрии prior_cov.
_PRIOR_SYM_RTOL = 1e-8


# ---------------------------------------------------------------------------
# Внутренние helpers
# ---------------------------------------------------------------------------


def _entry_offsets(space) -> Dict[str, Tuple[int, int]]:
    """Смещения trainable-записей ParameterSpace в плоском векторе φ."""
    offsets: Dict[str, Tuple[int, int]] = {}
    start = 0
    for name, entry in space.phi_entries.items():
        offsets[str(name)] = (start, entry.numel)
        start += entry.numel
    return offsets


def _flat_names(space, size: int) -> List[str]:
    """
    Имена φ-координат: full_name для скалярных trainable-записей,
    'name[i]' иначе (non-trainable записи реестра в φ не входят).
    """
    names: List[str] = []
    for name, entry in space.phi_entries.items():
        if entry.numel == 1:
            names.append(str(name))
        else:
            names.extend(f"{name}[{i}]" for i in range(entry.numel))
    if len(names) != size:
        raise ValueError(
            f"uq: ParameterSpace даёт {len(names)} координат, ожидается "
            f"{size} (несовпадение размерности φ)."
        )
    return names


def _counters(problem) -> Tuple[int, int]:
    return (
        int(getattr(problem, "n_forward", 0)),
        int(getattr(problem, "n_gradient", 0)),
    )


def _pack_current(problem, where: str) -> np.ndarray:
    """
    Текущий φ задачи (problem.pack()) с понятной ошибкой.

    pack(None) невозможен для пространств с tensor-записями (начальное
    состояние) — тогда ошибка подсказывает привести модель в нужное
    состояние через phi-аргумент вызывающей функции.
    """
    try:
        return np.asarray(problem.pack(), dtype=np.float64).reshape(-1)
    except ValueError as exc:
        raise ValueError(
            f"{where}: стартовую точку нельзя получить через problem.pack() "
            f"({exc}). Примените желаемые θ к модели заранее или "
            "используйте ParameterSpace без tensor-записей."
        ) from exc


def _resolve_evaluator(evaluator) -> Tuple[Callable, str]:
    """
    Приведение evaluator к callable(problem, phi0) -> результат.

    None → 'lbfgsb'; строка — ленивая резолюция через
    strategies.resolve_local_optimizer (без import-зависимости на этапе
    импорта модуля); callable — используется как есть.
    """
    if evaluator is None:
        evaluator = "lbfgsb"
    if callable(evaluator):
        return evaluator, str(getattr(evaluator, "__name__", repr(evaluator)))
    key = str(evaluator).lower()
    if key in ("lbfgsb", "least_squares"):
        from neuraltide.optimization.strategies import resolve_local_optimizer

        return resolve_local_optimizer(key)
    raise ValueError(
        f"uq: неизвестный evaluator={evaluator!r}; доступны 'lbfgsb', "
        "'least_squares' или callable(problem, phi0) -> результат с полями "
        "base.phi_best / base.loss_best."
    )


def _clean_tree(value: Any) -> Any:
    """Рекурсивно: нефинитные float -> None (JSON null), numpy -> list."""
    if isinstance(value, np.ndarray):
        return _clean_tree(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(k): _clean_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_tree(v) for v in value]
    return value


def _f(value: Any) -> Optional[float]:
    """float либо None, если нефинитен."""
    v = float(value)
    return v if np.isfinite(v) else None


def _symmetrize(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + a.T)


def _pd_inverse(precision: np.ndarray) -> Tuple[np.ndarray, float, bool]:
    """
    Обращение precision-матрицы с PD-гарантией через Cholesky.

    Пробуется нарастающий jitter (множитель от максимальной диагонали);
    при неудаче на всех ступенях — fallback через клиппинг спектра
    (флаг pd=False). Возвращает (covariance, jitter_used, pd); jitter_used
    нефинитен при fallback.
    """
    a = _symmetrize(np.asarray(precision, dtype=np.float64))
    n = a.shape[0]
    eye = np.eye(n, dtype=np.float64)
    scale = max(1.0, float(np.max(np.diag(a))) if n else 1.0)
    for jitter in _JITTER_STEPS:
        aj = a + (jitter * scale) * eye
        try:
            chol = np.linalg.cholesky(aj)
        except np.linalg.LinAlgError:
            continue
        cov = cho_solve((chol, True), eye)
        return _symmetrize(cov), float(jitter * scale), True
    # Fallback: спектральный клиппинг отрицательных собственных значений.
    eigvals, eigvecs = np.linalg.eigh(a)
    floor = scale * 1e-10
    clipped = np.clip(eigvals, floor, None)
    cov = (eigvecs / clipped) @ eigvecs.T
    return _symmetrize(cov), float("nan"), False


def _lag_autocorr(series: np.ndarray, max_lag: int) -> List[Optional[float]]:
    """
    Автокорреляция лагов 1..max_lag одномерного ряда (парная, по Пирсону).

    Пары с нефинитными компонентами отбрасываются; при вырожденной
    дисперсии или недостаточном числе пар возвращается None.
    """
    s = np.asarray(series, dtype=np.float64).reshape(-1)
    out: List[Optional[float]] = []
    for lag in range(1, int(max_lag) + 1):
        if lag >= s.size - 1:
            out.append(None)
            continue
        a, b = s[:-lag], s[lag:]
        ok = np.isfinite(a) & np.isfinite(b)
        if int(np.count_nonzero(ok)) < 3:
            out.append(None)
            continue
        a = a[ok] - a[ok].mean()
        b = b[ok] - b[ok].mean()
        denom = float(np.sqrt((a @ a) * (b @ b)))
        out.append(None if denom <= 0.0 else float((a @ b) / denom))
    return out


def _observation_blocks(problem):
    """
    Метаданные ObservationSpec ResidualLoss: [(spec, numel)] или None.

    Fallback None — для objective без атрибута observations.
    """
    objective = getattr(problem, "objective", None)
    observations = getattr(objective, "observations", None)
    if not observations:
        return None
    blocks = []
    for spec in observations:
        target_shape = spec.target.shape
        if target_shape.rank != 3 or target_shape.num_elements() is None:
            raise ValueError(
                "uq: residual-диагностика требует target спецификаций ранга 3 "
                f"([batch, T, n_units]) с конкретными размерами, получена "
                f"форма {target_shape} для spec '{spec.name}'."
            )
        blocks.append((spec, int(target_shape.num_elements())))
    return blocks


def _spec_mask(spec, shape) -> Optional[np.ndarray]:
    """Булева маска спецификации, приведённая к форме target (None — нет)."""
    if spec.mask is None:
        return None
    mask = np.asarray(spec.mask, dtype=np.float64)
    return np.broadcast_to(mask, shape) > 0.0


def _prior_precision(prior_cov, n: int, where: str) -> Optional[np.ndarray]:
    """
    P = inv(prior_cov) из prior_cov (скаляр — изотропный приор, [n, n] —
    полный). None — плоский приор. Проверки: размерность, симметрия,
    обратимость.
    """
    if prior_cov is None:
        return None
    arr = np.asarray(prior_cov, dtype=np.float64)
    if arr.ndim == 0:
        value = float(arr)
        if value <= 0.0:
            raise ValueError(f"{where}: prior_cov-скаляр должен быть > 0.")
        return np.eye(n, dtype=np.float64) / value
    if arr.shape != (n, n):
        raise ValueError(
            f"{where}: prior_cov имеет форму {arr.shape}, ожидается ({n}, "
            f"{n}) или скаляр."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{where}: prior_cov содержит нефинитные значения.")
    if float(np.max(np.abs(arr - arr.T))) > _PRIOR_SYM_RTOL * max(
        1.0, float(np.max(np.abs(arr)))
    ):
        raise ValueError(f"{where}: prior_cov должна быть симметричной.")
    try:
        return np.linalg.inv(_symmetrize(arr))
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            f"{where}: prior_cov вырождена (inv невозможен); задайте "
            "регулярную (например, добавьте ridge к диагонали)."
        ) from exc


def _theta_row(space, phi_row: np.ndarray) -> Dict[str, np.ndarray]:
    """φ-строка → физические θ {name: array формы записи}; NaN при отказе."""
    theta: Dict[str, np.ndarray] = {}
    try:
        unpacked = space.unpack(
            cast_if_needed(tf.convert_to_tensor(phi_row), neuraltide.config.get_dtype())
        )
        for name, value in unpacked.items():
            theta[str(name)] = np.asarray(value, dtype=np.float64)
    except Exception:  # noqa: BLE001 — нефинитная строка/трансформ: честный NaN
        for name, entry in space.phi_entries.items():
            theta[str(name)] = np.full(entry.shape.as_list(), np.nan)
    return theta


# ---------------------------------------------------------------------------
# DTO: диагностика невязок
# ---------------------------------------------------------------------------


@dataclass
class SpecResidualDiagnostics(_JsonDTO):
    """
    Диагностика невязок одной ObservationSpec.

    Атрибуты:
        name: имя спецификации;
        population: имя популяции;
        n_elements: число элементов блока (batch·T·n_units);
        n_masked: число элементов, исключённых маской из пулов;
        mean / std: среднее/СКО по немаскированным элементам (None при
            пустом пуле/вырожденном случае);
        max_abs: max|r| по всему блоку (включая маскированные нули);
        time_mean: [T] среднее по бэтчу/юнитам на шаг времени
            (None для полностью маскированного шага);
        autocorr: автокорреляция лагов 1..max_lag временного ряда
            time_mean (None там, где не определена);
        bin_stds: [n_bins] СКО по временным бинам (None для пустых бинов);
        bin_edges: [n_bins + 1] границы бинов по временной оси;
        levene_stat / levene_pvalue: тест Левена (центр — медиана, т.е.
            Brown–Forsythe) на равенство дисперсий бинов; None при
            недостаточном числе непустых бинов/выборке.
    """

    name: str = ""
    population: str = ""
    n_elements: int = 0
    n_masked: int = 0
    mean: Optional[float] = None
    std: Optional[float] = None
    max_abs: float = 0.0
    time_mean: List[Optional[float]] = field(default_factory=list)
    autocorr: List[Optional[float]] = field(default_factory=list)
    bin_stds: List[Optional[float]] = field(default_factory=list)
    bin_edges: List[float] = field(default_factory=list)
    levene_stat: Optional[float] = None
    levene_pvalue: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": str(self.name),
            "population": str(self.population),
            "n_elements": int(self.n_elements),
            "n_masked": int(self.n_masked),
            "mean": self.mean,
            "std": self.std,
            "max_abs": float(self.max_abs),
            "time_mean": _clean_tree(self.time_mean),
            "autocorr": _clean_tree(self.autocorr),
            "bin_stds": _clean_tree(self.bin_stds),
            "bin_edges": [float(v) for v in self.bin_edges],
            "levene_stat": self.levene_stat,
            "levene_pvalue": self.levene_pvalue,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SpecResidualDiagnostics":
        def _restore(values):
            return [None if v is None else float(v) for v in values]

        return cls(
            name=str(data.get("name", "")),
            population=str(data.get("population", "")),
            n_elements=int(data.get("n_elements", 0)),
            n_masked=int(data.get("n_masked", 0)),
            mean=data.get("mean"),
            std=data.get("std"),
            max_abs=float(data.get("max_abs", 0.0)),
            time_mean=_restore(data.get("time_mean", [])),
            autocorr=_restore(data.get("autocorr", [])),
            bin_stds=_restore(data.get("bin_stds", [])),
            bin_edges=[float(v) for v in data.get("bin_edges", [])],
            levene_stat=data.get("levene_stat"),
            levene_pvalue=data.get("levene_pvalue"),
        )


@dataclass
class WindowDiagnostics(_JsonDTO):
    """
    Сводка невязок по временному окну [start, stop) для всех spec.

    Атрибуты:
        start / stop: границы окна по временной оси;
        mean / std / max_abs: {spec.name: значение} по немаскированным
            элементам окна (None — пустой пул).
    """

    start: int = 0
    stop: int = 0
    mean: Dict[str, Optional[float]] = field(default_factory=dict)
    std: Dict[str, Optional[float]] = field(default_factory=dict)
    max_abs: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": int(self.start),
            "stop": int(self.stop),
            "mean": {str(k): v for k, v in self.mean.items()},
            "std": {str(k): v for k, v in self.std.items()},
            "max_abs": {str(k): float(v) for k, v in self.max_abs.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WindowDiagnostics":
        return cls(
            start=int(data.get("start", 0)),
            stop=int(data.get("stop", 0)),
            mean={str(k): v for k, v in data.get("mean", {}).items()},
            std={str(k): v for k, v in data.get("std", {}).items()},
            max_abs={str(k): float(v) for k, v in data.get("max_abs", {}).items()},
        )


@dataclass
class ResidualDiagnostics(_JsonDTO):
    """
    Итог диагностики невязок в точке φ.

    Атрибуты:
        specs: per-spec диагностика (порядок ObservationSpec objective);
        windows: сводки по временным окнам;
        max_abs: max|r| по всему вектору;
        phi: точка оценки;
        loss: 0.5·||r||² в точке;
        n_residuals: длина вектора невязок;
        max_lag / n_bins: использованные параметры;
        has_observation_metadata: False — fallback на плоский псевдо-spec;
        n_forward / wall_time: дельта счётчиков задачи за вызов.
    """

    specs: List[SpecResidualDiagnostics] = field(default_factory=list)
    windows: List[WindowDiagnostics] = field(default_factory=list)
    max_abs: float = 0.0
    phi: np.ndarray = field(default_factory=lambda: np.zeros(0))
    loss: float = 0.0
    n_residuals: int = 0
    max_lag: int = 10
    n_bins: int = 10
    has_observation_metadata: bool = True
    n_forward: int = 0
    wall_time: float = 0.0
    notes: str = ""

    def __post_init__(self) -> None:
        self.phi = np.asarray(self.phi, dtype=np.float64)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "specs": [s.to_dict() for s in self.specs],
            "windows": [w.to_dict() for w in self.windows],
            "max_abs": float(self.max_abs),
            "phi": self.phi.tolist(),
            "loss": float(self.loss),
            "n_residuals": int(self.n_residuals),
            "max_lag": int(self.max_lag),
            "n_bins": int(self.n_bins),
            "has_observation_metadata": bool(self.has_observation_metadata),
            "n_forward": int(self.n_forward),
            "wall_time": float(self.wall_time),
            "notes": str(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ResidualDiagnostics":
        return cls(
            specs=[SpecResidualDiagnostics.from_dict(d) for d in data.get("specs", [])],
            windows=[WindowDiagnostics.from_dict(d) for d in data.get("windows", [])],
            max_abs=float(data.get("max_abs", 0.0)),
            phi=np.asarray(data.get("phi", []), dtype=np.float64),
            loss=float(data.get("loss", 0.0)),
            n_residuals=int(data.get("n_residuals", 0)),
            max_lag=int(data.get("max_lag", 10)),
            n_bins=int(data.get("n_bins", 10)),
            has_observation_metadata=bool(data.get("has_observation_metadata", True)),
            n_forward=int(data.get("n_forward", 0)),
            wall_time=float(data.get("wall_time", 0.0)),
            notes=str(data.get("notes", "")),
        )


def _default_windows(t: int) -> List[Tuple[int, int]]:
    """Окна по умолчанию: 4 равных смежных интервала [start, stop)."""
    edges = np.unique(np.linspace(0, int(t), 5, dtype=int))
    return [(int(edges[i]), int(edges[i + 1])) for i in range(len(edges) - 1)]


def _validate_windows(
    windows: Sequence[Tuple[int, int]], t: int
) -> List[Tuple[int, int]]:
    out = []
    for w in windows:
        s, e = int(w[0]), int(w[1])
        if not (0 <= s < e <= t):
            raise ValueError(
                f"residual_diagnostics: окно ({s}, {e}) вне [0, {t}) или вырождено."
            )
        out.append((s, e))
    return out


def _bin_stats(
    block: np.ndarray, unmasked: np.ndarray, n_bins: int
) -> Tuple[List[Optional[float]], List[int]]:
    """СКО невязок по временным бинам + целочисленные границы бинов."""
    t = block.shape[1]
    edges = [int(v) for v in np.unique(np.linspace(0, t, int(n_bins) + 1, dtype=int))]
    stds: List[Optional[float]] = []
    for i in range(len(edges) - 1):
        s, e = edges[i], edges[i + 1]
        pool = block[:, s:e, :][unmasked[:, s:e, :]]
        stds.append(_f(np.std(pool)) if pool.size >= 2 else None)
    return stds, edges


def _levene_across_bins(
    bins: Sequence[np.ndarray],
) -> Tuple[Optional[float], Optional[float]]:
    groups = [np.asarray(b, dtype=np.float64) for b in bins if np.asarray(b).size >= 2]
    if len(groups) < 2:
        return None, None
    try:
        # Вырожденные группы (нулевая дисперсия) дают RuntimeWarning
        # scipy — статистика всё равно вычисляется; шум подавляем.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, pvalue = sp_stats.levene(*groups, center="median")
    except Exception:  # noqa: BLE001 — вырожденные группы: честный None
        return None, None
    return _f(stat), _f(pvalue)


def residual_diagnostics(
    problem,
    phi,
    *,
    max_lag: int = 10,
    n_bins: int = 10,
    windows: Optional[Sequence[Tuple[int, int]]] = None,
) -> ResidualDiagnostics:
    """
    Диагностика вектора невязок задачи в точке φ.

    Требует objective с residual-интерфейсом (problem.residual). При
    objective с метаданными ObservationSpec (ResidualLoss.observations)
    невязки раскладываются по спецификациям: mean/std/max|r| по
    немаскированным элементам, временной ряд (среднее по бэтчу/юнитам на
    шаг), автокорреляция лагов 1..max_lag этого ряда, гетероскедастичность
    по n_bins временным бинам (Levene/Brown–Forsythe по СКО бинов) и сводки
    по временным окнам (по умолчанию 4 равных интервала; windows задаётся
    явно как [(start, stop), ...]). Маскированные элементы (структурные
    нули residual) исключаются из всех пулов, кроме max_abs.

    Без метаданных observations — fallback: один псевдо-spec 'residual' по
    плоскому вектору (autocorr по плоскому ряду, бины по индексам, без
    временных окон).

    Args:
        problem: CalibrationProblem (residual-objective).
        phi: точка оценки (None — текущие значения модели).
        max_lag: число лагов автокорреляции (>= 1).
        n_bins: число временных бинов гетероскедастичности (>= 2).
        windows: явные временные окна [(start, stop), ...]; None — 4
            равных интервала по T целевой траектории.

    Returns:
        ResidualDiagnostics.
    """
    max_lag = int(max_lag)
    n_bins = int(n_bins)
    if max_lag < 1:
        raise ValueError(f"residual_diagnostics: max_lag >= 1, получен {max_lag}.")
    if n_bins < 2:
        raise ValueError(f"residual_diagnostics: n_bins >= 2, получен {n_bins}.")

    t0 = time.perf_counter()
    fwd0, _ = _counters(problem)
    r = np.asarray(problem.residual(phi), dtype=np.float64).reshape(-1)
    phi_arr = np.asarray(problem.pack(phi), dtype=np.float64).reshape(-1)
    loss = 0.5 * float(r @ r)

    blocks = _observation_blocks(problem)
    specs_out: List[SpecResidualDiagnostics] = []
    windows_out: List[WindowDiagnostics] = []
    window_bounds: List[Tuple[int, int]] = []

    if blocks is None:
        ts = r
        bin_stds: List[Optional[float]] = []
        edges = np.unique(np.linspace(0, ts.size, n_bins + 1, dtype=int))
        for i in range(len(edges) - 1):
            pool = ts[edges[i] : edges[i + 1]]
            bin_stds.append(_f(np.std(pool)) if pool.size >= 2 else None)
        bin_edges = [float(v) for v in edges]
        std_val = _f(np.std(ts)) if ts.size >= 2 else None
        specs_out.append(
            SpecResidualDiagnostics(
                name="residual",
                population="",
                n_elements=int(ts.size),
                n_masked=0,
                mean=_f(np.mean(ts)) if ts.size else None,
                std=std_val,
                max_abs=float(np.max(np.abs(ts))) if ts.size else 0.0,
                time_mean=[_f(v) for v in ts],
                autocorr=_lag_autocorr(ts, max_lag),
                bin_stds=bin_stds,
                bin_edges=bin_edges,
                levene_stat=None,
                levene_pvalue=None,
            )
        )
        has_meta = False
        notes = (
            "Objective не предоставляет observations — fallback на плоский "
            "псевдо-spec 'residual': временная структура и окна недоступны, "
            "автокорреляция и бины посчитаны по плоскому вектору невязок."
        )
    else:
        total = sum(numel for _, numel in blocks)
        if total != r.size:
            raise ValueError(
                f"residual_diagnostics: суммарный объём target'ов "
                f"observations ({total}) не совпадает с длиной residual "
                f"({r.size}) — objective изменился после прогона?"
            )
        shapes = [tuple(spec.target.shape.as_list()) for spec, _ in blocks]
        t_steps = {s[1] for s in shapes}
        if len(t_steps) != 1:
            raise ValueError(
                "residual_diagnostics: спецификации имеют разную длину "
                f"траектории T: {sorted(t_steps)} — временные окна не "
                "определены."
            )
        t = int(t_steps.pop())
        window_bounds = (
            _validate_windows(windows, t)
            if windows is not None
            else _default_windows(t)
        )

        offset = 0
        spec_blocks: List[Tuple[Any, np.ndarray, np.ndarray]] = []
        for spec, numel in blocks:
            shape = tuple(spec.target.shape.as_list())
            block = r[offset : offset + numel].reshape(shape)
            offset += numel
            mask = _spec_mask(spec, shape)
            if mask is None:
                mask = np.ones(shape, dtype=bool)
            spec_blocks.append((spec, block, mask))

        for spec, block, mask in spec_blocks:
            pool = block[mask]
            masked_count = int(block.size - np.count_nonzero(mask))
            # Временной ряд: среднее по немаскированным на каждом шаге.
            counts = mask.sum(axis=(0, 2))
            sums = np.where(mask, block, 0.0).sum(axis=(0, 2))
            with np.errstate(invalid="ignore", divide="ignore"):
                time_mean = sums / np.maximum(counts, 1)
            time_mean = [
                _f(v) if counts[i] > 0 else None for i, v in enumerate(time_mean)
            ]
            bin_stds, bin_edges = _bin_stats(block, mask, n_bins)
            levene_groups = [
                block[:, bin_edges[i] : int(bin_edges[i + 1]), :][
                    mask[:, int(bin_edges[i]) : int(bin_edges[i + 1]), :]
                ]
                for i in range(len(bin_edges) - 1)
            ]
            stat, pvalue = _levene_across_bins(levene_groups)
            specs_out.append(
                SpecResidualDiagnostics(
                    name=str(spec.name),
                    population=str(spec.population),
                    n_elements=int(block.size),
                    n_masked=masked_count,
                    mean=_f(np.mean(pool)) if pool.size else None,
                    std=_f(np.std(pool)) if pool.size >= 2 else None,
                    max_abs=float(np.max(np.abs(block))) if block.size else 0.0,
                    time_mean=time_mean,
                    autocorr=_lag_autocorr(
                        np.array([v if v is not None else np.nan for v in time_mean]),
                        max_lag,
                    ),
                    bin_stds=bin_stds,
                    bin_edges=[float(v) for v in bin_edges],
                    levene_stat=stat,
                    levene_pvalue=pvalue,
                )
            )

        for s, e in window_bounds:
            win = WindowDiagnostics(start=s, stop=e)
            for spec, block, mask in spec_blocks:
                pool = block[:, s:e, :][mask[:, s:e, :]]
                win.mean[str(spec.name)] = _f(np.mean(pool)) if pool.size else None
                win.std[str(spec.name)] = _f(np.std(pool)) if pool.size >= 2 else None
                win.max_abs[str(spec.name)] = (
                    float(np.max(np.abs(pool))) if pool.size else 0.0
                )
            windows_out.append(win)
        has_meta = True
        notes = (
            "Пулы (mean/std, autocorr, бины) — по немаскированным элементам; "
            "max_abs — по всему блоку (маскированные дают структурный ноль "
            "residual-нормировки). Автокорреляция — по временному ряду "
            "средних по шагу; гетероскедастичность — Levene (медиана, "
            "Brown–Forsythe) по СКО временных бинов. Окна по умолчанию — "
            "4 равных интервала [start, stop)."
        )

    fwd1, _ = _counters(problem)
    return ResidualDiagnostics(
        specs=specs_out,
        windows=windows_out,
        max_abs=float(np.max(np.abs(r))) if r.size else 0.0,
        phi=phi_arr,
        loss=loss,
        n_residuals=int(r.size),
        max_lag=max_lag,
        n_bins=n_bins,
        has_observation_metadata=has_meta,
        n_forward=fwd1 - fwd0,
        wall_time=time.perf_counter() - t0,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# DTO и функция: Laplace approximation
# ---------------------------------------------------------------------------


@dataclass
class LaplaceResult(_JsonDTO):
    """
    Гауссово приближение постериора в точке MAP.

    Атрибуты:
        phi_map: точка разложения (вектор φ);
        covariance: posterior covariance в φ-координатах [n, n];
        precision_gn: GN-часть precision (JᵀJ для 'gn', Гессиан потери для
            'fd-hessian') — используется noise_sensitivity;
        prior_precision: P = inv(prior_cov) или None (плоский приор);
        eigvals / eigvecs: спектр covariance (по убыванию; eigvecs —
            столбцы-направления);
        std: sqrt(diag(covariance)) — маргинальные СКО φ-координат;
        param_names: имена φ-координат ParameterSpace ('name' для numel==1,
            'name[i]' иначе);
        intervals: {flat_name: (lo, hi)} маргинальные интервалы уровня
            level в ФИЗИЧЕСКИХ θ (через transform.forward; границы min/max
            ParameterSpace в интервалы НЕ вшиты);
        phi_intervals: то же в φ-координатах (phi_map ± z·std);
        level / z: уровень интервалов и нормальный квантиль;
        method: 'gn' | 'fd-hessian';
        pd: True, если precision удачно обратилась через Cholesky
            (возможно, с jitter); False — спектральный fallback;
        jitter_used: применённый абсолютный jitter (None при fallback);
        loss: потеря в точке (0.5·||r||² для 'gn');
        n_forward: дельта счётчиков прямых прогонов задачи;
        wall_time: время вызова, секунды;
        max_params: использованный кап для residual_jacobian;
        notes: допущения и оговорки.
    """

    phi_map: np.ndarray = field(default_factory=lambda: np.zeros(0))
    covariance: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    precision_gn: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    prior_precision: Optional[np.ndarray] = None
    eigvals: np.ndarray = field(default_factory=lambda: np.zeros(0))
    eigvecs: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    std: np.ndarray = field(default_factory=lambda: np.zeros(0))
    param_names: List[str] = field(default_factory=list)
    intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    phi_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    level: float = 0.95
    z: float = _Z95
    method: str = "gn"
    pd: bool = True
    jitter_used: Optional[float] = None
    loss: float = 0.0
    n_forward: int = 0
    wall_time: float = 0.0
    max_params: int = 200
    notes: str = ""

    def __post_init__(self) -> None:
        self.phi_map = np.asarray(self.phi_map, dtype=np.float64)
        self.covariance = np.asarray(self.covariance, dtype=np.float64)
        self.precision_gn = np.asarray(self.precision_gn, dtype=np.float64)
        self.eigvals = np.asarray(self.eigvals, dtype=np.float64)
        self.eigvecs = np.asarray(self.eigvecs, dtype=np.float64)
        self.std = np.asarray(self.std, dtype=np.float64)
        if self.prior_precision is not None:
            self.prior_precision = np.asarray(self.prior_precision, dtype=np.float64)
        self.intervals = {
            str(k): (float(v[0]), float(v[1])) for k, v in self.intervals.items()
        }
        self.phi_intervals = {
            str(k): (float(v[0]), float(v[1])) for k, v in self.phi_intervals.items()
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phi_map": self.phi_map.tolist(),
            "covariance": self.covariance.tolist(),
            "precision_gn": self.precision_gn.tolist(),
            "prior_precision": (
                None if self.prior_precision is None else self.prior_precision.tolist()
            ),
            "eigvals": self.eigvals.tolist(),
            "eigvecs": self.eigvecs.tolist(),
            "std": self.std.tolist(),
            "param_names": [str(n) for n in self.param_names],
            "intervals": {k: [v[0], v[1]] for k, v in self.intervals.items()},
            "phi_intervals": {k: [v[0], v[1]] for k, v in self.phi_intervals.items()},
            "level": float(self.level),
            "z": float(self.z),
            "method": str(self.method),
            "pd": bool(self.pd),
            "jitter_used": self.jitter_used,
            "loss": float(self.loss),
            "n_forward": int(self.n_forward),
            "wall_time": float(self.wall_time),
            "max_params": int(self.max_params),
            "notes": str(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LaplaceResult":
        prior = data.get("prior_precision")

        def _pair_map(raw):
            return {str(k): (float(v[0]), float(v[1])) for k, v in raw.items()}

        return cls(
            phi_map=np.asarray(data.get("phi_map", []), dtype=np.float64),
            covariance=np.asarray(data.get("covariance", []), dtype=np.float64),
            precision_gn=np.asarray(data.get("precision_gn", []), dtype=np.float64),
            prior_precision=(
                None if prior is None else np.asarray(prior, dtype=np.float64)
            ),
            eigvals=np.asarray(data.get("eigvals", []), dtype=np.float64),
            eigvecs=np.asarray(data.get("eigvecs", []), dtype=np.float64),
            std=np.asarray(data.get("std", []), dtype=np.float64),
            param_names=[str(n) for n in data.get("param_names", [])],
            intervals=_pair_map(data.get("intervals", {})),
            phi_intervals=_pair_map(data.get("phi_intervals", {})),
            level=float(data.get("level", 0.95)),
            z=float(data.get("z", _Z95)),
            method=str(data.get("method", "gn")),
            pd=bool(data.get("pd", True)),
            jitter_used=data.get("jitter_used"),
            loss=float(data.get("loss", 0.0)),
            n_forward=int(data.get("n_forward", 0)),
            wall_time=float(data.get("wall_time", 0.0)),
            max_params=int(data.get("max_params", 200)),
            notes=str(data.get("notes", "")),
        )


def _intervals_from_covariance(space, phi, covariance, z):
    """Маргинальные интервалы φ_map ± z·std в φ и физических θ-координатах."""
    n = phi.size
    names = _flat_names(space, n)
    offsets = _entry_offsets(space)
    diag = np.maximum(np.diag(covariance), 0.0)
    std = np.sqrt(diag)
    intervals: Dict[str, Tuple[float, float]] = {}
    phi_intervals: Dict[str, Tuple[float, float]] = {}
    dtype = neuraltide.config.get_dtype()
    for name, entry in space.phi_entries.items():
        start, numel = offsets[name]
        piece = phi[start : start + numel]
        sd = std[start : start + numel]
        lo_p = piece - z * sd
        hi_p = piece + z * sd
        theta_lo = np.asarray(
            entry.transform.forward(tf.constant(lo_p, dtype=dtype)), dtype=np.float64
        ).reshape(-1)
        theta_hi = np.asarray(
            entry.transform.forward(tf.constant(hi_p, dtype=dtype)), dtype=np.float64
        ).reshape(-1)
        for i in range(numel):
            key = name if numel == 1 else f"{name}[{i}]"
            intervals[key] = (
                float(min(theta_lo[i], theta_hi[i])),
                float(max(theta_lo[i], theta_hi[i])),
            )
            phi_intervals[key] = (float(lo_p[i]), float(hi_p[i]))
    return names, intervals, phi_intervals


def _fd_hessian(problem, phi: np.ndarray, h: float) -> Tuple[np.ndarray, float]:
    """
    Центральные конечные разности градиента потери: H[:, j] ≈ ∂²L/∂φ∂φ_j.

    Требует 2n + 1 вычислений value_and_gradient. O(h²) погрешность;
    шаг h·max(|φ_j|, 1). Возвращает (симметризованный Гессиан, потерю в
    точке).
    """
    n = phi.size
    eval0, _ = problem.value_and_gradient(phi)
    hessian = np.empty((n, n), dtype=np.float64)
    for j in range(n):
        step = float(h) * max(abs(phi[j]), 1.0)
        e = np.zeros(n, dtype=np.float64)
        e[j] = step
        _, grad_plus = problem.value_and_gradient(phi + e)
        _, grad_minus = problem.value_and_gradient(phi - e)
        hessian[:, j] = (
            np.asarray(grad_plus, dtype=np.float64)
            - np.asarray(grad_minus, dtype=np.float64)
        ) / (2.0 * step)
    return _symmetrize(hessian), float(eval0.loss)


def laplace_approximation(
    problem,
    phi_map,
    *,
    jac: Optional[JacobianResult] = None,
    prior_cov=None,
    method: str = "gn",
    level: float = 0.95,
    max_params: int = 200,
    fd_h: float = 1e-5,
    max_fd_params: int = _FD_HESSIAN_MAX_PARAMS,
) -> LaplaceResult:
    """
    Гауссово приближение постериора p(φ | data) ≈ N(φ_map, covariance).

    Метод 'gn' (по умолчанию): covariance = (JᵀJ + P)⁻¹, где J — якобиан
    невязок по φ в единицах полного NLL (нормировка на sigma уже в residual;
    деление на N у ResidualLoss снимается множителями
    JacobianResult.likelihood_scale), P = inv(prior_cov) или 0 при
    prior_cov=None. J берётся из переданного jac (JacobianResult) либо
    вычисляется через sensitivities.residual_jacobian(problem, phi_map,
    max_params=max_params=200) с ленивым импортом sensitivities; n
    параметров сверх капа — ValueError от residual_jacobian.

    Метод 'fd-hessian' — только как явно документированный sanity-check:
    Гессиан потери центральными конечными разностями градиента (2n + 1
    прогонов с градиентом, O(h²) погрешность, чувствителен к точности
    градиента и кривизне вне минимума — вне минимума H ≠ JᵀJ из-за члена
    Σ rᵢ∇²rᵢ). Жёсткий кап: n > max_fd_params=50 → ValueError БЕЗ любых
    прогонов. jac с method='fd-hessian' несовместимы (ValueError).

    PD-проверка: Cholesky precision с нарастающим jitter (множитель от
    максимальной диагонали); флаг pd — успешность, jitter_used —
    применённый абсолютный jitter (None при спектральном fallback).
    Интервалы маргинальные, по именам ParameterSpace, в физических θ
    (intervals) и в φ (phi_intervals); границы min/max записей в них не
    вшиты — прижатая к границе MAP интервалы может выводить за пределы.

    Args:
        problem: CalibrationProblem (или duck-typed с parameter_space/
            pack/counters; для 'gn' без jac — ещё и с residual-objective).
        phi_map: точка MAP (None — текущие значения модели через pack()).
        jac: готовый JacobianResult (метод 'gn'); None — вычислить.
        prior_cov: None (плоский приор), скаляр (> 0, изотропный) или
            матрица [n, n] SPD (проверяется размерность/симметрия/обратимость).
        method: 'gn' | 'fd-hessian'.
        level: уровень маргинальных интервалов (0.95 → z ≈ 1.95996).
        max_params: кап числа параметров для внутреннего
            residual_jacobian.
        fd_h: относительный шаг конечных разностей (метод 'fd-hessian').
        max_fd_params: жёсткий кап размерности для 'fd-hessian'.

    Returns:
        LaplaceResult.

    Счётчики: n_forward — дельта счётчиков задачи (для 'gn' с готовым jac
    — 0; для 'fd-hessian' — 2n + 1 за вычисления градиента).
    """
    method = str(method).lower()
    if method not in ("gn", "fd-hessian"):
        raise ValueError(
            f"laplace_approximation: неизвестный method={method!r}; "
            "доступны 'gn' и 'fd-hessian'."
        )
    if jac is not None and method != "gn":
        raise ValueError(
            "laplace_approximation: jac совместим только с method='gn' — "
            "для 'fd-hessian' Гессиан строится конечными разностями."
        )
    level = float(level)
    if not 0.0 < level < 1.0:
        raise ValueError(
            f"laplace_approximation: level должен быть в (0, 1), got {level}."
        )

    space = getattr(problem, "parameter_space", None)
    if space is None:
        raise ValueError(
            "laplace_approximation: problem не предоставляет "
            "parameter_space — имена/интервалы по ParameterSpace "
            "невозможны."
        )

    if phi_map is None:
        phi = _pack_current(problem, "laplace_approximation")
    else:
        phi = np.asarray(problem.pack(phi_map), dtype=np.float64).reshape(-1)
    n = phi.size
    if n != int(space.size):
        raise ValueError(
            f"laplace_approximation: phi_map имеет {n} элементов, "
            f"ParameterSpace — {int(space.size)}."
        )
    if n == 0:
        raise ValueError("laplace_approximation: пустой ParameterSpace (size == 0).")

    t0 = time.perf_counter()
    fwd0, _ = _counters(problem)
    prior_precision = _prior_precision(prior_cov, n, "laplace_approximation")

    if method == "gn":
        if jac is None:
            from neuraltide.optimization import sensitivities

            jac = sensitivities.residual_jacobian(
                problem, phi, max_params=int(max_params)
            )
        to_nll = getattr(jac, "likelihood_jacobian", None)
        j_mat = np.asarray(to_nll() if to_nll is not None else jac.J, dtype=np.float64)
        if j_mat.ndim != 2 or j_mat.shape[1] != n:
            raise ValueError(
                f"laplace_approximation: jac.J имеет форму {j_mat.shape}, "
                f"ожидается [n_residuals, {n}]."
            )
        gn = _symmetrize(j_mat.T @ j_mat)
        loss = float(jac.loss)
        max_params_used = int(max_params)
        notes = (
            "GN-ковариация (JᵀJ + P)⁻¹ в единицах полного NLL: строки J "
            "умножены на JacobianResult.likelihood_scale (sqrt(N) для "
            "ResidualLoss с normalization='mean'). Интервалы маргинальные, "
            "границы ParameterSpace не вшиты; прижатая к границе MAP может "
            "давать интервалы за пределами [min, max]."
        )
    else:
        if n > int(max_fd_params):
            raise ValueError(
                f"laplace_approximation: method='fd-hessian' имеет жёсткий "
                f"кап max_fd_params={int(max_fd_params)} параметров, "
                f"получено n={n}. Кап защищает от 2n+1 прогонов с "
                "градиентом; используйте method='gn'."
            )
        with likelihood_normalization(problem):
            gn, loss = _fd_hessian(problem, phi, float(fd_h))
        max_params_used = int(max_params)
        notes = (
            "fd-hessian — явно приближённый sanity-check в единицах полного "
            "NLL (ResidualLoss временно с normalization='sum'): H — центральные "
            f"конечные разности градиента (шаг {fd_h}·max(|φ_j|, 1), O(h²) "
            f"погрешность, {2 * n + 1} прогонов с градиентом). Вне минимума "
            "H = JᵀJ + Σ rᵢ∇²rᵢ ≠ JᵀJ; ковариация осмысленна только около "
            "достаточно глубокого минимума. Границы ParameterSpace во "
            "интервалы не вшиты."
        )

    precision = gn if prior_precision is None else _symmetrize(gn + prior_precision)
    covariance, jitter, pd = _pd_inverse(precision)
    eigvals, eigvecs = np.linalg.eigh(covariance)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    std = np.sqrt(np.maximum(np.diag(covariance), 0.0))

    z = float(sp_stats.norm.ppf(0.5 + level / 2.0))
    names, intervals, phi_intervals = _intervals_from_covariance(
        space, phi, covariance, z
    )

    fwd1, _ = _counters(problem)
    return LaplaceResult(
        phi_map=phi,
        covariance=covariance,
        precision_gn=gn,
        prior_precision=prior_precision,
        eigvals=eigvals,
        eigvecs=eigvecs,
        std=std,
        param_names=names,
        intervals=intervals,
        phi_intervals=phi_intervals,
        level=level,
        z=z,
        method=method,
        pd=pd,
        jitter_used=None if not np.isfinite(jitter) else float(jitter),
        loss=loss,
        n_forward=fwd1 - fwd0,
        wall_time=time.perf_counter() - t0,
        max_params=max_params_used,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# DTO и функции: bootstrap
# ---------------------------------------------------------------------------


@dataclass
class BootstrapResult(_JsonDTO):
    """
    Результат bootstrap-процедуры (эксперименты/веса/остатки).

    Атрибуты:
        scheme: 'experiments' | 'random-weights' | 'residual';
        n_draws: число draw;
        seed: seed SeedSequence (или None);
        phi0: референсная стартовая точка (явно передавалась в evaluator
            каждой draw — мутация модели чужими рефитами не используется);
        phi_samples: [n_draws, n] — финальные φ draw (NaN-строки при
            неудачной draw);
        theta_samples: {full_name: [n_draws, *entry.shape]} физические θ
            (None, если ParameterSpace недоступен);
        losses: финальные потери draw (None при неудаче);
        statuses: 'ok' | 'failed' по draw;
        messages: сообщения/причины отказа;
        weights: per-draw веса схемы (counts экспериментов для
            'experiments', вектор весов для 'random-weights', None для
            'residual');
        evaluator: имя/представление использованного evaluator;
        n_forward: суммарная дельта прямых прогонов по draw;
        wall_time: общее время, секунды;
        notes: схема и честные оговорки.
    """

    scheme: str = ""
    n_draws: int = 0
    seed: Optional[int] = None
    phi0: np.ndarray = field(default_factory=lambda: np.zeros(0))
    phi_samples: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    theta_samples: Optional[Dict[str, np.ndarray]] = None
    losses: List[Optional[float]] = field(default_factory=list)
    statuses: List[str] = field(default_factory=list)
    messages: List[str] = field(default_factory=list)
    weights: List[Any] = field(default_factory=list)
    evaluator: str = ""
    n_forward: int = 0
    wall_time: float = 0.0
    notes: str = ""

    def __post_init__(self) -> None:
        self.phi0 = np.asarray(self.phi0, dtype=np.float64)
        self.phi_samples = np.asarray(self.phi_samples, dtype=np.float64)
        if self.theta_samples is not None:
            self.theta_samples = {
                str(k): np.asarray(v, dtype=np.float64)
                for k, v in self.theta_samples.items()
            }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scheme": str(self.scheme),
            "n_draws": int(self.n_draws),
            "seed": self.seed,
            "phi0": _clean_tree(self.phi0),
            "phi_samples": _clean_tree(self.phi_samples),
            "theta_samples": (
                None if self.theta_samples is None else _clean_tree(self.theta_samples)
            ),
            "losses": _clean_tree(self.losses),
            "statuses": [str(s) for s in self.statuses],
            "messages": [str(m) for m in self.messages],
            "weights": _clean_tree(self.weights),
            "evaluator": str(self.evaluator),
            "n_forward": int(self.n_forward),
            "wall_time": float(self.wall_time),
            "notes": str(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BootstrapResult":
        theta = data.get("theta_samples")
        return cls(
            scheme=str(data.get("scheme", "")),
            n_draws=int(data.get("n_draws", 0)),
            seed=data.get("seed"),
            phi0=np.asarray(data.get("phi0", []), dtype=np.float64),
            phi_samples=np.asarray(data.get("phi_samples", []), dtype=np.float64),
            theta_samples=(
                None
                if theta is None
                else {str(k): np.asarray(v, dtype=np.float64) for k, v in theta.items()}
            ),
            losses=[None if v is None else float(v) for v in data.get("losses", [])],
            statuses=[str(s) for s in data.get("statuses", [])],
            messages=[str(m) for m in data.get("messages", [])],
            weights=list(data.get("weights", [])),
            evaluator=str(data.get("evaluator", "")),
            n_forward=int(data.get("n_forward", 0)),
            wall_time=float(data.get("wall_time", 0.0)),
            notes=str(data.get("notes", "")),
        )


class _WeightedObjective:
    """
    Байесовский bootstrap случайных весов поверх residual-objective.

    loss = 0.5·Σ wᵢ·rᵢ², residual = √w ⊙ r (w — фиксированный вектор draw,
    не trainable). Интерфейс минимален (__call__/residual): все
    нормировки/маски/виды штрафа наследуются от базового objective.
    """

    def __init__(self, base, weights: np.ndarray):
        self._base = base
        self._weights = np.asarray(weights, dtype=np.float64)
        residual_fn = getattr(base, "residual", None)
        if residual_fn is None:
            raise TypeError(
                "uq: random-weights bootstrap требует objective с "
                f"residual(predictions), получен {type(base).__name__}."
            )

    def __call__(self, predictions, model) -> tf.Tensor:
        r = tf.reshape(self._base.residual(predictions), [-1])
        w = cast_if_needed(tf.convert_to_tensor(self._weights), r.dtype)
        return 0.5 * tf.reduce_sum(w * r * r)

    def residual(self, predictions) -> tf.Tensor:
        r = tf.reshape(self._base.residual(predictions), [-1])
        w = cast_if_needed(tf.convert_to_tensor(self._weights), r.dtype)
        return tf.sqrt(w) * r


def _collect_theta_samples(space, phi_samples: np.ndarray) -> Dict[str, np.ndarray]:
    """Стек физических θ по draw: {full_name: [n_draws, *entry.shape]}.

    Ключи — только trainable-записи (координаты φ): non-trainable
    реестровые записи в φ-строках отсутствуют.
    """
    columns: Dict[str, List[np.ndarray]] = {str(n): [] for n in space.phi_entries}
    for row in phi_samples:
        theta_row = _theta_row(space, row)
        for name in columns:
            columns[name].append(theta_row[name])
    return {name: np.stack(values, axis=0) for name, values in columns.items()}


def _boot_experiment_name(draw: int, position: int, original: str) -> str:
    """
    Уникальное служебное имя bootstrap-копии Experiment.

    Подвыборка с повторением обязана давать draw с разными именами —
    иначе валидация MultiExperimentProblem отклонила бы такую draw
    (эксперименты с одним именем делили бы exp-specific x0-записи этого
    имени). '/' в исходном имени заменяется на '_': exp-specific
    x0-записи кодируются как 'x0/<experiment>/<pop>/<slot>', слэш в имени
    эксперимента ломал бы это соглашение.
    """
    return f"boot_{int(draw)}_{int(position)}_{str(original).replace('/', '_')}"


def _exp_specific_x0_names(space) -> List[str]:
    """
    Полные имена exp-specific x0-записей ParameterSpace: trainable-записи
    kind 'initial_state'/'shooting_state' с locator.experiment != None.
    Пустой список — таких записей нет (или space недоступен).
    """
    if space is None:
        return []
    names = []
    for name, entry in getattr(space, "phi_entries", {}).items():
        if getattr(entry, "kind", None) not in ("initial_state", "shooting_state"):
            continue
        locator = getattr(entry, "locator", None)
        if locator is not None and getattr(locator, "experiment", None) is not None:
            names.append(str(name))
    return sorted(names)


def bootstrap_experiments(
    problem,
    *,
    n_draws: int,
    seed: Optional[int],
    evaluator=None,
) -> BootstrapResult:
    """
    Bootstrap распределения fitted φ.

    Схема выбирается по типу задачи:

    - MultiExperimentProblem (атрибут experiments): пары-bootstrap на
      уровне экспериментов — каждая draw строит задачу над подвыборкой
      experiments с повторением (counts подвыборки — в weights draw).
      Копии в draw получают уникальные служебные имена
      'boot_{draw}_{position}_{original_name}' ('/' в имени заменяется на
      '_', чтобы не ломать соглашение 'x0/<experiment>/<pop>/<slot>');
      исходные Experiment не мутируются (dataclasses.replace). Проблема
      переноса имён: exp-specific x0-записи исходных имён на служебные
      не переносятся, поэтому пространства с такими записями отклоняются
      ValueError до цикла draw. Корректность покрытия опирается на
      приблизительную iid-ность экспериментов;
    - иначе (одиночная CalibrationProblem): байесовский bootstrap
      случайных весов — w ~ Exp(1) на элементах невязок, потеря draw
      0.5·Σ wᵢ rᵢ² (через _WeightedObjective-представление поверх
      problem.with_objective; счётчики общие с базовой задачей).

    Каждая draw стартует с явно захваченного phi0 = problem.pack() ДО
    цикла: рефиты изменяют состояние модели, но стартовые точки draw
    от этого не зависят (muтация не используется). evaluator —
    callable(boot_problem, phi0) -> результат с полями base.phi_best/
    base.loss_best; None/'lbfgsb'/'least_squares' — ленивая резолюция
    локальных адаптеров (без import-зависимости на этапе импорта).

    Args:
        problem: CalibrationProblem или MultiExperimentProblem.
        n_draws: число draw (>= 1).
        seed: seed np.random.SeedSequence (детерминированность по seed).
        evaluator: None (lbfgsb) | 'lbfgsb' | 'least_squares' | callable.

    Returns:
        BootstrapResult.
    """
    n_draws = int(n_draws)
    if n_draws < 1:
        raise ValueError(
            f"bootstrap_experiments: n_draws должен быть >= 1, got {n_draws}."
        )
    evaluator_fn, evaluator_name = _resolve_evaluator(evaluator)
    space = getattr(problem, "parameter_space", None)
    exp_x0_names = _exp_specific_x0_names(space)
    if exp_x0_names:
        raise ValueError(
            "bootstrap_experiments: эксперимент-специфичные x0-записи "
            f"{exp_x0_names} не переносятся на bootstrap-копии экспериментов "
            "(каждая draw получает уникальные служебные имена "
            "'boot_{draw}_{position}_{original_name}'), поэтому bootstrap "
            "для пространств с exp-specific x0 пока не поддержан — "
            "используйте ParameterSpace без таких записей."
        )
    phi0 = _pack_current(problem, "bootstrap_experiments")

    experiments = getattr(problem, "experiments", None)
    scheme = "experiments" if experiments else "random-weights"

    ss = np.random.SeedSequence(seed)
    children = ss.spawn(n_draws)

    t_begin = time.perf_counter()
    fwd0, _ = _counters(problem)
    n_forward_extra = 0

    if scheme != "experiments":
        # Длина вектора весов — от одной оценки невязок в референсной точке.
        m = int(np.asarray(problem.residual(phi0)).reshape(-1).size)
        with_objective = getattr(problem, "with_objective", None)
        if not callable(with_objective):
            raise TypeError(
                "bootstrap_experiments: random-weights схема требует "
                "problem.with_objective(objective) (CalibrationProblem); "
                f"получен {type(problem).__name__}."
            )

    phi_samples = np.full((n_draws, phi0.size), np.nan, dtype=np.float64)
    losses: List[Optional[float]] = []
    statuses: List[str] = []
    messages: List[str] = []
    weights: List[Any] = []

    for i, child in enumerate(children):
        rng = np.random.default_rng(child)
        if scheme == "experiments":
            from neuraltide.optimization.problem import MultiExperimentProblem

            idx = rng.integers(0, len(experiments), size=len(experiments))
            # Подвыборка с повторением: копии с уникальными служебными
            # именами (replace — исходные Experiment не мутируют).
            boot_experiments = [
                replace(
                    experiments[int(j)],
                    name=_boot_experiment_name(i, k, experiments[int(j)].name),
                )
                for k, j in enumerate(idx)
            ]
            boot = MultiExperimentProblem(
                problem.network,
                problem.parameter_space,
                boot_experiments,
                grad_mode=getattr(problem, "grad_mode", "assign"),
            )
            weights.append(
                [int(v) for v in np.bincount(idx, minlength=len(experiments))]
            )
            counter_probe = boot
        else:
            w = rng.exponential(1.0, size=m)
            boot = problem.with_objective(_WeightedObjective(problem.objective, w))
            weights.append([float(v) for v in w])
            counter_probe = None

        try:
            result = evaluator_fn(boot, phi0)
            phi_b = np.asarray(result.base.phi_best, dtype=np.float64).reshape(-1)
            loss_b = float(result.base.loss_best)
            if phi_b.shape[0] != phi0.size or not np.all(np.isfinite(phi_b)):
                raise ValueError(
                    "evaluator вернул phi_best неподходящего размера "
                    f"({phi_b.shape[0]}) или нефинитный"
                )
            status = "ok"
            message = str(getattr(result, "message", ""))
        except Exception as exc:  # noqa: BLE001 — draw не должен ронять прогон
            phi_b = np.full(phi0.size, np.nan)
            loss_b = None
            status = "failed"
            message = f"{type(exc).__name__}: {exc}"
        phi_samples[i] = phi_b
        losses.append(loss_b)
        statuses.append(status)
        messages.append(message)
        if counter_probe is not None:
            n_forward_extra += int(getattr(counter_probe, "n_forward", 0))

    fwd1, _ = _counters(problem)
    if scheme == "experiments":
        n_forward_total = n_forward_extra
    else:
        n_forward_total = fwd1 - fwd0

    theta_samples = (
        _collect_theta_samples(space, phi_samples) if space is not None else None
    )
    notes = (
        f"Схема '{scheme}' ({n_draws} draw, seed={seed}). Стартовая точка "
        "каждой draw — явно захваченный phi0 (рефиты могут мутировать "
        "модель; скрытое состояние между draw не используется). "
        + (
            "Покрытие опирается на приблизительную iid-ность экспериментов; "
            "при сильной зависимости экспериментов интервалы занижены."
            if scheme == "experiments"
            else "Веса w ~ Exp(1) (байесовский bootstrap) на элементах "
            "невязок; эквивалентно повторному взвешиванию данных для "
            "гауссовых невязок."
        )
    )
    return BootstrapResult(
        scheme=scheme,
        n_draws=n_draws,
        seed=seed,
        phi0=phi0,
        phi_samples=phi_samples,
        theta_samples=theta_samples,
        losses=losses,
        statuses=statuses,
        messages=messages,
        weights=weights,
        evaluator=evaluator_name,
        n_forward=n_forward_total,
        wall_time=time.perf_counter() - t_begin,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# residual bootstrap: инверсия residual → ошибка модели
# ---------------------------------------------------------------------------


def _invert_residual_block(spec, block: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ошибка модели e (в kind-пространстве) из блока невязок.

    Инвертирует нормировку ResidualLoss: t_i = |r_i|·σ·√(N/(weight·mask_i)),
    |e| = inversion_psi(t). Возвращает (e, unmasked); на маскированных
    позициях e = NaN (исключены из пула ресемплинга).
    """
    shape = block.shape
    mask = _spec_mask(spec, shape)
    if mask is None:
        unmasked = np.ones(shape, dtype=bool)
    else:
        unmasked = mask
    if spec.mask is not None:
        n_norm = float(np.sum(np.asarray(spec.mask, dtype=np.float64)))
        mask_elements = np.broadcast_to(np.asarray(spec.mask, dtype=np.float64), shape)
    else:
        n_norm = float(block.size)
        mask_elements = np.ones(shape, dtype=np.float64)
    n_norm = max(n_norm, 1e-12)
    t = np.full(shape, np.nan, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = spec.sigma * np.sqrt(n_norm / spec.weight)
        t[unmasked] = (
            np.abs(block[unmasked])
            * scale
            / np.sqrt(np.maximum(mask_elements[unmasked], 1e-12))
        )
    sign = np.sign(block)
    delta = spec.delta
    if spec.kind in ("l2", "relative", "msle"):
        abs_e = t
    elif spec.kind == "huber":
        abs_e = np.where(t <= delta, t, t**2 / (2.0 * delta) + 0.5 * delta)
    elif spec.kind == "pseudo_huber":
        abs_e = delta * np.sqrt(
            np.maximum((1.0 + t**2 / (2.0 * delta**2)) ** 2 - 1.0, 0.0)
        )
    else:
        raise ValueError(
            f"residual_bootstrap: kind '{spec.kind}' не поддержан "
            "(нужна инверсия psi); поддержаны l2/relative/msle/huber/"
            "pseudo_huber."
        )
    e = sign * abs_e
    e = np.where(unmasked, e, np.nan)
    return e, unmasked


def _shift_target(spec, pred: np.ndarray, e_star: np.ndarray) -> np.ndarray:
    """
    Псевдо-наблюдение target* = pred − e* (данные = прогноз + шум).

    Для kind 'relative' ошибка — относительная: target* = pred − e*·(
    |target| + eps); для 'msle' — в лог-пространстве: target* =
    expm1(log1p(pred + eps) − e*) − eps (с обрезкой снизу нулём). Маскированные
    позиции (e* = 0) дают target* = pred — они исключены маской loss.
    """
    if spec.kind in ("l2", "huber", "pseudo_huber"):
        target_star = pred - e_star
    elif spec.kind == "relative":
        target = np.asarray(spec.target, dtype=np.float64)
        target_star = pred - e_star * (np.abs(target) + spec.eps)
    elif spec.kind == "msle":
        target_star = np.expm1(np.log1p(pred + spec.eps) - e_star) - spec.eps
        target_star = np.maximum(target_star, 0.0)
    else:
        raise ValueError(f"residual_bootstrap: kind '{spec.kind}' не поддержан.")
    return target_star


def residual_bootstrap(
    problem,
    *,
    n_draws: int,
    seed: Optional[int],
) -> BootstrapResult:
    """
    Residual (pairs-style) bootstrap: target* = pred − e*.

    В референсной точке phi0 = problem.pack() строится прогноз pred и
    вектор невязок; по блокам ObservationSpec ошибка модели e восстанавливается
    инверсией residual-нормировки (точно для kind l2/relative/msle/huber/
    pseudo_huber; spec.readout не поддержан — ValueError). Для каждой draw:
    центрированные e центрируются по среднему, ресемплируются iid с
    возвращением по немаскированным элементам, и задача с objective над
    target* доводится дефолтным evaluator (minimize_lbfgsb, ленивый
    импорт) от явного phi0.

    Честные оговорки (notes): ресемплинг iid игнорирует автокорреляцию и
    гетероскедастичность остатков; корректность опирается на адекватность
    подгонки (остаток ≈ шум); центрирование — по простому среднему;
    масштабная поправка σ·√(N/weight) предполагает вид нормировки
    ResidualLoss; для 'msle' псевдо-цели обрезаются снизу нулём.

    Args:
        problem: CalibrationProblem с ResidualLoss-like objective.
        n_draws: число draw (>= 1).
        seed: seed np.random.SeedSequence.

    Returns:
        BootstrapResult (scheme='residual', weights — None).
    """
    n_draws = int(n_draws)
    if n_draws < 1:
        raise ValueError(
            f"residual_bootstrap: n_draws должен быть >= 1, got {n_draws}."
        )
    from neuraltide.optimization.objectives import ObservationSpec, ResidualLoss

    evaluator_fn, evaluator_name = _resolve_evaluator(None)
    phi0 = _pack_current(problem, "residual_bootstrap")
    space = getattr(problem, "parameter_space", None)
    with_objective = getattr(problem, "with_objective", None)
    if not callable(with_objective):
        raise TypeError(
            "residual_bootstrap: требуется problem.with_objective(objective) "
            f"(CalibrationProblem), получен {type(problem).__name__}."
        )
    blocks = _observation_blocks(problem)
    if blocks is None:
        raise ValueError(
            "residual_bootstrap: objective не предоставляет observations "
            "(ResidualLoss-метаданные) — восстановление ошибки модели по "
            "невязкам невозможно."
        )

    t_begin = time.perf_counter()
    fwd0, _ = _counters(problem)

    output = problem.simulate(phi0)
    rates0 = {
        str(pop): np.asarray(rates, dtype=np.float64)
        for pop, rates in output.firing_rates.items()
    }
    r0 = np.asarray(problem.residual(phi0), dtype=np.float64).reshape(-1)

    spec_plans = []
    offset = 0
    for spec, numel in blocks:
        if spec.readout is not None:
            raise ValueError(
                "residual_bootstrap: spec.readout не поддержан — "
                "инверсия ошибки возможна только в пространстве частот."
            )
        shape = tuple(spec.target.shape.as_list())
        block = r0[offset : offset + numel].reshape(shape)
        offset += numel
        pred = np.broadcast_to(rates0[spec.population], shape)
        e, unmasked = _invert_residual_block(spec, block)
        spec_plans.append((spec, e, unmasked, pred))

    ss = np.random.SeedSequence(seed)
    children = ss.spawn(n_draws)

    phi_samples = np.full((n_draws, phi0.size), np.nan, dtype=np.float64)
    losses: List[Optional[float]] = []
    statuses: List[str] = []
    messages: List[str] = []

    for i, child in enumerate(children):
        rng = np.random.default_rng(child)
        new_specs = []
        try:
            for spec, e, unmasked, pred in spec_plans:
                pool = e[unmasked & np.isfinite(e)]
                if pool.size < 2:
                    raise ValueError(
                        f"spec '{spec.name}': меньше 2 немаскированных "
                        "остатков — ресемплинг невозможен."
                    )
                centered = pool - float(np.mean(pool))
                idx = rng.integers(0, centered.size, size=centered.size)
                e_star = np.zeros_like(e)
                e_star[unmasked] = centered[idx]
                target_star = _shift_target(spec, pred, e_star)
                new_specs.append(
                    ObservationSpec(
                        name=spec.name,
                        population=spec.population,
                        target=target_star,
                        sigma=spec.sigma,
                        mask=spec.mask,
                        weight=spec.weight,
                        kind=spec.kind,
                        eps=spec.eps,
                        delta=spec.delta,
                    )
                )
            boot = with_objective(ResidualLoss(new_specs))
            result = evaluator_fn(boot, phi0)
            phi_b = np.asarray(result.base.phi_best, dtype=np.float64).reshape(-1)
            loss_b = float(result.base.loss_best)
            if phi_b.shape[0] != phi0.size or not np.all(np.isfinite(phi_b)):
                raise ValueError(
                    "evaluator вернул phi_best неподходящего размера "
                    f"({phi_b.shape[0]}) или нефинитный"
                )
            status = "ok"
            message = str(getattr(result, "message", ""))
        except Exception as exc:  # noqa: BLE001 — draw не должен ронять прогон
            phi_b = np.full(phi0.size, np.nan)
            loss_b = None
            status = "failed"
            message = f"{type(exc).__name__}: {exc}"
        phi_samples[i] = phi_b
        losses.append(loss_b)
        statuses.append(status)
        messages.append(message)

    fwd1, _ = _counters(problem)
    theta_samples = (
        _collect_theta_samples(space, phi_samples) if space is not None else None
    )
    notes = (
        "Residual bootstrap: target* = pred − e*, e* — ресемплированные "
        "центрированные (по простому среднему) ошибки модели по "
        "немаскированным элементам; iid-ресемплинг ИГНОРИРУЕТ "
        "автокорреляцию и гетероскедастичность остатков. Корректность "
        "опирается на адекватность подгонки (остаток ≈ шум). Масштаб "
        "σ·√(N/weight) соответствует нормировке ResidualLoss; kind "
        "'relative'/'msle' инвертируются в своём пространстве ошибки; "
        "'msle' обрезает псевдо-цели снизу нулём. Старт каждой draw — "
        "явный phi0."
    )
    return BootstrapResult(
        scheme="residual",
        n_draws=n_draws,
        seed=seed,
        phi0=phi0,
        phi_samples=phi_samples,
        theta_samples=theta_samples,
        losses=losses,
        statuses=statuses,
        messages=messages,
        weights=[None] * n_draws,
        evaluator=evaluator_name,
        n_forward=fwd1 - fwd0,
        wall_time=time.perf_counter() - t_begin,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# DTO и функция: parameter ensemble
# ---------------------------------------------------------------------------


@dataclass
class EnsembleResult(_JsonDTO):
    """
    Ансамбль допустимых решений.

    Атрибуты:
        phi_samples: [n_samples, n] — φ готовых решений (порядок results);
        theta_samples: {full_name: [n_samples, *entry.shape]} физические θ
            (из base.theta_best результатов; None, если хотя бы один
            результат их не предоставляет — распакуйте φ через
            ParameterSpace снаружи);
        mean / std: среднее/СКО (ddof=0) по ансамблю для φ-координат;
        cov: выборочная ковариация φ [n, n] (ddof=1) при n_samples >= 2,
            иначе None;
        losses: loss_best решений;
        n_samples / size: размеры;
        notes: происхождение выборок и ограничения.
    """

    phi_samples: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    theta_samples: Optional[Dict[str, np.ndarray]] = None
    mean: np.ndarray = field(default_factory=lambda: np.zeros(0))
    std: np.ndarray = field(default_factory=lambda: np.zeros(0))
    cov: Optional[np.ndarray] = None
    losses: List[Optional[float]] = field(default_factory=list)
    n_samples: int = 0
    size: int = 0
    notes: str = ""

    def __post_init__(self) -> None:
        self.phi_samples = np.asarray(self.phi_samples, dtype=np.float64)
        self.mean = np.asarray(self.mean, dtype=np.float64)
        self.std = np.asarray(self.std, dtype=np.float64)
        if self.cov is not None:
            self.cov = np.asarray(self.cov, dtype=np.float64)
        if self.theta_samples is not None:
            self.theta_samples = {
                str(k): np.asarray(v, dtype=np.float64)
                for k, v in self.theta_samples.items()
            }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phi_samples": self.phi_samples.tolist(),
            "theta_samples": (
                None
                if self.theta_samples is None
                else {k: v.tolist() for k, v in self.theta_samples.items()}
            ),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "cov": None if self.cov is None else self.cov.tolist(),
            "losses": _clean_tree(self.losses),
            "n_samples": int(self.n_samples),
            "size": int(self.size),
            "notes": str(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EnsembleResult":
        theta = data.get("theta_samples")
        cov = data.get("cov")
        return cls(
            phi_samples=np.asarray(data.get("phi_samples", []), dtype=np.float64),
            theta_samples=(
                None
                if theta is None
                else {str(k): np.asarray(v, dtype=np.float64) for k, v in theta.items()}
            ),
            mean=np.asarray(data.get("mean", []), dtype=np.float64),
            std=np.asarray(data.get("std", []), dtype=np.float64),
            cov=None if cov is None else np.asarray(cov, dtype=np.float64),
            losses=[None if v is None else float(v) for v in data.get("losses", [])],
            n_samples=int(data.get("n_samples", 0)),
            size=int(data.get("size", 0)),
            notes=str(data.get("notes", "")),
        )


def parameter_ensemble(results) -> EnsembleResult:
    """
    Ансамбль φ/θ из готовых результатов оптимизации.

    Принимает любые объекты с полями phi_best/loss_best/theta_best —
    например OptimizationResult или base из LocalOptimizerResult/
    MultistartResult (передавайте сами результаты, образующие ансамбль:
    несколько LocalOptimizerResult, base лучших стартов и т.п.). theta
    берутся из theta_best; если хотя бы один результат их не имеет —
    theta_samples=None с пояснением в notes (φ всегда на месте).

    cov — выборочная ковариация (ddof=1) при n_samples >= 2 (np.cov),
    иначе None. std — СКО с ddof=0.
    """
    items = list(results)
    if not items:
        raise ValueError(
            "parameter_ensemble: непустая последовательность результатов обязательна."
        )
    bases = [getattr(item, "base", item) for item in items]
    phi_rows: List[np.ndarray] = []
    losses: List[Optional[float]] = []
    sizes: List[int] = []
    for base in bases:
        phi = np.asarray(getattr(base, "phi_best"), dtype=np.float64).reshape(-1)
        phi_rows.append(phi)
        sizes.append(int(phi.size))
        loss = getattr(base, "loss_best", None)
        losses.append(None if loss is None else float(loss))
    if len(set(sizes)) != 1:
        raise ValueError(
            f"parameter_ensemble: результаты имеют разную размерность φ: "
            f"{sorted(set(sizes))}."
        )

    # theta: ключи — от первого результата, предоставляющего theta_best;
    # пропуски и недостающие ключи — NaN-подушкой соответствующей формы.
    theta_best_list = [getattr(base, "theta_best", None) for base in bases]
    theta_samples: Optional[Dict[str, np.ndarray]] = None
    if any(tb is not None for tb in theta_best_list):
        reference = next(tb for tb in theta_best_list if tb is not None)
        cols: Dict[str, List[np.ndarray]] = {str(k): [] for k in reference}
        for theta_best in theta_best_list:
            for key, shape in ((k, np.asarray(v).shape) for k, v in reference.items()):
                value = None if theta_best is None else theta_best.get(key)
                cols[key].append(
                    np.full(shape, np.nan)
                    if value is None
                    else np.asarray(value, dtype=np.float64)
                )
        theta_samples = {k: np.stack(v, axis=0) for k, v in cols.items()}

    phi_samples = np.stack(phi_rows, axis=0)
    n_samples, size = phi_samples.shape
    cov = np.cov(phi_samples, rowvar=False) if n_samples >= 2 and size >= 1 else None
    if cov is not None:
        cov = np.atleast_2d(np.asarray(cov, dtype=np.float64))
    notes = (
        "Один элемент выборки — один переданный результат (phi_best). "
        "theta_samples из base.theta_best; cov — выборочная ковариация "
        "(ddof=1) при n_samples >= 2, иначе None; std — ddof=0. "
        + (
            "Все результаты предоставляют theta_best."
            if theta_samples is not None
            else "Часть результатов без theta_best — theta_samples=None; "
            "распакуйте φ через ParameterSpace снаружи."
        )
    )
    return EnsembleResult(
        phi_samples=phi_samples,
        theta_samples=theta_samples,
        mean=phi_samples.mean(axis=0),
        std=phi_samples.std(axis=0),
        cov=cov,
        losses=losses,
        n_samples=n_samples,
        size=size,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# DTO и функция: predictive intervals
# ---------------------------------------------------------------------------


@dataclass
class PredictiveResult(_JsonDTO):
    """
    Прогнозные интервалы по ансамблю φ.

    Атрибуты:
        level: уровень интервалов;
        populations: имена популяций (firing_rates сети);
        lower / upper: {population: [T, n_units]} квантили уровня
            (1 ± level)/2 по ансамблю прогонов;
        central: {population: [T, n_units]} поканальная медиана
            (центральная траектория);
        mean: {population: [T, n_units]} среднее по ансамблю;
        n_samples: число использованных φ (нефинитные строки пропущены);
        n_skipped_nonfinite: число пропущенных нефинитных строк;
        n_forward / wall_time: дельта счётчиков задачи;
        notes: определения и ограничения.
    """

    level: float = 0.95
    populations: List[str] = field(default_factory=list)
    lower: Dict[str, np.ndarray] = field(default_factory=dict)
    upper: Dict[str, np.ndarray] = field(default_factory=dict)
    central: Dict[str, np.ndarray] = field(default_factory=dict)
    mean: Dict[str, np.ndarray] = field(default_factory=dict)
    n_samples: int = 0
    n_skipped_nonfinite: int = 0
    n_forward: int = 0
    wall_time: float = 0.0
    notes: str = ""

    def __post_init__(self) -> None:
        for attr in ("lower", "upper", "central", "mean"):
            setattr(
                self,
                attr,
                {
                    str(k): np.asarray(v, dtype=np.float64)
                    for k, v in getattr(self, attr).items()
                },
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": float(self.level),
            "populations": [str(p) for p in self.populations],
            "lower": {k: v.tolist() for k, v in self.lower.items()},
            "upper": {k: v.tolist() for k, v in self.upper.items()},
            "central": {k: v.tolist() for k, v in self.central.items()},
            "mean": {k: v.tolist() for k, v in self.mean.items()},
            "n_samples": int(self.n_samples),
            "n_skipped_nonfinite": int(self.n_skipped_nonfinite),
            "n_forward": int(self.n_forward),
            "wall_time": float(self.wall_time),
            "notes": str(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PredictiveResult":
        def _maps(name):
            return {
                str(k): np.asarray(v, dtype=np.float64)
                for k, v in data.get(name, {}).items()
            }

        return cls(
            level=float(data.get("level", 0.95)),
            populations=[str(p) for p in data.get("populations", [])],
            lower=_maps("lower"),
            upper=_maps("upper"),
            central=_maps("central"),
            mean=_maps("mean"),
            n_samples=int(data.get("n_samples", 0)),
            n_skipped_nonfinite=int(data.get("n_skipped_nonfinite", 0)),
            n_forward=int(data.get("n_forward", 0)),
            wall_time=float(data.get("wall_time", 0.0)),
            notes=str(data.get("notes", "")),
        )


def predictive_intervals(
    problem,
    phi_samples,
    *,
    level: float = 0.95,
) -> PredictiveResult:
    """
    Прогнозные интервалы firing_rates по ансамблю φ.

    Каждая строка phi_samples прогоняется через problem.simulate(phi)
    (явный phi — контракт мутации CalibrationProblem); квантили уровня
    level берутся per-(population, time, unit) по ансамблю траекторий и
    монотонны по построению (lower <= central <= upper). Центральная
    траектория — поканальная медиана; batch обязан быть 1 (контракт
    NetworkRNN). Нефинитные строки phi_samples пропускаются с записью в
    n_skipped_nonfinite.

    Args:
        problem: CalibrationProblem (с simulate).
        phi_samples: [S, n] (или [n] — одна точка) векторов φ.
        level: уровень интервалов в (0, 1).

    Returns:
        PredictiveResult.
    """
    level = float(level)
    if not 0.0 < level < 1.0:
        raise ValueError(
            f"predictive_intervals: level должен быть в (0, 1), got {level}."
        )
    arr = np.atleast_2d(np.asarray(phi_samples, dtype=np.float64))
    expected = int(getattr(problem, "size", arr.shape[1]))
    if arr.shape[1] != expected:
        raise ValueError(
            f"predictive_intervals: phi_samples имеют {arr.shape[1]} "
            f"координат, ожидается {expected}."
        )

    t0 = time.perf_counter()
    fwd0, _ = _counters(problem)
    finite_rows = np.all(np.isfinite(arr), axis=1)
    skipped = int(np.count_nonzero(~finite_rows))

    runs: Dict[str, List[np.ndarray]] = {}
    for row in arr[finite_rows]:
        output = problem.simulate(np.asarray(row, dtype=np.float64))
        for pop, rates in output.firing_rates.items():
            a = np.asarray(rates, dtype=np.float64)
            if a.ndim != 3 or a.shape[0] != 1:
                raise ValueError(
                    "predictive_intervals: firing_rates должны иметь форму "
                    f"[1, T, n_units] (batch=1), получена {a.shape} для "
                    f"популяции '{pop}'."
                )
            runs.setdefault(str(pop), []).append(a[0])

    if not runs:
        raise ValueError(
            "predictive_intervals: нет ни одной нефинитной строки "
            "phi_samples — интервалы не определены."
        )

    q_lo = 0.5 * (1.0 - level)
    q_hi = 1.0 - q_lo
    lower, upper, central, mean = {}, {}, {}, {}
    for pop, stack_list in runs.items():
        stack = np.stack(stack_list, axis=0)
        lower[pop] = np.quantile(stack, q_lo, axis=0)
        upper[pop] = np.quantile(stack, q_hi, axis=0)
        central[pop] = np.quantile(stack, 0.5, axis=0)
        mean[pop] = stack.mean(axis=0)

    fwd1, _ = _counters(problem)
    return PredictiveResult(
        level=level,
        populations=sorted(runs),
        lower=lower,
        upper=upper,
        central=central,
        mean=mean,
        n_samples=int(sum(len(v) for v in runs.values())) // len(runs),
        n_skipped_nonfinite=skipped,
        n_forward=fwd1 - fwd0,
        wall_time=time.perf_counter() - t0,
        notes=(
            "Квантили per-(population, time, unit) по ансамблю прогонов "
            "simulate(phi); квантилы монотонны по построению. central — "
            "поканальная медиана, mean — среднее. Нефинитные строки "
            "phi_samples пропущены (n_skipped_nonfinite)."
        ),
    )


# ---------------------------------------------------------------------------
# DTO и функции: noise sensitivity
# ---------------------------------------------------------------------------


@dataclass
class NoiseFactorResult(_JsonDTO):
    """
    Пересчитанная Laplace-ковариация при масштабе шума σ → f·σ.

    Атрибуты:
        sigma_factor: f;
        covariance: [n, n] ковариация при f;
        std: sqrt(diag(covariance));
        phi_intervals: {flat_name: (lo, hi)} маргинальные интервалы в
            φ-координатах (phi_map ± z·std; физические θ — через
            transform.forward снаружи);
        pd / jitter_used: результаты PD-проверки (как в LaplaceResult).
    """

    sigma_factor: float = 1.0
    covariance: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    std: np.ndarray = field(default_factory=lambda: np.zeros(0))
    phi_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    pd: bool = True
    jitter_used: Optional[float] = None

    def __post_init__(self) -> None:
        self.sigma_factor = float(self.sigma_factor)
        self.covariance = np.asarray(self.covariance, dtype=np.float64)
        self.std = np.asarray(self.std, dtype=np.float64)
        self.phi_intervals = {
            str(k): (float(v[0]), float(v[1])) for k, v in self.phi_intervals.items()
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sigma_factor": float(self.sigma_factor),
            "covariance": self.covariance.tolist(),
            "std": self.std.tolist(),
            "phi_intervals": {k: [v[0], v[1]] for k, v in self.phi_intervals.items()},
            "pd": bool(self.pd),
            "jitter_used": self.jitter_used,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NoiseFactorResult":
        return cls(
            sigma_factor=float(data.get("sigma_factor", 1.0)),
            covariance=np.asarray(data.get("covariance", []), dtype=np.float64),
            std=np.asarray(data.get("std", []), dtype=np.float64),
            phi_intervals={
                str(k): (float(v[0]), float(v[1]))
                for k, v in data.get("phi_intervals", {}).items()
            },
            pd=bool(data.get("pd", True)),
            jitter_used=data.get("jitter_used"),
        )


@dataclass
class SensitivityResult(_JsonDTO):
    """
    Чувствительность Laplace-результата к масштабу шума.

    Атрибуты:
        entries: per-factor результаты (порядок sigma_factors);
        phi_map / param_names / level / z: из исходного LaplaceResult;
        notes: точное масштабирование для плоского приора и оговорки
            про prior.
    """

    entries: List[NoiseFactorResult] = field(default_factory=list)
    phi_map: np.ndarray = field(default_factory=lambda: np.zeros(0))
    param_names: List[str] = field(default_factory=list)
    level: float = 0.95
    z: float = _Z95
    notes: str = ""

    def __post_init__(self) -> None:
        self.phi_map = np.asarray(self.phi_map, dtype=np.float64)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entries": [e.to_dict() for e in self.entries],
            "phi_map": self.phi_map.tolist(),
            "param_names": [str(n) for n in self.param_names],
            "level": float(self.level),
            "z": float(self.z),
            "notes": str(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SensitivityResult":
        return cls(
            entries=[NoiseFactorResult.from_dict(d) for d in data.get("entries", [])],
            phi_map=np.asarray(data.get("phi_map", []), dtype=np.float64),
            param_names=[str(n) for n in data.get("param_names", [])],
            level=float(data.get("level", 0.95)),
            z=float(data.get("z", _Z95)),
            notes=str(data.get("notes", "")),
        )


def noise_sensitivity(
    laplace_result: LaplaceResult,
    sigma_factors=(0.5, 1.0, 2.0),
) -> SensitivityResult:
    """
    Масштабирование Laplace-ковариации при σ → f·σ.

    Невязки нормированы на σ, поэтому GN-precision масштабируется как
    G/f², где G = precision_gn (JᵀJ для 'gn', Гессиан для 'fd-hessian'):
    covariance(f) = (G/f² + P)⁻¹, P = prior_precision (None/0 — плоский
    приор). Для плоского приора масштабирование ТОЧНОЕ: covariance(f) =
    f²·covariance(1), std(f) = f·std(1) (∝ 1/σ); с приором масштабируется
    только GN-часть, std сублинейно по f. Интервалы — маргинальные, в
    φ-координатах (phi_map ± z·std(f)); физические θ — через
    transform.forward снаружи.

    Args:
        laplace_result: результат laplace_approximation.
        sigma_factors: последовательность f > 0 (по умолчанию
            (0.5, 1.0, 2.0)).

    Returns:
        SensitivityResult.
    """
    if not isinstance(laplace_result, LaplaceResult):
        raise TypeError(
            "noise_sensitivity: ожидается LaplaceResult, получен "
            f"{type(laplace_result).__name__}."
        )
    factors = [float(f) for f in sigma_factors]
    if not factors or any(f <= 0.0 for f in factors):
        raise ValueError(
            f"noise_sensitivity: sigma_factors должны быть > 0, got {factors}."
        )
    g_mat = np.asarray(laplace_result.precision_gn, dtype=np.float64)
    prior = laplace_result.prior_precision
    phi = np.asarray(laplace_result.phi_map, dtype=np.float64)
    z = float(laplace_result.z)
    names = list(laplace_result.param_names)

    entries: List[NoiseFactorResult] = []
    for factor in factors:
        scaled = g_mat / (factor * factor)
        precision = scaled if prior is None else _symmetrize(scaled + prior)
        covariance, jitter, pd = _pd_inverse(precision)
        std = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        phi_intervals = {
            names[i]: (float(phi[i] - z * std[i]), float(phi[i] + z * std[i]))
            for i in range(phi.size)
        }
        entries.append(
            NoiseFactorResult(
                sigma_factor=factor,
                covariance=covariance,
                std=std,
                phi_intervals=phi_intervals,
                pd=pd,
                jitter_used=None if not np.isfinite(jitter) else float(jitter),
            )
        )
    notes = (
        "covariance(f) = (G/f² + P)⁻¹, G = precision_gn; для плоского "
        "приора (P = 0) точно covariance(f) = f²·cov(1), std(f) = "
        "f·std(1) (∝ 1/σ); с приором масштабируется только GN-часть. "
        "Интервалы — маргинальные в φ-координатах; физические θ через "
        "transform.forward снаружи."
    )
    return SensitivityResult(
        entries=entries,
        phi_map=phi,
        param_names=names,
        level=float(laplace_result.level),
        z=z,
        notes=notes,
    )
