"""
Наблюдения и нормированные невязки для калибровки (этап P0).

Разделяет два интерфейса оптимизации:

- скалярный ``loss`` — для общих методов (L-BFGS-B, Adam, ...);
- вектор ``residual`` — честные невязки для nonlinear least squares
  (Gauss–Newton / LM / TRF), такие что
  ``0.5 * sum(residual ** 2) == loss``.

ObservationSpec описывает одно наблюдение: какая популяция сравнивается с
какой целевой траекторией, с какой моделью шума (sigma), весом, маской и
видом функции потерь. ResidualLoss агрегирует набор спецификаций.

Этап P2 добавляет робастные виды штрафа для тяжёлых выбросов
(kind 'cauchy' и 'tukey', масштаб scale c) и допускает тензорный вес
observation (0-d Tensor/Variable — например, обучаемый или адаптивный).

Семантика нормировки — masked mean (как у MSELoss/MSLELoss): вклад
наблюдения равен

    weight * sum(mask * psi(error)) / (N * sigma**2),

где psi(e) — штрафная функция вида, а N — нормировка: число
выбранных элементов ``max(sum(broadcast(mask)), eps)`` при заданной маске
(маска расширяется до формы траектории, поэтому маска из единиц любой
broadcast-формы эквивалентна её отсутствию) и число элементов траектории
без неё. Умножение маски на константу сокращается числителем и
знаменателем.

Нормировка ``normalization='mean'`` (по умолчанию) удобна для оптимизации:
масштаб loss не зависит от длины записи. Для статистики (FIM, Лаплас,
профили правдоподобия) нужен полный NLL — ``normalization='sum'`` (N = 1):
тогда для гауссова шума ``0.5 * ||r||^2`` равна NLL с точностью до
константы. Модули identifiability/uq переводят невязки в эти единицы сами:
через ``likelihood_row_scale(problem)`` (множитель sqrt(N) на строку
невязок) и контекст ``likelihood_normalization(problem)``.

Residual-вектор согласован с этой нормировкой поэлементно:

    r_i = sqrt(weight * mask_i / (N * sigma**2)) * sign(e_i) * sqrt(2 psi(e_i)),

поэтому ``0.5 * ||r||^2`` воспроизводит loss точно (для l2/relative/msle,
где psi = 0.5 e^2, множитель ``sign(e) * sqrt(2 psi)`` тождественно равен e).

Формы:
    target: [batch, T, n_units] — целевая траектория наблюдаемой величины;
    mask: broadcastable к target (например [batch, T, 1] или [1, T, n]);
    predictions.firing_rates[population]: [batch, T, n_units].
"""

import contextlib
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np
import tensorflow as tf

import neuraltide.config
from neuraltide.core.types import TensorType, cast_if_needed
from neuraltide.training.losses import BaseLoss, _MASK_DENOM_EPS
from neuraltide.training.readouts import BaseReadout, IdentityReadout

__all__ = [
    "ObservationSpec",
    "ResidualLoss",
    "likelihood_row_scale",
    "likelihood_normalization",
]

_NORMALIZATIONS = ("mean", "sum")

_KINDS = ("l2", "relative", "msle", "huber", "pseudo_huber", "cauchy", "tukey")

# Общий stateless-инстанс на все спецификации без readout.
_IDENTITY_READOUT = IdentityReadout()


@dataclass
class ObservationSpec:
    """
    Одно наблюдение для ResidualLoss.

    Атрибуты:
        name: уникальное имя наблюдения (ключ в components/target);
        population: имя популяции в NetworkGraph (ключ firing_rates);
        target: целевая траектория [batch, T, n_units];
        sigma: масштаб шума измерений (> 0); вклад делится на sigma**2;
        mask: опциональная маска, broadcastable к target; семантика
            masked mean — масштаб маски значение не меняет;
        weight: неотрицательный вес наблюдения в суммарной loss; допускает
            float либо 0-d Tensor/Variable (тогда приводится к глобальному
            dtype и используется как есть — напр. обучаемый/адаптивный вес);
        kind: вид штрафа — 'l2' | 'relative' | 'msle' | 'huber' |
            'pseudo_huber' | 'cauchy' | 'tukey';
        readout: опциональный BaseReadout, применяется к pred и target;
        eps: защита от деления на ноль / домен log1p (kind-зависимый);
        delta: порог Хьюбера (> 0) для 'huber' и 'pseudo_huber';
        scale: масштаб c (> 0) для 'cauchy' и 'tukey'.

    Преобразование ошибки e и штраф psi(e) по видам:
        l2:           e = pred - target,           psi = 0.5 e^2
        relative:     e = (pred - target) / (|target| + eps),
                                                psi = 0.5 e^2
        msle:         e = log1p(pred + eps) - log1p(target + eps),
                                                psi = 0.5 e^2
        huber:        e = pred - target,
            psi = 0.5 e^2 при |e| <= delta, иначе delta (|e| - 0.5 delta)
        pseudo_huber: e = pred - target,
            psi = delta^2 (sqrt(1 + (e/delta)^2) - 1)
        cauchy:       e = pred - target,
            psi = c^2/2 * log(1 + (e/c)^2)
        tukey:        e = pred - target,
            psi = c^2/6 * (1 - (1 - (e/c)^2)^3) при |e| <= c, иначе c^2/6

    Для 'msle' предсказания должны быть неотрицательными (частоты разрядов
    таковы по построению); eps защищает домен log1p около нуля. Cauchy и
    Tukey (biweight) ограничивают вклад выбросов: у cauchy хвост
    логарифмический, у tukey вне |e| > c штраф константен (нулевой вклад
    в градиент). Общая формула residual сохраняется:
    sign(e) * sqrt(2 psi) — для cauchy/tukey она не тождественна e, но
    0.5 * ||r||^2 == loss воспроизводится по-прежнему точно.
    """

    name: str
    population: str
    target: TensorType
    sigma: float = 1.0
    mask: Optional[TensorType] = None
    weight: float = 1.0
    kind: str = "l2"
    readout: Optional[BaseReadout] = None
    eps: float = 1e-8
    delta: float = 1.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("ObservationSpec: 'name' must be a non-empty string.")
        if not isinstance(self.population, str) or not self.population:
            raise ValueError(
                "ObservationSpec: 'population' must be a non-empty string."
            )
        if self.kind not in _KINDS:
            raise ValueError(
                f"ObservationSpec: unknown kind '{self.kind}'. "
                f"Available kinds: {list(_KINDS)}."
            )
        if self.sigma <= 0.0:
            raise ValueError("ObservationSpec: 'sigma' must be positive.")
        if self.delta <= 0.0:
            raise ValueError("ObservationSpec: 'delta' must be positive.")
        if self.scale <= 0.0:
            raise ValueError("ObservationSpec: 'scale' must be positive.")
        if self.eps < 0.0:
            raise ValueError("ObservationSpec: 'eps' must be non-negative.")
        dtype = neuraltide.config.get_dtype()
        self.weight = self._normalize_weight(self.weight, dtype)
        self.target = cast_if_needed(tf.convert_to_tensor(self.target), dtype)
        if self.mask is not None:
            self.mask = cast_if_needed(tf.convert_to_tensor(self.mask), dtype)

    @staticmethod
    def _normalize_weight(weight, dtype) -> Any:
        """
        Валидация веса: float как раньше, либо 0-d Tensor/Variable.

        Объект сохраняется как есть: Variable остаётся «живой» (чтение
        текущего значения — в точках использования через _weight_tensor),
        0-d Tensor — константой. Неотрицательность проверяется на месте.
        """
        del dtype  # приведение dtype — в _weight_tensor (на месте использования)
        if isinstance(weight, (tf.Tensor, tf.Variable)):
            w = tf.convert_to_tensor(weight)
            if w.shape.rank != 0:
                raise ValueError(
                    "ObservationSpec: tensor 'weight' must be 0-d "
                    f"(scalar), got shape {w.shape}."
                )
            if bool(tf.reduce_any(w < 0.0)):
                raise ValueError("ObservationSpec: 'weight' must be non-negative.")
            return weight
        w = float(weight)
        if w < 0.0:
            raise ValueError("ObservationSpec: 'weight' must be non-negative.")
        return w

    @staticmethod
    def _weight_tensor(spec: "ObservationSpec", dtype) -> TensorType:
        """Вес наблюдения как тензор вычислительного dtype (live Variable)."""
        if isinstance(spec.weight, (int, float)):
            # tf.convert_to_tensor(float) дал бы float32 с округлением —
            # константу строим сразу в целевом dtype.
            return tf.constant(spec.weight, dtype=dtype)
        return cast_if_needed(tf.convert_to_tensor(spec.weight), dtype)


class ResidualLoss(BaseLoss):
    """
    Сумма нормированных невязок по набору ObservationSpec.

    Args:
        observations: последовательность ObservationSpec; имена должны
            быть уникальны.
        normalization: 'mean' (по умолчанию) — вклад наблюдения делится на
            число элементов N (masked mean); 'sum' — без деления (N = 1),
            тогда для гауссова шума 0.5·||r||² — полный NLL. Атрибут можно
            временно переключать (см. likelihood_normalization).

    Скалярный интерфейс:
        __call__(predictions, model) — полная loss (сумма вкладов);
        components(predictions, model) — dict {spec.name: scalar};
        per_step_loss(firing_rates, target, mask=None) — аддитивный вклад
            временного шага (полная loss == sum_t per_step_loss);
        target — dict {spec.name: target} (совместимо с
            Trainer._extract_targets; для adjoint-режима имя наблюдения
            должно совпадать с именем популяции).

    Residual-интерфейс:
        residual(predictions) — плоский вектор невязок r, такой что
            0.5 * sum(r^2) == __call__(predictions, model)
        с учётом sigma/weight/mask/kind (см. модульную документацию).
    """

    def __init__(
        self, observations: Sequence[ObservationSpec], normalization: str = "mean"
    ):
        if normalization not in _NORMALIZATIONS:
            raise ValueError(
                f"ResidualLoss: normalization must be one of {_NORMALIZATIONS}, "
                f"got {normalization!r}."
            )
        self.normalization = normalization
        self.observations: List[ObservationSpec] = list(observations)
        if not self.observations:
            raise ValueError("ResidualLoss: at least one observation is required.")
        names = [spec.name for spec in self.observations]
        if len(set(names)) != len(names):
            raise ValueError(
                f"ResidualLoss: observation names must be unique, got {names}."
            )

    # ------------------------------------------------------------------
    # Свойства для совместимости с Trainer / CompositeLoss
    # ------------------------------------------------------------------
    @property
    def target(self) -> Dict[str, TensorType]:
        """dict {spec.name: target} — формат Trainer._extract_targets."""
        return {spec.name: spec.target for spec in self.observations}

    @property
    def mask(self) -> Dict[str, TensorType]:
        """dict {spec.name: mask} для спецификаций с маской."""
        return {
            spec.name: spec.mask for spec in self.observations if spec.mask is not None
        }

    # ------------------------------------------------------------------
    # Вычисление ошибки и штрафа
    # ------------------------------------------------------------------
    @staticmethod
    def _readout(spec: ObservationSpec) -> BaseReadout:
        return spec.readout if spec.readout is not None else _IDENTITY_READOUT

    def _error(self, spec: ObservationSpec, pred: TensorType, tgt: TensorType):
        """Преобразованная ошибка e (после readout)."""
        pred_r = self._readout(spec)(pred)
        tgt_r = self._readout(spec)(tgt)
        if spec.kind in ("l2", "huber", "pseudo_huber", "cauchy", "tukey"):
            return pred_r - tgt_r
        if spec.kind == "relative":
            return (pred_r - tgt_r) / (tf.abs(tgt_r) + spec.eps)
        # msle
        return tf.math.log1p(pred_r + spec.eps) - tf.math.log1p(tgt_r + spec.eps)

    @staticmethod
    def _psi(spec: ObservationSpec, e: TensorType) -> TensorType:
        """Штрафная функция psi(e) (неотрицательная)."""
        if spec.kind in ("l2", "relative", "msle"):
            return 0.5 * e**2
        if spec.kind == "huber":
            a = tf.abs(e)
            quadratic = 0.5 * e**2
            linear = spec.delta * (a - 0.5 * spec.delta)
            return tf.where(a <= spec.delta, quadratic, linear)
        if spec.kind == "pseudo_huber":
            ratio = e / spec.delta
            return spec.delta**2 * (tf.sqrt(1.0 + ratio**2) - 1.0)
        if spec.kind == "cauchy":
            ratio = e / spec.scale
            return 0.5 * spec.scale**2 * tf.math.log1p(ratio**2)
        # tukey (biweight): квадратичная около нуля, константна при |e| > c
        a = tf.abs(e)
        ratio = e / spec.scale
        quadratic = spec.scale**2 / 6.0 * (1.0 - (1.0 - ratio**2) ** 3)
        constant = tf.constant(spec.scale**2 / 6.0, dtype=e.dtype)
        return tf.where(a <= spec.scale, quadratic, constant)

    @staticmethod
    def _count(spec: ObservationSpec, full_shape, dtype) -> TensorType:
        """
        Число элементов N полной траектории: max(sum(broadcast(mask)), eps)
        либо число элементов траектории (без маски). Маска расширяется до
        full_shape, поэтому маска из единиц любой broadcast-формы даёт то же
        N, что и её отсутствие.
        """
        eps_const = tf.constant(_MASK_DENOM_EPS, dtype=dtype)
        ones = tf.ones(full_shape, dtype=dtype)
        if spec.mask is None:
            return tf.maximum(tf.reduce_sum(ones), eps_const)
        m = cast_if_needed(spec.mask, dtype) * ones
        return tf.maximum(tf.reduce_sum(m), eps_const)

    def _norm(self, spec: ObservationSpec, psi: TensorType) -> TensorType:
        """
        Нормировка вклада: N полной траектории при normalization='mean',
        1 при 'sum'. psi — штраф полной траектории (его форма задаёт форму
        расширения маски).
        """
        if self.normalization == "sum":
            return tf.constant(1.0, dtype=psi.dtype)
        return self._count(spec, tf.shape(psi), psi.dtype)

    def _full_shape(self, spec: ObservationSpec):
        """Форма полной траектории наблюдения (для per_step_loss)."""
        shape = tf.shape(spec.target)
        if spec.mask is not None:
            shape = tf.broadcast_dynamic_shape(shape, tf.shape(spec.mask))
        return shape

    def likelihood_row_scale(self) -> np.ndarray:
        """
        Множители строк невязок для перевода в единицы полного NLL:
        sqrt(N_k) для элементов наблюдения k при normalization='mean'
        (1 при 'sum'). r_sum = scale * r, J_sum = scale[:, None] * J.

        Длина равна длине residual(predictions) при предсказаниях формы
        target (контракт ObservationSpec).
        """
        pieces: List[np.ndarray] = []
        for spec in self.observations:
            shape = self._full_shape(spec)
            numel = int(tf.reduce_prod(shape))
            if self.normalization == "sum":
                factor = 1.0
            else:
                n = self._count(spec, shape, tf.float64)
                factor = float(np.sqrt(float(n)))
            pieces.append(np.full(numel, factor, dtype=np.float64))
        return np.concatenate(pieces)

    def _spec_terms(self, spec: ObservationSpec, pred: TensorType, tgt: TensorType):
        """e, psi, mask (broadcastable) и нормировка N для полной траектории."""
        e = self._error(spec, pred, tgt)
        psi = self._psi(spec, e)
        if spec.mask is None:
            m = tf.ones_like(psi)
        else:
            m = cast_if_needed(spec.mask, psi.dtype)
        n = self._norm(spec, psi)
        return e, psi, m, n

    @staticmethod
    def _spec_loss(
        spec: ObservationSpec, psi: TensorType, m: TensorType, n: TensorType
    ) -> TensorType:
        """Вклад наблюдения: weight * sum(mask * psi) / (N * sigma^2)."""
        w = ObservationSpec._weight_tensor(spec, psi.dtype)
        return w * tf.reduce_sum(m * psi) / (n * spec.sigma**2)

    # ------------------------------------------------------------------
    # Скалярный интерфейс BaseLoss
    # ------------------------------------------------------------------
    def _rates(self, predictions, spec: ObservationSpec) -> TensorType:
        rates = predictions.firing_rates
        if spec.population not in rates:
            raise KeyError(
                f"ResidualLoss: population '{spec.population}' not found in "
                f"predictions.firing_rates (have {sorted(rates)})."
            )
        return rates[spec.population]

    def __call__(self, predictions, model) -> TensorType:
        dtype = neuraltide.config.get_dtype()
        total = tf.constant(0.0, dtype=dtype)
        for spec in self.observations:
            pred = self._rates(predictions, spec)
            _, psi, m, n = self._spec_terms(spec, pred, spec.target)
            total += self._spec_loss(spec, psi, m, n)
        return total

    def components(self, predictions, model) -> Dict[str, TensorType]:
        """Взвешенные вклады наблюдений по отдельности: {spec.name: scalar}."""
        out: Dict[str, TensorType] = {}
        for spec in self.observations:
            pred = self._rates(predictions, spec)
            _, psi, m, n = self._spec_terms(spec, pred, spec.target)
            out[spec.name] = self._spec_loss(spec, psi, m, n)
        return out

    def per_step_loss(
        self,
        firing_rates: Dict[str, TensorType],
        target: Dict[str, TensorType],
        mask: Optional[Dict[str, TensorType]] = None,
    ) -> TensorType:
        """
        Аддитивный вклад временного шага.

        Конвенция: полная loss == sum_t per_step_loss(rates_t, target_t).
        Нормировка N берётся от полной маски наблюдения (или полного
        target'а), поэтому сумма срезов воспроизводит полную loss; при
        передаче точных срезов маски через mask={spec.name: m_t} совпадение
        точное, без срезов — для масок, постоянных во времени (как у
        MSELoss/MSLELoss).
        """
        dtype = neuraltide.config.get_dtype()
        total = tf.constant(0.0, dtype=dtype)
        for spec in self.observations:
            pred_t = firing_rates.get(spec.population)
            tgt_t = target.get(spec.name)
            if pred_t is None or tgt_t is None:
                continue
            e = self._error(spec, pred_t, tgt_t)
            psi = self._psi(spec, e)

            mask_t: Optional[TensorType] = None
            if mask is not None and spec.name in mask:
                mask_t = mask[spec.name]
            elif spec.mask is not None:
                m_full = cast_if_needed(spec.mask, psi.dtype)
                if m_full.shape.rank == psi.shape.rank + 1:
                    # среднее по времени: точно для масок, постоянных во времени
                    mask_t = tf.reduce_mean(m_full, axis=1)
                else:
                    mask_t = m_full

            if mask_t is None:
                m = tf.ones_like(psi)
            else:
                m = cast_if_needed(mask_t, psi.dtype)
            if self.normalization == "sum":
                n = tf.constant(1.0, dtype=psi.dtype)
            else:
                n = self._count(spec, self._full_shape(spec), psi.dtype)
            total += self._spec_loss(spec, psi, m, n)
        return total

    # ------------------------------------------------------------------
    # Residual-интерфейс
    # ------------------------------------------------------------------
    def residual(self, predictions) -> TensorType:
        """
        Честный вектор невязок r: 0.5 * sum(r^2) == __call__(predictions).

        Поэлементная структура (см. документацию модуля):
            r_i = sqrt(weight * mask_i / (N sigma^2)) * sign(e_i) * sqrt(2 psi(e_i)).
        """
        pieces: List[TensorType] = []
        for spec in self.observations:
            pred = self._rates(predictions, spec)
            e = self._error(spec, pred, spec.target)
            psi = self._psi(spec, e)
            if spec.mask is None:
                m = tf.ones_like(psi)
            else:
                m = cast_if_needed(spec.mask, psi.dtype)
            n = self._norm(spec, psi)
            w = ObservationSpec._weight_tensor(spec, psi.dtype)
            scale = tf.sqrt(w / (n * spec.sigma**2))
            r = scale * tf.sqrt(m) * self._signed_root(spec, e, psi)
            pieces.append(tf.reshape(r, [-1]))
        return tf.concat(pieces, axis=0)

    @staticmethod
    def _signed_root(
        spec: ObservationSpec, e: TensorType, psi: TensorType
    ) -> TensorType:
        """
        sign(e)·sqrt(2 psi(e)) с конечной производной в e = 0.

        Для l2/relative/msle (psi = e²/2) это ровно e. Для робастных видов
        psi ≈ e²/2 около нуля, поэтому в точках psi = 0 подставляется e:
        значение то же (0), производная равна пределу (1). Прямая запись
        sign(e)·sqrt(2 psi) давала 0·∞ = NaN в якобиане при точном
        совпадении предсказания и цели.
        """
        if spec.kind in ("l2", "relative", "msle"):
            return e
        positive = psi > 0.0
        psi_safe = tf.where(positive, psi, tf.ones_like(psi))
        return tf.where(positive, tf.sign(e) * tf.sqrt(2.0 * psi_safe), e)


# ---------------------------------------------------------------------------
# Единицы правдоподобия для статистических модулей
# ---------------------------------------------------------------------------


def _residual_objectives(problem) -> Optional[List[Any]]:
    """
    Objective'ы, чьи невязки составляют problem.residual, в порядке
    конкатенации; None, если состав невязок неизвестен (duck-задачи,
    multiple shooting с невязками разрывов).
    """
    from neuraltide.optimization.shooting import MultipleShootingProblem

    if isinstance(problem, MultipleShootingProblem):
        return None
    inner = getattr(problem, "_problem", None)
    if inner is not None and hasattr(problem, "_compose"):  # FrozenProblem
        return _residual_objectives(inner)
    experiments = getattr(problem, "experiments", None)
    if isinstance(experiments, (list, tuple)):  # MultiExperimentProblem
        return [exp.objective for exp in experiments]
    if hasattr(problem, "objective"):
        return [problem.objective]
    return None


def likelihood_row_scale(problem) -> Optional[np.ndarray]:
    """
    Множители строк problem.residual для перевода в единицы полного NLL.

    Для ResidualLoss с normalization='mean' невязки нормированы так, что
    0.5·||r||² = NLL/N; статистике (FIM, Лаплас) нужен NLL. Возвращает
    вектор s (sqrt(N_k) на элементы наблюдения k), такой что
    r_nll = s·r и J_nll = s[:, None]·J. None — если задача не построена на
    ResidualLoss (невязки duck-задач считаются уже заданными в единицах NLL).
    """
    objectives = _residual_objectives(problem)
    if not objectives:
        return None
    pieces = []
    for obj in objectives:
        fn = getattr(obj, "likelihood_row_scale", None)
        if fn is None:
            return None
        pieces.append(fn())
    return np.concatenate(pieces)


@contextlib.contextmanager
def likelihood_normalization(problem) -> Iterator[None]:
    """
    Временно переводит ResidualLoss задачи в normalization='sum'.

    Внутри контекста loss и невязки задачи — в единицах полного NLL
    (гауссов шум с известной sigma), поэтому пороги χ² профилей и
    гессиан потерь имеют статистический смысл. Задачи без ResidualLoss
    не меняются. Исходные значения восстанавливаются при выходе.
    """
    objectives = _residual_objectives(problem) or []
    saved = []
    for obj in objectives:
        if hasattr(obj, "likelihood_row_scale") and hasattr(obj, "normalization"):
            saved.append((obj, obj.normalization))
            obj.normalization = "sum"
    try:
        yield
    finally:
        for obj, value in saved:
            obj.normalization = value
