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

где psi(e) — штрафная функция вида, а N — нормировка:
``max(sum(mask), eps)`` при заданной маске и число элементов траектории
без неё. Маска масштаб не меняет: умножение маски на константу сокращается
числителем и знаменателем.

Residual-вектор согласован с этой нормировкой поэлементно:

    r_i = sqrt(weight * mask_i / (N * sigma**2)) * sign(e_i) * sqrt(2 psi(e_i)),

поэтому ``0.5 * ||r||^2`` воспроизводит loss точно (для l2/relative/msle,
где psi = 0.5 e^2, множитель ``sign(e) * sqrt(2 psi)`` тождественно равен e).

Формы:
    target: [batch, T, n_units] — целевая траектория наблюдаемой величины;
    mask: broadcastable к target (например [batch, T, 1] или [1, T, n]);
    predictions.firing_rates[population]: [batch, T, n_units].
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import tensorflow as tf

import neuraltide.config
from neuraltide.core.types import TensorType, cast_if_needed
from neuraltide.training.losses import BaseLoss, _MASK_DENOM_EPS
from neuraltide.training.readouts import BaseReadout, IdentityReadout

__all__ = ["ObservationSpec", "ResidualLoss"]

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

    def __init__(self, observations: Sequence[ObservationSpec]):
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
    def _norm(spec: ObservationSpec, psi: TensorType) -> TensorType:
        """
        Нормировка N полной траектории: max(sum(mask), eps) либо число
        элементов target'а (без маски).
        """
        dtype = psi.dtype
        eps_const = tf.constant(_MASK_DENOM_EPS, dtype=dtype)
        if spec.mask is None:
            n = tf.cast(tf.size(spec.target), dtype)
            return tf.maximum(n, eps_const)
        m = cast_if_needed(spec.mask, dtype)
        return tf.maximum(tf.reduce_sum(m), eps_const)

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
            n = self._norm(spec, psi)
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
        dtype = neuraltide.config.get_dtype()
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
            r = (
                scale
                * tf.sqrt(m)
                * tf.sign(e)
                * tf.sqrt(tf.maximum(2.0 * psi, tf.constant(0.0, dtype=dtype)))
            )
            pieces.append(tf.reshape(r, [-1]))
        return tf.concat(pieces, axis=0)
