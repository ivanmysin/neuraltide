import tensorflow as tf
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional

import neuraltide
import neuraltide.config
from neuraltide.core.network import NetworkOutput, NetworkRNN
from neuraltide.core.types import TensorType, cast_if_needed
from neuraltide.training.readouts import IdentityReadout, BaseReadout

# Нижняя граница знаменателя маски: полностью нулевая маска даёт loss = 0,
# а не NaN.
_MASK_DENOM_EPS = 1e-8


def _masked_mean(err: TensorType, mask_val: TensorType) -> TensorType:
    """
    Masked mean: sum(mask * err) / max(sum(mask), eps).

    Маска масштаб не меняет: умножение маски на константу сокращается
    числителем и знаменателем.
    """
    m = cast_if_needed(mask_val, err.dtype)
    denom = tf.maximum(tf.reduce_sum(m), tf.constant(_MASK_DENOM_EPS, dtype=err.dtype))
    return tf.reduce_sum(m * err) / denom


class BaseLoss(ABC):
    """Базовый класс для функции потерь."""

    @abstractmethod
    def __call__(self, predictions: NetworkOutput, model: NetworkRNN) -> TensorType:
        raise NotImplementedError

    def per_step_loss(
        self,
        firing_rates: Dict[str, TensorType],
        target: Dict[str, TensorType],
        mask: Optional[Dict[str, TensorType]] = None,
    ) -> TensorType:
        """
        Аддитивный вклад одного временного шага (для adjoint-метода).

        Конвенция: полная loss ≡ sum_t per_step_loss(rates_t, target_t, ...).
        Потери, не поддерживающие разложение по времени, возвращают 0.

        Args:
            firing_rates: частоты разрядов на шаге t: {name: [batch, n_units]}.
            target: целевые значения на шаге t: {name: [batch, n_units]}.
            mask: опциональные срезы маски на шаге t; если None, потерьный
                объект использует собственную маску (при наличии).

        Returns:
            Скаляр — вклад шага t.
        """
        return self._default_per_step_loss(firing_rates, target)

    def _default_per_step_loss(
        self,
        firing_rates: Dict[str, TensorType],
        target: Dict[str, TensorType],
    ) -> TensorType:
        """Default implementation using full NetworkOutput."""
        dtype = neuraltide.config.get_dtype()
        return tf.constant(0.0, dtype=dtype)


class _TrajectoryLoss(BaseLoss):
    """
    Общая логика MSE/MSLE: pointwise ошибка траектории + опциональная маска.

    Полная loss — среднее по всем элементам (masked mean при заданной маске).
    per_step_loss — аддитивный вклад шага: без маски mean(err_t)/T, с маской
    sum(m_t * err_t) / max(sum(m_full), eps); T берётся из сохранённого
    полного target.
    """

    def __init__(
        self,
        target: Dict[str, TensorType],
        readout: Optional[BaseReadout] = None,
        mask: Optional[Dict[str, TensorType]] = None,
    ):
        self.target = target
        self.readout = readout if readout is not None else IdentityReadout()
        self.mask = mask

    def _error(self, pred_val: TensorType, target_val: TensorType) -> TensorType:
        """Квадрат pointwise ошибки после readout."""
        pred_readout = self.readout(pred_val)
        target_readout = self.readout(target_val)
        return (pred_readout - target_readout) ** 2

    def _target_T(self, pop_name: str) -> TensorType:
        """Число временных шагов из сохранённого target [batch, T, n]."""
        return tf.shape(self.target[pop_name])[1]

    def _full_mask(self, pop_name: str) -> Optional[TensorType]:
        if self.mask is None or pop_name not in self.mask:
            return None
        return self.mask[pop_name]

    def _per_step_mask_view(
        self, pop_name: str, err: TensorType
    ) -> Optional[TensorType]:
        """
        Представление полной маски для вызова per_step_loss без среза.

        Если маска содержит временную ось (ранг на 1 больше ранга ошибки),
        используется среднее по времени — точное совпадение значений полной
        loss сохраняется для масок, постоянных во времени; adjoint передаёт
        точные срезы через аргумент mask=.
        """
        m_full = self._full_mask(pop_name)
        if m_full is None:
            return None
        if m_full.shape.rank == err.shape.rank + 1:
            return tf.reduce_mean(m_full, axis=1)
        return m_full

    def __call__(self, predictions: NetworkOutput, model: NetworkRNN) -> TensorType:
        total_loss = tf.constant(0.0, dtype=neuraltide.config.get_dtype())

        for pop_name, target_val in self.target.items():
            pred_val = predictions.firing_rates[pop_name]
            err = self._error(pred_val, target_val)

            m_full = self._full_mask(pop_name)
            if m_full is not None:
                total_loss += _masked_mean(err, m_full)
            else:
                total_loss += tf.reduce_mean(err)

        return total_loss

    def per_step_loss(
        self,
        firing_rates: Dict[str, TensorType],
        target: Dict[str, TensorType],
        mask: Optional[Dict[str, TensorType]] = None,
    ) -> TensorType:
        """Аддитивный вклад шага t; сумма по t даёт полную loss."""
        total_loss = tf.constant(0.0, dtype=neuraltide.config.get_dtype())

        for pop_name, target_t in target.items():
            pred_t = firing_rates.get(pop_name)
            if pred_t is None:
                continue

            err = self._error(pred_t, target_t)

            mask_t: Optional[TensorType] = None
            if mask is not None and pop_name in mask:
                mask_t = mask[pop_name]
            else:
                mask_t = self._per_step_mask_view(pop_name, err)

            if mask_t is None:
                T = tf.cast(self._target_T(pop_name), err.dtype)
                total_loss += tf.reduce_mean(err) / T
            else:
                m = cast_if_needed(mask_t, err.dtype)
                m_full = self._full_mask(pop_name)
                if m_full is not None:
                    mass = tf.reduce_sum(cast_if_needed(m_full, err.dtype))
                else:
                    mass = tf.reduce_sum(m)
                denom = tf.maximum(mass, tf.constant(_MASK_DENOM_EPS, dtype=err.dtype))
                total_loss += tf.reduce_sum(m * err) / denom

        return total_loss


class MSELoss(_TrajectoryLoss):
    """
    MSE между предсказанной и целевой активностью.

    Args:
        target: dict[str, tf.Tensor] — целевые траектории [batch, T, n].
        readout: BaseReadout — предобработка перед сравнением.
        observable_key: str — какую наблюдаемую использовать.
        mask: dict[str, tf.Tensor] или None — маска по времени/каналам.
            Семантика — masked mean: вклад популяции
            sum(mask * err^2) / max(sum(mask), eps); масштаб маски
            не влияет на значение loss.
    """

    def __init__(
        self,
        target: Dict[str, TensorType],
        readout: Optional[BaseReadout] = None,
        observable_key: str = "firing_rate",
        mask: Optional[Dict[str, TensorType]] = None,
    ):
        super().__init__(target=target, readout=readout, mask=mask)
        self.observable_key = observable_key


class MSLELoss(_TrajectoryLoss):
    """
    Mean Squared Logarithmic Error: mean((log(eps+pred) - log(eps+target))^2).

    Better for comparing rates across different scales. Поддерживает
    опциональную маску с семантикой masked mean (как у MSELoss).
    """

    def __init__(
        self,
        target: Dict[str, TensorType],
        readout: Optional[BaseReadout] = None,
        eps: float = 1.0,
        mask: Optional[Dict[str, TensorType]] = None,
    ):
        super().__init__(target=target, readout=readout, mask=mask)
        self.eps = eps

    def _error(self, pred_val: TensorType, target_val: TensorType) -> TensorType:
        pred_readout = self.readout(pred_val)
        target_readout = self.readout(target_val)
        log_pred = tf.math.log(pred_readout + self.eps)
        log_target = tf.math.log(target_readout + self.eps)
        return (log_pred - log_target) ** 2


class StabilityPenalty(BaseLoss):
    """
    Штраф за численную нестабильность.
    Возвращает predictions.stability_loss.
    """

    def __call__(self, predictions: NetworkOutput, model: NetworkRNN) -> TensorType:
        return predictions.stability_loss

    def per_step_loss(
        self,
        firing_rates: Dict[str, TensorType],
        target: Dict[str, TensorType],
        mask: Optional[Dict[str, TensorType]] = None,
    ) -> TensorType:
        """StabilityPenalty doesn't have per-step gradient (handled separately)."""
        return tf.constant(0.0, dtype=neuraltide.config.get_dtype())


class L2RegularizationLoss(BaseLoss):
    """L2-регуляризация: Σ ||w||²."""

    def __call__(self, predictions: NetworkOutput, model: NetworkRNN) -> TensorType:
        loss = tf.constant(0.0, dtype=neuraltide.config.get_dtype())
        for var in model.trainable_variables:
            loss += tf.reduce_sum(var**2)
        return loss


class ParameterBoundLoss(BaseLoss):
    """
    Мягкие границы на параметры.
    Штраф нарастает при выходе за [min, max].
    """

    def __init__(self, bounds: Dict[str, Tuple[float, float]]):
        self.bounds = bounds

    def __call__(self, predictions: NetworkOutput, model: NetworkRNN) -> TensorType:
        loss = tf.constant(0.0, dtype=neuraltide.config.get_dtype())

        for var in model.trainable_variables:
            var_name = var.name
            if var_name in self.bounds:
                lo, hi = self.bounds[var_name]
                violation_low = tf.reduce_sum(tf.nn.relu(lo - var))
                violation_high = tf.reduce_sum(tf.nn.relu(var - hi))
                loss += violation_low + violation_high

        return loss


class AntiPhaseLoss(BaseLoss):
    """
    Loss for mutually inhibitory populations: anti-correlation + activity maintenance.

    Designed for networks where populations should oscillate in anti-phase
    (e.g. two mutually inhibiting populations). The loss has two terms:

    1. Correlation term: penalises deviation from target_correlation.
       With target_correlation = -1.0, this pushes the pair toward perfect
       anti-phase.  Mutual inhibition provides the physical substrate.

    2. Activity term: penalises mean firing rates below activity_target.
       Prevents the trivial solution where both populations go silent.

    Args:
        pop_pairs: list of (name_a, name_b) tuples.
        target_correlation: desired Pearson correlation (−1.0 = full anti-phase).
        correlation_weight: weight of the correlation term.
        activity_target: minimum acceptable mean firing rate (Hz).
        activity_weight: weight of the activity maintenance term.
        transient_steps: number of initial time steps to exclude.
        eps: small constant for numerical stability.
    """

    def __init__(
        self,
        pop_pairs: List[Tuple[str, str]],
        target_correlation: float = -1.0,
        correlation_weight: float = 1.0,
        activity_target: float = 20.0,
        activity_weight: float = 0.5,
        transient_steps: int = 0,
        eps: float = 1e-8,
    ):
        self.pop_pairs = pop_pairs
        self.target_correlation = target_correlation
        self.correlation_weight = correlation_weight
        self.activity_target = activity_target
        self.activity_weight = activity_weight
        self.transient_steps = transient_steps
        self.eps = eps

        all_pops = set()
        for a, b in pop_pairs:
            all_pops.add(a)
            all_pops.add(b)
        self.all_populations = sorted(all_pops)

    def __call__(self, predictions: NetworkOutput, model: NetworkRNN) -> TensorType:
        dtype = neuraltide.config.get_dtype()
        loss = tf.constant(0.0, dtype=dtype)

        rates = predictions.firing_rates  # dict[str, Tensor[B, T, N]]

        # --- 1. Anti-correlation term ---
        for pop_a, pop_b in self.pop_pairs:
            r_a = rates[pop_a]  # [1, T, n_units]
            r_b = rates[pop_b]

            if self.transient_steps > 0:
                r_a = r_a[:, self.transient_steps :, :]
                r_b = r_b[:, self.transient_steps :, :]

            r_a_flat = tf.reshape(r_a, [-1])
            r_b_flat = tf.reshape(r_b, [-1])

            mean_a = tf.reduce_mean(r_a_flat)
            mean_b = tf.reduce_mean(r_b_flat)
            r_a_c = r_a_flat - mean_a
            r_b_c = r_b_flat - mean_b

            cov = tf.reduce_mean(r_a_c * r_b_c)
            std_a = tf.sqrt(tf.reduce_mean(r_a_c**2) + self.eps)
            std_b = tf.sqrt(tf.reduce_mean(r_b_c**2) + self.eps)
            corr = cov / (std_a * std_b + self.eps)

            loss += self.correlation_weight * (corr - self.target_correlation) ** 2

        # --- 2. Activity maintenance term ---
        for pop_name in self.all_populations:
            r = rates[pop_name]  # [1, T, n_units]
            if self.transient_steps > 0:
                r = r[:, self.transient_steps :, :]
            mean_rate = tf.reduce_mean(r)
            loss += self.activity_weight * tf.nn.relu(self.activity_target - mean_rate)

        return loss

    def per_step_loss(
        self,
        firing_rates: Dict[str, TensorType],
        target: Dict[str, TensorType],
        mask: Optional[Dict[str, TensorType]] = None,
    ) -> TensorType:
        dtype = neuraltide.config.get_dtype()
        return tf.constant(0.0, dtype=dtype)


class CompositeLoss:
    """
    Составная функция потерь: L = Σ weight_i * L_i.

    per_step_loss распределяет аддитивный вклад шага по термам, так что
    полная loss остаётся равной сумме per-step вкладов (конвенция BaseLoss).
    """

    def __init__(self, terms: List[Tuple[float, BaseLoss]]):
        self.terms = terms

    def components(
        self, predictions: NetworkOutput, model: NetworkRNN
    ) -> Dict[str, TensorType]:
        """
        Взвешенные компоненты потери по отдельности.

        Returns:
            dict с детерминированными ключами вида "term_i:ClassName",
            значения — weight_i * L_i(predictions, model).
        """
        out: Dict[str, TensorType] = {}
        for i, (weight, loss_obj) in enumerate(self.terms):
            key = f"term_{i}:{type(loss_obj).__name__}"
            out[key] = weight * loss_obj(predictions, model)
        return out

    def __call__(self, predictions: NetworkOutput, model: NetworkRNN) -> TensorType:
        total = tf.constant(0.0, dtype=neuraltide.config.get_dtype())
        for value in self.components(predictions, model).values():
            total += value
        return total

    def per_step_loss(
        self,
        firing_rates: Dict[str, TensorType],
        target: Dict[str, TensorType],
        mask: Optional[Dict[str, TensorType]] = None,
    ) -> TensorType:
        total = tf.constant(0.0, dtype=neuraltide.config.get_dtype())
        for weight, loss_obj in self.terms:
            total += weight * loss_obj.per_step_loss(firing_rates, target, mask=mask)
        return total
