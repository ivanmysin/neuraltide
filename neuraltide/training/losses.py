import tensorflow as tf
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional

import neuraltide
import neuraltide.config
from neuraltide.core.network import NetworkOutput, NetworkRNN
from neuraltide.core.types import TensorType
from neuraltide.training.readouts import IdentityReadout, BaseReadout


class BaseLoss(ABC):
    """Базовый класс для функции потерь."""

    @abstractmethod
    def __call__(self, predictions: NetworkOutput, model: NetworkRNN) -> TensorType:
        raise NotImplementedError

    def per_step_loss(
        self,
        firing_rates: Dict[str, TensorType],
        target: Dict[str, TensorType],
    ) -> TensorType:
        """
        Compute loss for a single time step (for adjoint method).
        
        Args:
            firing_rates: Dict of firing rates at time t
            target: Dict of target values at time t
        
        Returns:
            Scalar loss value
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


class MSELoss(BaseLoss):
    """
    MSE между предсказанной и целевой активностью.

    Args:
        target: dict[str, tf.Tensor] — целевые траектории.
        readout: BaseReadout — предобработка перед сравнением.
        observable_key: str — какую наблюдаемую использовать.
        mask: dict[str, tf.Tensor] или None — маска по времени.
    """

    def __init__(
        self,
        target: Dict[str, TensorType],
        readout: Optional[BaseReadout] = None,
        observable_key: str = 'firing_rate',
        mask: Optional[Dict[str, TensorType]] = None,
    ):
        self.target = target
        self.readout = readout if readout is not None else IdentityReadout()
        self.observable_key = observable_key
        self.mask = mask

    def __call__(self, predictions: NetworkOutput, model: NetworkRNN) -> TensorType:
        total_loss = tf.constant(0.0, dtype=neuraltide.config.get_dtype())

        for pop_name, target_val in self.target.items():
            pred_val = predictions.firing_rates[pop_name]

            pred_readout = self.readout(pred_val)
            target_readout = self.readout(target_val)

            loss = tf.reduce_mean((pred_readout - target_readout) ** 2)

            if self.mask is not None and pop_name in self.mask:
                mask_val = self.mask[pop_name]
                loss = loss * mask_val

            total_loss += loss

        return total_loss

    def per_step_loss(
        self,
        firing_rates: Dict[str, TensorType],
        target: Dict[str, TensorType],
    ) -> TensorType:
        """Compute MSE loss for a single time step."""
        total_loss = tf.constant(0.0, dtype=neuraltide.config.get_dtype())

        for pop_name, target_val in target.items():
            pred_val = firing_rates.get(pop_name)
            if pred_val is None:
                continue

            pred_readout = self.readout(pred_val)
            target_readout = self.readout(target_val)

            loss = tf.reduce_mean((pred_readout - target_readout) ** 2)

            if self.mask is not None and pop_name in self.mask:
                mask_val = self.mask[pop_name]
                loss = loss * mask_val

            total_loss += loss

        return total_loss


class MSLELoss(BaseLoss):
    """
    Mean Squared Logarithmic Error: mean((log(1+pred) - log(1+target))^2).

    Better for comparing rates across different scales.
    """

    def __init__(
        self,
        target: Dict[str, TensorType],
        readout: Optional[BaseReadout] = None,
        eps: float = 1.0,
    ):
        self.target = target
        self.readout = readout if readout is not None else IdentityReadout()
        self.eps = eps

    def __call__(self, predictions: NetworkOutput, model: NetworkRNN) -> TensorType:
        total_loss = tf.constant(0.0, dtype=neuraltide.config.get_dtype())

        for pop_name, target_val in self.target.items():
            pred_val = predictions.firing_rates[pop_name]

            pred_readout = self.readout(pred_val)
            target_readout = self.readout(target_val)

            pred_clamped = pred_readout + self.eps
            log_pred = tf.math.log(pred_clamped)
            log_target = tf.math.log(target_readout + self.eps)

            loss = tf.reduce_mean((log_pred - log_target) ** 2)

            total_loss += loss

        return total_loss

    def per_step_loss(
        self,
        firing_rates: Dict[str, TensorType],
        target: Dict[str, TensorType],
    ) -> TensorType:
        """Compute MSLE loss for a single time step."""
        total_loss = tf.constant(0.0, dtype=neuraltide.config.get_dtype())

        for pop_name, target_val in target.items():
            pred_val = firing_rates.get(pop_name)
            if pred_val is None:
                continue

            pred_readout = self.readout(pred_val)
            target_readout = self.readout(target_val)

            pred_clamped = pred_readout + self.eps
            log_pred = tf.math.log(pred_clamped)
            log_target = tf.math.log(target_readout + self.eps)

            loss = tf.reduce_mean((log_pred - log_target) ** 2)

            if self.mask is not None and pop_name in self.mask:
                mask_val = self.mask[pop_name]
                loss = loss * mask_val

            total_loss += loss

        return total_loss


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
    ) -> TensorType:
        """StabilityPenalty doesn't have per-step gradient (handled separately)."""
        return tf.constant(0.0, dtype=neuraltide.config.get_dtype())


class L2RegularizationLoss(BaseLoss):
    """L2-регуляризация: Σ ||w||²."""

    def __call__(self, predictions: NetworkOutput, model: NetworkRNN) -> TensorType:
        loss = tf.constant(0.0, dtype=neuraltide.config.get_dtype())
        for var in model.trainable_variables:
            loss += tf.reduce_sum(var ** 2)
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


class CompositeLoss:
    """
    Составная функция потерь: L = Σ weight_i * L_i.
    """

    def __init__(self, terms: List[Tuple[float, BaseLoss]]):
        self.terms = terms

    def __call__(self, predictions: NetworkOutput, model: NetworkRNN) -> TensorType:
        total = tf.constant(0.0, dtype=neuraltide.config.get_dtype())
        for weight, loss_obj in self.terms:
            total += weight * loss_obj(predictions, model)
        return total
