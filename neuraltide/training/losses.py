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


class StabilityPenalty(BaseLoss):
    """
    Штраф за численную нестабильность.
    Возвращает predictions.stability_loss.
    """

    def __call__(self, predictions: NetworkOutput, model: NetworkRNN) -> TensorType:
        return predictions.stability_loss


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
