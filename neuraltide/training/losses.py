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
                r_a = r_a[:, self.transient_steps:, :]
                r_b = r_b[:, self.transient_steps:, :]

            r_a_flat = tf.reshape(r_a, [-1])
            r_b_flat = tf.reshape(r_b, [-1])

            mean_a = tf.reduce_mean(r_a_flat)
            mean_b = tf.reduce_mean(r_b_flat)
            r_a_c = r_a_flat - mean_a
            r_b_c = r_b_flat - mean_b

            cov = tf.reduce_mean(r_a_c * r_b_c)
            std_a = tf.sqrt(tf.reduce_mean(r_a_c ** 2) + self.eps)
            std_b = tf.sqrt(tf.reduce_mean(r_b_c ** 2) + self.eps)
            corr = cov / (std_a * std_b + self.eps)

            loss += self.correlation_weight * (corr - self.target_correlation) ** 2

        # --- 2. Activity maintenance term ---
        for pop_name in self.all_populations:
            r = rates[pop_name]  # [1, T, n_units]
            if self.transient_steps > 0:
                r = r[:, self.transient_steps:, :]
            mean_rate = tf.reduce_mean(r)
            loss += self.activity_weight * tf.nn.relu(
                self.activity_target - mean_rate
            )

        return loss

    def per_step_loss(
        self,
        firing_rates: Dict[str, TensorType],
        target: Dict[str, TensorType],
    ) -> TensorType:
        dtype = neuraltide.config.get_dtype()
        return tf.constant(0.0, dtype=dtype)


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
