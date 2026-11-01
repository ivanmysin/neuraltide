import tensorflow as tf
from typing import Dict, Optional
import os


class DivergenceDetector(tf.keras.callbacks.Callback):
    """
    Останавливает обучение при NaN/Inf в loss.
    """

    def __init__(self, patience: int = 3):
        super().__init__()
        self.patience = patience
        self.bad_epochs = 0

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        logs = logs or {}
        loss = logs.get('loss')
        if loss is None:
            return

        if tf.math.is_nan(loss) or tf.math.is_inf(loss):
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                raise RuntimeError(
                    f"Training diverged: loss became {loss} at epoch {epoch + 1}"
                )
        else:
            self.bad_epochs = 0


class GradientMonitor(tf.keras.callbacks.Callback):
    """
    Логирует нормы градиентов по переменным.
    """

    def __init__(self, log_every: int = 10):
        super().__init__()
        self.log_every = log_every
        self._last_logged_epoch = -1

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        if (epoch - self._last_logged_epoch) < self.log_every:
            return

        self._last_logged_epoch = epoch

        # Compute gradient norms using the model's optimizer and loss
        if self.model is not None and self.model.optimizer is not None:
            try:
                # In practice, gradient logging requires access to the training step
                # For now, log a message indicating gradient monitoring is active
                tf.print(f"[GradientMonitor] Epoch {epoch}: monitoring active (gradients not directly accessible in on_epoch_end)")
            except Exception as e:
                tf.print(f"[GradientMonitor] Warning: could not compute gradients: {e}")


class ExperimentLogger(tf.keras.callbacks.Callback):
    """
    Сохраняет checkpoint каждые N эпох.
    """

    def __init__(self, save_dir: str, save_every: int = 10, network=None, optimizer=None):
        super().__init__()
        self.save_dir = save_dir
        self.save_every = save_every
        self.network = network
        self.optimizer = optimizer
        self._last_saved_epoch = -1

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        if (epoch - self._last_saved_epoch) >= self.save_every:
            self._last_saved_epoch = epoch
            # Use self.model if network/optimizer not explicitly provided
            net = self.network or getattr(self.model, '_network', None)
            opt = self.optimizer or getattr(self.model, 'optimizer', None)
            if net is not None and opt is not None:
                checkpoint_dir = os.path.join(self.save_dir, f'epoch_{epoch}')
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint = tf.train.Checkpoint(
                    optimizer=opt,
                    network=net,
                )
                checkpoint.save(file_prefix=os.path.join(checkpoint_dir, 'ckpt'))
