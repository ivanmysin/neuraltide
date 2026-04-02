import tensorflow as tf
from typing import Dict, List, Optional
import os


class DivergenceDetector:
    """
    Останавливает обучение при NaN/Inf в loss.
    """

    def __init__(self, patience: int = 3):
        self.patience = patience
        self.bad_epochs = 0

    def on_epoch_end(self, epoch: int, logs: Dict) -> None:
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


class GradientMonitor:
    """
    Логирует нормы градиентов по переменным.
    """

    def __init__(self, log_every: int = 10):
        self.log_every = log_every
        self._last_logged_epoch = -1

    def on_epoch_end(self, epoch: int, logs: Dict) -> None:
        if (epoch - self._last_logged_epoch) < self.log_every:
            return

        self._last_logged_epoch = epoch


class ExperimentLogger:
    """
    Сохраняет checkpoint каждые N эпох.
    """

    def __init__(self, save_dir: str, save_every: int = 10, network=None, optimizer=None):
        self.save_dir = save_dir
        self.save_every = save_every
        self.network = network
        self.optimizer = optimizer
        self._last_saved_epoch = -1

    def on_epoch_end(self, epoch: int, logs: Dict) -> None:
        if (epoch - self._last_saved_epoch) >= self.save_every:
            self._last_saved_epoch = epoch
            if self.network is not None and self.optimizer is not None:
                checkpoint_dir = os.path.join(self.save_dir, f'epoch_{epoch}')
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint = tf.train.Checkpoint(
                    optimizer=self.optimizer,
                    network=self.network,
                )
                checkpoint.save(file_prefix=os.path.join(checkpoint_dir, 'ckpt'))
