import tensorflow as tf
from typing import Dict, List, Optional
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
        loss = logs.get("loss")
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

    Градиенты читаются из logs["grads"], которые Trainer пробрасывает
    в callbacks на каждой эпохе (значения — до клиппинга). История
    сохраняется в self.history: список dict с ключами epoch, global_norm,
    norms (и variables — имена переменных, если они известны).

    Вне Trainer (например, в голом Keras fit) градиентов в logs нет:
    колбэк печатает пояснение и не падает.
    """

    def __init__(
        self,
        log_every: int = 10,
        variables: Optional[List[tf.Variable]] = None,
    ):
        super().__init__()
        self.log_every = log_every
        self.variables = variables
        # -log_every: эпоха 0 логируется сразу (baseline), далее каждые
        # log_every эпох.
        self._last_logged_epoch = -log_every
        self.history: List[Dict[str, object]] = []

    def _resolve_variables(
        self, grads: List[Optional[tf.Tensor]]
    ) -> Optional[List[tf.Variable]]:
        if self.variables is not None:
            return self.variables
        if self.model is not None:
            return list(self.model.trainable_variables)
        return None

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        if (epoch - self._last_logged_epoch) < self.log_every:
            return

        self._last_logged_epoch = epoch

        logs = logs or {}
        grads = logs.get("grads")
        if grads is None:
            tf.print(
                f"[GradientMonitor] epoch {epoch}: no 'grads' in logs; "
                "gradient norms are available only via Trainer.fit "
                "(BPTT or adjoint), which passes grads to callbacks"
            )
            return

        variables = self._resolve_variables(grads)
        if variables is not None:
            pairs = [(g, v) for g, v in zip(grads, variables) if g is not None]
        else:
            pairs = [(g, None) for g in grads if g is not None]

        flat = [tf.reshape(g, [-1]) for g, _ in pairs]
        global_norm = float(tf.linalg.global_norm(flat)) if flat else 0.0

        entry: Dict[str, object] = {
            "epoch": epoch,
            "global_norm": global_norm,
            "norms": [float(tf.norm(g)) for g, _ in pairs],
        }
        if variables is not None:
            entry["variables"] = [v.name if v is not None else "?" for _, v in pairs]
        self.history.append(entry)

        tf.print(f"[GradientMonitor] epoch {epoch}: global_grad_norm=", global_norm)


class ExperimentLogger(tf.keras.callbacks.Callback):
    """
    Сохраняет checkpoint каждые N эпох.
    """

    def __init__(
        self, save_dir: str, save_every: int = 10, network=None, optimizer=None
    ):
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
            net = self.network or getattr(self.model, "_network", None)
            opt = self.optimizer or getattr(self.model, "optimizer", None)
            if net is not None and opt is not None:
                checkpoint_dir = os.path.join(self.save_dir, f"epoch_{epoch}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint = tf.train.Checkpoint(
                    optimizer=opt,
                    network=net,
                )
                checkpoint.save(file_prefix=os.path.join(checkpoint_dir, "ckpt"))
