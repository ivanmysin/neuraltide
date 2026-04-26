import csv
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Callable, Literal, Tuple, Union

import tensorflow as tf

import neuraltide
import neuraltide.config
from neuraltide.core.network import NetworkRNN, NetworkOutput
from neuraltide.training.losses import CompositeLoss
from neuraltide.core.types import TensorType, StateList

try:
    from neuraltide.training.adjoint import compute_gradients as adjoint_compute_gradients
except ImportError:
    adjoint_compute_gradients = None


@dataclass
class TrainingHistory:
    """История обучения."""
    loss_history: List[float]
    epochs: int

    def to_dict(self) -> dict:
        return {
            'loss_history': self.loss_history,
            'epochs': self.epochs,
        }


class Trainer:
    """
    Высокоуровневый API обучения.

    Args:
        network: сеть.
        loss_fn: функция потерь.
        optimizer: оптимизатор.
        grad_method: метод вычисления градиентов: "bptt" (по умолчанию) или "adjoint".
        grad_clip_norm: максимальная норма градиента (0.0 — без клиппинга).
        run_eagerly: отключить tf.function для отладки.
    """

    def __init__(
        self,
        network: NetworkRNN,
        loss_fn: CompositeLoss,
        optimizer: tf.keras.optimizers.Optimizer,
        grad_method: Literal["bptt", "adjoint"] = "bptt",
        grad_clip_norm: float = 1.0,
        run_eagerly: bool = False,
    ):
        self.network = network
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.grad_method = grad_method
        self.grad_clip_norm = grad_clip_norm
        self.run_eagerly = run_eagerly

        if grad_method == "adjoint" and adjoint_compute_gradients is None:
            raise ImportError("adjoint method not available")

        if not run_eagerly:
            self._train_step = tf.function(self._train_step)

    def _train_step(self, t_sequence: TensorType) -> Dict[str, float]:
        with tf.GradientTape() as tape:
            output = self.network(t_sequence, training=True)
            loss = self.loss_fn(output, self.network)

        if self.grad_method == "adjoint":
            grads = adjoint_compute_gradients(self.network, t_sequence, self.loss_fn)
        else:
            grads = tape.gradient(loss, self.network.trainable_variables)

        grads_and_vars = [(g, v) for g, v in zip(grads, self.network.trainable_variables) if g is not None]
        
        if not grads_and_vars:
            return {'loss': loss}

        if self.grad_clip_norm > 0:
            grads_only = [g for g, v in grads_and_vars]
            clipped_grads, _ = tf.clip_by_global_norm(grads_only, self.grad_clip_norm)
            grads_and_vars = [(g, v) for g, (_, v) in zip(clipped_grads, grads_and_vars)]

        self.optimizer.apply_gradients(grads_and_vars)

        return {'loss': loss}

    def train_step(self, t_sequence: TensorType) -> Dict[str, float]:
        if self.grad_method == "adjoint":
            if self.run_eagerly:
                return self._train_step_adjoint(t_sequence)
            return self._train_step_adjoint(t_sequence)
        if self.run_eagerly:
            return self._train_step_eager(t_sequence)
        return self._train_step(t_sequence)

    def _train_step_adjoint(self, t_sequence: TensorType) -> Dict[str, float]:
        with tf.GradientTape() as tape:
            output = self.network(t_sequence, training=True)
            loss = self.loss_fn(output, self.network)

        # Pass stability_loss if available
        stability_loss = getattr(output, 'stability_loss', None)
        grads = self._adjoint_computer.compute_gradients(loss, t_sequence, stability_loss)

        trainable_vars = self.network.trainable_variables

        grads_and_vars = []
        for v in trainable_vars:
            g = grads.get(v.name)
            if g is not None:
                grads_and_vars.append((g, v))

        if not grads_and_vars:
            return {'loss': loss}

        if self.grad_clip_norm > 0:
            grads_only = [g for g, v in grads_and_vars]
            clipped_grads, _ = tf.clip_by_global_norm(grads_only, self.grad_clip_norm)
            grads_and_vars = [(g, v) for g, (_, v) in zip(clipped_grads, grads_and_vars)]

        self.optimizer.apply_gradients(grads_and_vars)

        return {'loss': loss}

    def _train_step_eager(self, t_sequence: TensorType) -> Dict[str, float]:
        with tf.GradientTape() as tape:
            output = self.network(t_sequence, training=True)
            loss = self.loss_fn(output, self.network)

        if self.grad_method == "adjoint":
            grads = adjoint_compute_gradients(self.network, t_sequence, self.loss_fn)
        else:
            grads = tape.gradient(loss, self.network.trainable_variables)

        grads_and_vars = [(g, v) for g, v in zip(grads, self.network.trainable_variables) if g is not None]
        
        if not grads_and_vars:
            return {'loss': loss}

        if self.grad_clip_norm > 0:
            grads_only = [g for g, v in grads_and_vars]
            clipped_grads, _ = tf.clip_by_global_norm(grads_only, self.grad_clip_norm)
            grads_and_vars = [(g, v) for g, (_, v) in zip(clipped_grads, grads_and_vars)]

        self.optimizer.apply_gradients(grads_and_vars)

        return {'loss': loss}

    def fit(
        self,
        t_sequence: TensorType,
        epochs: int,
        callbacks: Optional[List] = None,
        verbose: int = 1,
    ) -> TrainingHistory:
        """Обучение на заданное число эпох."""
        history = TrainingHistory(loss_history=[], epochs=epochs)

        for epoch in range(epochs):
            step_result = self.train_step(t_sequence)
            loss_val = float(step_result['loss'])

            if verbose > 0:
                print(f"Epoch {epoch + 1}/{epochs} - loss: {loss_val:.6f}")

            history.loss_history.append(loss_val)

            if callbacks:
                for callback in callbacks:
                    if hasattr(callback, 'on_epoch_end'):
                        callback.on_epoch_end(epoch, {'loss': loss_val})

        self._last_history = history
        return history

    def predict(self, t_sequence: TensorType) -> NetworkOutput:
        """Предсказание без обучения."""
        return self.network(t_sequence, training=False)

    def predict_with_state(
        self,
        t_sequence: TensorType,
        initial_state: Optional[Tuple[StateList, StateList]] = None,
        return_final_state: bool = False,
    ) -> NetworkOutput:
        """
        Предсказание с заданным начальным состоянием.

        Args:
            t_sequence: tf.Tensor shape [batch, T, 1] — временная последовательность.
            initial_state: начальное состояние (pop_states, syn_states).
            return_final_state: вернуть финальное состояние в output.hidden_states.

        Returns:
            NetworkOutput с firing_rates.
        """
        return self.network(
            t_sequence,
            initial_state=initial_state,
            training=False,
            reset_state=False,
        )

    def train_step_with_state(
        self,
        t_sequence: TensorType,
        initial_state: Optional[Tuple[StateList, StateList]] = None,
    ) -> Dict[str, any]:
        """
        Один шаг обучения с заданным начальным состоянием.

        Args:
            t_sequence: tf.Tensor shape [batch, T, 1].
            initial_state: начальное состояние (pop_states, syn_states).

        Returns:
            dict с 'loss' и опционально 'final_state'.
        """
        with tf.GradientTape() as tape:
            output = self.network(t_sequence, initial_state=initial_state, training=True)
            loss = self.loss_fn(output, self.network)

        if self.grad_method == "adjoint":
            grads = adjoint_compute_gradients(self.network, t_sequence, self.loss_fn)
        else:
            grads = tape.gradient(loss, self.network.trainable_variables)

        grads_and_vars = [(g, v) for g, v in zip(grads, self.network.trainable_variables) if g is not None]
        
        if not grads_and_vars:
            return {'loss': loss}

        if self.grad_clip_norm > 0:
            grads_only = [g for g, v in grads_and_vars]
            clipped_grads, _ = tf.clip_by_global_norm(grads_only, self.grad_clip_norm)
            grads_and_vars = [(g, v) for g, (_, v) in zip(clipped_grads, grads_and_vars)]

        self.optimizer.apply_gradients(grads_and_vars)

        return {'loss': loss}

    def save_experiment(self, path: str) -> None:
        """Сохранение состояния эксперимента."""
        os.makedirs(path, exist_ok=True)

        checkpoint_dir = os.path.join(path, 'checkpoint')
        checkpoint = tf.train.Checkpoint(
            optimizer=self.optimizer,
            network=self.network,
        )
        checkpoint.save(file_prefix=os.path.join(checkpoint_dir, 'ckpt'))

        config_data = {
            'stability_penalty_weight': self.network._stability_penalty_weight,
            'grad_clip_norm': self.grad_clip_norm,
        }
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(config_data, f)

        versions_data = {
            'neuraltide': neuraltide.__version__,
            'tensorflow': tf.__version__,
        }
        with open(os.path.join(path, 'versions.json'), 'w') as f:
            json.dump(versions_data, f)

    @classmethod
    def load_experiment(cls, path: str, network: NetworkRNN,
                        loss_fn: CompositeLoss,
                        optimizer: tf.keras.optimizers.Optimizer) -> 'Trainer':
        """Загрузка состояния эксперимента."""
        checkpoint_dir = os.path.join(path, 'checkpoint')
        checkpoint = tf.train.Checkpoint(
            optimizer=optimizer,
            network=network,
        )
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        if latest is not None:
            checkpoint.restore(latest)

        with open(os.path.join(path, 'config.json'), 'r') as f:
            config_data = json.load(f)

        trainer = cls(
            network=network,
            loss_fn=loss_fn,
            optimizer=optimizer,
            grad_clip_norm=config_data.get('grad_clip_norm', 1.0),
        )

        return trainer

    def export_results(
        self,
        path_or_fd: Union[str, os.PathLike],
        format: Literal["json", "csv"] = "json",
        include_config: bool = True,
    ) -> None:
        """
        Экспорт результатов оптимизации в файл.

        Args:
            path_or_fd: путь к файлу или файловый дескриптор.
            format: формат вывода — "json" или "csv".
            include_config: включить конфигурацию сети.
        """
        if format == "json":
            self._export_json(path_or_fd, include_config)
        elif format == "csv":
            self._export_csv(path_or_fd)
        else:
            raise ValueError(f"Unknown format: {format}")

    def _export_json(self, path_or_fd: Union[str, os.PathLike], include_config: bool) -> None:
        results = {
            'trainable_variables': [],
            'loss_history': [],
        }

        for v in self.network.trainable_variables:
            results['trainable_variables'].append({
                'name': v.name,
                'value': float(v.numpy()),
            })

        if hasattr(self, '_last_history'):
            results['loss_history'] = self._last_history.loss_history

        if include_config:
            results['config'] = self._get_config()

        if isinstance(path_or_fd, (str, os.PathLike)):
            with open(path_or_fd, 'w') as f:
                json.dump(results, f, indent=2)
        else:
            json.dump(results, path_or_fd, indent=2)

    def _export_csv(self, path_or_fd: Union[str, os.PathLike]) -> None:
        rows = []

        rows.append(['name', 'value'])

        for v in self.network.trainable_variables:
            rows.append([v.name, float(v.numpy())])

        if isinstance(path_or_fd, (str, os.PathLike)):
            with open(path_or_fd, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(rows)
        else:
            writer = csv.writer(path_or_fd)
            writer.writerows(rows)

    def _get_config(self) -> dict:
        config = {}

        for name, pop in self.network.populations.items():
            config[f'pop_{name}'] = {}
            for param_name, param_spec in pop.params.items():
                value = param_spec['value']
                if hasattr(value, 'numpy'):
                    value = float(value.numpy())
                config[f'pop_{name}'][param_name] = value

        for name in self.network.synapses:
            config[f'syn_{name}'] = {}
            syn = self.network.synapses[name].model
            for param_name, param_spec in syn.params.items():
                value = param_spec['value']
                if hasattr(value, 'numpy'):
                    value = float(value.numpy())
                config[f'syn_{name}'][param_name] = value

        return config
