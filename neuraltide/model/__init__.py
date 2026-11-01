"""
Keras-совместимая обёртка для NetworkRNN.

Позволяет использовать стандартный интерфейс keras:
    from neuraltide.model import BrainModelKeras
    model = BrainModelKeras(graph, integrator, dt, loss_fn)
    model.compile(optimizer='adam')
    model.fit(inputs, targets, epochs=100)
"""
from typing import Dict, List, Optional, Tuple

import tensorflow as tf

import neuraltide
from neuraltide.core.network import NetworkGraph, NetworkRNN, NetworkOutput
from neuraltide.core.base import BaseInputGenerator
from neuraltide.core.types import TensorType, StateList
from neuraltide.integrators.base import BaseIntegrator
from neuraltide.training.losses import BaseLoss, CompositeLoss


class BrainModelKeras(tf.keras.Model):
    """
    Keras Model, оборачивающий NetworkRNN.

    Автоматически генерирует t_sequence из dt и формы входа.

    Args:
        graph: NetworkGraph (с declare_input и add_population/add_synapse)
        integrator: интегратор (EulerIntegrator, RK4Integrator и т.д.)
        dt: шаг интегрирования (мс)
        loss_fn: функция потерь
        stability_penalty_weight: вес штрафа за стабильность

    Example:
        graph = NetworkGraph(dt=0.5)
        graph.declare_input('theta', n_units=1)
        graph.add_population('exc', pop)
        graph.add_synapse('theta->exc', syn, src='theta', tgt='exc')

        model = BrainModelKeras(graph, RK4Integrator(), dt=0.5,
                                loss_fn=MSELoss(target))
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
        model.fit(inputs, targets, epochs=100)
    """

    def __init__(
        self,
        graph: NetworkGraph,
        integrator: BaseIntegrator,
        dt: float,
        loss_fn: Optional[BaseLoss] = None,
        stability_penalty_weight: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._dt = dt
        self._loss_fn = loss_fn
        self._network = NetworkRNN(
            graph, integrator,
            stability_penalty_weight=stability_penalty_weight,
        )
        self._graph = graph

    def call(
        self,
        inputs: TensorType,
        training: bool = False,
        initial_state: Optional[Tuple[StateList, StateList]] = None,
    ) -> NetworkOutput:
        """
        Forward pass.

        Args:
            inputs: [T, total_input_units] или [batch, T, total_input_units]
                firing rates входов. NetworkRNN поддерживает batch=1.
            training: флаг обучения
            initial_state: начальное состояние (опционально)

        Returns:
            NetworkOutput
        """
        # NetworkRNN поддерживает batch=1. Приводим к [1, T, total_units].
        if inputs.shape.rank == 2:
            # [T, units] -> [1, T, units]
            inputs = inputs[tf.newaxis]
        elif inputs.shape.rank == 3:
            # [batch, T, units] -> [1, T, units] (берём первый样本)
            inputs = tf.reshape(inputs[0], [1, tf.shape(inputs)[1], tf.shape(inputs)[2]])

        # Генерируем t_sequence из dt и T
        T = tf.shape(inputs)[1]
        t_values = tf.cast(tf.range(T), inputs.dtype) * self._dt
        t_sequence = t_values[tf.newaxis, :, tf.newaxis]  # [1, T, 1]

        return self._network(t_sequence, inputs=inputs,
                             initial_state=initial_state,
                             training=training)

    def train_step(self, data):
        """Один шаг обучения (Keras convention)."""
        if isinstance(data, tuple):
            x, y = data
        else:
            x = data
            y = None

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)

            # Вычисляем loss
            if self._loss_fn is not None:
                loss = self._loss_fn(y_pred, self._network)
            elif y is not None:
                # Fallback: MSE по всем firing rates
                loss = tf.constant(0.0, dtype=neuraltide.config.get_dtype())
                for name, pred_rate in y_pred.firing_rates.items():
                    if isinstance(y, dict):
                        target_rate = y[name]
                    else:
                        target_rate = y
                    loss = loss + tf.reduce_mean(tf.square(pred_rate - target_rate))
            else:
                raise ValueError("No loss_fn provided and no target data")

        # Обновляем веса
        trainable_vars = self._network.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        grads_and_vars = [(g, v) for g, v in zip(grads, trainable_vars) if g is not None]
        if grads_and_vars:
            self.optimizer.apply_gradients(grads_and_vars)

        # Метрики
        metrics = {'loss': loss}
        if hasattr(self, 'metrics') and self.metrics:
            for metric in self.metrics:
                if metric.name == 'loss':
                    metric.update_state(loss)

        return metrics

    def test_step(self, data):
        """Один шаг валидации."""
        if isinstance(data, tuple):
            x, y = data
        else:
            x = data
            y = None

        y_pred = self(x, training=False)

        if self._loss_fn is not None:
            loss = self._loss_fn(y_pred, self._network)
        elif y is not None:
            loss = tf.constant(0.0, dtype=neuraltide.config.get_dtype())
            for name, pred_rate in y_pred.firing_rates.items():
                if isinstance(y, dict):
                    target_rate = y[name]
                else:
                    target_rate = y
                loss = loss + tf.reduce_mean(tf.square(pred_rate - target_rate))
        else:
            loss = tf.constant(0.0, dtype=neuraltide.config.get_dtype())

        return {'loss': loss}

    @property
    def network(self) -> NetworkRNN:
        """Доступ к внутреннему NetworkRNN."""
        return self._network

    @property
    def trainable_variables(self) -> List[tf.Variable]:
        """Обучаемые переменные сети."""
        return self._network.trainable_variables
