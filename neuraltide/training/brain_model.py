"""
BrainModel — высокоуровневый класс, объединяющий генераторы входов и NetworkRNN.

Оркестрирует:
    1. Генераторы вычисляют firing rates из (t, extra_inputs)
    2. pack_inputs() складывает их в один тензор
    3. NetworkRNN симулирует динамику
    4. Loss вычисляется на выходе

Поддерживает два optimizer'а:
    - network_optimizer: обновляет параметры NetworkRNN
    - generator_optimizer: обновляет параметры генераторов (опционально)
"""
from typing import Dict, List, Optional, Tuple

import tensorflow as tf

from neuraltide.core.network import NetworkRNN, NetworkOutput
from neuraltide.core.base import BaseInputGenerator
from neuraltide.core.types import TensorType, StateList
from neuraltide.training.losses import CompositeLoss
from neuraltide.training.trainer import TrainingHistory


class BrainModel(tf.keras.layers.Layer):
    """
    Высокоуровневая обёртка: input generators + NetworkRNN.

    Example:
        gen = VonMisesGenerator(dt=dt, params={...})
        pop = IzhikevichMeanField(dt=dt, params={...})
        syn = TsodyksMarkramSynapse(n_pre=1, n_post=2, dt=dt, params={...})

        graph = NetworkGraph(dt=dt)
        graph.declare_input('theta', n_units=gen.n_units)
        graph.add_population('exc', pop)
        graph.add_synapse('theta->exc', syn, src='theta', tgt='exc')

        model = BrainModel(
            input_generators={'theta': gen},
            network=NetworkRNN(graph, RK4Integrator()),
            network_optimizer=tf.keras.optimizers.Adam(1e-3),
            loss_fn=CompositeLoss([(1.0, MSELoss(target))]),
        )

        history = model.fit(
            t_sequence=t_seq,
            extra_inputs={'theta': xy_tensor},
            target={'exc': target_rates},
            epochs=1000,
        )
    """

    def __init__(
        self,
        input_generators: Dict[str, BaseInputGenerator],
        network: NetworkRNN,
        network_optimizer: tf.keras.optimizers.Optimizer,
        generator_optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
        loss_fn: Optional[CompositeLoss] = None,
        grad_clip_norm: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._generators = input_generators
        self._network = network
        self._network_optimizer = network_optimizer
        self._generator_optimizer = generator_optimizer
        self._loss_fn = loss_fn
        self._grad_clip_norm = grad_clip_norm

    def _precompute_inputs(
        self,
        t_sequence: TensorType,
        extra_inputs: Optional[Dict[str, TensorType]] = None,
    ) -> TensorType:
        """Запускает генераторы и складывает в один тензор.

        Returns:
            Tensor[batch, T, total_input_units]
        """
        input_dict = {}
        for name, gen in self._generators.items():
            extra = extra_inputs.get(name) if extra_inputs else None
            rates = gen(t_sequence, extra_inputs=extra)  # [batch, T, n_units]
            input_dict[name] = rates
        return self._network._graph.pack_inputs(input_dict)

    def forward(
        self,
        t_sequence: TensorType,
        extra_inputs: Optional[Dict[str, TensorType]] = None,
        initial_state: Optional[Tuple[StateList, StateList]] = None,
    ) -> NetworkOutput:
        """Forward pass: генераторы → входы → сеть."""
        inputs = self._precompute_inputs(t_sequence, extra_inputs)
        return self._network(t_sequence, inputs=inputs,
                             initial_state=initial_state)

    def train_step(
        self,
        t_sequence: TensorType,
        target: Dict[str, TensorType],
        extra_inputs: Optional[Dict[str, TensorType]] = None,
        initial_state: Optional[Tuple[StateList, StateList]] = None,
    ) -> Dict[str, float]:
        """Один шаг обучения с двойным потоком градиентов."""
        with tf.GradientTape(persistent=True) as tape:
            inputs = self._precompute_inputs(t_sequence, extra_inputs)
            output = self._network(t_sequence, inputs=inputs,
                                   training=True, initial_state=initial_state)
            loss = self._loss_fn(output, self._network)

        # Градиенты по сети
        network_grads = tape.gradient(loss, self._network.trainable_variables)
        # Градиенты по генераторам
        gen_vars = []
        for gen in self._generators.values():
            gen_vars.extend(gen.trainable_variables)
        gen_grads = tape.gradient(loss, gen_vars)
        del tape

        # Применение градиентов сети
        nv_pairs = [(g, v) for g, v in zip(network_grads,
                     self._network.trainable_variables) if g is not None]
        if nv_pairs:
            if self._grad_clip_norm > 0:
                gs = [g for g, _ in nv_pairs]
                clipped, _ = tf.clip_by_global_norm(gs, self._grad_clip_norm)
                nv_pairs = [(g, v) for (_, v), g in zip(nv_pairs, clipped)]
            self._network_optimizer.apply_gradients(nv_pairs)

        # Применение градиентов генераторов
        if self._generator_optimizer is not None and gen_vars:
            gv_pairs = [(g, v) for g, v in zip(gen_grads, gen_vars)
                        if g is not None]
            if gv_pairs:
                if self._grad_clip_norm > 0:
                    gs = [g for g, _ in gv_pairs]
                    clipped, _ = tf.clip_by_global_norm(gs, self._grad_clip_norm)
                    gv_pairs = [(g, v) for (_, v), g in zip(gv_pairs, clipped)]
                self._generator_optimizer.apply_gradients(gv_pairs)

        return {'loss': loss}

    def fit(
        self,
        t_sequence: TensorType,
        epochs: int,
        target: Dict[str, TensorType],
        extra_inputs: Optional[Dict[str, TensorType]] = None,
        initial_state: Optional[Tuple[StateList, StateList]] = None,
        callbacks: Optional[List] = None,
        verbose: int = 1,
    ) -> TrainingHistory:
        """Обучение на заданное число эпох."""
        history = TrainingHistory(loss_history=[], epochs=epochs)
        for epoch in range(epochs):
            result = self.train_step(t_sequence, target, extra_inputs,
                                     initial_state)
            loss_val = float(result['loss'])
            history.loss_history.append(loss_val)
            if verbose > 0 and (epoch + 1) % max(1, epochs // 20) == 0:
                print(f"Epoch {epoch + 1}/{epochs} - loss: {loss_val:.6f}")
            if callbacks:
                for cb in callbacks:
                    if hasattr(cb, 'on_epoch_end'):
                        cb.on_epoch_end(epoch, {'loss': loss_val})
        return history

    @property
    def trainable_variables(self) -> List[tf.Variable]:
        """Все обучаемые переменные (сеть + генераторы)."""
        vars_ = list(self._network.trainable_variables)
        for gen in self._generators.values():
            vars_.extend(gen.trainable_variables)
        return vars_
