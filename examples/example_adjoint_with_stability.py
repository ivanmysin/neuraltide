"""Пример использования adjoint-метода с штрафом за численную нестабильность.

Показывает, как штраф stability_loss влияет на градиенты при использовании gradient_method='adjoint'.
"""
import numpy as np
import tensorflow as tf

import neuraltide
from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.populations import IzhikevichMeanField
from neuraltide.synapses import TsodyksMarkramSynapse
from neuraltide.inputs import VonMisesGenerator
from neuraltide.integrators import RK4Integrator
from neuraltide.training import Trainer, CompositeLoss, MSELoss
from neuraltide.training.adjoint import AdjointGradientComputer


def create_network_with_instability(dt=0.05):
    """Создаёт сеть, склонную к нестабильности (высокий I_ext)."""
    graph = NetworkGraph(dt=dt)

    # Входной генератор
    gen = VonMisesGenerator(dt=dt, params={
        'mean_rate': {'value': 25.0, 'trainable': False},
        'R': {'value': 0.8, 'trainable': False},
        'freq': {'value': 8.0, 'trainable': False},
        'phase': {'value': 0.0, 'trainable': False},
    })

    # Популяция с высоким внешним током → склонна к нестабильности
    pop_params = {
        'tau_pop': {'value': 1.0, 'trainable': False},
        'alpha': {'value': 0.6, 'trainable': False},
        'a': {'value': 0.02, 'trainable': False},
        'b': {'value': 0.2, 'trainable': False},
        'w_jump': {'value': 0.1, 'trainable': False},
        'Delta_I': {'value': 0.8, 'trainable': True, 'min': 0.1, 'max': 3.0},
        'I_ext': {'value': 2.5, 'trainable': True},   # высокое значение → нестабильность
    }

    pop = IzhikevichMeanField(dt=dt, params=pop_params, name="exc")

    syn_params = {
        'gsyn_max': {'value': [[0.4]], 'trainable': True},
        'tau_f': {'value': 20.0, 'trainable': True},
        'tau_d': {'value': 8.0, 'trainable': True},
        'tau_r': {'value': 100.0, 'trainable': True},
        'Uinc': {'value': 0.3, 'trainable': True},
        'pconn': {'value': [[1.0]], 'trainable': False},
        'e_r': {'value': 0.0, 'trainable': False},
    }

    syn = TsodyksMarkramSynapse(n_pre=1, n_post=1, dt=dt, params=syn_params)

    graph.add_input_population('theta', gen)
    graph.add_population('exc', pop)
    graph.add_synapse('theta->exc', syn, src='theta', tgt='exc')

    return NetworkRNN(
        graph, 
        RK4Integrator(),
        stability_penalty_weight=0.5,   # включаем штраф
        return_hidden_states=False
    )


def main():
    dt = 0.05
    T = 80
    t_values = np.arange(T, dtype=np.float32) * dt
    t_seq = tf.constant(t_values[None, :, None])

    network = create_network_with_instability(dt)

    # Целевая активность — умеренная (чтобы был конфликт со стабильностью)
    target = tf.ones([1, T, 1], dtype=tf.float32) * 0.8
    target_dict = {'exc': target}

    loss_fn = CompositeLoss([
        (1.0, MSELoss(target_dict)),
    ])

    optimizer = tf.keras.optimizers.Adam(0.02)

    print("=== Пример: Adjoint + штраф за нестабильность ===\n")
    print(f"stability_penalty_weight = {network._stability_penalty_weight}\n")

    trainer = Trainer(
        network=network,
        loss_fn=loss_fn,
        optimizer=optimizer,
        gradient_method="adjoint",
        run_eagerly=True,          # для отладки
        grad_clip_norm=1.0
    )

    # Первый шаг — смотрим loss и stability_loss
    result = trainer.train_step(t_seq)
    print(f"Initial loss: {result['loss']:.4f}")

    # Делаем 10 шага обучения
    for i in range(10):
        result = trainer.train_step(t_seq)
        print(f"Step {i+1:2d} - loss: {result['loss']:.4f}")

    print("\nОбучение с adjoint + stability penalty завершено.")
    print("Adjoint-метод учитывает штраф за численную нестабильность.")


if __name__ == "__main__":
    main()
