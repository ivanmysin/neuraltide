"""Пример использования adjoint-метода с штрафом за численную нестабильность.

Показывает, как штраф stability_loss влияет на градиенты.
"""
import numpy as np
import tensorflow as tf

import neuraltide
from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.populations import IzhikevichMeanField
from neuraltide.synapses import TsodyksMarkramSynapse
from neuraltide.inputs import VonMisesGenerator
from neuraltide.integrators import RK4Integrator
from neuraltide.training import Trainer, CompositeLoss, MSELoss, StabilityPenalty
from neuraltide.utils import seed_everything


def create_network_with_instability(dt=0.05):
    """Создаёт сеть, склонную к нестабильности (высокий I_ext)."""
    graph = NetworkGraph(dt=dt)

    gen = VonMisesGenerator(dt=dt, params={
        'mean_rate': {'value': 25.0, 'trainable': False},
        'R': {'value': 0.8, 'trainable': False},
        'freq': {'value': 8.0, 'trainable': False},
        'phase': {'value': 0.0, 'trainable': False},
    })

    pop = IzhikevichMeanField(dt=dt, params={
        'tau_pop': {'value': 1.0, 'trainable': False},
        'alpha': {'value': 0.6, 'trainable': False},
        'a': {'value': 0.02, 'trainable': False},
        'b': {'value': 0.2, 'trainable': False},
        'w_jump': {'value': 0.1, 'trainable': False},
        'Delta_I': {'value': 0.8, 'trainable': True, 'min': 0.1, 'max': 3.0},
        'I_ext': {'value': 2.5, 'trainable': True},
    })

    syn = TsodyksMarkramSynapse(n_pre=1, n_post=1, dt=dt, params={
        'gsyn_max': {'value': [[0.4]], 'trainable': True},
        'tau_f': {'value': 20.0, 'trainable': True},
        'tau_d': {'value': 8.0, 'trainable': True},
        'tau_r': {'value': 100.0, 'trainable': True},
        'Uinc': {'value': 0.3, 'trainable': True},
        'pconn': {'value': [[1.0]], 'trainable': False},
        'e_r': {'value': 0.0, 'trainable': False},
    })

    graph.declare_input('theta', n_units=gen.n_units)
    graph.add_population('exc', pop)
    graph.add_synapse('theta->exc', syn, src='theta', tgt='exc')

    return graph, gen, NetworkRNN(
        graph,
        RK4Integrator(),
        stability_penalty_weight=0.5,
    )


def main():
    dt = 0.05
    T = 10
    n_steps = int(T / dt)

    seed_everything(42)

    graph, gen, network = create_network_with_instability(dt)

    t_values = np.arange(n_steps, dtype=np.float32) * dt
    t_seq = tf.constant(t_values[None, :, None])
    inputs = graph.pack_inputs({'theta': gen(t_seq)})

    target = {
        'exc': tf.constant(
            (10.0 + 3.0 * np.sin(2 * np.pi * 8.0 * t_values / 1000.0))[None, :, None],
            dtype=tf.float32
        )
    }

    loss_fn = CompositeLoss([
        (1.0, MSELoss(target)),
        (0.5, StabilityPenalty()),
    ])

    trainer = Trainer(network, loss_fn,
                      optimizer=tf.keras.optimizers.Adam(1e-3))

    print("Training with stability penalty...")
    history = trainer.fit(t_seq, inputs=inputs, epochs=100, verbose=2)

    print(f"\nFinal loss: {history.loss_history[-1]:.4f}")
    print("Training completed successfully!")


if __name__ == '__main__':
    main()
