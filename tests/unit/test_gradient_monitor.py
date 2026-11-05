"""
GradientMonitor получает градиенты из logs["grads"], которые Trainer
пробрасывает в callbacks, сохраняет history с global_norm и per-variable
norms; вне Trainer (без grads в logs) деградирует понятно и не падает.
"""

import numpy as np
import pytest
import tensorflow as tf

from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.populations import IzhikevichMeanField
from neuraltide.synapses import TsodyksMarkramSynapse
from neuraltide.inputs import VonMisesGenerator
from neuraltide.integrators import RK4Integrator
from neuraltide.training import MSELoss, CompositeLoss, Trainer, GradientMonitor


def _make_network(dt=0.1):
    pop = IzhikevichMeanField(
        dt=dt,
        params={
            "tau_pop": {"value": 1.0, "trainable": False},
            "alpha": {"value": 0.5, "trainable": False},
            "a": {"value": 0.02, "trainable": False},
            "b": {"value": 0.2, "trainable": False},
            "w_jump": {"value": 0.1, "trainable": False},
            "Delta_I": {"value": 0.5, "trainable": True},
            "I_ext": {"value": 0.1, "trainable": True},
        },
    )
    gen = VonMisesGenerator(
        dt=dt,
        params={"mean_rate": 20.0, "R": 0.5, "freq": 8.0, "phase": 0.0},
        name="theta_gen",
    )
    syn = TsodyksMarkramSynapse(
        n_pre=1,
        n_post=1,
        dt=dt,
        params={
            "gsyn_max": {"value": [[0.1]], "trainable": True},
            "tau_f": {"value": 20.0, "trainable": False},
            "tau_d": {"value": 5.0, "trainable": False},
            "tau_r": {"value": 200.0, "trainable": False},
            "Uinc": {"value": 0.2, "trainable": False},
            "pconn": {"value": [[1.0]], "trainable": False},
            "e_r": {"value": 0.0, "trainable": False},
        },
    )
    graph = NetworkGraph(dt=dt)
    graph.declare_input("theta", n_units=gen.n_units)
    graph.add_population("exc", pop)
    graph.add_synapse("theta->exc", syn, src="theta", tgt="exc")
    return NetworkRNN(graph, integrator=RK4Integrator()), gen


def _make_data(network, gen, T=30):
    dt = network._graph.dt
    t_values = np.arange(T, dtype=np.float32) * dt
    t_seq = tf.constant(t_values[None, :, None])
    inputs = network._graph.pack_inputs({"theta": gen(t_seq)})
    output = network(t_seq, inputs=inputs, training=False)
    target = {k: v + 0.5 for k, v in output.firing_rates.items()}
    return t_seq, inputs, target


class TestGradientMonitor:
    def test_trainer_passes_grads_and_monitor_records(self):
        network, gen = _make_network()
        t_seq, inputs, target = _make_data(network, gen)
        loss_fn = CompositeLoss([(1.0, MSELoss(target))])
        trainer = Trainer(
            network,
            loss_fn,
            tf.keras.optimizers.Adam(1e-3),
            grad_method="bptt",
        )
        monitor = GradientMonitor(log_every=1)
        trainer.fit(t_seq, inputs, epochs=2, callbacks=[monitor], verbose=0)

        assert len(monitor.history) == 2
        for entry in monitor.history:
            assert entry["global_norm"] > 0.0
            assert len(entry["norms"]) > 0
            assert all(n >= 0.0 for n in entry["norms"])

    def test_global_norm_matches_manual(self):
        network, gen = _make_network()
        t_seq, inputs, target = _make_data(network, gen)
        loss_fn = CompositeLoss([(1.0, MSELoss(target))])

        # градиенты в logs — значения до обновления весов, поэтому эталон
        # считаем до fit
        with tf.GradientTape() as tape:
            output = network(t_seq, inputs=inputs, training=True)
            loss = loss_fn(output, network)
        grads = tape.gradient(loss, network.trainable_variables)
        flat = [tf.reshape(g, [-1]) for g in grads if g is not None]
        expected = float(tf.linalg.global_norm(flat))

        trainer = Trainer(
            network, loss_fn, tf.keras.optimizers.Adam(1e-3), grad_method="bptt"
        )
        monitor = GradientMonitor(log_every=1)
        trainer.fit(t_seq, inputs, epochs=1, callbacks=[monitor], verbose=0)

        assert monitor.history[0]["global_norm"] == pytest.approx(expected, rel=1e-5)

    def test_variables_recorded_when_available(self):
        network, gen = _make_network()
        t_seq, inputs, target = _make_data(network, gen)
        loss_fn = CompositeLoss([(1.0, MSELoss(target))])
        trainer = Trainer(
            network, loss_fn, tf.keras.optimizers.Adam(1e-3), grad_method="bptt"
        )
        monitor = GradientMonitor(log_every=1, variables=network.trainable_variables)
        trainer.fit(t_seq, inputs, epochs=1, callbacks=[monitor], verbose=0)

        entry = monitor.history[0]
        assert "variables" in entry
        assert len(entry["variables"]) == len(entry["norms"])

    def test_log_every_respected(self):
        network, gen = _make_network()
        t_seq, inputs, target = _make_data(network, gen)
        loss_fn = CompositeLoss([(1.0, MSELoss(target))])
        trainer = Trainer(
            network, loss_fn, tf.keras.optimizers.Adam(1e-3), grad_method="bptt"
        )
        monitor = GradientMonitor(log_every=2)
        trainer.fit(t_seq, inputs, epochs=6, callbacks=[monitor], verbose=0)

        assert [e["epoch"] for e in monitor.history] == [0, 2, 4]

    def test_adjoint_grads_in_logs(self):
        network, gen = _make_network()
        t_seq, inputs, target = _make_data(network, gen)
        loss_fn = CompositeLoss([(1.0, MSELoss(target))])
        trainer = Trainer(
            network, loss_fn, tf.keras.optimizers.Adam(1e-3), grad_method="adjoint"
        )
        monitor = GradientMonitor(log_every=1)
        trainer.fit(t_seq, inputs, epochs=1, callbacks=[monitor], verbose=0)

        assert len(monitor.history) == 1
        assert monitor.history[0]["global_norm"] > 0.0

    def test_degrades_gracefully_without_grads(self):
        monitor = GradientMonitor(log_every=1)
        # Keras-style вызов без Trainer: grads в logs отсутствуют
        monitor.on_epoch_end(0, {"loss": 1.0})
        monitor.on_epoch_end(1, {"loss": 0.9})
        assert monitor.history == []

    def test_none_logs_do_not_raise(self):
        monitor = GradientMonitor(log_every=1)
        monitor.on_epoch_end(0, None)
        assert monitor.history == []
