"""
float64-путь NetworkRNN целиком: forward, BPTT-градиенты, сравнение
с float32. set_dtype(tf.float64) вызывается до создания объектов;
conftest сбрасывает dtype после каждого теста.
"""

import numpy as np
import tensorflow as tf

import neuraltide.config
from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.populations import IzhikevichMeanField
from neuraltide.synapses import TsodyksMarkramSynapse
from neuraltide.inputs import VonMisesGenerator
from neuraltide.integrators import RK4Integrator
from neuraltide.training import MSELoss, CompositeLoss, Trainer


def _build(dt=0.1):
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


def _run(network, gen, T=30):
    dt = network._graph.dt
    dtype = neuraltide.config.get_dtype().as_numpy_dtype()
    t_values = np.arange(T, dtype=dtype) * dt
    t_seq = tf.constant(t_values[None, :, None])
    inputs = network._graph.pack_inputs({"theta": gen(t_seq)})
    return t_seq, inputs


class TestFloat64Path:
    def test_forward_preserves_dtype(self):
        neuraltide.config.set_dtype(tf.float64)
        network, gen = _build()
        t_seq, inputs = _run(network, gen)

        output = network(t_seq, inputs=inputs, training=False)
        rates = output.firing_rates["exc"]
        assert rates.dtype == tf.float64
        assert output.stability_loss.dtype == tf.float64
        assert np.all(np.isfinite(rates.numpy()))

    def test_bptt_gradients_float64(self):
        neuraltide.config.set_dtype(tf.float64)
        network, gen = _build()
        t_seq, inputs = _run(network, gen)
        output = network(t_seq, inputs=inputs, training=False)
        target = {k: v + 0.5 for k, v in output.firing_rates.items()}
        loss_fn = CompositeLoss([(1.0, MSELoss(target))])

        with tf.GradientTape() as tape:
            out = network(t_seq, inputs=inputs, training=True)
            loss = loss_fn(out, network)
        grads = tape.gradient(loss, network.trainable_variables)

        assert loss.dtype == tf.float64
        assert len(grads) == len(network.trainable_variables)
        for g in grads:
            assert g is not None
            assert g.dtype == tf.float64
            assert np.all(np.isfinite(g.numpy()))

    def test_trainer_bptt_float64(self):
        neuraltide.config.set_dtype(tf.float64)
        network, gen = _build()
        t_seq, inputs = _run(network, gen)
        output = network(t_seq, inputs=inputs, training=False)
        target = {k: v + 0.5 for k, v in output.firing_rates.items()}
        loss_fn = CompositeLoss([(1.0, MSELoss(target))])

        trainer = Trainer(
            network, loss_fn, tf.keras.optimizers.Adam(1e-3), grad_method="bptt"
        )
        monitor_logs = []

        class _Recorder:
            def on_epoch_end(self, epoch, logs=None):
                monitor_logs.append(logs)

        trainer.fit(t_seq, inputs, epochs=1, callbacks=[_Recorder()], verbose=0)

        assert len(monitor_logs) == 1
        assert monitor_logs[0]["grads"] is not None
        assert all(g.dtype == tf.float64 for g in monitor_logs[0]["grads"])
        assert np.isfinite(monitor_logs[0]["loss"])

    def test_float64_close_to_float32(self):
        # та же конфигурация в float32 и float64 даёт близкие потери
        neuraltide.config.set_dtype(tf.float32)
        network32, gen32 = _build()
        t32, inputs32 = _run(network32, gen32)
        out32 = network32(t32, inputs=inputs32)
        loss32 = float(tf.reduce_mean(out32.firing_rates["exc"]))

        neuraltide.config.set_dtype(tf.float64)
        network64, gen64 = _build()
        t64, inputs64 = _run(network64, gen64)
        out64 = network64(t64, inputs=inputs64)
        loss64 = float(tf.reduce_mean(out64.firing_rates["exc"]))

        assert abs(loss64 - loss32) / max(abs(loss32), 1e-12) < 1e-2

    def test_float32_path_unchanged(self):
        # поведение по умолчанию (float32) не изменилось
        neuraltide.config.set_dtype(tf.float32)
        network, gen = _build()
        t_seq, inputs = _run(network, gen)
        output = network(t_seq, inputs=inputs, training=False)
        assert output.firing_rates["exc"].dtype == tf.float32
