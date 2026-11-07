"""Этап 4: чанкинг длинных временных рядов (память O(L) вместо O(T)).

Сегмент = переиспользование self._scan_compiled на срезе протокола с
carry = final_state предыдущего сегмента. 'checkpointed' — точный BPTT
через tf.custom_gradient (пересчёт сегмента при backward; tf.recompute_grad
на этой конфигурации TF теряет градиенты по переменным сети);
'truncated' — stop_gradient на carry (дёшево, градиент смещён). Покрытие:
эквивалентность прямого прохода и градиентов целому прогону (в т.ч.
неравный последний сегмент и взвешенный stability-штраф), eager/граф,
compiled-ядро, Trainer, батч, shooting, валидации, float64, длинный T.
"""

import time

import numpy as np
import pytest
import tensorflow as tf

import neuraltide.config
from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.integrators import EulerIntegrator, RK4Integrator
from neuraltide.optimization import (
    CalibrationProblem,
    Experiment,
    MultiExperimentProblem,
    MultipleShootingProblem,
    ObservationSpec,
    ParameterSpace,
    ResidualLoss,
    ShootingConfig,
)
from neuraltide.populations import BrunelFokkerPlanck, WilsonCowan
from neuraltide.synapses import StaticSynapse
from neuraltide.training import CompositeLoss, MSELoss, Trainer

DT = 0.5
T = 60

RTOL32, ATOL32 = 1e-5, 1e-6


def _t_seq(n, dt=DT):
    dtype = neuraltide.config.get_dtype().as_numpy_dtype()
    return tf.constant((np.arange(n, dtype=dtype) * dt)[None, :, None])


def build_l0():
    """L0-подобная сеть (параметры как в conftest._build_l0)."""
    pop = WilsonCowan(
        dt=DT,
        params={
            "tau": {"value": [8.0], "trainable": True, "min": 1.0, "max": 50.0},
            "a": 1.0,
            "theta": 0.0,
            "w": 0.2,
            "I_ext": {"value": [0.8], "trainable": True},
            "max_rate": 100.0,
        },
        name="exc",
    )
    syn = StaticSynapse(
        n_pre=1,
        n_post=1,
        dt=DT,
        params={
            "gsyn_max": {"value": [[0.05]], "trainable": True},
            "pconn": [[1.0]],
            "e_r": 0.0,
        },
    )
    graph = NetworkGraph(dt=DT)
    graph.declare_input("in", n_units=1)
    graph.add_population("exc", pop)
    graph.add_synapse("in->exc", syn, src="in", tgt="exc")
    return NetworkRNN(graph, EulerIntegrator())


def build_l0_data(network, T=T):
    from neuraltide.inputs import ConstantRateGenerator

    t_seq = _t_seq(T)
    gen = ConstantRateGenerator(dt=DT, params={"rate": 10.0})
    inputs = network._graph.pack_inputs({"in": gen(t_seq)})
    return t_seq, inputs


def build_brunel():
    pop = BrunelFokkerPlanck(
        dt=DT,
        params={
            "tau_m": 20.0,
            "v_theta": 20.0,
            "v_reset": 10.0,
            "tau_rp": 2.0,
            "mu_0": 20.0,
            "sigma_ext": 2.0,
        },
        grid_size=120,
        v_min=-80.0,
    )
    graph = NetworkGraph(dt=DT)
    with pytest.warns(UserWarning, match="has no incoming synapses"):
        graph.add_population("brunel", pop)
        network = NetworkRNN(graph, EulerIntegrator())
    return network


class TestChunkedForward:
    @pytest.mark.parametrize("chunk_length", [16, 32, 7])
    @pytest.mark.parametrize("chunk_grad", ["checkpointed", "truncated"])
    def test_chunked_forward_equals_whole(self, chunk_length, chunk_grad):
        net = build_l0()
        t_seq, inputs = build_l0_data(net)
        out_w = net(t_seq, inputs=inputs)
        out_c = net(
            t_seq,
            inputs=inputs,
            chunk_length=chunk_length,
            chunk_grad=chunk_grad,
        )
        np.testing.assert_allclose(
            out_c.firing_rates["exc"].numpy(),
            out_w.firing_rates["exc"].numpy(),
            rtol=RTOL32,
            atol=ATOL32,
        )
        for a, b in zip(out_c.final_state[0], out_w.final_state[0]):
            np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=RTOL32, atol=ATOL32)
        for a, b in zip(out_c.final_state[1], out_w.final_state[1]):
            np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=RTOL32, atol=ATOL32)

    def test_chunked_brunel(self):
        net = build_brunel()
        t_seq = _t_seq(40)
        out_w = net(t_seq)
        out_c = net(t_seq, chunk_length=16)
        np.testing.assert_allclose(
            out_c.firing_rates["brunel"].numpy(),
            out_w.firing_rates["brunel"].numpy(),
            rtol=1e-4,
            atol=1e-5,
        )

    def test_chunked_no_rates(self):
        net = build_l0()
        t_seq, inputs = build_l0_data(net)
        out = net(t_seq, inputs=inputs, chunk_length=16, return_rates=False)
        assert out.firing_rates == {}
        out_w = net(t_seq, inputs=inputs)
        np.testing.assert_allclose(
            out.final_state[0][0].numpy(),
            out_w.final_state[0][0].numpy(),
            rtol=RTOL32,
            atol=ATOL32,
        )

    def test_chunked_eager_vs_graph(self):
        net = build_l0()
        t_seq, inputs = build_l0_data(net)

        out_e = net(t_seq, inputs=inputs, chunk_length=16)

        @tf.function
        def graph_run():
            return net(t_seq, inputs=inputs, chunk_length=16)

        out_g = graph_run()
        np.testing.assert_allclose(
            out_g.firing_rates["exc"].numpy(),
            out_e.firing_rates["exc"].numpy(),
            rtol=RTOL32,
            atol=ATOL32,
        )

    def test_chunked_batch(self):
        net = build_l0()
        t_seq, inputs = build_l0_data(net)
        t_b = tf.concat([t_seq] * 3, axis=0)
        i_b = tf.concat([inputs] * 3, axis=0)
        out_b = net(t_b, inputs=i_b, chunk_length=16)
        out_1 = net(t_seq, inputs=inputs, chunk_length=16)
        for slot in range(3):
            np.testing.assert_allclose(
                out_b.firing_rates["exc"].numpy()[slot],
                out_1.firing_rates["exc"].numpy()[0],
                rtol=RTOL32,
                atol=ATOL32,
            )

    def test_weighted_stability_formula(self):
        """stability_loss чанкованного прогона = sum(stab_k·L_k)/T
        (проверка формулы на RK4-сети с ненулевым local_err; сегментные
        stab независимо получены прямыми сегментными прогонами)."""
        pop = WilsonCowan(
            dt=DT,
            params={
                "tau": [8.0],
                "a": 1.0,
                "theta": 0.0,
                "w": 0.2,
                "I_ext": [0.8],
                "max_rate": 100.0,
            },
            name="exc",
        )
        graph = NetworkGraph(dt=DT)
        graph.add_population("exc", pop)
        net = NetworkRNN(graph, RK4Integrator(), stability_penalty_weight=2.5)
        t_seq = _t_seq(37)  # неравный последний сегмент
        L = 16
        carry_p, carry_s = tuple(net._init_pop_states), tuple(net._init_syn_states)
        manual = 0.0
        for k in range(-(-37 // L)):
            s = k * L
            e = min(37, s + L)
            _, st, fp, fs, _, _ = net._scan_compiled(
                t_seq[:, s:e],
                carry_p,
                carry_s,
                tf.zeros([1, e - s, 0], dtype=neuraltide.config.get_dtype()),
            )
            manual += float(st) * (e - s)
            carry_p, carry_s = fp, fs
        out_c = net(t_seq, chunk_length=L)
        assert float(out_c.stability_loss) == pytest.approx(manual / 37, rel=1e-5)


class TestChunkedGradients:
    def test_checkpointed_gradient_equals_full(self):
        def build(chunk):
            net = build_l0()
            t_seq, inputs = build_l0_data(net)
            out = net(t_seq, inputs=inputs)
            target = {k: v + 0.5 for k, v in out.firing_rates.items()}
            loss_fn = CompositeLoss([(1.0, MSELoss(target))])
            return (
                net,
                t_seq,
                inputs,
                Trainer(
                    net, loss_fn, tf.keras.optimizers.SGD(0.01), chunk_length=chunk
                ),
            )

        net_w, t_w, i_w, tr_w = build(None)
        net_c, t_c, i_c, tr_c = build(16)
        for _ in range(2):
            tr_w.train_step(t_w, i_w)
            tr_c.train_step(t_c, i_c)
        for v_w, v_c in zip(net_w.trainable_variables, net_c.trainable_variables):
            np.testing.assert_allclose(v_c.numpy(), v_w.numpy(), rtol=1e-4, atol=1e-5)

    def test_checkpointed_gradient_float64(self):
        neuraltide.config.set_dtype(tf.float64)
        net = build_l0()
        t_seq, inputs = build_l0_data(net)
        target = {k: v + 0.5 for k, v in net(t_seq, inputs=inputs).firing_rates.items()}
        loss_fn = CompositeLoss([(1.0, MSELoss(target))])

        def grads(chunk):
            with tf.GradientTape() as tape:
                out = net(t_seq, inputs=inputs, chunk_length=chunk)
                loss = loss_fn(out, net)
            return [g.numpy() for g in tape.gradient(loss, net.trainable_variables)]

        g_w = grads(None)
        g_c = grads(16)
        for a, b in zip(g_c, g_w):
            np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-11)

    def test_truncated_gradient_differs(self):
        net = build_l0()
        t_seq, inputs = build_l0_data(net)
        target = {k: v + 0.5 for k, v in net(t_seq, inputs=inputs).firing_rates.items()}
        loss_fn = CompositeLoss([(1.0, MSELoss(target))])

        def grads(chunk_grad):
            with tf.GradientTape() as tape:
                out = net(t_seq, inputs=inputs, chunk_length=16, chunk_grad=chunk_grad)
                loss = loss_fn(out, net)
            return tf.concat(
                [
                    tf.reshape(g, [-1])
                    for g in tape.gradient(loss, net.trainable_variables)
                ],
                axis=0,
            ).numpy()

        g_full = grads("checkpointed")
        g_trunc = grads("truncated")
        # Смещение truncated: градиенты различаются существенно.
        assert not np.allclose(g_full, g_trunc, rtol=1e-3, atol=1e-5)
        # И прямой проход идентичен (смещение только в градиенте).
        out_f = net(t_seq, inputs=inputs, chunk_length=16, chunk_grad="checkpointed")
        out_t = net(t_seq, inputs=inputs, chunk_length=16, chunk_grad="truncated")
        np.testing.assert_allclose(
            out_t.firing_rates["exc"].numpy(),
            out_f.firing_rates["exc"].numpy(),
            rtol=RTOL32,
            atol=ATOL32,
        )

    def test_problem_compiled_chunked(self):
        def build(chunk, compiled):
            net = build_l0()
            t_seq, inputs = build_l0_data(net)
            target = net(t_seq, inputs=inputs).firing_rates["exc"]
            loss = ResidualLoss(
                [
                    ObservationSpec(
                        name="obs", population="exc", target=target, sigma=2.0
                    )
                ]
            )
            space = ParameterSpace.from_network(net)
            return CalibrationProblem(
                net,
                space,
                t_seq,
                inputs=inputs,
                objective=loss,
                grad_mode="chain",
                compiled=compiled,
                chunk_length=chunk,
            )

        rng = np.random.default_rng(0)
        p_w = build(None, False)
        p_c = build(16, False)
        p_cc = build(16, True)
        phi = (
            p_w.pack().numpy() + rng.standard_normal(p_w.size).astype(np.float32) * 0.02
        )
        ev_w, g_w = p_w.value_and_gradient(phi)
        ev_c, g_c = p_c.value_and_gradient(phi)
        ev_cc, g_cc = p_cc.value_and_gradient(phi)
        assert ev_c.loss == pytest.approx(ev_w.loss, rel=RTOL32, abs=ATOL32)
        np.testing.assert_allclose(g_c.numpy(), g_w.numpy(), rtol=1e-4, atol=1e-5)
        assert ev_cc.loss == pytest.approx(ev_w.loss, rel=RTOL32, abs=ATOL32)
        np.testing.assert_allclose(g_cc.numpy(), g_w.numpy(), rtol=1e-4, atol=1e-5)
        r_w = p_w.residual(phi)
        r_c = p_cc.residual(phi)
        np.testing.assert_allclose(r_c, r_w, rtol=RTOL32, atol=ATOL32)
        # Ядра с разными chunk-параметрами — разные записи кэша.
        assert len(p_cc._core_cache) == 1  # одно ядро на задачу
        p_cc.evaluate(phi)
        assert len(p_cc._core_cache) == 1

    def test_long_T_memory(self):
        """Косвенный тест памяти: T=20000 с лентой выполняется, градиент
        конечен (без жёстких порогов — главное, что проходит)."""
        net = build_l0()
        t_seq, inputs = build_l0_data(net, T=20000)
        t0 = time.perf_counter()
        with tf.GradientTape() as tape:
            out = net(t_seq, inputs=inputs, chunk_length=200)
            loss = tf.reduce_mean(out.firing_rates["exc"])
        grads = tape.gradient(loss, net.trainable_variables)
        dt_ = time.perf_counter() - t0
        assert all(np.all(np.isfinite(g.numpy())) for g in grads if g is not None)
        assert out.firing_rates["exc"].shape[1] == 20000
        assert dt_ < 120.0  # мягкий предохранитель, не порог памяти


class TestChunkedValidations:
    def test_dynamic_T_raises(self):
        net = build_l0()
        t_seq, inputs = build_l0_data(net)

        @tf.function(input_signature=[tf.TensorSpec([None, None, 1], tf.float32)])
        def dyn(x):
            return net(x, chunk_length=8)

        with pytest.raises(ValueError, match="statically known time length"):
            dyn(t_seq)

    def test_bad_chunk_length_raises(self):
        net = build_l0()
        t_seq, inputs = build_l0_data(net)
        with pytest.raises(ValueError, match="positive int"):
            net(t_seq, inputs=inputs, chunk_length=0)
        with pytest.raises(ValueError, match="positive int"):
            net(t_seq, inputs=inputs, chunk_length=-3)

    def test_bad_chunk_grad_raises(self):
        net = build_l0()
        t_seq, inputs = build_l0_data(net)
        with pytest.raises(ValueError, match="chunk_grad"):
            net(t_seq, inputs=inputs, chunk_length=8, chunk_grad="bad")

    def test_adjoint_chunk_raises(self):
        net = build_l0()
        t_seq, inputs = build_l0_data(net)
        target = net(t_seq, inputs=inputs).firing_rates["exc"]
        loss = ResidualLoss(
            [ObservationSpec(name="obs", population="exc", target=target, sigma=2.0)]
        )
        space = ParameterSpace.from_network(net)
        with pytest.raises(ValueError, match="adjoint"):
            CalibrationProblem(
                net,
                space,
                t_seq,
                inputs=inputs,
                objective=loss,
                grad_mode="adjoint",
                chunk_length=8,
            )
        with pytest.raises(ValueError, match="adjoint"):
            Trainer(
                net,
                CompositeLoss([(1.0, MSELoss({"exc": target}))]),
                tf.keras.optimizers.SGD(0.01),
                grad_method="adjoint",
                chunk_length=8,
            )


class TestChunkedShooting:
    def test_chunked_shooting(self):
        def build(chunk):
            net = build_l0()
            t_seq, inputs = build_l0_data(net)
            target = net(t_seq, inputs=inputs).firing_rates["exc"]
            loss = ResidualLoss(
                [
                    ObservationSpec(
                        name="obs", population="exc", target=target, sigma=2.0
                    )
                ]
            )
            space = ParameterSpace.from_network(net)
            base = CalibrationProblem(
                net,
                space,
                t_seq,
                inputs=inputs,
                objective=loss,
                grad_mode="chain",
                chunk_length=chunk,
            )
            return MultipleShootingProblem(
                base, ShootingConfig(3), continuity_weight=100.0
            )

        rng = np.random.default_rng(1)

        def phi_of(problem):
            values = {}
            for n, e in problem._space.phi_entries.items():
                values[n] = (
                    e.variable.numpy()
                    if e.variable is not None
                    else np.zeros(e.shape, np.float32)
                )
            phi = np.asarray(problem._space.pack(values=values), np.float32).reshape(-1)
            return phi + rng.standard_normal(problem.size).astype(np.float32) * 0.01

        ms_w = build(None)
        ms_c = build(16)
        phi = phi_of(ms_w)
        ev_w, g_w = ms_w.value_and_gradient(phi)
        ev_c, g_c = ms_c.value_and_gradient(phi)
        assert ev_c.loss == pytest.approx(ev_w.loss, rel=RTOL32, abs=ATOL32)
        np.testing.assert_allclose(g_c.numpy(), g_w.numpy(), rtol=1e-4, atol=1e-5)
        assert ms_c.finalize(phi) is not None
        assert ms_c.to_single_shooting_problem(phi) is not None


class TestChunkedMultiExperiment:
    def test_vectorized_chunked(self):
        def build(chunk):
            net = build_l0()
            t_seq, inputs = build_l0_data(net)
            inputs2 = inputs * 0.5
            t1 = net(t_seq, inputs=inputs).firing_rates["exc"]
            t2 = net(t_seq, inputs=inputs2).firing_rates["exc"]

            def _loss(target):
                return ResidualLoss(
                    [
                        ObservationSpec(
                            name="obs", population="exc", target=target, sigma=2.0
                        )
                    ]
                )

            experiments = [
                Experiment(
                    name="e1", t_sequence=t_seq, inputs=inputs, objective=_loss(t1)
                ),
                Experiment(
                    name="e2", t_sequence=t_seq, inputs=inputs2, objective=_loss(t2)
                ),
            ]
            space = ParameterSpace.from_network(net)
            return MultiExperimentProblem(
                net,
                space,
                experiments,
                grad_mode="chain",
                vectorize=True,
                chunk_length=chunk,
            )

        rng = np.random.default_rng(2)
        p_w = build(None)
        p_c = build(16)
        phi = (
            p_w.pack().numpy() + rng.standard_normal(p_w.size).astype(np.float32) * 0.02
        )
        ev_w, g_w = p_w.value_and_gradient(phi)
        ev_c, g_c = p_c.value_and_gradient(phi)
        assert ev_c.loss == pytest.approx(ev_w.loss, rel=RTOL32, abs=ATOL32)
        np.testing.assert_allclose(g_c.numpy(), g_w.numpy(), rtol=1e-4, atol=1e-5)


class TestChunkedX0AndLosses:
    def test_checkpointed_x0_gradient(self):
        """Trainable x0-запись (common) + chunk_length: градиент (eager и
        compiled) == задача без чанкинга; x0-градиент ненулевой."""
        from neuraltide.inputs import ConstantRateGenerator

        def build(chunk, compiled):
            pop = WilsonCowan(
                dt=DT,
                params={
                    "tau": [8.0],
                    "a": 1.0,
                    "theta": 0.0,
                    "w": 0.2,
                    "I_ext": [0.8],
                    "max_rate": 100.0,
                },
                name="exc",
            )
            syn = StaticSynapse(
                n_pre=1,
                n_post=1,
                dt=DT,
                params={"gsyn_max": [[0.05]], "pconn": [[1.0]], "e_r": 0.0},
            )
            graph = NetworkGraph(dt=DT)
            graph.declare_input("in", n_units=1)
            graph.add_population("exc", pop)
            graph.add_synapse("in->exc", syn, src="in", tgt="exc")
            network = NetworkRNN(graph, EulerIntegrator())
            space = ParameterSpace.from_network(network)
            space.add_initial_state_entries(network, include=[("exc", 0)])
            t_seq, inputs = (
                _t_seq(T),
                graph.pack_inputs(
                    {
                        "in": ConstantRateGenerator(dt=DT, params={"rate": 10.0})(
                            _t_seq(T)
                        )
                    }
                ),
            )
            target = network(t_seq, inputs=inputs).firing_rates["exc"]
            loss = ResidualLoss(
                [
                    ObservationSpec(
                        name="obs", population="exc", target=target, sigma=2.0
                    )
                ]
            )
            return CalibrationProblem(
                network,
                space,
                t_seq,
                inputs=inputs,
                objective=loss,
                grad_mode="chain",
                compiled=compiled,
                chunk_length=chunk,
            )

        rng = np.random.default_rng(3)
        p_w = build(None, False)
        p_c = build(16, False)
        p_cc = build(16, True)
        values = {}
        for n, e in p_w.parameter_space.phi_entries.items():
            values[n] = (
                e.variable.numpy()
                if e.variable is not None
                else np.zeros(e.shape, np.float32)
            )
        phi = (
            np.asarray(p_w.parameter_space.pack(values=values), np.float32).reshape(-1)
            + rng.standard_normal(p_w.size).astype(np.float32) * 0.02
        )
        ev_w, g_w = p_w.value_and_gradient(phi)
        ev_c, g_c = p_c.value_and_gradient(phi)
        ev_cc, g_cc = p_cc.value_and_gradient(phi)
        assert ev_c.loss == pytest.approx(ev_w.loss, rel=RTOL32, abs=ATOL32)
        np.testing.assert_allclose(g_c.numpy(), g_w.numpy(), rtol=1e-4, atol=1e-5)
        assert ev_cc.loss == pytest.approx(ev_w.loss, rel=RTOL32, abs=ATOL32)
        np.testing.assert_allclose(g_cc.numpy(), g_w.numpy(), rtol=1e-4, atol=1e-5)
        idx_x0 = list(p_w.parameter_space.phi_entries).index("x0/exc/0")
        assert abs(float(g_c[idx_x0])) > 0
        assert abs(float(g_cc[idx_x0])) > 0

    def test_antiphase_chunked(self):
        """Неаддитивная потеря (AntiPhaseLoss) + chunk_length: forward и
        градиент == целый прогон."""
        from neuraltide.training import AntiPhaseLoss

        def build_net():
            p1 = WilsonCowan(
                dt=DT,
                params={
                    "tau": [8.0],
                    "a": 1.0,
                    "theta": 0.0,
                    "w": [0.1],
                    "I_ext": [0.8],
                    "max_rate": 100.0,
                },
                name="p1",
            )
            p2 = WilsonCowan(
                dt=DT,
                params={
                    "tau": [9.0],
                    "a": 1.0,
                    "theta": 0.0,
                    "w": [0.1],
                    "I_ext": [0.7],
                    "max_rate": 100.0,
                },
                name="p2",
            )
            inh = StaticSynapse(
                n_pre=1,
                n_post=1,
                dt=DT,
                params={"gsyn_max": [[0.3]], "pconn": [[1.0]], "e_r": -5.0},
            )
            graph = NetworkGraph(dt=DT)
            graph.add_population("p1", p1)
            graph.add_population("p2", p2)
            graph.add_synapse("p2->p1", inh, src="p2", tgt="p1")
            graph.add_synapse("p1->p2", inh, src="p1", tgt="p2")
            return NetworkRNN(graph, EulerIntegrator())

        net = build_net()
        t_seq = _t_seq(T)
        loss_fn = AntiPhaseLoss(
            [("p1", "p2")], target_correlation=-1.0, activity_target=0.0
        )

        def loss_and_grads(chunk):
            with tf.GradientTape() as tape:
                out = net(t_seq, chunk_length=chunk)
                loss = loss_fn(out, net)
            return loss, tape.gradient(loss, net.trainable_variables)

        l_w, g_w = loss_and_grads(None)
        l_c, g_c = loss_and_grads(16)
        assert float(l_c) == pytest.approx(float(l_w), rel=RTOL32, abs=ATOL32)
        for a, b in zip(g_c, g_w):
            assert a is not None
            np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-4, atol=1e-5)

    def test_checkpointed_gradient_with_local_error(self):
        """Heun (local_err > 0) + chunk_length checkpointed: градиент
        конечен и == целый прогон."""
        from neuraltide.integrators import HeunIntegrator

        pop = WilsonCowan(
            dt=DT,
            params={
                "tau": [8.0],
                "a": 1.0,
                "theta": 0.0,
                "w": 0.2,
                "I_ext": [0.8],
                "max_rate": 100.0,
            },
            name="exc",
        )
        graph = NetworkGraph(dt=DT)
        graph.add_population("exc", pop)
        net = NetworkRNN(graph, HeunIntegrator())
        t_seq = _t_seq(T)

        def grads(chunk):
            with tf.GradientTape() as tape:
                out = net(t_seq, chunk_length=chunk)
                loss = tf.reduce_mean(out.firing_rates["exc"])
            return [g.numpy() for g in tape.gradient(loss, net.trainable_variables)]

        g_w = grads(None)
        g_c = grads(16)
        for a, b in zip(g_c, g_w):
            assert np.all(np.isfinite(a))
            np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)

    def test_brunel_fp_chunked_gradient(self):
        """BrunelFokkerPlanck + chunk_length: градиент (по trainable
        mu_0) == целый прогон."""
        pop = BrunelFokkerPlanck(
            dt=DT,
            params={
                "tau_m": 20.0,
                "v_theta": 20.0,
                "v_reset": 10.0,
                "tau_rp": 2.0,
                "mu_0": {"value": 20.0, "trainable": True, "min": 0.0, "max": 40.0},
                "sigma_ext": 2.0,
            },
            grid_size=100,
            v_min=-80.0,
        )
        graph = NetworkGraph(dt=DT)
        with pytest.warns(UserWarning, match="has no incoming synapses"):
            graph.add_population("brunel", pop)
            net = NetworkRNN(graph, EulerIntegrator())
        t_seq = _t_seq(30)

        def grads(chunk):
            with tf.GradientTape() as tape:
                out = net(t_seq, chunk_length=chunk)
                loss = tf.reduce_mean(out.firing_rates["brunel"])
            return [g.numpy() for g in tape.gradient(loss, net.trainable_variables)]

        g_w = grads(None)
        g_c = grads(16)
        for a, b in zip(g_c, g_w):
            assert np.all(np.isfinite(a))
            np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)
