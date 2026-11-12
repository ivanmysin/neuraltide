"""Этап 2: batch B>1 в ядре симуляции NetworkRNN.

Один батч-прогон (состояния [B, n], синапсы [B, n_pre, n_post]) эквивалентен
B индивидуальным прогонам с одинаковым протоколом — по значениям (rates,
final_state) и по градиентам (батч-градиент == среднее индивидуальных).
B=1 — прежнее per-sample поведение. Покрываются: три конфигурации сетей
(WC+Static, Izhikevich+TsodyksMarkram, BrunelFokkerPlanck), user-supplied
initial_state [B, n] с разными значениями по слотам, отсутствие ретрейсинга
при смене B, эквивалентность eager/граф, compiled-ядро задачи калибровки,
AntiPhaseLoss per-batch-item, float64 и guard'ы жёстких интеграторов.
"""

import numpy as np
import pytest
import tensorflow as tf

import neuraltide.config
from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.integrators import (
    EulerIntegrator,
    ExponentialEulerIntegrator,
    IMEXEulerIntegrator,
    RK4Integrator,
    RosenbrockIntegrator,
)
from neuraltide.inputs import ConstantRateGenerator, VonMisesGenerator
from neuraltide.optimization import (
    CalibrationProblem,
    ObservationSpec,
    ParameterSpace,
    ResidualLoss,
)
from neuraltide.populations import (
    BrunelFokkerPlanck,
    IzhikevichMeanField,
    WilsonCowan,
)
from neuraltide.synapses import (
    CompositeSynapse,
    NMDASynapse,
    SecondOrderSynapse,
    StaticSynapse,
    TsodyksMarkramSynapse,
)
from neuraltide.training import AntiPhaseLoss, CompositeLoss, MSELoss, Trainer
from neuraltide.training.adjoint import AdjointSolver

DT = 0.5
T = 60
B = 3

RTOL32, ATOL32 = 1e-5, 1e-4
RTOL64, ATOL64 = 1e-9, 1e-11


def _t_seq(T, dt=DT):
    dtype = neuraltide.config.get_dtype().as_numpy_dtype()
    return tf.constant((np.arange(T, dtype=dtype) * dt)[None, :, None])


def _batch_t(t, b):
    return tf.concat([t] * b, axis=0)


def build_wc():
    """L0-подобная сеть: WilsonCowan(1) + StaticSynapse от ConstantRate."""
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


def wc_inputs(graph, t, rates):
    """Упакованные входы [1, T, 1] со constant rate из списка rates."""
    parts = []
    for r in rates:
        gen = ConstantRateGenerator(dt=DT, params={"rate": float(r)})
        parts.append(graph.pack_inputs({"in": gen(t)}))
    return parts


def build_izh_tm():
    pop = IzhikevichMeanField(
        dt=DT,
        params={
            "tau_pop": 1.0,
            "alpha": 0.5,
            "a": 0.02,
            "b": 0.2,
            "w_jump": 0.1,
            "Delta_I": 0.5,
            "I_ext": 0.1,
        },
    )
    gen = VonMisesGenerator(
        dt=DT, params={"mean_rate": 20.0, "R": 0.5, "freq": 8.0, "phase": 0.0}
    )
    syn = TsodyksMarkramSynapse(
        n_pre=1,
        n_post=1,
        dt=DT,
        params={
            "gsyn_max": [[0.1]],
            "tau_f": 20.0,
            "tau_d": 5.0,
            "tau_r": 200.0,
            "Uinc": 0.2,
            "pconn": [[1.0]],
            "e_r": 0.0,
        },
    )
    graph = NetworkGraph(dt=DT)
    graph.declare_input("theta", n_units=gen.n_units)
    graph.add_population("exc", pop)
    graph.add_synapse("theta->exc", syn, src="theta", tgt="exc")
    return NetworkRNN(graph, RK4Integrator())


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


def _assert_batch_matches_individual(out_b, outs, rtol, atol):
    """out_b: батч-выход [B, ...]; outs: список индивидуальных выходов."""
    assert len(outs) == int(out_b.firing_rates[next(iter(out_b.firing_rates))].shape[0])
    for slot, out_i in enumerate(outs):
        for name in out_i.firing_rates:
            np.testing.assert_allclose(
                out_b.firing_rates[name].numpy()[slot],
                out_i.firing_rates[name].numpy()[0],
                rtol=rtol,
                atol=atol,
            )
        for j, s_i in enumerate(out_i.final_state[0]):
            np.testing.assert_allclose(
                out_b.final_state[0][j].numpy()[slot],
                s_i.numpy()[0],
                rtol=rtol,
                atol=atol,
            )
        for j, s_i in enumerate(out_i.final_state[1]):
            np.testing.assert_allclose(
                out_b.final_state[1][j].numpy()[slot],
                s_i.numpy()[0],
                rtol=rtol,
                atol=atol,
            )


class TestBatchEquivalence:
    def test_wc_static_batch_matches_individual(self):
        t = _t_seq(T)
        net_b = build_wc()
        rates = [10.0, 10.0, 25.0]
        parts = wc_inputs(net_b._graph, t, rates)
        inputs_b = tf.concat(parts, axis=0)
        out_b = net_b(_batch_t(t, B), inputs=inputs_b)
        outs = []
        for r in rates:
            net_i = build_wc()
            ins = net_i._graph.pack_inputs(
                {"in": ConstantRateGenerator(dt=DT, params={"rate": r})(t)}
            )
            outs.append(net_i(t, inputs=ins))
        _assert_batch_matches_individual(out_b, outs, RTOL32, ATOL32)

    def test_izh_tm_batch_matches_individual(self):
        t = _t_seq(T)
        net_b = build_izh_tm()
        gen_a = VonMisesGenerator(
            dt=DT, params={"mean_rate": 20.0, "R": 0.5, "freq": 8.0, "phase": 0.0}
        )
        gen_b = VonMisesGenerator(
            dt=DT, params={"mean_rate": 30.0, "R": 0.5, "freq": 8.0, "phase": 1.5}
        )
        gens = [gen_a, gen_a, gen_b]
        inputs_b = tf.concat(
            [net_b._graph.pack_inputs({"theta": g(t)}) for g in gens], axis=0
        )
        out_b = net_b(_batch_t(t, B), inputs=inputs_b)
        outs = []
        for g in gens:
            net_i = build_izh_tm()
            ins = net_i._graph.pack_inputs({"theta": g(t)})
            outs.append(net_i(t, inputs=ins))
        _assert_batch_matches_individual(out_b, outs, RTOL32, ATOL32)

    def test_brunel_batch_matches_individual(self):
        t = _t_seq(T)
        net_b = build_brunel()
        out_b = net_b(_batch_t(t, B))  # без входов; одинаковые слоты
        net_i = build_brunel()
        out_i = net_i(t)
        for slot in range(B):
            np.testing.assert_allclose(
                out_b.firing_rates["brunel"].numpy()[slot],
                out_i.firing_rates["brunel"].numpy()[0],
                rtol=RTOL32,
                atol=ATOL32,
            )
        for j, s_i in enumerate(out_i.final_state[0]):
            for slot in range(B):
                np.testing.assert_allclose(
                    out_b.final_state[0][j].numpy()[slot],
                    s_i.numpy()[0],
                    rtol=RTOL32,
                    atol=ATOL32,
                )

    def test_network_wide_rk4_batch_matches_individual(self):
        """integration_mode='network' (_JointNetworkRHS) батч-агностичен."""
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
            params={"gsyn_max": [[0.05]], "pconn": [[1.0]], "e_r": 5.0},
        )
        graph = NetworkGraph(dt=DT)
        graph.declare_input("in", n_units=1)
        graph.add_population("exc", pop)
        graph.add_synapse("in->exc", syn, src="in", tgt="exc")
        graph.add_synapse("exc->exc", syn, src="exc", tgt="exc")
        net_b = NetworkRNN(graph, RK4Integrator(), integration_mode="network")
        t = _t_seq(T)
        rates = [10.0, 18.0, 25.0]
        parts = []
        for r in rates:
            gen = ConstantRateGenerator(dt=DT, params={"rate": float(r)})
            parts.append(graph.pack_inputs({"in": gen(t)}))
        inputs_b = tf.concat(parts, axis=0)
        out_b = net_b(_batch_t(t, B), inputs=inputs_b)
        outs = []
        for r in rates:
            g = NetworkGraph(dt=DT)
            g.declare_input("in", n_units=1)
            p2 = WilsonCowan(
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
            s2 = StaticSynapse(
                n_pre=1,
                n_post=1,
                dt=DT,
                params={"gsyn_max": [[0.05]], "pconn": [[1.0]], "e_r": 5.0},
            )
            g.add_population("exc", p2)
            g.add_synapse("in->exc", s2, src="in", tgt="exc")
            g.add_synapse("exc->exc", s2, src="exc", tgt="exc")
            net_i = NetworkRNN(g, RK4Integrator(), integration_mode="network")
            ins = g.pack_inputs(
                {"in": ConstantRateGenerator(dt=DT, params={"rate": float(r)})(t)}
            )
            outs.append(net_i(t, inputs=ins))
        _assert_batch_matches_individual(out_b, outs, RTOL32, ATOL32)

    def test_user_initial_state_per_slot(self):
        """initial_state [B, n] с разными значениями по слотам."""
        t = _t_seq(T)
        net_b = build_wc()
        inputs_b = tf.concat(wc_inputs(net_b._graph, t, [10.0, 10.0, 10.0]), axis=0)
        pop0, syn0 = net_b.get_initial_state()  # per-sample [1, n]
        e0_values = [0.0, 0.5, 0.9]
        init_e0 = tf.constant(
            [[v] for v in e0_values], dtype=neuraltide.config.get_dtype()
        )
        init_pop = [init_e0] + [
            tf.broadcast_to(s, [B] + s.shape[1:].as_list()) for s in pop0[1:]
        ]
        init_syn = [tf.broadcast_to(s, [B] + s.shape[1:].as_list()) for s in syn0]
        out_b = net_b(
            _batch_t(t, B), inputs=inputs_b, initial_state=(init_pop, init_syn)
        )
        outs = []
        for v in e0_values:
            net_i = build_wc()
            ins = net_i._graph.pack_inputs(
                {"in": ConstantRateGenerator(dt=DT, params={"rate": 10.0})(t)}
            )
            p0, s0 = net_i.get_initial_state()
            st = (
                [tf.constant([[v]], dtype=neuraltide.config.get_dtype())] + p0[1:],
                s0,
            )
            outs.append(net_i(t, inputs=ins, initial_state=st))
        _assert_batch_matches_individual(out_b, outs, RTOL32, ATOL32)

    def test_eager_impl_and_graph_both_equivalent(self):
        """Батч-прогон через call (граф) и напрямую через _scan_forward_impl
        (eager) эквивалентны индивидуальным прогонам."""
        t = _t_seq(T)
        net_b = build_wc()
        rates = [10.0, 14.0, 25.0]
        parts = wc_inputs(net_b._graph, t, rates)
        inputs_b = tf.concat(parts, axis=0)
        t_b = _batch_t(t, B)
        out_graph = net_b(t_b, inputs=inputs_b)
        pop0, syn0 = net_b.get_initial_state(batch_size=B)
        rates_e, _, fp, fs, _, _ = net_b._scan_forward_impl(
            t_b, tuple(pop0), tuple(syn0), inputs_b
        )
        out_i_list = []
        for r in rates:
            net_i = build_wc()
            ins = net_i._graph.pack_inputs(
                {"in": ConstantRateGenerator(dt=DT, params={"rate": float(r)})(t)}
            )
            out_i_list.append(net_i(t, inputs=ins).firing_rates["exc"].numpy()[0])
        rb = out_graph.firing_rates["exc"].numpy()
        re = rates_e["exc"].numpy()
        for slot in range(B):
            np.testing.assert_allclose(
                rb[slot], out_i_list[slot], rtol=RTOL32, atol=ATOL32
            )
            np.testing.assert_allclose(
                re[slot], out_i_list[slot], rtol=RTOL32, atol=ATOL32
            )


class TestBatchGradients:
    def test_batch_gradient_is_mean_of_individual(self):
        """Trainer (BPTT) на батче: значения trainable_variables после 2 шагов
        SGD == среднему по индивидуальным прогонам (идентичные копии сети,
        каждая со своим входом)."""
        t = _t_seq(T)
        rates = [10.0, 14.0, 25.0]

        net_b = build_wc()
        inputs_b = tf.concat(wc_inputs(net_b._graph, t, rates), axis=0)
        out_b = net_b(_batch_t(t, B), inputs=inputs_b)
        target_b = {k: v + 0.5 for k, v in out_b.firing_rates.items()}
        loss_b = CompositeLoss([(1.0, MSELoss(target_b))])
        trainer_b = Trainer(
            net_b, loss_b, tf.keras.optimizers.SGD(0.01), run_eagerly=True
        )
        for _ in range(2):
            trainer_b.train_step(_batch_t(t, B), inputs_b)

        singles = []
        for r in rates:
            net_i = build_wc()
            gen = ConstantRateGenerator(dt=DT, params={"rate": float(r)})
            ins = net_i._graph.pack_inputs({"in": gen(t)})
            out_i = net_i(t, inputs=ins)
            tgt = {k: v + 0.5 for k, v in out_i.firing_rates.items()}
            lf = CompositeLoss([(1.0, MSELoss(tgt))])
            tr_i = Trainer(net_i, lf, tf.keras.optimizers.SGD(0.01), run_eagerly=True)
            for _ in range(2):
                tr_i.train_step(t, ins)
            singles.append(net_i)

        for vi, var_b in enumerate(net_b.trainable_variables):
            mean_val = np.mean(
                [n2.trainable_variables[vi].numpy() for n2 in singles], axis=0
            )
            np.testing.assert_allclose(var_b.numpy(), mean_val, rtol=1e-4, atol=1e-5)


class TestBatchNoRetrace:
    def test_batch_change_no_retrace(self):
        t = _t_seq(T)
        net = build_wc()
        inputs1 = tf.concat(wc_inputs(net._graph, t, [10.0]), axis=0)
        net(t, inputs=inputs1)  # B=1
        inputs3 = tf.concat(wc_inputs(net._graph, t, [10.0, 14.0, 25.0]), axis=0)
        net(_batch_t(t, 3), inputs=inputs3)  # B=3
        inputs2 = tf.concat(wc_inputs(net._graph, t, [10.0, 14.0]), axis=0)
        net(_batch_t(t, 2), inputs=inputs2)  # B=2
        net(t, inputs=inputs1)  # B=1 снова
        concrete = net._scan_compiled._list_all_concrete_functions()
        assert len(concrete) == 1


class TestBatchCompiledProblem:
    def test_compiled_problem_batch_protocol(self):
        t = _t_seq(T)
        net_b = build_wc()
        rates = [10.0, 14.0, 25.0]
        inputs_b = tf.concat(wc_inputs(net_b._graph, t, rates), axis=0)
        t_b = _batch_t(t, B)
        out_ref = net_b(t_b, inputs=inputs_b)
        target = out_ref.firing_rates["exc"]
        loss = ResidualLoss(
            [ObservationSpec(name="obs", population="exc", target=target, sigma=2.0)]
        )
        space_b = ParameterSpace.from_network(net_b)
        prob_c = CalibrationProblem(
            net_b, space_b, t_b, inputs=inputs_b, objective=loss, compiled=True
        )
        prob_e = CalibrationProblem(
            net_b, space_b, t_b, inputs=inputs_b, objective=loss, compiled=False
        )
        phi = prob_e.pack()
        ev_c, g_c = prob_c.value_and_gradient(phi)
        ev_e, g_e = prob_e.value_and_gradient(phi)
        assert ev_c.loss == pytest.approx(ev_e.loss, rel=RTOL32, abs=ATOL32)
        np.testing.assert_allclose(g_c.numpy(), g_e.numpy(), rtol=1e-4, atol=1e-5)

        # батч-градиент == среднее индивидуальных (eager-задачи на копиях)
        grads = []
        for r in rates:
            net_i = build_wc()
            gen = ConstantRateGenerator(dt=DT, params={"rate": float(r)})
            ins = net_i._graph.pack_inputs({"in": gen(t)})
            out_i = net_i(t, inputs=ins)
            li = ResidualLoss(
                [
                    ObservationSpec(
                        name="obs",
                        population="exc",
                        target=out_i.firing_rates["exc"],
                        sigma=2.0,
                    )
                ]
            )
            sp = ParameterSpace.from_network(net_i)
            pr = CalibrationProblem(net_i, sp, t, inputs=ins, objective=li)
            _, g_i = pr.value_and_gradient(pr.pack())
            grads.append(g_i.numpy())
        np.testing.assert_allclose(
            g_c.numpy(), np.mean(grads, axis=0), rtol=1e-4, atol=1e-5
        )


class TestAntiPhasePerItem:
    def test_per_item_correlation_and_b1_identity(self):
        """B=2: anti-phase в слоте 0, in-phase в слоте 1 — loss больше, чем
        при двух anti-phase; B=1 тождественна прежней пуловой формуле."""
        dtype = neuraltide.config.get_dtype()
        t = np.linspace(0, 4 * np.pi, 200, dtype=np.float32)
        a = np.sin(t)[None, :, None]
        b_anti = np.sin(t + np.pi)[None, :, None]
        b_in = np.sin(t + 0.1)[None, :, None]

        rates_anti = {
            "pa": tf.constant(np.concatenate([a, a], axis=0), dtype=dtype),
            "pb": tf.constant(np.concatenate([b_anti, b_anti], axis=0), dtype=dtype),
        }
        rates_mix = {
            "pa": tf.constant(np.concatenate([a, a], axis=0), dtype=dtype),
            "pb": tf.constant(np.concatenate([b_anti, b_in], axis=0), dtype=dtype),
        }

        class _Out:
            def __init__(self, fr):
                self.firing_rates = fr
                self.stability_loss = tf.constant(0.0, dtype=dtype)
                self.final_state = ([], [])

        loss_fn = AntiPhaseLoss(
            [("pa", "pb")], target_correlation=-1.0, activity_target=0.0
        )
        l_two_anti = float(loss_fn(_Out(rates_anti), None))
        l_mix = float(loss_fn(_Out(rates_mix), None))
        # в слоте 1 корреляция ≈ +1 → штраф (corr−(−1))² ≈ 4 против ≈ 0
        assert l_mix > l_two_anti + 1.0

        # B=1: совпадает с прежней пуловой формулой (ручной расчёт)
        r_a = tf.constant(a, dtype=dtype)
        r_b = tf.constant(b_anti, dtype=dtype)
        l_b1 = float(
            loss_fn(
                _Out({"pa": r_a, "pb": r_b}),
                None,
            )
        )
        af = tf.reshape(r_a, [-1])
        bf = tf.reshape(r_b, [-1])
        ac = af - tf.reduce_mean(af)
        bc = bf - tf.reduce_mean(bf)
        corr = tf.reduce_mean(ac * bc) / (
            tf.sqrt(tf.reduce_mean(ac**2) + 1e-8)
            * tf.sqrt(tf.reduce_mean(bc**2) + 1e-8)
        )
        l_manual = float((corr - (-1.0)) ** 2)
        assert l_b1 == pytest.approx(l_manual, rel=1e-5, abs=1e-8)


class TestBatchFloat64:
    def test_batch_matches_individual_float64(self):
        neuraltide.config.set_dtype(tf.float64)
        t = _t_seq(T)
        net_b = build_wc()
        rates = [10.0, 14.0, 25.0]
        inputs_b = tf.concat(wc_inputs(net_b._graph, t, rates), axis=0)
        out_b = net_b(_batch_t(t, B), inputs=inputs_b)
        outs = []
        for r in rates:
            net_i = build_wc()
            ins = net_i._graph.pack_inputs(
                {"in": ConstantRateGenerator(dt=DT, params={"rate": float(r)})(t)}
            )
            outs.append(net_i(t, inputs=ins))
        _assert_batch_matches_individual(out_b, outs, RTOL64, ATOL64)


class TestStiffIntegratorGuards:
    @pytest.mark.parametrize(
        "integrator",
        [
            RosenbrockIntegrator(),
            IMEXEulerIntegrator(),
        ],
    )
    def test_population_step_rejects_batch(self, integrator):
        """Split-шаги жёстких интеграторов (per-unit якобиан) при batch>1 —
        понятная ошибка (eager: assert_equal; в графе: InvalidArgumentError
        с тем же текстом)."""
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
        state = [tf.zeros([B, 1], dtype=neuraltide.config.get_dtype())]
        inp = {
            "I_syn": tf.zeros([B, 1], dtype=neuraltide.config.get_dtype()),
            "g_syn": tf.zeros([B, 1], dtype=neuraltide.config.get_dtype()),
        }
        with pytest.raises((ValueError, tf.errors.InvalidArgumentError), match="batch"):
            integrator.step(pop, state, inp)

    @pytest.mark.parametrize(
        "integrator",
        [RosenbrockIntegrator(), ExponentialEulerIntegrator()],
    )
    def test_population_step_guard_via_network(self, integrator):
        """Guard batch=1-only срабатывает и ВНУТРИ графа скана (динамические
        батч-оси; n_units>=2): сеть + batch-протокол -> InvalidArgumentError
        с текстом про batch."""
        pop = WilsonCowan(
            dt=DT,
            params={
                "tau": [8.0, 9.0],
                "a": 1.0,
                "theta": 0.0,
                "w": [0.2, 0.2],
                "I_ext": [0.8, 0.8],
                "max_rate": 100.0,
            },
            name="exc",
        )
        graph = NetworkGraph(dt=DT)
        graph.declare_input("in", n_units=2)
        graph.add_population("exc", pop)
        net = NetworkRNN(graph, integrator)
        t = _t_seq(10)
        gen = ConstantRateGenerator(dt=DT, params={"rate": [10.0, 12.0]})
        inputs = tf.concat([graph.pack_inputs({"in": gen(t)})] * B, axis=0)
        with pytest.raises((ValueError, tf.errors.InvalidArgumentError), match="batch"):
            net(_batch_t(t, B), inputs=inputs)

    def test_rosenbrock_network_step_supports_batch(self):
        """step_network (_JointNetworkRHS + плотный якобиан) батч-агностичен:
        batch-прогон == индивидуальные прогоны (допуски float32: плотный
        solve вносит ошибку порядка 1e-5 даже между идентичными слотами)."""
        t = _t_seq(20)
        rates = [10.0, 18.0, 25.0]

        def _build():
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
                params={"gsyn_max": [[0.05]], "pconn": [[1.0]], "e_r": 5.0},
            )
            graph = NetworkGraph(dt=DT)
            graph.declare_input("in", n_units=1)
            graph.add_population("exc", pop)
            graph.add_synapse("in->exc", syn, src="in", tgt="exc")
            graph.add_synapse("exc->exc", syn, src="exc", tgt="exc")
            return NetworkRNN(graph, RosenbrockIntegrator(), integration_mode="network")

        net_b = _build()
        inputs_b = tf.concat(
            [
                net_b._graph.pack_inputs(
                    {"in": ConstantRateGenerator(dt=DT, params={"rate": float(r)})(t)}
                )
                for r in rates
            ],
            axis=0,
        )
        out_b = net_b(_batch_t(t, B), inputs=inputs_b)
        outs = []
        for r in rates:
            net_i = _build()
            ins = net_i._graph.pack_inputs(
                {"in": ConstantRateGenerator(dt=DT, params={"rate": float(r)})(t)}
            )
            outs.append(net_i(t, inputs=ins))
        _assert_batch_matches_individual(out_b, outs, RTOL32, ATOL32)


class TestBatchValidation:
    def test_initial_state_rank_mismatch_raises(self):
        t = _t_seq(10)
        net = build_wc()
        inputs = tf.concat(wc_inputs(net._graph, t, [10.0] * B), axis=0)
        pop0, syn0 = net.get_initial_state(batch_size=B)
        bad_pop = list(pop0)
        bad_pop[0] = tf.zeros([B], dtype=neuraltide.config.get_dtype())  # rank 1
        with pytest.raises(ValueError, match="rank"):
            net(_batch_t(t, B), inputs=inputs, initial_state=(bad_pop, syn0))

    def test_batch_mismatch_t_vs_inputs_raises(self):
        t = _t_seq(10)
        net = build_wc()
        inputs = tf.concat(wc_inputs(net._graph, t, [10.0] * 2), axis=0)
        with pytest.raises(ValueError, match="batch"):
            net(_batch_t(t, B), inputs=inputs)


class TestBatchCompositeSynapses:
    def _build(self):
        """WC(1) с тремя синапсами от одного входа: NMDA, SecondOrder и
        Composite(TM + Static)."""
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
        nmda = NMDASynapse(
            n_pre=1,
            n_post=1,
            dt=DT,
            params={
                "gsyn_max_nmda": [[0.05]],
                "tau1_nmda": 2.0,
                "tau2_nmda": 20.0,
                "Mgb": 1.0,
                "av_nmda": 0.08,
                "pconn_nmda": [[1.0]],
                "e_r_nmda": 5.0,
                "v_ref": 0.0,
            },
        )
        so = SecondOrderSynapse(
            n_pre=1,
            n_post=1,
            dt=DT,
            params={
                "gsyn_max": [[0.04]],
                "tau1": 1.0,
                "tau2": 10.0,
                "pconn": [[1.0]],
                "e_r": 0.0,
            },
        )
        tm = TsodyksMarkramSynapse(
            n_pre=1,
            n_post=1,
            dt=DT,
            params={
                "gsyn_max": [[0.06]],
                "tau_f": 20.0,
                "tau_d": 5.0,
                "tau_r": 200.0,
                "Uinc": 0.2,
                "pconn": [[1.0]],
                "e_r": 3.0,
            },
        )
        st = StaticSynapse(
            n_pre=1,
            n_post=1,
            dt=DT,
            params={"gsyn_max": [[0.03]], "pconn": [[1.0]], "e_r": -5.0},
        )
        comp = CompositeSynapse(
            n_pre=1, n_post=1, dt=DT, components=[("tm", tm), ("st", st)]
        )
        graph = NetworkGraph(dt=DT)
        graph.declare_input("in", n_units=1)
        graph.add_population("exc", pop)
        graph.add_synapse("nmda->exc", nmda, src="in", tgt="exc")
        graph.add_synapse("so->exc", so, src="in", tgt="exc")
        graph.add_synapse("comp->exc", comp, src="in", tgt="exc")
        return NetworkRNN(graph, RK4Integrator())

    def test_batch_matches_individual(self):
        t = _t_seq(T)
        rates = [10.0, 18.0, 25.0]
        net_b = self._build()
        inputs_b = tf.concat(
            [
                net_b._graph.pack_inputs(
                    {"in": ConstantRateGenerator(dt=DT, params={"rate": float(r)})(t)}
                )
                for r in rates
            ],
            axis=0,
        )
        out_b = net_b(_batch_t(t, B), inputs=inputs_b)
        outs = []
        for r in rates:
            net_i = self._build()
            ins = net_i._graph.pack_inputs(
                {"in": ConstantRateGenerator(dt=DT, params={"rate": float(r)})(t)}
            )
            outs.append(net_i(t, inputs=ins))
        _assert_batch_matches_individual(out_b, outs, RTOL32, ATOL32)


class TestBatchAdjoint:
    def test_adjoint_batch_finite_and_matches_bptt(self):
        """AdjointSolver при B=3: прогон и градиенты конечны, градиенты
        совпадают с BPTT по тому же батч-прогону (та же конвенция потери;
        сверка с индивидуальными — транзитивно через тест BPTT-эквивалентности
        батча)."""
        t = _t_seq(30)
        rates = [10.0, 10.0, 10.0]  # идентичные слоты

        def _build():
            return build_wc()

        net_a = _build()
        inputs_b = tf.concat(
            [
                net_a._graph.pack_inputs(
                    {"in": ConstantRateGenerator(dt=DT, params={"rate": float(r)})(t)}
                )
                for r in rates
            ],
            axis=0,
        )
        t_b = _batch_t(t, B)
        out_a = net_a(t_b, inputs=inputs_b)
        target = {k: v + 0.5 for k, v in out_a.firing_rates.items()}
        loss_fn = CompositeLoss([(1.0, MSELoss(target))])

        solver = AdjointSolver(net_a, net_a._integrator)
        grads_a, variables, output = solver.compute_gradients(
            t_b, inputs_b, target, loss_fn
        )
        assert len(grads_a) == len(variables)
        assert all(g is not None and np.all(np.isfinite(g.numpy())) for g in grads_a)
        assert np.all(np.isfinite(output.firing_rates["exc"].numpy()))

        # BPTT по тому же батч-прогону на идентичной копии сети
        net_b = _build()
        with tf.GradientTape() as tape:
            out_b = net_b(t_b, inputs=inputs_b)
            loss_b = loss_fn(out_b, net_b)
        grads_b = tape.gradient(loss_b, net_b.trainable_variables)
        for g_a, g_b in zip(grads_a, grads_b):
            np.testing.assert_allclose(g_a.numpy(), g_b.numpy(), rtol=1e-3, atol=1e-4)


class TestBatchInitialStateParity:
    """MAJOR-2: единая семантика начального состояния в eager и compiled
    режимах (_broadcast_or_validate_init)."""

    def _problem(self, net, t_b, inputs_b, initial_state, compiled, target_init=None):
        # target строится с валидным начальным состоянием (target_init,
        # по умолчанию — то же, но с per-sample broadcast);
        # initial_state — то, что получает сама задача.
        tgt_init = target_init if target_init is not None else initial_state
        if tgt_init is not None:
            ip, isyn = tgt_init
            b = int(t_b.shape[0])
            ip = tuple(
                tf.broadcast_to(x, [b] + x.shape[1:].as_list())
                if x.shape[0] == 1
                else x
                for x in ip
            )
            isyn = tuple(
                tf.broadcast_to(x, [b] + x.shape[1:].as_list())
                if x.shape[0] == 1
                else x
                for x in isyn
            )
            tgt_init = (ip, isyn)
        out = net(t_b, inputs=inputs_b, initial_state=tgt_init)
        target = out.firing_rates["exc"]
        loss = ResidualLoss(
            [ObservationSpec(name="obs", population="exc", target=target, sigma=2.0)]
        )
        space = ParameterSpace.from_network(net)
        problem = CalibrationProblem(
            net,
            space,
            t_b,
            inputs=inputs_b,
            objective=loss,
            compiled=compiled,
        )
        if initial_state is not None:
            problem.initial_state = initial_state
        return problem

    def test_per_sample_initial_state_broadcasts_in_both_modes(self):
        t = _t_seq(T)
        pop0, syn0 = None, None
        results = {}
        for compiled in (False, True):
            net = build_wc()
            if pop0 is None:
                pop0, syn0 = net.get_initial_state()  # per-sample [1, n]
            inputs_b = tf.concat(wc_inputs(net._graph, t, [10.0, 14.0, 25.0]), axis=0)
            t_b = _batch_t(t, B)
            prob = self._problem(net, t_b, inputs_b, (pop0, syn0), compiled)
            ev, g = prob.value_and_gradient(prob.pack())
            results[compiled] = (ev.loss, g.numpy())
        assert results[False][0] == pytest.approx(
            results[True][0], rel=RTOL32, abs=ATOL32
        )
        np.testing.assert_allclose(
            results[True][1], results[False][1], rtol=1e-4, atol=1e-5
        )

    def test_mismatched_initial_state_batch_raises_in_both_modes(self):
        t = _t_seq(20)
        bad_e0 = tf.zeros([2, 1], dtype=neuraltide.config.get_dtype())
        for compiled in (False, True):
            net = build_wc()
            pop0, syn0 = net.get_initial_state()
            init_pop = [bad_e0] + list(pop0[1:])
            inputs_b = tf.concat(wc_inputs(net._graph, t, [10.0, 14.0, 25.0]), axis=0)
            prob = self._problem(
                net,
                _batch_t(t, B),
                inputs_b,
                (init_pop, syn0),
                compiled,
                target_init=(pop0, syn0),
            )
            with pytest.raises(ValueError, match="batch"):
                prob.value_and_gradient(prob.pack())
