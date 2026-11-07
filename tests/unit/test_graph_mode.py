"""Граф-режим NetworkRNN (этап 1a плана context/gpu_batching_plan.md).

Закреплённые input_signature скана (self._scan_compiled): отсутствие
ретрейсинга при смене T / ранга t_sequence / initial_state=None-vs-заданном,
эквивалентность значений и градиентов eager↔graph (включая честный eager-скан
_сверх_ GradientTape), jit_compile (XLA) — прямой прогон для всех встроенных
интеграторов и train step (кроме AdaptiveRK/Exponential: их градиент не
компилируется под XLA), float64.

Эталоном eager служит прямой вызов _scan_forward_impl (тело скана без
tf.function): исполняется как обычный Python-код. Публичный-внутренний
вход _scan_forward_states делегирует в _scan_compiled и используется
AdjointSolver.
"""

import numpy as np
import pytest
import tensorflow as tf

import neuraltide
import neuraltide.config
from neuraltide.core.network import NetworkGraph, NetworkOutput, NetworkRNN
from neuraltide.integrators import (
    AdaptiveRKIntegrator,
    EulerIntegrator,
    ExponentialEulerIntegrator,
    HeunIntegrator,
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
from neuraltide.synapses import StaticSynapse, TsodyksMarkramSynapse
from neuraltide.training import CompositeLoss, MSELoss, Trainer
from neuraltide.training.readouts import BandpassReadout

DT = 0.5

# Допуски graph-vs-eager (калиброваны на CPU; фактическое расхождение
# float32 ~1e-5 на траекториях T≤100, float64 ~1e-14).
RTOL32, ATOL32 = 1e-5, 1e-5
RTOL64, ATOL64 = 1e-9, 1e-11

# Интеграторы, чей ГРАДИЕНТ (train step) не компилируется под XLA на
# TF 2.21 (CPU): InvalidArgumentError «XLA compilation requires a fixed
# tensor list size» (AdaptiveRK — вложенный while_loop; ExponentialEuler —
# tf.linalg.expm) либо «BroadcastGradientArgs must be a compile-time
# constant» (Rosenbrock split — динамические батч-оси скана после этапа 2;
# на статических формах этапа 1 этот комбо компилировался). Прямой прогон
# под XLA у всех интеграторов работает; Rosenbrock network-wide
# (step_network) с градиентом под XLA не проверен — при batch>1 split-путь
# отклоняется guard'ом.
KNOWN_XLA_GRADIENT_LIMITATIONS = {
    "adaptiverk": (
        "вложенный tf.while_loop адаптивного шага: обратный проход "
        "требует tensor list фиксированного размера"
    ),
    "exponential": (
        "tf.linalg.expm в шаге: обратный проход matrix_exponential/while "
        "требует tensor list фиксированного размера"
    ),
    "rosenbrock": (
        "split-шаг population: динамические батч-оси скана дают "
        "BroadcastTo с неизвестной формой (BroadcastGradientArgs под XLA); "
        "используйте Trainer(jit_compile=False) или integration_mode="
        "'network'"
    ),
}


def _t_seq(T: int, dt: float = DT) -> tf.Tensor:
    dtype = neuraltide.config.get_dtype().as_numpy_dtype()
    values = np.arange(T, dtype=dtype) * dt
    return tf.constant(values[None, :, None])


def _zero_inputs(T: int) -> tf.Tensor:
    return tf.zeros([1, T, 0], dtype=neuraltide.config.get_dtype())


def _eager_reference(network: NetworkRNN, t_seq, inputs):
    """Прямой eager-прогон тела скана (без tf.function)."""
    init_pop = tuple(network._init_pop_states)
    init_syn = tuple(network._init_syn_states)
    rates, stability, final_pop, final_syn, _, _ = network._scan_forward_impl(
        t_seq, init_pop, init_syn, inputs
    )
    return rates, stability, final_pop, final_syn


def _build_wc(jit_compile: bool = False) -> NetworkRNN:
    """WC(1)+StaticSynapse от ConstantRate (топология L0, своя сборка —
    позволяет задать jit_compile и обучаемые параметры)."""
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
    return NetworkRNN(graph, EulerIntegrator(), jit_compile=jit_compile)


def _wc_data(network: NetworkRNN, T: int):
    t_seq = _t_seq(T)
    gen = ConstantRateGenerator(dt=DT, params={"rate": 10.0})
    inputs = network._graph.pack_inputs({"in": gen(t_seq)})
    return t_seq, inputs


def _build_izh_tm() -> NetworkRNN:
    """IzhikevichMeanField(1)+TsodyksMarkram от VonMises, RK4."""
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


def _build_brunel() -> NetworkRNN:
    """BrunelFokkerPlanck без внешних входов: inputs=None -> ширина 0."""
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
        grid_size=150,
        v_min=-80.0,
    )
    graph = NetworkGraph(dt=DT)
    with pytest.warns(UserWarning, match="has no incoming synapses"):
        graph.add_population("brunel", pop)
        network = NetworkRNN(graph, EulerIntegrator())
    return network


def _assert_graph_matches_eager(ref, out, rtol: float, atol: float):
    (r_rates, r_stab, r_pop, r_syn) = ref
    o_rates, o_stab = out.firing_rates, out.stability_loss
    assert set(o_rates) == set(r_rates)
    for name in r_rates:
        np.testing.assert_allclose(
            o_rates[name].numpy(), r_rates[name].numpy(), rtol=rtol, atol=atol
        )
    np.testing.assert_allclose(float(o_stab), float(r_stab), rtol=rtol, atol=atol)
    o_fp, o_fs = out.final_state
    assert len(o_fp) == len(r_pop) and len(o_fs) == len(r_syn)
    for got, want in zip(list(o_fp) + list(o_fs), list(r_pop) + list(r_syn)):
        np.testing.assert_allclose(got.numpy(), want.numpy(), rtol=rtol, atol=atol)


class TestFixedSignatures:
    def test_no_retrace_fixed_signatures(self, l0_factory):
        network, _, _, _, _ = l0_factory()

        def make_inputs(T):
            return tf.fill([1, T, 1], tf.constant(10.0))

        out1 = network(_t_seq(50), inputs=make_inputs(50))
        # ранг 2, другая длина
        network(_t_seq(80)[:, :, 0], inputs=make_inputs(80))
        # заданное initial_state (продолжение из final_state)
        network(_t_seq(50), inputs=make_inputs(50), initial_state=out1.final_state)
        # ещё две длины, включая повтор
        network(_t_seq(45), inputs=make_inputs(45))
        network(_t_seq(80), inputs=make_inputs(80))

        concrete = network._scan_compiled._list_all_concrete_functions()
        assert len(concrete) == 1

    def test_no_retrace_no_inputs_network(self):
        network = _build_brunel()
        network(_t_seq(50))
        network(_t_seq(70))
        concrete = network._scan_compiled._list_all_concrete_functions()
        assert len(concrete) == 1

    def test_scan_forward_states_direct_call_default_inputs(self):
        # Прямой вызов _scan_forward_states без inputs (прежний контракт
        # с default None, используется в examples/example_mutual_inhibition.py).
        network = _build_brunel()
        t_seq = _t_seq(30)
        init_pop = tuple(network._init_pop_states)
        init_syn = tuple(network._init_syn_states)
        out = network._scan_forward_states(t_seq, init_pop, init_syn)
        out_ref = network._scan_forward_states(
            t_seq, init_pop, init_syn, _zero_inputs(30)
        )

        def flat(result):
            rates, stability, fp, fs, aps, ass_ = result
            return (
                list(rates.values())
                + [stability]
                + list(fp)
                + list(fs)
                + list(aps)
                + list(ass_)
            )

        for a, b in zip(flat(out), flat(out_ref)):
            np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=0, atol=0)

    def test_initial_state_wrong_length_raises(self, l0_factory):
        network, _, t_seq, inputs, _ = l0_factory()
        out = network(t_seq, inputs=inputs)
        with pytest.raises(ValueError, match="initial_state\\[0\\] has 0"):
            network(
                t_seq,
                inputs=inputs,
                initial_state=(out.final_state[0][:-1], out.final_state[1]),
            )
        # Проверка длины syn-состояния: у StaticSynapse состояния нет,
        # поэтому используем сеть с динамическим синапсом (TsodyksMarkram).
        network = _build_izh_tm()
        t_seq = _t_seq(20)
        inputs = network._graph.pack_inputs(
            {
                "theta": VonMisesGenerator(
                    dt=DT,
                    params={"mean_rate": 20.0, "R": 0.5, "freq": 8.0, "phase": 0.0},
                )(t_seq)
            }
        )
        out = network(t_seq, inputs=inputs)
        with pytest.raises(ValueError, match="initial_state\\[1\\] has"):
            network(
                t_seq,
                inputs=inputs,
                initial_state=(out.final_state[0], out.final_state[1][:-1]),
            )


class TestGraphEagerEquivalence:
    @pytest.mark.parametrize("T", [80])
    def test_wilson_cowan_static(self, T):
        network = _build_wc()
        t_seq, inputs = _wc_data(network, T)
        ref = _eager_reference(network, t_seq, inputs)
        out = network(t_seq, inputs=inputs)
        _assert_graph_matches_eager(ref, out, rtol=RTOL32, atol=ATOL32)

    @pytest.mark.parametrize("T", [80])
    def test_izhikevich_tsodyks_markram(self, T):
        network = _build_izh_tm()
        t_seq = _t_seq(T)
        inputs = network._graph.pack_inputs(
            {
                "theta": VonMisesGenerator(
                    dt=DT,
                    params={"mean_rate": 20.0, "R": 0.5, "freq": 8.0, "phase": 0.0},
                )(t_seq)
            }
        )
        ref = _eager_reference(network, t_seq, inputs)
        out = network(t_seq, inputs=inputs)
        _assert_graph_matches_eager(ref, out, rtol=RTOL32, atol=ATOL32)

    @pytest.mark.parametrize("T", [60])
    def test_brunel_fokker_planck_no_inputs(self, T):
        network = _build_brunel()
        t_seq = _t_seq(T)
        ref = _eager_reference(network, t_seq, _zero_inputs(T))
        out = network(t_seq)  # inputs=None -> материализуется ширина 0
        _assert_graph_matches_eager(ref, out, rtol=RTOL32, atol=ATOL32)

    def test_graph_matches_eager_gradients(self, l0_factory):
        """Три ветки одного train step: честный eager-скан
        (_scan_forward_impl + GradientTape, без tf.function внутри),
        Trainer(run_eagerly=True) (eager-лента вокруг уже скомпилированного
        скана) и Trainer(run_eagerly=False) (скомпилированный train step).
        Градиенты сверяются в стартовой точке, траектории — по 3 шагам.
        """
        net_impl, _, t_seq, inputs, _ = l0_factory()
        net_e, _, _, _, _ = l0_factory()
        net_g, _, _, _, _ = l0_factory()
        target_impl = {
            k: v + 0.5 for k, v in net_impl(t_seq, inputs=inputs).firing_rates.items()
        }
        target_e = {
            k: v + 0.5 for k, v in net_e(t_seq, inputs=inputs).firing_rates.items()
        }
        target_g = {
            k: v + 0.5 for k, v in net_g(t_seq, inputs=inputs).firing_rates.items()
        }
        loss_impl = CompositeLoss([(1.0, MSELoss(target_impl))])
        loss_e = CompositeLoss([(1.0, MSELoss(target_e))])
        loss_g = CompositeLoss([(1.0, MSELoss(target_g))])

        trainer_e = Trainer(
            net_e, loss_e, tf.keras.optimizers.SGD(0.01), run_eagerly=True
        )
        trainer_g = Trainer(
            net_g, loss_g, tf.keras.optimizers.SGD(0.01), run_eagerly=False
        )

        # 1) Градиенты в стартовой точке (все ветки видят одинаковые веса).
        init_pop = tuple(net_impl._init_pop_states)
        init_syn = tuple(net_impl._init_syn_states)
        with tf.GradientTape() as tape:
            rates, stability, fp, fs, _, _ = net_impl._scan_forward_impl(
                t_seq, init_pop, init_syn, inputs
            )
            out_impl = NetworkOutput(
                firing_rates=rates,
                stability_loss=stability,
                final_state=(list(fp), list(fs)),
            )
            loss_value_impl = loss_impl(out_impl, net_impl)
        grads_impl = tape.gradient(loss_value_impl, net_impl.trainable_variables)

        step_e = trainer_e.train_step(t_seq, inputs)
        step_g = trainer_g.train_step(t_seq, inputs)

        losses0 = [float(loss_value_impl), float(step_e["loss"]), float(step_g["loss"])]
        np.testing.assert_allclose(losses0[1], losses0[0], rtol=RTOL32, atol=ATOL32)
        np.testing.assert_allclose(losses0[2], losses0[0], rtol=RTOL32, atol=ATOL32)
        for branch_name, grads in (
            ("eager-trainer", step_e["grads"]),
            ("graph-trainer", step_g["grads"]),
        ):
            assert len(grads) == len(grads_impl)
            for g_ref, g in zip(grads_impl, grads):
                assert g is not None, f"{branch_name}: None gradient"
                np.testing.assert_allclose(
                    g.numpy(), g_ref.numpy(), rtol=1e-4, atol=1e-5
                )

        # 2) Траектории обучения eager- vs graph-тренера (3 шага всего).
        losses_e = [float(step_e["loss"])]
        losses_g = [float(step_g["loss"])]
        for _ in range(2):
            losses_e.append(float(trainer_e.train_step(t_seq, inputs)["loss"]))
            losses_g.append(float(trainer_g.train_step(t_seq, inputs)["loss"]))
        np.testing.assert_allclose(losses_g, losses_e, rtol=RTOL32, atol=ATOL32)
        assert len(net_e.trainable_variables) == len(net_g.trainable_variables)
        for v_e, v_g in zip(net_e.trainable_variables, net_g.trainable_variables):
            np.testing.assert_allclose(v_g.numpy(), v_e.numpy(), rtol=1e-4, atol=1e-6)

    def test_graph_matches_eager_values_float64(self, l0_factory):
        neuraltide.config.set_dtype(tf.float64)
        network = _build_wc()
        t_seq, inputs = _wc_data(network, 60)
        ref = _eager_reference(network, t_seq, inputs)
        out = network(t_seq, inputs=inputs)
        _assert_graph_matches_eager(ref, out, rtol=RTOL64, atol=ATOL64)


class TestJitCompile:
    def test_jit_compile_forward_smoke(self):
        network = _build_wc(jit_compile=True)
        assert network.jit_compile is True
        t_seq, inputs = _wc_data(network, 60)
        out = network(t_seq, inputs=inputs)
        rates = out.firing_rates["exc"]
        assert np.all(np.isfinite(rates.numpy()))
        assert len(network._scan_compiled._list_all_concrete_functions()) == 1

    def test_jit_compile_gradient_smoke(self, l0_factory):
        network, _, t_seq, inputs, _ = l0_factory()
        target = {
            k: v + 0.5 for k, v in network(t_seq, inputs=inputs).firing_rates.items()
        }
        loss_fn = CompositeLoss([(1.0, MSELoss(target))])
        trainer = Trainer(
            network, loss_fn, tf.keras.optimizers.SGD(1e-3), jit_compile=True
        )
        result = trainer.train_step(t_seq, inputs)
        assert np.isfinite(float(result["loss"]))
        assert all(
            np.all(np.isfinite(g.numpy())) for g in result["grads"] if g is not None
        )

    @pytest.mark.parametrize(
        "integrator_name",
        ["euler", "heun", "rk4", "imex", "adaptiverk", "rosenbrock", "exponential"],
    )
    def test_jit_compile_integrators(self, integrator_name):
        integrators = {
            "euler": EulerIntegrator(),
            "heun": HeunIntegrator(),
            "rk4": RK4Integrator(),
            "rosenbrock": RosenbrockIntegrator(),
            "imex": IMEXEulerIntegrator(),
            "adaptiverk": AdaptiveRKIntegrator(),
            "exponential": ExponentialEulerIntegrator(),
        }
        # Собираем сеть заново с нужным интегратором.
        pop = WilsonCowan(
            dt=DT,
            params={
                "tau": {"value": [8.0], "trainable": True},
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
        network = NetworkRNN(graph, integrators[integrator_name], jit_compile=True)

        t_seq = _t_seq(20)
        gen = ConstantRateGenerator(dt=DT, params={"rate": 10.0})
        inputs = graph.pack_inputs({"in": gen(t_seq)})

        # Прямой прогон под XLA: все встроенные интеграторы.
        out = network(t_seq, inputs=inputs)
        rates = out.firing_rates["exc"]
        assert np.all(np.isfinite(rates.numpy()))

        # Train step под XLA: см. KNOWN_XLA_GRADIENT_LIMITATIONS.
        target = {"exc": rates + 0.5}
        loss_fn = CompositeLoss([(1.0, MSELoss(target))])
        trainer = Trainer(
            network, loss_fn, tf.keras.optimizers.SGD(1e-3), jit_compile=True
        )
        try:
            result = trainer.train_step(t_seq, inputs)
        except tf.errors.InvalidArgumentError:
            if integrator_name in KNOWN_XLA_GRADIENT_LIMITATIONS:
                pytest.xfail(KNOWN_XLA_GRADIENT_LIMITATIONS[integrator_name])
            raise
        assert all(
            np.all(np.isfinite(g.numpy())) for g in result["grads"] if g is not None
        )

    def test_trainer_flag_validation(self, l0_factory):
        network, _, _, _, loss = l0_factory()
        optimizer = tf.keras.optimizers.SGD(0.01)
        with pytest.raises(ValueError, match="несовместимы"):
            Trainer(
                network,
                loss,
                optimizer,
                run_eagerly=True,
                jit_compile=True,
            )
        with pytest.raises(NotImplementedError, match="adjoint"):
            Trainer(
                network,
                loss,
                optimizer,
                grad_method="adjoint",
                jit_compile=True,
            )


class TestBandpassReadoutTracing:
    def test_unbuilt_layer_tracing_raises(self):
        readout = BandpassReadout(f_low=5.0, f_high=40.0, dt=DT)
        x = tf.ones([1, 100, 2], dtype=neuraltide.config.get_dtype())

        @tf.function
        def filtered(inp):
            return readout(inp)

        with pytest.raises(RuntimeError, match="scipy.signal.firwin"):
            filtered(x)

    def test_eager_call_then_trace(self):
        readout = BandpassReadout(f_low=5.0, f_high=40.0, dt=DT)
        x = tf.ones([1, 100, 2], dtype=neuraltide.config.get_dtype())
        y_eager = readout(x)
        assert readout.built

        @tf.function
        def filtered(inp):
            return readout(inp)

        y_graph = filtered(x)
        np.testing.assert_allclose(y_graph.numpy(), y_eager.numpy(), rtol=0, atol=0)
        # Построение произошло ровно один раз (ядро не пересоздаётся).
        y_again = readout(x)
        np.testing.assert_allclose(y_again.numpy(), y_eager.numpy(), rtol=0, atol=0)


_GPU_AVAILABLE = bool(tf.config.list_physical_devices("GPU"))


@pytest.mark.skipif(not _GPU_AVAILABLE, reason="no GPU")
class TestGpuSmoke:
    """GPU smoke (пункт приёмки этапа 1 плана): прогон сети, train step и
    compiled-задача выполняются под /GPU:0; выход сети размещён на GPU.
    Локально (CPU) — skip."""

    def test_forward_train_step_and_compiled_problem_on_gpu(self):
        with tf.device("/GPU:0"):
            network = _build_wc()
            t_seq, inputs = _wc_data(network, 60)
            out = network(t_seq, inputs=inputs)
            rates = out.firing_rates["exc"]
            assert "/GPU:0" in rates.backing_device, rates.backing_device

            target = {k: v + 0.5 for k, v in out.firing_rates.items()}
            loss_fn = CompositeLoss([(1.0, MSELoss(target))])
            trainer = Trainer(network, loss_fn, tf.keras.optimizers.SGD(1e-3))
            result = trainer.train_step(t_seq, inputs)
            assert np.isfinite(float(result["loss"]))

            objective = ResidualLoss(
                [ObservationSpec(name="obs", population="exc", target=rates, sigma=2.0)]
            )
            space = ParameterSpace.from_network(network)
            problem = CalibrationProblem(
                network,
                space,
                t_seq,
                inputs=inputs,
                objective=objective,
                compiled=True,
            )
            evaluation, grad = problem.value_and_gradient(problem.pack())
            assert np.isfinite(evaluation.loss)
            assert np.all(np.isfinite(grad.numpy()))
