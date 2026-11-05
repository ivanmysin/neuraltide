"""
Тесты network-wide интегрирования (integration_mode='network', этап после В6).

Покрываются:
- порядок сходимости RK4 по dt на связанной E–E сети Уилсона–Коуэна с
  синапсом Цодыкса–Маркрама (network: порядок > 3.5; контроль split:
  порядок ≈ 2 — зафиксированное различие режимов);
- устойчивость ROS2 network-wide на жёсткой связанной 2-популяционной
  задаче, где RK4 network-wide расходится; согласованность при мелком dt;
- parity/shape Эйлера и Хьюна network-wide (в т.ч. пустые derivatives
  синапсов и неизменность состояния InputPopulation);
- ValueError для AdaptiveRK/IMEX/Exponential в NetworkRNN(network);
- AD-градиент RK4 network-wide против конечных разностей;
- parity BrunelFP: Euler network-wide == Euler split (ν-буфер сохраняется);
- smoke multiple shooting с network-wide Эйлером (loss_ms == loss_single);
- stiffness_report(scope='network'): спектр полного якобиана связанной
  сети больше популяционного и конечен.
"""

import numpy as np
import pytest
import tensorflow as tf

import neuraltide.config
from neuraltide.core.base import PopulationModel
from neuraltide.core.network import NetworkGraph, NetworkRNN, _step_fn
from neuraltide.inputs import ConstantRateGenerator
from neuraltide.integrators import (
    AdaptiveRKIntegrator,
    EulerIntegrator,
    ExponentialEulerIntegrator,
    HeunIntegrator,
    IMEXEulerIntegrator,
    RK4Integrator,
    RosenbrockIntegrator,
    classify_failure,
    stiffness_report,
)
from neuraltide.optimization import (
    CalibrationProblem,
    ObservationSpec,
    ParameterSpace,
    ResidualLoss,
)
from neuraltide.optimization.shooting import MultipleShootingProblem, ShootingConfig
from neuraltide.populations import BrunelFokkerPlanck, InputPopulation, WilsonCowan
from neuraltide.synapses import StaticSynapse, TsodyksMarkramSynapse

DT = 0.5


def _t_seq(n, dt):
    return tf.constant((np.arange(n, dtype=np.float64) * dt)[None, :, None])


# --------------------------------------------------------------------- #
# Фикстуры сетей
# --------------------------------------------------------------------- #


def _ee_tm_network(dt, integrator, mode):
    """
    E–E сеть Уилсона–Коуэна с рекуррентным синапсом Цодыкса–Маркрама
    (локальный аналог test_audit_regressions::_ee_tm_network). Автономная
    (без внешних входов), e_r != 0 — синапс реально влияет на динамику.
    """
    pop = WilsonCowan(
        dt=dt,
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
    syn = TsodyksMarkramSynapse(
        n_pre=1,
        n_post=1,
        dt=dt,
        params={
            "gsyn_max": {"value": [[0.5]], "trainable": True, "min": 0.001, "max": 5.0},
            "tau_f": {"value": [[500.0]], "trainable": False},
            "tau_d": {"value": [[200.0]], "trainable": False},
            "tau_r": {"value": [[50.0]], "trainable": False},
            "Uinc": {"value": [[0.2]], "trainable": False},
            "pconn": [[1.0]],
            "e_r": 1.0,
        },
        name="exc->exc",
    )
    graph = NetworkGraph(dt=dt)
    graph.add_population("exc", pop)
    graph.add_synapse("exc->exc", syn, src="exc", tgt="exc")
    return NetworkRNN(graph, integrator, integration_mode=mode), pop, syn


def _build_l0_like(mode, integrator=None):
    """
    L0-подобная сеть: WC + StaticSynapse от ConstantRate (аналог
    conftest._build_l0 в заданном режиме интегрирования).
    Возвращает (network, space, t_seq, inputs, loss).
    """
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
            "gsyn_max": {
                "value": [[0.05]],
                "trainable": True,
                "min": 0.001,
                "max": 1.0,
            },
            "pconn": [[1.0]],
            "e_r": 0.0,
        },
        name="in->exc",
    )
    gen = ConstantRateGenerator(dt=DT, params={"rate": 10.0})
    graph = NetworkGraph(dt=DT)
    graph.declare_input("in", n_units=1)
    graph.add_population("exc", pop)
    graph.add_synapse("in->exc", syn, src="in", tgt="exc")
    network = NetworkRNN(
        graph,
        integrator if integrator is not None else EulerIntegrator(),
        integration_mode=mode,
    )

    t_seq = _t_seq(60, DT)
    inputs = graph.pack_inputs({"in": gen(t_seq)})
    target = network(t_seq, inputs=inputs).firing_rates["exc"]
    loss = ResidualLoss(
        [ObservationSpec(name="obs", population="exc", target=target, sigma=2.0)]
    )
    space = ParameterSpace.from_network(network)
    return network, space, t_seq, inputs, loss


class StiffCoupledPopulation(PopulationModel):
    """Линейная жёсткая популяция с синаптическим входом: y' = A·y + b·I_syn.

    Быстрая компонента λ = −1000 (нестабильна для явных RK при dt = 0.5),
    медленная λ = −1. Связь через StaticSynapse (pre_rate = state[0]).
    """

    def __init__(self, a_matrix, b_vec, dt, name):
        super().__init__(n_units=1, dt=dt, name=name)
        dtype = neuraltide.config.get_dtype()
        self.a_matrix = tf.constant(a_matrix, dtype=dtype)
        self.b_vec = tf.constant(b_vec, dtype=dtype)[:, tf.newaxis]
        self.state_size = [tf.TensorShape([1, 1]), tf.TensorShape([1, 1])]

    def get_initial_state(self, batch_size=1):
        dtype = neuraltide.config.get_dtype()
        return [tf.ones([1, 1], dtype=dtype), tf.ones([1, 1], dtype=dtype)]

    def derivatives(self, state, total_synaptic_input):
        flat = tf.stack([state[0][0], state[1][0]])
        out = self.a_matrix @ flat + self.b_vec * total_synaptic_input["I_syn"][0, 0]
        return [out[0][tf.newaxis, :], out[1][tf.newaxis, :]]

    def get_firing_rate(self, state):
        return state[0]

    @property
    def parameter_spec(self):
        return {}


STIFF_A = [[-1000.0, 0.0], [0.0, -1.0]]
B_VEC = [0.0, 1.0]
STIFF_T_END = 2.0


def _stiff_coupled_network(dt, integrator, mode):
    """Две жёсткие линейные популяции с перекрёстной статической связью."""
    graph = NetworkGraph(dt=dt)
    graph.add_population("p1", StiffCoupledPopulation(STIFF_A, B_VEC, dt, name="p1"))
    graph.add_population("p2", StiffCoupledPopulation(STIFF_A, B_VEC, dt, name="p2"))
    syn_params = {"gsyn_max": [[1.0]], "pconn": [[1.0]], "e_r": 1000.0}
    graph.add_synapse(
        "p1->p2", StaticSynapse(1, 1, dt, params=syn_params), src="p1", tgt="p2"
    )
    graph.add_synapse(
        "p2->p1", StaticSynapse(1, 1, dt, params=syn_params), src="p2", tgt="p1"
    )
    return NetworkRNN(graph, integrator, integration_mode=mode)


def _final_pop_vector(output):
    return np.concatenate([np.asarray(s).ravel() for s in output.final_state[0]])


# --------------------------------------------------------------------- #
# Порядок RK4 network-wide против split
# --------------------------------------------------------------------- #


class TestRK4NetworkOrder:
    T_END = 20.0

    def _final_E(self, dt, mode):
        network, _, _ = _ee_tm_network(dt, RK4Integrator(), mode=mode)
        n = int(round(self.T_END / dt))
        out = network(_t_seq(n, dt))
        return float(np.asarray(out.final_state[0][0]).ravel()[0])

    def _errors(self, mode):
        ref = self._final_E(0.5 / 64, mode)
        errs = [abs(self._final_E(dt, mode) - ref) for dt in (0.5, 0.25, 0.125)]
        return ref, errs

    def test_rk4_network_wide_order_above_3_5(self):
        """Связанная сеть со STP интегрируется согласованно: порядок 4."""
        neuraltide.config.set_dtype(tf.float64)
        _, errs = self._errors(mode="network")
        assert errs[0] > errs[1] > errs[2] > 0.0, errs
        order = np.log2(errs[1] / errs[2])
        assert order > 3.5, f"network order {order:.2f}, errs {errs}"

    def test_rk4_split_order_is_two_control(self):
        """Контроль: тот же сети в split-режиме падает до порядка ~2."""
        neuraltide.config.set_dtype(tf.float64)
        _, errs = self._errors(mode="split")
        assert errs[1] > 0.0 and errs[2] > 0.0, errs
        order = np.log2(errs[1] / errs[2])
        assert 1.5 < order < 2.6, f"split order {order:.2f}, errs {errs}"


# --------------------------------------------------------------------- #
# ROS2 network-wide на жёсткой связанной задаче
# --------------------------------------------------------------------- #


class TestRosenbrockNetworkStiff:
    def test_rk4_network_diverges_rosenbrock_stable(self):
        """λ·h = −500: RK4 network-wide расходится, ROS2 — устойчив."""
        neuraltide.config.set_dtype(tf.float64)
        n = int(round(STIFF_T_END / DT))
        t_seq = _t_seq(n, DT)

        out_rk = _stiff_coupled_network(DT, RK4Integrator(), "network")(t_seq)
        failure = classify_failure(out_rk.final_state)
        assert failure["status"] in ("nan", "diverged"), failure

        out_ros = _stiff_coupled_network(
            DT, RosenbrockIntegrator(n_substeps=1), "network"
        )(t_seq)
        s_ros = _final_pop_vector(out_ros)
        assert classify_failure(out_ros.final_state)["status"] == "ok"
        # Быстрая компонента подавлена (|x| << 1), медленная ≈ e^{-2}.
        assert np.max(np.abs(s_ros[[0, 2]])) < 1e-2
        np.testing.assert_allclose(s_ros[[1, 3]], np.exp(-STIFF_T_END), rtol=5e-2)

    def test_rosenbrock_network_consistent_with_fine_dt(self):
        """Субшаги согласуются с RK4-референсом на устойчивом шаге."""
        neuraltide.config.set_dtype(tf.float64)
        # Референс: RK4 с мелким шагом (|λ·h| = 2 — на границе устойчивости).
        dt_fine = 0.002
        n_fine = int(round(STIFF_T_END / dt_fine))
        out_ref = _stiff_coupled_network(dt_fine, RK4Integrator(), "network")(
            _t_seq(n_fine, dt_fine)
        )
        s_ref = _final_pop_vector(out_ref)

        n = int(round(STIFF_T_END / DT))
        t_seq = _t_seq(n, DT)
        out_ros = _stiff_coupled_network(
            DT, RosenbrockIntegrator(n_substeps=8), "network"
        )(t_seq)
        s_ros = _final_pop_vector(out_ros)
        rel = np.linalg.norm(s_ros - s_ref) / np.linalg.norm(s_ref)
        assert rel < 5e-3, f"rel err {rel:.3g}"

    def test_rosenbrock_network_fd_jacobian_matches_jvp(self):
        """method='fd' даёт ту же траекторию, что и 'jvp'."""
        neuraltide.config.set_dtype(tf.float64)
        n = int(round(STIFF_T_END / DT))
        t_seq = _t_seq(n, DT)
        out_fd = _stiff_coupled_network(
            DT, RosenbrockIntegrator(n_substeps=4, jacobian_method="fd"), "network"
        )(t_seq)
        out_jvp = _stiff_coupled_network(
            DT, RosenbrockIntegrator(n_substeps=4), "network"
        )(t_seq)
        s_fd = _final_pop_vector(out_fd)
        s_jvp = _final_pop_vector(out_jvp)
        rel = np.linalg.norm(s_fd - s_jvp) / np.linalg.norm(s_jvp)
        assert rel < 5e-2, f"rel err {rel:.3g}"


# --------------------------------------------------------------------- #
# Euler/Heun: parity, shapes, пустые derivatives
# --------------------------------------------------------------------- #


class TestEulerHeunNetworkWide:
    def test_euler_network_equals_split_with_static_synapse(self):
        """Euler network-wide == Euler split (статический синапс, ZOH вход)."""
        neuraltide.config.set_dtype(tf.float64)
        net_split, _, t_seq, inputs, _ = _build_l0_like(mode="split")
        net_net, _, _, _, _ = _build_l0_like(mode="network")

        out_split = net_split(t_seq, inputs=inputs)
        out_net = net_net(t_seq, inputs=inputs)

        np.testing.assert_allclose(
            out_net.firing_rates["exc"].numpy(),
            out_split.firing_rates["exc"].numpy(),
            rtol=1e-12,
            atol=1e-14,
        )
        for s_n, s_s in zip(out_net.final_state[0], out_split.final_state[0]):
            np.testing.assert_allclose(s_n.numpy(), s_s.numpy(), rtol=1e-12, atol=1e-14)
        # stability_loss в network-режиме не аккумулируется.
        assert float(out_net.stability_loss) == 0.0

    def test_heun_network_smoke_and_shapes(self):
        """Heun network-wide: конечные траектории, полные списки состояний."""
        neuraltide.config.set_dtype(tf.float64)
        network, _, t_seq, inputs, _ = _build_l0_like(
            mode="network", integrator=HeunIntegrator()
        )
        init_pop, init_syn = network.get_initial_state()
        out = network(t_seq, inputs=inputs)

        rates = out.firing_rates["exc"].numpy()
        assert rates.shape == (1, 60, 1)
        assert np.all(np.isfinite(rates))
        # Полные плоские списки той же структуры, что начальное состояние:
        # 1 тензор популяции + 0 тензоров статического синапса.
        assert len(out.final_state[0]) == len(init_pop)
        assert len(out.final_state[1]) == len(init_syn)
        for s, s0 in zip(out.final_state[0], init_pop):
            assert s.shape == s0.shape
            assert np.all(np.isfinite(s.numpy()))
        # stability_loss в network-режиме не аккумулируется.
        assert float(out.stability_loss) == 0.0

    def test_euler_network_keeps_input_population_state(self):
        """InputPopulation даёт пустые derivatives — состояние неизменно."""
        neuraltide.config.set_dtype(tf.float64)
        gen = ConstantRateGenerator(dt=DT, params={"rate": 10.0})
        inp = InputPopulation(gen)
        pop = WilsonCowan(
            dt=DT,
            params={
                "tau": 10.0,
                "a": 1.0,
                "theta": 0.0,
                "w": 0.2,
                "I_ext": 0.8,
                "max_rate": 100.0,
            },
            name="exc",
        )
        syn = StaticSynapse(
            n_pre=1,
            n_post=1,
            dt=DT,
            params={"gsyn_max": 0.05, "pconn": [[1.0]], "e_r": 0.0},
        )
        graph = NetworkGraph(dt=DT)
        graph.add_population("ext", inp)
        graph.add_population("exc", pop)
        graph.add_synapse("ext->exc", syn, src="ext", tgt="exc")

        with pytest.warns(UserWarning, match="no incoming synapses"):
            network = NetworkRNN(graph, EulerIntegrator(), integration_mode="network")

        init_pop, init_syn = network.get_initial_state()
        assert len(init_pop) == 3  # ext: [t, extra] + exc: [E]

        # Полный scan с InputPopulation не поддерживается (как в split) —
        # проверяем eager-вызов _step_fn, см. test_network_wide.py.
        pop_carry, syn_carry, _ = _step_fn(
            (
                tuple(init_pop),
                tuple(init_syn),
                tf.zeros([1], dtype=neuraltide.config.get_dtype()),
            ),
            tf.constant([[0.0]], dtype=neuraltide.config.get_dtype()),
            None,
            graph,
            EulerIntegrator(),
            integration_mode="network",
        )
        # Состояние InputPopulation (2 слота) неизменно, E популяции обновилось.
        for k in range(2):
            np.testing.assert_array_equal(
                np.asarray(pop_carry[k]), np.asarray(init_pop[k])
            )
        e0 = float(init_pop[2][0, 0])
        e1 = float(pop_carry[2][0, 0])
        assert e1 != pytest.approx(e0)


# --------------------------------------------------------------------- #
# ValueError для неподдерживающих интеграторов
# --------------------------------------------------------------------- #


class TestNetworkModeRejectsUnsupportedIntegrators:
    @pytest.mark.parametrize(
        "integrator",
        [
            AdaptiveRKIntegrator(),
            IMEXEulerIntegrator(),
            ExponentialEulerIntegrator(),
        ],
        ids=["adaptive_rk", "imex", "exponential"],
    )
    def test_value_error_in_network_mode(self, integrator):
        neuraltide.config.set_dtype(tf.float64)
        pop = WilsonCowan(
            dt=DT,
            params={
                "tau": 10.0,
                "a": 1.0,
                "theta": 0.0,
                "w": 0.2,
                "I_ext": 0.8,
                "max_rate": 100.0,
            },
            name="exc",
        )
        graph = NetworkGraph(dt=DT)
        graph.add_population("exc", pop)
        with pytest.raises(ValueError, match="supports_network_step"):
            NetworkRNN(graph, integrator, integration_mode="network")
        assert integrator.supports_network_step is False

    def test_base_step_network_advises_alternatives(self):
        """Базовый step_network советует split или поддерживающие интеграторы."""
        integrator = AdaptiveRKIntegrator()
        with pytest.raises(NotImplementedError, match="integration_mode='split'"):
            integrator.step_network(None, [], [], DT)


# --------------------------------------------------------------------- #
# Градиент RK4 network-wide против FD
# --------------------------------------------------------------------- #


class TestRK4NetworkGradient:
    def test_gradient_matches_finite_differences(self):
        neuraltide.config.set_dtype(tf.float64)
        network, pop, _ = _ee_tm_network(DT, RK4Integrator(), mode="network")
        t_seq = _t_seq(20, DT)

        def loss():
            rates = network(t_seq).firing_rates["exc"]
            return tf.reduce_sum(rates**2)

        with tf.GradientTape() as tape:
            value = loss()
        grad_ad = float(tape.gradient(value, pop.tau)[0])
        assert not np.isnan(grad_ad)

        tau0 = float(pop.tau[0])
        estimates = []
        for eps in (3e-5, 1e-4):
            pop.tau.assign([tau0 + eps])
            loss_plus = float(loss())
            pop.tau.assign([tau0 - eps])
            loss_minus = float(loss())
            estimates.append((loss_plus - loss_minus) / (2.0 * eps))
        pop.tau.assign([tau0])

        for est in estimates:
            rel = abs(est - grad_ad) / max(abs(grad_ad), 1e-12)
            assert rel < 1e-3, f"AD {grad_ad:.8g} vs FD {est:.8g} (rel {rel:.3g})"


# --------------------------------------------------------------------- #
# BrunelFP: Euler network-wide == split Euler
# --------------------------------------------------------------------- #


class TestBrunelFPNetworkParity:
    @pytest.mark.filterwarnings("ignore:Population .* has no incoming synapses")
    def test_euler_network_equals_split(self):
        """Паритет Эйлера: derivatives тот же, ν-буфер сдвигается одинаково."""
        neuraltide.config.set_dtype(tf.float64)
        params = {
            "tau_m": {"value": 20.0, "trainable": False},
            "v_theta": {"value": 20.0, "trainable": False},
            "v_reset": {"value": 10.0, "trainable": False},
            "tau_rp": {"value": 2.0, "trainable": False},
            "mu_0": {"value": 18.0, "trainable": False},
            "sigma_ext": {"value": 3.0, "trainable": False},
        }
        pop = BrunelFokkerPlanck(dt=DT, params=params, grid_size=120, v_min=-80.0)
        graph = NetworkGraph(dt=DT)
        graph.add_population("brunel", pop)

        t_seq = _t_seq(40, DT)
        out_split = NetworkRNN(graph, EulerIntegrator())(t_seq)
        out_net = NetworkRNN(graph, EulerIntegrator(), integration_mode="network")(
            t_seq
        )

        np.testing.assert_allclose(
            out_net.firing_rates["brunel"].numpy(),
            out_split.firing_rates["brunel"].numpy(),
            rtol=1e-12,
            atol=1e-15,
        )
        # Состояние [P, p_r, nu_buf] совпадает, включая ν-буфер задержки.
        assert len(out_net.final_state[0]) == 3
        for s_n, s_s in zip(out_net.final_state[0], out_split.final_state[0]):
            np.testing.assert_allclose(s_n.numpy(), s_s.numpy(), rtol=1e-12, atol=1e-15)
        assert out_net.final_state[0][2].shape == (1, int(round(2.0 / DT)))


# --------------------------------------------------------------------- #
# Multiple shooting с network-wide Эйлером
# --------------------------------------------------------------------- #


class TestShootingSmokeNetworkWide:
    def test_loss_matches_single_shooting(self):
        """MultipleShootingProblem поверх network-wide Эйлера: loss_ms ==
        loss_single в инициализированной точке (разрывы ≈ 0)."""
        neuraltide.config.set_dtype(tf.float64)
        network, space, t_seq, inputs, loss = _build_l0_like(mode="network")
        problem = CalibrationProblem(
            network, space, t_seq, inputs=inputs, objective=loss
        )
        ms = MultipleShootingProblem(problem, ShootingConfig(3))
        phi_ms = np.asarray(ms.initialize_from_single_shooting(), dtype=np.float64)
        phi_base = phi_ms[: problem.parameter_space.size]

        ev_ms, _ = ms.value_and_gradient(phi_ms)
        ev_base, _ = problem.value_and_gradient(phi_base)
        assert ev_ms.loss == pytest.approx(ev_base.loss, rel=1e-8)
        gaps = ms.continuity_gaps(phi_ms)
        assert gaps["finite"]
        assert gaps["max"] < 1e-6


# --------------------------------------------------------------------- #
# stiffness_report(scope='network'): полный спектр связанной сети
# --------------------------------------------------------------------- #


class TestStiffnessReportNetworkScope:
    def test_network_scope_spectrum_larger_and_finite(self):
        """На связанной TM-сети scope='network' даёт спектр полного
        якобиана ([P, D] по объединённому состоянию популяций и синапсов) —
        больше популяционного блока ([P, n, d] при замороженных синаптических
        входах) — и весь конечен."""
        neuraltide.config.set_dtype(tf.float64)
        network, _, _ = _ee_tm_network(DT, EulerIntegrator(), mode="network")

        rep_pop = stiffness_report(network, t_points=2, scope="populations")
        rep_net = stiffness_report(network, t_points=2, scope="network")

        ev_pop = rep_pop["populations"]["exc"]["eigenvalues"]
        ev_net = rep_net["network"]["eigenvalues"]
        # WC: 1 состояние у 1 юнита -> 1 собственное значение на точку;
        # network: E популяции + (R, U, A) синапса TM -> 4 на точку.
        assert ev_pop.shape == (2, 1, 1)
        assert ev_net.shape == (2, 4)
        assert ev_net.shape[-1] > ev_pop.shape[-2] * ev_pop.shape[-1]
        assert np.all(np.isfinite(ev_pop.real)) and np.all(np.isfinite(ev_pop.imag))
        assert np.all(np.isfinite(ev_net.real)) and np.all(np.isfinite(ev_net.imag))
        # Метаданные отчёта согласованы.
        assert rep_pop["scope"] == "populations"
        assert rep_net["scope"] == "network"
        assert rep_net["t_points"] == 2
        assert rep_net["dt_network"] == DT
        assert "euler_stable" in rep_net
        assert "lambda_max" in rep_net
