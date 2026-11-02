"""
Тесты P1: multiple shooting (ShootingConfig, MultipleShootingProblem).

Покрываются: валидация узлов; эквивалентность одного сегмента базовой
задаче (loss/gradient, в том числе против base problem с x0-записью);
initialize_from_single_shooting даёт нулевые разрывы; градиентное
уменьшение разрывов и finalize (loss_single ≈ loss_ms, consistent);
фиксированный residual layout с тождеством 0.5·||r||² == loss_ms;
градиент свободных shooting-координат согласован с FD (float64);
честный NotImplementedError для BrunelFokkerPlanck; восстановление
свободного прогона через to_single_shooting_problem; include_syn_states
на сети с TsodyksMarkramSynapse; scalar-only objective (evaluate без
residual-интерфейса); общие счётчики и bounds расширенного пространства.
"""

import numpy as np
import pytest
import tensorflow as tf

import neuraltide.config
from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.integrators import EulerIntegrator
from neuraltide.optimization import (
    CalibrationProblem,
    ObservationSpec,
    ParameterSpace,
    ResidualLoss,
)
from neuraltide.optimization.shooting import (
    MultipleShootingProblem,
    ShootingConfig,
    SingleShootingCheck,
)
from neuraltide.populations import BrunelFokkerPlanck, WilsonCowan
from neuraltide.synapses import TsodyksMarkramSynapse


def _build_ms(l0_factory, nodes, continuity_weight=1.0, include_syn_states=False):
    """L0-сеть в float64 + MultipleShootingProblem (base, ms)."""
    neuraltide.config.set_dtype(tf.float64)
    network, space, t_seq, inputs, loss = l0_factory()
    problem = CalibrationProblem(network, space, t_seq, inputs=inputs, objective=loss)
    ms = MultipleShootingProblem(
        problem,
        ShootingConfig(nodes),
        continuity_weight=continuity_weight,
        include_syn_states=include_syn_states,
    )
    return problem, ms


class TestShootingConfig:
    def test_int_nodes_uniform(self):
        assert ShootingConfig(4).resolve(60) == [0, 15, 30, 45, 60]

    def test_single_segment(self):
        assert ShootingConfig(1).resolve(60) == [0, 60]

    def test_explicit_nodes(self):
        assert ShootingConfig([0, 10, 60]).resolve(60) == [0, 10, 60]

    def test_first_node_must_be_zero(self):
        with pytest.raises(ValueError, match="first node"):
            ShootingConfig([1, 60]).resolve(60)

    def test_last_node_must_be_T(self):
        with pytest.raises(ValueError, match="last node"):
            ShootingConfig([0, 30]).resolve(60)

    def test_nodes_must_increase(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            ShootingConfig([0, 30, 30, 60]).resolve(60)
        with pytest.raises(ValueError, match="strictly increasing"):
            ShootingConfig(5).resolve(3)  # пустые сегменты

    def test_nonpositive_segments(self):
        with pytest.raises(ValueError, match="positive integer"):
            ShootingConfig(0).resolve(60)

    def test_too_short_sequence(self):
        with pytest.raises(ValueError, match="at least two"):
            ShootingConfig([0]).resolve(60)


class TestSingleSegmentEquivalence:
    def test_loss_and_gradient_match_base(self, l0_factory):
        neuraltide.config.set_dtype(tf.float64)
        network, space, t_seq, inputs, loss = l0_factory()
        base = CalibrationProblem(network, space, t_seq, inputs=inputs, objective=loss)
        ms = MultipleShootingProblem(base, ShootingConfig(1))
        phi_ms = np.array(ms.initialize_from_single_shooting(), dtype=np.float64)
        phi_base = phi_ms[: base.size]

        ev_ms, grad_ms = ms.value_and_gradient(phi_ms)
        ev_base, grad_base = base.value_and_gradient(phi_base)
        assert ev_ms.loss == pytest.approx(ev_base.loss, rel=1e-10)
        np.testing.assert_allclose(
            np.asarray(grad_ms)[: base.size],
            np.asarray(grad_base),
            rtol=1e-9,
            atol=1e-11,
        )

    def test_matches_base_with_x0_entry(self, l0_factory):
        """Один сегмент == base problem с той же x0-записью (float64)."""
        neuraltide.config.set_dtype(tf.float64)
        network, space, t_seq, inputs, loss = l0_factory()
        base = CalibrationProblem(network, space, t_seq, inputs=inputs, objective=loss)
        ms = MultipleShootingProblem(base, ShootingConfig(1))
        phi_ms = np.array(ms.initialize_from_single_shooting(), dtype=np.float64)

        space2 = ParameterSpace.from_network(network)
        space2.add_initial_state_entries(network)
        base_x0 = CalibrationProblem(
            network, space2, t_seq, inputs=inputs, objective=loss
        )
        values = {
            name: (
                entry.variable.numpy()
                if entry.variable is not None
                else np.zeros(entry.shape, dtype=np.float64)
            )
            for name, entry in space2.entries.items()
        }
        phi2 = np.asarray(space2.pack(values), dtype=np.float64)

        assert ms.size == space2.size
        ev_ms, grad_ms = ms.value_and_gradient(phi_ms)
        ev2, grad2 = base_x0.value_and_gradient(phi2)
        assert ev_ms.loss == pytest.approx(ev2.loss, rel=1e-10)
        np.testing.assert_allclose(
            np.asarray(grad_ms), np.asarray(grad2), rtol=1e-8, atol=1e-10
        )


class TestInitialize:
    def test_gaps_are_zero_after_initialize(self, l0_factory):
        problem, ms = _build_ms(l0_factory, 4)
        phi = np.array(ms.initialize_from_single_shooting(), dtype=np.float64)
        # initialize выполняет K-1 цепочных прогонов
        assert ms.n_forward == 3
        gaps = ms.continuity_gaps(phi)
        assert ms.n_forward == 3 + 4
        assert gaps["max"] == pytest.approx(0.0, abs=1e-9)
        assert gaps["mean"] == pytest.approx(0.0, abs=1e-9)
        assert gaps["finite"]
        # компоненты continuity присутствуют и нулевые
        ev = ms.evaluate(phi)
        assert ev.components["continuity/max"] == pytest.approx(0.0, abs=1e-9)
        assert ev.components["continuity/mean"] == pytest.approx(0.0, abs=1e-9)
        # стыкованная траектория == непрерывный прогон base problem
        assert ev.loss == pytest.approx(
            problem.evaluate(phi[: problem.size]).loss, rel=1e-10
        )
        # счётчики общие: evaluate добавляет K прогонов
        assert ms.n_forward == 3 + 4 + 4 + 1
        assert ms.n_forward == problem.n_forward

    def test_perturbed_states_then_gradient_descent(self, l0_factory):
        problem, ms = _build_ms(l0_factory, [0, 10, 30, 60])
        phi0 = np.array(ms.initialize_from_single_shooting(), dtype=np.float64)
        base_size = problem.size
        perturbed = phi0.copy()
        perturbed[base_size + 1] += 0.4  # начало сегмента 1
        perturbed[base_size + 2] -= 0.3  # начало сегмента 2
        gaps0 = ms.continuity_gaps(perturbed)
        assert gaps0["max"] > 1e-2

        # градиентный спуск по shooting-координатам (base заморожен)
        cur = perturbed.copy()
        lr = 0.2
        for _ in range(400):
            _, grad = ms.value_and_gradient(cur)
            g = np.array(grad, dtype=np.float64)
            g[:base_size] = 0.0
            cur = cur - lr * g
        gaps1 = ms.continuity_gaps(cur)
        assert gaps1["max"] < gaps0["max"]
        assert gaps1["max"] < 1e-3

        check = ms.finalize(cur, gap_tol=1e-3)
        assert isinstance(check, SingleShootingCheck)
        assert check.consistent
        assert check.loss_single == pytest.approx(check.loss_ms, abs=1e-2)

    def test_finalize_consistent_at_initialize(self, l0_factory):
        problem, ms = _build_ms(l0_factory, [0, 20, 60])
        phi = np.array(ms.initialize_from_single_shooting(), dtype=np.float64)
        check = ms.finalize(phi, gap_tol=1e-9)
        assert check.consistent
        assert check.loss_ms == pytest.approx(check.loss_single, rel=1e-8)


class TestResidual:
    @pytest.mark.parametrize("weight", [1.0, 2.0])
    def test_residual_reconstruction(self, l0_factory, weight):
        problem, ms = _build_ms(l0_factory, 3, continuity_weight=weight)
        phi = np.array(ms.initialize_from_single_shooting(), dtype=np.float64)
        # активные разрывы: данные и continuity оба дают вклад
        phi[problem.size + 1] += 0.2
        r = ms.residual(phi)
        ev = ms.evaluate(phi)
        assert 0.5 * float(r @ r) == pytest.approx(ev.loss, rel=1e-10)
        # layout: r_data + (K-1) стыков по одному элементу (L0: 1 pop slot)
        r_base = problem.residual(phi[: problem.size])
        assert r.shape[0] == r_base.shape[0] + (ms.n_segments - 1)

    def test_residual_typeerror_for_scalar_objective(self, l0_factory):
        neuraltide.config.set_dtype(tf.float64)
        network, space, t_seq, inputs, loss = l0_factory()
        target = loss.observations[0].target

        def scalar_objective(predictions, model):
            return tf.reduce_mean((predictions.firing_rates["exc"] - target) ** 2)

        problem = CalibrationProblem(
            network, space, t_seq, inputs=inputs, objective=scalar_objective
        )
        ms = MultipleShootingProblem(problem, ShootingConfig(2))
        phi = np.array(ms.initialize_from_single_shooting(), dtype=np.float64)
        ev = ms.evaluate(phi)
        assert np.isfinite(ev.loss)
        assert ev.status == "no_residual_interface"
        assert ev.residuals.shape == (0,)
        with pytest.raises(TypeError, match="residual"):
            ms.residual(phi)
        # scalar evaluate совпадает с прогоном base problem (один сегмент)
        ms1 = MultipleShootingProblem(problem, ShootingConfig(1))
        phi1 = np.asarray(ms1.initialize_from_single_shooting(), dtype=np.float64)
        assert ms1.evaluate(phi1).loss == pytest.approx(
            problem.evaluate(phi1[: problem.size]).loss, rel=1e-10
        )


class TestGradient:
    def test_gradient_matches_finite_differences(self, l0_factory):
        problem, ms = _build_ms(l0_factory, [0, 12, 30, 60])
        phi = np.array(ms.initialize_from_single_shooting(), dtype=np.float64)
        phi[problem.size + 1] += 0.25  # активный разрыв на стыке 1->2
        _, grad = ms.value_and_gradient(phi)
        h = 1e-5
        fd = np.zeros_like(phi)
        for i in range(phi.shape[0]):
            step = np.zeros_like(phi)
            step[i] = h
            fp = ms.evaluate(phi + step).loss
            fm = ms.evaluate(phi - step).loss
            fd[i] = (fp - fm) / (2.0 * h)
        np.testing.assert_allclose(np.asarray(grad), fd, rtol=1e-5, atol=1e-7)


class TestBrunelRefusal:
    def test_brunel_fp_raises_not_implemented(self):
        neuraltide.config.set_dtype(tf.float64)
        pop = BrunelFokkerPlanck(
            dt=0.5,
            params={
                "tau_m": {"value": 20.0, "trainable": False},
                "v_theta": {"value": 20.0, "trainable": False},
                "v_reset": {"value": 10.0, "trainable": False},
                "tau_rp": {"value": 2.0, "trainable": False},
                "mu_0": {"value": 18.0, "trainable": False},
                "sigma_ext": {"value": 3.0, "trainable": False},
            },
            grid_size=64,
            v_min=-80.0,
            name="brunel",
        )
        graph = NetworkGraph(dt=0.5)
        graph.add_population("brunel", pop)
        with pytest.warns(UserWarning, match="no incoming synapses"):
            network = NetworkRNN(graph, EulerIntegrator())
        t_seq = tf.constant(np.zeros([1, 8, 1], dtype=np.float64))
        space = ParameterSpace.from_network(network)
        problem = CalibrationProblem(network, space, t_seq, objective=None)
        with pytest.raises(NotImplementedError, match="BrunelFokkerPlanck"):
            MultipleShootingProblem(problem, ShootingConfig(2))


class TestToSingleShooting:
    def test_restores_free_run(self, l0_factory):
        problem, ms = _build_ms(l0_factory, 4)
        phi = np.array(ms.initialize_from_single_shooting(), dtype=np.float64)
        ev_ms = ms.evaluate(phi)  # применяет theta_base к модели
        single = ms.to_single_shooting_problem(phi)
        assert isinstance(single, CalibrationProblem)
        assert single.parameter_space is problem.parameter_space
        assert single.t_sequence is problem.t_sequence

        ev_single = single.evaluate(None)
        assert ev_single.loss == pytest.approx(ev_ms.loss, rel=1e-8)
        ev_base = problem.evaluate(phi[: problem.size])
        assert ev_single.loss == pytest.approx(ev_base.loss, rel=1e-8)

        out_single = single.simulate(None)
        out_base = problem.simulate(phi[: problem.size])
        np.testing.assert_allclose(
            np.asarray(out_single.firing_rates["exc"]),
            np.asarray(out_base.firing_rates["exc"]),
            rtol=1e-9,
            atol=1e-11,
        )

    def test_segment_simulation_matches_slice_of_free_run(self, l0_factory):
        problem, ms = _build_ms(l0_factory, [0, 15, 60])
        phi = np.array(ms.initialize_from_single_shooting(), dtype=np.float64)
        full = problem.simulate(phi[: problem.size])
        seg1 = ms.simulate_segment(phi, 1)
        np.testing.assert_allclose(
            np.asarray(seg1.firing_rates["exc"]),
            np.asarray(full.firing_rates["exc"])[:, 15:60],
            rtol=1e-9,
            atol=1e-11,
        )


class TestIncludeSynStates:
    def _build_tm_network(self):
        """E–E сеть с TsodyksMarkramSynapse (syn-состояние [R, U, A])."""
        neuraltide.config.set_dtype(tf.float64)
        pop = WilsonCowan(
            dt=0.5,
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
            dt=0.5,
            params={
                "gsyn_max": {
                    "value": [[0.05]],
                    "trainable": True,
                    "min": 0.001,
                    "max": 1.0,
                },
                "tau_f": {"value": [[500.0]], "trainable": False},
                "tau_d": {"value": [[200.0]], "trainable": False},
                "tau_r": {"value": [[50.0]], "trainable": False},
                "Uinc": {"value": [[0.2]], "trainable": False},
                "pconn": [[1.0]],
                "e_r": 0.0,
            },
            name="exc->exc",
        )
        graph = NetworkGraph(dt=0.5)
        graph.add_population("exc", pop)
        graph.add_synapse("exc->exc", syn, src="exc", tgt="exc")
        network = NetworkRNN(graph, EulerIntegrator())
        t_seq = tf.constant((np.arange(40, dtype=np.float64) * 0.5)[None, :, None])
        target = network(t_seq).firing_rates["exc"]
        loss = ResidualLoss(
            [ObservationSpec(name="obs", population="exc", target=target, sigma=2.0)]
        )
        space = ParameterSpace.from_network(network)
        return network, space, t_seq, loss

    def test_syn_states_in_entries_gaps_and_residual(self):
        network, space, t_seq, loss = self._build_tm_network()
        problem = CalibrationProblem(network, space, t_seq, objective=loss)
        ms_syn = MultipleShootingProblem(
            problem, ShootingConfig(2), include_syn_states=True
        )
        ms_pop = MultipleShootingProblem(problem, ShootingConfig(2))
        # 2 сегмента × (1 pop-слот + 3 syn-слота) = 8 shooting-координат
        assert ms_syn.size == ms_pop.size + 2 * 3
        names = ms_syn.parameter_space.names
        assert "x0s/0/exc/0" in names
        assert "x0s/1/exc->exc/2" in names

        phi = np.asarray(ms_syn.initialize_from_single_shooting(), dtype=np.float64)
        gaps = ms_syn.continuity_gaps(phi)
        assert gaps["max"] == pytest.approx(0.0, abs=1e-9)
        r = ms_syn.residual(phi)
        ev = ms_syn.evaluate(phi)
        assert 0.5 * float(r @ r) == pytest.approx(ev.loss, rel=1e-10)
        check = ms_syn.finalize(phi, gap_tol=1e-8)
        assert check.consistent


class TestDuckAPI:
    def test_size_bounds_unpack_and_shared_counters(self, l0_problem):
        problem = l0_problem
        ms = MultipleShootingProblem(problem, ShootingConfig(3))
        assert ms.size == problem.size + 3  # 3 сегмента × 1 pop-слот
        assert ms.n_segments == 3
        assert ms.nodes == [0, 20, 40, 60]

        lo, hi = ms.bounds_phi()
        lo_b, hi_b = problem.bounds_phi()
        assert lo.shape == hi.shape == (ms.size,)
        np.testing.assert_allclose(lo[: problem.size], lo_b)
        np.testing.assert_allclose(hi[: problem.size], hi_b)
        assert np.all(np.isinf(lo[problem.size :]))
        assert np.all(np.isinf(hi[problem.size :]))

        phi = np.array(ms.initialize_from_single_shooting())
        theta = ms.unpack(phi)
        assert "x0s/0/exc/0" in theta
        assert "x0s/2/exc/0" in theta
        assert "exc/tau" in theta

        with pytest.raises(ValueError, match="elements"):
            ms.pack(np.zeros(ms.size + 1))

        base_forward = problem.n_forward
        ms.evaluate(phi)
        assert problem.n_forward == base_forward + 3
        assert ms.n_forward == problem.n_forward
        assert ms.n_gradient == problem.n_gradient

    def test_non_finite_status(self, l0_factory):
        """Честный status при расходящемся прогоне (огромный разрыв)."""
        neuraltide.config.set_dtype(tf.float64)
        network, space, t_seq, inputs, loss = l0_factory()
        problem = CalibrationProblem(
            network, space, t_seq, inputs=inputs, objective=loss
        )
        ms = MultipleShootingProblem(problem, ShootingConfig(2))
        phi = np.array(ms.initialize_from_single_shooting(), dtype=np.float64)
        phi[problem.size] = 1e9  # сегмент 0 стартует из гигантского E
        ev = ms.evaluate(phi)
        assert ev.status in ("non_finite", "ok")  # WC ограничен, но статус честный
        assert np.isfinite(ev.loss) == (ev.status != "non_finite")
