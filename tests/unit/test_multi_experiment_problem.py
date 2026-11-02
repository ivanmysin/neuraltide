"""
Тесты P1: MultiExperimentProblem — несколько экспериментов над одной
сетью и одним ParameterSpace.

Покрываются: потеря — сумма потерь экспериментов; градиент (сумма
вкладов в одной GradientTape) равен FD-градиенту суммарной потери
(float64); residual — конкатенация невязок с reconstruction
0.5·||r||² == loss; компоненты с префиксом '{exp}/'; held-out не входят
в evaluate и доступны только через validate; simulate(phi, experiment);
счётчики n_forward/n_gradient; тот же duck-type API (size/pack/unpack/
bounds_phi).
"""

import numpy as np
import pytest
import tensorflow as tf

import neuraltide.config
from neuraltide.optimization import (
    CalibrationProblem,
    Experiment,
    MultiExperimentProblem,
    ObservationSpec,
    ResidualLoss,
)


def _loss(target, sigma=2.0):
    return ResidualLoss(
        [ObservationSpec(name="obs", population="exc", target=target, sigma=sigma)]
    )


@pytest.fixture
def multi(l0_factory):
    """Два обучающих эксперимента + один held-out, float64.

    Вход при e_r=0 не влияет на динамику L0-сети, поэтому протоколы
    различаются начальным состоянием (initial_state эксперимента).
    """
    neuraltide.config.set_dtype(tf.float64)
    network, space, t_seq, inputs1, _ = l0_factory()
    inputs_held = inputs1 * 0.5

    pop0, syn0 = network.get_initial_state()
    state2 = (
        [tf.constant([[0.7]], dtype=tf.float64)] + list(pop0[1:]),
        list(syn0),
    )
    state_held = (
        [tf.constant([[0.4]], dtype=tf.float64)] + list(pop0[1:]),
        list(syn0),
    )

    target1 = network(t_seq, inputs=inputs1).firing_rates["exc"]
    target2 = network(t_seq, inputs=inputs1, initial_state=state2).firing_rates["exc"]
    target_held = network(
        t_seq, inputs=inputs_held, initial_state=state_held
    ).firing_rates["exc"]

    exp1 = Experiment(
        name="exp1", t_sequence=t_seq, inputs=inputs1, objective=_loss(target1)
    )
    exp2 = Experiment(
        name="exp2",
        t_sequence=t_seq,
        inputs=inputs1,
        objective=_loss(target2),
        initial_state=state2,
    )
    held = Experiment(
        name="held",
        t_sequence=t_seq,
        inputs=inputs_held,
        objective=_loss(target_held),
        initial_state=state_held,
    )
    problem = MultiExperimentProblem(network, space, [exp1, exp2], held_out=[held])

    singles = {
        "exp1": CalibrationProblem(
            network, space, t_seq, inputs=inputs1, objective=exp1.objective
        ),
        "exp2": CalibrationProblem(
            network,
            space,
            t_seq,
            inputs=inputs1,
            objective=exp2.objective,
            initial_state=state2,
        ),
        "held": CalibrationProblem(
            network,
            space,
            t_seq,
            inputs=inputs_held,
            objective=held.objective,
            initial_state=state_held,
        ),
    }
    return problem, singles, (exp1, exp2, held)


class TestDuckApi:
    def test_size_pack_unpack_bounds(self, multi):
        problem, singles, _ = multi
        assert problem.size == problem.parameter_space.size
        phi0 = problem.pack()
        assert phi0.shape == (problem.size,)
        theta = problem.unpack(phi0)
        assert set(theta) == set(problem.parameter_space.entries)
        lower, upper = problem.bounds_phi()
        assert lower.shape == upper.shape == (problem.size,)
        assert lower[0] == pytest.approx(1.0)  # tau: min
        assert lower[2] == pytest.approx(0.001)  # gsyn_max: min
        assert np.isneginf(lower[1])  # I_ext: без границ
        assert np.isposinf(upper[1])

    def test_training_experiment_requires_objective(self, multi):
        problem, _, experiments = multi
        exp1 = experiments[0]
        bad = Experiment(name="bad", t_sequence=exp1.t_sequence, inputs=exp1.inputs)
        with pytest.raises(ValueError, match="objective"):
            MultiExperimentProblem(
                problem.network, problem.parameter_space, [exp1, bad]
            )


class TestEvaluate:
    def test_evaluate_is_sum_of_experiments(self, multi):
        problem, singles, _ = multi
        phi0 = np.asarray(problem.pack(), dtype=np.float64) + 0.1
        ev = problem.evaluate(phi0)
        ref = singles["exp1"].evaluate(phi0).loss + singles["exp2"].evaluate(phi0).loss
        assert ev.loss == pytest.approx(ref, rel=1e-10)

    def test_components_prefixed_by_experiment(self, multi):
        problem, _, _ = multi
        phi0 = np.asarray(problem.pack(), dtype=np.float64)
        ev = problem.evaluate(phi0)
        assert set(ev.components) == {"exp1/obs", "exp2/obs"}
        assert ev.components["exp1/obs"] == pytest.approx(
            ev.components["exp2/obs"], rel=1e-6
        )

    def test_held_out_not_in_evaluate(self, multi):
        problem, _, experiments = multi
        exp1, exp2, held = experiments
        phi0 = np.asarray(problem.pack(), dtype=np.float64) + 0.1
        ev = problem.evaluate(phi0)
        # задача без held-out даёт ту же evaluate-потерю
        problem_no_held = MultiExperimentProblem(
            problem.network, problem.parameter_space, [exp1, exp2]
        )
        assert problem_no_held.evaluate(phi0).loss == pytest.approx(ev.loss, rel=1e-10)
        # изменение held-out таргета не влияет на evaluate
        held_mod = Experiment(
            name="held",
            t_sequence=held.t_sequence,
            inputs=held.inputs,
            objective=_loss(held.objective.observations[0].target * 3.0),
        )
        problem_mod = MultiExperimentProblem(
            problem.network, problem.parameter_space, [exp1, exp2], held_out=[held_mod]
        )
        assert problem_mod.evaluate(phi0).loss == pytest.approx(ev.loss, rel=1e-10)

    def test_validate_returns_held_out_only(self, multi):
        problem, singles, _ = multi
        phi0 = np.asarray(problem.pack(), dtype=np.float64) + 0.1
        val = problem.validate(phi0)
        assert set(val) == {"held"}
        assert val["held"] == pytest.approx(singles["held"].evaluate(phi0).loss)


class TestGradient:
    def test_gradient_matches_fd_sum(self, multi):
        problem, _, _ = multi
        phi0 = np.asarray(problem.pack(), dtype=np.float64) + 0.1
        ev, grad = problem.value_and_gradient(phi0)
        h = 1e-5
        fd = np.zeros_like(phi0)
        for i in range(phi0.shape[0]):
            step = np.zeros_like(phi0)
            step[i] = h
            fp = problem.evaluate(phi0 + step).loss
            fm = problem.evaluate(phi0 - step).loss
            fd[i] = (fp - fm) / (2.0 * h)
        np.testing.assert_allclose(np.asarray(grad), fd, rtol=1e-5, atol=1e-8)
        # ненулевой градиент хотя бы по одной координате
        assert np.any(np.abs(np.asarray(grad)) > 0.0)

    def test_gradient_equals_sum_of_single_gradients(self, multi):
        problem, singles, _ = multi
        phi0 = np.asarray(problem.pack(), dtype=np.float64) + 0.1
        _, grad = problem.value_and_gradient(phi0)
        _, g1 = singles["exp1"].value_and_gradient(phi0)
        _, g2 = singles["exp2"].value_and_gradient(phi0)
        np.testing.assert_allclose(
            np.asarray(grad), np.asarray(g1) + np.asarray(g2), rtol=1e-8, atol=1e-10
        )


class TestResidual:
    def test_residual_is_concatenation(self, multi):
        problem, singles, _ = multi
        phi0 = np.asarray(problem.pack(), dtype=np.float64) + 0.1
        r = problem.residual(phi0)
        r1 = singles["exp1"].residual(phi0)
        r2 = singles["exp2"].residual(phi0)
        np.testing.assert_allclose(r, np.concatenate([r1, r2]), rtol=1e-10)

    def test_residual_reconstructs_loss(self, multi):
        problem, _, _ = multi
        phi0 = np.asarray(problem.pack(), dtype=np.float64) + 0.1
        r = problem.residual(phi0)
        ev = problem.evaluate(phi0)
        assert 0.5 * float(r @ r) == pytest.approx(ev.loss, rel=1e-10)


class TestSimulateAndCounters:
    def test_simulate_selects_experiment(self, multi):
        problem, _, experiments = multi
        phi0 = np.asarray(problem.pack(), dtype=np.float64)
        out_default = problem.simulate(phi0)
        out_exp2 = problem.simulate(phi0, experiment="exp2")
        assert set(out_default.firing_rates) == {"exc"}
        assert out_exp2.firing_rates["exc"].shape == (
            out_default.firing_rates["exc"].shape
        )
        # exp2 отличается начальным состоянием → другая траектория
        assert not np.allclose(
            out_default.firing_rates["exc"], out_exp2.firing_rates["exc"]
        )
        with pytest.raises(ValueError, match="unknown experiment"):
            problem.simulate(phi0, experiment="nope")

    def test_counters_per_experiment(self, multi):
        problem, _, _ = multi
        problem.evaluate()
        assert problem.n_forward == 2
        assert problem.n_gradient == 0
        problem.value_and_gradient()
        assert problem.n_forward == 4
        assert problem.n_gradient == 1
        problem.residual()
        assert problem.n_forward == 6
        problem.validate()
        assert problem.n_forward == 7  # один held-out прогон
        assert problem.wall_time > 0.0
