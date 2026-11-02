"""
Тесты P1: view-хелперы CalibrationProblem, градиент по x0-записям и
FrozenProblem.

Покрываются: time_slice/with_objective/with_inputs/with_initial_state
дают ту же потерю, что и ручная CalibrationProblem на соответствующем
протоколе; счётчики view общие с базовой задачей; градиент по
x0-координатам проходит направленный Taylor test (float64, оба
grad_mode) и согласован с конечными разностями; FrozenProblem: свободные
координаты согласованы с FD, фиксированные исключены и заморожены.
"""

import numpy as np
import pytest
import tensorflow as tf

import neuraltide.config
from neuraltide.optimization import (
    CalibrationProblem,
    Experiment,
    FrozenProblem,
    ObservationSpec,
    ResidualLoss,
    taylor_test,
)


def _vg(problem):
    def vg(phi_np):
        ev, grad = problem.value_and_gradient(phi_np)
        return ev.loss, np.asarray(grad, dtype=np.float64)

    return vg


def _x0_values(space, fill=0.3):
    return {
        name: (
            entry.variable.numpy()
            if entry.variable is not None
            else np.full(entry.shape, fill, dtype=np.float64)
        )
        for name, entry in space.entries.items()
    }


class TestTimeSlice:
    def test_matches_manual_problem_on_slice(self, l0_problem):
        problem = l0_problem
        target = problem.objective.observations[0].target
        start, stop = 10, 50
        sliced_target = target[:, start:stop]
        sliced_loss = ResidualLoss(
            [
                ObservationSpec(
                    name="obs", population="exc", target=sliced_target, sigma=2.0
                )
            ]
        )
        view = problem.time_slice(start, stop).with_objective(sliced_loss)
        manual = CalibrationProblem(
            problem.network,
            problem.parameter_space,
            problem.t_sequence[:, start:stop],
            inputs=problem.inputs[:, start:stop],
            objective=sliced_loss,
        )
        phi0 = np.asarray(problem.pack())
        assert view.evaluate(phi0).loss == pytest.approx(
            manual.evaluate(phi0).loss, rel=1e-6
        )

    def test_slices_inputs_along_time(self, l0_problem):
        problem = l0_problem
        view = problem.time_slice(0, 30)
        assert view.t_sequence.shape[1] == 30
        assert view.inputs.shape[1] == 30
        # протокол базовой задачи не изменился
        assert problem.t_sequence.shape[1] == 60


class TestWithObjectiveAndInputs:
    def test_with_objective_matches_manual(self, l0_problem):
        problem = l0_problem
        target = problem.objective.observations[0].target
        loss2 = ResidualLoss(
            [
                ObservationSpec(
                    name="obs", population="exc", target=target * 1.5, sigma=2.0
                )
            ]
        )
        view = problem.with_objective(loss2)
        manual = CalibrationProblem(
            problem.network,
            problem.parameter_space,
            problem.t_sequence,
            inputs=problem.inputs,
            objective=loss2,
        )
        phi0 = np.asarray(problem.pack())
        assert view.evaluate(phi0).loss == pytest.approx(
            manual.evaluate(phi0).loss, rel=1e-6
        )

    def test_with_inputs_matches_manual(self, l0_problem):
        problem = l0_problem
        inputs2 = problem.inputs * 0.0
        view = problem.with_inputs(inputs2)
        manual = CalibrationProblem(
            problem.network,
            problem.parameter_space,
            problem.t_sequence,
            inputs=inputs2,
            objective=problem.objective,
        )
        phi0 = np.asarray(problem.pack())
        assert view.evaluate(phi0).loss == pytest.approx(
            manual.evaluate(phi0).loss, rel=1e-6
        )

    def test_with_initial_state_matches_manual(self, l0_problem):
        problem = l0_problem
        pop0, syn0 = problem.network.get_initial_state()
        new_pop = [tf.constant([[0.5]], dtype=pop0[0].dtype)] + list(pop0[1:])
        state = (new_pop, syn0)
        view = problem.with_initial_state(state)
        manual = CalibrationProblem(
            problem.network,
            problem.parameter_space,
            problem.t_sequence,
            inputs=problem.inputs,
            objective=problem.objective,
            initial_state=state,
        )
        phi0 = np.asarray(problem.pack())
        assert view.evaluate(phi0).loss == pytest.approx(
            manual.evaluate(phi0).loss, rel=1e-6
        )


class TestValidateHeldOut:
    def test_validate_uses_held_out_experiments(self, l0_factory):
        neuraltide.config.set_dtype(tf.float64)
        network, space, t_seq, inputs, loss = l0_factory()
        inputs_held = inputs * 0.5
        target_held = network(t_seq, inputs=inputs_held).firing_rates["exc"]
        loss_held = ResidualLoss(
            [
                ObservationSpec(
                    name="obs", population="exc", target=target_held, sigma=2.0
                )
            ]
        )
        held = Experiment(
            name="held", t_sequence=t_seq, inputs=inputs_held, objective=loss_held
        )
        problem = CalibrationProblem(
            network, space, t_seq, inputs=inputs, objective=loss, held_out=[held]
        )
        phi0 = np.asarray(problem.pack(), dtype=np.float64) + 0.1
        val = problem.validate(phi0)
        assert set(val) == {"held"}
        manual = CalibrationProblem(
            network, space, t_seq, inputs=inputs_held, objective=loss_held
        )
        assert val["held"] == pytest.approx(manual.evaluate(phi0).loss)
        # validate увеличивает n_forward на число held-out прогонов
        assert problem.n_forward >= 1

    def test_validate_empty_without_held_out(self, l0_problem):
        assert l0_problem.validate() == {}


class TestSharedCounters:
    def test_views_share_counters_with_base(self, l0_problem):
        problem = l0_problem
        # objective под срез времени (таргет срезается вместе с протоколом)
        target = problem.objective.observations[0].target[:, 5:20]
        sliced_loss = ResidualLoss(
            [ObservationSpec(name="obs", population="exc", target=target, sigma=2.0)]
        )
        base_forward = problem.n_forward
        view = problem.time_slice(5, 20).with_objective(sliced_loss)
        view.evaluate()
        assert problem.n_forward == base_forward + 1
        assert view.n_forward == problem.n_forward

        other = view.with_objective(
            ResidualLoss(
                [
                    ObservationSpec(
                        name="obs", population="exc", target=target, sigma=2.0
                    )
                ]
            )
        )
        other.evaluate()
        assert problem.n_forward == base_forward + 2

        base_grad = problem.n_gradient
        view.value_and_gradient()
        assert problem.n_gradient == base_grad + 1
        assert view.n_gradient == problem.n_gradient
        assert problem.wall_time > 0.0

    def test_views_do_not_replace_base_protocol(self, l0_problem):
        problem = l0_problem
        phi0 = np.asarray(problem.pack())
        loss_ref = problem.evaluate(phi0).loss
        problem.with_inputs(problem.inputs * 0.0).evaluate(phi0)
        assert problem.evaluate(phi0).loss == pytest.approx(loss_ref, rel=1e-6)


class TestInitialStateGradient:
    """Градиент по x0-координатам (float64)."""

    def _problem_with_x0(self, l0_factory, grad_mode="assign"):
        neuraltide.config.set_dtype(tf.float64)
        network, space, t_seq, inputs, loss = l0_factory()
        space.add_initial_state_entries(network)
        problem = CalibrationProblem(
            network, space, t_seq, inputs=inputs, objective=loss, grad_mode=grad_mode
        )
        phi0 = np.asarray(space.pack(_x0_values(space)), dtype=np.float64)
        return problem, phi0

    @pytest.mark.parametrize("grad_mode", ["assign", "chain"])
    def test_taylor_test_x0_gradient(self, l0_factory, grad_mode):
        problem, phi0 = self._problem_with_x0(l0_factory, grad_mode)
        report = taylor_test(_vg(problem), phi0)
        assert 0.9 < report["slope_R0_median"] < 1.1
        assert 1.8 < report["slope_R1_median"] < 2.2

    def test_x0_gradient_matches_fd(self, l0_factory):
        problem, phi0 = self._problem_with_x0(l0_factory)
        _, grad = problem.value_and_gradient(phi0)
        h = 1e-5
        fd = np.zeros_like(phi0)
        for i in range(phi0.shape[0]):
            step = np.zeros_like(phi0)
            step[i] = h
            fp = problem.evaluate(phi0 + step).loss
            fm = problem.evaluate(phi0 - step).loss
            fd[i] = (fp - fm) / (2.0 * h)
        # x0-координата вносит ненулевой вклад
        assert abs(fd[-1]) > 0.0
        np.testing.assert_allclose(np.asarray(grad), fd, rtol=1e-5, atol=1e-8)

    def test_both_modes_agree_with_x0(self, l0_factory):
        problem, phi0 = self._problem_with_x0(l0_factory, grad_mode="assign")
        _, g_assign = problem.value_and_gradient(phi0)
        problem.grad_mode = "chain"
        _, g_chain = problem.value_and_gradient(phi0)
        np.testing.assert_allclose(
            np.asarray(g_assign), np.asarray(g_chain), rtol=1e-8, atol=1e-10
        )


class TestFrozenProblem:
    def _build(self, l0_factory):
        neuraltide.config.set_dtype(tf.float64)
        network, space, t_seq, inputs, loss = l0_factory()
        problem = CalibrationProblem(
            network, space, t_seq, inputs=inputs, objective=loss
        )
        return problem

    def test_free_coords_match_fd(self, l0_factory):
        problem = self._build(l0_factory)
        frozen = FrozenProblem(problem, free_names=["exc/tau", "exc/I_ext"])
        assert frozen.size == 2
        assert frozen.free_names == ["exc/tau", "exc/I_ext"]
        phi0 = np.asarray(frozen.pack(), dtype=np.float64) + 0.05
        ev, grad = frozen.value_and_gradient(phi0)
        assert np.asarray(grad).shape == (frozen.size,)
        h = 1e-6
        fd = np.zeros_like(phi0)
        for i in range(phi0.shape[0]):
            step = np.zeros_like(phi0)
            step[i] = h
            fp = frozen.evaluate(phi0 + step).loss
            fm = frozen.evaluate(phi0 - step).loss
            fd[i] = (fp - fm) / (2.0 * h)
        np.testing.assert_allclose(np.asarray(grad), fd, rtol=1e-5, atol=1e-8)

    def test_fixed_coordinates_are_frozen(self, l0_factory):
        problem = self._build(l0_factory)
        frozen = FrozenProblem(
            problem, free_names=["exc/I_ext"], fixed={"exc/tau": 20.0}
        )
        assert frozen.size == 1
        phi0 = np.asarray(frozen.pack())
        loss_ref = frozen.evaluate(phi0).loss
        # ручное изменение замороженной переменной не влияет на frozen
        problem.parameter_space.entries["exc/tau"].variable.assign([33.0])
        assert frozen.evaluate(phi0).loss == pytest.approx(loss_ref, rel=1e-10)
        # reduced-градиент не содержит замороженных координат
        _, grad = frozen.value_and_gradient(phi0)
        assert np.asarray(grad).shape == (1,)
        # unpack возвращает только свободные записи
        assert set(frozen.unpack(phi0)) == {"exc/I_ext"}

    def test_fixed_value_applied_to_evaluation(self, l0_factory):
        problem = self._build(l0_factory)
        space = problem.parameter_space
        frozen = FrozenProblem(problem, fixed={"exc/tau": 12.0})
        phi0 = np.asarray(frozen.pack())
        ev = frozen.evaluate(phi0)
        names = list(space.entries)
        tau_idx = names.index("exc/tau")
        full = np.array(problem.pack(), dtype=np.float64)  # копия (writable)
        full[tau_idx] = 12.0  # tau: identity transform в L0-сети
        assert ev.loss == pytest.approx(problem.evaluate(full).loss, rel=1e-10)
        assert set(frozen.unpack(phi0)) == {n for n in names if n != "exc/tau"}

    def test_frozen_shares_counters_and_bounds_subset(self, l0_factory):
        problem = self._build(l0_factory)
        frozen = FrozenProblem(problem, free_names=["exc/tau"])
        base_forward = problem.n_forward
        frozen.evaluate()
        assert problem.n_forward == base_forward + 1
        assert frozen.n_forward == problem.n_forward
        lo, hi = frozen.bounds_phi()
        assert lo.shape == hi.shape == (1,)
        assert lo[0] == pytest.approx(1.0)  # физический min tau (identity)
        assert hi[0] == pytest.approx(50.0)

    def test_unknown_names_raise(self, l0_factory):
        problem = self._build(l0_factory)
        with pytest.raises(ValueError, match="unknown free"):
            FrozenProblem(problem, free_names=["nope/param"])
        with pytest.raises(ValueError, match="unknown fixed"):
            FrozenProblem(problem, fixed={"nope/param": 1.0})
        with pytest.raises(ValueError, match="both free and fixed"):
            FrozenProblem(problem, free_names=["exc/tau"], fixed={"exc/tau": 8.0})
