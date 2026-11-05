"""
Тесты P1: MultiExperimentProblem — несколько экспериментов над одной
сетью и одним ParameterSpace.

Покрываются: потеря — сумма потерь экспериментов; градиент (сумма
вкладов в одной GradientTape) равен FD-градиенту суммарной потери
(float64); residual — конкатенация невязок с reconstruction
0.5·||r||² == loss; компоненты с префиксом '{exp}/'; held-out не входят
в evaluate и доступны только через validate; simulate(phi, experiment);
счётчики n_forward/n_gradient; тот же duck-type API (size/pack/unpack/
bounds_phi); эксперимент-специфичные x0-записи 'x0/<experiment>/<pop>/
<slot>': приоритет над common, изоляция градиентов, transforms/bounds,
FrozenProblem, held-out, осиротевшие эксперименты, non-trainable записи.
"""

import warnings

import numpy as np
import pytest
import tensorflow as tf

import neuraltide.config
from neuraltide.optimization import (
    CalibrationProblem,
    Experiment,
    FrozenProblem,
    MultiExperimentProblem,
    ObservationSpec,
    OptimizationResult,
    ParameterEntry,
    ResidualLoss,
    ScaledSigmoidTransform,
)
from neuraltide.optimization import problem as problem_module


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

    def test_duplicate_experiment_names_raise(self, multi):
        """Два одинаковых имени делили бы exp-specific x0 молча — отказ."""
        problem, _, experiments = multi
        exp1 = experiments[0]
        with pytest.raises(ValueError, match="duplicate"):
            MultiExperimentProblem(
                problem.network, problem.parameter_space, [exp1, exp1]
            )

    def test_train_held_out_name_collision_raises(self, multi):
        problem, _, experiments = multi
        held = experiments[2]
        with pytest.raises(ValueError, match="held-out"):
            MultiExperimentProblem(
                problem.network, problem.parameter_space, [held], held_out=[held]
            )

    def test_duplicate_held_out_names_raise(self, multi):
        """Дубли внутри held_out затирали бы компоненты validate — отказ."""
        problem, _, experiments = multi
        exp1 = experiments[0]
        held = experiments[2]
        with pytest.raises(ValueError, match="duplicate held-out"):
            MultiExperimentProblem(
                problem.network,
                problem.parameter_space,
                [exp1],
                held_out=[held, held],
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


class TestNonTrainableEntry:
    """C8: ручная non-trainable запись вне φ — duck-type API не ломается."""

    def test_nontrainable_entry_excluded_from_multi_problem(self, l0_factory):
        neuraltide.config.set_dtype(tf.float64)
        network, space, t_seq, inputs, loss = l0_factory()
        var = tf.Variable([0.5], dtype=tf.float64, name="manual_const")
        space.add_entry(
            ParameterEntry(
                full_name="manual/const",
                shape=tf.TensorShape([1]),
                trainable=False,
                variable=var,
                group="population",
            )
        )
        problem = MultiExperimentProblem(
            network,
            space,
            [Experiment(name="exp1", t_sequence=t_seq, inputs=inputs, objective=loss)],
        )
        assert "manual/const" in problem.parameter_space.entries
        assert "manual/const" not in problem.parameter_space.phi_entries
        assert problem.size == 3
        phi0 = np.asarray(problem.pack(), dtype=np.float64)
        assert phi0.shape == (problem.size,)
        theta = problem.unpack(phi0)
        assert "manual/const" not in theta
        ev, grad = problem.value_and_gradient(phi0)
        assert np.asarray(grad).shape == (problem.size,)
        lower, upper = problem.bounds_phi()
        assert lower.shape == upper.shape == (problem.size,)


# ---------------------------------------------------------------------------
# Эксперимент-специфичные x0 (docs/next_steps.md, п. 2)
# ---------------------------------------------------------------------------


def _model_values(space):
    """θ-значения trainable model_var-записей из текущих переменных."""
    return {
        name: entry.variable.numpy()
        for name, entry in space.entries.items()
        if entry.variable is not None
    }


def _x0_phi(space, extra_values):
    """φ из текущих model-параметров + явных значений x0-записей."""
    values = _model_values(space)
    values.update(extra_values)
    return np.asarray(space.pack(values), dtype=np.float64)


def _same_experiments(network, t_seq, inputs, target, names=("exp1", "exp2")):
    return [
        Experiment(name=n, t_sequence=t_seq, inputs=inputs, objective=_loss(target))
        for n in names
    ]


def _run_with_slot(network, t_seq, inputs, slot_value):
    """Ручный прогон сети со slot_value в слоте 0 популяции exc."""
    pop0, syn0 = network.get_initial_state()
    state = (
        [tf.constant([[slot_value]], dtype=pop0[0].dtype)] + list(pop0[1:]),
        list(syn0),
    )
    return network(t_seq, inputs=inputs, initial_state=state)


class TestExperimentSpecificX0:
    """Материализация 'x0/<experiment>/<pop>/<slot>' в MultiExperimentProblem."""

    @staticmethod
    def _build(l0_factory, experiments=("exp2",), common=True):
        neuraltide.config.set_dtype(tf.float64)
        network, space, t_seq, inputs, loss = l0_factory()
        if common:
            space.add_initial_state_entries(network)
        for name in experiments:
            space.add_experiment_initial_state_entries(network, name)
        return network, space, t_seq, inputs, loss

    # -- приоритет: common заполняет слот, exp-запись перезаписывает ----
    def test_common_then_experiment_priority(self, l0_factory):
        network, space, t_seq, inputs, _ = self._build(l0_factory)
        values = _model_values(space)
        values["x0/exc/0"] = [[0.7]]
        values["x0/exp2/exc/0"] = [[0.9]]
        phi0 = np.asarray(space.pack(values), dtype=np.float64)
        target = network(t_seq, inputs=inputs).firing_rates["exc"]
        problem = MultiExperimentProblem(
            network, space, _same_experiments(network, t_seq, inputs, target)
        )

        out1 = problem.simulate(phi0, experiment="exp1")
        out2 = problem.simulate(phi0, experiment="exp2")
        ref_common = _run_with_slot(network, t_seq, inputs, 0.7)
        ref_exp = _run_with_slot(network, t_seq, inputs, 0.9)
        # exp1 — common x0; exp2 — exp-specific x0 (перезаписал слот)
        np.testing.assert_allclose(
            out1.firing_rates["exc"], ref_common.firing_rates["exc"], rtol=1e-10
        )
        np.testing.assert_allclose(
            out2.firing_rates["exc"], ref_exp.firing_rates["exc"], rtol=1e-10
        )
        assert not np.allclose(out1.firing_rates["exc"], out2.firing_rates["exc"])

        # exp2-запись перезаписала слот: common-координата не влияет на exp2
        values_common = dict(values)
        values_common["x0/exc/0"] = [[0.2]]
        phi0_b = np.asarray(space.pack(values_common), dtype=np.float64)
        out2_b = problem.simulate(phi0_b, experiment="exp2")
        np.testing.assert_allclose(
            out2_b.firing_rates["exc"], out2.firing_rates["exc"], rtol=1e-10
        )
        out1_b = problem.simulate(phi0_b, experiment="exp1")
        assert not np.allclose(out1_b.firing_rates["exc"], out1.firing_rates["exc"])

        # FD по обеим x0-координатам согласован с value_and_gradient
        _, grad = problem.value_and_gradient(phi0)
        names = space.trainable_names
        for name in ("x0/exc/0", "x0/exp2/exc/0"):
            i = names.index(name)
            step = np.zeros_like(phi0)
            step[i] = 1e-5
            fd = (
                problem.evaluate(phi0 + step).loss - problem.evaluate(phi0 - step).loss
            ) / (2e-5)
            assert fd == pytest.approx(np.asarray(grad)[i], rel=1e-5, abs=1e-8)

    # -- изоляция градиентов --------------------------------------------
    def test_gradient_isolation_between_experiments(self, l0_factory):
        network, space, t_seq, inputs, _ = self._build(l0_factory)
        values = _model_values(space)
        values["x0/exc/0"] = [[0.6]]
        values["x0/exp2/exc/0"] = [[0.9]]
        phi0 = np.asarray(space.pack(values), dtype=np.float64)
        target = network(t_seq, inputs=inputs).firing_rates["exc"]

        def build(target1):
            exps = [
                Experiment(
                    name="exp1",
                    t_sequence=t_seq,
                    inputs=inputs,
                    objective=_loss(target1),
                ),
                Experiment(
                    name="exp2",
                    t_sequence=t_seq,
                    inputs=inputs,
                    objective=_loss(target),
                ),
            ]
            return MultiExperimentProblem(network, space, exps)

        problem = build(target)
        _, grad = problem.value_and_gradient(phi0)
        grad = np.asarray(grad, dtype=np.float64)
        # изменение таргета exp1 не меняет градиент по exp2-x0,
        # но меняет common-градиент (common участвует в обоих экспериментах)
        problem_b = build(target * 3.0)
        _, grad_b = problem_b.value_and_gradient(phi0)
        grad_b = np.asarray(grad_b, dtype=np.float64)
        names = space.trainable_names
        i_common = names.index("x0/exc/0")
        i_exp2 = names.index("x0/exp2/exc/0")
        assert grad[i_exp2] != 0.0
        assert grad_b[i_exp2] == pytest.approx(grad[i_exp2], rel=1e-10)
        assert not np.isclose(grad_b[i_common], grad[i_common])

        # градиент по exp2-x0 против FD суммарной потери
        step = np.zeros_like(phi0)
        step[i_exp2] = 1e-5
        fd = (
            problem.evaluate(phi0 + step).loss - problem.evaluate(phi0 - step).loss
        ) / (2e-5)
        assert fd == pytest.approx(grad[i_exp2], rel=1e-5, abs=1e-8)

    # -- φ-операции согласованы (pack/unpack/bounds_phi/project) ---------
    def test_phi_ops_consistent_with_experiment_entries(self, l0_factory):
        network, space, t_seq, inputs, _ = self._build(l0_factory)
        entry = space.entries["x0/exp2/exc/0"]
        entry.transform = ScaledSigmoidTransform(min=0.0, max=2.0)
        entry.min, entry.max = 0.3, 1.5
        values = _model_values(space)
        values["x0/exc/0"] = [[0.7]]
        values["x0/exp2/exc/0"] = [[0.9]]
        phi0 = np.asarray(space.pack(values), dtype=np.float64)

        assert space.size == 5  # 3 model + common x0 + exp-specific x0
        theta = space.unpack(phi0)
        assert set(theta) == set(space.phi_entries)
        assert np.asarray(theta["x0/exp2/exc/0"]).item() == pytest.approx(
            0.9, abs=1e-12
        )
        # round-trip pack(unpack(phi)) == phi
        np.testing.assert_allclose(
            np.asarray(space.pack(theta)), phi0, rtol=1e-10, atol=1e-12
        )
        # bounds_phi: границы exp-координаты — inverse физических (0.3, 1.5)
        names = space.trainable_names
        i_exp = names.index("x0/exp2/exc/0")
        problem = MultiExperimentProblem(
            network,
            space,
            _same_experiments(
                network,
                t_seq,
                inputs,
                network(t_seq, inputs=inputs).firing_rates["exc"],
            ),
        )
        assert problem.size == space.size
        lower, upper = problem.bounds_phi()
        assert lower.shape == upper.shape == (problem.size,)
        assert lower[i_exp] == pytest.approx(
            float(entry.transform.inverse(tf.constant(0.3, tf.float64)))
        )
        assert upper[i_exp] == pytest.approx(
            float(entry.transform.inverse(tf.constant(1.5, tf.float64)))
        )
        # project: θ клиппится в физические границы exp-записи
        phi_far = phi0.copy()
        phi_far[i_exp] = 5.0  # forward = 2·sigmoid(5) ≈ 1.987 > 1.5
        phi_proj = np.asarray(space.project(phi_far), dtype=np.float64)
        theta_proj = space.unpack(phi_proj)
        assert np.asarray(theta_proj["x0/exp2/exc/0"]).item() == pytest.approx(
            1.5, abs=1e-12
        )
        # внутренние точки project не двигает
        phi_in = np.asarray(space.project(phi0), dtype=np.float64)
        np.testing.assert_allclose(phi_in, phi0, rtol=1e-10, atol=1e-12)

    # -- transform на exp-записи в value_and_gradient --------------------
    @pytest.mark.parametrize("grad_mode", ["assign", "chain"])
    def test_experiment_x0_transform_matches_fd(self, l0_factory, grad_mode):
        neuraltide.config.set_dtype(tf.float64)
        network, space, t_seq, inputs, _ = l0_factory()
        space.add_experiment_initial_state_entries(
            network,
            "exp2",
            transforms={"x0/exp2/exc/0": ScaledSigmoidTransform(min=0.0, max=2.0)},
        )
        phi0 = _x0_phi(space, {"x0/exp2/exc/0": [[0.9]]})
        target = network(t_seq, inputs=inputs).firing_rates["exc"]
        problem = MultiExperimentProblem(
            network,
            space,
            _same_experiments(network, t_seq, inputs, target),
            grad_mode=grad_mode,
        )
        _, grad = problem.value_and_gradient(phi0)
        i_exp = space.trainable_names.index("x0/exp2/exc/0")
        assert np.asarray(grad)[i_exp] != 0.0
        step = np.zeros_like(phi0)
        step[i_exp] = 1e-5
        fd = (
            problem.evaluate(phi0 + step).loss - problem.evaluate(phi0 - step).loss
        ) / (2e-5)
        assert fd == pytest.approx(np.asarray(grad)[i_exp], rel=1e-5, abs=1e-8)

    # -- FrozenProblem с exp-именами --------------------------------------
    def test_frozen_problem_with_experiment_names(self, l0_factory):
        network, space, t_seq, inputs, _ = self._build(
            l0_factory, experiments=("exp2",), common=False
        )
        values = _model_values(space)
        values["x0/exp2/exc/0"] = [[0.9]]
        phi0 = np.asarray(space.pack(values), dtype=np.float64)
        target = network(t_seq, inputs=inputs).firing_rates["exc"]
        problem = MultiExperimentProblem(
            network, space, _same_experiments(network, t_seq, inputs, target)
        )
        names = space.trainable_names
        i_exp = names.index("x0/exp2/exc/0")

        # 1) exp-запись свободна, model-параметры заморожены на текущих
        frozen = FrozenProblem(problem, free_names=["x0/exp2/exc/0"])
        assert frozen.size == 1
        assert frozen.free_names == ["x0/exp2/exc/0"]
        phi_free = phi0[[i_exp]]
        ev, grad = frozen.value_and_gradient(phi_free)
        assert np.asarray(grad).shape == (1,)
        # identity transform: composed-точка совпадает с phi0
        assert ev.loss == pytest.approx(problem.evaluate(phi0).loss, rel=1e-12)
        fd = (
            frozen.evaluate(phi_free + 1e-5).loss
            - frozen.evaluate(phi_free - 1e-5).loss
        ) / (2e-5)
        assert fd == pytest.approx(np.asarray(grad)[0], rel=1e-5, abs=1e-8)

        # 2) exp-запись fixed на явном значении, model-параметры свободны
        frozen2 = FrozenProblem(problem, fixed={"x0/exp2/exc/0": 0.55})
        assert frozen2.size == 3
        phi_model = np.delete(phi0, i_exp)
        full = phi0.copy()
        full[i_exp] = 0.55
        assert frozen2.evaluate(phi_model).loss == pytest.approx(
            problem.evaluate(full).loss, rel=1e-12
        )
        _, grad2 = frozen2.value_and_gradient(phi_model)
        assert np.asarray(grad2).shape == (3,)
        h = 1e-5
        for i in range(3):
            step = np.zeros_like(phi_model)
            step[i] = h
            fd = (
                frozen2.evaluate(phi_model + step).loss
                - frozen2.evaluate(phi_model - step).loss
            ) / (2 * h)
            assert fd == pytest.approx(np.asarray(grad2)[i], rel=1e-5, abs=1e-8)

        # 3) неизвестное exp-имя — понятная ошибка
        with pytest.raises(ValueError, match="unknown free"):
            FrozenProblem(problem, free_names=["x0/ghost/exc/0"])

    # -- held-out получает exp-specific x0 --------------------------------
    def test_held_out_receives_experiment_specific_x0(self, l0_factory):
        network, space, t_seq, inputs, _ = self._build(
            l0_factory, experiments=("exp2", "held")
        )
        values = _model_values(space)
        values["x0/exc/0"] = [[0.7]]
        values["x0/exp2/exc/0"] = [[0.9]]
        values["x0/held/exc/0"] = [[0.4]]
        phi0 = np.asarray(space.pack(values), dtype=np.float64)
        target = network(t_seq, inputs=inputs).firing_rates["exc"]
        exps = _same_experiments(network, t_seq, inputs, target, names=("exp1", "exp2"))
        held = Experiment(
            name="held", t_sequence=t_seq, inputs=inputs, objective=_loss(target)
        )
        problem = MultiExperimentProblem(network, space, exps, held_out=[held])

        val = problem.validate(phi0)
        # held-out прогон использует held-x0 (0.4), а не common (0.7)
        space.apply_to_model(space.unpack(phi0))
        ref_held = _run_with_slot(network, t_seq, inputs, 0.4)
        ref_common = _run_with_slot(network, t_seq, inputs, 0.7)
        assert val["held"] == pytest.approx(
            float(_loss(target)(ref_held, network)), rel=1e-10
        )
        assert val["held"] != pytest.approx(
            float(_loss(target)(ref_common, network)), rel=1e-10
        )

    # -- осиротевшие exp-записи: UserWarning при каждом конструировании ----
    def test_orphaned_experiment_records_warn(self, l0_factory):
        neuraltide.config.set_dtype(tf.float64)
        network, space, t_seq, inputs, loss = l0_factory()
        space.add_experiment_initial_state_entries(network, "ghost")
        exp1 = Experiment(name="exp1", t_sequence=t_seq, inputs=inputs, objective=loss)
        with pytest.warns(UserWarning, match="ghost"):
            MultiExperimentProblem(network, space, [exp1])
        # совпадающие имена не предупреждают
        space2 = space  # тот же реестр, но эксперимент с именем 'ghost'
        exp_ghost = Experiment(
            name="ghost", t_sequence=t_seq, inputs=inputs, objective=loss
        )
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            MultiExperimentProblem(network, space2, [exp_ghost])
        assert not any("ghost" in str(w.message) for w in record)

    # -- non-trainable exp-запись вне φ не ломает duck-API -----------------
    def test_nontrainable_experiment_entry_outside_phi(self, l0_factory):
        neuraltide.config.set_dtype(tf.float64)
        network, space, t_seq, inputs, loss = l0_factory()
        space.add_experiment_initial_state_entries(network, "exp2")
        space.entries["x0/exp2/exc/0"].trainable = False
        target = network(t_seq, inputs=inputs).firing_rates["exc"]
        problem = MultiExperimentProblem(
            network, space, _same_experiments(network, t_seq, inputs, target)
        )
        assert "x0/exp2/exc/0" in problem.parameter_space.entries
        assert "x0/exp2/exc/0" not in problem.parameter_space.phi_entries
        assert problem.size == 3
        phi0 = np.asarray(problem.pack(), dtype=np.float64)
        theta = problem.unpack(phi0)
        assert "x0/exp2/exc/0" not in theta
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            ev, grad = problem.value_and_gradient(phi0)
        # non-trainable exp-запись не предупреждает и не материализуется
        assert not any("экспериментного контекста" in str(w.message) for w in record)
        assert np.asarray(grad).shape == (3,)
        # exp1 и exp2 — одинаковые траектории (exp-x0 не применён)
        out1 = problem.simulate(phi0, experiment="exp1")
        out2 = problem.simulate(phi0, experiment="exp2")
        np.testing.assert_allclose(
            out1.firing_rates["exc"], out2.firing_rates["exc"], rtol=1e-10
        )
        lower, upper = problem.bounds_phi()
        assert lower.shape == upper.shape == (3,)

    # -- вне экспериментного контекста: пропуск с одноразовым warning ------
    def test_experiment_x0_skipped_outside_experiment_context(self, l0_factory):
        problem_module._X0_OUTSIDE_CONTEXT_WARNED.clear()
        neuraltide.config.set_dtype(tf.float64)
        network, space, t_seq, inputs, loss = l0_factory()
        space.add_experiment_initial_state_entries(network, "exp2")
        phi0 = _x0_phi(space, {"x0/exp2/exc/0": [[0.9]]})
        problem = CalibrationProblem(
            network, space, t_seq, inputs=inputs, objective=loss
        )
        with pytest.warns(UserWarning, match="экспериментного контекста"):
            problem.evaluate(phi0)
        # one-time: повторные прогоны не предупреждают
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            problem.evaluate(phi0)
        assert not any("экспериментного контекста" in str(w.message) for w in record)
        # exp-x0 не применён: траектория совпадает с прогоном по умолчанию
        space.apply_to_model(space.unpack(phi0))
        ref = network(t_seq, inputs=inputs)
        out = problem.simulate(phi0)
        np.testing.assert_allclose(
            out.firing_rates["exc"], ref.firing_rates["exc"], rtol=1e-10
        )

    # -- OptimizationResult roundtrip с exp-ключами ------------------------
    def test_optimization_result_roundtrip_with_experiment_keys(self, tmp_path):
        result = OptimizationResult(
            phi_best=np.array([1.0, 2.0, 3.0, -0.2]),
            theta_best={
                "exc/tau": np.array([8.0]),
                "x0/exc/0": np.array([[0.7]]),
                "x0/exp2/exc/0": np.array([[0.9]]),
            },
            loss_best=0.5,
        )
        restored = OptimizationResult.from_dict(result.to_dict())
        assert set(restored.theta_best) == set(result.theta_best)
        for key in result.theta_best:
            np.testing.assert_allclose(restored.theta_best[key], result.theta_best[key])
        path = tmp_path / "result.json"
        result.save(path)
        loaded = OptimizationResult.load(path)
        assert set(loaded.theta_best) == {
            "exc/tau",
            "x0/exc/0",
            "x0/exp2/exc/0",
        }
        np.testing.assert_allclose(loaded.phi_best, result.phi_best)
        assert loaded.loss_best == pytest.approx(0.5)
