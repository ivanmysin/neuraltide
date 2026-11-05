"""
Тесты P1: continuation-стратегии и scheduler.

dt_continuation/subset_continuation (strategies.py) и ContinuationScheduler
со helpers-расписаниями (schedules.py) проверяются на малой L0-сети
(WilsonCowan + StaticSynapse от ConstantRate, float64, горизонт 30 мс):
цель — прогон при параметрах конструктора, старт — сдвинутый phi0
(как в test_local_optimizers). Teacher forcing: смешивание только во
входах, последний alpha=0, free_run_check на оригинальных входах.
"""

import json

import numpy as np
import pytest
import tensorflow as tf

import neuraltide.config
from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.integrators import EulerIntegrator
from neuraltide.inputs import ConstantRateGenerator
from neuraltide.optimization import (
    CalibrationProblem,
    ObservationSpec,
    ParameterSpace,
    ResidualLoss,
)
from neuraltide.optimization.local import minimize_lbfgsb
from neuraltide.optimization.results import (
    Evaluation,
    LocalOptimizerResult,
)
from neuraltide.optimization.schedules import (
    ContinuationScheduler,
    ScheduleResult,
    Stage,
    horizon_growth,
    objective_schedule,
    smooth_objective,
    teacher_forcing_schedule,
)
from neuraltide.optimization.strategies import (
    ContinuationResult,
    dt_continuation,
    subset_continuation,
)
from neuraltide.populations import WilsonCowan
from neuraltide.synapses import StaticSynapse

T_STEPS = 60


def _build_l0_dt(dt: float) -> CalibrationProblem:
    """L0-сеть с параметризованным шагом интегрирования (float64)."""
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
    syn = StaticSynapse(
        n_pre=1,
        n_post=1,
        dt=dt,
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
    gen = ConstantRateGenerator(dt=dt, params={"rate": 10.0})

    graph = NetworkGraph(dt=dt)
    graph.declare_input("in", n_units=1)
    graph.add_population("exc", pop)
    graph.add_synapse("in->exc", syn, src="in", tgt="exc")
    network = NetworkRNN(graph, EulerIntegrator())

    t_seq = tf.constant((np.arange(T_STEPS, dtype=np.float64) * dt)[None, :, None])
    inputs = graph.pack_inputs({"in": gen(t_seq)})
    target = network(t_seq, inputs=inputs).firing_rates["exc"]
    loss = ResidualLoss(
        [ObservationSpec(name="obs", population="exc", target=target, sigma=2.0)]
    )
    space = ParameterSpace.from_network(network)
    return CalibrationProblem(network, space, t_seq, inputs=inputs, objective=loss)


def _problem64() -> CalibrationProblem:
    neuraltide.config.set_dtype(tf.float64)
    return _build_l0_dt(0.5)


def _offsets(problem) -> dict:
    offsets = {}
    start = 0
    for name, entry in problem.parameter_space.entries.items():
        offsets[name] = (start, entry.numel)
        start += entry.numel
    return offsets


def _shifted_start(problem) -> np.ndarray:
    """Сдвинутый от theta* старт (внутри bounds): tau=20, I_ext=0.5, gsyn=0.2."""
    phi = np.asarray(problem.pack(), dtype=np.float64).reshape(-1).copy()
    off = _offsets(problem)
    phi[off["exc/tau"][0]] = 20.0
    phi[off["exc/I_ext"][0]] = 0.5
    phi[off["in->exc/gsyn_max"][0]] = 0.2
    return phi


def _set_model(problem, phi) -> float:
    """Явный перевод модели в точку phi и потеря в ней."""
    return problem.evaluate(np.asarray(phi, dtype=np.float64)).loss


# ---------------------------------------------------------------------------
# dt_continuation
# ---------------------------------------------------------------------------


class TestDtContinuation:
    def test_carries_theta_between_stages(self):
        neuraltide.config.set_dtype(tf.float64)
        first = _build_l0_dt(1.0)
        start = _shifted_start(first)
        start_loss = _set_model(first, start)

        result = dt_continuation(
            _build_l0_dt, [1.0, 0.5], optimizer="lbfgsb", maxiter=60
        )

        assert isinstance(result, ContinuationResult)
        assert result.final_dt == 0.5
        assert len(result.stage_results) == 2
        assert result.final_result is result.stage_results[-1]
        # Финальная потеря лучше начальной и theta перенесён на fine-сетку.
        assert result.final_result.base.loss_best < start_loss * 0.1
        tau = np.asarray(result.final_result.base.theta_best["exc/tau"]).reshape(-1)
        np.testing.assert_allclose(tau, [8.0], rtol=5e-2)
        i_ext = np.asarray(result.final_result.base.theta_best["exc/I_ext"]).reshape(-1)
        np.testing.assert_allclose(i_ext, [0.8], atol=5e-2)
        # История по стадиям.
        assert [h["dt"] for h in result.history] == [1.0, 0.5]
        assert all(h["loss"] is not None for h in result.history)
        assert result.wall_time > 0.0

    def test_single_dt(self):
        neuraltide.config.set_dtype(tf.float64)
        problem = _build_l0_dt(0.5)
        start_loss = _set_model(problem, _shifted_start(problem))
        result = dt_continuation(_build_l0_dt, [0.5], maxiter=60)
        assert len(result.stage_results) == 1
        assert result.final_dt == 0.5
        assert result.final_result.base.loss_best < start_loss * 0.1

    def test_callable_optimizer(self):
        neuraltide.config.set_dtype(tf.float64)
        problem = _build_l0_dt(1.0)
        start_loss = _set_model(problem, _shifted_start(problem))
        result = dt_continuation(
            _build_l0_dt,
            [1.0, 0.5],
            optimizer=lambda p, phi0, **kw: minimize_lbfgsb(p, phi0, **kw),
            maxiter=60,
        )
        assert result.final_result.base.loss_best < start_loss * 0.1

    def test_empty_dts_rejected(self):
        with pytest.raises(ValueError, match="dts"):
            dt_continuation(_build_l0_dt, [])

    def test_serialization(self, tmp_path):
        neuraltide.config.set_dtype(tf.float64)
        problem = _build_l0_dt(1.0)
        _set_model(problem, _shifted_start(problem))
        result = dt_continuation(_build_l0_dt, [1.0, 0.5], maxiter=60)
        data = result.to_dict()
        json.dumps(data)
        restored = ContinuationResult.from_dict(data)
        assert restored.final_dt == 0.5
        assert len(restored.stage_results) == 2
        assert isinstance(restored.final_result, LocalOptimizerResult)
        assert restored.final_result.base.loss_best == pytest.approx(
            result.final_result.base.loss_best
        )
        np.testing.assert_allclose(
            restored.final_result.base.phi_best,
            result.final_result.base.phi_best,
        )
        path = tmp_path / "continuation.json"
        result.save(path)
        loaded = ContinuationResult.load(path)
        assert loaded.final_dt == 0.5
        assert loaded.history == result.history


# ---------------------------------------------------------------------------
# subset_continuation
# ---------------------------------------------------------------------------


class TestSubsetContinuation:
    def test_subset_then_auto_joint(self):
        problem = _problem64()
        start_loss = _set_model(problem, _shifted_start(problem))

        result = subset_continuation(
            problem,
            [["exc/tau"], ["exc/tau", "exc/I_ext"]],
            local="lbfgsb",
            maxiter=60,
        )

        # Две заданные стадии + автоматическая совместная доводка.
        assert len(result.stage_results) == 3
        assert set(result.stage_results[0].base.theta_best) == {"exc/tau"}
        assert set(result.stage_results[1].base.theta_best) == {
            "exc/tau",
            "exc/I_ext",
        }
        final_names = set(result.stage_results[2].base.theta_best)
        assert final_names == set(problem.parameter_space.entries)
        # Совместная доводка лучше начальной точки.
        assert result.final_result.base.loss_best < start_loss * 0.1
        tau = np.asarray(result.final_result.base.theta_best["exc/tau"]).reshape(-1)
        np.testing.assert_allclose(tau, [8.0], rtol=5e-2)
        assert result.final_result is result.stage_results[-1]

    def test_explicit_joint_last_stage_not_duplicated(self):
        problem = _problem64()
        _set_model(problem, _shifted_start(problem))
        all_names = list(problem.parameter_space.entries)
        result = subset_continuation(problem, [["exc/tau"], all_names], maxiter=60)
        assert len(result.stage_results) == 2

    def test_frozen_first_stage_moves_only_free(self):
        problem = _problem64()
        phi0 = _shifted_start(problem)
        start_loss = _set_model(problem, phi0)
        result = subset_continuation(problem, [["exc/I_ext"]], maxiter=60)
        # Заданная стадия + автоматическая совместная доводка.
        assert len(result.stage_results) == 2
        # Первая стадия двигает ТОЛЬКО свободную запись и улучшает потерю;
        # её решение не обязано совпадать с совместным оптимумом (поэтому
        # joint-стадия обязательна).
        assert set(result.stage_results[0].base.theta_best) == {"exc/I_ext"}
        assert result.stage_results[0].base.loss_best < start_loss
        final_names = set(result.stage_results[-1].base.theta_best)
        assert final_names == set(problem.parameter_space.entries)
        assert result.final_result.base.loss_best < start_loss * 0.1

    def test_unknown_free_name_rejected(self):
        problem = _problem64()
        with pytest.raises(ValueError, match="неизвестные"):
            subset_continuation(problem, [["exc/nope"]])

    def test_empty_stages_rejected(self):
        problem = _problem64()
        with pytest.raises(ValueError, match="stages"):
            subset_continuation(problem, [])

    def test_result_fields(self):
        problem = _problem64()
        start_loss = _set_model(problem, _shifted_start(problem))
        result = subset_continuation(problem, [["exc/tau"]], maxiter=60)
        assert result.final_dt == 0.5  # dt графа базовой задачи
        assert len(result.history) == 2  # заданная стадия + авто joint
        assert result.wall_time > 0.0
        assert result.final_result.base.loss_best < start_loss


# ---------------------------------------------------------------------------
# ContinuationScheduler
# ---------------------------------------------------------------------------


class TestScheduler:
    def test_run_records_stage_results_and_free_run(self):
        problem = _problem64()
        start_loss = _set_model(problem, _shifted_start(problem))
        stages = [
            Stage("warm", problem.with_objective(problem.objective), maxiter=40),
            Stage("final", problem.with_objective(problem.objective), maxiter=40),
        ]
        result = ContinuationScheduler(stages).run(problem)

        assert isinstance(result, ScheduleResult)
        assert len(result.stage_results) == 2
        assert result.final_result is result.stage_results[-1]
        assert result.alpha_history == []
        # Обязательная повторная оценка на оригинальной задаче.
        assert isinstance(result.free_run_check, Evaluation)
        assert np.isfinite(result.free_run_check.loss)
        manual = problem.evaluate(result.final_result.base.phi_best)
        assert result.free_run_check.loss == pytest.approx(manual.loss, rel=1e-8)
        # Стадии улучшили старт.
        assert result.final_result.base.loss_best < start_loss * 0.1
        # История.
        assert [h["stage"] for h in result.history] == ["warm", "final"]
        assert all(h["optimizer"] == "lbfgsb" for h in result.history)
        assert result.history[0]["loss"] >= result.history[1]["loss"]
        assert result.wall_time > 0.0

    def test_callable_optimizer(self):
        problem = _problem64()
        start_loss = _set_model(problem, _shifted_start(problem))
        stages = [
            Stage(
                "custom",
                problem,
                optimizer=lambda p, phi0: minimize_lbfgsb(p, phi0, maxiter=40),
            )
        ]
        result = ContinuationScheduler(stages).run(problem)
        assert result.final_result.base.loss_best < start_loss * 0.1
        assert result.history[0]["optimizer"] == "<lambda>"

    def test_empty_stages_and_wrong_type_rejected(self):
        with pytest.raises(ValueError, match="stages"):
            ContinuationScheduler([])
        with pytest.raises(TypeError, match="Stage"):
            ContinuationScheduler([object()])

    def test_alpha_guard_last_alpha_must_be_zero(self):
        problem = _problem64()
        _set_model(problem, _shifted_start(problem))
        stage = Stage("tf", problem, meta={"alpha": 0.5})
        with pytest.raises(ValueError, match="alpha"):
            ContinuationScheduler([stage]).run(problem)

    def test_serialization(self, tmp_path):
        problem = _problem64()
        _set_model(problem, _shifted_start(problem))
        stages = horizon_growth(problem, [20, 60])
        for stage in stages:
            stage.maxiter = 40
        result = ContinuationScheduler(stages).run(problem)
        data = result.to_dict()
        json.dumps(data)
        restored = ScheduleResult.from_dict(data)
        assert restored.alpha_history == []
        assert len(restored.stage_results) == 2
        assert isinstance(restored.free_run_check, Evaluation)
        assert restored.free_run_check.loss == pytest.approx(result.free_run_check.loss)
        assert restored.history[0]["stage"] == result.history[0]["stage"]
        assert restored.wall_time == pytest.approx(result.wall_time)


# ---------------------------------------------------------------------------
# horizon_growth
# ---------------------------------------------------------------------------


class TestHorizonGrowth:
    def test_stages_sliced_and_runnable(self):
        problem = _problem64()
        start_loss = _set_model(problem, _shifted_start(problem))
        stages = horizon_growth(problem, [20, 60])
        assert len(stages) == 2
        assert stages[0].problem.t_sequence.shape[1] == 20
        assert stages[1].problem.t_sequence.shape[1] == 60
        assert stages[0].meta["stop"] == 20
        # Целевые траектории срезаны под горизонт: evaluate работает.
        phi0 = np.asarray(problem.pack(), dtype=np.float64)
        assert np.isfinite(stages[0].problem.evaluate(phi0).loss)
        # Прогон расписания.
        for stage in stages:
            stage.maxiter = 40
        result = ContinuationScheduler(stages).run(problem)
        assert result.final_result.base.loss_best < start_loss * 0.5

    def test_tuple_spans_and_build_inputs(self):
        problem = _problem64()
        stages = horizon_growth(
            problem,
            [(0, 20), (10, 60)],
            build_inputs=lambda inputs, s, e: inputs[:, s:e] * 2.0,
        )
        assert len(stages) == 2
        doubled = np.asarray(stages[1].problem.inputs, dtype=np.float64)
        reference = 2.0 * np.asarray(problem.inputs[:, 10:60], dtype=np.float64)
        np.testing.assert_allclose(doubled, reference, rtol=1e-6)
        assert stages[1].meta["start"] == 10

    def test_invalid_spans_rejected(self):
        problem = _problem64()
        with pytest.raises(ValueError, match="горизонт"):
            horizon_growth(problem, [999])
        with pytest.raises(ValueError, match="горизонт"):
            horizon_growth(problem, [(40, 10)])

    def test_non_residual_objective_rejected(self):
        problem = _problem64()
        view = problem.with_objective(lambda preds, net: 0.0)
        with pytest.raises(TypeError, match="observations"):
            horizon_growth(view, [20])

    def test_temporal_mask_sliced_with_target_and_loss_consistent(self):
        """Маска [1, T, 1] с осью времени срезается вместе с target.

        Раньше срезалась только target — evaluate срезанной view падал
        с broadcast-ошибкой форм. Здесь: формы среза согласованы, loss на
        срезе конечна, 0.5·||r||² == loss и совпадает с ручным masked mean.
        """
        problem = _problem64()
        t_total = int(problem.t_sequence.shape[1])
        half = t_total // 2
        target = problem.objective.observations[0].target
        mask = np.ones((1, t_total, 1), dtype=np.float64)
        mask[:, half:] = 0.0  # наблюдаем только первую половину горизонта
        masked = ResidualLoss(
            [
                ObservationSpec(
                    name="obs",
                    population="exc",
                    target=target,
                    sigma=2.0,
                    mask=mask,
                )
            ]
        )
        base = problem.with_objective(masked)
        stops = [half, t_total]
        stages = horizon_growth(base, stops)

        phi0 = np.asarray(problem.pack(), dtype=np.float64)
        for stage, stop in zip(stages, stops):
            spec = stage.problem.objective.observations[0]
            # Цели и временные маски срезаны согласованно.
            assert spec.target.shape.as_list() == [1, stop, 1]
            assert spec.mask.shape.as_list() == [1, stop, 1]
            loss = float(stage.problem.evaluate(phi0).loss)
            assert np.isfinite(loss)
            # Контракт residual: 0.5·||r||² == loss на срезе.
            r = np.asarray(stage.problem.residual(phi0), dtype=np.float64)
            assert 0.5 * float(r @ r) == pytest.approx(loss, rel=1e-8)
            # Loss на срезе — точный masked mean по срезанной маске.
            pred = np.asarray(
                stage.problem.simulate(phi0).firing_rates["exc"], dtype=np.float64
            )
            e = pred - np.asarray(spec.target, dtype=np.float64)
            m = np.asarray(spec.mask, dtype=np.float64)
            expected = (
                0.5 * float(np.sum(m * e**2)) / (float(np.sum(m)) * spec.sigma**2)
            )
            assert loss == pytest.approx(expected, rel=1e-8)
        # Вторая половина маски нулевая: срез [0:half] не видит её вклад.
        first_spec = stages[0].problem.objective.observations[0]
        assert float(np.sum(np.asarray(first_spec.mask))) == float(half)

        # Прогон расписания раньше падал на первой же оценке стадии.
        for stage in stages:
            stage.maxiter = 40
        result = ContinuationScheduler(stages).run(base)
        assert np.isfinite(result.free_run_check.loss)

    def test_time_invariant_broadcast_mask_kept_as_is(self):
        """Маска [1, 1, 1] без оси времени не срезается — broadcast валиден."""
        problem = _problem64()
        t_total = int(problem.t_sequence.shape[1])
        target = problem.objective.observations[0].target
        unit_weights = np.array([[[0.25]]], dtype=np.float64)  # [1, 1, 1]
        masked = ResidualLoss(
            [
                ObservationSpec(
                    name="obs",
                    population="exc",
                    target=target,
                    sigma=2.0,
                    mask=unit_weights,
                )
            ]
        )
        base = problem.with_objective(masked)
        stages = horizon_growth(base, [t_total // 2, t_total])
        phi0 = np.asarray(problem.pack(), dtype=np.float64)
        for stage, stop in zip(stages, [t_total // 2, t_total]):
            spec = stage.problem.objective.observations[0]
            assert spec.mask.shape.as_list() == [1, 1, 1]
            assert spec.target.shape.as_list() == [1, stop, 1]
            loss = float(stage.problem.evaluate(phi0).loss)
            assert np.isfinite(loss)
            r = np.asarray(stage.problem.residual(phi0), dtype=np.float64)
            assert 0.5 * float(r @ r) == pytest.approx(loss, rel=1e-8)


# ---------------------------------------------------------------------------
# teacher_forcing_schedule
# ---------------------------------------------------------------------------


class TestTeacherForcing:
    def test_last_alpha_must_be_zero(self):
        problem = _problem64()
        target = np.asarray(problem.inputs, dtype=np.float64)
        with pytest.raises(ValueError, match="alpha"):
            teacher_forcing_schedule(problem, target, [0.5, 0.2])
        with pytest.raises(ValueError, match="alpha"):
            teacher_forcing_schedule(problem, target, [1.0])

    def test_alpha_range_and_empty(self):
        problem = _problem64()
        target = np.asarray(problem.inputs, dtype=np.float64)
        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            teacher_forcing_schedule(problem, target, [1.5, 0.0])
        with pytest.raises(ValueError, match="alphas"):
            teacher_forcing_schedule(problem, target, [])

    def test_inputs_none_rejected(self):
        problem = _problem64()
        bare = CalibrationProblem(
            problem.network,
            problem.parameter_space,
            problem.t_sequence,
            inputs=None,
            objective=problem.objective,
        )
        with pytest.raises(ValueError, match="inputs"):
            teacher_forcing_schedule(bare, np.zeros((1, 2, 1)), [0.0])

    def test_shape_mismatch_rejected(self):
        problem = _problem64()
        with pytest.raises(ValueError, match="формы"):
            teacher_forcing_schedule(
                problem, np.zeros((1, 10, 1), dtype=np.float64), [0.0]
            )

    def test_mixing_formula_and_alpha_meta(self):
        problem = _problem64()
        target = np.zeros_like(np.asarray(problem.inputs, dtype=np.float64))
        stages = teacher_forcing_schedule(problem, target, [0.25, 0.0])
        assert len(stages) == 2
        original = np.asarray(problem.inputs, dtype=np.float64)
        mixed = np.asarray(stages[0].problem.inputs, dtype=np.float64)
        np.testing.assert_allclose(mixed, 0.75 * original, rtol=1e-6)
        closed = np.asarray(stages[1].problem.inputs, dtype=np.float64)
        np.testing.assert_allclose(closed, original, rtol=1e-6)
        assert stages[0].meta["alpha"] == 0.25
        assert stages[1].meta["alpha"] == 0.0
        assert stages[0].problem.inputs is not problem.inputs

    def test_scheduler_end_to_end_with_free_run(self):
        problem = _problem64()
        start_loss = _set_model(problem, _shifted_start(problem))
        target = np.zeros_like(np.asarray(problem.inputs, dtype=np.float64))
        stages = teacher_forcing_schedule(problem, target, [1.0, 0.5, 0.0])
        for stage in stages:
            stage.maxiter = 40
        result = ContinuationScheduler(stages).run(problem)

        assert result.alpha_history == [1.0, 0.5, 0.0]
        assert len(result.stage_results) == 3
        # Free run: оригинальные входы (не смешанные).
        assert isinstance(result.free_run_check, Evaluation)
        free_inputs = np.asarray(problem.inputs, dtype=np.float64)
        assert not np.allclose(
            free_inputs, np.zeros_like(free_inputs)
        )  # входы задачи не изменены scheduler'ом
        assert result.free_run_check.loss == pytest.approx(
            problem.evaluate(result.final_result.base.phi_best).loss, rel=1e-8
        )
        # Финальная стадия (alpha=0) улучшила старт на замкнутой сети.
        assert result.final_result.base.loss_best < start_loss


# ---------------------------------------------------------------------------
# objective_schedule / smooth_objective
# ---------------------------------------------------------------------------


class TestObjectiveSchedule:
    def test_stages_and_run(self):
        problem = _problem64()
        _set_model(problem, _shifted_start(problem))
        target = problem.objective.observations[0].target
        shifted_target = ResidualLoss(
            [
                ObservationSpec(
                    name="obs", population="exc", target=target * 1.5, sigma=2.0
                )
            ]
        )
        stages = objective_schedule(problem, [problem.objective, shifted_target])
        assert len(stages) == 2
        assert stages[0].problem.objective is problem.objective
        assert stages[1].problem.objective is shifted_target
        assert stages[0].meta["objective"] == 0
        for stage in stages:
            stage.maxiter = 40
        result = ContinuationScheduler(stages).run(problem)
        assert len(result.stage_results) == 2
        assert result.final_result.base.loss_best >= 0.0
        assert np.isfinite(result.free_run_check.loss)

    def test_empty_and_noncallable_rejected(self):
        problem = _problem64()
        with pytest.raises(ValueError, match="objectives"):
            objective_schedule(problem, [])
        with pytest.raises(TypeError, match="callable"):
            objective_schedule(problem, [123])


class TestSmoothObjective:
    def test_smoothing_changes_target_keeps_shape(self):
        problem = _problem64()
        smoothed = smooth_objective(problem.objective, 5)
        assert isinstance(smoothed, ResidualLoss)
        original_target = problem.objective.observations[0].target
        smooth_target = smoothed.observations[0].target
        assert smooth_target.shape.as_list() == original_target.shape.as_list()
        assert not np.allclose(np.asarray(smooth_target), np.asarray(original_target))
        # Сглаженная цель глажче: меньше энергия высоких разностей.
        orig_arr = np.asarray(original_target, dtype=np.float64).reshape(-1)
        smooth_arr = np.asarray(smooth_target, dtype=np.float64).reshape(-1)
        assert np.sum(np.diff(smooth_arr) ** 2) < np.sum(np.diff(orig_arr) ** 2)

    def test_view_with_smoothed_objective_evaluates(self):
        problem = _problem64()
        view = problem.with_objective(smooth_objective(problem.objective, 7))
        phi0 = np.asarray(problem.pack(), dtype=np.float64)
        assert np.isfinite(view.evaluate(phi0).loss)

    def test_typeerror_for_non_residual_objective(self):
        problem = _problem64()
        with pytest.raises(TypeError, match="observations"):
            smooth_objective(lambda preds, net: 0.0, 5)
        with pytest.raises(TypeError, match="observations"):
            smooth_objective(problem.with_objective(lambda p, n: 0.0).objective, 5)

    def test_window_validation(self):
        problem = _problem64()
        with pytest.raises(ValueError, match="window"):
            smooth_objective(problem.objective, 1)

    def test_combined_objective_schedule_smooth_to_pointwise(self):
        problem = _problem64()
        start_loss = _set_model(problem, _shifted_start(problem))
        stages = objective_schedule(
            problem,
            [smooth_objective(problem.objective, 9), problem.objective],
        )
        for stage in stages:
            stage.maxiter = 40
        result = ContinuationScheduler(stages).run(problem)
        assert result.final_result.base.loss_best < start_loss * 0.1
