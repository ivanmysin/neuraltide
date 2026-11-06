"""
Тесты этапа 1b: компилируемое ядро optimization-слоя (compiled=True).

CalibrationProblem / MultiExperimentProblem с флагом compiled выполняют
value_and_gradient / evaluate / residual через скомпилированное
tf.function-ядро (_CompiledCore): мутация модели (assign), счётчики и
numpy-сборка Evaluation остаются снаружи в eager. Проверяется
эквивалентность compiled/eager (loss/grad/comps/residual, оба grad_mode,
x0-записи, exp-specific x0, view-хелперы, FrozenProblem, validate,
float64) и отсутствие ретрейсинга ядра при смене φ.
"""

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
    ResidualLoss,
)
from neuraltide.training import CompositeLoss, MSELoss

RTOL = 1e-5
ATOL = 1e-6


def _loss(target, sigma=2.0):
    return ResidualLoss(
        [ObservationSpec(name="obs", population="exc", target=target, sigma=sigma)]
    )


def _phi_from_model(space, jitter=0.0, rng=None):
    """φ из текущих значений модели (+ опциональный шум).

    Работает и для пространств с tensor-записями, где pack() невозможен.
    Координаты L0-близки к физическим (identity-преобразования), поэтому
    малый шум даёт валидную физическую точку (tau не уходит к 0).
    """
    values = {}
    for name, entry in space.phi_entries.items():
        if entry.variable is not None:
            values[name] = entry.variable.numpy()
        else:
            values[name] = np.zeros(entry.shape, dtype=np.float32)
    phi = np.asarray(space.pack(values=values), dtype=np.float32).reshape(-1)
    if jitter:
        phi = phi + rng.standard_normal(phi.size).astype(np.float32) * jitter
    return phi


def _build_pair(l0_factory, compiled_kwargs=None):
    """Пара идентичных задач (eager, compiled) на L0."""
    net_e, space_e, t_seq_e, inputs_e, loss_e = l0_factory()
    net_c, space_c, t_seq_c, inputs_c, loss_c = l0_factory()
    common = dict(inputs=inputs_e, objective=loss_e)
    prob_e = CalibrationProblem(net_e, space_e, t_seq_e, **common)
    prob_c = CalibrationProblem(
        net_c,
        space_c,
        t_seq_c,
        inputs=inputs_c,
        objective=loss_c,
        compiled=True,
        **(compiled_kwargs or {}),
    )
    return prob_e, prob_c


class TestCompiledMatchesEager:
    @pytest.mark.parametrize("grad_mode", ["assign", "chain"])
    def test_value_gradient_evaluate_residual(self, l0_factory, grad_mode):
        prob_e, prob_c = _build_pair(l0_factory, {"grad_mode": grad_mode})
        rng = np.random.default_rng(0)
        phi = _phi_from_model(prob_e.parameter_space, jitter=0.02, rng=rng)

        ev_e, g_e = prob_e.value_and_gradient(phi)
        ev_c, g_c = prob_c.value_and_gradient(phi)
        assert ev_e.loss == pytest.approx(ev_c.loss, rel=RTOL, abs=ATOL)
        np.testing.assert_allclose(g_c.numpy(), g_e.numpy(), rtol=RTOL, atol=ATOL)
        assert ev_c.grad is not None
        np.testing.assert_allclose(
            np.asarray(ev_c.grad), np.asarray(ev_e.grad), rtol=RTOL, atol=ATOL
        )
        assert ev_e.components.keys() == ev_c.components.keys()
        for k in ev_e.components:
            assert ev_e.components[k] == pytest.approx(
                ev_c.components[k], rel=RTOL, abs=ATOL
            )
        np.testing.assert_allclose(
            np.asarray(ev_c.residuals), np.asarray(ev_e.residuals), rtol=RTOL, atol=ATOL
        )
        assert ev_e.status == ev_c.status == "ok"

        ev_e = prob_e.evaluate(phi)
        ev_c = prob_c.evaluate(phi)
        assert ev_e.loss == pytest.approx(ev_c.loss, rel=RTOL, abs=ATOL)

        r_e = prob_e.residual(phi)
        r_c = prob_c.residual(phi)
        np.testing.assert_allclose(r_c, r_e, rtol=RTOL, atol=ATOL)

        # Счётчики семантически совпадают.
        assert prob_e.n_forward == prob_c.n_forward
        assert prob_e.n_gradient == prob_c.n_gradient

    def test_grad_mode_assign_literal_path(self, l0_factory):
        """'assign' с чисто-x0 пространством (нет model_var) — literal-путь
        tape.gradient(loss, phi) внутри графа."""
        net_e, space_e, t_seq, inputs, _ = l0_factory()
        net_c, space_c, _, _, _ = l0_factory()
        for space in (space_e, space_c):
            for name in ("exc/tau", "exc/I_ext", "in->exc/gsyn_max"):
                space.entries[name].trainable = False
        space_e.add_initial_state_entries(net_e, include=[("exc", 0)])
        space_c.add_initial_state_entries(net_c, include=[("exc", 0)])
        loss_e = _loss(net_e(t_seq, inputs=inputs).firing_rates["exc"])
        loss_c = _loss(net_c(t_seq, inputs=inputs).firing_rates["exc"])
        prob_e = CalibrationProblem(
            net_e, space_e, t_seq, inputs=inputs, objective=loss_e, grad_mode="assign"
        )
        prob_c = CalibrationProblem(
            net_c,
            space_c,
            t_seq,
            inputs=inputs,
            objective=loss_c,
            grad_mode="assign",
            compiled=True,
        )
        rng = np.random.default_rng(1)
        phi = _phi_from_model(space_e, jitter=0.02, rng=rng)
        ev_e, g_e = prob_e.value_and_gradient(phi)
        ev_c, g_c = prob_c.value_and_gradient(phi)
        assert ev_e.loss == pytest.approx(ev_c.loss, rel=RTOL, abs=ATOL)
        np.testing.assert_allclose(g_c.numpy(), g_e.numpy(), rtol=RTOL, atol=ATOL)
        assert prob_e.size == prob_c.size == 1

    def test_float64(self, l0_factory):
        neuraltide.config.set_dtype(tf.float64)
        prob_e, prob_c = _build_pair(l0_factory)
        rng = np.random.default_rng(2)
        phi = _phi_from_model(prob_e.parameter_space, jitter=0.02, rng=rng)
        ev_e, g_e = prob_e.value_and_gradient(phi)
        ev_c, g_c = prob_c.value_and_gradient(phi)
        assert ev_e.loss == pytest.approx(ev_c.loss, rel=1e-12, abs=1e-14)
        np.testing.assert_allclose(g_c.numpy(), g_e.numpy(), rtol=1e-12, atol=1e-14)
        r_e = prob_e.residual(phi)
        r_c = prob_c.residual(phi)
        np.testing.assert_allclose(r_c, r_e, rtol=1e-12, atol=1e-14)


class TestCompiledX0:
    def test_common_x0_entries_chain(self, l0_factory):
        net_e, space_e, t_seq, inputs, _ = l0_factory()
        net_c, space_c, _, _, _ = l0_factory()
        space_e.add_initial_state_entries(net_e, include=[("exc", 0)])
        space_c.add_initial_state_entries(net_c, include=[("exc", 0)])
        loss_e = _loss(net_e(t_seq, inputs=inputs).firing_rates["exc"])
        loss_c = _loss(net_c(t_seq, inputs=inputs).firing_rates["exc"])
        prob_e = CalibrationProblem(
            net_e, space_e, t_seq, inputs=inputs, objective=loss_e, grad_mode="chain"
        )
        prob_c = CalibrationProblem(
            net_c,
            space_c,
            t_seq,
            inputs=inputs,
            objective=loss_c,
            grad_mode="chain",
            compiled=True,
        )
        rng = np.random.default_rng(3)
        phi = _phi_from_model(space_e, jitter=0.02, rng=rng)
        ev_e, g_e = prob_e.value_and_gradient(phi)
        ev_c, g_c = prob_c.value_and_gradient(phi)
        assert ev_e.loss == pytest.approx(ev_c.loss, rel=RTOL, abs=ATOL)
        np.testing.assert_allclose(g_c.numpy(), g_e.numpy(), rtol=RTOL, atol=ATOL)
        # x0-координата (последняя) получает ненулевой градиент в обеих.
        assert abs(float(g_e[-1])) > 0
        np.testing.assert_allclose(
            np.asarray(prob_c.residual(phi)),
            np.asarray(prob_e.residual(phi)),
            rtol=RTOL,
            atol=ATOL,
        )


class TestCompiledNoRetrace:
    def test_no_retrace_across_phi(self, l0_factory):
        _, prob_c = _build_pair(l0_factory)
        rng = np.random.default_rng(4)
        base = _phi_from_model(prob_c.parameter_space)
        for i in range(3):
            phi = base + rng.standard_normal(prob_c.size).astype(np.float32) * 0.01
            prob_c.value_and_gradient(phi)
            prob_c.evaluate(phi)
            prob_c.residual(phi)
        for core in prob_c._core_cache.values():
            # _list_all_concrete_functions — приватный TF API (пинован к
            # TF 2.21): считает concrete-функции ядра для проверки
            # отсутствия ретрейсинга при смене φ.
            assert len(core.grad_core._list_all_concrete_functions()) == 1
            assert len(core.eval_core._list_all_concrete_functions()) == 1


def _build_multi_pair(l0_factory, with_exp_x0=True):
    """Пара MultiExperimentProblem (eager, compiled): exp1 — базовый
    протокол, exp2 — другие входы/таргет/начальное состояние (+exp-x0)."""
    net_e, space_e, t_seq, inputs1, _ = l0_factory()
    net_c, space_c, _, _, _ = l0_factory()
    if with_exp_x0:
        space_e.add_experiment_initial_state_entries(
            net_e, "exp2", include=[("exc", 0)]
        )
        space_c.add_experiment_initial_state_entries(
            net_c, "exp2", include=[("exc", 0)]
        )
    inputs2 = inputs1 * 0.5
    pop0, syn0 = net_e.get_initial_state()
    state2 = (
        [tf.constant([[0.7]], dtype=neuraltide.config.get_dtype())] + list(pop0[1:]),
        list(syn0),
    )
    tg1 = net_e(t_seq, inputs=inputs1).firing_rates["exc"]
    tg2 = net_e(t_seq, inputs=inputs2, initial_state=state2).firing_rates["exc"]
    exp1 = Experiment(
        name="exp1", t_sequence=t_seq, inputs=inputs1, objective=_loss(tg1)
    )
    exp2 = Experiment(
        name="exp2",
        t_sequence=t_seq,
        inputs=inputs2,
        objective=_loss(tg2),
        initial_state=state2,
    )
    prob_e = MultiExperimentProblem(
        net_e, space_e, [exp1, exp2], grad_mode="chain", compiled=False
    )
    prob_c = MultiExperimentProblem(
        net_c, space_c, [exp1, exp2], grad_mode="chain", compiled=True
    )
    return prob_e, prob_c, space_e


class TestCompiledMultiExperiment:
    def test_matches_eager_with_exp_x0(self, l0_factory):
        prob_e, prob_c, space = _build_multi_pair(l0_factory, with_exp_x0=True)
        rng = np.random.default_rng(5)
        phi = _phi_from_model(space, jitter=0.02, rng=rng)

        ev_e, g_e = prob_e.value_and_gradient(phi)
        ev_c, g_c = prob_c.value_and_gradient(phi)
        assert ev_e.loss == pytest.approx(ev_c.loss, rel=RTOL, abs=ATOL)
        np.testing.assert_allclose(g_c.numpy(), g_e.numpy(), rtol=RTOL, atol=ATOL)
        assert ev_e.components.keys() == ev_c.components.keys()
        for k in ev_e.components:
            assert ev_e.components[k] == pytest.approx(
                ev_c.components[k], rel=RTOL, abs=ATOL
            )
        np.testing.assert_allclose(
            np.asarray(ev_c.residuals), np.asarray(ev_e.residuals), rtol=RTOL, atol=ATOL
        )
        # Счётчики: n_forward += число экспериментов, n_gradient += 1.
        assert prob_e.n_forward == prob_c.n_forward == 2
        assert prob_e.n_gradient == prob_c.n_gradient == 1

        ev_e = prob_e.evaluate(phi)
        ev_c = prob_c.evaluate(phi)
        assert ev_e.loss == pytest.approx(ev_c.loss, rel=RTOL, abs=ATOL)
        r_e = prob_e.residual(phi)
        r_c = prob_c.residual(phi)
        np.testing.assert_allclose(r_c, r_e, rtol=RTOL, atol=ATOL)
        # vg (2 прогона) + evaluate (2) + residual (2)
        assert prob_e.n_forward == prob_c.n_forward == 6

    def test_gradient_isolation_exp_x0(self, l0_factory):
        """Градиент по exp-specific x0 не смешивается между экспериментами:
        сверка с разностью потерь (шаг по этой координате меняет только
        exp2)."""
        prob_e, prob_c, space = _build_multi_pair(l0_factory, with_exp_x0=True)
        rng = np.random.default_rng(6)
        phi = _phi_from_model(space, jitter=0.02, rng=rng)
        _, g_c = prob_c.value_and_gradient(phi)
        # Координата exp-specific x0 — последняя; общие x0 нет.
        x0_idx = int(np.argmax([n.startswith("x0/exp2/") for n in space.phi_entries]))
        step = np.zeros_like(phi)
        step[x0_idx] = 1e-3
        lo = prob_c.evaluate(phi - step).loss
        hi = prob_c.evaluate(phi + step).loss
        fd = (hi - lo) / (2e-3)
        assert float(g_c[x0_idx]) == pytest.approx(fd, rel=1e-2, abs=1e-2)


class TestCompiledViews:
    def test_time_slice_with_objective_with_inputs(self, l0_factory):
        prob_e, prob_c = _build_pair(l0_factory)
        rng = np.random.default_rng(7)
        phi = _phi_from_model(prob_e.parameter_space, jitter=0.02, rng=rng)
        # time_slice требует срезанной цели (как в реальном использовании).
        sl_e = prob_e.time_slice(10, 50).with_objective(
            _loss(
                prob_e.network(prob_e.t_sequence, inputs=prob_e.inputs).firing_rates[
                    "exc"
                ][:, 10:50]
            )
        )
        sl_c = prob_c.time_slice(10, 50).with_objective(
            _loss(
                prob_c.network(prob_c.t_sequence, inputs=prob_c.inputs).firing_rates[
                    "exc"
                ][:, 10:50]
            )
        )
        new_obj_e = _loss(
            prob_e.network(prob_e.t_sequence, inputs=prob_e.inputs).firing_rates["exc"]
            * 1.1
        )
        new_obj_c = _loss(
            prob_c.network(prob_c.t_sequence, inputs=prob_c.inputs).firing_rates["exc"]
            * 1.1
        )
        variants_e = [
            sl_e,
            prob_e.with_objective(new_obj_e),
            prob_e.with_inputs(prob_e.inputs * 0.5),
        ]
        variants_c = [
            sl_c,
            prob_c.with_objective(new_obj_c),
            prob_c.with_inputs(prob_c.inputs * 0.5),
        ]
        for view_e, view_c in zip(variants_e, variants_c):
            assert view_c.compiled
            ev_e, g_e = view_e.value_and_gradient(phi)
            ev_c, g_c = view_c.value_and_gradient(phi)
            assert ev_e.loss == pytest.approx(ev_c.loss, rel=RTOL, abs=ATOL)
            np.testing.assert_allclose(g_c.numpy(), g_e.numpy(), rtol=RTOL, atol=ATOL)


class TestCompiledFrozen:
    def test_frozen_over_compiled(self, l0_factory):
        prob_e, prob_c = _build_pair(l0_factory)
        fr_e = FrozenProblem(prob_e, free_names=["exc/tau", "in->exc/gsyn_max"])
        fr_c = FrozenProblem(prob_c, free_names=["exc/tau", "in->exc/gsyn_max"])
        rng = np.random.default_rng(8)
        phi_full = _phi_from_model(prob_e.parameter_space, jitter=0.02, rng=rng)
        phi_free = phi_full[fr_c._free_index]
        ev_e, g_e = fr_e.value_and_gradient(phi_free)
        ev_c, g_c = fr_c.value_and_gradient(phi_free)
        assert ev_e.loss == pytest.approx(ev_c.loss, rel=RTOL, abs=ATOL)
        np.testing.assert_allclose(g_c.numpy(), g_e.numpy(), rtol=RTOL, atol=ATOL)
        assert fr_c.size == fr_e.size == 2


class TestCompiledValidate:
    def test_validate_same_in_both_modes(self, l0_factory):
        net_e, space_e, t_seq, inputs, loss = l0_factory()
        net_c, space_c, _, _, _ = l0_factory()
        inputs_h = inputs * 0.5
        pop0, syn0 = net_e.get_initial_state()
        state_h = (
            [tf.constant([[0.4]], dtype=neuraltide.config.get_dtype())]
            + list(pop0[1:]),
            list(syn0),
        )
        tg_h = net_e(t_seq, inputs=inputs_h, initial_state=state_h).firing_rates["exc"]
        held = Experiment(
            name="held",
            t_sequence=t_seq,
            inputs=inputs_h,
            objective=_loss(tg_h),
            initial_state=state_h,
        )
        prob_e = CalibrationProblem(
            net_e, space_e, t_seq, inputs=inputs, objective=loss, held_out=[held]
        )
        prob_c = CalibrationProblem(
            net_c,
            space_c,
            t_seq,
            inputs=inputs,
            objective=loss,
            held_out=[held],
            compiled=True,
        )
        rng = np.random.default_rng(9)
        phi = _phi_from_model(space_e, jitter=0.02, rng=rng)
        v_e = prob_e.validate(phi)
        v_c = prob_c.validate(phi)
        assert v_e.keys() == v_c.keys()
        for k in v_e:
            assert v_e[k] == pytest.approx(v_c[k], rel=RTOL, abs=ATOL)


class TestCompiledEdgeCases:
    def test_no_residual_interface_status(self, l0_factory):
        """objective без residual-интерфейса (CompositeLoss/MSELoss):
        status 'no_residual_interface', residuals пусты, loss совпадает
        с eager."""
        net_e, space_e, t_seq, inputs, _ = l0_factory()
        net_c, space_c, _, _, _ = l0_factory()
        target = net_e(t_seq, inputs=inputs).firing_rates["exc"]
        obj_e = CompositeLoss([(1.0, MSELoss({"exc": target}))])
        obj_c = CompositeLoss([(1.0, MSELoss({"exc": target}))])
        prob_e = CalibrationProblem(
            net_e, space_e, t_seq, inputs=inputs, objective=obj_e
        )
        prob_c = CalibrationProblem(
            net_c,
            space_c,
            t_seq,
            inputs=inputs,
            objective=obj_c,
            compiled=True,
        )
        rng = np.random.default_rng(10)
        phi = _phi_from_model(space_e, jitter=0.02, rng=rng)
        ev_e = prob_e.evaluate(phi)
        ev_c = prob_c.evaluate(phi)
        assert ev_e.status == ev_c.status == "no_residual_interface"
        assert ev_c.residuals.shape == (0,)
        assert ev_e.loss == pytest.approx(ev_c.loss, rel=RTOL, abs=ATOL)

    def test_multi_residual_typeerror(self, l0_factory):
        """compiled residual у Multi с objective без интерфейса → TypeError
        (как в eager)."""
        net, space, t_seq, inputs1, _ = l0_factory()
        inputs2 = inputs1 * 0.5
        target = net(t_seq, inputs=inputs1).firing_rates["exc"]
        exp1 = Experiment(
            name="exp1", t_sequence=t_seq, inputs=inputs1, objective=_loss(target)
        )
        exp2 = Experiment(
            name="exp2",
            t_sequence=t_seq,
            inputs=inputs2,
            objective=CompositeLoss([(1.0, MSELoss({"exc": target}))]),
        )
        prob_c = MultiExperimentProblem(net, space, [exp1, exp2], compiled=True)
        prob_e = MultiExperimentProblem(net, space, [exp1, exp2], compiled=False)
        phi = np.zeros(prob_c.size, dtype=np.float32)
        with pytest.raises(TypeError, match="residual"):
            prob_c.residual(phi)
        with pytest.raises(TypeError, match="residual"):
            prob_e.residual(phi)

    def test_phi_none(self, l0_factory):
        """phi=None: pack из текущих значений модели, сверка с eager."""
        prob_e, prob_c = _build_pair(l0_factory)
        ev_e, g_e = prob_e.value_and_gradient(None)
        ev_c, g_c = prob_c.value_and_gradient(None)
        assert ev_e.loss == pytest.approx(ev_c.loss, rel=RTOL, abs=ATOL)
        np.testing.assert_allclose(g_c.numpy(), g_e.numpy(), rtol=RTOL, atol=ATOL)
        assert prob_e.n_gradient == prob_c.n_gradient == 1

    def test_explicit_protocol_initial_state(self, l0_factory):
        """with_initial_state (явное начальное состояние протокола,
        отличное от дефолтного): сверка с eager."""
        prob_e, prob_c = _build_pair(l0_factory)
        pop0, syn0 = prob_e.network.get_initial_state()
        state = (
            [tf.constant([[0.7]], dtype=neuraltide.config.get_dtype())]
            + list(pop0[1:]),
            list(syn0),
        )
        prob_e = prob_e.with_initial_state(state)
        prob_c = prob_c.with_initial_state(state)
        rng = np.random.default_rng(12)
        phi = _phi_from_model(prob_e.parameter_space, jitter=0.02, rng=rng)
        ev_e, g_e = prob_e.value_and_gradient(phi)
        ev_c, g_c = prob_c.value_and_gradient(phi)
        assert ev_e.loss == pytest.approx(ev_c.loss, rel=RTOL, abs=ATOL)
        np.testing.assert_allclose(g_c.numpy(), g_e.numpy(), rtol=RTOL, atol=ATOL)

    def test_initial_state_inplace_mutation(self, l0_factory):
        """Регрессия MINOR-1: in-place смена problem.initial_state
        отражается немедленно (начальное состояние — аргумент ядра, не
        замыкание), значения совпадают с eager после той же мутации."""
        prob_e, prob_c = _build_pair(l0_factory)
        rng = np.random.default_rng(13)
        phi = _phi_from_model(prob_e.parameter_space, jitter=0.02, rng=rng)
        prob_c.value_and_gradient(phi)  # ядро построено и закэшировано
        pop0, syn0 = prob_e.network.get_initial_state()
        new_state = (
            [tf.constant([[0.9]], dtype=neuraltide.config.get_dtype())]
            + list(pop0[1:]),
            list(syn0),
        )
        prob_e.initial_state = new_state
        prob_c.initial_state = new_state
        ev_e, g_e = prob_e.value_and_gradient(phi)
        ev_c, g_c = prob_c.value_and_gradient(phi)
        assert ev_e.loss == pytest.approx(ev_c.loss, rel=RTOL, abs=ATOL)
        np.testing.assert_allclose(g_c.numpy(), g_e.numpy(), rtol=RTOL, atol=ATOL)

    def test_multi_common_x0_accumulation(self, l0_factory):
        """Общая x0-запись у двух экспериментов: градиент по ней
        накапливает вклады обоих экспериментов (сверка с eager, add_n
        вне ядер)."""
        net_e, space_e, t_seq, inputs1, _ = l0_factory()
        net_c, space_c, _, _, _ = l0_factory()
        for net, space in ((net_e, space_e), (net_c, space_c)):
            space.add_initial_state_entries(net, include=[("exc", 0)])
        inputs2 = inputs1 * 0.5
        tg1 = net_e(t_seq, inputs=inputs1).firing_rates["exc"]
        tg2 = net_e(t_seq, inputs=inputs2).firing_rates["exc"]
        exp1 = Experiment(
            name="exp1", t_sequence=t_seq, inputs=inputs1, objective=_loss(tg1)
        )
        exp2 = Experiment(
            name="exp2", t_sequence=t_seq, inputs=inputs2, objective=_loss(tg2)
        )
        prob_e = MultiExperimentProblem(
            net_e, space_e, [exp1, exp2], grad_mode="chain", compiled=False
        )
        prob_c = MultiExperimentProblem(
            net_c, space_c, [exp1, exp2], grad_mode="chain", compiled=True
        )
        rng = np.random.default_rng(14)
        phi = _phi_from_model(space_e, jitter=0.02, rng=rng)
        ev_e, g_e = prob_e.value_and_gradient(phi)
        ev_c, g_c = prob_c.value_and_gradient(phi)
        assert ev_e.loss == pytest.approx(ev_c.loss, rel=RTOL, abs=ATOL)
        np.testing.assert_allclose(g_c.numpy(), g_e.numpy(), rtol=RTOL, atol=ATOL)
        # Общая x0 — последняя координата; вклад ненулевой в обеих ветках.
        x0_idx = prob_e.size - 1
        assert abs(float(g_e[x0_idx])) > 0
        assert float(g_e[x0_idx]) == pytest.approx(
            float(g_c[x0_idx]), rel=RTOL, abs=ATOL
        )
