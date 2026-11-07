"""Этап 3: векторный режим MultiExperimentProblem (vectorize=True).

Все обучающие эксперименты одним батч-прогоном сети. Эквивалентность
цикловому режиму (vectorize=False): loss_vec == mean_e loss_e (цикловой
/E), grad_vec == grad_loop / E, residual — конкатенация, компоненты с
префиксом '{exp}/'. Покрываются: x0-записи (common broadcast + exp-specific
one-hot изоляция), per-exp initial_state, compiled-ядро (одно на батч,
no-retrace), валидации (T/профиль входов), simulate/validate (в т.ч.
held-out батчем и по одному), float64, adjoint-запрет.
"""

import numpy as np
import pytest
import tensorflow as tf

import neuraltide.config
from neuraltide.inputs import ConstantRateGenerator
from neuraltide.optimization import (
    Experiment,
    MultiExperimentProblem,
    ObservationSpec,
    ResidualLoss,
)
from neuraltide.optimization.problem import _VectorizedObjective

DT = 0.5
T = 60
E = 3  # число обучающих экспериментов

RTOL = 1e-5
ATOL = 1e-6


def _loss(target, sigma=2.0):
    return ResidualLoss(
        [ObservationSpec(name="obs", population="exc", target=target, sigma=sigma)]
    )


def _phi_from_model(problem, jitter=0.0, rng=None):
    """φ из текущих значений модели (+ шум); работает и с x0-записями."""
    values = {}
    for name, entry in problem.parameter_space.phi_entries.items():
        if entry.variable is not None:
            values[name] = entry.variable.numpy()
        else:
            values[name] = np.zeros(entry.shape, dtype=np.float32)
    phi = np.asarray(
        problem.parameter_space.pack(values=values), dtype=np.float32
    ).reshape(-1)
    if jitter:
        phi = phi + rng.standard_normal(problem.size).astype(np.float32) * jitter
    return phi


def build_experiments(l0_factory, with_x0=False, exp_x0_for=None, per_exp_state=False):
    """Три обучающих эксперимента на L0 с разными входами/таргетами.

    Returns (problem_kwargs: dict для MultiExperimentProblem, phi_hint_space).
    """
    network, space, t_seq, inputs1, _ = l0_factory()
    inputs2 = inputs1 * 0.5
    inputs3 = network._graph.pack_inputs(
        {"in": ConstantRateGenerator(dt=DT, params={"rate": 25.0})(t_seq)}
    )
    if with_x0:
        space.add_initial_state_entries(network, include=[("exc", 0)])
    if exp_x0_for is not None:
        space.add_experiment_initial_state_entries(
            network, exp_x0_for, include=[("exc", 0)]
        )
    pop0, syn0 = network.get_initial_state()
    state3 = None
    if per_exp_state:
        state3 = (
            [tf.constant([[0.7]], dtype=neuraltide.config.get_dtype())]
            + list(pop0[1:]),
            list(syn0),
        )
    t1 = network(t_seq, inputs=inputs1).firing_rates["exc"]
    t2 = network(t_seq, inputs=inputs2).firing_rates["exc"]
    if state3 is not None:
        t3 = network(t_seq, inputs=inputs3, initial_state=state3).firing_rates["exc"]
    else:
        t3 = network(t_seq, inputs=inputs3).firing_rates["exc"]
    experiments = [
        Experiment(name="exp1", t_sequence=t_seq, inputs=inputs1, objective=_loss(t1)),
        Experiment(name="exp2", t_sequence=t_seq, inputs=inputs2, objective=_loss(t2)),
        Experiment(
            name="exp3",
            t_sequence=t_seq,
            inputs=inputs3,
            objective=_loss(t3),
            initial_state=state3,
        ),
    ]
    return {"network": network, "space": space, "experiments": experiments}


def build_pair(
    l0_factory,
    vectorize_a=True,
    vectorize_b=False,
    compiled_a=False,
    compiled_b=False,
    **exp_kwargs,
):
    ka = build_experiments(l0_factory, **exp_kwargs)
    kb = build_experiments(l0_factory, **exp_kwargs)
    pa = MultiExperimentProblem(
        ka["network"],
        ka["space"],
        ka["experiments"],
        grad_mode="chain",
        compiled=compiled_a,
        vectorize=vectorize_a,
    )
    pb = MultiExperimentProblem(
        kb["network"],
        kb["space"],
        kb["experiments"],
        grad_mode="chain",
        compiled=compiled_b,
        vectorize=vectorize_b,
    )
    return pa, pb


class TestVectorizedMatchesLoop:
    def test_value_gradient_evaluate_residual(self, l0_factory):
        pv, pl = build_pair(l0_factory)
        rng = np.random.default_rng(0)
        phi = _phi_from_model(pv, jitter=0.02, rng=rng)

        ev_v, g_v = pv.value_and_gradient(phi)
        ev_l, g_l = pl.value_and_gradient(phi)
        assert ev_v.loss == pytest.approx(ev_l.loss / E, rel=RTOL, abs=ATOL)
        np.testing.assert_allclose(g_v.numpy(), g_l.numpy() / E, rtol=RTOL, atol=ATOL)
        assert ev_v.components.keys() == ev_l.components.keys()
        for k in ev_v.components:
            assert ev_v.components[k] == pytest.approx(
                ev_l.components[k], rel=RTOL, abs=ATOL
            )
        np.testing.assert_allclose(
            np.asarray(ev_v.residuals), np.asarray(ev_l.residuals), rtol=RTOL, atol=ATOL
        )
        assert ev_v.status == "ok"
        # Счётчики: векторный режим — 1 батч-прогон, цикловой — E.
        assert pv.n_forward == 1
        assert pl.n_forward == E

        ev_v = pv.evaluate(phi)
        ev_l = pl.evaluate(phi)
        assert ev_v.loss == pytest.approx(ev_l.loss / E, rel=RTOL, abs=ATOL)

        r_v = pv.residual(phi)
        r_l = pl.residual(phi)
        np.testing.assert_allclose(r_v, r_l, rtol=RTOL, atol=ATOL)

    def test_float64(self, l0_factory):
        neuraltide.config.set_dtype(tf.float64)
        pv, pl = build_pair(l0_factory)
        rng = np.random.default_rng(1)
        phi = _phi_from_model(pv, jitter=0.02, rng=rng)
        ev_v, g_v = pv.value_and_gradient(phi)
        ev_l, g_l = pl.value_and_gradient(phi)
        assert ev_v.loss == pytest.approx(ev_l.loss / E, rel=1e-12, abs=1e-14)
        np.testing.assert_allclose(g_v.numpy(), g_l.numpy() / E, rtol=1e-12, atol=1e-14)


class TestVectorizedX0:
    def test_common_and_exp_specific_x0(self, l0_factory):
        pv, pl = build_pair(l0_factory, with_x0=True, exp_x0_for="exp2")
        rng = np.random.default_rng(2)
        phi = _phi_from_model(pv, jitter=0.02, rng=rng)
        ev_v, g_v = pv.value_and_gradient(phi)
        ev_l, g_l = pl.value_and_gradient(phi)
        assert ev_v.loss == pytest.approx(ev_l.loss / E, rel=RTOL, abs=ATOL)
        np.testing.assert_allclose(g_v.numpy(), g_l.numpy() / E, rtol=1e-4, atol=1e-5)
        names = list(pv.parameter_space.phi_entries)
        idx_common = names.index("x0/exc/0")
        idx_exp = names.index("x0/exp2/exc/0")
        # common: вклад всех экспериментов (broadcast backward суммирует слоты)
        assert abs(float(g_v[idx_common])) > 0
        # exp-specific: ненулевой вклад только своего эксперимента (one-hot),
        # численная сверка с цикловым (изоляция покрыта test_multi_experiment).
        assert abs(float(g_v[idx_exp])) > 0
        assert float(g_v[idx_exp]) == pytest.approx(
            float(g_l[idx_exp] / E), rel=1e-4, abs=1e-5
        )

    def test_per_experiment_initial_state(self, l0_factory):
        pv, pl = build_pair(l0_factory, per_exp_state=True)
        rng = np.random.default_rng(3)
        phi = _phi_from_model(pv, jitter=0.02, rng=rng)
        ev_v, g_v = pv.value_and_gradient(phi)
        ev_l, g_l = pl.value_and_gradient(phi)
        assert ev_v.loss == pytest.approx(ev_l.loss / E, rel=RTOL, abs=ATOL)
        np.testing.assert_allclose(g_v.numpy(), g_l.numpy() / E, rtol=1e-4, atol=1e-5)


class TestVectorizedValidations:
    def test_different_T_raises(self, l0_factory):
        network, space, t_seq, inputs1, _ = l0_factory()
        t_short = t_seq[:, :30]
        gen = ConstantRateGenerator(dt=DT, params={"rate": 10.0})
        inputs_short = network._graph.pack_inputs({"in": gen(t_short)})
        target = network(t_seq, inputs=inputs1).firing_rates["exc"]
        target_short = network(t_short, inputs=inputs_short).firing_rates["exc"]
        experiments = [
            Experiment(
                name="e1", t_sequence=t_seq, inputs=inputs1, objective=_loss(target)
            ),
            Experiment(
                name="e2",
                t_sequence=t_short,
                inputs=inputs_short,
                objective=_loss(target_short),
            ),
        ]
        with pytest.raises(ValueError, match="одинаковую длину T"):
            MultiExperimentProblem(network, space, experiments, vectorize=True)

    def test_inputs_profile_mismatch_raises(self, l0_factory):
        network, space, t_seq, inputs1, _ = l0_factory()
        target = network(t_seq, inputs=inputs1).firing_rates["exc"]
        experiments = [
            Experiment(
                name="e1", t_sequence=t_seq, inputs=inputs1, objective=_loss(target)
            ),
            Experiment(
                name="e2", t_sequence=t_seq, inputs=None, objective=_loss(target)
            ),
        ]
        # Проверка профиля входов срабатывает в конструкторе (до прогона).
        with pytest.raises(ValueError, match="профиль входов"):
            MultiExperimentProblem(network, space, experiments, vectorize=True)

    def test_simulate_returns_experiment_slice(self, l0_factory):
        pv, pl = build_pair(l0_factory)
        phi = _phi_from_model(pv)
        for exp in ("exp1", "exp2", "exp3"):
            out_v = pv.simulate(phi, experiment=exp)
            out_l = pl.simulate(phi, experiment=exp)
            np.testing.assert_allclose(
                out_v.firing_rates["exc"].numpy(),
                out_l.firing_rates["exc"].numpy(),
                rtol=RTOL,
                atol=ATOL,
            )
        # experiment=None — срез первого эксперимента
        out_v = pv.simulate(phi)
        out_l = pl.simulate(phi)
        np.testing.assert_allclose(
            out_v.firing_rates["exc"].numpy(),
            out_l.firing_rates["exc"].numpy(),
            rtol=RTOL,
            atol=ATOL,
        )
        assert out_v.firing_rates["exc"].shape[0] == 1

    def test_validate_batched_and_loop(self, l0_factory):
        """Held-out одной длины/профиля — батчем (значения == цикловому);
        held-out другой длины — по одному."""
        neuraltide.config.set_dtype(tf.float32)
        ka = build_experiments(l0_factory)
        network, space, t_seq = (
            ka["network"],
            ka["space"],
            ka["experiments"][0].t_sequence,
        )
        inputs_held = ka["experiments"][0].inputs * 0.7
        pop0, syn0 = network.get_initial_state()
        state_held = (
            [tf.constant([[0.4]], dtype=neuraltide.config.get_dtype())]
            + list(pop0[1:]),
            list(syn0),
        )
        target_held = network(
            t_seq, inputs=inputs_held, initial_state=state_held
        ).firing_rates["exc"]
        held = Experiment(
            name="held",
            t_sequence=t_seq,
            inputs=inputs_held,
            objective=_loss(target_held),
            initial_state=state_held,
        )
        pv = MultiExperimentProblem(
            network, space, ka["experiments"], held_out=[held], vectorize=True
        )
        kb = build_experiments(l0_factory)
        network2, space2 = kb["network"], kb["space"]
        target_held2 = network2(
            t_seq, inputs=inputs_held, initial_state=state_held
        ).firing_rates["exc"]
        held2 = Experiment(
            name="held",
            t_sequence=t_seq,
            inputs=inputs_held,
            objective=_loss(target_held2),
            initial_state=state_held,
        )
        pl = MultiExperimentProblem(
            network2, space2, kb["experiments"], held_out=[held2], vectorize=False
        )
        phi = _phi_from_model(pv, jitter=0.02, rng=np.random.default_rng(5))
        v_v = pv.validate(phi)
        v_l = pl.validate(phi)
        assert v_v.keys() == v_l.keys()
        for k in v_v:
            assert v_v[k] == pytest.approx(v_l[k], rel=RTOL, abs=ATOL)
        # Векторный validate — один батч-прогон на все held-out.
        assert pv.n_forward == 1
        assert pl.n_forward == 1
        # held-out другой длины — по одному (без ошибок, значения совпадают)
        t_short = t_seq[:, :30]
        gen = ConstantRateGenerator(dt=DT, params={"rate": 10.0})
        inputs_short = network._graph.pack_inputs({"in": gen(t_short)})
        target_short = network(t_short, inputs=inputs_short).firing_rates["exc"]
        held_short = Experiment(
            name="held_short",
            t_sequence=t_short,
            inputs=inputs_short,
            objective=_loss(target_short),
        )
        pv2 = MultiExperimentProblem(
            network,
            space,
            ka["experiments"],
            held_out=[held, held_short],
            vectorize=True,
        )
        v2 = pv2.validate(phi)
        assert set(v2) == {"held", "held_short"}


class TestVectorizedCompiled:
    def test_compiled_matches_eager_no_retrace(self, l0_factory):
        pv_c, pv_e = build_pair(
            l0_factory,
            vectorize_a=True,
            vectorize_b=True,
            compiled_a=True,
            compiled_b=False,
        )
        rng = np.random.default_rng(4)
        base = _phi_from_model(pv_e)
        phi1 = base + rng.standard_normal(pv_e.size).astype(np.float32) * 0.01
        phi2 = base - rng.standard_normal(pv_e.size).astype(np.float32) * 0.01
        for _ in range(5):
            for phi in (phi1, phi2):
                ev_c, g_c = pv_c.value_and_gradient(phi)
                ev_e, g_e = pv_e.value_and_gradient(phi)
                assert ev_c.loss == pytest.approx(ev_e.loss, rel=RTOL, abs=ATOL)
                np.testing.assert_allclose(
                    g_c.numpy(), g_e.numpy(), rtol=1e-5, atol=1e-6
                )
        # Ключ кэша по id протоколов экспериментов: без промахов при
        # повторных вызовах (тензоры батч-протокола пересоздаются).
        assert len(pv_c._vcore_cache) == 1
        for core in pv_c._vcore_cache.values():
            # _list_all_concrete_functions — приватный TF API (пин TF 2.21).
            assert len(core.grad_core._list_all_concrete_functions()) == 1

    def test_compiled_with_x0_matches_eager(self, l0_factory):
        """compiled=True + vectorize=True с x0-записями (common и
        exp-specific): градиенты совпадают с eager-vectorize, x0-градиенты
        ненулевые (подстановка x0 внутри графа ядра)."""
        pv_c, pv_e = build_pair(
            l0_factory,
            vectorize_a=True,
            vectorize_b=True,
            compiled_a=True,
            compiled_b=False,
            with_x0=True,
            exp_x0_for="exp2",
        )
        rng = np.random.default_rng(8)
        phi = _phi_from_model(pv_e, jitter=0.02, rng=rng)
        ev_c, g_c = pv_c.value_and_gradient(phi)
        ev_e, g_e = pv_e.value_and_gradient(phi)
        assert ev_c.loss == pytest.approx(ev_e.loss, rel=RTOL, abs=ATOL)
        np.testing.assert_allclose(g_c.numpy(), g_e.numpy(), rtol=1e-5, atol=1e-6)
        names = list(pv_e.parameter_space.phi_entries)
        idx_common = names.index("x0/exc/0")
        idx_exp = names.index("x0/exp2/exc/0")
        assert abs(float(g_c[idx_common])) > 0
        assert abs(float(g_c[idx_exp])) > 0


class TestVectorizedX0Order:
    """Регрессия High-1: результат не зависит от порядка добавления
    x0-записей (exp-specific до common)."""

    @pytest.mark.parametrize("exp_first", [False, True])
    def test_x0_order_matches_loop(self, l0_factory, exp_first):
        def _build():
            network, space, t_seq, inputs1, _ = l0_factory()
            inputs2 = inputs1 * 0.5
            inputs3 = network._graph.pack_inputs(
                {"in": ConstantRateGenerator(dt=DT, params={"rate": 25.0})(t_seq)}
            )
            if exp_first:
                space.add_experiment_initial_state_entries(
                    network, "exp2", include=[("exc", 0)]
                )
                space.add_initial_state_entries(network, include=[("exc", 0)])
            else:
                space.add_initial_state_entries(network, include=[("exc", 0)])
                space.add_experiment_initial_state_entries(
                    network, "exp2", include=[("exc", 0)]
                )
            t1 = network(t_seq, inputs=inputs1).firing_rates["exc"]
            t2 = network(t_seq, inputs=inputs2).firing_rates["exc"]
            t3 = network(t_seq, inputs=inputs3).firing_rates["exc"]
            experiments = [
                Experiment(
                    name="exp1", t_sequence=t_seq, inputs=inputs1, objective=_loss(t1)
                ),
                Experiment(
                    name="exp2", t_sequence=t_seq, inputs=inputs2, objective=_loss(t2)
                ),
                Experiment(
                    name="exp3", t_sequence=t_seq, inputs=inputs3, objective=_loss(t3)
                ),
            ]
            return network, space, experiments

        net_a, space_a, exps_a = _build()
        net_b, space_b, exps_b = _build()
        pv = MultiExperimentProblem(
            net_a, space_a, exps_a, grad_mode="chain", vectorize=True
        )
        pl = MultiExperimentProblem(
            net_b, space_b, exps_b, grad_mode="chain", vectorize=False
        )
        rng = np.random.default_rng(9)
        phi = _phi_from_model(pv, jitter=0.02, rng=rng)
        ev_v, g_v = pv.value_and_gradient(phi)
        ev_l, g_l = pl.value_and_gradient(phi)
        assert ev_v.loss == pytest.approx(ev_l.loss / E, rel=RTOL, abs=ATOL)
        np.testing.assert_allclose(g_v.numpy(), g_l.numpy() / E, rtol=1e-4, atol=1e-5)

    def test_exp_specific_isolation_against_single_experiment(self, l0_factory):
        """Явная изоляция: grad по exp-specific записи == grad задачи,
        состоящей только из этого эксперимента, / E."""
        network, space, t_seq, inputs1, _ = l0_factory()
        inputs2 = inputs1 * 0.5
        space.add_initial_state_entries(network, include=[("exc", 0)])
        space.add_experiment_initial_state_entries(
            network, "exp2", include=[("exc", 0)]
        )
        t1 = network(t_seq, inputs=inputs1).firing_rates["exc"]
        t2 = network(t_seq, inputs=inputs2).firing_rates["exc"]
        experiments = [
            Experiment(
                name="exp1", t_sequence=t_seq, inputs=inputs1, objective=_loss(t1)
            ),
            Experiment(
                name="exp2", t_sequence=t_seq, inputs=inputs2, objective=_loss(t2)
            ),
        ]
        pv = MultiExperimentProblem(
            network, space, experiments, grad_mode="chain", vectorize=True
        )
        # Одноэкспериментная задача с той же exp-specific записью.
        network2, space2, t_seq2, inputs1b, _ = l0_factory()
        inputs2b = inputs1b * 0.5
        space2.add_initial_state_entries(network2, include=[("exc", 0)])
        space2.add_experiment_initial_state_entries(
            network2, "exp2", include=[("exc", 0)]
        )
        t2b = network2(t_seq2, inputs=inputs2b).firing_rates["exc"]
        single = MultiExperimentProblem(
            network2,
            space2,
            [
                Experiment(
                    name="exp2",
                    t_sequence=t_seq2,
                    inputs=inputs2b,
                    objective=_loss(t2b),
                )
            ],
            grad_mode="chain",
            vectorize=False,
        )
        rng = np.random.default_rng(10)
        # Общий φ: координаты common+x0 совпадают по построению
        # (l0_factory детерминирован), поэтому pack-значения одинаковы.
        phi = _phi_from_model(pv, jitter=0.02, rng=rng)
        _, g_v = pv.value_and_gradient(phi)
        _, g_s = single.value_and_gradient(phi[: single.size])
        names = list(pv.parameter_space.phi_entries)
        idx_exp = names.index("x0/exp2/exc/0")
        names_s = list(single.parameter_space.phi_entries)
        idx_exp_s = names_s.index("x0/exp2/exc/0")
        E2 = 2
        assert float(g_v[idx_exp]) == pytest.approx(
            float(g_s[idx_exp_s]) / E2, rel=1e-4, abs=1e-5
        )


class TestVectorizedNoResidual:
    def test_status_and_typeerror(self, l0_factory):
        """objective без residual-интерфейса: status
        'no_residual_interface' и TypeError из residual()."""
        from neuraltide.training import CompositeLoss, MSELoss

        network, space, t_seq, inputs1, _ = l0_factory()
        target = network(t_seq, inputs=inputs1).firing_rates["exc"]
        obj = CompositeLoss([(1.0, MSELoss({"exc": target}))])
        experiments = [
            Experiment(name="e1", t_sequence=t_seq, inputs=inputs1, objective=obj),
        ]
        pv = MultiExperimentProblem(
            network, space, experiments, grad_mode="chain", vectorize=True
        )
        phi = _phi_from_model(pv)
        ev = pv.evaluate(phi)
        assert ev.status == "no_residual_interface"
        assert ev.residuals.shape == (0,)
        with pytest.raises(TypeError, match="residual"):
            pv.residual(phi)


class TestVectorizedValidateHetero:
    def test_mixed_T_heldout_values_match_loop(self, l0_factory):
        """Разнородные held-out (разные T): validate по одному, значения
        совпадают с цикловым режимом."""
        ka = build_experiments(l0_factory)
        network, space = ka["network"], ka["space"]
        t_seq = ka["experiments"][0].t_sequence
        inputs1 = ka["experiments"][0].inputs
        t_short = t_seq[:, :30]
        gen = ConstantRateGenerator(dt=DT, params={"rate": 10.0})
        inputs_short = network._graph.pack_inputs({"in": gen(t_short)})
        target_short = network(t_short, inputs=inputs_short).firing_rates["exc"]
        held_short = Experiment(
            name="held_short",
            t_sequence=t_short,
            inputs=inputs_short,
            objective=_loss(target_short),
        )
        target_held = network(t_seq, inputs=inputs1 * 0.7).firing_rates["exc"]
        held = Experiment(
            name="held",
            t_sequence=t_seq,
            inputs=inputs1 * 0.7,
            objective=_loss(target_held),
        )
        pv = MultiExperimentProblem(
            network,
            space,
            ka["experiments"],
            held_out=[held, held_short],
            vectorize=True,
        )
        kb = build_experiments(l0_factory)
        network2, space2 = kb["network"], kb["space"]
        target_held2 = network2(t_seq, inputs=inputs1 * 0.7).firing_rates["exc"]
        target_short2 = network2(t_short, inputs=inputs_short).firing_rates["exc"]
        pl = MultiExperimentProblem(
            network2,
            space2,
            kb["experiments"],
            held_out=[
                Experiment(
                    name="held",
                    t_sequence=t_seq,
                    inputs=inputs1 * 0.7,
                    objective=_loss(target_held2),
                ),
                Experiment(
                    name="held_short",
                    t_sequence=t_short,
                    inputs=inputs_short,
                    objective=_loss(target_short2),
                ),
            ],
            vectorize=False,
        )
        phi = _phi_from_model(pv, jitter=0.02, rng=np.random.default_rng(11))
        v_v = pv.validate(phi)
        v_l = pl.validate(phi)
        assert set(v_v) == {"held", "held_short"}
        for k in v_v:
            assert v_v[k] == pytest.approx(v_l[k], rel=RTOL, abs=ATOL)


class TestVectorizedAdjointRejected:
    def test_per_step_loss_not_implemented(self, l0_factory):
        ka = build_experiments(l0_factory)
        vobj = _VectorizedObjective(ka["experiments"], ka["network"])
        out = ka["network"](
            ka["experiments"][0].t_sequence, inputs=ka["experiments"][0].inputs
        )
        with pytest.raises(NotImplementedError, match="adjoint"):
            vobj.per_step_loss(out.firing_rates, {}, mask=None)
