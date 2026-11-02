"""
Интеграционные тесты этапа P1: сквозной рабочий процесс и публичный импорт.

Покрываются:
- сквозная цепочка на L0-сети (фикстура conftest l0_factory в float64):
  multistart (Sobol -> screening -> доводка least_squares/TRF) ->
  residual_jacobian (forward sensitivities) -> compute_fim ->
  likelihood_profile по exc/tau -> честный free-run прогон;
  ограничение: в L0-фикстуре синапс in->exc имеет e_r=0.0, поэтому
  gsyn_max не идентифицируем (post_v отсутствует у WilsonCowan) —
  восстановление проверяется по tau/I_ext, а идентифицируемость
  gsyn_max честно отражается в FIM/профиле;
- smoke-тесты публичного импорта PEP-562: все P1-имена доступны через
  neuraltide.optimization (карта _LAZY_MAP/__all__).

Время прогона цепочки — порядка 30-60 с на CPU (budget < 2 мин).
"""

import numpy as np
import pytest
import tensorflow as tf

import neuraltide.config
import neuraltide.optimization as opt
from neuraltide.optimization import (
    CalibrationProblem,
    CollinearityPair,
    ContinuationResult,
    ContinuationScheduler,
    Experiment,
    FIMDirection,
    FIMResult,
    FrozenProblem,
    JacobianResult,
    LocalOptimizerResult,
    MultistartResult,
    MultipleShootingProblem,
    ProfileResult,
    RunCheckpoint,
    ScheduleResult,
    ShootingConfig,
    SingleShootingCheck,
    Stage,
    StartResult,
    SubsetResult,
    TruncatedResult,
    CheckpointedGradient,
    compute_fim,
    dt_continuation,
    horizon_growth,
    least_squares,
    likelihood_profile,
    minimize_lbfgsb,
    multistart,
    objective_schedule,
    residual_jacobian,
    residual_jvp,
    sample_phi0s,
    select_subset,
    smooth_objective,
    subset_continuation,
    teacher_forcing_schedule,
    truncated_bptt,
    value_and_gradient_checkpointed,
)

# Полный список публичных P1-имён (должен совпадать с __all__ пакета).
PUBLIC_P1_NAMES = [
    # local
    "minimize_lbfgsb",
    "least_squares",
    # sensitivities
    "residual_jvp",
    "residual_jacobian",
    "value_and_gradient_checkpointed",
    "truncated_bptt",
    "CheckpointedGradient",
    "TruncatedResult",
    # shooting
    "ShootingConfig",
    "MultipleShootingProblem",
    "SingleShootingCheck",
    # strategies
    "sample_phi0s",
    "multistart",
    "dt_continuation",
    "subset_continuation",
    "ContinuationResult",
    # schedules
    "Stage",
    "ScheduleResult",
    "ContinuationScheduler",
    "horizon_growth",
    "teacher_forcing_schedule",
    "objective_schedule",
    "smooth_objective",
    # identifiability
    "compute_fim",
    "likelihood_profile",
    "select_subset",
    "FIMResult",
    "FIMDirection",
    "CollinearityPair",
    "ProfileResult",
    "SubsetResult",
    # CORE results / problem (DTO этапа P1)
    "LocalOptimizerResult",
    "StartResult",
    "MultistartResult",
    "RunCheckpoint",
    "JacobianResult",
    "Experiment",
    "MultiExperimentProblem",
    "FrozenProblem",
]


class TestPublicImports:
    """Smoke: все публичные P1-имена импортируются из пакета (PEP-562)."""

    @pytest.mark.parametrize("name", PUBLIC_P1_NAMES)
    def test_lazy_import_by_name(self, name):
        assert getattr(opt, name) is not None, f"{name} недоступно в пакете"

    def test_names_in_all(self):
        for name in PUBLIC_P1_NAMES:
            assert name in opt.__all__, f"{name} отсутствует в __all__"

    def test_from_import_all(self):
        # Проверка, что from-importы сверху файла собрались: каждое
        # импортированное имя используется здесь (не None).
        imported = (
            minimize_lbfgsb,
            least_squares,
            residual_jvp,
            residual_jacobian,
            value_and_gradient_checkpointed,
            truncated_bptt,
            CheckpointedGradient,
            TruncatedResult,
            ShootingConfig,
            MultipleShootingProblem,
            SingleShootingCheck,
            sample_phi0s,
            multistart,
            dt_continuation,
            subset_continuation,
            ContinuationResult,
            Stage,
            ScheduleResult,
            ContinuationScheduler,
            horizon_growth,
            teacher_forcing_schedule,
            objective_schedule,
            smooth_objective,
            compute_fim,
            likelihood_profile,
            select_subset,
            FIMResult,
            FIMDirection,
            CollinearityPair,
            ProfileResult,
            SubsetResult,
            LocalOptimizerResult,
            StartResult,
            MultistartResult,
            RunCheckpoint,
            JacobianResult,
            Experiment,
            FrozenProblem,
        )
        assert all(obj is not None for obj in imported)


class TestP1Workflow:
    """Сквозная цепочка P1 на малой сети (< 2 мин на CPU)."""

    def test_multistart_jacobian_fim_profile_chain(self, l0_factory):
        neuraltide.config.set_dtype(tf.float64)
        network, space, t_seq, inputs, loss = l0_factory()
        problem = CalibrationProblem(
            network, space, t_seq, inputs=inputs, objective=loss
        )
        # Сдвиг от "истинных" значений конструктора фикстуры.
        theta_star = {"exc/tau": 8.0, "exc/I_ext": 0.8, "in->exc/gsyn_max": 0.05}
        space.apply_to_model(
            {
                "exc/tau": tf.constant([[20.0]]),
                "exc/I_ext": tf.constant([[0.2]]),
                "in->exc/gsyn_max": tf.constant([[0.2]]),
            }
        )
        ev0 = problem.evaluate()
        assert ev0.loss > 1.0  # стартовая точка действительно плохая

        # 1. Мультистарт: screening -> доводка top-k (TRF).
        ms = multistart(
            problem,
            n_starts=6,
            method="sobol",
            refine_top_k=2,
            local="least_squares",
            seed=17,
        )
        assert isinstance(ms, MultistartResult)
        assert ms.n_starts == 6
        assert ms.n_refined == 2
        assert ms.best_index >= 0
        assert ms.base.loss_best < 1e-6, (
            f"мультистарт не сошёлся: loss={ms.base.loss_best}"
        )
        phi_best = np.asarray(ms.base.phi_best, dtype=np.float64)

        # Восстановление идентифицируемых параметров (gsyn_max в L0 с
        # e_r=0.0 молчит — он не восстанавливается, это задокументировано).
        theta_fit = space.unpack(phi_best)
        for name in ("exc/tau", "exc/I_ext"):
            got = float(tf.reshape(theta_fit[name], [-1])[0])
            assert abs(got - theta_star[name]) < 1e-3, (name, got)

        # 2. Якобиан невязок (forward sensitivities) в точке решения.
        jac = residual_jacobian(problem, phi_best)
        assert isinstance(jac, JacobianResult)
        assert jac.J.ndim == 2
        assert jac.J.shape[1] == problem.size
        assert jac.n_forward > 0
        # Якобиан согласован с невязками: loss == 0.5 * ||r||^2.
        assert jac.loss == pytest.approx(0.5 * float(jac.r @ jac.r), rel=1e-9)

        # 3. FIM = J^T J: спектр, обусловленность, чувствительности.
        fim = compute_fim(jac, param_names=space.names)
        assert isinstance(fim, FIMResult)
        assert fim.fim.shape == (problem.size, problem.size)
        # FIM симметрична и положительно полуопределена.
        np.testing.assert_allclose(fim.fim, fim.fim.T, atol=1e-12)
        assert np.all(fim.eigenvalues >= -1e-12)
        assert 1 <= fim.effective_rank <= problem.size
        assert fim.condition_number > 0
        assert len(fim.param_sensitivity) == problem.size

        # 4. Профиль правдоподобия по exc/tau (гауссов χ²-интервал).
        profile = likelihood_profile(
            problem,
            "exc/tau",
            grid=np.linspace(5.0, 11.0, 5),
            optimizer="lbfgsb",
            warm_start=True,
        )
        assert isinstance(profile, ProfileResult)
        assert len(profile.theta_values) == 5
        assert sum(profile.successes) >= 4, (
            f"слишком много неуспешных узлов: {profile.successes}"
        )
        assert profile.intervals["0.95"] is not None
        lo, hi = profile.intervals["0.95"]
        assert lo <= theta_star["exc/tau"] <= hi, (lo, hi)
        # Профильный минимум близок к оптимуму доводки.
        ok_losses = [lv for lv, s in zip(profile.losses, profile.successes) if s]
        assert min(ok_losses) <= ms.base.loss_best + 1e-6

        # 5. Честный free-run: прогон без teacher forcing на исходных входах.
        ev_free = problem.evaluate(phi_best)
        assert ev_free.status == "ok"
        assert ev_free.loss == pytest.approx(ms.base.loss_best, rel=1e-6)

        # Метрики для отчёта (печатаются только при -s или падении).
        print(
            f"\nP1 chain: loss {ev0.loss:.3g} -> {ms.base.loss_best:.3g}, "
            f"n_forward={problem.n_forward}, "
            f"FIM rank={fim.effective_rank}/{problem.size}, "
            f"kappa={fim.condition_number:.3g}, "
            f"profile 95%: [{lo:.2f}, {hi:.2f}]"
        )

    def test_frozen_fixed_values_keep_float64(self, l0_factory):
        """Регрессия: fixed-значения FrozenProblem не квантуются в float32."""
        neuraltide.config.set_dtype(tf.float64)
        network, space, t_seq, inputs, loss = l0_factory()
        problem = CalibrationProblem(
            network, space, t_seq, inputs=inputs, objective=loss
        )
        value64 = 8.1234567890123
        frozen = FrozenProblem(
            problem, free_names=["exc/I_ext"], fixed={"exc/tau": value64}
        )
        theta = space.unpack(frozen._compose(frozen.pack()))
        got = float(tf.reshape(theta["exc/tau"], [-1])[0])
        assert abs(got - value64) < 1e-12
