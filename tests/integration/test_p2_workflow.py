"""
Интеграционные тесты этапа P2: сквозной рабочий процесс и публичный импорт.

Покрываются:
- smoke-тесты публичного импорта PEP-562: все P2-имена доступны через
  neuraltide.optimization (карта _LAZY_MAP/__all__);
- сквозная цепочка P2 на малой E–I сети (float64, цель зашумлена с sigma
  ObservationSpec, обучаемы exc/tau и inh/tau):
  SPSA (global_search, малый бюджет) -> доводка minimize_lbfgsb ->
  balanced_train (GradNormWeights + PCGrad, smoke) ->
  laplace_approximation -> predictive_intervals -> residual_diagnostics;
  UQ-блок намеренно запускается на свежем экземпляре задачи: forward-режим
  чувствительностей внутри laplace несовместим в пределах одного экземпляра
  NetworkRNN с предшествующими gradient-прогонами (@tf.function-кэш TF,
  см. «Ограничения этапа P2» в docs/optimization.md);
- эквивалентность SerialPool: evaluate_candidates (n_workers=1) даёт те
  же потери, что ручной цикл problem.evaluate.

Время прогона цепочки — порядка 15-30 с на CPU (budget < 2 мин).
"""

import numpy as np
import pytest
import tensorflow as tf

import neuraltide.config
import neuraltide.optimization as opt
from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.inputs import SinusoidalGenerator
from neuraltide.integrators import EulerIntegrator
from neuraltide.populations import WilsonCowan
from neuraltide.synapses import StaticSynapse
from neuraltide.optimization import (
    ALResult,
    AugmentedLagrangianState,
    BalancedResult,
    BootstrapResult,
    CalibrationProblem,
    CandidateTask,
    ConstraintSpec,
    ConstrainedProblem,
    EnsembleResult,
    EvalRecord,
    FixedWeights,
    GlobalOptimizerResult,
    GradNormWeights,
    LaplaceResult,
    NoiseFactorResult,
    ObservationSpec,
    OptionalDependencyError,
    ParameterSpace,
    ParetoResult,
    PredictiveResult,
    ProcessPool,
    ResidualDiagnostics,
    ResidualLoss,
    SelfAdaptiveWeights,
    SensitivityResult,
    SerialPool,
    SpecResidualDiagnostics,
    UncertaintyWeighting,
    WeightStrategy,
    WindowDiagnostics,
    adaptive_window_sampler,
    augmented_lagrangian,
    balanced_gradient,
    balanced_train,
    bootstrap_experiments,
    cagrad_project,
    epsilon_constraint,
    evaluate_candidates,
    fd_jacobian,
    global_search,
    laplace_approximation,
    make_evaluator,
    minimize_lbfgsb,
    noise_sensitivity,
    parameter_ensemble,
    pcgrad_project,
    predictive_intervals,
    residual_bootstrap,
    residual_diagnostics,
)

# Полный список публичных P2-имён (должен совпадать с __all__ пакета).
PUBLIC_P2_NAMES = [
    # global_search
    "global_search",
    "GlobalOptimizerResult",
    "ParetoResult",
    "OptionalDependencyError",
    # robust
    "WeightStrategy",
    "FixedWeights",
    "UncertaintyWeighting",
    "SelfAdaptiveWeights",
    "GradNormWeights",
    "pcgrad_project",
    "cagrad_project",
    "balanced_gradient",
    "AdaptiveWindowFocus",
    "adaptive_window_sampler",
    "balanced_train",
    "BalancedResult",
    # constraints
    "ConstraintSpec",
    "AugmentedLagrangianState",
    "ConstrainedProblem",
    "fd_jacobian",
    "epsilon_constraint",
    "augmented_lagrangian",
    "ALResult",
    # uq
    "residual_diagnostics",
    "SpecResidualDiagnostics",
    "WindowDiagnostics",
    "ResidualDiagnostics",
    "laplace_approximation",
    "LaplaceResult",
    "bootstrap_experiments",
    "residual_bootstrap",
    "BootstrapResult",
    "parameter_ensemble",
    "EnsembleResult",
    "predictive_intervals",
    "PredictiveResult",
    "noise_sensitivity",
    "NoiseFactorResult",
    "SensitivityResult",
    # parallel
    "CandidateTask",
    "EvalRecord",
    "SerialPool",
    "ProcessPool",
    "evaluate_candidates",
    "make_evaluator",
]

# Истинные параметры малой E–I сети цепочки (обе tau обучаемы).
THETA_STAR = {"exc/tau": 8.0, "inh/tau": 12.0}
L2_DT = 0.5
L2_T = 60
L2_SIGMA = 1.0


def build_ei_problem(t_steps: int = L2_T):
    """
    Фабрика E–I задачи калибровки (float64 — вызывать после set_dtype).

    Цель снята при theta* и зашумлена с тем же sigma, что в ObservationSpec:
    MAP-невязки ненулевые (exact-fit — вырожденный случай для residual-
    нормировки и forward-чувствительностей). Обучаемы только постоянные
    времени: синаптические усиления в такой сети по зашумлённым данным
    идентифицируются плохо (почти коллинеарны с входным током), что честно
    отражалось бы раздутыми Laplace-интервалами.
    """
    exc = WilsonCowan(
        dt=L2_DT,
        params={
            "tau": {
                "value": [THETA_STAR["exc/tau"]],
                "trainable": True,
                "min": 1.0,
                "max": 50.0,
            },
            "a": 1.0,
            "theta": 0.0,
            "w": 0.2,
            "I_ext": 0.8,
            "max_rate": 100.0,
        },
        name="exc",
    )
    inh = WilsonCowan(
        dt=L2_DT,
        params={
            "tau": {
                "value": [THETA_STAR["inh/tau"]],
                "trainable": True,
                "min": 1.0,
                "max": 50.0,
            },
            "a": 1.0,
            "theta": 0.0,
            "w": 0.1,
            "I_ext": 0.3,
            "max_rate": 100.0,
        },
        name="inh",
    )
    syn_in = StaticSynapse(
        n_pre=1,
        n_post=1,
        dt=L2_DT,
        params={"gsyn_max": [[30.0]], "pconn": [[1.0]], "e_r": 1.0},
        name="in->exc",
    )
    syn_ie = StaticSynapse(
        n_pre=1,
        n_post=1,
        dt=L2_DT,
        params={"gsyn_max": [[10.0]], "pconn": [[1.0]], "e_r": -0.5},
        name="inh->exc",
    )
    gen = SinusoidalGenerator(
        dt=L2_DT,
        params={"amplitude": 8.0, "freq": 8.0, "phase": 0.0, "offset": 10.0},
    )
    graph = NetworkGraph(dt=L2_DT)
    graph.declare_input("in", n_units=1)
    graph.add_population("exc", exc)
    graph.add_population("inh", inh)
    graph.add_synapse("in->exc", syn_in, src="in", tgt="exc")
    graph.add_synapse("inh->exc", syn_ie, src="inh", tgt="exc")
    network = NetworkRNN(graph, EulerIntegrator())

    t_seq = tf.constant((np.arange(t_steps, dtype=np.float64) * L2_DT)[None, :, None])
    inputs = graph.pack_inputs({"in": gen(t_seq)})
    out = network(t_seq, inputs=inputs)
    rng = np.random.default_rng(123)
    observations = []
    for pop in ("exc", "inh"):
        rates = np.asarray(out.firing_rates[pop], dtype=np.float64)
        target = tf.constant(rates + rng.normal(0.0, L2_SIGMA, rates.shape))
        observations.append(
            ObservationSpec(name=pop, population=pop, target=target, sigma=L2_SIGMA)
        )
    return CalibrationProblem(
        network,
        ParameterSpace.from_network(network),
        t_seq,
        inputs=inputs,
        objective=ResidualLoss(observations),
    )


class TestPublicImports:
    """Smoke: все публичные P2-имена импортируются из пакета (PEP-562)."""

    @pytest.mark.parametrize("name", PUBLIC_P2_NAMES)
    def test_lazy_import_by_name(self, name):
        assert getattr(opt, name) is not None, f"{name} недоступно в пакете"

    def test_names_in_all(self):
        for name in PUBLIC_P2_NAMES:
            assert name in opt.__all__, f"{name} отсутствует в __all__"

    def test_from_import_all(self):
        imported = (
            global_search,
            GlobalOptimizerResult,
            ParetoResult,
            OptionalDependencyError,
            WeightStrategy,
            FixedWeights,
            UncertaintyWeighting,
            SelfAdaptiveWeights,
            GradNormWeights,
            pcgrad_project,
            cagrad_project,
            balanced_gradient,
            adaptive_window_sampler,
            balanced_train,
            BalancedResult,
            ConstraintSpec,
            AugmentedLagrangianState,
            ConstrainedProblem,
            fd_jacobian,
            epsilon_constraint,
            augmented_lagrangian,
            ALResult,
            residual_diagnostics,
            SpecResidualDiagnostics,
            WindowDiagnostics,
            ResidualDiagnostics,
            laplace_approximation,
            LaplaceResult,
            bootstrap_experiments,
            residual_bootstrap,
            BootstrapResult,
            parameter_ensemble,
            EnsembleResult,
            predictive_intervals,
            PredictiveResult,
            noise_sensitivity,
            NoiseFactorResult,
            SensitivityResult,
            CandidateTask,
            EvalRecord,
            SerialPool,
            ProcessPool,
            evaluate_candidates,
            make_evaluator,
        )
        assert all(obj is not None for obj in imported)


class TestP2Workflow:
    """Сквозная цепочка P2 на малой сети (< 2 мин на CPU)."""

    def test_global_local_balanced_uq_chain(self):
        neuraltide.config.set_dtype(tf.float64)
        problem = build_ei_problem()
        space = problem.parameter_space
        space.apply_to_model(
            {
                "exc/tau": tf.constant([[20.0]]),
                "inh/tau": tf.constant([[25.0]]),
            }
        )
        ev0 = problem.evaluate()
        assert ev0.loss > 10.0  # стартовая точка действительно плохая

        # 1. Глобальный поиск малого бюджета (SPSA — bounds не нужны).
        gs = global_search(problem, method="spsa", max_evals=120, seed=7)
        assert isinstance(gs, GlobalOptimizerResult)
        assert gs.n_evals <= 120
        assert np.isfinite(gs.base.loss_best)
        assert gs.base.loss_best < ev0.loss / 10.0
        phi_global = np.asarray(gs.base.phi_best, dtype=np.float64)

        # 2. Доводка L-BFGS-B от лучшей точки глобальной фазы.
        lbfgs = minimize_lbfgsb(problem, phi_global, maxiter=60)
        assert np.isfinite(lbfgs.base.loss_best)
        assert lbfgs.base.loss_best <= gs.base.loss_best + 1e-9
        phi_map = np.asarray(lbfgs.base.phi_best, dtype=np.float64)
        theta_fit = space.unpack(phi_map)
        for name, star in THETA_STAR.items():
            got = float(tf.reshape(theta_fit[name], [-1])[0])
            assert abs(got - star) < 0.5, (name, got, star)

        # 3. Балансировка компонент (smoke): GradNormWeights + PCGrad;
        #    balanced_train мутирует модель — после smoke вернём phi_map.
        bt = balanced_train(
            problem,
            phi_global,
            GradNormWeights(alpha=1.0),
            optimizer="adam",
            max_steps=3,
            projector=pcgrad_project,
            seed=7,
        )
        assert isinstance(bt, BalancedResult)
        assert len(bt.weight_history) == 3
        names = [f"w:{p}" for p in ("exc", "inh")]
        for record in bt.weight_history:
            assert all(n in record for n in names)
            assert all(np.isfinite(record[n]) for n in names)
        assert np.isfinite(bt.base.loss_best)
        assert "best_loss" in bt.baseline
        problem.evaluate(phi_map)  # вернуть модель в MAP-точку

        # 4. Laplace на свежем экземпляре задачи (forward-режим
        #    чувствительностей несовместим с предшествующими
        #    gradient-прогонами в пределах одного NetworkRNN — см.
        #    docs/optimization.md, «Ограничения этапа P2»).
        problem_uq = build_ei_problem()
        lap = laplace_approximation(problem_uq, phi_map, method="gn")
        assert isinstance(lap, LaplaceResult)
        assert lap.method == "gn"
        assert lap.covariance.shape == (problem.size, problem.size)
        assert np.all(np.isfinite(lap.covariance))
        assert np.all(lap.eigvals >= -1e-9)
        for name, star in THETA_STAR.items():
            lo, hi = lap.intervals[name]
            assert lo <= star <= hi, (name, lo, hi, star)

        # 5. Прогнозные интервалы по ансамблю posterior: монотонность
        #    lower <= central <= upper. Полоса — неопределённость параметров
        #    (без шума наблюдений), поэтому сверяется с истинной бесшумной
        #    траекторией при theta*: она должна в основном лежать внутри,
        #    а сама полоса — быть уже шума σ. Прежняя проверка покрытия
        #    зашумлённой цели проходила только при завышенной в √N раз
        #    ковариации (К1 ревью).
        clean = build_ei_problem().simulate(None)  # модель при theta*
        rng = np.random.default_rng(7)
        cov = lap.covariance + 1e-12 * np.eye(lap.phi_map.size)
        # 200 сэмплов: квантили 2.5/97.5 % по 16 сэмплам — почти min/max, и
        # полоса случайно может не накрыть истину, лежащую внутри 95 %-
        # интервала параметров (при seed 7 все 16 сэмплов inh/tau < 12).
        phi_samples = rng.multivariate_normal(lap.phi_map, cov, size=200)
        pred = predictive_intervals(problem_uq, phi_samples, level=0.95)
        assert isinstance(pred, PredictiveResult)
        assert pred.n_samples == 200
        assert pred.n_skipped_nonfinite == 0
        assert set(pred.populations) == {"exc", "inh"}
        for pop in pred.populations:
            lo, mid, hi = pred.lower[pop], pred.central[pop], pred.upper[pop]
            assert lo.shape == mid.shape == hi.shape == (L2_T, 1)
            assert np.all(lo <= mid + 1e-12)
            assert np.all(mid <= hi + 1e-12)
            truth = np.asarray(clean.firing_rates[pop], dtype=np.float64)[0]
            coverage = float(np.mean((lo <= truth) & (truth <= hi)))
            assert coverage >= 0.5, (pop, coverage)
            half_width = float(np.mean(hi - lo)) / 2.0
            assert half_width < L2_SIGMA, (pop, half_width)

        # 6. Диагностика остатков в MAP-точке.
        diag = residual_diagnostics(problem_uq, phi_map, max_lag=5)
        assert isinstance(diag, ResidualDiagnostics)
        assert len(diag.specs) == 2
        assert {s.name for s in diag.specs} == {"exc", "inh"}
        for spec in diag.specs:
            assert np.isfinite(spec.std)
            assert spec.max_abs < 1.0
        assert np.isfinite(diag.max_abs)

        print(
            f"\nP2 chain: loss {ev0.loss:.3g} -> {lbfgs.base.loss_best:.3g} "
            f"(SPSA {gs.n_evals} evals), balanced "
            f"{float(bt.baseline['best_loss']):.3g} -> "
            f"{bt.base.loss_best:.3g}, "
            f"Laplace pd={lap.pd}, "
            f"exc/tau 95%: [{lap.intervals['exc/tau'][0]:.2f}, "
            f"{lap.intervals['exc/tau'][1]:.2f}]"
        )

    def test_serial_pool_matches_direct_evaluation(self):
        """SerialPool (n_workers=1) воспроизводит ручной цикл evaluate."""
        neuraltide.config.set_dtype(tf.float64)
        problem = build_ei_problem()
        space = problem.parameter_space
        space.apply_to_model(
            {
                "exc/tau": tf.constant([[20.0]]),
                "inh/tau": tf.constant([[25.0]]),
            }
        )
        phi0 = np.asarray(problem.pack(), dtype=np.float64)
        rng = np.random.default_rng(3)
        candidates = [
            phi0,
            phi0 + rng.standard_normal(phi0.size) * 0.5,
            phi0 + rng.standard_normal(phi0.size) * 0.5,
        ]

        records = evaluate_candidates(build_ei_problem, candidates, n_workers=1, seed=3)
        assert len(records) == len(candidates)
        assert all(isinstance(r, EvalRecord) for r in records)
        assert [r.index for r in records] == [0, 1, 2]  # ordered=True
        for rec, phi in zip(records, candidates):
            assert rec.error is None
            assert np.isfinite(rec.loss)
            direct = problem.evaluate(np.asarray(phi, dtype=np.float64)).loss
            assert rec.loss == pytest.approx(direct, rel=1e-12, abs=1e-12)
            assert rec.n_forward == 1

        # CandidateTask с явными сидами тоже принимается.
        task = CandidateTask(index=0, phi=candidates[0], seed=3)
        records2 = evaluate_candidates(build_ei_problem, [task], n_workers=1, seed=None)
        assert len(records2) == 1
        assert records2[0].loss == pytest.approx(records[0].loss, rel=1e-12)

        # make_evaluator (n_workers=1) — тот же контракт для global_search.
        evaluator = make_evaluator(build_ei_problem, n_workers=1, seed=3)
        try:
            losses = list(evaluator(candidates))
        finally:
            evaluator.close()
        assert len(losses) == len(candidates)
        for loss, rec in zip(losses, records):
            assert loss == pytest.approx(rec.loss, rel=1e-12, abs=1e-12)
