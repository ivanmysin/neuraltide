"""
neuraltide.optimization — слой калибровки поверх моделей сети.

Состав пакета:

- transforms: дифференцируемые преобразования координат параметров
  (θ ↔ φ) — identity/affine/log/logit/scaled_sigmoid/softplus/bounded_log;
- ParameterSpace: единый реестр оптимизируемых величин с pack/unpack/
  apply_to_model/project; записи начального состояния 'x0/<pop>/<slot>'
  (add_initial_state_entries, StateLocator);
- objectives: ObservationSpec и ResidualLoss — нормированные невязки
  (masked mean, sigma, weight, l2/relative/msle/huber/pseudo_huber) с
  честным residual-интерфейсом 0.5·||r||² == loss;
- problem: CalibrationProblem — единые операции simulate/evaluate/
  value_and_gradient/residual/validate; view-хелперы (time_slice,
  with_objective, with_inputs, with_initial_state) с общими счётчиками;
  Experiment, MultiExperimentProblem (сумма потерь по экспериментам в
  одной GradientTape), FrozenProblem (подпространство φ);
- results: Evaluation, OptimizationResult и производные DTO результатов
  оптимизации (LocalOptimizerResult, StartResult, MultistartResult,
  RunCheckpoint, JacobianResult) с единой JSON-сериализацией;
- diagnostics: FD/Taylor/dot-product/dt/dtype проверки градиентов;
- local: SciPy-адаптеры minimize_lbfgsb / least_squares (TRF/LM) поверх
  CalibrationProblem;
- sensitivities: residual_jvp / residual_jacobian (forward sensitivities),
  value_and_gradient_checkpointed (сегментный прогон + checkpointing),
  truncated_bptt (явно приближённый режим с free-run проверкой);
- shooting: MultipleShootingProblem — сегменты с независимыми начальными
  состояниями и continuity-штрафом (penalty method, finalize-чек);
- strategies: sample_phi0s (Sobol/LHS/random), multistart (screening ->
  доводка top-k, checkpoint/resume), dt_continuation / subset_continuation;
- schedules: Stage, ContinuationScheduler и helpers (horizon_growth,
  teacher_forcing_schedule с обязательным alpha=0, objective_schedule,
  smooth_objective); teacher forcing честен: смешивание только во входах,
  обязательный free_run_check;
- identifiability: compute_fim (FIM = JᵀJ по готовому якобиану),
  likelihood_profile (профиль правдоподобия, χ²-интервалы), select_subset
  (доводка подмножества с оценкой смещения);
- global_search: единый глобальный безградиентный поиск
  (global_search с method='de'|'cma'|'spsa'|'basinhopping'|'nsga2',
  бюджет max_evals, refine_top_k, checkpoint/resume для cma/spsa;
  extras 'global'/'pareto' для cma/pymoo);
- robust: балансировка компонент потерь — WeightStrategy
  (FixedWeights/UncertaintyWeighting/SelfAdaptiveWeights/GradNormWeights),
  PCGrad/CAGrad-проекции, balanced_gradient/balanced_train с обязательным
  baseline, adaptive_window_sampler (focused view по остатку);
- constraints: ограничения на прогнозы — ConstraintSpec,
  ConstrainedProblem (AL-член одной лентой), augmented_lagrangian
  (внешние итерации, checkpoint/resume), epsilon_constraint;
- uq: неопределённость — residual_diagnostics, laplace_approximation
  (GN/FD-Гессиан, маргинальные интервалы), bootstrap_experiments /
  residual_bootstrap, parameter_ensemble, predictive_intervals,
  noise_sensitivity;
- parallel: пакетная оценка кандидатов — CandidateTask/EvalRecord,
  SerialPool, ProcessPool (spawn-процессы), evaluate_candidates,
  make_evaluator (инъекция в global_search).

Имена базовых модулей (transforms/parameters/objectives/problem/results/
diagnostics) импортированы eager-импортом; имена остальных модулей
(local/sensitivities/shooting/strategies/schedules/identifiability/
global_search/robust/constraints/uq/parallel) доступны лениво через
PEP-562 __getattr__ по _LAZY_MAP. Все модули — eager-режим, batch=1
(контракт NetworkRNN); публичный API — через __all__ ниже.
"""

import importlib
from typing import Any, Dict

from neuraltide.optimization.diagnostics import (
    compare_bptt_adjoint,
    dot_product_test,
    dt_convergence,
    dtype_consistency_check,
    finite_difference_scan,
    taylor_test,
)
from neuraltide.optimization.objectives import ObservationSpec, ResidualLoss
from neuraltide.optimization.parameters import ParameterEntry, ParameterSpace
from neuraltide.optimization.problem import CalibrationProblem
from neuraltide.optimization.results import Evaluation, OptimizationResult
from neuraltide.optimization.transforms import (
    TRANSFORM_REGISTRY,
    AffineTransform,
    BoundedLogTransform,
    IdentityTransform,
    LogTransform,
    LogitTransform,
    ScaledSigmoidTransform,
    SoftplusTransform,
    Transform,
    make_transform,
    register_transform,
)

# Карта публичных имён на модули (PEP-562). Существующие имена уже
# импортированы eager-импортом выше; карта служит единым реестром для
# ленивого доступа к остальным именам.
_LAZY_MAP: Dict[str, str] = {
    # transforms
    "Transform": "neuraltide.optimization.transforms",
    "IdentityTransform": "neuraltide.optimization.transforms",
    "AffineTransform": "neuraltide.optimization.transforms",
    "LogTransform": "neuraltide.optimization.transforms",
    "LogitTransform": "neuraltide.optimization.transforms",
    "ScaledSigmoidTransform": "neuraltide.optimization.transforms",
    "SoftplusTransform": "neuraltide.optimization.transforms",
    "BoundedLogTransform": "neuraltide.optimization.transforms",
    "make_transform": "neuraltide.optimization.transforms",
    "register_transform": "neuraltide.optimization.transforms",
    "TRANSFORM_REGISTRY": "neuraltide.optimization.transforms",
    # parameters
    "ParameterEntry": "neuraltide.optimization.parameters",
    "ParameterSpace": "neuraltide.optimization.parameters",
    "StateLocator": "neuraltide.optimization.parameters",
    # objectives
    "ObservationSpec": "neuraltide.optimization.objectives",
    "ResidualLoss": "neuraltide.optimization.objectives",
    # problem
    "CalibrationProblem": "neuraltide.optimization.problem",
    "Experiment": "neuraltide.optimization.problem",
    "MultiExperimentProblem": "neuraltide.optimization.problem",
    "FrozenProblem": "neuraltide.optimization.problem",
    # results
    "Evaluation": "neuraltide.optimization.results",
    "OptimizationResult": "neuraltide.optimization.results",
    "LocalOptimizerResult": "neuraltide.optimization.results",
    "StartResult": "neuraltide.optimization.results",
    "MultistartResult": "neuraltide.optimization.results",
    "RunCheckpoint": "neuraltide.optimization.results",
    "JacobianResult": "neuraltide.optimization.results",
    # diagnostics
    "finite_difference_scan": "neuraltide.optimization.diagnostics",
    "taylor_test": "neuraltide.optimization.diagnostics",
    "dot_product_test": "neuraltide.optimization.diagnostics",
    "dt_convergence": "neuraltide.optimization.diagnostics",
    "dtype_consistency_check": "neuraltide.optimization.diagnostics",
    "compare_bptt_adjoint": "neuraltide.optimization.diagnostics",
    # local
    "minimize_lbfgsb": "neuraltide.optimization.local",
    "least_squares": "neuraltide.optimization.local",
    # sensitivities
    "residual_jvp": "neuraltide.optimization.sensitivities",
    "residual_jacobian": "neuraltide.optimization.sensitivities",
    "value_and_gradient_checkpointed": "neuraltide.optimization.sensitivities",
    "truncated_bptt": "neuraltide.optimization.sensitivities",
    "CheckpointedGradient": "neuraltide.optimization.sensitivities",
    "TruncatedResult": "neuraltide.optimization.sensitivities",
    # shooting
    "ShootingConfig": "neuraltide.optimization.shooting",
    "MultipleShootingProblem": "neuraltide.optimization.shooting",
    "SingleShootingCheck": "neuraltide.optimization.shooting",
    # strategies
    "sample_phi0s": "neuraltide.optimization.strategies",
    "multistart": "neuraltide.optimization.strategies",
    "dt_continuation": "neuraltide.optimization.strategies",
    "subset_continuation": "neuraltide.optimization.strategies",
    "ContinuationResult": "neuraltide.optimization.strategies",
    # schedules
    "Stage": "neuraltide.optimization.schedules",
    "ScheduleResult": "neuraltide.optimization.schedules",
    "ContinuationScheduler": "neuraltide.optimization.schedules",
    "horizon_growth": "neuraltide.optimization.schedules",
    "teacher_forcing_schedule": "neuraltide.optimization.schedules",
    "objective_schedule": "neuraltide.optimization.schedules",
    "smooth_objective": "neuraltide.optimization.schedules",
    # identifiability
    "compute_fim": "neuraltide.optimization.identifiability",
    "likelihood_profile": "neuraltide.optimization.identifiability",
    "select_subset": "neuraltide.optimization.identifiability",
    "FIMResult": "neuraltide.optimization.identifiability",
    "FIMDirection": "neuraltide.optimization.identifiability",
    "CollinearityPair": "neuraltide.optimization.identifiability",
    "ProfileResult": "neuraltide.optimization.identifiability",
    "SubsetResult": "neuraltide.optimization.identifiability",
    # global_search
    "global_search": "neuraltide.optimization.global_search",
    "GlobalOptimizerResult": "neuraltide.optimization.global_search",
    "ParetoResult": "neuraltide.optimization.global_search",
    "OptionalDependencyError": "neuraltide.optimization.global_search",
    # robust
    "WeightStrategy": "neuraltide.optimization.robust",
    "FixedWeights": "neuraltide.optimization.robust",
    "UncertaintyWeighting": "neuraltide.optimization.robust",
    "SelfAdaptiveWeights": "neuraltide.optimization.robust",
    "GradNormWeights": "neuraltide.optimization.robust",
    "pcgrad_project": "neuraltide.optimization.robust",
    "cagrad_project": "neuraltide.optimization.robust",
    "balanced_gradient": "neuraltide.optimization.robust",
    "AdaptiveWindowFocus": "neuraltide.optimization.robust",
    "adaptive_window_sampler": "neuraltide.optimization.robust",
    "balanced_train": "neuraltide.optimization.robust",
    "BalancedResult": "neuraltide.optimization.robust",
    # constraints
    "ConstraintSpec": "neuraltide.optimization.constraints",
    "AugmentedLagrangianState": "neuraltide.optimization.constraints",
    "ConstrainedProblem": "neuraltide.optimization.constraints",
    "fd_jacobian": "neuraltide.optimization.constraints",
    "epsilon_constraint": "neuraltide.optimization.constraints",
    "augmented_lagrangian": "neuraltide.optimization.constraints",
    "ALResult": "neuraltide.optimization.constraints",
    # uq
    "residual_diagnostics": "neuraltide.optimization.uq",
    "SpecResidualDiagnostics": "neuraltide.optimization.uq",
    "WindowDiagnostics": "neuraltide.optimization.uq",
    "ResidualDiagnostics": "neuraltide.optimization.uq",
    "laplace_approximation": "neuraltide.optimization.uq",
    "LaplaceResult": "neuraltide.optimization.uq",
    "bootstrap_experiments": "neuraltide.optimization.uq",
    "residual_bootstrap": "neuraltide.optimization.uq",
    "BootstrapResult": "neuraltide.optimization.uq",
    "parameter_ensemble": "neuraltide.optimization.uq",
    "EnsembleResult": "neuraltide.optimization.uq",
    "predictive_intervals": "neuraltide.optimization.uq",
    "PredictiveResult": "neuraltide.optimization.uq",
    "noise_sensitivity": "neuraltide.optimization.uq",
    "NoiseFactorResult": "neuraltide.optimization.uq",
    "SensitivityResult": "neuraltide.optimization.uq",
    # parallel
    "CandidateTask": "neuraltide.optimization.parallel",
    "EvalRecord": "neuraltide.optimization.parallel",
    "SerialPool": "neuraltide.optimization.parallel",
    "ProcessPool": "neuraltide.optimization.parallel",
    "evaluate_candidates": "neuraltide.optimization.parallel",
    "make_evaluator": "neuraltide.optimization.parallel",
}

__all__ = [
    # transforms
    "Transform",
    "IdentityTransform",
    "AffineTransform",
    "LogTransform",
    "LogitTransform",
    "ScaledSigmoidTransform",
    "SoftplusTransform",
    "BoundedLogTransform",
    "make_transform",
    "register_transform",
    "TRANSFORM_REGISTRY",
    # parameters
    "ParameterEntry",
    "ParameterSpace",
    "StateLocator",
    # objectives
    "ObservationSpec",
    "ResidualLoss",
    # problem / results
    "CalibrationProblem",
    "Experiment",
    "MultiExperimentProblem",
    "FrozenProblem",
    "Evaluation",
    "OptimizationResult",
    "LocalOptimizerResult",
    "StartResult",
    "MultistartResult",
    "RunCheckpoint",
    "JacobianResult",
    # diagnostics
    "finite_difference_scan",
    "taylor_test",
    "dot_product_test",
    "dt_convergence",
    "dtype_consistency_check",
    "compare_bptt_adjoint",
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


def __getattr__(name: str) -> Any:
    """
    Ленивый доступ к именам из _LAZY_MAP (PEP-562).

    Поднимает AttributeError (не ImportError) для неизвестных имён.

    Импорт подмодуля ставит его как атрибут пакета (import-машинерия
    parent.<child> = module), что теневит одноимённую функцию
    (модуль global_search.py теневит global_search()): обращение к
    GlobalOptimizerResult импортировало бы модуль, и последующий
    from neuraltide.optimization import global_search дал бы модуль.
    Поэтому после импорта кэшируем ВСЕ публичные имена этого модуля.
    """
    module_name = _LAZY_MAP.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name)
    value = getattr(module, name)
    for public_name, mapped_module in _LAZY_MAP.items():
        if mapped_module == module_name:
            globals()[public_name] = getattr(module, public_name)
    return value


def __dir__() -> list:
    return sorted(set(globals()) | set(__all__))
