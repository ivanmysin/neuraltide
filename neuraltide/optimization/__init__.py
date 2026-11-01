"""
neuraltide.optimization — слой калибровки поверх моделей сети.

Этап P0 дорожной карты оптимизации:

- transforms: дифференцируемые преобразования координат параметров
  (θ ↔ φ) — identity/affine/log/logit/scaled_sigmoid/softplus/bounded_log;
- ParameterSpace: единый реестр оптимизируемых величин с pack/unpack/
  apply_to_model/project;
- objectives: ObservationSpec и ResidualLoss — нормированные невязки
  (masked mean, sigma, weight, l2/relative/msle/huber/pseudo_huber) с
  честным residual-интерфейсом 0.5·||r||² == loss;
- problem: CalibrationProblem — единые операции simulate/evaluate/
  value_and_gradient/residual для всех будущих оптимизаторов;
- results: Evaluation и OptimizationResult — единые форматы результатов;
- diagnostics: FD/Taylor/dot-product/dt/dtype проверки градиентов.

SciPy-оптимизаторы, мультистарт, checkpointing и multiple shooting —
следующие этапы дорожной карты (docs/optimization_roadmap.md).
"""

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
    # objectives
    "ObservationSpec",
    "ResidualLoss",
    # problem / results
    "CalibrationProblem",
    "Evaluation",
    "OptimizationResult",
    # diagnostics
    "finite_difference_scan",
    "taylor_test",
    "dot_product_test",
    "dt_convergence",
    "dtype_consistency_check",
    "compare_bptt_adjoint",
]
