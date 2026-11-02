"""
Continuation scheduler и расписания (curriculum) калибровки (этап P1,
ранг 12 дорожной карты).

- Stage: именованная стадия — задача (view CalibrationProblem поверх
  той же сети/ParameterSpace), оптимизатор ('lbfgsb' | 'least_squares' |
  callable) и meta;
- ContinuationScheduler(stages).run(problem): последовательный запуск
  стадий с переносом phi (unpack/pack общего ParameterSpace view-задач),
  историей loss по стадиям и обязательной финальной оценкой
  free_run_check на ОРИГИНАЛЬНОЙ задаче (без teacher forcing и прочих
  искажений протокола);
- helpers: horizon_growth (короткий горизонт -> длинный),
  teacher_forcing_schedule (смешивание входов (1-alpha)*u +
  alpha*target_rates с обязательным финальным alpha=0),
  objective_schedule (feature/smooth -> pointwise loss),
  smooth_objective (скользящее среднее целевых траекторий ResidualLoss).

Teacher forcing honesty: смешивание выполняется ТОЛЬКО во входах
inputs; последний alpha обязан быть 0 (иначе ValueError ещё на этапе
построения расписания и на этапе запуска scheduler'а); free_run_check —
повторная оценка финальной phi на оригинальных входах; alpha_history
сохраняется в результате.

Ограничение: scheduler работает со стадиями над ОДНИМ ParameterSpace
(view-задачи time_slice/with_objective/with_inputs); цепочки FrozenProblem
— через subset_continuation из strategies.py.
"""

import dataclasses
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import tensorflow as tf

import neuraltide.config
from neuraltide.core.types import cast_if_needed
from neuraltide.optimization.objectives import ResidualLoss
from neuraltide.optimization.results import (
    Evaluation,
    LocalOptimizerResult,
    _JsonDTO,
)
from neuraltide.optimization.strategies import resolve_local_optimizer

__all__ = [
    "Stage",
    "ScheduleResult",
    "ContinuationScheduler",
    "horizon_growth",
    "teacher_forcing_schedule",
    "objective_schedule",
    "smooth_objective",
]


# ---------------------------------------------------------------------------
# Stage / ScheduleResult
# ---------------------------------------------------------------------------


@dataclass
class Stage:
    """
    Одна стадия continuation-расписания.

    Атрибуты:
        name: имя стадии (в history результата);
        problem: задача стадии — обычно view базовой CalibrationProblem
            (time_slice/with_objective/with_inputs) с тем же
            ParameterSpace; pack/evaluate принимают полный phi;
        optimizer: 'lbfgsb' | 'least_squares' | callable(problem, phi0);
        maxiter: бюджет итераций для строкового адаптера (в local.py);
        meta: произвольные метаданные; ключ 'alpha' зарезервирован для
            teacher forcing (см. teacher_forcing_schedule).
    """

    name: str
    problem: Any
    optimizer: Union[str, Callable] = "lbfgsb"
    maxiter: int = 200
    meta: Dict[str, Any] = field(default_factory=dict)


def _jsonable(value: Any) -> Any:
    """Рекурсивное приведение к JSON-сериализуемому виду."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _result_to_dict(result: Any) -> Any:
    to_dict = getattr(result, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    return {"repr": repr(result)}


def _result_from_dict(data: Any) -> Any:
    if isinstance(data, Mapping) and "base" in data:
        return LocalOptimizerResult.from_dict(data)
    return data


def _evaluation_to_dict(evaluation: Optional[Evaluation]) -> Any:
    return None if evaluation is None else evaluation.to_dict()


def _evaluation_from_dict(data: Any) -> Optional[Evaluation]:
    if data is None:
        return None
    return Evaluation(
        loss=float(data.get("loss", 0.0)),
        components={str(k): float(v) for k, v in data.get("components", {}).items()},
        residuals=np.asarray(data.get("residuals", []), dtype=np.float64),
        grad=None if data.get("grad") is None else np.asarray(data["grad"]),
        status=str(data.get("status", "ok")),
        n_forward=int(data.get("n_forward", 0)),
        n_gradient=int(data.get("n_gradient", 0)),
        wall_time=float(data.get("wall_time", 0.0)),
        dtype=str(data.get("dtype", "float32")),
        dt=float(data.get("dt", 0.0)),
    )


@dataclass
class ScheduleResult(_JsonDTO):
    """
    Итог ContinuationScheduler.run.

    Атрибуты:
        stage_results: результаты оптимизаторов по стадиям;
        final_result: результат последней стадии;
        alpha_history: коэффициенты teacher forcing по стадиям (пустой
            список, если расписание без teacher forcing);
        free_run_check: повторная оценка финальной phi на ОРИГИНАЛЬНОЙ
            задаче (Evaluation) — обязательно не None после run();
        history: краткая история {stage, optimizer, loss, wall_time,
            meta};
        wall_time: суммарное время, секунды.
    """

    stage_results: List[Any] = field(default_factory=list)
    final_result: Any = None
    alpha_history: List[float] = field(default_factory=list)
    free_run_check: Optional[Evaluation] = None
    history: List[Dict[str, Any]] = field(default_factory=list)
    wall_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_results": [_result_to_dict(r) for r in self.stage_results],
            "final_result": _result_to_dict(self.final_result),
            "alpha_history": [float(a) for a in self.alpha_history],
            "free_run_check": _evaluation_to_dict(self.free_run_check),
            "history": _jsonable(self.history),
            "wall_time": float(self.wall_time),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ScheduleResult":
        return cls(
            stage_results=[_result_from_dict(d) for d in data.get("stage_results", [])],
            final_result=_result_from_dict(data.get("final_result")),
            alpha_history=[float(a) for a in data.get("alpha_history", [])],
            free_run_check=_evaluation_from_dict(data.get("free_run_check")),
            history=list(data.get("history", [])),
            wall_time=float(data.get("wall_time", 0.0)),
        )


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class ContinuationScheduler:
    """
    Последовательный запуск стадий continuation-расписания.

    run(problem):
      1. стартует из problem.pack() (текущее состояние модели);
      2. для каждой стадии вызывает оптимизатор (строка -> адаптер local.py
         с maxiter=stage.maxiter; callable -> stage.optimizer(problem,
         phi0)) и переносит phi = result.base.phi_best между стадиями
         (view-задачи делят ParameterSpace, поэтому перенос — тождественный
         unpack->pack);
      3. если у стадий есть meta['alpha'] (teacher forcing), последний
         alpha обязан быть 0.0 — иначе ValueError ДО запуска;
      4. завершается обязательным free_run_check: problem.evaluate(phi)
         на оригинальной задаче и её исходных входах.

    Счётчики n_forward/n_gradient и wall_time накапливаются в общих
    счётчиках view-задач и базовой задачи.
    """

    def __init__(self, stages: Sequence[Stage]):
        self.stages = list(stages)
        if not self.stages:
            raise ValueError("ContinuationScheduler: непустой stages обязателен.")
        for stage in self.stages:
            if not isinstance(stage, Stage):
                raise TypeError(
                    "ContinuationScheduler: элементы stages должны быть "
                    f"Stage, получен {type(stage).__name__}."
                )

    def _run_stage(self, stage: Stage, phi: np.ndarray):
        if callable(stage.optimizer):
            result = stage.optimizer(stage.problem, phi)
            opt_name = str(getattr(stage.optimizer, "__name__", repr(stage.optimizer)))
        else:
            fn, opt_name = resolve_local_optimizer(stage.optimizer)
            result = fn(stage.problem, phi, maxiter=int(stage.maxiter))
        return result, opt_name

    def run(self, problem) -> ScheduleResult:
        """
        Выполняет расписание поверх базовой задачи problem.

        problem — оригинальная (не искажённая) задача: используется для
        начальной точки и обязательного free_run_check на исходных
        входах/объективе.
        """
        alphas = [
            float(stage.meta["alpha"]) for stage in self.stages if "alpha" in stage.meta
        ]
        if alphas and alphas[-1] != 0.0:
            raise ValueError(
                "ContinuationScheduler: расписание с teacher forcing "
                f"заканчивается на alpha={alphas[-1]} != 0; замкнутая "
                "сеть обязана быть финальной стадией."
            )
        t_begin = time.perf_counter()
        phi = np.asarray(problem.pack(), dtype=np.float64).reshape(-1)
        stage_results: List[Any] = []
        history: List[Dict[str, Any]] = []
        for stage in self.stages:
            t0 = time.perf_counter()
            result, opt_name = self._run_stage(stage, phi)
            phi = np.asarray(result.base.phi_best, dtype=np.float64).reshape(-1)
            stage_results.append(result)
            history.append(
                {
                    "stage": str(stage.name),
                    "optimizer": opt_name,
                    "loss": float(result.base.loss_best),
                    "wall_time": time.perf_counter() - t0,
                    "meta": dict(stage.meta),
                }
            )
        # Free run: оригинальная задача на исходных входах.
        free_run_check = problem.evaluate(phi)
        return ScheduleResult(
            stage_results=stage_results,
            final_result=stage_results[-1],
            alpha_history=alphas,
            free_run_check=free_run_check,
            history=history,
            wall_time=time.perf_counter() - t_begin,
        )


# ---------------------------------------------------------------------------
# Helpers: расписания стадий
# ---------------------------------------------------------------------------


def _slice_mask(mask, target, start: int, stop: int):
    """
    Срез маски по оси времени [start:stop), согласованный с target.

    Маска с полноценной осью времени — ранг совпадает с target и длина
    оси времени равна длине target'а (например [batch, T, n] или
    [1, T, n]) — срезается по axis=1; маска рангом на единицу меньше с
    осью времени 0 (например [T, n], [T, 1]) — по axis=0. Time-invariant
    и прочие broadcast-маски без оси времени (например [batch, 1, n],
    [1, 1, n], [n]) передаются без изменений: их broadcast-семантика на
    срезе та же, что и на полном горизонте.
    """
    if mask is None:
        return None
    m_shape = mask.shape
    t_shape = target.shape
    t_rank = t_shape.rank
    t_time = t_shape[1] if t_rank is not None and t_rank >= 2 else None
    if m_shape.rank is None or t_time is None:
        return mask
    if (
        m_shape.rank == t_rank
        and m_shape.rank >= 2
        and m_shape[1] is not None
        and int(m_shape[1]) == int(t_time)
    ):
        return mask[:, start:stop]
    if (
        m_shape.rank == t_rank - 1
        and m_shape.rank >= 1
        and m_shape[0] is not None
        and int(m_shape[0]) == int(t_time)
    ):
        return mask[start:stop]
    return mask


def _slice_objective(objective, start: int, stop: int):
    """
    Срез целевых траекторий и масок objective по времени [start:stop).

    Требует ResidualLoss-подобный objective с атрибутом observations
    (иначе TypeError): цели срезаются, маски — через _slice_mask (с осью
    времени срезаются, time-invariant/broadcast — как есть), спецификации
    пересобираются через dataclasses.replace.
    """
    observations = getattr(objective, "observations", None)
    if observations is None:
        raise TypeError(
            "horizon_growth: objective должен предоставлять 'observations' "
            "для автоматического среза целей; для прочих objective'ов "
            "соберите Stage вручную."
        )
    sliced = [
        dataclasses.replace(
            spec,
            target=spec.target[:, start:stop],
            mask=_slice_mask(spec.mask, spec.target, start, stop),
        )
        for spec in observations
    ]
    return type(objective)(sliced)


def horizon_growth(base_problem, horizons, *, build_inputs=None) -> List[Stage]:
    """
    Расписание «короткий горизонт -> длинный».

    Args:
        base_problem: исходная CalibrationProblem (целевые траектории
            берутся из её objective — ResidualLoss-like);
        horizons: последовательность стоп-индексов (int, старт 0) или
            кортежей (start, stop); каждый stop <= T, start < stop;
        build_inputs: опциональный callable(inputs, start, stop) ->
            входы среза; по умолчанию inputs[:, start:stop] (как у
            time_slice).

    Returns:
        Список Stage с view-задачами base_problem.time_slice(...)
        (с пересобранным objective под срез).
    """
    t_seq = base_problem.t_sequence
    total = int(t_seq.shape[1])
    spans: List[tuple] = []
    for horizon in horizons:
        if isinstance(horizon, (tuple, list)):
            start, stop = int(horizon[0]), int(horizon[1])
        else:
            start, stop = 0, int(horizon)
        if not (0 <= start < stop <= total):
            raise ValueError(
                f"horizon_growth: недопустимый горизонт ({start}, {stop}) "
                f"при T={total}."
            )
        spans.append((start, stop))
    stages: List[Stage] = []
    for start, stop in spans:
        view = base_problem.time_slice(start, stop)
        if build_inputs is not None:
            view = view.with_inputs(build_inputs(base_problem.inputs, start, stop))
        view = view.with_objective(
            _slice_objective(base_problem.objective, start, stop)
        )
        stages.append(
            Stage(
                name=f"horizon[{start}:{stop}]",
                problem=view,
                meta={"start": start, "stop": stop, "horizon": stop},
            )
        )
    return stages


def teacher_forcing_schedule(base_problem, target_rates, alphas) -> List[Stage]:
    """
    Расписание teacher forcing «открытая сеть -> замкнутая».

    Смешивание выполняется ТОЛЬКО во входах:
    inputs_alpha = (1 - alpha) * u + alpha * target_rates.
    Последний alpha обязан быть 0.0 (иначе ValueError) — финальная стадия
    работает на исходных входах; честность результата обеспечивает
    free_run_check scheduler'а на оригинальной задаче.

    Args:
        base_problem: исходная CalibrationProblem (inputs не None);
        target_rates: целевые пресинаптические частоты, форма ровно
            [batch, T, total_input_units] (как у base_problem.inputs);
        alphas: непустая последовательность в [0, 1] с alphas[-1] == 0.

    Returns:
        Список Stage с view-задачами base_problem.with_inputs(mixed) и
        meta['alpha'].
    """
    if base_problem.inputs is None:
        raise ValueError(
            "teacher_forcing_schedule: base_problem.inputs is None — "
            "смешивание входов невозможно."
        )
    alphas = [float(a) for a in alphas]
    if not alphas:
        raise ValueError("teacher_forcing_schedule: непустой alphas обязателен.")
    for alpha in alphas:
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"teacher_forcing_schedule: alpha вне [0, 1]: {alpha}.")
    if alphas[-1] != 0.0:
        raise ValueError(
            "teacher_forcing_schedule: последний alpha обязан быть 0 "
            f"(замкнутая сеть), получен {alphas[-1]}."
        )
    inputs = tf.convert_to_tensor(base_problem.inputs)
    target = cast_if_needed(tf.convert_to_tensor(target_rates), inputs.dtype)
    if target.shape.as_list() != inputs.shape.as_list():
        raise ValueError(
            "teacher_forcing_schedule: target_rates формы "
            f"{target.shape.as_list()} не совпадает с inputs "
            f"{inputs.shape.as_list()}."
        )
    dtype = neuraltide.config.get_dtype()
    stages: List[Stage] = []
    for alpha in alphas:
        mixed = cast_if_needed((1.0 - alpha) * inputs + alpha * target, dtype)
        stages.append(
            Stage(
                name=f"teacher_forcing[a={alpha:g}]",
                problem=base_problem.with_inputs(mixed),
                meta={"alpha": alpha},
            )
        )
    return stages


def objective_schedule(base_problem, objectives) -> List[Stage]:
    """
    Расписание «сначала один objective -> затем другой» (например,
    feature-based -> pointwise): каждая стадия — view с новым objective.
    """
    objectives = list(objectives)
    if not objectives:
        raise ValueError("objective_schedule: непустой objectives обязателен.")
    stages: List[Stage] = []
    for index, objective in enumerate(objectives):
        if not callable(objective):
            raise TypeError(
                "objective_schedule: objective должен быть callable, "
                f"получен {type(objective).__name__}."
            )
        stages.append(
            Stage(
                name=f"objective[{index}]",
                problem=base_problem.with_objective(objective),
                meta={"objective": index},
            )
        )
    return stages


def smooth_objective(objective, window: int) -> ResidualLoss:
    """
    Сглаженная копия ResidualLoss-подобного objective.

    Целевые траектории всех ObservationSpec сглаживаются скользящим
    средним по времени (axis=1) с окном window и padding 'SAME'
    (краевые значения усредняются с нулевым паддингом — лёгкий краевой
    сдвиг; это инициализационная стадия, финальная доводка идёт на
    исходных целях). Спецификации пересобираются через
    dataclasses.replace, тип objective сохраняется.

    Raises:
        TypeError: objective без атрибута observations (не
            ResidualLoss-подобный) — честный отказ вместо silently
            обёртки, меняющей постановку.
        ValueError: window < 2.
    """
    observations = getattr(objective, "observations", None)
    if observations is None:
        raise TypeError(
            "smooth_objective: требуется ResidualLoss-подобный objective "
            "с атрибутом 'observations'; прочие objective'ы не "
            "поддерживаются (сглаживание меняет постановку задачи)."
        )
    window = int(window)
    if window < 2:
        raise ValueError(f"smooth_objective: window должен быть >= 2, {window}.")
    smoothed = []
    for spec in observations:
        target = tf.convert_to_tensor(spec.target)
        smooth_target = tf.nn.avg_pool1d(
            target, ksize=window, strides=1, padding="SAME"
        )
        smoothed.append(dataclasses.replace(spec, target=smooth_target))
    return type(objective)(smoothed)
