"""
CalibrationProblem: единая постановка задачи калибровки.

Инкапсулирует модель (NetworkRNN), пространство параметров (ParameterSpace),
протокол (t_sequence, inputs, initial_state) и целевую функцию (objective),
предоставляя одинаковые операции для всех будущих оптимизаторов
(Keras, SciPy, безградиентные методы):

    simulate(phi)           — прямой прогон сети;
    evaluate(phi)           — скалярная потеря, компоненты и невязки;
    value_and_gradient(phi) — потеря и градиент по φ;
    residual(phi)           — вектор нормированных невязок для GN/LM/TRF;
    validate(phi)           — потери на held-out экспериментах.

Дополнительно к базовым операциям:

    * tensor-backed записи ParameterSpace (kind 'initial_state'/
      'shooting_state', см. ParameterSpace.add_initial_state_entries):
      начальное состояние материализуется как дифференцируемая функция
      от φ — θ_x0 = transform.forward(кусок φ) подставляется в
      network.call(initial_state=...); вклад в градиент в chain-режиме —
      tape.gradient(loss, x0_tensor) * transform.jacobian (для model_var
      путь прежний: градиент по переменным × jacobian);
    * эксперимент-специфичные x0-записи 'x0/<experiment>/<population>/
      <slot>' (см. ParameterSpace.add_experiment_initial_state_entries):
      в MultiExperimentProblem материализуются у эксперимента с
      совпадающим именем с приоритетом над общими 'x0/<pop>/<slot>'
      (двухпроходная подстановка); общие model-параметры и общий x0
      участвуют во всех экспериментах, вклады общего x0 в градиент
      суммируются по экспериментам, экспериментные x0 между экспериментами
      не смешиваются; вне экспериментного контекста (CalibrationProblem,
      shooting, sensitivities) exp-specific записи пропускаются с
      одноразовым UserWarning;
    * view-хелперы time_slice/with_objective/with_inputs/with_initial_state,
      возвращающие новую CalibrationProblem с ОБЩИМИ счётчиками;
    * Experiment — протокол одного эксперимента; held_out=[Experiment]
      в конструкторе задачи — отложенные протоколы только для validate;
    * MultiExperimentProblem — одна сеть + один ParameterSpace, потеря
      сумма по экспериментам в одной GradientTape, residual конкатенация,
      компоненты с префиксом '{exp}/';
    * FrozenProblem — подпространство φ (free_names/fixed) поверх той же
      задачи для continuation/profiles/subset.

Контракт мутации модели
-----------------------
phi=None: модель не изменяется — оценка выполняется при текущих значениях
    переменных (начальное φ получают через pack()). Для пространств с
    tensor-записями pack(None) невозможен (x0 не хранится в модели) —
    требуется явный phi.
phi != None: модель явно переводится в состояние θ = unpack(phi) перед
    прогоном и остаётся в нём после вызова (model_var → переменные,
    tensor-записи → начальное состояние прогона). Чтобы сохранить и
    восстановить состояние, используйте
    ``saved = problem.pack()`` и
    ``parameter_space.apply_to_model(parameter_space.unpack(saved))``.

Режимы градиента (grad_mode)
----------------------------
'assign' (по умолчанию): literal-путь tape.watch(phi) → ... допустим
    только когда нет записей model_var с переменными (иначе градиент
    через variable.assign недоступен/неполон) — тогда иначе цепочка.
'chain': dloss/dθ по переменным модели и dloss/dx0 по подставленным
    тензорам начального состояния, затем умножение на dθ/dφ (jacobian
    transforms). Применение φ к модели перед прогоном обязательно в
    обоих режимах: иначе значение потери и её градиент окажутся вычислены
    в разных точках.

Счётчики n_forward / n_gradient и накопленный wall_time хранятся в
разделяемом словаре self._counters: view-задачи и FrozenProblem делят
счётчики с базовой задачей. CalibrationProblem считает каждый прогон
сети за единицу n_forward; MultiExperimentProblem — каждый прогон
каждого эксперимента.

Ограничения: по умолчанию eager-режим (compiled=False — прежнее
поведение без изменений); compiled=True — value_and_gradient/evaluate/
residual идут через скомпилированное tf.function-ядро (см. CalibrationProblem,
этап 1b), мутация модели/счётчики/Evaluation/simulate/validate остаются
eager; batch=1 (контракт NetworkRNN); записи ParameterSpace kind 'model_var'
без привязанной переменной не включаются в градиент — их вклад
тождественно нулевой. Все φ-операции задач (bounds_phi/search_box_phi,
материализация x0, FrozenProblem) работают только с trainable-записями
ParameterSpace (``phi_entries``/``trainable_names``): non-trainable
записи — реестровые константы вне вектора φ, они не оптимизируются и
не влияют на его размерность.
"""

import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf

import neuraltide.config
from neuraltide.core.network import NetworkOutput, NetworkRNN
from neuraltide.core.types import TensorType, cast_if_needed
from neuraltide.optimization._compiled import _CompiledCore
from neuraltide.optimization._state import (
    InitialState,
    _TENSOR_KINDS,
    _broadcast_or_validate_init,
    _concat_base_states,
    _make_vectorized_x0_substitute,
    _materialize_batched_initial_state,
    _materialize_initial_state,
    _phi_gradient,
    _x0_experiment,
)

# Реэкспорты для обратной совместимости: sensitivities/shooting/robust и
# тесты импортируют эти имена из problem (явный alias — намеренный
# реэкспорт).
from neuraltide.optimization._state import (
    _X0_OUTSIDE_CONTEXT_WARNED as _X0_OUTSIDE_CONTEXT_WARNED,
)
from neuraltide.optimization._state import _chain_gradient as _chain_gradient
from neuraltide.optimization.parameters import ParameterSpace
from neuraltide.optimization.results import Evaluation

__all__ = [
    "CalibrationProblem",
    "Experiment",
    "MultiExperimentProblem",
    "FrozenProblem",
]

_GRAD_MODES = ("assign", "chain")


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


@dataclass
class Experiment:
    """
    Один эксперимент (протокол + данные) для калибровки.

    Атрибуты:
        name: уникальное имя эксперимента (префикс компонент
            MultiExperimentProblem: '{name}/{component}');
        t_sequence: временная последовательность [batch, T, 1] или
            [batch, T] (мс);
        inputs: упакованные внешние входы [batch, T, total_input_units]
            или None, если входов нет;
        objective: callable(predictions, network) -> scalar; для
            обучающих экспериментов MultiExperimentProblem обязателен;
            None — при validate используется objective задачи;
        initial_state: опциональное начальное состояние протокола
            (pop, syn) — база, в которую подставляются x0-записи
            ParameterSpace: общие 'x0/<pop>/<slot>' и эксперимент-
            специфичные 'x0/<name>/<pop>/<slot>' (только у эксперимента
            с совпадающим name, с приоритетом над общими).
    """

    name: str
    t_sequence: TensorType
    inputs: Optional[TensorType] = None
    objective: Optional[Any] = None
    initial_state: InitialState = None


# ---------------------------------------------------------------------------
# Модульные хелперы (общие для Calibration/Multi/Frozen)
# ---------------------------------------------------------------------------


def _bounds_phi_from_space(space: ParameterSpace) -> Tuple[np.ndarray, np.ndarray]:
    """Границы вектора φ из физических min/max trainable-записей ParameterSpace."""
    lower: list = []
    upper: list = []
    dtype = neuraltide.config.get_dtype()
    for entry in space.phi_entries.values():
        lo, hi = -np.inf, np.inf
        if entry.min is not None:
            lo = float(entry.transform.inverse(tf.constant(entry.min, dtype)))
        if entry.max is not None:
            hi = float(entry.transform.inverse(tf.constant(entry.max, dtype)))
        lower.extend([lo] * entry.numel)
        upper.extend([hi] * entry.numel)
    return (
        np.asarray(lower, dtype=np.float64),
        np.asarray(upper, dtype=np.float64),
    )


# Отступ поискового бокса глобальных методов от физических границ (доля span).
SEARCH_BOX_REL_MARGIN = 0.01


def _search_box_phi_from_space(
    space: ParameterSpace, rel_margin: float = SEARCH_BOX_REL_MARGIN
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Поисковый бокс φ для глобальных методов (DE, CMA-ES, ...).

    По trainable-записям ParameterSpace. bounds_phi берёт inverse(min/max):
    у sigmoid-образных преобразований
    это граница отступа машинной точности (±11 во float32, ±31 во float64),
    у log с min=0 — log(tiny). Равномерная выборка в таком боксе почти вся
    ложится к краям физического диапазона и зависит от dtype. Здесь бокс
    строится по θ ∈ [min + δ·span, max − δ·span], δ = rel_margin, и
    переводится в φ; записи без одной из границ дают ±inf (глобальные
    методы с конечными границами их отвергают).
    """
    if not 0.0 <= float(rel_margin) < 0.5:
        raise ValueError(
            f"search box: rel_margin должен быть в [0, 0.5), got {rel_margin}."
        )
    lower: list = []
    upper: list = []
    dtype = neuraltide.config.get_dtype()
    for entry in space.phi_entries.values():
        lo, hi = -np.inf, np.inf
        if entry.min is not None and entry.max is not None:
            span = float(entry.max) - float(entry.min)
            lo_t = float(entry.min) + float(rel_margin) * span
            hi_t = float(entry.max) - float(rel_margin) * span
            lo = float(entry.transform.inverse(tf.constant(lo_t, dtype)))
            hi = float(entry.transform.inverse(tf.constant(hi_t, dtype)))
            lo, hi = min(lo, hi), max(lo, hi)
        elif entry.min is not None or entry.max is not None:
            lo_b, hi_b = _bounds_phi_from_space_entry(entry, dtype)
            lo, hi = lo_b, hi_b
        lower.extend([lo] * entry.numel)
        upper.extend([hi] * entry.numel)
    return (
        np.asarray(lower, dtype=np.float64),
        np.asarray(upper, dtype=np.float64),
    )


def _bounds_phi_from_space_entry(entry, dtype) -> Tuple[float, float]:
    """bounds_phi одной записи (для односторонних границ)."""
    lo, hi = -np.inf, np.inf
    if entry.min is not None:
        lo = float(entry.transform.inverse(tf.constant(entry.min, dtype)))
    if entry.max is not None:
        hi = float(entry.transform.inverse(tf.constant(entry.max, dtype)))
    return lo, hi


def _make_evaluation(
    *,
    loss: float,
    components: Optional[Dict[str, float]],
    residuals: Optional[np.ndarray],
    grad: Optional[tf.Tensor],
    status: str,
    counters: Dict[str, Any],
    elapsed: float,
    dt: float,
) -> Evaluation:
    """Сборка Evaluation с фиксацией счётчиков и wall time."""
    counters["wall_time"] += elapsed
    return Evaluation(
        loss=float(loss),
        components={} if components is None else dict(components),
        residuals=np.zeros(0) if residuals is None else np.asarray(residuals),
        grad=None if grad is None else np.asarray(grad),
        status=status,
        n_forward=counters["n_forward"],
        n_gradient=counters["n_gradient"],
        wall_time=elapsed,
        dtype=str(neuraltide.config.get_dtype().name),
        dt=float(dt),
    )


def _components_of(objective, output, network, prefix: str = "") -> Dict[str, float]:
    components_fn = getattr(objective, "components", None)
    if components_fn is None:
        return {}
    return {f"{prefix}{k}": float(v) for k, v in components_fn(output, network).items()}


def _residual_of(objective, output) -> Optional[np.ndarray]:
    residual_fn = getattr(objective, "residual", None)
    if residual_fn is None:
        return None
    return np.asarray(residual_fn(output))


# ---------------------------------------------------------------------------
# CalibrationProblem
# ---------------------------------------------------------------------------


class CalibrationProblem:
    """
    Задача калибровки: сеть + параметры + протокол + целевая функция.

    Args:
        network: NetworkRNN.
        parameter_space: ParameterSpace, построенный от этой сети
            (ParameterSpace.from_network, опционально с
            add_initial_state_entries).
        t_sequence: временная последовательность [batch, T, 1] или
            [batch, T] (мс).
        inputs: упакованные внешние входы [batch, T, total_input_units]
            или None, если входов нет.
        objective: callable(predictions, network) -> scalar —
            ResidualLoss, CompositeLoss, BaseLoss или пользовательская
            функция; None допускается, если objective будет задан позже
            (присваиванием problem.objective); методы вычисления
            потребуют его понятной ошибкой.
        initial_state: опциональное начальное состояние протокола
            (pop, syn) — база для x0-записей ParameterSpace.
        grad_mode: 'assign' (по умолчанию) или 'chain' — см. модульную
            документацию.
        held_out: опциональная последовательность Experiment —
            отложенные протоколы только для validate().
        chunk_length: None (default) — целый прогон одним сканом; иначе
            чанкованный прогон сегментами длины chunk_length с памятью
            градиента O(chunk_length) (см. NetworkRNN.call). Несовместим с
            grad_mode='adjoint' (adjoint имеет собственную память O(1)).
        chunk_grad: 'checkpointed' (default, точный BPTT с пересчётом
            сегментов) | 'truncated' (дёшево, градиент смещён).
        compiled: если True, value_and_gradient / evaluate / residual
            выполняются через скомпилированное tf.function-ядро
            (этап 1b): граф прогона+ленты+градиента трассируется один
            раз на (objective × протокол времени/входов) и кэшируется.
            Мутация модели (assign φ → θ), счётчики и numpy-сборка
            Evaluation остаются снаружи в eager; simulate() и
            validate() — eager в обоих режимах. Явное initial_state
            протокола — аргумент ядра, материализуется снаружи при
            каждом вызове: in-place смена problem.initial_state
            отражается немедленно; view-хелперы
            (time_slice/with_objective/...) по-прежнему создают новые
            ядра лениво. Ключ кэша ядер — id объектов (задача держит
            ссылки); мутирование objective по месту ядром не
            отслеживается.

    Счётчики n_forward / n_gradient и накопленный wall_time хранятся в
    разделяемом self._counters и копируются в каждый Evaluation.
    """

    def __init__(
        self,
        network: NetworkRNN,
        parameter_space: ParameterSpace,
        t_sequence: TensorType,
        inputs: Optional[TensorType] = None,
        objective: Optional[Any] = None,
        initial_state: InitialState = None,
        grad_mode: str = "assign",
        held_out: Optional[Sequence[Experiment]] = None,
        compiled: bool = False,
        chunk_length: Optional[int] = None,
        chunk_grad: str = "checkpointed",
    ):
        if grad_mode not in _GRAD_MODES:
            raise ValueError(
                f"CalibrationProblem: unknown grad_mode '{grad_mode}'. "
                f"Available modes: {list(_GRAD_MODES)}."
            )
        if grad_mode == "adjoint" and chunk_length is not None:
            raise ValueError(
                "CalibrationProblem: chunk_length с grad_mode='adjoint' "
                "не поддержан (adjoint имеет собственную память O(1); "
                "смешение не реализовано). Используйте grad_mode='bptt' "
                "или chunk_length=None."
            )
        if chunk_length is not None and chunk_grad not in (
            "checkpointed",
            "truncated",
        ):
            raise ValueError(
                f"CalibrationProblem: unknown chunk_grad '{chunk_grad}'; "
                "expected 'checkpointed' or 'truncated'."
            )
        if objective is not None and not callable(objective):
            raise TypeError(
                "CalibrationProblem: 'objective' must be callable "
                f"(predictions, network) -> scalar, got "
                f"{type(objective).__name__}."
            )
        self.network = network
        self.parameter_space = parameter_space
        self.t_sequence = t_sequence
        self.inputs = inputs
        self.objective = objective
        self.initial_state = initial_state
        self.grad_mode = grad_mode
        self.held_out: List[Experiment] = list(held_out or [])
        self.compiled = bool(compiled)
        self.chunk_length = chunk_length
        self.chunk_grad = chunk_grad
        # Кэш compiled-ядер: ключ (experiment, id(objective), id(t),
        # id(inputs), chunk_length, chunk_grad) — см. _get_compiled_core.
        # Начальное состояние протокола — аргумент ядра, материализуется
        # при каждом вызове.
        self._core_cache: Dict[Any, _CompiledCore] = {}

        self._counters: Dict[str, Any] = {
            "n_forward": 0,
            "n_gradient": 0,
            "wall_time": 0.0,
        }

    # ------------------------------------------------------------------
    # Счётчики (view-задачи делят self._counters)
    # ------------------------------------------------------------------
    @property
    def n_forward(self) -> int:
        return self._counters["n_forward"]

    @n_forward.setter
    def n_forward(self, value: int) -> None:
        self._counters["n_forward"] = int(value)

    @property
    def n_gradient(self) -> int:
        return self._counters["n_gradient"]

    @n_gradient.setter
    def n_gradient(self, value: int) -> None:
        self._counters["n_gradient"] = int(value)

    @property
    def wall_time(self) -> float:
        return self._counters["wall_time"]

    @wall_time.setter
    def wall_time(self, value: float) -> None:
        self._counters["wall_time"] = float(value)

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------
    def _require_objective(self) -> None:
        if self.objective is None:
            raise ValueError(
                "CalibrationProblem: objective is None. Передайте "
                "objective=... в конструкторе или присвойте "
                "problem.objective = ... перед вызовом."
            )

    # ------------------------------------------------------------------
    # pack
    # ------------------------------------------------------------------
    @property
    def size(self) -> int:
        """Размерность вектора φ."""
        return self.parameter_space.size

    def pack(self, phi: Optional[Any] = None) -> tf.Tensor:
        """
        φ-вектор в координатах ParameterSpace.

        phi=None — текущее значение переменных модели; иначе phi
        проверяется на размер и приводится к глобальному dtype. Для
        пространств с tensor-записями (начальное состояние) phi=None
        невозможен — x0 не хранится в модели; передайте явный phi.
        """
        if phi is None:
            return self.parameter_space.pack()
        dtype = neuraltide.config.get_dtype()
        phi_t = cast_if_needed(tf.convert_to_tensor(phi), dtype)
        phi_t = tf.reshape(phi_t, [-1])
        if phi_t.shape[0] is not None and int(phi_t.shape[0]) != self.size:
            raise ValueError(
                f"CalibrationProblem: phi has {int(phi_t.shape[0])} "
                f"elements, expected {self.size}."
            )
        tf.debugging.assert_equal(
            tf.size(phi_t), self.size, message="CalibrationProblem: bad phi size"
        )
        return phi_t

    # ------------------------------------------------------------------
    # Ядро: прямой прогон
    # ------------------------------------------------------------------
    def _apply_phi(self, phi_t: tf.Tensor) -> None:
        """Модель := θ = unpack(phi) (явное намерение, см. контракт)."""
        theta = self.parameter_space.unpack(phi_t)
        self.parameter_space.apply_to_model(theta)

    def _prepare(self, phi: Optional[Any]) -> tf.Tensor:
        """pack (+apply) без прогона; увеличивает n_forward."""
        phi_t = self.pack(phi)
        if phi is not None:
            self._apply_phi(phi_t)
        self._counters["n_forward"] += 1
        return phi_t

    def _forward(self, phi: Optional[Any]) -> Tuple[tf.Tensor, NetworkOutput]:
        """pack (+apply) и прямой прогон; увеличивает n_forward."""
        phi_t = self._prepare(phi)
        theta = self.parameter_space.unpack(phi_t)
        initial_state, _ = _materialize_initial_state(
            self.network, self.parameter_space, theta, self.initial_state
        )
        initial_state = _broadcast_or_validate_init(
            initial_state,
            tf.shape(self.t_sequence)[0],
            what="CalibrationProblem",
        )
        output = self.network(
            self.t_sequence,
            inputs=self.inputs,
            initial_state=initial_state,
            chunk_length=self.chunk_length,
            chunk_grad=self.chunk_grad,
        )
        return phi_t, output

    # ------------------------------------------------------------------
    # compiled-ядро (compiled=True, см. __init__)
    # ------------------------------------------------------------------
    def _materialized_base_init(self) -> Tuple[Tuple, Tuple]:
        """Протокольное начальное состояние полными (pop, syn) кортежами.

        Без кэша (eager-конструирование тензоров, дёшево): пересоздаётся
        при каждом вызове, поэтому in-place смена problem.initial_state
        отражается немедленно. x0-записи ParameterSpace подставляются
        поверх внутри ядра — здесь только база.
        """
        if self.initial_state is None:
            pop, syn = self.network.get_initial_state(batch_size=1)
            return (tuple(pop), tuple(syn))
        return (tuple(self.initial_state[0]), tuple(self.initial_state[1]))

    def _get_compiled_core(
        self,
        *,
        objective,
        t_sequence,
        inputs,
        experiment: Optional[str] = None,
    ) -> _CompiledCore:
        """Ленивое (построение при первом вызове) получение ядра из кэша.

        Ключ — id объектов (задача держит ссылки на objective/протокол,
        поэтому id стабилен); view-хелперы заменяют объекты → новый ключ
        → новое ядро. Начальное состояние протокола НЕ входит в ключ —
        оно аргумент ядра и материализуется снаружи при каждом вызове.
        Мутирование objective по месту ядром не отслеживается.
        """
        key = (
            experiment,
            id(objective),
            id(t_sequence),
            id(inputs),
            self.chunk_length,
            self.chunk_grad,
        )
        core = self._core_cache.get(key)
        if core is None:
            core = _CompiledCore(
                self.network,
                self.parameter_space,
                t_sequence,
                inputs,
                objective,
                experiment_name=experiment,
                grad_mode=self.grad_mode,
                chunk_length=self.chunk_length,
                chunk_grad=self.chunk_grad,
            )
            core.build()
            self._core_cache[key] = core
        return core

    def _build_compiled_evaluation(
        self,
        loss_t: tf.Tensor,
        comps: Dict[str, tf.Tensor],
        residual_t: tf.Tensor,
        grad: Optional[tf.Tensor],
        t0: float,
    ) -> Evaluation:
        """numpy-сборка Evaluation из тензорных выходов ядра (eager)."""
        has_residual = getattr(self.objective, "residual", None) is not None
        components = {k: float(v) for k, v in comps.items()}
        if has_residual:
            residuals = np.asarray(residual_t)
            status = "ok"
        else:
            residuals = np.zeros(0)
            status = "no_residual_interface"
        return _make_evaluation(
            loss=float(loss_t),
            components=components,
            residuals=residuals,
            grad=grad,
            status=status,
            counters=self._counters,
            elapsed=time.perf_counter() - t0,
            dt=float(self.network._graph.dt),
        )

    # ------------------------------------------------------------------
    # Публичные операции
    # ------------------------------------------------------------------
    def simulate(self, phi: Optional[Any] = None) -> NetworkOutput:
        """
        Прямой прогон сети.

        phi=None — при текущих значениях модели; phi != None — модель
        сначала переводится в θ = unpack(phi) и остаётся в нём (см.
        контракт мутации в документации модуля).
        """
        _, output = self._forward(phi)
        return output

    def evaluate(self, phi: Optional[Any] = None) -> Evaluation:
        """Скалярная потеря, компоненты и невязки (без градиента)."""
        self._require_objective()
        t0 = time.perf_counter()
        if self.compiled:
            phi_t = self._prepare(phi)
            base_init = self._materialized_base_init()
            core = self._get_compiled_core(
                objective=self.objective,
                t_sequence=self.t_sequence,
                inputs=self.inputs,
            )
            loss_t, comps, residual_t = core.eval_core(phi_t, *base_init)
            return self._build_compiled_evaluation(
                loss_t, comps, residual_t, grad=None, t0=t0
            )
        _, output = self._forward(phi)
        eval_result = self._build_evaluation(output, grad=None, t0=t0)
        return eval_result

    def value_and_gradient(
        self, phi: Optional[Any] = None
    ) -> Tuple[Evaluation, tf.Tensor]:
        """
        Потеря и градиент по φ одним прогоном.

        Начальное состояние материализуется внутри GradientTape как
        дифференцируемая функция от φ (см. модульную документацию).

        Returns:
            (Evaluation с заполненным grad, grad — tf.Tensor формы [size]).
        """
        self._require_objective()
        t0 = time.perf_counter()
        if self.compiled:
            phi_t = self._prepare(phi)
            base_init = self._materialized_base_init()
            core = self._get_compiled_core(
                objective=self.objective,
                t_sequence=self.t_sequence,
                inputs=self.inputs,
            )
            loss_t, grad_t, comps, residual_t = core.grad_core(phi_t, *base_init)
            self._counters["n_gradient"] += 1
            eval_result = self._build_compiled_evaluation(
                loss_t, comps, residual_t, grad=grad_t, t0=t0
            )
            return eval_result, grad_t
        phi_t = self._prepare(phi)
        # persistent: assign-режим сначала пробует literal-путь по phi,
        # и при None переиспользует ту же ленту для цепного вычисления.
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(phi_t)
            theta = self.parameter_space.unpack(phi_t)
            initial_state, x0_map = _materialize_initial_state(
                self.network, self.parameter_space, theta, self.initial_state
            )
            initial_state = _broadcast_or_validate_init(
                initial_state,
                tf.shape(self.t_sequence)[0],
                what="CalibrationProblem",
            )
            output = self.network(
                self.t_sequence,
                inputs=self.inputs,
                initial_state=initial_state,
                chunk_length=self.chunk_length,
                chunk_grad=self.chunk_grad,
            )
            loss_t = self.objective(output, self.network)
        grad_t = _phi_gradient(
            self.parameter_space, self.grad_mode, tape, loss_t, phi_t, x0_map
        )
        # persistent-лента больше не нужна — явно освобождаем.
        del tape
        self._counters["n_gradient"] += 1
        eval_result = self._build_evaluation(
            output, grad=grad_t, t0=t0, loss_override=loss_t
        )
        return eval_result, grad_t

    def residual(self, phi: Optional[Any] = None) -> np.ndarray:
        """
        Вектор нормированных невязок (numpy) для GN/LM/TRF.

        Требует objective с методом residual (например ResidualLoss), такой
        что 0.5 * ||r||^2 == loss.
        """
        self._require_objective()
        residual_fn = getattr(self.objective, "residual", None)
        if residual_fn is None:
            raise TypeError(
                "CalibrationProblem: objective "
                f"'{type(self.objective).__name__}' не предоставляет "
                "residual(predictions). Для least-squares постановки "
                "используйте ResidualLoss; иначе ограничьтесь evaluate / "
                "value_and_gradient."
            )
        if self.compiled:
            phi_t = self._prepare(phi)
            base_init = self._materialized_base_init()
            core = self._get_compiled_core(
                objective=self.objective,
                t_sequence=self.t_sequence,
                inputs=self.inputs,
            )
            _, _, residual_t = core.eval_core(phi_t, *base_init)
            return np.asarray(residual_t)
        _, output = self._forward(phi)
        r = residual_fn(output)
        return np.asarray(r)

    def validate(self, phi: Optional[Any] = None) -> Dict[str, float]:
        """
        Потери на held-out экспериментах (конструктор: held_out=[...]).

        phi=None не трогает модель (оценка при текущих значениях);
        phi != None переводит модель в θ = unpack(phi) один раз, далее
        прогоны всех held-out экспериментов. Каждый прогон увеличивает
        n_forward; objective берётся из эксперимента, при None —
        problem.objective.

        Returns:
            {experiment.name: loss}.
        """
        if not self.held_out:
            return {}
        phi_t = self.pack(phi)
        if phi is not None:
            self._apply_phi(phi_t)
        theta = self.parameter_space.unpack(phi_t)
        results: Dict[str, float] = {}
        t0 = time.perf_counter()
        for exp in self.held_out:
            objective = exp.objective if exp.objective is not None else self.objective
            if objective is None:
                raise ValueError(
                    f"CalibrationProblem: held-out experiment '{exp.name}' "
                    "has no objective and problem.objective is None."
                )
            initial_state, _ = _materialize_initial_state(
                self.network, self.parameter_space, theta, exp.initial_state
            )
            initial_state = _broadcast_or_validate_init(
                initial_state,
                tf.shape(exp.t_sequence)[0],
                what=f"held-out experiment '{exp.name}'",
            )
            output = self.network(
                exp.t_sequence,
                inputs=exp.inputs,
                initial_state=initial_state,
                chunk_length=self.chunk_length,
                chunk_grad=self.chunk_grad,
            )
            results[exp.name] = float(objective(output, self.network))
            self._counters["n_forward"] += 1
        self._counters["wall_time"] += time.perf_counter() - t0
        return results

    # ------------------------------------------------------------------
    # View-хелперы (общие счётчики, тот же network/parameter_space)
    # ------------------------------------------------------------------
    def _spawn(self, **overrides: Any) -> "CalibrationProblem":
        """Новая CalibrationProblem поверх тех же объектов с общими counters."""
        kwargs: Dict[str, Any] = dict(
            network=self.network,
            parameter_space=self.parameter_space,
            t_sequence=self.t_sequence,
            inputs=self.inputs,
            objective=self.objective,
            initial_state=self.initial_state,
            grad_mode=self.grad_mode,
            held_out=self.held_out,
            compiled=self.compiled,
            chunk_length=self.chunk_length,
            chunk_grad=self.chunk_grad,
        )
        kwargs.update(overrides)
        view = CalibrationProblem(**kwargs)
        view._counters = self._counters
        return view

    def time_slice(self, start: int, stop: int) -> "CalibrationProblem":
        """
        View на временной срез [start:stop) той же задачи.

        Срезаются t_sequence и inputs; initial_state — начальное
        состояние протокола — сохраняется. Целевая траектория объективно
        срезается пользователем через with_objective.
        """
        t_seq = self.t_sequence[:, start:stop]
        inputs = None if self.inputs is None else self.inputs[:, start:stop]
        return self._spawn(t_sequence=t_seq, inputs=inputs)

    def with_objective(self, objective: Optional[Any]) -> "CalibrationProblem":
        """View с другой целевой функцией (тот же протокол)."""
        return self._spawn(objective=objective)

    def with_inputs(self, inputs: Optional[TensorType]) -> "CalibrationProblem":
        """View с другими упакованными входами (тот же протокол)."""
        return self._spawn(inputs=inputs)

    def with_initial_state(self, initial_state: InitialState) -> "CalibrationProblem":
        """View с другим начальным состоянием протокола (база для x0)."""
        return self._spawn(initial_state=initial_state)

    # ------------------------------------------------------------------
    # Вспомогательные bounds в координатах φ
    # ------------------------------------------------------------------
    def bounds_phi(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Границы вектора φ из физических min/max trainable-записей
        ParameterSpace (non-trainable записи вне φ не учитываются).

        Записи без границ дают -inf / +inf. Границы преобразуются через
        inverse соответствующего transform'а; предназначены для
        SciPy-адаптеров (L-BFGS-B / TRF) — сами по себе задачу не
        ограничивают.
        """
        return _bounds_phi_from_space(self.parameter_space)

    def search_box_phi(
        self, rel_margin: float = SEARCH_BOX_REL_MARGIN
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Поисковый бокс φ для глобальных методов: θ ∈ [min + δ·span,
        max − δ·span] в φ-координатах (не зависит от dtype и машинного
        отступа inverse у границ). См. _search_box_phi_from_space.
        """
        return _search_box_phi_from_space(self.parameter_space, rel_margin)

    # ------------------------------------------------------------------
    # Сборка Evaluation
    # ------------------------------------------------------------------
    def _build_evaluation(
        self,
        output: NetworkOutput,
        grad: Optional[tf.Tensor],
        t0: float,
        loss_override: Optional[tf.Tensor] = None,
    ) -> Evaluation:
        if loss_override is not None:
            loss_t = loss_override
        else:
            loss_t = self.objective(output, self.network)

        components = _components_of(self.objective, output, self.network)

        residuals = _residual_of(self.objective, output)
        if residuals is not None:
            status = "ok"
        else:
            residuals = np.zeros(0)
            status = "no_residual_interface"

        return _make_evaluation(
            loss=float(loss_t),
            components=components,
            residuals=residuals,
            grad=grad,
            status=status,
            counters=self._counters,
            elapsed=time.perf_counter() - t0,
            dt=float(self.network._graph.dt),
        )


# ---------------------------------------------------------------------------
# _VectorizedObjective — objective-обёртка векторного режима MultiExperiment
# ---------------------------------------------------------------------------


class _VectorizedObjective:
    """
    Objective-обёртка векторного режима MultiExperimentProblem: батч-выход
    сети нарезается по слотам, каждый под-объектив получает срез своего
    эксперимента.

    Семантика потерь — СРЕДНЕЕ по экспериментам (mean_e), в отличие от
    циклового режима (сумма): loss_vec = (1/E)·Σ_e loss_e. Градиент по
    общим параметрам — среднее градиентов (эквивалентно весам 1/E в
    цикловом режиме).

    Args:
        experiments: обучающие Experiment (порядок = порядок слотов батча).
        network: NetworkRNN (прокидывается в под-объективы).

    Методы:
        __call__(output, model) -> mean_e objective_e(slice_e);
        components(output, model) -> {'{name}/{k}': v};
        residual(output) -> concat_e residual_e(slice_e); если хотя бы у
            одного под-объектива нет residual — has_residual=False и
            residual() поднимает TypeError (как цикловой режим);
        per_step_loss — НЕ поддержан (adjoint несовместим с векторным
            режимом): поднимает NotImplementedError.

    Внимание: stability_loss в срезах — батч-агрегат (общий для всех
    слотов): при stability_penalty_weight > 0 градиент штрафа смешивает
    слоты батча (кросс-слотовая утечка). Используйте vectorize=True с
    stability_penalty_weight=0 (default).
    """

    def __init__(self, experiments, network):
        self.experiments = list(experiments)
        self.network = network
        self._residual_fns = [
            getattr(e.objective, "residual", None) for e in self.experiments
        ]
        self.has_residual = all(fn is not None for fn in self._residual_fns)

    @staticmethod
    def _slice(output: NetworkOutput, slot: int) -> NetworkOutput:
        """Срез слота батча как отдельный NetworkOutput."""
        return NetworkOutput(
            firing_rates={
                k: v[slot : slot + 1] for k, v in output.firing_rates.items()
            },
            stability_loss=output.stability_loss,
            final_state=(
                tuple(t[slot : slot + 1] for t in output.final_state[0]),
                tuple(t[slot : slot + 1] for t in output.final_state[1]),
            ),
        )

    def __call__(self, output, model):
        n = len(self.experiments)
        total = None
        for i, exp in enumerate(self.experiments):
            loss_e = exp.objective(self._slice(output, i), model)
            total = loss_e if total is None else total + loss_e
        return total / float(n)

    def components(self, output, model):
        comps = {}
        for i, exp in enumerate(self.experiments):
            fn = getattr(exp.objective, "components", None)
            if fn is None:
                continue
            for k, v in fn(self._slice(output, i), model).items():
                comps[f"{exp.name}/{k}"] = v
        return comps

    def residual(self, output):
        if not self.has_residual:
            raise TypeError(
                "MultiExperimentProblem (vectorize=True): objective "
                "не предоставляет residual(predictions)."
            )
        return tf.concat(
            [fn(self._slice(output, i)) for i, fn in enumerate(self._residual_fns)],
            axis=0,
        )

    def per_step_loss(self, *args, **kwargs):
        raise NotImplementedError(
            "Векторный режим MultiExperimentProblem (vectorize=True) "
            "несовместим с adjoint-методом: per-step разложения потерь "
            "по батч-срезам нет."
        )


# ---------------------------------------------------------------------------
# MultiExperimentProblem
# ---------------------------------------------------------------------------


class MultiExperimentProblem:
    """
    Несколько экспериментов над одной сетью и одним ParameterSpace.

    Потеря — сумма потерь обучающих экспериментов (experiments), каждый
    со своим протоколом/входами/целевой функцией; value_and_gradient
    выполняет все прогоны в одной GradientTape, поэтому градиенты общих
    параметров суммируются по протоколам автоматически. residual —
    конкатенация невязок обучающих экспериментов; компоненты несут
    префикс '{exp.name}/'. Held-out эксперименты (held_out=[Experiment])
    в evaluate/value_and_gradient/residual не входят — только validate.

    Начальные состояния: общие x0-записи 'x0/<pop>/<slot>' участвуют во
    всех экспериментах; эксперимент-специфичные 'x0/<experiment>/<pop>/
    <slot>' материализуются только у эксперимента с совпадающим именем
    (включая held-out в validate) и перезаписывают общие значения в
    своих слотах; записи, не совпадающие ни с одним именем, никогда не
    материализуются — при каждом конструировании выдаётся UserWarning
    со списком таких «осиротевших» экспериментов.

    Тот же duck-type API, что у CalibrationProblem: size / pack / unpack /
    bounds_phi / evaluate / value_and_gradient / residual / simulate /
    validate. n_forward увеличивается на число прогонённых экспериментов
    (по одному на обучающий эксперимент за операцию, по одному на
    held-out в validate); счётчики — разделяемый self._counters.

    Args:
        network: NetworkRNN (одна сеть для всех экспериментов).
        parameter_space: ParameterSpace (общие параметры; общий x0
            подставляется в начальное состояние каждого эксперимента,
            exp-specific x0 — только в свой).
        experiments: непустая последовательность Experiment с objective;
            имена обучающих экспериментов должны быть уникальны и не
            пересекаться с held_out (иначе ValueError — два одинаковых
            имени делили бы exp-specific x0 одного имени молча).
        grad_mode: 'assign' (по умолчанию) или 'chain'.
        held_out: отложенные Experiment для validate; имена не должны
            дублироваться внутри held_out и пересекаться с обучающими
            (иначе ValueError); objective у held-out эксперимента
            обязателен (иначе validate поднимет ValueError).
        compiled: если True, value_and_gradient / evaluate / residual
            выполняются через скомпилированное ядро (см.
            CalibrationProblem): отдельное ядро на каждый обучающий
            эксперимент (ключ — имя эксперимента), суммы потерь/градиентов
            — eager-сложение тензоров снаружи (d/dφ Σ = Σ d/dφ);
            начальное состояние эксперимента — аргумент ядра
            (in-place смена exp.initial_state отражается немедленно).
        vectorize: если True, обучающие эксперименты прогоняются ОДНИМ
            батч-прогоном сети (этап 3): требуются одинаковые длины T и
            согласованный профиль входов (None у всех / тензоры одинаковой
            ширины; иначе ValueError). Потеря и градиент — СРЕДНЕЕ по
            экспериментам (mean_e), в отличие от циклового режима (сумма):
            переключение режимов меняет масштаб loss/grad ровно в E раз
            (эквивалентно весам 1/E у циклового). Счётчик n_forward
            увеличивается на 1 за батч-операцию (в цикловом — на число
            экспериментов). x0-записи ParameterSpace материализуются
            батчево: common — broadcast на все слоты (градиент
            аккумулирует вклады всех экспериментов), exp-specific — своя
            подстановка через one-hot маску (вклад только своего слота;
            изоляция как в цикловом режиме). residual — конкатенация
            невязок экспериментов; компоненты — с префиксом '{exp.name}/'.
            simulate() — срез запрошенного эксперимента (experiment=None —
            первого); validate() — один батч-прогон для held-out той же
            длины/профиля, иначе по одному. adjoint-метод несовместим
            (per-step разложения по батч-срезам нет,
            _VectorizedObjective.per_step_loss поднимает NotImplementedError).
            compiled=True + vectorize=True — один ядро на весь батч
            (кэш по id objective-обёртки и протоколов экспериментов).
        chunk_length / chunk_grad: чанкинг каждого прогона (см.
            CalibrationProblem и NetworkRNN.call): векторный батч чанкуется
            по времени в каждом слоте; несовместим с grad_mode='adjoint'.
            stability-штраф: stability_loss в батч-режиме — агрегат по
            всем слотам, при stability_penalty_weight > 0 его градиент
            смешивает слоты (кросс-слотовая утечка) — используйте
            vectorize=True с stability_penalty_weight=0 (default).
            simulate() и validate() — eager в обоих режимах.
    """

    def __init__(
        self,
        network: NetworkRNN,
        parameter_space: ParameterSpace,
        experiments: Sequence[Experiment],
        grad_mode: str = "assign",
        held_out: Optional[Sequence[Experiment]] = None,
        compiled: bool = False,
        vectorize: bool = False,
        chunk_length: Optional[int] = None,
        chunk_grad: str = "checkpointed",
    ):
        if grad_mode not in _GRAD_MODES:
            raise ValueError(
                f"MultiExperimentProblem: unknown grad_mode '{grad_mode}'. "
                f"Available modes: {list(_GRAD_MODES)}."
            )
        if grad_mode == "adjoint" and chunk_length is not None:
            raise ValueError(
                "MultiExperimentProblem: chunk_length с grad_method="
                "'adjoint' не поддержан (adjoint имеет собственную память "
                "O(1); смешение не реализовано). Используйте grad_mode="
                "'bptt' или chunk_length=None."
            )
        if chunk_length is not None and chunk_grad not in (
            "checkpointed",
            "truncated",
        ):
            raise ValueError(
                f"MultiExperimentProblem: unknown chunk_grad '{chunk_grad}'; "
                "expected 'checkpointed' or 'truncated'."
            )
        self.network = network
        self.parameter_space = parameter_space
        self.experiments: List[Experiment] = list(experiments)
        self.held_out: List[Experiment] = list(held_out or [])
        self.grad_mode = grad_mode
        if not self.experiments:
            raise ValueError(
                "MultiExperimentProblem: at least one experiment is required."
            )
        for exp in self.experiments:
            if exp.objective is None:
                raise ValueError(
                    f"MultiExperimentProblem: training experiment "
                    f"'{exp.name}' requires an objective."
                )
        train_names = [exp.name for exp in self.experiments]
        duplicates = sorted({n for n in train_names if train_names.count(n) > 1})
        if duplicates:
            raise ValueError(
                "MultiExperimentProblem: duplicate experiment names "
                f"{duplicates}: имена обучающих экспериментов должны быть "
                "уникальны, иначе эксперименты с одним именем делили бы "
                "exp-specific x0-записи этого имени."
            )
        held_names = [e.name for e in self.held_out]
        held_duplicates = sorted({n for n in held_names if held_names.count(n) > 1})
        if held_duplicates:
            raise ValueError(
                "MultiExperimentProblem: duplicate held-out experiment "
                f"names {held_duplicates}: имена должны быть уникальны, "
                "иначе validate() затирал бы компоненты одного имени "
                "другим в возвращаемом словаре."
            )
        overlap = sorted(set(train_names) & set(held_names))
        if overlap:
            raise ValueError(
                "MultiExperimentProblem: experiment names used both as "
                f"training and held-out: {overlap} — переименуйте одну из "
                "групп, иначе held-out неотличим от обучающего эксперимента."
            )
        known = {e.name for e in self.experiments} | {e.name for e in self.held_out}
        orphaned = sorted(
            {
                _x0_experiment(entry)
                for entry in self.parameter_space.phi_entries.values()
                if entry.kind in _TENSOR_KINDS
                and _x0_experiment(entry) is not None
                and _x0_experiment(entry) not in known
            }
        )
        if orphaned:
            warnings.warn(
                "MultiExperimentProblem: эксперимент-специфичные "
                f"x0-записи для экспериментов {orphaned} не совпадают ни "
                "с одним именем train/held-out эксперимента и никогда не "
                "материализуются (проверьте имена в "
                "add_experiment_initial_state_entries).",
                UserWarning,
                stacklevel=2,
            )
        self._counters: Dict[str, Any] = {
            "n_forward": 0,
            "n_gradient": 0,
            "wall_time": 0.0,
        }
        self.compiled = bool(compiled)
        # Кэш compiled-ядер: ключ (experiment_name, id(objective), id(t),
        # id(inputs)) — та же семантика, что у
        # CalibrationProblem._get_compiled_core. Начальное состояние
        # эксперимента — аргумент ядра, материализуется при каждом вызове.
        self._core_cache: Dict[Any, _CompiledCore] = {}
        self.chunk_length = chunk_length
        self.chunk_grad = chunk_grad
        self.vectorize = bool(vectorize)
        if self.vectorize:
            # Одинаковая длина T (статическая проверка; динамические оси
            # допускаем — несовпадение отловит tf.concat при прогоне).
            t0 = self.experiments[0].t_sequence.shape[1]
            for exp in self.experiments[1:]:
                te = exp.t_sequence.shape[1]
                if t0 is not None and te is not None and int(te) != int(t0):
                    raise ValueError(
                        "MultiExperimentProblem(vectorize=True): все "
                        "эксперименты должны иметь одинаковую длину T "
                        f"(got {int(t0)} and {int(te)} for "
                        f"'{exp.name}')."
                    )
            # Согласованный профиль входов: либо у всех None, либо у всех
            # тензоры одинаковой последней размерности.
            in0 = self.experiments[0].inputs
            for exp in self.experiments[1:]:
                ie = exp.inputs
                if (in0 is None) != (ie is None):
                    raise ValueError(
                        "MultiExperimentProblem(vectorize=True): профиль "
                        "входов несогласован — inputs=None у части "
                        f"экспериментов (см. '{exp.name}'). Либо у всех "
                        "None, либо у всех тензоры."
                    )
                if (
                    in0 is not None
                    and ie is not None
                    and in0.shape[-1] is not None
                    and ie.shape[-1] is not None
                    and int(in0.shape[-1]) != int(ie.shape[-1])
                ):
                    raise ValueError(
                        "MultiExperimentProblem(vectorize=True): "
                        "последняя размерность inputs должна совпадать у "
                        f"всех экспериментов (got {int(in0.shape[-1])} and "
                        f"{int(ie.shape[-1])} for '{exp.name}')."
                    )
            self._vectorized_objective = _VectorizedObjective(
                self.experiments, self.network
            )
        else:
            self._vectorized_objective = None
        self._vcore_cache: Dict[Any, _CompiledCore] = {}

    # ------------------------------------------------------------------
    # Векторный режим (vectorize=True): один батч-прогон на все эксперименты
    # ------------------------------------------------------------------
    def _prepare_vectorized(self, phi: Optional[Any]) -> tf.Tensor:
        """pack (+apply); n_forward += 1 (один батч-прогон на всех)."""
        phi_t = self.pack(phi)
        if phi is not None:
            self._apply_phi(phi_t)
        self._counters["n_forward"] += 1
        return phi_t

    def _batched_protocol(self, experiments):
        """Батч-протокол: concat t/inputs по слотам, per-exp base-состояния.

        Без кэша (конкатенация тензоров дёшево): in-place смена протоколов
        отражается немедленно. Порядок слотов = порядок experiments.
        """
        t_batched = tf.concat([e.t_sequence for e in experiments], axis=0)
        inputs_batched = None
        if experiments[0].inputs is not None:
            inputs_batched = tf.concat([e.inputs for e in experiments], axis=0)
        base_states = []
        for e in experiments:
            if e.initial_state is None:
                base_states.append(None)
            else:
                base_states.append(
                    (tuple(e.initial_state[0]), tuple(e.initial_state[1]))
                )
        names = [e.name for e in experiments]
        return t_batched, inputs_batched, base_states, names

    def _get_vectorized_core(self, t_batched, inputs_batched) -> _CompiledCore:
        """Ленивое получение compiled-ядра векторного батч-прогона.

        Ключ — id объектов ПРОТОКОЛОВ ЭКСПЕРИМЕНТОВ (стабильны, т.к. задача
        держит ссылки), а не конкатенированных тензоров: _batched_protocol
        пересоздаёт их при каждом вызове, ключ по их id промахивался бы
        при каждом вызове (retrace-шторм).
        """
        key = (
            id(self._vectorized_objective),
            tuple(e.name for e in self.experiments),
            tuple(id(e.t_sequence) for e in self.experiments),
            tuple(
                id(e.inputs) if e.inputs is not None else None for e in self.experiments
            ),
            self.chunk_length,
            self.chunk_grad,
        )
        core = self._vcore_cache.get(key)
        if core is None:
            core = _CompiledCore(
                self.network,
                self.parameter_space,
                t_batched,
                inputs_batched,
                self._vectorized_objective,
                experiment_name=None,
                grad_mode=self.grad_mode,
                x0_substitute=_make_vectorized_x0_substitute(
                    [e.name for e in self.experiments]
                ),
                chunk_length=self.chunk_length,
                chunk_grad=self.chunk_grad,
            )
            core.build()
            self._vcore_cache[key] = core
        return core

    def _held_out_can_batch(self) -> bool:
        """True, если held-out можно прогнать одним батчем: непустые,
        одинаковые T между собой и с обучающими, согласованный профиль
        входов (None у всех / тензоры одинаковой ширины).
        """
        if not self.held_out:
            return False
        train_t = self.experiments[0].t_sequence.shape[1]
        train_in = self.experiments[0].inputs
        for e in self.held_out:
            te = e.t_sequence.shape[1]
            if train_t is not None and te is not None and int(te) != int(train_t):
                return False
            ie = e.inputs
            if (train_in is None) != (ie is None):
                return False
            if (
                train_in is not None
                and ie is not None
                and train_in.shape[-1] is not None
                and ie.shape[-1] is not None
                and int(train_in.shape[-1]) != int(ie.shape[-1])
            ):
                return False
        return True

    def _vectorized_forward(self, phi_t, with_tape: bool):
        """Один батч-прогон векторного режима.

        Returns:
            (loss_t, grad_t или None, comps, residual_t или None).
            comps: dict {имя: тензор} с префиксами '{exp}/';
            residual_t: None, если у какого-то под-объектива нет
            residual-интерфейса (has_residual=False — статус решает
            вызывающий).
        """
        vobj = self._vectorized_objective
        t_b, inputs_b, base_states, names = self._batched_protocol(self.experiments)
        if self.compiled:
            init_pop, init_syn = _concat_base_states(self.network, base_states)
            core = self._get_vectorized_core(t_b, inputs_b)
            if with_tape:
                loss_t, grad_t, comps, residual_t = core.grad_core(
                    phi_t, init_pop, init_syn
                )
                self._counters["n_gradient"] += 1
            else:
                loss_t, comps, residual_t = core.eval_core(phi_t, init_pop, init_syn)
                grad_t = None
            return loss_t, grad_t, comps, residual_t
        if with_tape:
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(phi_t)
                # theta строится ВНУТРИ ленты: x0-записи дифференцируемы
                # по φ (как в цикловом режиме).
                theta = self.parameter_space.unpack(phi_t)
                initial_state, x0_map = _materialize_batched_initial_state(
                    self.network,
                    self.parameter_space,
                    theta,
                    base_states,
                    names,
                )
                output = self.network(
                    t_b,
                    inputs=inputs_b,
                    initial_state=initial_state,
                    chunk_length=self.chunk_length,
                    chunk_grad=self.chunk_grad,
                )
                loss_t = vobj(output, self.network)
            grad_t = _phi_gradient(
                self.parameter_space, self.grad_mode, tape, loss_t, phi_t, x0_map
            )
            del tape
            self._counters["n_gradient"] += 1
        else:
            theta = self.parameter_space.unpack(phi_t)
            initial_state, _ = _materialize_batched_initial_state(
                self.network,
                self.parameter_space,
                theta,
                base_states,
                names,
            )
            output = self.network(
                t_b,
                inputs=inputs_b,
                initial_state=initial_state,
                chunk_length=self.chunk_length,
                chunk_grad=self.chunk_grad,
            )
            loss_t = vobj(output, self.network)
            grad_t = None
        comps = vobj.components(output, self.network)
        residual_t = vobj.residual(output) if vobj.has_residual else None
        return loss_t, grad_t, comps, residual_t

    def _vectorized_evaluation(self, loss_t, comps, residual_t, grad, t0):
        """numpy-сборка Evaluation векторного режима."""
        vobj = self._vectorized_objective
        if vobj.has_residual:
            residuals = np.zeros(0) if residual_t is None else np.asarray(residual_t)
            status = "ok"
        else:
            residuals, status = np.zeros(0), "no_residual_interface"
        return _make_evaluation(
            loss=float(loss_t),
            components={k: float(v) for k, v in comps.items()},
            residuals=residuals,
            grad=grad,
            status=status,
            counters=self._counters,
            elapsed=time.perf_counter() - t0,
            dt=float(self.network._graph.dt),
        )

    def _exp_base_init(self, exp: Experiment) -> Tuple[Tuple, Tuple]:
        """Начальное состояние протокола эксперимента полными кортежами.

        Без кэша: пересоздаётся при каждом вызове, поэтому in-place смена
        exp.initial_state отражается немедленно.
        """
        if exp.initial_state is None:
            pop, syn = self.network.get_initial_state(batch_size=1)
            return (tuple(pop), tuple(syn))
        return (tuple(exp.initial_state[0]), tuple(exp.initial_state[1]))

    def _get_compiled_core(self, exp: Experiment) -> _CompiledCore:
        """Ленивое получение ядра эксперимента из кэша."""

        key = (
            exp.name,
            id(exp.objective),
            id(exp.t_sequence),
            id(exp.inputs),
            self.chunk_length,
            self.chunk_grad,
        )
        core = self._core_cache.get(key)
        if core is None:
            core = _CompiledCore(
                self.network,
                self.parameter_space,
                exp.t_sequence,
                exp.inputs,
                exp.objective,
                experiment_name=exp.name,
                grad_mode=self.grad_mode,
                chunk_length=self.chunk_length,
                chunk_grad=self.chunk_grad,
            )
            core.build()
            self._core_cache[key] = core
        return core

    # ------------------------------------------------------------------
    # Счётчики
    # ------------------------------------------------------------------
    @property
    def n_forward(self) -> int:
        return self._counters["n_forward"]

    @n_forward.setter
    def n_forward(self, value: int) -> None:
        self._counters["n_forward"] = int(value)

    @property
    def n_gradient(self) -> int:
        return self._counters["n_gradient"]

    @n_gradient.setter
    def n_gradient(self, value: int) -> None:
        self._counters["n_gradient"] = int(value)

    @property
    def wall_time(self) -> float:
        return self._counters["wall_time"]

    @wall_time.setter
    def wall_time(self, value: float) -> None:
        self._counters["wall_time"] = float(value)

    # ------------------------------------------------------------------
    # pack / unpack (общий ParameterSpace)
    # ------------------------------------------------------------------
    @property
    def size(self) -> int:
        return self.parameter_space.size

    def pack(self, phi: Optional[Any] = None) -> tf.Tensor:
        """См. CalibrationProblem.pack (общий ParameterSpace)."""
        if phi is None:
            return self.parameter_space.pack()
        dtype = neuraltide.config.get_dtype()
        phi_t = cast_if_needed(tf.convert_to_tensor(phi), dtype)
        phi_t = tf.reshape(phi_t, [-1])
        if phi_t.shape[0] is not None and int(phi_t.shape[0]) != self.size:
            raise ValueError(
                f"MultiExperimentProblem: phi has {int(phi_t.shape[0])} "
                f"elements, expected {self.size}."
            )
        tf.debugging.assert_equal(
            tf.size(phi_t), self.size, message="MultiExperimentProblem: bad phi size"
        )
        return phi_t

    def unpack(self, phi) -> Dict[str, tf.Tensor]:
        """φ → физические θ (общий ParameterSpace)."""
        return self.parameter_space.unpack(phi)

    def bounds_phi(self) -> Tuple[np.ndarray, np.ndarray]:
        """Границы φ общего ParameterSpace."""
        return _bounds_phi_from_space(self.parameter_space)

    def search_box_phi(
        self, rel_margin: float = SEARCH_BOX_REL_MARGIN
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Поисковый бокс φ общего ParameterSpace (см. CalibrationProblem)."""
        return _search_box_phi_from_space(self.parameter_space, rel_margin)

    # ------------------------------------------------------------------
    # Ядро
    # ------------------------------------------------------------------
    def _apply_phi(self, phi_t: tf.Tensor) -> None:
        theta = self.parameter_space.unpack(phi_t)
        self.parameter_space.apply_to_model(theta)

    def _prepare(self, phi: Optional[Any]) -> tf.Tensor:
        """pack (+apply); n_forward += число обучающих экспериментов."""
        phi_t = self.pack(phi)
        if phi is not None:
            self._apply_phi(phi_t)
        self._counters["n_forward"] += len(self.experiments)
        return phi_t

    def _resolve_experiment(self, experiment: Union[int, str, Experiment, None]):
        if experiment is None:
            return self.experiments[0]
        if isinstance(experiment, int):
            return self.experiments[experiment]
        if isinstance(experiment, str):
            for exp in self.experiments:
                if exp.name == experiment:
                    return exp
            raise ValueError(
                f"MultiExperimentProblem: unknown experiment '{experiment}'."
            )
        return experiment

    def _run_experiment(self, exp: Experiment, theta: Mapping[str, tf.Tensor]):
        initial_state, x0_map = _materialize_initial_state(
            self.network,
            self.parameter_space,
            theta,
            exp.initial_state,
            experiment=exp.name,
        )
        initial_state = _broadcast_or_validate_init(
            initial_state,
            tf.shape(exp.t_sequence)[0],
            what=f"experiment '{exp.name}'",
        )
        output = self.network(
            exp.t_sequence,
            inputs=exp.inputs,
            initial_state=initial_state,
            chunk_length=self.chunk_length,
            chunk_grad=self.chunk_grad,
        )
        return output, x0_map

    def _aggregate(
        self, outputs: Sequence[NetworkOutput]
    ) -> Tuple[Dict[str, float], np.ndarray, str]:
        components: Dict[str, float] = {}
        residuals: List[np.ndarray] = []
        has_residual = True
        for exp, output in zip(self.experiments, outputs):
            components.update(
                _components_of(
                    exp.objective, output, self.network, prefix=f"{exp.name}/"
                )
            )
            r = _residual_of(exp.objective, output)
            if r is None:
                has_residual = False
            else:
                residuals.append(r)
        if not has_residual:
            return components, np.zeros(0), "no_residual_interface"
        if not residuals:
            return components, np.zeros(0), "ok"
        return components, np.concatenate(residuals), "ok"

    # ------------------------------------------------------------------
    # Публичные операции
    # ------------------------------------------------------------------
    def simulate(
        self,
        phi: Optional[Any] = None,
        experiment: Union[int, str, Experiment, None] = None,
    ) -> NetworkOutput:
        """
        Прямой прогон одного эксперимента (по умолчанию первого).

        Контракт мутации как у CalibrationProblem: phi=None не трогает
        модель; phi != None переводит модель в θ = unpack(phi).

        vectorize=True: один батч-прогон; experiment=None возвращает срез
        ПЕРВОГО эксперимента (батч-output не возвращается — у слотов общий
        протокол; для полного батча используйте evaluate).
        """
        exp = self._resolve_experiment(experiment)
        phi_t = self.pack(phi)
        if phi is not None:
            self._apply_phi(phi_t)
        theta = self.parameter_space.unpack(phi_t)
        if self.vectorize:
            t_b, inputs_b, base_states, names = self._batched_protocol(self.experiments)
            initial_state, _ = _materialize_batched_initial_state(
                self.network,
                self.parameter_space,
                theta,
                base_states,
                names,
            )
            output = self.network(
                t_b,
                inputs=inputs_b,
                initial_state=initial_state,
                chunk_length=self.chunk_length,
                chunk_grad=self.chunk_grad,
            )
            self._counters["n_forward"] += 1
            names = [e.name for e in self.experiments]
            try:
                slot = names.index(exp.name)
            except ValueError:
                raise ValueError(
                    "MultiExperimentProblem(vectorize=True): simulate "
                    f"эксперимент '{exp.name}' не входит в обучающие "
                    f"эксперименты {names}."
                ) from None
            return _VectorizedObjective._slice(output, slot)
        output, _ = self._run_experiment(exp, theta)
        self._counters["n_forward"] += 1
        return output

    def evaluate(self, phi: Optional[Any] = None) -> Evaluation:
        """Суммарная потеря обучающих экспериментов (без градиента).

        vectorize=True: потеря — СРЕДНЕЕ по экспериментам (один
        батч-прогон), не сумма."""
        t0 = time.perf_counter()
        if self.vectorize:
            phi_t = self._prepare_vectorized(phi)
            loss_t, _, comps, residual_t = self._vectorized_forward(
                phi_t, with_tape=False
            )
            return self._vectorized_evaluation(
                loss_t, comps, residual_t, grad=None, t0=t0
            )
        phi_t = self._prepare(phi)
        if self.compiled:
            total: Optional[tf.Tensor] = None
            components: Dict[str, float] = {}
            residuals: List[np.ndarray] = []
            has_residual = True
            for exp in self.experiments:
                core = self._get_compiled_core(exp)
                loss_e, comps_e, residual_e = core.eval_core(
                    phi_t, *self._exp_base_init(exp)
                )
                total = loss_e if total is None else total + loss_e
                components.update(
                    {f"{exp.name}/{k}": float(v) for k, v in comps_e.items()}
                )
                if getattr(exp.objective, "residual", None) is None:
                    has_residual = False
                else:
                    residuals.append(np.asarray(residual_e))
            if not has_residual:
                residuals_np, status = np.zeros(0), "no_residual_interface"
            elif not residuals:
                residuals_np, status = np.zeros(0), "ok"
            else:
                residuals_np, status = np.concatenate(residuals), "ok"
            return _make_evaluation(
                loss=float(total),
                components=components,
                residuals=residuals_np,
                grad=None,
                status=status,
                counters=self._counters,
                elapsed=time.perf_counter() - t0,
                dt=float(self.network._graph.dt),
            )
        theta = self.parameter_space.unpack(phi_t)
        total: Optional[tf.Tensor] = None
        outputs: List[NetworkOutput] = []
        for exp in self.experiments:
            output, _ = self._run_experiment(exp, theta)
            outputs.append(output)
            loss_e = exp.objective(output, self.network)
            total = loss_e if total is None else total + loss_e
        components, residuals, status = self._aggregate(outputs)
        return _make_evaluation(
            loss=float(total),
            components=components,
            residuals=residuals,
            grad=None,
            status=status,
            counters=self._counters,
            elapsed=time.perf_counter() - t0,
            dt=float(self.network._graph.dt),
        )

    def value_and_gradient(
        self, phi: Optional[Any] = None
    ) -> Tuple[Evaluation, tf.Tensor]:
        """
        Суммарная потеря и градиент по φ (все прогоны в одной ленте).

        compiled=True: отдельное ядро на эксперимент, суммы потерь и
        градиентов — eager-сложение тензоров снаружи (d/dφ Σ = Σ d/dφ).

        vectorize=True: один батч-прогон; потеря и градиент — СРЕДНЕЕ по
        экспериментам (mean_e), эквивалентно весам 1/E в цикловом режиме.

        Returns:
            (Evaluation, grad — tf.Tensor формы [size]).
        """
        t0 = time.perf_counter()
        if self.vectorize:
            phi_t = self._prepare_vectorized(phi)
            loss_t, grad_t, comps, residual_t = self._vectorized_forward(
                phi_t, with_tape=True
            )
            eval_result = self._vectorized_evaluation(
                loss_t, comps, residual_t, grad=grad_t, t0=t0
            )
            return eval_result, grad_t
        phi_t = self._prepare(phi)
        if self.compiled:
            losses: List[tf.Tensor] = []
            grads: List[tf.Tensor] = []
            components: Dict[str, float] = {}
            residuals: List[np.ndarray] = []
            has_residual = True
            for exp in self.experiments:
                core = self._get_compiled_core(exp)
                loss_e, grad_e, comps_e, residual_e = core.grad_core(
                    phi_t, *self._exp_base_init(exp)
                )
                losses.append(loss_e)
                grads.append(grad_e)
                components.update(
                    {f"{exp.name}/{k}": float(v) for k, v in comps_e.items()}
                )
                if getattr(exp.objective, "residual", None) is None:
                    has_residual = False
                else:
                    residuals.append(np.asarray(residual_e))
            loss_t = tf.add_n(losses)
            grad_t = tf.add_n(grads)
            self._counters["n_gradient"] += 1
            if not has_residual:
                residuals_np, status = np.zeros(0), "no_residual_interface"
            elif not residuals:
                residuals_np, status = np.zeros(0), "ok"
            else:
                residuals_np, status = np.concatenate(residuals), "ok"
            eval_result = _make_evaluation(
                loss=float(loss_t),
                components=components,
                residuals=residuals_np,
                grad=grad_t,
                status=status,
                counters=self._counters,
                elapsed=time.perf_counter() - t0,
                dt=float(self.network._graph.dt),
            )
            return eval_result, grad_t
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(phi_t)
            theta = self.parameter_space.unpack(phi_t)
            losses: List[tf.Tensor] = []
            outputs: List[NetworkOutput] = []
            # Тензор x0 создаётся заново в каждом эксперименте: собираем все,
            # иначе градиент по x0 учитывал бы только последний эксперимент.
            x0_map: Dict[str, List[tf.Tensor]] = {}
            for exp in self.experiments:
                output, x0_e = self._run_experiment(exp, theta)
                outputs.append(output)
                for key, tensor in x0_e.items():
                    x0_map.setdefault(key, []).append(tensor)
                losses.append(exp.objective(output, self.network))
            loss_t = tf.add_n(losses)
        grad_t = _phi_gradient(
            self.parameter_space, self.grad_mode, tape, loss_t, phi_t, x0_map
        )
        del tape
        self._counters["n_gradient"] += 1
        components, residuals, status = self._aggregate(outputs)
        eval_result = _make_evaluation(
            loss=float(loss_t),
            components=components,
            residuals=residuals,
            grad=grad_t,
            status=status,
            counters=self._counters,
            elapsed=time.perf_counter() - t0,
            dt=float(self.network._graph.dt),
        )
        return eval_result, grad_t

    def residual(self, phi: Optional[Any] = None) -> np.ndarray:
        """Конкатенированные невязки обучающих экспериментов (numpy)."""
        t0 = time.perf_counter()
        if self.vectorize:
            phi_t = self._prepare_vectorized(phi)
            vobj = self._vectorized_objective
            if not vobj.has_residual:
                bad = next(
                    e
                    for i, e in enumerate(self.experiments)
                    if vobj._residual_fns[i] is None
                )
                raise TypeError(
                    "MultiExperimentProblem (vectorize=True): objective "
                    f"'{type(bad.objective).__name__}' эксперимента "
                    f"'{bad.name}' не предоставляет residual(predictions)."
                )
            _, _, _, residual_t = self._vectorized_forward(phi_t, with_tape=False)
            self._counters["wall_time"] += time.perf_counter() - t0
            return np.asarray(residual_t)
        phi_t = self._prepare(phi)
        if self.compiled:
            for exp in self.experiments:
                if getattr(exp.objective, "residual", None) is None:
                    raise TypeError(
                        "MultiExperimentProblem: objective "
                        f"'{type(exp.objective).__name__}' эксперимента "
                        f"'{exp.name}' не предоставляет residual(predictions)."
                    )
            residuals: List[np.ndarray] = []
            for exp in self.experiments:
                core = self._get_compiled_core(exp)
                _, _, residual_e = core.eval_core(phi_t, *self._exp_base_init(exp))
                residuals.append(np.asarray(residual_e))
            self._counters["wall_time"] += time.perf_counter() - t0
            return np.concatenate(residuals)
        theta = self.parameter_space.unpack(phi_t)
        residuals: List[np.ndarray] = []
        for exp in self.experiments:
            output, _ = self._run_experiment(exp, theta)
            r = _residual_of(exp.objective, output)
            if r is None:
                raise TypeError(
                    "MultiExperimentProblem: objective "
                    f"'{type(exp.objective).__name__}' эксперимента "
                    f"'{exp.name}' не предоставляет residual(predictions)."
                )
            residuals.append(r)
        self._counters["wall_time"] += time.perf_counter() - t0
        return np.concatenate(residuals)

    def validate(self, phi: Optional[Any] = None) -> Dict[str, float]:
        """
        Потери на held-out экспериментах (только они, не обучающие).

        phi=None не трогает модель; phi != None переводит модель в
        θ = unpack(phi) один раз. objective берётся из эксперимента.

        vectorize=True: если у всех held-out одинаковые T и профиль входов
        с обучающими (и между собой) — один батч-прогон со срезами;
        иначе — по одному, как цикловой режим.
        """
        if not self.held_out:
            return {}
        phi_t = self.pack(phi)
        if phi is not None:
            self._apply_phi(phi_t)
        theta = self.parameter_space.unpack(phi_t)
        results: Dict[str, float] = {}
        t0 = time.perf_counter()
        if self.vectorize and self._held_out_can_batch():
            for exp in self.held_out:
                if exp.objective is None:
                    raise ValueError(
                        f"MultiExperimentProblem: held-out experiment "
                        f"'{exp.name}' has no objective."
                    )
            t_b, inputs_b, base_states, names = self._batched_protocol(self.held_out)
            initial_state, _ = _materialize_batched_initial_state(
                self.network,
                self.parameter_space,
                theta,
                base_states,
                names,
            )
            output = self.network(
                t_b,
                inputs=inputs_b,
                initial_state=initial_state,
                chunk_length=self.chunk_length,
                chunk_grad=self.chunk_grad,
            )
            for i, exp in enumerate(self.held_out):
                results[exp.name] = float(
                    exp.objective(_VectorizedObjective._slice(output, i), self.network)
                )
            self._counters["n_forward"] += 1
            self._counters["wall_time"] += time.perf_counter() - t0
            return results
        for exp in self.held_out:
            if exp.objective is None:
                raise ValueError(
                    f"MultiExperimentProblem: held-out experiment "
                    f"'{exp.name}' has no objective."
                )
            output, _ = self._run_experiment(exp, theta)
            results[exp.name] = float(exp.objective(output, self.network))
            self._counters["n_forward"] += 1
        self._counters["wall_time"] += time.perf_counter() - t0
        return results


# ---------------------------------------------------------------------------
# FrozenProblem
# ---------------------------------------------------------------------------


class FrozenProblem:
    """
    Подпространство φ поверх CalibrationProblem/MultiExperimentProblem.

    Свободными остаются записи free_names (по умолчанию — все trainable,
    кроме fixed; non-trainable записи ParameterSpace — реестровые константы
    вне вектора φ и в подпространство никогда не входят); остальные
    координаты заморожены: fixed — на явных физических значениях θ из
    словаря fixed, прочие — на текущих значениях модели на момент
    конструирования. Все операции делегируются базовой задаче с полным
    вектором phi_full = compose(phi_free); счётчики общие (делятся с базовой
    задачей). Градиент по замороженным координатам тождественно ноль — они
    исключены из reduced-градиента.

    Args:
        problem: базовая задача (CalibrationProblem или
            MultiExperimentProblem).
        free_names: имена trainable-записей ParameterSpace, остающихся
            свободными; None — все trainable-записи, кроме fixed.
            Non-trainable имя в free_names/fixed отклоняется ValueError
            («не входит в оптимизируемый φ»), как и неизвестное имя.
        fixed: {full_name: физическое θ} — записи, зафиксированные на
            явных значениях.

    Контракт мутации: phi != None переводит базовую модель в точку с
    phi_full (замороженные координаты в ней всегда на зафиксированных
    значениях). phi=None без fixed не трогает модель. phi=None при
    непустом fixed оценивает точку compose(pack()) — текущие свободные
    координаты и значения fixed — и поэтому переводит модель в неё: иначе
    потеря считалась бы при значениях модели, а не при fixed.
    """

    def __init__(
        self,
        problem,
        free_names: Optional[Sequence[str]] = None,
        fixed: Optional[Mapping[str, Any]] = None,
    ):
        self._problem = problem
        space = problem.parameter_space
        # φ подпространство строится только по trainable-записям; реестр
        # (space.entries) нужен лишь чтобы отличить «неизвестное имя» от
        # «известной non-trainable записи вне оптимизируемого φ».
        entries = space.phi_entries
        registry = space.entries
        fixed = {str(k): v for k, v in (fixed or {}).items()}
        for name in fixed:
            if name not in entries:
                if name in registry:
                    raise ValueError(
                        f"FrozenProblem: fixed name '{name}' не входит в "
                        "оптимизируемый φ (запись trainable=False; "
                        "фиксировать можно только trainable-записи)."
                    )
                raise ValueError(f"FrozenProblem: unknown fixed name '{name}'.")
        if free_names is None:
            free = [n for n in entries if n not in fixed]
        else:
            free = [str(n) for n in free_names]
            for name in free:
                if name not in entries:
                    if name in registry:
                        raise ValueError(
                            f"FrozenProblem: free name '{name}' не входит в "
                            "оптимизируемый φ (запись trainable=False; "
                            "свободными могут быть только trainable-записи)."
                        )
                    raise ValueError(f"FrozenProblem: unknown free name '{name}'.")
            overlap = sorted(set(free) & set(fixed))
            if overlap:
                raise ValueError(
                    f"FrozenProblem: names both free and fixed: {overlap}."
                )
        self._free_names: List[str] = free
        self._has_fixed = bool(fixed)

        dtype = neuraltide.config.get_dtype()
        self._full_offsets: Dict[str, Tuple[int, int]] = {}
        start = 0
        for name, entry in entries.items():
            self._full_offsets[name] = (start, entry.numel)
            start += entry.numel

        # Базовый полный вектор: замороженные координаты корректны,
        # свободные перезаписываются compose() при каждом вызове.
        phi_base = np.zeros(space.size, dtype=np.float64)
        for name, entry in entries.items():
            if name in self._free_names:
                continue
            if name in fixed:
                theta = space._value_to_tensor(fixed[name], entry)
                phi_piece = entry.transform.inverse(theta)
            else:
                if entry.variable is None:
                    raise ValueError(
                        f"FrozenProblem: entry '{name}' заморожена, но у неё "
                        "нет переменной и нет fixed-значения (tensor-записи "
                        "должны входить в free_names или fixed)."
                    )
                phi_piece = entry.transform.inverse(
                    cast_if_needed(entry.variable, dtype)
                )
            s, n = self._full_offsets[name]
            phi_base[s : s + n] = np.asarray(phi_piece, dtype=np.float64).reshape(-1)
        self._phi_base = phi_base

        free_index: List[np.ndarray] = []
        pos = 0
        for name in self._free_names:
            s, n = self._full_offsets[name]
            free_index.append(np.arange(s, s + n))
            pos += n
        self._free_index = (
            np.concatenate(free_index) if free_index else np.zeros(0, dtype=int)
        )

    # ------------------------------------------------------------------
    # Свойства
    # ------------------------------------------------------------------
    @property
    def problem(self):
        """Базовая задача."""
        return self._problem

    @property
    def parameter_space(self) -> ParameterSpace:
        return self._problem.parameter_space

    @property
    def free_names(self) -> List[str]:
        return list(self._free_names)

    @property
    def size(self) -> int:
        """Размерность reduced-вектора φ (свободные координаты)."""
        return int(len(self._free_index))

    @property
    def n_forward(self) -> int:
        return self._problem.n_forward

    @property
    def n_gradient(self) -> int:
        return self._problem.n_gradient

    @property
    def wall_time(self) -> float:
        return self._problem.wall_time

    # ------------------------------------------------------------------
    # pack / unpack / compose
    # ------------------------------------------------------------------
    def _compose(self, phi_free) -> tf.Tensor:
        """Reduced φ → полный вектор φ базовой задачи."""
        phi_np = np.asarray(phi_free, dtype=np.float64).reshape(-1)
        if phi_np.shape[0] != self.size:
            raise ValueError(
                f"FrozenProblem: phi has {int(phi_np.shape[0])} elements, "
                f"expected {self.size}."
            )
        full = self._phi_base.copy()
        full[self._free_index] = phi_np
        return tf.constant(full, dtype=neuraltide.config.get_dtype())

    def pack(self, phi: Optional[Any] = None) -> tf.Tensor:
        """
        Reduced φ текущей точки.

        phi=None — свободные координаты текущего полного pack() базовой
        модели; phi != None — проверка размера и приведение reduced-φ.
        """
        if phi is None:
            phi_full = np.asarray(self._problem.pack(), dtype=np.float64).reshape(-1)
            return tf.constant(
                phi_full[self._free_index], dtype=neuraltide.config.get_dtype()
            )
        phi_np = np.asarray(phi, dtype=np.float64).reshape(-1)
        if phi_np.shape[0] != self.size:
            raise ValueError(
                f"FrozenProblem: phi has {int(phi_np.shape[0])} elements, "
                f"expected {self.size}."
            )
        return tf.constant(phi_np, dtype=neuraltide.config.get_dtype())

    def unpack(self, phi) -> Dict[str, tf.Tensor]:
        """Reduced φ → физические θ свободных записей."""
        theta_full = self.parameter_space.unpack(self._compose(phi))
        return {n: theta_full[n] for n in self._free_names}

    def bounds_phi(self) -> Tuple[np.ndarray, np.ndarray]:
        """Границы reduced φ (строки bounds_phi базовой задачи)."""
        lower, upper = self._problem.bounds_phi()
        return lower[self._free_index], upper[self._free_index]

    def search_box_phi(
        self, rel_margin: float = SEARCH_BOX_REL_MARGIN
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Поисковый бокс reduced φ (строки бокса базовой задачи)."""
        fn = getattr(self._problem, "search_box_phi", None)
        lower, upper = fn(rel_margin) if callable(fn) else self._problem.bounds_phi()
        return lower[self._free_index], upper[self._free_index]

    # ------------------------------------------------------------------
    # Операции (делегирование с compose)
    # ------------------------------------------------------------------
    def _full_phi(self, phi: Optional[Any]) -> Optional[tf.Tensor]:
        """
        phi_full для делегирования: compose(phi); при phi=None — None (без
        мутации), если fixed пуст, и compose(pack()) иначе (см. контракт).
        """
        if phi is not None:
            return self._compose(phi)
        if self._has_fixed:
            return self._compose(self.pack())
        return None

    def evaluate(self, phi: Optional[Any] = None) -> Evaluation:
        """Потеря базовой задачи в точке phi_full = compose(phi)."""
        return self._problem.evaluate(self._full_phi(phi))

    def value_and_gradient(
        self, phi: Optional[Any] = None
    ) -> Tuple[Evaluation, tf.Tensor]:
        """
        Потеря и reduced-градиент (координаты free_names).

        Returns:
            (Evaluation базовой задачи, grad — tf.Tensor формы [size]).
        """
        eval_result, grad_full = self._problem.value_and_gradient(self._full_phi(phi))
        grad_free = tf.gather(tf.convert_to_tensor(grad_full), self._free_index)
        return eval_result, grad_free

    def residual(self, phi: Optional[Any] = None) -> np.ndarray:
        """Невязки базовой задачи в точке phi_full = compose(phi)."""
        return self._problem.residual(self._full_phi(phi))

    def validate(self, phi: Optional[Any] = None) -> Dict[str, float]:
        """validate базовой задачи в точке phi_full = compose(phi)."""
        return self._problem.validate(self._full_phi(phi))

    def simulate(self, phi: Optional[Any] = None, experiment=None) -> NetworkOutput:
        """simulate базовой задачи (experiment — для MultiExperimentProblem)."""
        full = self._full_phi(phi)
        if experiment is None:
            return self._problem.simulate(full)
        return self._problem.simulate(full, experiment=experiment)
