"""
CalibrationProblem: единая постановка задачи калибровки (этапы P0/P1).

Инкапсулирует модель (NetworkRNN), пространство параметров (ParameterSpace),
протокол (t_sequence, inputs, initial_state) и целевую функцию (objective),
предоставляя одинаковые операции для всех будущих оптимизаторов
(Keras, SciPy, безградиентные методы):

    simulate(phi)           — прямой прогон сети;
    evaluate(phi)           — скалярная потеря, компоненты и невязки;
    value_and_gradient(phi) — потеря и градиент по φ;
    residual(phi)           — вектор нормированных невязок для GN/LM/TRF;
    validate(phi)           — потери на held-out экспериментах.

Этап P1 добавляет поверх P0:

    * tensor-backed записи ParameterSpace (kind 'initial_state'/
      'shooting_state', см. ParameterSpace.add_initial_state_entries):
      начальное состояние материализуется как дифференцируемая функция
      от φ — θ_x0 = transform.forward(кусок φ) подставляется в
      network.call(initial_state=...); вклад в градиент в chain-режиме —
      tape.gradient(loss, x0_tensor) * transform.jacobian (для model_var
      путь прежний: градиент по переменным × jacobian);
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

Ограничения: eager-режим (без tf.function вокруг методов задачи);
batch=1 (контракт NetworkRNN); записи ParameterSpace kind 'model_var'
без привязанной переменной не включаются в градиент — их вклад
тождественно нулевой.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf

import neuraltide.config
from neuraltide.core.network import NetworkOutput, NetworkRNN
from neuraltide.core.types import TensorType, cast_if_needed
from neuraltide.optimization.parameters import ParameterSpace
from neuraltide.optimization.results import Evaluation

__all__ = [
    "CalibrationProblem",
    "Experiment",
    "MultiExperimentProblem",
    "FrozenProblem",
]

_GRAD_MODES = ("assign", "chain")
_TENSOR_KINDS = ("initial_state", "shooting_state")

InitialState = Optional[Tuple[Sequence[TensorType], Sequence[TensorType]]]


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
            ParameterSpace.
    """

    name: str
    t_sequence: TensorType
    inputs: Optional[TensorType] = None
    objective: Optional[Any] = None
    initial_state: InitialState = None


# ---------------------------------------------------------------------------
# Модульные хелперы (общие для Calibration/Multi/Frozen)
# ---------------------------------------------------------------------------


def _tensor_entries(space: ParameterSpace):
    return [(n, e) for n, e in space.entries.items() if e.kind in _TENSOR_KINDS]


def _materialize_initial_state(
    network: NetworkRNN,
    space: ParameterSpace,
    theta: Mapping[str, tf.Tensor],
    base_initial_state: InitialState,
) -> Tuple[InitialState, Dict[str, tf.Tensor]]:
    """
    Строит начальное состояние для network.call с подставленными x0.

    theta — физические значения из space.unpack (дифференцируемы по φ,
    если построены внутри GradientTape). base_initial_state — начальное
    состояние протокола (pop, syn) или None → сетевое по умолчанию.

    Returns:
        (initial_state, x0_map): initial_state — None, если нет ни
        tensor-записей, ни base (сеть возьмёт своё дефолтное); иначе
        (init_pop, init_syn). x0_map: full_name → тензор, фактически
        подставленный в состояние (источник для chain-градиента).
    """
    entries = _tensor_entries(space)
    if not entries:
        return base_initial_state, {}
    dtype = neuraltide.config.get_dtype()
    if base_initial_state is not None:
        init_pop = list(base_initial_state[0])
        init_syn = list(base_initial_state[1])
    else:
        pop, syn = network.get_initial_state(batch_size=1)
        init_pop = list(pop)
        init_syn = list(syn)
    offsets = network.pop_state_offsets
    x0_map: Dict[str, tf.Tensor] = {}
    for name, entry in entries:
        locator = entry.locator
        if locator is None or locator.population not in offsets:
            raise ValueError(
                f"CalibrationProblem: locator записи '{name}' ссылается на "
                f"неизвестную популяцию "
                f"'{locator.population if locator else None}'."
            )
        start, count = offsets[locator.population]
        if not (0 <= locator.slot < count):
            raise ValueError(
                f"CalibrationProblem: locator slot {locator.slot} вне "
                f"[0, {count}) для популяции '{locator.population}'."
            )
        x0 = cast_if_needed(tf.convert_to_tensor(theta[name]), dtype)
        x0 = tf.reshape(x0, entry.shape)
        init_pop[start + locator.slot] = x0
        x0_map[name] = x0
    return (init_pop, init_syn), x0_map


def _chain_gradient(
    space: ParameterSpace,
    tape: tf.GradientTape,
    loss_t: tf.Tensor,
    phi_t: tf.Tensor,
    x0_map: Mapping[str, Any],
) -> tf.Tensor:
    """
    dloss/dφ = dloss/dисточник × dисточник/dφ (поэлементно × jacobian).

    model_var: dloss/dθ по переменным модели × transform.jacobian;
    tensor-записи: dloss/dx0 по подставленному тензору × transform.jacobian.
    x0_map[name] может быть списком тензоров — по одному на эксперимент,
    в котором подставлялась запись (MultiExperimentProblem): вклады
    суммируются. Записи без доступного источника дают нулевой вклад.
    """
    dtype = neuraltide.config.get_dtype()
    entries = space.entries
    sources: List[tf.Tensor] = []
    source_names: List[str] = []
    for name, entry in entries.items():
        if entry.kind == "model_var":
            if entry.variable is not None:
                sources.append(entry.variable)
                source_names.append(name)
        else:
            x0 = x0_map.get(name)
            if x0 is None:
                continue
            for tensor in x0 if isinstance(x0, (list, tuple)) else [x0]:
                sources.append(tensor)
                source_names.append(name)
    grads = tape.gradient(loss_t, sources) if sources else []
    grad_by_name: Dict[str, tf.Tensor] = {}
    for name, g in zip(source_names, grads):
        if g is None:
            continue
        grad_by_name[name] = g if name not in grad_by_name else grad_by_name[name] + g

    pieces = []
    start = 0
    for name, entry in entries.items():
        numel = entry.numel
        phi_piece = phi_t[start : start + numel]
        g_src = grad_by_name.get(name)
        if g_src is None:
            contrib = tf.zeros([numel], dtype=dtype)
        else:
            jac = entry.transform.jacobian(tf.reshape(phi_piece, entry.shape))
            contrib = tf.reshape(g_src, [-1]) * tf.reshape(jac, [-1])
        pieces.append(contrib)
        start += numel
    if not pieces:
        return tf.zeros([0], dtype=dtype)
    return tf.concat(pieces, axis=0)


def _phi_gradient(
    space: ParameterSpace,
    grad_mode: str,
    tape: tf.GradientTape,
    loss_t: tf.Tensor,
    phi_t: tf.Tensor,
    x0_map: Mapping[str, tf.Tensor],
) -> tf.Tensor:
    """
    Градиент dloss/dφ в выбранном режиме.

    'assign': literal-путь корректен только без model_var-записей с
        переменными — при них градиент через variable.assign недоступен,
        а literal вернул бы неполный градиент (только x0-часть).
    'chain': всегда цепное вычисление.
    """
    has_model_vars = any(
        entry.kind == "model_var" and entry.variable is not None
        for entry in space.entries.values()
    )
    if grad_mode == "assign" and not has_model_vars:
        g = tape.gradient(loss_t, phi_t)
        if g is not None:
            return g
    return _chain_gradient(space, tape, loss_t, phi_t, x0_map)


def _bounds_phi_from_space(space: ParameterSpace) -> Tuple[np.ndarray, np.ndarray]:
    """Границы вектора φ из физических min/max записей ParameterSpace."""
    lower: list = []
    upper: list = []
    dtype = neuraltide.config.get_dtype()
    for entry in space.entries.values():
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

    bounds_phi берёт inverse(min/max): у sigmoid-образных преобразований
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
    for entry in space.entries.values():
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
    ):
        if grad_mode not in _GRAD_MODES:
            raise ValueError(
                f"CalibrationProblem: unknown grad_mode '{grad_mode}'. "
                f"Available modes: {list(_GRAD_MODES)}."
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
        output = self.network(
            self.t_sequence,
            inputs=self.inputs,
            initial_state=initial_state,
        )
        return phi_t, output

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
        phi_t = self._prepare(phi)
        # persistent: assign-режим сначала пробует literal-путь по phi,
        # и при None переиспользует ту же ленту для цепного вычисления.
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(phi_t)
            theta = self.parameter_space.unpack(phi_t)
            initial_state, x0_map = _materialize_initial_state(
                self.network, self.parameter_space, theta, self.initial_state
            )
            output = self.network(
                self.t_sequence,
                inputs=self.inputs,
                initial_state=initial_state,
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
            output = self.network(
                exp.t_sequence, inputs=exp.inputs, initial_state=initial_state
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
        Границы вектора φ из физических min/max записей ParameterSpace.

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

    Тот же duck-type API, что у CalibrationProblem: size / pack / unpack /
    bounds_phi / evaluate / value_and_gradient / residual / simulate /
    validate. n_forward увеличивается на число прогонённых экспериментов
    (по одному на обучающий эксперимент за операцию, по одному на
    held-out в validate); счётчики — разделяемый self._counters.

    Args:
        network: NetworkRNN (одна сеть для всех экспериментов).
        parameter_space: ParameterSpace (общие параметры; x0-записи
            подставляются в начальное состояние каждого эксперимента).
        experiments: непустая последовательность Experiment с objective.
        grad_mode: 'assign' (по умолчанию) или 'chain'.
        held_out: отложенные Experiment для validate; objective у
            held-out эксперимента обязателен (иначе validate поднимет
            ValueError).
    """

    def __init__(
        self,
        network: NetworkRNN,
        parameter_space: ParameterSpace,
        experiments: Sequence[Experiment],
        grad_mode: str = "assign",
        held_out: Optional[Sequence[Experiment]] = None,
    ):
        if grad_mode not in _GRAD_MODES:
            raise ValueError(
                f"MultiExperimentProblem: unknown grad_mode '{grad_mode}'. "
                f"Available modes: {list(_GRAD_MODES)}."
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
        self._counters: Dict[str, Any] = {
            "n_forward": 0,
            "n_gradient": 0,
            "wall_time": 0.0,
        }

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
            self.network, self.parameter_space, theta, exp.initial_state
        )
        output = self.network(
            exp.t_sequence, inputs=exp.inputs, initial_state=initial_state
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
        """
        exp = self._resolve_experiment(experiment)
        phi_t = self.pack(phi)
        if phi is not None:
            self._apply_phi(phi_t)
        theta = self.parameter_space.unpack(phi_t)
        output, _ = self._run_experiment(exp, theta)
        self._counters["n_forward"] += 1
        return output

    def evaluate(self, phi: Optional[Any] = None) -> Evaluation:
        """Суммарная потеря обучающих экспериментов (без градиента)."""
        t0 = time.perf_counter()
        phi_t = self._prepare(phi)
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

        Returns:
            (Evaluation, grad — tf.Tensor формы [size]).
        """
        t0 = time.perf_counter()
        phi_t = self._prepare(phi)
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
        phi_t = self._prepare(phi)
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

    Свободными остаются записи free_names (по умолчанию — все, кроме
    fixed); остальные координаты заморожены: fixed — на явных физических
    значениях θ из словаря fixed, прочие — на текущих значениях модели
    на момент конструирования. Все операции делегируются базовой задаче
    с полным вектором phi_full = compose(phi_free); счётчики общие
    (делятся с базовой задачей). Градиент по замороженным координатам
    тождественно ноль — они исключены из reduced-градиента.

    Args:
        problem: базовая задача (CalibrationProblem или
            MultiExperimentProblem).
        free_names: имена записей ParameterSpace, остающихся свободными;
            None — все записи, кроме fixed.
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
        entries = space.entries
        fixed = {str(k): v for k, v in (fixed or {}).items()}
        for name in fixed:
            if name not in entries:
                raise ValueError(f"FrozenProblem: unknown fixed name '{name}'.")
        if free_names is None:
            free = [n for n in entries if n not in fixed]
        else:
            free = [str(n) for n in free_names]
            for name in free:
                if name not in entries:
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
