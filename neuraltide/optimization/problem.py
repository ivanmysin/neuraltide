"""
CalibrationProblem: единая постановка задачи калибровки (этап P0).

Инкапсулирует модель (NetworkRNN), пространство параметров (ParameterSpace),
протокол (t_sequence, inputs, initial_state) и целевую функцию (objective),
предоставляя одинаковые операции для всех будущих оптимизаторов
(Keras, SciPy, безградиентные методы):

    simulate(phi)          — прямой прогон сети;
    evaluate(phi)          — скалярная потеря, компоненты и невязки;
    value_and_gradient(phi) — потеря и градиент по φ;
    residual(phi)          — вектор нормированных невязок для GN/LM/TRF.

Контракт мутации модели
-----------------------
phi=None: модель не изменяется — оценка выполняется при текущих значениях
    переменных (начальное φ получают через pack()).
phi != None: модель явно переводится в состояние θ = unpack(phi) перед
    прогоном и остаётся в нём после вызова. Это явное намерение: повторные
    вызовы с разными φ не «портят» модель, а последовательно задают её
    состояние. Чтобы сохранить и восстановить состояние, используйте
    ``saved = problem.pack()`` и
    ``parameter_space.apply_to_model(parameter_space.unpack(saved))``.

Режимы градиента (grad_mode)
----------------------------
'assign' (по умолчанию): предполагается путь tape.watch(phi) →
    unpack(phi) → apply_to_model(theta) → forward → tape.gradient(loss, phi).
    В текущем eager TensorFlow градиент через variable.assign не
    дифференцируется (возвращается None), поэтому используется
    математически эквивалентная цепочка: сначала dloss/dθ по переменным
    модели, затем умножение на dθ/dφ (jacobian transforms).

'chain': то же цепное вычисление явно: dloss/dθ по переменным модели,
    затем умножение на transform.jacobian. Применение φ к модели перед
    прогоном обязательно в обоих режимах: иначе значение потери и её
    градиент окажутся вычислены в разных точках.

Оба режима дают одинаковый градиент (проверяется taylor_test'ом);
'assign' сохранён как публичный контракт на случай будущих версий TF,
где literal-путь заработает.

Ограничения P0: eager-режим (без tf.function вокруг методов задачи);
batch=1 (контракт NetworkRNN); записи ParameterSpace без привязанной
переменной (ручная регистрация через add_entry) не включаются в градиент —
их вклад тождественно нулевой.
"""

import time
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

import neuraltide.config
from neuraltide.core.network import NetworkOutput, NetworkRNN
from neuraltide.core.types import TensorType, cast_if_needed
from neuraltide.optimization.parameters import ParameterSpace
from neuraltide.optimization.results import Evaluation

__all__ = ["CalibrationProblem"]

_GRAD_MODES = ("assign", "chain")


class CalibrationProblem:
    """
    Задача калибровки: сеть + параметры + протокол + целевая функция.

    Args:
        network: NetworkRNN.
        parameter_space: ParameterSpace, построенный от этой сети
            (ParameterSpace.from_network).
        t_sequence: временная последовательность [batch, T, 1] или
            [batch, T] (мс).
        inputs: упакованные внешние входы [batch, T, total_input_units]
            или None, если входов нет.
        objective: callable(predictions, network) -> scalar —
            ResidualLoss, CompositeLoss, BaseLoss или пользовательская
            функция; None допускается, если objective будет задан позже
            (присваиванием problem.objective); методы вычисления
            потребуют его понятной ошибкой.
        initial_state: опциональное начальное состояние (pop, syn).
        grad_mode: 'assign' (по умолчанию) или 'chain' — см. модульную
            документацию.

    Счётчики n_forward / n_gradient и накопленное wall_time хранятся на
    объекте и копируются в каждый Evaluation.
    """

    def __init__(
        self,
        network: NetworkRNN,
        parameter_space: ParameterSpace,
        t_sequence: TensorType,
        inputs: Optional[TensorType] = None,
        objective: Optional[Any] = None,
        initial_state: Optional[Tuple[Sequence[TensorType], Sequence[TensorType]]] = (
            None
        ),
        grad_mode: str = "assign",
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

        self.n_forward: int = 0
        self.n_gradient: int = 0
        self.wall_time: float = 0.0

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
        проверяется на размер и приводится к глобальному dtype.
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
        self.n_forward += 1
        return phi_t

    def _forward(self, phi: Optional[Any]) -> Tuple[tf.Tensor, NetworkOutput]:
        """pack (+apply) и прямой прогон; увеличивает n_forward."""
        phi_t = self._prepare(phi)
        output = self.network(
            self.t_sequence,
            inputs=self.inputs,
            initial_state=self.initial_state,
        )
        return phi_t, output

    # ------------------------------------------------------------------
    # Градиент по φ
    # ------------------------------------------------------------------
    def _chain_gradient(
        self, tape: tf.GradientTape, loss_t: tf.Tensor, phi_t: tf.Tensor
    ) -> tf.Tensor:
        """
        dloss/dφ = dloss/dθ (по переменным модели) × dθ/dφ (jacobians).

        Записи ParameterSpace без переменной дают нулевой вклад.
        """
        dtype = neuraltide.config.get_dtype()
        entries = self.parameter_space.entries
        var_names = [
            name for name, entry in entries.items() if entry.variable is not None
        ]
        grad_by_name: Dict[str, Optional[tf.Tensor]] = {}
        if var_names:
            grads = tape.gradient(loss_t, [entries[n].variable for n in var_names])
            grad_by_name = dict(zip(var_names, grads))

        pieces = []
        start = 0
        for name, entry in entries.items():
            numel = entry.numel
            phi_piece = phi_t[start : start + numel]
            g_var = grad_by_name.get(name)
            if g_var is None:
                contrib = tf.zeros([numel], dtype=dtype)
            else:
                jac = entry.transform.jacobian(tf.reshape(phi_piece, entry.shape))
                contrib = tf.reshape(g_var, [-1]) * tf.reshape(jac, [-1])
            pieces.append(contrib)
            start += numel
        if not pieces:
            return tf.zeros([0], dtype=dtype)
        return tf.concat(pieces, axis=0)

    def _phi_gradient(
        self, tape: tf.GradientTape, loss_t: tf.Tensor, phi_t: tf.Tensor
    ) -> tf.Tensor:
        if self.grad_mode == "assign":
            # Literal-путь: в eager TF градиент через variable.assign
            # недоступен (None) — тогда эквивалентная цепочка.
            g = tape.gradient(loss_t, phi_t)
            if g is not None:
                return g
        return self._chain_gradient(tape, loss_t, phi_t)

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
            output = self.network(
                self.t_sequence,
                inputs=self.inputs,
                initial_state=self.initial_state,
            )
            loss_t = self.objective(output, self.network)
        grad_t = self._phi_gradient(tape, loss_t, phi_t)
        # persistent-лента больше не нужна — явно освобождаем.
        del tape
        self.n_gradient += 1
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

    # ------------------------------------------------------------------
    # Вспомогательные bounds в координатах φ
    # ------------------------------------------------------------------
    def bounds_phi(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Границы вектора φ из физических min/max записей ParameterSpace.

        Записи без границ дают -inf / +inf. Границы преобразуются через
        inverse соответствующего transform'а; предназначены для будущих
        SciPy-адаптеров (L-BFGS-B / TRF) — сами по себе задачу не ограничивают.
        """
        lower: list = []
        upper: list = []
        dtype = neuraltide.config.get_dtype()
        for entry in self.parameter_space.entries.values():
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

        components_fn = getattr(self.objective, "components", None)
        if components_fn is not None:
            components = {
                str(k): float(v) for k, v in components_fn(output, self.network).items()
            }
        else:
            components = {}

        residual_fn = getattr(self.objective, "residual", None)
        if residual_fn is not None:
            residuals = np.asarray(residual_fn(output))
            status = "ok"
        else:
            residuals = np.zeros(0)
            status = "no_residual_interface"

        elapsed = time.perf_counter() - t0
        self.wall_time += elapsed

        return Evaluation(
            loss=float(loss_t),
            components=components,
            residuals=residuals,
            grad=None if grad is None else np.asarray(grad),
            status=status,
            n_forward=self.n_forward,
            n_gradient=self.n_gradient,
            wall_time=elapsed,
            dtype=str(neuraltide.config.get_dtype().name),
            dt=float(self.network._graph.dt),
        )
