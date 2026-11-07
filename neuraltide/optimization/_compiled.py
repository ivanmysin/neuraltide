"""
Графовое ядро compiled=True для задач калибровки (этап 1b).

_CompiledCore — один скомпилированный (tf.function) прогон+лента для
фиксированной комбинации (задача × эксперимент × objective × протокол
времени/входов). Мутация модели (variable.assign через
parameter_space.apply_to_model), счётчики и numpy-сборка Evaluation
остаются снаружи в eager: ядро читает переменные модели внутри графа и
принимает φ и начальное состояние протокола аргументами (in-place смена
problem.initial_state/exp.initial_state отражается немедленно — материализация
снаружи при каждом вызове). Трассировка — одна на ядро; владелец (задача)
кэширует ядра по (experiment, objective, протокол времени/входов) и строит
их лениво при первом вызове.
"""

from typing import Optional

import tensorflow as tf

import neuraltide.config
from neuraltide.core.network import NetworkRNN
from neuraltide.optimization._state import (
    _broadcast_or_validate_init,
    _materialize_initial_state,
    _phi_gradient,
)
from neuraltide.optimization.parameters import ParameterSpace


class _CompiledCore:
    """
    Скомпилированное ядро одной (задача × эксперимент × objective ×
    протокол времени/входов)-связки.

    Args:
        network: NetworkRNN (прогон внутри графа читает её переменные —
            мутация модели делается снаружи, до вызова ядра).
        parameter_space: ParameterSpace — unpack(phi) и _phi_gradient
            внутри графа.
        t_sequence / inputs: протокол времени/входов; захватываются
            замыканием (фиксирован у задачи, не аргументы ядра).
        objective: callable(output, network) -> scalar; components/
            residual берутся с того же output внутри графа.
        experiment_name: имя эксперимента для exp-specific x0 (None —
            внеэкспериментный контекст).
        grad_mode: 'assign' | 'chain' — см. _phi_gradient.
        chunk_length / chunk_grad: параметры чанкинга прогона (см.
            NetworkRNN.call); входят в семантику ядра — владелец
            включает их в ключ кэша.
        x0_substitute: опциональная замена материализатора x0 сигнатуры
            (network, space, theta, init_pop, init_syn) ->
            (initial_state, x0_map). По умолчанию — одноэкспериментный
            _materialize_initial_state; векторный режим MultiExperimentProblem
            передаёт батч-подстановку (_make_vectorized_x0_substitute), т.к.
            x0-подстановка должна выполняться внутри графа ядра.

    После build() доступны два tf.function (обычные, без jit_compile):
        eval_core(phi, init_pop..., init_syn...) -> (loss, comps, residual)
        grad_core(phi, init_pop..., init_syn...) -> (loss, grad, comps, residual)

    Аргументы init_pop/init_syn — явное начальное состояние протокола
    (полные кортежи тензоров, материализованные владельцем снаружи при
    каждом вызове): x0-записи ParameterSpace подставляются поверх них
    внутри графа, как в eager-ветках задач.
    """

    def __init__(
        self,
        network: NetworkRNN,
        parameter_space: ParameterSpace,
        t_sequence,
        inputs,
        objective,
        experiment_name: Optional[str] = None,
        grad_mode: str = "chain",
        x0_substitute=None,
        chunk_length: Optional[int] = None,
        chunk_grad: str = "checkpointed",
    ):
        self.network = network
        self.parameter_space = parameter_space
        self.t_sequence = t_sequence
        self.inputs = inputs
        self.objective = objective
        self.experiment_name = experiment_name
        self.grad_mode = grad_mode
        self.x0_substitute = x0_substitute
        self.chunk_length = chunk_length
        self.chunk_grad = chunk_grad
        self.eval_core = None
        self.grad_core = None

    def build(self) -> None:
        """Создаёт tf.function ядра (одна трассировка при первом вызове)."""
        if self.eval_core is not None:
            return
        network = self.network
        space = self.parameter_space
        t_sequence = self.t_sequence
        inputs = self.inputs
        objective = self.objective
        experiment = self.experiment_name
        grad_mode = self.grad_mode
        x0_substitute = self.x0_substitute
        chunk_length = self.chunk_length
        chunk_grad = self.chunk_grad
        dtype = neuraltide.config.get_dtype()

        def _materialize(theta, init_pop, init_syn):
            # x0-записи ParameterSpace подставляются поверх явного начального
            # состояния внутри графа (дифференцируемо по φ). Батч-семантика
            # начального состояния — общая с eager (_broadcast_or_validate_init
            # внутри подстановщиков).
            if x0_substitute is not None:
                return x0_substitute(network, space, theta, init_pop, init_syn)
            return _materialize_initial_state(
                network, space, theta, (init_pop, init_syn), experiment=experiment
            )

        def _comps_and_residual(output):
            components_fn = getattr(objective, "components", None)
            comps = (
                {} if components_fn is None else dict(components_fn(output, network))
            )
            residual_fn = getattr(objective, "residual", None)
            # objective может декларировать отсутствие residual-интерфейса
            # флагом has_residual (векторная обёртка MultiExperimentProblem).
            if residual_fn is None or getattr(objective, "has_residual", True) is False:
                residual = tf.zeros([0], dtype=dtype)
            else:
                residual = residual_fn(output)
            return comps, residual

        def _eval_fn(phi, init_pop, init_syn):
            batch = tf.shape(t_sequence)[0]
            init_pop, init_syn = _broadcast_or_validate_init(
                (init_pop, init_syn),
                batch,
                what="compiled core initial_state",
            )
            theta = space.unpack(phi)
            initial_state, _ = _materialize(theta, init_pop, init_syn)
            output = network(
                t_sequence,
                inputs=inputs,
                initial_state=initial_state,
                chunk_length=chunk_length,
                chunk_grad=chunk_grad,
            )
            loss = objective(output, network)
            comps, residual = _comps_and_residual(output)
            return loss, comps, residual

        def _grad_fn(phi, init_pop, init_syn):
            batch = tf.shape(t_sequence)[0]
            init_pop, init_syn = _broadcast_or_validate_init(
                (init_pop, init_syn),
                batch,
                what="compiled core initial_state",
            )
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(phi)
                theta = space.unpack(phi)
                initial_state, x0_map = _materialize(theta, init_pop, init_syn)
                output = network(
                    t_sequence,
                    inputs=inputs,
                    initial_state=initial_state,
                    chunk_length=chunk_length,
                    chunk_grad=chunk_grad,
                )
                loss = objective(output, network)
                comps, residual = _comps_and_residual(output)
            grad = _phi_gradient(space, grad_mode, tape, loss, phi, x0_map)
            # persistent-лента больше не нужна — явно освобождаем.
            del tape
            return loss, grad, comps, residual

        self.eval_core = tf.function(_eval_fn)
        self.grad_core = tf.function(_grad_fn)
