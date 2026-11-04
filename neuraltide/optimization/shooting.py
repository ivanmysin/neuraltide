"""
Multiple shooting поверх CalibrationProblem (этап P1).

MultipleShootingProblem оборачивает базовую задачу калибровки и разбивает
временной горизонт на сегменты с независимыми начальными состояниями
``s_k`` (свободные shooting-переменные), общими физическими параметрами и
continuity-штрафом на стыках:

    loss_ms(φ) = L_data(stitched trajectory) + 0.5·w·Σ_k ||x_end,k − s_{k+1}||²,

где stitched trajectory — конкатенация firing-rate траекторий сегментов,
``w = continuity_weight``. Данный член data loss вычисляется БАЗОВЫМ
objective на склеенной траектории, поэтому при нулевых разрывах
``loss_ms`` совпадает с потерей честного свободного single-shooting прогона
(см. ``finalize``).

Вектор φ расширенного пространства:

    φ = [φ_base; vec(s_0); vec(s_1); ...; vec(s_{K-1})]

с записями вида ``x0s/<k>/<pop>/<slot>`` (kind='shooting_state',
locator=StateLocator, shape [1, n_units] слота состояния, transforms
по умолчанию identity). Свой ParameterSpace = пространство базовой задачи
плюс shooting-записи; trainable-записи базового пространства разделяются
по ссылке (те же tf.Variable), а его non-trainable записи реестра (вне
вектора φ) не копируются. Сегмент k стартует из s_k: записи подставляются в
начальное состояние сегмента (дополняя/переопределяя начальное состояние
протокола и x0-записи базового пространства — shooting-записи имеют
приоритет на своём сегменте). При ``fix_first_segment=True`` s_0 не
вводится: записи ``x0s/0/...`` отсутствуют, сегмент 0 стартует из
начального состояния базовой задачи (base initial state с общими
x0-записями базового пространства), стыки 0->1, 1->2, ... сохраняются
и φ = [φ_base; vec(s_1); ...; vec(s_{K-1})]. Параметр обратно совместим:
по умолчанию False и прежнее поведение бит-в-бит.

Duck-type API повторяет CalibrationProblem: size / pack / unpack /
bounds_phi / evaluate / value_and_gradient / residual / validate
(делегируется базовой задаче) + специфичные:
simulate_segment(phi, k), continuity_gaps(phi), initialize_from_
single_shooting(), finalize(phi), to_single_shooting_problem(phi).

Residual layout (фиксирован): r = [r_data; √w·gap_0; ...; √w·gap_{K-2}],
где r_data = objective.residual(stitched_output). Тогда
0.5·||r||² == loss_ms точно (objective обязан давать честный residual,
иначе residual() поднимает TypeError со ссылкой на ResidualLoss; скалярный
evaluate для общих objective работает — см. ниже).

Градиент считается единой GradientTape: сегментные прогоны + стыковка +
continuity-член, затем цепное правило через хелперы CORE
(``_chain_gradient``, ``_materialize_initial_state``) — градиент по
φ_base идёт через переменные модели и x0-записи базового пространства,
по shooting-координатам — через подставленные тензоры начальных состояний
× jacobian transform'ов. Режим 'assign' недоступен (модельные переменные
+ tensor-записи одновременно) — всегда используется chain-путь.

Отклонения и пробелы CORE (обходятся локально, CORE не меняется):
  * ``_materialize_initial_state`` подставляет только pop-слоты; для
    syn-слотов (include_syn_states=True) подстановка выполняется
    собственным хелпером по тем же конвенциям (flat-список syn-состояний
    в порядке graph.synapse_names);
  * публичного доступа к syn-offsets нет — вычисляются локально
    (зеркалят NetworkRNN._build);
  * синтез начального состояния сегмента как функции от φ и части
    chain-градиента по shooting-записям реализованы в модуле поверх
    тех же примитивов ParameterSpace.

Честный отказ: при наличии в графе BrunelFokkerPlanck конструктор
поднимает NotImplementedError (буфер задержки ν(t−τ_rp) и внутреннее
CFL-субшагивание несовместимы с точной сегментацией горизонта).

Контракт мутации модели унаследован от CalibrationProblem: phi=None не
трогает модель (pack(None) при shooting-записях невозможен — передайте
явный phi); phi != None переводит модель в θ = unpack_base(φ) перед
прогоном. Счётчики n_forward / n_gradient / wall_time общие с базовой
задачей (shared dict); каждый сегментный прогон считается за единицу
n_forward. Eager-режим, batch=1, dtype — через neuraltide.config.
"""

import time
import warnings
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf

import neuraltide.config
from neuraltide.core.network import NetworkOutput
from neuraltide.core.types import TensorType, cast_if_needed
from neuraltide.optimization.parameters import (
    ParameterEntry,
    ParameterSpace,
    StateLocator,
)
from neuraltide.optimization.problem import (
    SEARCH_BOX_REL_MARGIN,
    CalibrationProblem,
    _bounds_phi_from_space,
    _chain_gradient,
    _make_evaluation,
    _materialize_initial_state,
    _search_box_phi_from_space,
)
from neuraltide.optimization.results import Evaluation
from neuraltide.optimization.transforms import IdentityTransform
from neuraltide.populations import BrunelFokkerPlanck

__all__ = [
    "ShootingConfig",
    "SingleShootingCheck",
    "MultipleShootingProblem",
]

# Популяции, несовместимые с точной сегментацией прогона (явный список).
_UNSUPPORTED_POPULATIONS = (BrunelFokkerPlanck,)

_GAP_ZERO = {"max": 0.0, "mean": 0.0, "l2": 0.0}


def _syn_state_offsets(network) -> Dict[str, Tuple[int, int]]:
    """(start, count) каждого синапса в плоском списке syn-состояний.

    Публичного аналога pop_state_offsets для синапсов в CORE нет;
    вычисление зеркалит NetworkRNN._build по graph.synapse_names.
    """
    offsets: Dict[str, Tuple[int, int]] = {}
    idx = 0
    graph = network._graph
    for name in graph.synapse_names:
        count = len(graph._synapses[name].model.state_size)
        offsets[name] = (idx, count)
        idx += count
    return offsets


def _gap_stats(arr: np.ndarray) -> Dict[str, float]:
    """max/mean/L2 статистика разрывов одного стыка (abs-максимумы)."""
    if arr.size == 0:
        return dict(_GAP_ZERO)
    abs_arr = np.abs(arr)
    return {
        "max": float(np.max(abs_arr)),
        "mean": float(np.mean(abs_arr)),
        "l2": float(np.sqrt(np.sum(arr**2))),
    }


def _gap_dict(gap_arrays: Sequence[np.ndarray]) -> Dict[str, Any]:
    """Агрегаты разрывов по всем стыкам + per-joint статистика."""
    joints: Dict[str, Dict[str, float]] = {}
    pieces: List[np.ndarray] = []
    for k, arr in enumerate(gap_arrays):
        flat = np.asarray(arr, dtype=np.float64).reshape(-1)
        joints[f"{k}->{k + 1}"] = _gap_stats(flat)
        pieces.append(flat)
    if pieces:
        all_gaps = np.concatenate(pieces)
        out = _gap_stats(all_gaps)
        out["finite"] = bool(np.all(np.isfinite(all_gaps)))
    else:
        out = dict(_GAP_ZERO)
        out["finite"] = True
    out["joints"] = joints
    return out


@dataclass
class ShootingConfig:
    """
    Узлы сегментации временной оси.

    Атрибуты:
        nodes: int — число сегментов с равномерными границами, либо
            Sequence[int] — явные индексы времени границ (включая 0 и T).

    Валидация (в resolve): первая граница 0, последняя T, строгое
    возрастание; для равномерного разбиения требуется nodes >= 1 и
    nodes <= T (иначе возникают пустые сегменты).
    """

    nodes: Union[int, Sequence[int]]

    def resolve(self, T: int) -> List[int]:
        """Возвращает список границ [0, ..., T] для горизонта T шагов."""
        if isinstance(self.nodes, Integral):
            n = int(self.nodes)
            if n < 1:
                raise ValueError(
                    f"ShootingConfig: nodes must be a positive integer, got {n}."
                )
            bounds = [(i * T) // n for i in range(n + 1)]
        else:
            bounds = [int(b) for b in self.nodes]
            if len(bounds) < 2:
                raise ValueError(
                    "ShootingConfig: explicit nodes must contain at least two "
                    f"indices (0 and T), got {bounds}."
                )
        if bounds[0] != 0:
            raise ValueError(f"ShootingConfig: first node must be 0, got {bounds[0]}.")
        if bounds[-1] != T:
            raise ValueError(
                f"ShootingConfig: last node must be T={T}, got {bounds[-1]}."
            )
        for prev, cur in zip(bounds, bounds[1:]):
            if cur <= prev:
                raise ValueError(
                    f"ShootingConfig: nodes must be strictly increasing, got {bounds}."
                )
        return bounds


@dataclass
class SingleShootingCheck:
    """
    Результат финальной проверки multiple-shooting решения.

    Атрибуты:
        loss_ms: потеря multiple shooting в точке φ;
        loss_single: потеря честного свободного single-shooting прогона
            (полный горизонт, x0 = s_0, те же параметры);
        gaps: словарь continuity_gaps (max/mean/l2/finite/joints);
        consistent: True, если разрывы не превосходят допуска, обе
            потери конечны и совпадают в пределах допуска — тогда
            MS-траектория является траекторией единого прогона.
    """

    loss_ms: float
    loss_single: float
    gaps: Dict[str, Any]
    consistent: bool


@dataclass
class _SegmentProtocol:
    """Протокол одного сегмента: срезы времени и входов."""

    t_sequence: TensorType
    inputs: Optional[TensorType]


class MultipleShootingProblem:
    """
    Задача калибровки в постановке multiple shooting.

    Args:
        problem: базовая CalibrationProblem (та же сеть/пространство).
        config: ShootingConfig или сырой nodes (int | Sequence[int]).
        continuity_weight: неотрицательный вес continuity-штрафа w.
        include_syn_states: включать syn-слоты в shooting-переменные и
            continuity-разрывы (по умолчанию True). При False каждый сегмент
            стартует с начального состояния синапсов протокола (для
            TsodyksMarkram — R = 1, U = 0, A = 0), склейка не совпадает со
            свободным прогоном, и при синапсах с состоянием выдаётся
            предупреждение.
        fix_first_segment: фиксировать начальное состояние сегмента 0 за
            начальным состоянием базовой задачи вместо свободной
            shooting-переменной s_0 (по умолчанию False). При True записи
            ``x0s/0/...`` не создаются, размерность φ уменьшается ровно на
            число слотов первого сегмента, стыки 0->1, 1->2, ... и
            finalize/to_single_shooting_problem не меняются (single
            shooting стартует из base_state/общих x0 базовой задачи).

    Duck-type API совместим с CalibrationProblem; градиент всегда
    chain-режимом (см. модульную документацию).
    """

    def __init__(
        self,
        problem: CalibrationProblem,
        config: Union[ShootingConfig, int, Sequence[int]],
        *,
        continuity_weight: float = 1.0,
        include_syn_states: bool = True,
        fix_first_segment: bool = False,
    ):
        if not isinstance(config, ShootingConfig):
            config = ShootingConfig(config)
        if continuity_weight < 0.0:
            raise ValueError(
                "MultipleShootingProblem: continuity_weight must be "
                f"non-negative, got {continuity_weight}."
            )
        self._problem = problem
        self.config = config
        self.continuity_weight = float(continuity_weight)
        self.include_syn_states = bool(include_syn_states)
        self.fix_first_segment = bool(fix_first_segment)
        self.network = problem.network
        if not self.include_syn_states:
            stateful = [
                name
                for name in self.network._graph.synapse_names
                if self.network._graph._synapses[name].model.state_size
            ]
            if stateful:
                warnings.warn(
                    "MultipleShootingProblem: include_syn_states=False при "
                    f"синапсах с состоянием {stateful}: их состояние "
                    "сбрасывается к начальному на каждом узле, склейка не "
                    "совпадает со свободным прогоном (finalize это покажет).",
                    UserWarning,
                    stacklevel=2,
                )
        self._base_space = problem.parameter_space
        self._base_size = self._base_space.size

        graph = self.network._graph
        for pop_name in graph.dynamic_population_names:
            pop = graph._populations[pop_name]
            if isinstance(pop, _UNSUPPORTED_POPULATIONS):
                raise NotImplementedError(
                    "MultipleShootingProblem: популяция "
                    f"'{pop_name}' типа {type(pop).__name__} не поддерживается "
                    "— буфер задержки ν(t−τ_rp) и внутреннее CFL-"
                    "субшагивание несовместимы с точной сегментацией "
                    "горизонта. Используйте single shooting."
                )

        t_seq = problem.t_sequence
        if t_seq.shape.rank == 2:
            t_seq = t_seq[:, :, tf.newaxis]
        t_steps = t_seq.shape[1]
        if t_steps is None:
            raise ValueError(
                "MultipleShootingProblem: t_sequence must have a statically "
                "known number of time steps (shape[1])."
            )
        self._T = int(t_steps)
        self.nodes = config.resolve(self._T)
        self._segments: List[Tuple[int, int]] = list(zip(self.nodes, self.nodes[1:]))
        self.n_segments = len(self._segments)

        # Протоколы сегментов: через view-хелперы CORE (общие счётчики),
        # при недоступности time_slice — прямой срез тензоров.
        slicer = getattr(problem, "time_slice", None)
        self._segment_views: List[_SegmentProtocol] = []
        for start, stop in self._segments:
            if callable(slicer):
                view = slicer(start, stop)
                self._segment_views.append(
                    _SegmentProtocol(t_sequence=view.t_sequence, inputs=view.inputs)
                )
            else:
                inputs = (
                    None if problem.inputs is None else problem.inputs[:, start:stop]
                )
                self._segment_views.append(
                    _SegmentProtocol(t_sequence=t_seq[:, start:stop], inputs=inputs)
                )

        # Расширенное пространство: trainable-записи базового пространства
        # (по ссылке) + shooting. Non-trainable записи реестра базового
        # пространства в φ не входят и сюда не копируются.
        self._space = ParameterSpace()
        for name, entry in self._base_space.phi_entries.items():
            self._space.add_entry(entry)
        self._shooting: List[List[Tuple[str, ParameterEntry]]] = []
        self._subst: List[List[Tuple[str, ParameterEntry, bool, int]]] = []
        self._phi_index: Dict[str, Tuple[int, int]] = {}
        pos = self._base_size
        pop_offsets = self.network.pop_state_offsets
        syn_offsets = (
            _syn_state_offsets(self.network) if self.include_syn_states else {}
        )
        for k in range(self.n_segments):
            entries_k: List[Tuple[str, ParameterEntry]] = []
            subst_k: List[Tuple[str, ParameterEntry, bool, int]] = []
            if not (self.fix_first_segment and k == 0):
                for pop_name in graph.dynamic_population_names:
                    pop = graph._populations[pop_name]
                    start, count = pop_offsets[pop_name]
                    for slot in range(count):
                        shape = tf.TensorShape(pop.state_size[slot])
                        if shape.num_elements() == 0:
                            continue
                        name = f"x0s/{k}/{pop_name}/{slot}"
                        entry = ParameterEntry(
                            full_name=name,
                            shape=shape,
                            transform=IdentityTransform(),
                            group="shooting_state",
                            trainable=True,
                            variable=None,
                            kind="shooting_state",
                            locator=StateLocator(population=pop_name, slot=slot),
                        )
                        entries_k.append((name, entry))
                        subst_k.append((name, entry, False, start + slot))
                for syn_name, (start, count) in syn_offsets.items():
                    if syn_name in graph._populations:
                        raise ValueError(
                            "MultipleShootingProblem: имя синапса "
                            f"'{syn_name}' совпадает с именем популяции — "
                            "имена shooting-записей неоднозначны."
                        )
                    model = graph._synapses[syn_name].model
                    for slot in range(count):
                        shape = tf.TensorShape(model.state_size[slot])
                        if shape.num_elements() == 0:
                            continue
                        name = f"x0s/{k}/{syn_name}/{slot}"
                        entry = ParameterEntry(
                            full_name=name,
                            shape=shape,
                            transform=IdentityTransform(),
                            group="shooting_state",
                            trainable=True,
                            variable=None,
                            kind="shooting_state",
                            locator=StateLocator(population=syn_name, slot=slot),
                        )
                        entries_k.append((name, entry))
                        subst_k.append((name, entry, True, start + slot))
            for name, entry in entries_k:
                self._space.add_entry(entry)
                self._phi_index[name] = (pos, entry.numel)
                pos += entry.numel
            self._shooting.append(entries_k)
            self._subst.append(subst_k)

        counters = getattr(problem, "_counters", None)
        self._counters: Dict[str, Any] = (
            counters
            if counters is not None
            else {"n_forward": 0, "n_gradient": 0, "wall_time": 0.0}
        )

    # ------------------------------------------------------------------
    # Свойства (duck parity с CalibrationProblem)
    # ------------------------------------------------------------------
    @property
    def problem(self) -> CalibrationProblem:
        """Базовая задача."""
        return self._problem

    @property
    def parameter_space(self) -> ParameterSpace:
        """Расширенное пространство (base + shooting-записи)."""
        return self._space

    @property
    def objective(self):
        return self._problem.objective

    @objective.setter
    def objective(self, value) -> None:
        self._problem.objective = value

    @property
    def t_sequence(self) -> TensorType:
        return self._problem.t_sequence

    @property
    def inputs(self) -> Optional[TensorType]:
        return self._problem.inputs

    @property
    def size(self) -> int:
        """Размерность вектора φ (base + shooting)."""
        return self._space.size

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
    # pack / unpack / bounds
    # ------------------------------------------------------------------
    def pack(self, phi: Optional[Any] = None) -> tf.Tensor:
        """
        φ-вектор в координатах расширенного ParameterSpace.

        phi=None требует переменных модели для всех записей; shooting-
        записей переменных нет — передайте явный phi (контракт как у
        tensor-записей базового пространства).
        """
        if phi is None:
            return self._space.pack()
        dtype = neuraltide.config.get_dtype()
        phi_t = cast_if_needed(tf.convert_to_tensor(phi), dtype)
        phi_t = tf.reshape(phi_t, [-1])
        if phi_t.shape[0] is not None and int(phi_t.shape[0]) != self.size:
            raise ValueError(
                f"MultipleShootingProblem: phi has {int(phi_t.shape[0])} "
                f"elements, expected {self.size}."
            )
        tf.debugging.assert_equal(
            tf.size(phi_t),
            self.size,
            message="MultipleShootingProblem: bad phi size",
        )
        return phi_t

    def unpack(self, phi: Optional[Any] = None) -> Dict[str, tf.Tensor]:
        """φ → физические θ всех записей (base + shooting)."""
        return self._space.unpack(self.pack(phi))

    def bounds_phi(self) -> Tuple[np.ndarray, np.ndarray]:
        """Границы φ; shooting-записи без физических границ → ±inf."""
        return _bounds_phi_from_space(self._space)

    def search_box_phi(
        self, rel_margin: float = SEARCH_BOX_REL_MARGIN
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Поисковый бокс φ для глобальных методов (см. problem.py)."""
        return _search_box_phi_from_space(self._space, rel_margin)

    # ------------------------------------------------------------------
    # Внутренние хелперы
    # ------------------------------------------------------------------
    def _require_objective(self) -> None:
        if self.objective is None:
            raise ValueError(
                "MultipleShootingProblem: objective is None. Передайте "
                "objective базовой задаче перед вызовом."
            )

    def _apply_base_phi(self, phi_t: tf.Tensor) -> None:
        """Модель := θ_base = unpack_base(φ[:base_size]) (контракт мутации)."""
        theta = self._base_space.unpack(phi_t[: self._base_size])
        self._base_space.apply_to_model(theta)

    def _base_state(self, phi_t: tf.Tensor):
        """Начальное состояние протокола с подставленными x0-записями."""
        theta_base = self._base_space.unpack(phi_t[: self._base_size])
        base_state, base_map = _materialize_initial_state(
            self.network, self._base_space, theta_base, self._problem.initial_state
        )
        if base_state is None:
            pop0, syn0 = self.network.get_initial_state(batch_size=1)
            base_state = (list(pop0), list(syn0))
        return base_state, base_map

    def _segment_state(self, k: int, phi_t: tf.Tensor, base_state):
        """
        Начальное состояние сегмента k: base_state с подставленными
        shooting-значениями s_k как дифференцируемыми функциями от φ.
        Для фиксированного сегмента 0 (fix_first_segment=True) subst-пусто:
        возвращается base_state без подстановок.
        """
        dtype = neuraltide.config.get_dtype()
        init_pop = list(base_state[0])
        init_syn = list(base_state[1])
        x0_map: Dict[str, tf.Tensor] = {}
        for name, entry, is_syn, idx in self._subst[k]:
            start, numel = self._phi_index[name]
            piece = tf.reshape(phi_t[start : start + numel], entry.shape)
            value = cast_if_needed(entry.transform.forward(piece), dtype)
            if is_syn:
                init_syn[idx] = value
            else:
                init_pop[idx] = value
            x0_map[name] = value
        return (init_pop, init_syn), x0_map

    def _gap_tensors(self, seg_outputs, start_states) -> List[tf.Tensor]:
        """Плоские разрывы x_end,k − s_{k+1} по зарегистрированным слотам."""
        dtype = neuraltide.config.get_dtype()
        gaps: List[tf.Tensor] = []
        for k in range(self.n_segments - 1):
            end_pop, end_syn = seg_outputs[k].final_state
            nxt_pop, nxt_syn = start_states[k + 1]
            pieces = []
            for _name, _entry, is_syn, idx in self._subst[k + 1]:
                x_end = (end_syn if is_syn else end_pop)[idx]
                x_start = (nxt_syn if is_syn else nxt_pop)[idx]
                pieces.append(tf.reshape(x_end - x_start, [-1]))
            if pieces:
                gaps.append(tf.concat(pieces, axis=0))
            else:
                gaps.append(tf.zeros([0], dtype=dtype))
        return gaps

    def _forward_core(self, phi_t: tf.Tensor):
        """
        Сегментированный прогон (вызывать внутри GradientTape для
        градиента). Возвращает (loss_t, stitched, gaps, x0_map, outputs).
        """
        dtype = neuraltide.config.get_dtype()
        self._counters["n_forward"] += self.n_segments
        base_state, base_map = self._base_state(phi_t)
        x0_map: Dict[str, tf.Tensor] = dict(base_map)
        start_states: List[Tuple[List, List]] = []
        seg_outputs: List[NetworkOutput] = []
        for k in range(self.n_segments):
            state_k, x0_k = self._segment_state(k, phi_t, base_state)
            x0_map.update(x0_k)
            start_states.append(state_k)
            proto = self._segment_views[k]
            out = self.network(
                proto.t_sequence,
                inputs=proto.inputs,
                initial_state=state_k,
            )
            seg_outputs.append(out)
        stitched_rates = {
            name: tf.concat([out.firing_rates[name] for out in seg_outputs], axis=1)
            for name in seg_outputs[0].firing_rates
        }
        stitched = NetworkOutput(
            firing_rates=stitched_rates,
            stability_loss=tf.add_n([out.stability_loss for out in seg_outputs]),
            final_state=seg_outputs[-1].final_state,
        )
        gaps = self._gap_tensors(seg_outputs, start_states)
        cont = tf.constant(0.0, dtype=dtype)
        for gap in gaps:
            cont = cont + 0.5 * self.continuity_weight * tf.reduce_sum(gap * gap)
        if self.objective is None:
            data_loss = tf.constant(0.0, dtype=dtype)
        else:
            data_loss = self.objective(stitched, self.network)
        loss_t = data_loss + cont
        return loss_t, stitched, gaps, x0_map, seg_outputs

    def _build_evaluation(
        self,
        t0: float,
        loss_t: tf.Tensor,
        stitched: NetworkOutput,
        gaps: Sequence[tf.Tensor],
        grad: Optional[tf.Tensor],
    ) -> Evaluation:
        components: Dict[str, float] = {}
        comp_fn = getattr(self.objective, "components", None)
        if callable(comp_fn):
            components = {
                str(k): float(v) for k, v in comp_fn(stitched, self.network).items()
            }
        gap_info = _gap_dict([np.asarray(g, dtype=np.float64) for g in gaps])
        components["continuity/max"] = gap_info["max"]
        components["continuity/mean"] = gap_info["mean"]
        components["continuity/l2"] = gap_info["l2"]
        residual_fn = getattr(self.objective, "residual", None)
        if callable(residual_fn):
            residuals = np.asarray(residual_fn(stitched))
            status = "ok"
        else:
            residuals = np.zeros(0)
            status = "no_residual_interface"
        loss_float = float(loss_t)
        if not np.isfinite(loss_float):
            status = "non_finite"
        return _make_evaluation(
            loss=loss_float,
            components=components,
            residuals=residuals,
            grad=grad,
            status=status,
            counters=self._counters,
            elapsed=time.perf_counter() - t0,
            dt=float(self.network._graph.dt),
        )

    # ------------------------------------------------------------------
    # Публичные операции (duck parity с CalibrationProblem)
    # ------------------------------------------------------------------
    def evaluate(self, phi: Optional[Any] = None) -> Evaluation:
        """Скалярная потеря multiple shooting, компоненты и невязки."""
        self._require_objective()
        t0 = time.perf_counter()
        phi_t = self.pack(phi)
        if phi is not None:
            self._apply_base_phi(phi_t)
        loss_t, stitched, gaps, _, _ = self._forward_core(phi_t)
        return self._build_evaluation(t0, loss_t, stitched, gaps, grad=None)

    def value_and_gradient(
        self, phi: Optional[Any] = None
    ) -> Tuple[Evaluation, tf.Tensor]:
        """
        Потеря и градиент по φ одним проходом (chain-режим).

        Returns:
            (Evaluation с заполненным grad, grad — tf.Tensor формы [size]).
        """
        self._require_objective()
        t0 = time.perf_counter()
        phi_t = self.pack(phi)
        if phi is not None:
            self._apply_base_phi(phi_t)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(phi_t)
            loss_t, stitched, gaps, x0_map, _ = self._forward_core(phi_t)
        grad_t = _chain_gradient(self._space, tape, loss_t, phi_t, x0_map)
        del tape
        self._counters["n_gradient"] += 1
        eval_result = self._build_evaluation(t0, loss_t, stitched, gaps, grad=grad_t)
        return eval_result, grad_t

    def residual(self, phi: Optional[Any] = None) -> np.ndarray:
        """
        Вектор невязок r = [r_data; √w·gap_0; ...], 0.5·||r||² == loss_ms.

        Требует objective с методом residual (ResidualLoss); иначе TypeError
        со ссылкой на scalar-only путь (evaluate / value_and_gradient).
        """
        self._require_objective()
        residual_fn = getattr(self.objective, "residual", None)
        if not callable(residual_fn):
            raise TypeError(
                "MultipleShootingProblem: objective "
                f"'{type(self.objective).__name__}' не предоставляет "
                "residual(predictions); фиксированный layout "
                "r = [r_data; sqrt(w)·(x_end,k − s_{k+1})] требует "
                "ResidualLoss. Для общего scalar objective используйте "
                "evaluate / value_and_gradient."
            )
        phi_t = self.pack(phi)
        if phi is not None:
            self._apply_base_phi(phi_t)
        _, stitched, gaps, _, _ = self._forward_core(phi_t)
        dtype = neuraltide.config.get_dtype()
        scale = tf.sqrt(tf.constant(self.continuity_weight, dtype))
        pieces = [tf.reshape(residual_fn(stitched), [-1])]
        for gap in gaps:
            pieces.append(scale * tf.cast(gap, dtype))
        return np.asarray(tf.concat(pieces, axis=0))

    def validate(self, phi: Optional[Any] = None) -> Dict[str, float]:
        """Делегируется базовой задаче (held-out протоколы)."""
        if phi is None:
            return self._problem.validate()
        phi_t = self.pack(phi)
        return self._problem.validate(phi_t[: self._base_size])

    def simulate_segment(self, phi: Optional[Any], k: int) -> NetworkOutput:
        """
        Прямой прогон сегмента k из начального состояния s_k(φ).

        phi=None — при текущих значениях модели (shooting-значения при
        этом неизвестны — передайте явный phi).
        """
        if not 0 <= int(k) < self.n_segments:
            raise ValueError(
                f"MultipleShootingProblem: segment index {k} out of range "
                f"[0, {self.n_segments})."
            )
        k = int(k)
        phi_t = self.pack(phi)
        if phi is not None:
            self._apply_base_phi(phi_t)
        base_state, _ = self._base_state(phi_t)
        state_k, _ = self._segment_state(k, phi_t, base_state)
        self._counters["n_forward"] += 1
        proto = self._segment_views[k]
        return self.network(
            proto.t_sequence,
            inputs=proto.inputs,
            initial_state=state_k,
        )

    def continuity_gaps(self, phi: Optional[Any] = None) -> Dict[str, Any]:
        """
        Разрывы стыков: max/mean/L2 по модулям (все стыки вместе и по
        отдельности). Только pop-состояния или pop+syn в зависимости от
        include_syn_states. 'finite' — признак конечности всех разрывов.
        """
        phi_t = self.pack(phi)
        if phi is not None:
            self._apply_base_phi(phi_t)
        _, _, gaps, _, _ = self._forward_core(phi_t)
        return _gap_dict([np.asarray(g, dtype=np.float64) for g in gaps])

    # ------------------------------------------------------------------
    # Инициализация и финализация
    # ------------------------------------------------------------------
    def initialize_from_single_shooting(
        self, phi_base: Optional[Any] = None
    ) -> tf.Tensor:
        """
        Инициализация shooting-состояний цепочкой сегментных прогонов
        (carry final_state → начало следующего сегмента) при параметрах
        базовой задачи. Возвращает готовый вектор φ (gaps ≈ 0).
        Переменные для фиксированного сегмента 0
        (fix_first_segment=True) не создаются, цепочка прогонов
        сохраняется.

        Args:
            phi_base: вектор φ базового пространства; None — текущие
                значения модели (невозможно, если у базового пространства
                есть tensor-записи — передайте phi_base явно).
        """
        if phi_base is None:
            try:
                phi_base_t = self._problem.pack()
            except ValueError as exc:
                raise ValueError(
                    "initialize_from_single_shooting: pack(None) базовой "
                    "задачи невозможен (tensor-записи начального состояния) "
                    "— передайте phi_base явно."
                ) from exc
        else:
            phi_base_t = self._problem.pack(phi_base)
        theta_base = self._base_space.unpack(phi_base_t)
        self._base_space.apply_to_model(theta_base)
        dtype = neuraltide.config.get_dtype()
        base_state, _ = self._base_state(phi_base_t)
        carry = (list(base_state[0]), list(base_state[1]))
        pieces = [
            tf.reshape(cast_if_needed(phi_base_t, dtype), [-1]),
        ]
        for k in range(self.n_segments):
            for name, entry, is_syn, idx in self._subst[k]:
                value = (carry[1] if is_syn else carry[0])[idx]
                phi_piece = entry.transform.inverse(cast_if_needed(value, dtype))
                pieces.append(tf.reshape(phi_piece, [-1]))
            if k < self.n_segments - 1:
                proto = self._segment_views[k]
                out = self.network(
                    proto.t_sequence,
                    inputs=proto.inputs,
                    initial_state=carry,
                )
                self._counters["n_forward"] += 1
                carry = (list(out.final_state[0]), list(out.final_state[1]))
        return tf.concat(pieces, axis=0)

    def to_single_shooting_problem(self, phi: Any) -> CalibrationProblem:
        """
        Задача single shooting на полном горизонте: base problem с
        начальным состоянием x0 = s_0(φ) (константы); при
        fix_first_segment=True s_0 не существует и начальным состоянием
        остаётся base_state (начальное состояние базовой задачи с общими
        x0-записями базового пространства). Модель не изменяется:
        оценивайте результат при phi=None после evaluate(phi) или
        передайте phi_base явно.
        """
        spawn = getattr(self._problem, "_spawn", None)
        if not callable(spawn):
            raise TypeError(
                "MultipleShootingProblem: базовая задача не предоставляет "
                "view-конструктор _spawn — to_single_shooting_problem "
                "поддерживается для CalibrationProblem."
            )
        phi_t = self.pack(phi)
        base_state, _ = self._base_state(phi_t)
        init_pop = list(base_state[0])
        init_syn = list(base_state[1])
        dtype = neuraltide.config.get_dtype()
        for name, entry, is_syn, idx in self._subst[0]:
            start, numel = self._phi_index[name]
            piece = tf.reshape(phi_t[start : start + numel], entry.shape)
            value = tf.constant(np.asarray(entry.transform.forward(piece)), dtype=dtype)
            if is_syn:
                init_syn[idx] = value
            else:
                init_pop[idx] = value
        return spawn(initial_state=(init_pop, init_syn))

    def finalize(
        self,
        phi: Any,
        gap_tol: float = 1e-6,
        loss_rtol: float = 1e-6,
        loss_atol: float = 1e-10,
    ) -> SingleShootingCheck:
        """
        Финальная проверка: свободный single-shooting прогон с x0 = s_0
        на полном горизонте и сравнение потерь/разрывов.

        consistent = разрывы <= gap_tol, обе потери конечны и
        |loss_ms − loss_single| <= loss_atol + loss_rtol·|loss_single|.
        Сравнение потерь ловит расхождение, которое разрывы по отобранным
        слотам не видят (например, сброс состояний синапсов при
        include_syn_states=False после того, как оптимизатор свёл разрывы
        популяций к нулю).
        """
        ev_ms = self.evaluate(phi)
        gaps = self.continuity_gaps(phi)
        single = self.to_single_shooting_problem(phi)
        ev_single = single.evaluate(None)
        losses_match = bool(
            abs(ev_ms.loss - ev_single.loss)
            <= float(loss_atol) + float(loss_rtol) * abs(ev_single.loss)
        )
        consistent = bool(
            gaps["finite"]
            and gaps["max"] <= gap_tol
            and np.isfinite(ev_ms.loss)
            and np.isfinite(ev_single.loss)
            and losses_match
        )
        return SingleShootingCheck(
            loss_ms=ev_ms.loss,
            loss_single=ev_single.loss,
            gaps=gaps,
            consistent=consistent,
        )
