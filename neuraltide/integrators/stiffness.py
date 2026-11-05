"""
Диагностика жёсткости и хелперы network-wide интегрирования.

Модуль предоставляет:

- хелперы уплощения состояния и JVP-якобиана векторного поля популяции
  (общие для Rosenbrock/Exponential-интеграторов и отчёта о жёсткости);
- joint-хелперы для step_network() (integration_mode='network'):
  flatten_joint_state()/unflatten_joint_state() — уплощение объединённого
  состояния сети, joint_block_layout()/joint_dynamic_indices() — «scope»
  блоков популяций/синапсов у _JointNetworkRHS,
  flat_vector_field_jacobian() — плотный якобиан плоского векторного поля
  (RosenbrockIntegrator.step_network);
- stiffness_report() — спектр якобиана правой части в точках траектории,
  stiffness ratio и явный лимит шага Эйлера;
- classify_failure() — классификация сбоя прогона (NaN / расходимость / ок).

Ограничения: в scope='populations' (по умолчанию) спектр вычисляется
для блоков популяций при замороженных синаптических входах
(I_syn/g_syn = 0); межюнитная связность через синапсы в якобиан не входит
— это соответствует линеаризации, используемой самими жёсткими
интеграторами внутри шага. scope='network' линеаризует полную связанную правую часть
(_JointNetworkRHS) по объединённому состоянию сети — тот же базис, что у
joint_vector_field-линеаризации RosenbrockIntegrator.step_network; явный
лимит шага Эйлера в этом scope относится уже к полному спектру связанной
сети. В обоих scope внешние входы заморожены (zero-order hold: значение
последнего шага сегмента), как внутри шага у network-wide интеграторов.
"""

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

import neuraltide.config
from neuraltide.core.base import PopulationModel
from neuraltide.core.types import TensorType, StateList, cast_if_needed

if TYPE_CHECKING:  # циклический импорт: core.network -> integrators.base
    from neuraltide.core.network import NetworkGraph, NetworkRNN


def flatten_state(state: StateList) -> tf.Tensor:
    """
    Уплощает StateList ([1, n_units] × d тензоров) в матрицу [n_units, d].

    Столбец j соответствует state[j]. Юниты независимы внутри популяции
    (связность только через синапсы), поэтому такое уплощение — естественный
    базис для блочного якобиана [n_units, d, d].
    """
    return tf.stack([tf.reshape(s, [-1]) for s in state], axis=1)


def unflatten_state(flat: tf.Tensor, like: StateList) -> StateList:
    """Обратное к flatten_state преобразование; формы берутся из `like`."""
    return [tf.reshape(flat[:, i], s.shape) for i, s in enumerate(like)]


def flatten_joint_state(pop_states: StateList, syn_states: StateList) -> tf.Tensor:
    """
    Уплощает объединённое состояние связанной сети в вектор [D].

    Конкатенация reshape(-1) всех тензоров pop_states и syn_states
    (популяции, затем синапсы) — тот же порядок, в котором
    _JointNetworkRHS возвращает плоские списки производных. Батч всегда 1,
    поэтому ведущая ось [1, n] стирается без потерь.
    """
    all_states = list(pop_states) + list(syn_states)
    if not all_states:
        return tf.zeros([0], dtype=neuraltide.config.get_dtype())
    return tf.concat([tf.reshape(s, [-1]) for s in all_states], axis=0)


def unflatten_joint_state(
    flat: tf.Tensor,
    pop_states: StateList,
    syn_states: StateList,
) -> Tuple[StateList, StateList]:
    """
    Обратное к flatten_joint_state преобразование.

    Число и формы слотов берутся из переданных списков состояний; длина
    вектора flat должна равняться сумме их элементов.
    """
    all_states = list(pop_states) + list(syn_states)
    shapes = [s.shape for s in all_states]
    sizes = [int(shape.num_elements()) for shape in shapes]
    pieces = tf.split(flat, sizes)
    n_pop = len(pop_states)
    new_pop = [tf.reshape(p, sh) for p, sh in zip(pieces[:n_pop], shapes[:n_pop])]
    new_syn = [tf.reshape(p, sh) for p, sh in zip(pieces[n_pop:], shapes[n_pop:])]
    return new_pop, new_syn


def joint_block_layout(rhs: Any) -> Tuple[List[int], List[int]]:
    """
    «Network scope»: числа тензоров состояния по блокам популяций и
    синапсов связанной сети.

    Структура читается из кэшей графа у rhs (_JointNetworkRHS._graph).
    Модели без динамики (InputPopulation, StaticSynapse) имеют пустые
    derivatives — их слоты step_network() оставляет неизменными.

    Raises:
        TypeError: если rhs не экспонирует кэши графа (ожидается
            _JointNetworkRHS из neuraltide.core.network).
    """
    graph = getattr(rhs, "_graph", None)
    if graph is None or not (
        hasattr(graph, "_pop_info_cache") and hasattr(graph, "_syn_info_cache")
    ):
        raise TypeError(
            "joint_block_layout: ожидается rhs с кэшами графа "
            "(_JointNetworkRHS из neuraltide.core.network), получен "
            f"'{type(rhs).__name__}'."
        )
    pop_blocks = [int(n) for _, _, n in graph._pop_info_cache]
    syn_blocks = [int(n) for _, _, n in graph._syn_info_cache]
    return pop_blocks, syn_blocks


def joint_derivative_ranges(blocks: List[int], n_derivs: int) -> List[Tuple[int, int]]:
    """
    Раскладывает n_derivs плоских производных по блокам состояния.

    Конвенция network-режима: блок даёт либо ровно len(state) производных
    (динамический блок), либо ни одной (модель без динамики). Производные
    размещаются жадно в порядке блоков; если после этого остаются
    неразмещённые производные — структура rhs не согласована с кэшами
    графа (ValueError).

    Returns:
        Список (offset, count) производных по блокам (count == 0 для
        блоков без динамики).
    """
    ranges = []
    idx = 0
    for n in blocks:
        count = n if n_derivs - idx >= n else 0
        ranges.append((idx, count))
        idx += count
    if idx != n_derivs:
        raise ValueError(
            "joint_derivative_ranges: не удалось согласовать плоские "
            f"производные ({n_derivs}) с блоками состояния {blocks} — "
            "ожидается, что каждый блок даёт ровно len(state) производных "
            "либо ни одной."
        )
    return ranges


def joint_dynamic_indices(
    rhs: Any, n_dpop: int, n_dsyn: int
) -> Tuple[List[int], List[int]]:
    """
    Плоские индексы слотов состояния, для которых rhs дал производные.

    Args:
        rhs: _JointNetworkRHS (структура блоков — см. joint_block_layout).
        n_dpop: len(dpop_flat) из rhs(pop_states, syn_states).
        n_dsyn: len(dsyn_flat) из того же вызова.

    Returns:
        (pop_indices, syn_indices) — позиции в плоских списках pop_states
        и syn_states. Слоты блоков без динамики в списки не входят.
    """
    pop_blocks, syn_blocks = joint_block_layout(rhs)
    pop_idx: List[int] = []
    s = 0
    for n, (_, cnt) in zip(pop_blocks, joint_derivative_ranges(pop_blocks, n_dpop)):
        pop_idx.extend(range(s, s + cnt))
        s += n
    syn_idx: List[int] = []
    s = 0
    for n, (_, cnt) in zip(syn_blocks, joint_derivative_ranges(syn_blocks, n_dsyn)):
        syn_idx.extend(range(s, s + cnt))
        s += n
    return pop_idx, syn_idx


def flat_vector_field_jacobian(
    f_flat: Callable[[tf.Tensor], tf.Tensor],
    z: tf.Tensor,
    method: str = "jvp",
    fd_eps: Optional[float] = None,
) -> tf.Tensor:
    """
    Плотный якобиан J = ∂f/∂z плоского векторного поля: [D, D].

    Используется RosenbrockIntegrator.step_network(): f_flat — замыкание
    над _JointNetworkRHS, отображающее вектор динамических компонент
    состояния сети [D] в плоскую правую часть [D] того же размера.

    Args:
        f_flat: callable(z) -> [D].
        z: точка линеаризации, вектор [D].
        method: 'jvp' (D JVP-вызовов через ForwardAccumulator, точно,
            по умолчанию) или 'fd' (центральные разности).
        fd_eps: шаг разности для method='fd' (по умолчанию sqrt(eps)).

    Returns:
        tf.Tensor [D, D] в dtype вектора z.
    """
    d = int(z.shape[0])
    if d == 0:
        return tf.zeros([0, 0], dtype=z.dtype)

    if method == "jvp":
        columns = []
        for j in range(d):
            tangents = tf.one_hot(j, d, dtype=z.dtype)
            with tf.autodiff.ForwardAccumulator(primals=z, tangents=tangents) as acc:
                out = f_flat(z)
            columns.append(acc.jvp(out))
        return tf.stack(columns, axis=1)

    if method == "fd":
        eps = fd_eps
        if eps is None:
            eps = float(np.sqrt(np.finfo(z.dtype.as_numpy_dtype).eps))
        columns = []
        for j in range(d):
            e = tf.one_hot(j, d, dtype=z.dtype) * eps
            columns.append((f_flat(z + e) - f_flat(z - e)) / (2.0 * eps))
        return tf.stack(columns, axis=1)

    raise ValueError(f"Unknown jacobian method '{method}', expected 'jvp' or 'fd'.")


def state_vector_field_jacobian(
    population: PopulationModel,
    state: StateList,
    total_synaptic_input: Dict[str, TensorType],
    method: str = "jvp",
    fd_eps: Optional[float] = None,
) -> tf.Tensor:
    """
    Якобиан J правой части популяции по её состоянию: [n_units, d, d].

    J[u, i, j] = ∂f_i / ∂state_j в юните u (блочно-диагональная структура:
    юниты внутри популяции не связаны). Вычисляется d JVP-вызовами через
    ForwardAccumulator (method='jvp') или центральными разностями
    (method='fd').

    Args:
        population: экземпляр PopulationModel.
        state: текущее состояние.
        total_synaptic_input: синаптический вход (заморожен внутри шага).
        method: 'jvp' (точно, по умолчанию) или 'fd'.
        fd_eps: шаг разности для method='fd' (по умолчанию sqrt(eps)).

    Returns:
        tf.Tensor [n_units, d, d] в глобальном dtype.
    """
    dtype = neuraltide.config.get_dtype()
    state = list(state)  # ForwardAccumulator требует единую структуру primals/tangents
    d = len(state)
    if d == 0:
        n = int(state[0].shape[1]) if state else 1
        return tf.zeros([n, 0, 0], dtype=dtype)

    if method == "jvp":
        columns = []
        for j in range(d):
            tangents = [
                tf.ones_like(state[i]) if i == j else tf.zeros_like(state[i])
                for i in range(d)
            ]
            with tf.autodiff.ForwardAccumulator(
                primals=state, tangents=tangents
            ) as acc:
                out = population.derivatives(state, total_synaptic_input)
            # acc.jvp(out[i]) имеет форму примала [1, n] — схлопываем до [n].
            columns.append(
                tf.stack([tf.reshape(acc.jvp(o), [-1]) for o in out], axis=1)
            )
        return tf.stack(columns, axis=2)

    if method == "fd":
        eps = fd_eps
        if eps is None:
            eps = float(np.sqrt(np.finfo(dtype.as_numpy_dtype).eps))
        columns = []
        flat_state = [tf.reshape(s, [-1]) for s in state]
        for j in range(d):
            plus = [
                tf.reshape(flat_state[i] + (eps if i == j else 0.0), s.shape)
                for i, s in enumerate(state)
            ]
            minus = [
                tf.reshape(flat_state[i] - (eps if i == j else 0.0), s.shape)
                for i, s in enumerate(state)
            ]
            f_plus = flatten_state(population.derivatives(plus, total_synaptic_input))
            f_minus = flatten_state(population.derivatives(minus, total_synaptic_input))
            columns.append((f_plus - f_minus) / (2.0 * eps))
        return tf.stack(columns, axis=2)

    raise ValueError(f"Unknown jacobian method '{method}', expected 'jvp' or 'fd'.")


def _extract_state_tensors(output: Any) -> List[tf.Tensor]:
    """Извлекает плоский список тензоров состояния из разных форм вывода."""
    from neuraltide.core.network import NetworkOutput

    if isinstance(output, NetworkOutput):
        pop_states, syn_states = output.final_state
        return list(pop_states) + list(syn_states)
    if (
        isinstance(output, (tuple, list))
        and len(output) == 2
        and all(isinstance(x, (tuple, list)) for x in output)
    ):
        # (pop_states, syn_states)
        return list(output[0]) + list(output[1])
    if isinstance(output, (tuple, list)):
        return [t for t in output if isinstance(t, tf.Tensor)]
    if isinstance(output, dict):
        return [t for t in output.values() if isinstance(t, tf.Tensor)]
    raise TypeError(
        "classify_failure: unsupported output type "
        f"'{type(output).__name__}'. Pass NetworkOutput, "
        "(pop_states, syn_states) tuple, or a list of tensors."
    )


def explicit_euler_dt_limit(eigenvalues: np.ndarray) -> float:
    """
    Лимит шага явного Эйлера по спектру одной точки: min 2|Re λ|/|λ|².

    Область устойчивости явного Эйлера — круг |1 + hλ| <= 1; для моды
    с Re λ < 0 это даёт h <= 2|Re λ|/|λ|² (для вещественной отрицательной
    моды совпадает с классическим 2/|λ|). Для комплексных мод лимит
    существенно жёстче: у пары λ = −1 ± 10i он ≈ 0.0198 против 0.199
    у формулы 2/|λ|.

    Если в спектре есть мода с Re λ >= 0, устойчивого положительного
    шага не существует — лимит равен 0.0.
    """
    ev = np.asarray(eigenvalues, dtype=np.complex128).reshape(-1)
    if ev.size == 0:
        return 0.0
    limit = float("inf")
    for lam in ev:
        re = float(lam.real)
        if re >= 0.0:
            return 0.0
        limit = min(limit, -2.0 * re / abs(lam) ** 2)
    return float(limit)


def classify_failure(output: Any, diverged_threshold: float = 1e6) -> Dict[str, Any]:
    """
    Классифицирует результат прогона сети.

    Args:
        output: NetworkOutput, (pop_states, syn_states) или список тензоров.
        diverged_threshold: порог |x|_max для расходимости (по умолчанию 1e6).

    Returns:
        dict: {'status': 'ok' | 'nan' | 'diverged', 'max_abs': float}.
    """
    tensors = _extract_state_tensors(output)
    if not tensors:
        return {"status": "ok", "max_abs": 0.0}

    has_nan = False
    max_abs = 0.0
    for t in tensors:
        t64 = cast_if_needed(t, tf.float64)
        if bool(tf.reduce_any(tf.math.is_nan(t64))):
            has_nan = True
        if bool(tf.reduce_any(tf.math.is_inf(t64))):
            has_nan = True
        max_abs = max(max_abs, float(tf.reduce_max(tf.abs(t64))))
        if has_nan:
            return {"status": "nan", "max_abs": float("nan")}
    if max_abs > diverged_threshold:
        return {"status": "diverged", "max_abs": max_abs}
    return {"status": "ok", "max_abs": max_abs}


def _segment_bounds(total: int, n_segments: int) -> List[Tuple[int, int]]:
    """Границы почти равных сегментов [start, end) покрывающих [0, total)."""
    base = total // n_segments
    rem = total % n_segments
    bounds = []
    start = 0
    for i in range(n_segments):
        length = base + (1 if i < rem else 0)
        bounds.append((start, start + length))
        start += length
    return bounds


def _resolve_network_and_protocol(
    network_or_problem: Any,
    t_points: int,
) -> Tuple["NetworkRNN", Optional[tf.Tensor], Optional[tf.Tensor], Any, int]:
    """
    Возвращает (network, t_sequence|None, inputs, initial_state, n_segments).

    Для CalibrationProblem берётся её протокол; для голого NetworkRNN
    синтезируется протокол из t_points сегментов по ~1 мс без входов.
    """
    from neuraltide.core.network import NetworkRNN

    if hasattr(network_or_problem, "network") and hasattr(
        network_or_problem, "t_sequence"
    ):
        problem = network_or_problem
        network = problem.network
        t_seq = problem.t_sequence
        n = int(t_seq.shape[1])
        n_segments = max(1, min(t_points, n))
        return (
            network,
            t_seq,
            problem.inputs,
            getattr(problem, "initial_state", None),
            n_segments,
        )

    if isinstance(network_or_problem, NetworkRNN):
        network = network_or_problem
        graph = network._graph
        if graph._input_names:
            raise ValueError(
                "stiffness_report: bare NetworkRNN with declared external "
                "inputs — pass CalibrationProblem instead so the protocol "
                "(t_sequence, inputs) is available."
            )
        n_segments = max(1, int(t_points))
        steps_per_segment = max(1, int(round(1.0 / float(graph.dt))))
        total = n_segments * steps_per_segment
        dtype = neuraltide.config.get_dtype()
        t_seq = tf.constant(
            (np.arange(total, dtype=np.float64) * float(graph.dt))[None, :, None],
            dtype=dtype,
        )
        return network, t_seq, None, None, n_segments

    raise TypeError(
        "stiffness_report: expected NetworkRNN or CalibrationProblem, got "
        f"'{type(network_or_problem).__name__}'."
    )


def _joint_network_rhs_jacobian(
    graph: "NetworkGraph",
    pop_states: StateList,
    syn_states: StateList,
    input_rates: Optional[Dict[str, TensorType]],
    method: str = "jvp",
) -> tf.Tensor:
    """
    Плотный якобиан [D, D] полной связанной правой части сети.

    Линеаризация _JointNetworkRHS по динамическим компонентам
    объединённого состояния (базис тот же, что у
    RosenbrockIntegrator.step_network): состояние уплощается
    (flatten_joint_state), rhs даёт производные динамических слотов
    (joint_dynamic_indices), якобиан строится flat_vector_field_jacobian
    по плоскому векторному полю. Внешние входы заморожены (zero-order
    hold): input_rates передаются в rhs как есть.

    Args:
        graph: NetworkGraph с построенными кэшами (validate()).
        pop_states: плоский список состояний популяций.
        syn_states: плоский список состояний синапсов.
        input_rates: {name: tensor[batch, n_units]} — замороженные
            частоты внешних входов (None эквивалентен {}).
        method: 'jvp' (по умолчанию) или 'fd'.

    Returns:
        tf.Tensor [D, D] в dtype состояния, D — число динамических
        компонент объединённого состояния.

    Raises:
        ValueError: если у сети нет динамических компонент.
    """
    from neuraltide.core.network import _JointNetworkRHS

    rhs = _JointNetworkRHS(graph, input_rates)
    dpop, dsyn = rhs(pop_states, syn_states)
    if not dpop and not dsyn:
        raise ValueError(
            "stiffness_report(scope='network'): сеть не имеет динамических "
            "компонент — полный якобиан вырожден."
        )
    pop_idx, syn_idx = joint_dynamic_indices(rhs, len(dpop), len(dsyn))
    n_pop = len(pop_states)
    dyn_idx = list(pop_idx) + [n_pop + i for i in syn_idx]

    flat0 = flatten_joint_state(pop_states, syn_states)
    shapes = [s.shape for s in list(pop_states) + list(syn_states)]
    sizes = [int(sh.num_elements()) for sh in shapes]
    dyn_sizes = [sizes[i] for i in dyn_idx]
    flat_pieces = tf.split(flat0, sizes)
    z0 = tf.concat([flat_pieces[i] for i in dyn_idx], axis=0)

    def scatter_z(z: tf.Tensor) -> tf.Tensor:
        """Вставляет динамические компоненты z в полный плоский вектор."""
        upd = list(flat_pieces)
        z_pieces = tf.split(z, dyn_sizes)
        for k, i in enumerate(dyn_idx):
            upd[i] = z_pieces[k]
        return tf.concat(upd, axis=0)

    def rhs_flat(z: tf.Tensor) -> tf.Tensor:
        """Плоская правая часть по динамическим компонентам [D]."""
        new_pop, new_syn = unflatten_joint_state(scatter_z(z), pop_states, syn_states)
        fp, fs = rhs(new_pop, new_syn)
        return tf.concat([tf.reshape(t, [-1]) for t in list(fp) + list(fs)], axis=0)

    return flat_vector_field_jacobian(rhs_flat, z0, method=method)


def _spectrum_metrics(
    per_point: List[np.ndarray],
) -> Tuple[Dict[str, Any], List[float], List[float], List[float]]:
    """
    Метрики спектра по списку точек: блок отчёта + агрегаты.

    Args:
        per_point: список [P] массивов собственных значений (complex).

    Returns:
        (block, lambda_max_abs, stiffness_ratio, explicit_dt_limit):
            block — {'eigenvalues': np.stack(per_point) [P, ...],
                     'lambda_max_abs': [P], 'stiffness_ratio': [P],
                     'explicit_dt_limit': [P]}; агрегаты — те же списки
            для свёртки в глобальные метрики отчёта.
    """
    lam_max_abs = [float(np.max(np.abs(ev))) for ev in per_point]
    ratios = []
    dt_limits = []
    for ev in per_point:
        re = ev.real
        stable = re[re < 0.0]
        if stable.size == 0:
            ratios.append(float("inf"))
        else:
            ratios.append(float(np.max(np.abs(stable)) / np.min(np.abs(stable))))
        dt_limits.append(explicit_euler_dt_limit(ev))
    block = {
        "eigenvalues": np.stack(per_point, axis=0),
        "lambda_max_abs": lam_max_abs,
        "stiffness_ratio": ratios,
        "explicit_dt_limit": dt_limits,
    }
    return block, lam_max_abs, ratios, dt_limits


def stiffness_report(
    network_or_problem: Any,
    phi: Optional[np.ndarray] = None,
    t_points: int = 8,
    method: str = "jvp",
    scope: str = "populations",
) -> Dict[str, Any]:
    """
    Отчёт о жёсткости: спектр якобиана правой части в точках траектории.

    Прогон разбивается на t_points сегментов; в конце каждого сегмента
    вычисляется якобиан правой части (метод 'jvp' через ForwardAccumulator
    или 'fd' — центральные разности) и его собственные значения.

    Args:
        network_or_problem: NetworkRNN (без внешних входов) или
            CalibrationProblem (берётся его протокол и parameter_space).
        phi: опциональный вектор параметров в пространстве задачи —
            применяется к модели перед прогоном (с восстановлением после).
        t_points: число точек траектории (сегментов прогона).
        method: 'jvp' (по умолчанию) или 'fd'.
        scope: 'populations' (по умолчанию) — блочно-диагональные
            якобианы динамических популяций при нулевом синаптическом
            входе (межюнитная связность не входит в спектр);
            'network' — полный якобиан [D, D] связанной сети по
            объединённому состоянию (популяции + синапсы) через
            _JointNetworkRHS; синаптическая связность входит в спектр.
            Внешние входы в обоих scope заморожены (zero-order hold:
            значение последнего шага сегмента).

    Returns:
        dict:
            'method': использованный метод якобиана;
            'scope': использованный scope;
            't_points': фактическое число точек;
            'populations' (scope='populations'): {name:
                {'eigenvalues': np.ndarray [P, n, d] complex,
                 'lambda_max_abs': [P], 'stiffness_ratio': [P],
                 'explicit_dt_limit': [P]}};
            'network' (scope='network'): {'eigenvalues': np.ndarray
                [P, D] complex, 'lambda_max_abs': [P],
                'stiffness_ratio': [P], 'explicit_dt_limit': [P]} — D
                равно числу динамических компонент объединённого состояния;
            'lambda_max': max |λ| по всем точкам (и блокам);
            'stiffness_ratio': max ratio по всем точкам;
            'explicit_dt_limit': min по точкам лимита min 2|Re λ|/|λ|²
                по модам (0.0, если в спектре точки есть мода с
                Re λ >= 0 — устойчивого шага Эйлера не существует);
            'dt_network': шаг сети graph.dt;
            'euler_stable': True если dt_network <= explicit_dt_limit.
    """
    if method not in ("jvp", "fd"):
        raise ValueError(f"Unknown method '{method}', expected 'jvp' or 'fd'.")
    if scope not in ("populations", "network"):
        raise ValueError(
            f"Unknown scope '{scope}', expected 'populations' or 'network'."
        )

    (
        network,
        t_seq_full,
        inputs_full,
        initial_state,
        n_segments,
    ) = _resolve_network_and_protocol(network_or_problem, t_points)
    graph = network._graph

    # --- phi: применить и потом восстановить ---
    saved_phi = None
    if phi is not None:
        space = getattr(network_or_problem, "parameter_space", None)
        if space is None:
            raise ValueError(
                "stiffness_report: phi requires a CalibrationProblem with a "
                "parameter_space."
            )
        saved_phi = np.asarray(network_or_problem.pack(), dtype=np.float64)
        space.apply_to_model(space.unpack(np.asarray(phi, dtype=np.float64)))
    try:
        bounds = _segment_bounds(int(t_seq_full.shape[1]), n_segments)
        pop_names = list(graph.dynamic_population_names)

        # eigenvalues[pop] -> list [P] of np.ndarray [n, d]
        spectra: Dict[str, list] = {name: [] for name in pop_names}
        # scope='network': list [P] of np.ndarray [D] полного спектра
        network_spectra: List[np.ndarray] = []
        # Нулевой синаптический вход — per-population формы [1, n_units_pop]:
        # кэш по имени популяции, а не по первой встреченной n_units.
        zero_inputs: Dict[str, Dict[str, tf.Tensor]] = {}
        carry_state = initial_state
        for k0, k1 in bounds:
            t_seg = t_seq_full[:, k0:k1, :]
            inputs_seg = inputs_full[:, k0:k1, :] if inputs_full is not None else None
            kwargs: Dict[str, Any] = {}
            if inputs_seg is not None:
                kwargs["inputs"] = inputs_seg
            if carry_state is not None:
                kwargs["initial_state"] = carry_state
            output = network(t_seg, **kwargs)
            failure = classify_failure(output.final_state)
            if failure["status"] != "ok":
                raise FloatingPointError(
                    f"stiffness_report: simulation failed at segment "
                    f"[{k0}, {k1}): {failure}."
                )
            pop_states, syn_states = output.final_state
            if scope == "network":
                # Внешние входы заморожены ZOH: значение последнего шага
                # сегмента (тот вход, который rhs видит внутри шага).
                input_rates: Dict[str, TensorType] = {}
                if inputs_seg is not None:
                    last = inputs_seg[:, -1, :]
                    for name in graph._input_names:
                        off = graph._input_offsets[name]
                        size = graph._external_inputs[name]
                        input_rates[name] = last[:, off : off + size]
                jac = _joint_network_rhs_jacobian(
                    graph, pop_states, syn_states, input_rates, method=method
                )
                network_spectra.append(
                    np.linalg.eigvals(jac.numpy().astype(np.complex128))
                )
            else:
                for name in pop_names:
                    pop = graph._populations[name]
                    start = network._pop_state_offsets[name]
                    n_state = len(pop.state_size)
                    state = list(pop_states[start : start + n_state])
                    n_units = int(state[0].shape[1])
                    zero_inp = zero_inputs.get(name)
                    if zero_inp is None:
                        dtype = neuraltide.config.get_dtype()
                        zero_inp = {
                            "I_syn": tf.zeros([1, n_units], dtype=dtype),
                            "g_syn": tf.zeros([1, n_units], dtype=dtype),
                        }
                        zero_inputs[name] = zero_inp
                    jac = state_vector_field_jacobian(
                        pop, state, zero_inp, method=method
                    )
                    spectra[name].append(
                        np.linalg.eigvals(jac.numpy().astype(np.complex128))
                    )
            carry_state = output.final_state
    finally:
        if saved_phi is not None:
            network_or_problem.parameter_space.apply_to_model(
                network_or_problem.parameter_space.unpack(saved_phi)
            )

    # (lambda_max_abs, stiffness_ratio, explicit_dt_limit) по блокам
    metrics: List[Tuple[List[float], List[float], List[float]]] = []
    lambda_max_global = 0.0
    ratio_global = 0.0
    dt_limit_global = np.inf
    if scope == "populations":
        populations: Dict[str, Dict[str, Any]] = {}
        for name in pop_names:
            block, lam, ratios, dt_limits = _spectrum_metrics(spectra[name])
            populations[name] = block
            metrics.append((lam, ratios, dt_limits))
    else:
        block, lam, ratios, dt_limits = _spectrum_metrics(network_spectra)
        metrics.append((lam, ratios, dt_limits))
    for lam, ratios, dt_limits in metrics:
        lambda_max_global = max(lambda_max_global, *lam)
        ratio_global = max(ratio_global, *ratios)
        dt_limit_global = min(dt_limit_global, *dt_limits)

    dt_network = float(graph.dt)
    report: Dict[str, Any] = {
        "method": method,
        "scope": scope,
        "t_points": n_segments,
        "lambda_max": lambda_max_global,
        "stiffness_ratio": float(ratio_global),
        "explicit_dt_limit": float(dt_limit_global),
        "dt_network": dt_network,
        "euler_stable": bool(dt_network <= dt_limit_global),
    }
    if scope == "populations":
        report["populations"] = populations
    else:
        report["network"] = block
    return report
