"""
Тензорные хелперы материализации начального состояния и градиента по φ.

Общая база problem.py (eager-ветки) и _compiled.py (графовое ядро
compiled=True): единственный источник, чтобы eager и compiled пути
давали идентичные результаты. Импортированы в problem.py по именам,
поэтому прежние пути ``from neuraltide.optimization.problem import
_materialize_initial_state`` и т.п. продолжают работать.
"""

import warnings
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import tensorflow as tf

import neuraltide.config
from neuraltide.core.types import TensorType, cast_if_needed
from neuraltide.optimization.parameters import ParameterSpace

_TENSOR_KINDS = ("initial_state", "shooting_state")

InitialState = Optional[Tuple[Sequence[TensorType], Sequence[TensorType]]]

# «One-time» защита от спама предупреждением о пропуске эксперимент-
# специфичных x0-записей вне экспериментного контекста: наборы имён,
# по которым предупреждение уже выданы (materialize вызывается на каждый
# прогон — без дедупликации оптимизатор получил бы тысячи копий).
_X0_OUTSIDE_CONTEXT_WARNED: set = set()


def _tensor_entries(space: ParameterSpace):
    """
    Trainable tensor-записи (kind 'initial_state'/'shooting_state').

    Только они участвуют в материализации x0: non-trainable записи —
    реестровые константы вне вектора φ, их слоты состояния сохраняют
    base/network initial state.
    """
    return [(n, e) for n, e in space.phi_entries.items() if e.kind in _TENSOR_KINDS]


def _x0_experiment(entry) -> Optional[str]:
    """experiment записи x0 (None — common); locator=None трактуем как common."""
    return entry.locator.experiment if entry.locator is not None else None


def _materialize_initial_state(
    network,
    space: ParameterSpace,
    theta: Mapping[str, tf.Tensor],
    base_initial_state: InitialState,
    experiment: Optional[str] = None,
) -> Tuple[InitialState, Dict[str, tf.Tensor]]:
    """
    Строит начальное состояние для network.call с подставленными x0.

    theta — физические значения из space.unpack (дифференцируемы по φ,
    если построены внутри GradientTape). base_initial_state — начальное
    состояние протокола (pop, syn) или None → сетевое по умолчанию.
    Материализуются только trainable tensor-записи: non-trainable записи
    пропускаются (без валидации locator), их слоты сохраняют значения
    base/network initial state.

    Подстановка двухпроходная с явным приоритетом эксперимент-специфичных
    записей (kind 'initial_state'/'shooting_state' с
    locator.experiment != None, см. ParameterSpace.
    add_experiment_initial_state_entries):

    * pass 1 — common-записи (locator.experiment is None) заполняют
      свои слоты начального состояния;
    * pass 2 — записи с locator.experiment == experiment перезаписывают
      те же слоты (перезаписанная common-запись удаляется из x0_map —
      её тензор больше не участвует в состоянии);
    * записи с другим locator.experiment пропускаются.

    experiment=None — внеэкспериментный контекст (CalibrationProblem,
    shooting, sensitivities, robust): применяются только common-записи,
    trainable exp-specific записи пропускаются с одноразовым UserWarning
    (предупреждение выдаётся один раз на набор имён, а не на каждый
    прогон).

    Returns:
        (initial_state, x0_map): initial_state — None, если нет ни
        применимых записей, ни base (сеть возьмёт своё дефолтное); иначе
        (init_pop, init_syn). x0_map: full_name → тензор, фактически
        подставленный в состояние (источник для chain-градиента).
    """
    entries = _tensor_entries(space)
    common = [(n, e) for n, e in entries if _x0_experiment(e) is None]
    if experiment is None:
        skipped = sorted(n for n, e in entries if _x0_experiment(e) is not None)
        if skipped:
            key = tuple(skipped)
            if key not in _X0_OUTSIDE_CONTEXT_WARNED:
                _X0_OUTSIDE_CONTEXT_WARNED.add(key)
                warnings.warn(
                    "Эксперимент-специфичные начальные состояния "
                    f"{skipped} пропущены вне экспериментного контекста "
                    "(experiment=None): они материализуются только в "
                    "MultiExperimentProblem у эксперимента с совпадающим "
                    "именем.",
                    UserWarning,
                    stacklevel=2,
                )
        applicable = common
    else:
        applicable = common + [
            (n, e) for n, e in entries if _x0_experiment(e) == experiment
        ]
    if not applicable:
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
    slot_owner: Dict[int, str] = {}
    for name, entry in applicable:
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
        flat = start + locator.slot
        x0 = cast_if_needed(tf.convert_to_tensor(theta[name]), dtype)
        x0 = tf.reshape(x0, entry.shape)
        # Exp-запись перезаписывает слот common-записи: прежний тензор
        # выходит из состояния — убираем его из x0_map (chain-градиент
        # не должен получать вклад по уже не подставленному тензору).
        prev = slot_owner.get(flat)
        if prev is not None:
            x0_map.pop(prev, None)
        init_pop[flat] = x0
        x0_map[name] = x0
        slot_owner[flat] = name
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
    суммируются, поэтому градиент по общему x0 аккумулирует вклады всех
    экспериментов, а эксперимент-специфичные x0-записи попадают в x0_map
    только своего эксперимента и между экспериментами не смешиваются.
    Записи без доступного источника дают нулевой вклад.
    """
    dtype = neuraltide.config.get_dtype()
    # Смещения по trainable-записям — в согласии с ParameterSpace.unpack
    # и длиной phi_t (non-trainable записи вне φ).
    entries = space.phi_entries
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
        for entry in space.phi_entries.values()
    )
    if grad_mode == "assign" and not has_model_vars:
        g = tape.gradient(loss_t, phi_t)
        if g is not None:
            return g
    return _chain_gradient(space, tape, loss_t, phi_t, x0_map)
