"""
ParameterSpace: единый реестр оптимизируемых параметров сети.

Сканирует `network._graph._populations` и `network._graph._synapses`,
сопоставляет trainable-переменные владельцам по коротким именам
(подход ``neuraltide/training/adjoint.py:_get_named_vars``) и строит
плоский вектор φ оптимизатора связанный с физическими θ через
поэлементные Transform'ы.

Стабильные полные имена имеют вид ``<owner>/<param>`` (например
``exc/Delta_I`` или ``inh->exc/tau_f``): owner — имя популяции/синапса
в NetworkGraph, param — имя переменной без служебного суффикса ``:0``.
Компоненты начального состояния регистрируются отдельно именами вида
``x0/<population>/<slot>`` (метод add_initial_state_entries) и адресуются
StateLocator(population, slot) в плоском списке начальных состояний сети
(см. NetworkRNN.pop_state_offsets).

Физические min/max извлекаются из MinMaxConstraint переменной
(т.е. из ``min``/``max`` пользовательского params-словаря). Без явно
заданного transform используется identity — старый params-формат
продолжает работать без изменений.

Вектор φ строится только по записям с ``trainable=True``. Записи с
``trainable=False`` (например ручные ``add_entry(..., trainable=False)``)
— реестровые константы: они видны в ``entries``/``names`` (интроспекция,
duplicate-check), но исключены из ``size``, ``pack``, ``unpack``,
``project`` и ``apply_to_model``.

Ограничения P0/P1:
  * автоматически охватываются только trainable-параметры популяций
    и синапсов графа. Параметры генераторов входов и readout-слои в
    ``network.trainable_variables`` обычно не входят — они не
    автоматизируются и не обещаются; для nuisance-параметров
    (сдвиги, масштабы, параметры шума) используйте ручную регистрацию
    ``add_entry``;
  * неизвестные trainable variables сети пропускаются с warnings.warn;
  * записи начального состояния (kind 'initial_state'/'shooting_state')
    не привязаны к переменным модели: pack(None) требует явных values
    для них, apply_to_model их пропускает.
"""

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

import tensorflow as tf

import neuraltide.config
from neuraltide.core.types import cast_if_needed
from neuraltide.optimization.transforms import IdentityTransform, Transform

__all__ = ["ParameterEntry", "ParameterSpace", "StateLocator"]

_ENTRY_KINDS = ("model_var", "initial_state", "shooting_state")
_TENSOR_KINDS = ("initial_state", "shooting_state")


def _short_name(var: tf.Variable) -> str:
    """Короткое имя переменной: без префикса слоя и суффикса ':0'."""
    raw = var.name
    if "/" in raw:
        raw = raw.rsplit("/", 1)[-1]
    if ":" in raw:
        raw = raw.rsplit(":", 1)[0]
    return raw


def _full_name(owner: str, var: tf.Variable) -> str:
    """
    Полное имя вида '<owner>/<param>'.

    В Keras 3 ``var.name`` уже короткое ('Delta_I'), префикс добавляем
    сами. В Keras 2 / TF1-style scoping ``var.name`` может уже содержать
    путь ('exc/Delta_I:0') — тогда оно используется как есть (минус ':0').
    """
    raw = var.name
    if raw.endswith(":0"):
        raw = raw[:-2]
    if "/" in raw:
        return raw
    return f"{owner}/{raw}"


def _bounds_from_variable(var: tf.Variable):
    """Физические (min, max) из MinMaxConstraint переменной, иначе (None, None)."""
    constraint = getattr(var, "constraint", None)
    if constraint is None:
        return None, None
    min_val = getattr(constraint, "min_val", None)
    max_val = getattr(constraint, "max_val", None)
    return min_val, max_val


@dataclass
class StateLocator:
    """
    Адрес компоненты состояния в плоском списке начальных состояний сети.

    Атрибуты:
        population: имя популяции в NetworkGraph;
        slot: индекс тензора внутри состояния популяции
            (0 <= slot < len(pop.state_size)).

    Плоский индекс в списке pop-начальных состояний вычисляется как
    ``start + slot``, где ``(start, count) = network.pop_state_offsets[
    population]``.
    """

    population: str
    slot: int


@dataclass
class ParameterEntry:
    """
    Одна оптимизируемая величина ParameterSpace.

    Атрибуты:
        full_name: стабильное полное имя вида 'exc/Delta_I' или
            'x0/exc/0' для компонент начального состояния;
        shape: исходная форма тензора параметра;
        transform: преобразование координат θ = forward(φ);
        min/max: физические границы (None — не ограничен);
        group: 'population' | 'synapse' | 'initial_state' |
            пользовательская группа;
        trainable: признак оптимизируемости; False — реестровая константа:
            запись остаётся в реестре (entries/names, duplicate-check), но
            не входит в вектор φ и игнорируется size/pack/unpack/project/
            apply_to_model;
        kind: 'model_var' — параметр модели (переменная или ручная
            запись), 'initial_state'/'shooting_state' — тензорная
            компонента начального состояния, не привязанная к
            переменной модели;
        locator: адрес состояния для kind 'initial_state'/'shooting_state',
            иначе None;
        variable: связанная tf.Variable (None для ручных записей и
            тензорных записей состояния — kind != 'model_var' требует
            variable=None).
    """

    full_name: str
    shape: tf.TensorShape
    transform: Transform = field(default_factory=IdentityTransform)
    min: Optional[float] = None
    max: Optional[float] = None
    group: str = "population"
    trainable: bool = True
    variable: Optional[tf.Variable] = None
    kind: str = "model_var"
    locator: Optional[StateLocator] = None

    def __post_init__(self) -> None:
        self.shape = tf.TensorShape(self.shape)
        if self.kind not in _ENTRY_KINDS:
            raise ValueError(
                f"ParameterEntry: unknown kind '{self.kind}'; "
                f"expected one of {list(_ENTRY_KINDS)}."
            )
        if self.kind != "model_var" and self.variable is not None:
            raise ValueError(
                f"ParameterEntry: kind '{self.kind}' requires variable=None "
                f"('{self.full_name}')."
            )
        if self.kind in _TENSOR_KINDS and self.locator is None:
            raise ValueError(
                f"ParameterEntry: kind '{self.kind}' requires a StateLocator "
                f"('{self.full_name}')."
            )

    @property
    def numel(self) -> int:
        return int(self.shape.num_elements())


class ParameterSpace:
    """
    Плоский вектор φ оптимизатора над trainable-параметрами сети.

    φ строится только по записям с ``trainable=True`` (свойства
    ``trainable_names`` / ``phi_entries`` — ordered trainable-only view).
    Записи с ``trainable=False`` — реестровые константы: ``entries`` и
    ``names`` по-прежнему содержат их (интроспекция, duplicate-check по
    всем записям), но размерность φ и все операции с ним (size, pack,
    unpack, project, apply_to_model) их игнорируют.

    Все методы (pack/unpack/apply_to_model/project) используют один и тот
    же вектор φ, что исключает рассинхронизацию между адаптерами
    оптимизаторов (Keras, SciPy, безградиентные методы).
    """

    def __init__(self) -> None:
        self._entries: Dict[str, ParameterEntry] = {}

    # ------------------------------------------------------------------
    # Построение
    # ------------------------------------------------------------------
    @classmethod
    def from_network(
        cls,
        network,
        transforms: Optional[Mapping[str, Transform]] = None,
    ) -> "ParameterSpace":
        """
        Собирает ParameterSpace из NetworkRNN.

        Args:
            network: NetworkRNN (доступ к графу через network._graph).
            transforms: необязательный словарь {full_name: Transform}.
                Записи без явного transform получают identity — старый
                params-формат продолжает работать без изменений.

        Warns:
            UserWarning: про transforms для неизвестных имён и про
                trainable variables, не сопоставленные ни одному
                владельцу (readout, генераторы входов и т.п.).
        """
        space = cls()
        graph = network._graph
        remaining: Dict[str, Transform] = dict(transforms or {})

        matched = set()

        def _add_owner(owner_name: str, group: str, model) -> None:
            for var in model.trainable_variables:
                full = _full_name(owner_name, var)
                if full in space._entries:
                    raise ValueError(
                        f"ParameterSpace: duplicate parameter name '{full}'."
                    )
                transform = remaining.pop(full, None)
                if transform is None:
                    transform = IdentityTransform()
                elif not isinstance(transform, Transform):
                    raise TypeError(
                        f"ParameterSpace: transform for '{full}' must be a "
                        f"Transform instance, got {type(transform).__name__}."
                    )
                min_val, max_val = _bounds_from_variable(var)
                space._entries[full] = ParameterEntry(
                    full_name=full,
                    shape=var.shape,
                    transform=transform,
                    min=min_val,
                    max=max_val,
                    group=group,
                    trainable=bool(var.trainable),
                    variable=var,
                )
                matched.add(id(var))

        for name, pop in graph._populations.items():
            _add_owner(name, "population", pop)
        for name, entry in graph._synapses.items():
            _add_owner(name, "synapse", entry.model)

        if remaining:
            warnings.warn(
                "ParameterSpace: transforms provided for unknown "
                f"parameters: {sorted(remaining)}.",
                UserWarning,
                stacklevel=2,
            )

        for var in network.trainable_variables:
            if id(var) not in matched:
                warnings.warn(
                    "ParameterSpace: trainable variable "
                    f"'{_full_name('?', var)}' was not matched to any "
                    "population/synapse and will be skipped.",
                    UserWarning,
                    stacklevel=2,
                )
        return space

    def add_entry(self, entry: ParameterEntry) -> None:
        """
        Ручная регистрация записи (например nuisance-параметра шума,
        который не привязан к переменным модели).

        Raises:
            ValueError: при дублировании full_name.
        """
        if entry.full_name in self._entries:
            raise ValueError(
                f"ParameterSpace: duplicate parameter name '{entry.full_name}'."
            )
        self._entries[entry.full_name] = entry

    def add_initial_state_entries(
        self,
        network,
        include: Optional[object] = None,
        transforms: Optional[Mapping[str, Transform]] = None,
        bounds: Optional[Mapping[str, Tuple[Optional[float], Optional[float]]]] = None,
    ) -> List[str]:
        """
        Регистрирует оптимизируемые компоненты начального состояния сети.

        Для каждой динамической популяции графа и каждого слота её
        state_size создаётся запись 'x0/<population>/<slot>' формы
        [1, n_units] с group='initial_state', kind='initial_state' и
        locator=StateLocator(population, slot). Адресация слотов —
        через network.pop_state_offsets. Слоты с нулевым числом
        элементов пропускаются.

        Args:
            network: NetworkRNN — источник популяций и offsets.
            include: опциональный фильтр — перечисление полных имён
                ('x0/exc/0') и/или кортежей (population, slot). Без него
                регистрируются все слоты всех динамических популяций.
            transforms: необязательный словарь {full_name: Transform};
                записи без явного transform получают identity.
            bounds: необязательный словарь {full_name: (min, max)} —
                физические границы компонент состояния.

        Returns:
            Список зарегистрированных имён (в порядке создания).

        Raises:
            TypeError: при transform не-Transform.
            ValueError: при дублировании имени или неизвестном include.
        """
        remaining_transforms: Dict[str, Transform] = dict(transforms or {})
        remaining_bounds: Dict[str, Tuple[Optional[float], Optional[float]]] = dict(
            bounds or {}
        )
        include_set = self._normalize_include(include)
        registered: List[str] = []

        for pop_name in network._graph.dynamic_population_names:
            pop = network._graph._populations[pop_name]
            for slot, shape in enumerate(pop.state_size):
                full = f"x0/{pop_name}/{slot}"
                if include_set is not None and full not in include_set:
                    continue
                shape = tf.TensorShape(shape)
                if shape.num_elements() == 0:
                    continue
                if full in self._entries:
                    raise ValueError(
                        f"ParameterSpace: duplicate parameter name '{full}'."
                    )
                transform = remaining_transforms.pop(full, None)
                if transform is None:
                    transform = IdentityTransform()
                elif not isinstance(transform, Transform):
                    raise TypeError(
                        f"ParameterSpace: transform for '{full}' must be a "
                        f"Transform instance, got {type(transform).__name__}."
                    )
                min_val, max_val = remaining_bounds.pop(full, (None, None))
                self._entries[full] = ParameterEntry(
                    full_name=full,
                    shape=shape,
                    transform=transform,
                    min=min_val,
                    max=max_val,
                    group="initial_state",
                    trainable=True,
                    variable=None,
                    kind="initial_state",
                    locator=StateLocator(population=pop_name, slot=slot),
                )
                registered.append(full)

        if remaining_transforms:
            warnings.warn(
                "ParameterSpace: transforms provided for unknown "
                f"initial-state entries: {sorted(remaining_transforms)}.",
                UserWarning,
                stacklevel=2,
            )
        if remaining_bounds:
            warnings.warn(
                "ParameterSpace: bounds provided for unknown "
                f"initial-state entries: {sorted(remaining_bounds)}.",
                UserWarning,
                stacklevel=2,
            )
        return registered

    @staticmethod
    def _normalize_include(include: Optional[object]) -> Optional[set]:
        """Приводит include к множеству полных имён; None — без фильтра."""
        if include is None:
            return None
        names = set()
        for item in include:
            if isinstance(item, str):
                names.add(item)
            elif isinstance(item, tuple) and len(item) == 2:
                names.add(f"x0/{item[0]}/{item[1]}")
            else:
                raise ValueError(
                    "ParameterSpace: 'include' items must be full names "
                    f"('x0/exc/0') or (population, slot) tuples, got {item!r}."
                )
        return names

    # ------------------------------------------------------------------
    # Интроспекция
    # ------------------------------------------------------------------
    @property
    def entries(self) -> Mapping[str, ParameterEntry]:
        """Только для чтения: full_name → ParameterEntry."""
        return dict(self._entries)

    @property
    def names(self) -> List[str]:
        return list(self._entries.keys())

    @property
    def trainable_names(self) -> List[str]:
        """Имена trainable-записей в порядке регистрации — координаты φ."""
        return [n for n, e in self._entries.items() if e.trainable]

    @property
    def phi_entries(self) -> Mapping[str, ParameterEntry]:
        """
        Ordered trainable-only view реестра: full_name → ParameterEntry.

        Порядок — порядок регистрации; значения — те же объекты записей,
        что и в ``entries``. Именно это подмножество определяет вектор φ:
        его длина равна ``size``, а pack/unpack/project/apply_to_model
        работают только с этими записями.
        """
        return {n: e for n, e in self._entries.items() if e.trainable}

    @property
    def model_entry_names(self) -> List[str]:
        """Имена записей параметров модели (kind == 'model_var')."""
        return [n for n, e in self._entries.items() if e.kind == "model_var"]

    @property
    def tensor_entry_names(self) -> List[str]:
        """Имена тензорных записей состояния ('initial_state'/'shooting_state')."""
        return [n for n, e in self._entries.items() if e.kind in _TENSOR_KINDS]

    @property
    def size(self) -> int:
        """Длина вектора φ: суммарное число элементов trainable-записей."""
        return int(sum(e.numel for e in self.phi_entries.values()))

    def shape(self, name: str) -> tf.TensorShape:
        return self._entries[name].shape

    def group(self, name: str) -> str:
        return self._entries[name].group

    # ------------------------------------------------------------------
    # pack / unpack
    # ------------------------------------------------------------------
    def _offsets(self) -> Dict[str, tuple]:
        offsets = {}
        start = 0
        for name, entry in self.phi_entries.items():
            offsets[name] = (start, entry.numel)
            start += entry.numel
        return offsets

    @staticmethod
    def _value_to_tensor(value, entry: ParameterEntry) -> tf.Tensor:
        # python scalar/list/ndarray конвертируем СРАЗУ в целевом dtype:
        # convert_to_tensor без dtype квантует float64 в float32 (теряется
        # точность fixed-значений FrozenProblem / values в pack). Уже
        # существующие tf.Tensor/tf.Variable приводим через cast (у них
        # dtype зафиксирован и convert с dtype= поднял бы ValueError).
        dtype = neuraltide.config.get_dtype()
        if tf.is_tensor(value):
            tensor = cast_if_needed(value, dtype)
        else:
            tensor = tf.convert_to_tensor(value, dtype=dtype)
        numel = tensor.shape.num_elements()
        if numel is not None and numel != entry.numel:
            if numel == 1 and entry.numel > 1:
                tensor = tf.broadcast_to(tensor, entry.shape)
            else:
                raise ValueError(
                    f"ParameterSpace: value for '{entry.full_name}' has "
                    f"{numel} elements, expected {entry.numel} "
                    f"(shape {entry.shape})."
                )
        return tf.reshape(tensor, entry.shape)

    def pack(self, values: Optional[Mapping[str, object]] = None) -> tf.Tensor:
        """
        Физические θ → плоский вектор φ = inverse(θ) trainable-записей.

        Args:
            values: словарь {full_name: θ}; ключи — полные имена.
                Требуются только trainable-записи; неизвестные и
                non-trainable ключи молча игнорируются. None — текущие
                значения переменных модели trainable-записей (trainable-
                запись без переменной в этом режиме вызывает ValueError).

        Returns:
            tf.Tensor формы [size] в neuraltide.config.get_dtype(); для
            пустого trainable-множества — tf.zeros([0], dtype).
        """
        pieces = []
        for name, entry in self.phi_entries.items():
            if values is None:
                if entry.variable is None:
                    raise ValueError(
                        "ParameterSpace: pack(None) requires model "
                        f"variables; entry '{name}' has none. Pass "
                        "'values' with explicit numbers instead."
                    )
                theta = cast_if_needed(entry.variable, neuraltide.config.get_dtype())
            else:
                if name not in values:
                    raise ValueError(f"ParameterSpace: 'values' missing key '{name}'.")
                theta = self._value_to_tensor(values[name], entry)
            pieces.append(tf.reshape(entry.transform.inverse(theta), [-1]))
        dtype = neuraltide.config.get_dtype()
        if not pieces:
            return tf.zeros([0], dtype=dtype)
        return tf.concat(pieces, axis=0)

    def unpack(self, phi) -> Dict[str, tf.Tensor]:
        """
        Плоский вектор φ → физические θ = forward(φ) по trainable-именам.

        Дифференцируемо по phi (GradientTape-совместимо). Non-trainable
        записи в результат не включаются.

        Raises:
            ValueError: при несовпадении размера phi с self.size.
        """
        phi = cast_if_needed(tf.convert_to_tensor(phi), neuraltide.config.get_dtype())
        phi = tf.reshape(phi, [-1])
        if phi.shape[0] is not None and phi.shape[0] != self.size:
            raise ValueError(
                f"ParameterSpace: phi has {int(phi.shape[0])} elements, "
                f"expected {self.size}."
            )
        tf.debugging.assert_equal(
            tf.size(phi), self.size, message="ParameterSpace: bad phi size"
        )
        offsets = self._offsets()
        result: Dict[str, tf.Tensor] = {}
        for name, entry in self.phi_entries.items():
            start, numel = offsets[name]
            piece = tf.reshape(phi[start : start + numel], entry.shape)
            result[name] = entry.transform.forward(piece)
        return result

    # ------------------------------------------------------------------
    # Применение значений и проекция
    # ------------------------------------------------------------------
    def apply_to_model(self, theta_dict: Mapping[str, object]) -> None:
        """
        Записывает физические θ из theta_dict в переменные модели.

        Обрабатываются только trainable-записи: non-trainable записи
        пропускаются даже с привязанной variable (это реестровые константы,
        которые оптимизатор не меняет). Записи без привязанной переменной
        (add_entry) пропускаются. Формы и dtype приводятся к переменной
        автоматически.
        """
        for name, entry in self.phi_entries.items():
            if entry.variable is None:
                continue
            if name not in theta_dict:
                raise ValueError(f"ParameterSpace: 'theta_dict' missing key '{name}'.")
            theta = self._value_to_tensor(theta_dict[name], entry)
            entry.variable.assign(tf.cast(theta, entry.variable.dtype))

    def project(self, phi) -> tf.Tensor:
        """
        Проецирует φ в допустимую область.

        Выбранный подход: φ → θ = forward(φ) → клиппинг θ в [min, max]
        (физические границы записи) → inverse → φ'. Тогда
        forward(project(φ)) гарантированно лежит внутри заданных
        физических границ. Для bounded-преобразований (logit,
        scaled_sigmoid, bounded_log, softplus, log) forward уже
        отображает внутрь области; клиппинг здесь — дополнительный
        страховочный слой. Записи без границ не изменяются (кроме
        прохождения через round-trip transform'а).

        Для неограниченных записей с сильно выросшим φ (например exp-
        переполнение у log) вызывающий код обязан ограничивать шаг
        оптимизатора — project() не имеет оснований для клиппинга.

        Работает только с trainable-записями: non-trainable записи и их
        границы на результат не влияют.
        """
        phi = cast_if_needed(tf.convert_to_tensor(phi), neuraltide.config.get_dtype())
        phi = tf.reshape(phi, [-1])
        if phi.shape[0] is not None and phi.shape[0] != self.size:
            raise ValueError(
                f"ParameterSpace: phi has {int(phi.shape[0])} elements, "
                f"expected {self.size}."
            )
        offsets = self._offsets()
        pieces = []
        for name, entry in self.phi_entries.items():
            start, numel = offsets[name]
            piece = tf.reshape(phi[start : start + numel], entry.shape)
            theta = entry.transform.forward(piece)
            if entry.min is not None:
                theta = tf.maximum(theta, entry.min)
            if entry.max is not None:
                theta = tf.minimum(theta, entry.max)
            pieces.append(tf.reshape(entry.transform.inverse(theta), [-1]))
        dtype = neuraltide.config.get_dtype()
        if not pieces:
            return tf.zeros([0], dtype=dtype)
        return tf.concat(pieces, axis=0)
