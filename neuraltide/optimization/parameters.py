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

Физические min/max извлекаются из MinMaxConstraint переменной
(т.е. из ``min``/``max`` пользовательского params-словаря). Без явно
заданного transform используется identity — старый params-формат
продолжает работать без изменений.

Ограничения P0:
  * автоматически охватываются только trainable-параметры популяций
    и синапсов графа. Параметры генераторов входов и readout-слои в
    ``network.trainable_variables`` обычно не входят — они не
    автоматизируются и не обещаются; для nuisance-параметров
    (сдвиги, масштабы, параметры шума) используйте ручную регистрацию
    ``add_entry``;
  * неизвестные trainable variables сети пропускаются с warnings.warn.
"""

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional

import tensorflow as tf

import neuraltide.config
from neuraltide.core.types import cast_if_needed
from neuraltide.optimization.transforms import IdentityTransform, Transform

__all__ = ["ParameterEntry", "ParameterSpace"]


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
class ParameterEntry:
    """
    Одна оптимизируемая величина ParameterSpace.

    Атрибуты:
        full_name: стабильное полное имя вида 'exc/Delta_I';
        shape: исходная форма тензора параметра;
        transform: преобразование координат θ = forward(φ);
        min/max: физические границы (None — не ограничен);
        group: 'population' | 'synapse' | пользовательская группа;
        trainable: признак оптимизируемости;
        variable: связанная tf.Variable (None для ручных записей).
    """

    full_name: str
    shape: tf.TensorShape
    transform: Transform = field(default_factory=IdentityTransform)
    min: Optional[float] = None
    max: Optional[float] = None
    group: str = "population"
    trainable: bool = True
    variable: Optional[tf.Variable] = None

    def __post_init__(self) -> None:
        self.shape = tf.TensorShape(self.shape)

    @property
    def numel(self) -> int:
        return int(self.shape.num_elements())


class ParameterSpace:
    """
    Плоский вектор φ оптимизатора над trainable-параметрами сети.

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
    def size(self) -> int:
        return int(sum(e.numel for e in self._entries.values()))

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
        for name, entry in self._entries.items():
            offsets[name] = (start, entry.numel)
            start += entry.numel
        return offsets

    @staticmethod
    def _value_to_tensor(value, entry: ParameterEntry) -> tf.Tensor:
        tensor = cast_if_needed(
            tf.convert_to_tensor(value), neuraltide.config.get_dtype()
        )
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
        Физические θ → плоский вектор φ = inverse(θ) всех записей.

        Args:
            values: словарь {full_name: θ}; ключи — полные имена.
                None — текущие значения переменных модели (записи без
                переменной в этом режиме вызывают ValueError).

        Returns:
            tf.Tensor формы [size] в neuraltide.config.get_dtype().
        """
        pieces = []
        for name, entry in self._entries.items():
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
        Плоский вектор φ → физические θ = forward(φ) по именам.

        Дифференцируемо по phi (GradientTape-совместимо).

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
        for name, entry in self._entries.items():
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

        Записи без привязанной переменной (add_entry) пропускаются.
        Формы и dtype приводятся к переменной автоматически.
        """
        for name, entry in self._entries.items():
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
        for name, entry in self._entries.items():
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
