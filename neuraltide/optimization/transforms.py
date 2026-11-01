"""
Преобразования координат оптимизируемых параметров (θ ↔ φ).

Физические параметры модели θ связаны с внутренними координатами
оптимизатора φ дифференцируемым поэлементным преобразованием
θ = forward(φ). Обратное преобразование inverse(θ) сначала клиппит θ
в допустимую область с безопасным численным margin, а затем возвращает φ.
jacobian(φ) возвращает dθ/dφ поэлементно, в той же форме что φ.

Предупреждение о насыщении: sigmoid-образные преобразования
(`logit`, `scaled_sigmoid`, `bounded_log`) имеют производную, стремящуюся
к нулю у границ интервала: оптимизатор «прилипает» к границе, фактически
не достигая её. Изменение масштаба s лишь смещает зону насыщения.
Для параметров с естественной нижней границей у нуля предпочтительнее
`softplus` (производная затухает как exp(φ), а не как u·(1−u)) или `log`;
у верхней границы альтернативой являются настоящие bounds у L-BFGS-B/TRF.

Все константы создаются в `neuraltide.config.get_dtype()`; входы
приводятся к нему через `cast_if_needed`, поэтому реализации совместимы
и со скалярами, и с тензорами произвольной формы.
"""

from typing import Dict, Optional, Type

import numpy as np
import tensorflow as tf

import neuraltide.config
from neuraltide.core.types import cast_if_needed

__all__ = [
    "Transform",
    "IdentityTransform",
    "AffineTransform",
    "LogTransform",
    "LogitTransform",
    "ScaledSigmoidTransform",
    "SoftplusTransform",
    "BoundedLogTransform",
    "TRANSFORM_REGISTRY",
    "make_transform",
    "register_transform",
]


def _dtype() -> tf.DType:
    return neuraltide.config.get_dtype()


def _to_tensor(value) -> tf.Tensor:
    return cast_if_needed(tf.convert_to_tensor(value), _dtype())


def _eps() -> float:
    return float(np.finfo(_dtype().as_numpy_dtype).eps)


def _tiny() -> float:
    return float(np.finfo(_dtype().as_numpy_dtype).tiny)


def _margin() -> float:
    """Относительный безопасный margin для клиппинга около границ."""
    return 100.0 * _eps()


class Transform:
    """
    Базовый класс преобразования координат θ = forward(φ).

    Контракт:
        forward(phi)  — дифференцируемое прямое преобразование;
        inverse(theta) — обратное преобразование с клиппингом theta
            в допустимую область и безопасным margin;
        jacobian(phi)  — dtheta/dphi поэлементно, форма совпадает с phi.

    Атрибуты:
        name: строковое имя вида преобразования (используется фабрикой).
    """

    name: str = "base"

    def forward(self, phi) -> tf.Tensor:
        """φ → θ (дифференцируемо)."""
        raise NotImplementedError

    def inverse(self, theta) -> tf.Tensor:
        """θ → φ с клиппингом θ в допустимую область."""
        raise NotImplementedError

    def jacobian(self, phi) -> tf.Tensor:
        """dθ/dφ поэлементно, той же формы что phi."""
        raise NotImplementedError


class IdentityTransform(Transform):
    """θ = φ. Сохраняет текущее поведение params-формата без transform."""

    name = "identity"

    def forward(self, phi) -> tf.Tensor:
        return _to_tensor(phi)

    def inverse(self, theta) -> tf.Tensor:
        return _to_tensor(theta)

    def jacobian(self, phi) -> tf.Tensor:
        return tf.ones_like(_to_tensor(phi))


class AffineTransform(Transform):
    """θ = a·φ + b — центрирование и масштабирование (a ≠ 0)."""

    name = "affine"

    def __init__(self, a: float = 1.0, b: float = 0.0):
        if a == 0.0:
            raise ValueError("AffineTransform: slope 'a' must be non-zero.")
        self.a = float(a)
        self.b = float(b)

    def forward(self, phi) -> tf.Tensor:
        return self.a * _to_tensor(phi) + self.b

    def inverse(self, theta) -> tf.Tensor:
        return (_to_tensor(theta) - self.b) / self.a

    def jacobian(self, phi) -> tf.Tensor:
        return tf.ones_like(_to_tensor(phi)) * self.a


class LogTransform(Transform):
    """
    θ = exp(φ) — положительные параметры, меняющиеся по порядкам величины.

    inverse клиппит θ снизу: значения <= floor·(1 + margin) отображаются
    в log(floor·(1 + margin)), поэтому round-trip около нижней границы
    возвращает floor, а не −inf/NaN. floor по умолчанию зависит от dtype
    (min normal: ~1.18e-38 для float32, ~2.2e-308 для float64).

    Производная exp(φ) затухает экспоненциально при φ → −∞, но остаётся
    ненулевой везде, где forward ещё представим в данном dtype.
    """

    name = "log"

    def __init__(self, floor: Optional[float] = None):
        if floor is not None and floor <= 0.0:
            raise ValueError("LogTransform: 'floor' must be positive.")
        self.floor = None if floor is None else float(floor)

    def _theta_min(self) -> tf.Tensor:
        floor = self.floor if self.floor is not None else _tiny()
        return tf.constant(floor * (1.0 + _margin()), dtype=_dtype())

    def forward(self, phi) -> tf.Tensor:
        return tf.exp(_to_tensor(phi))

    def inverse(self, theta) -> tf.Tensor:
        theta = tf.maximum(_to_tensor(theta), self._theta_min())
        return tf.math.log(theta)

    def jacobian(self, phi) -> tf.Tensor:
        return tf.exp(_to_tensor(phi))


class ScaledSigmoidTransform(Transform):
    """
    θ = lo + (hi − lo)·sigmoid(φ/s) — параметр внутри (lo, hi).

    .. warning::
        Производная ∝ u·(1−u), u = sigmoid(φ/s), исчезает у обеих границ:
        преобразование насыщается, градиентный поток к границе затухает.
        Это осознанное ограничение, а не дефект реализации; для задач,
        где граница должна достигаться, используйте настоящие bounds
        (L-BFGS-B/TRF) или softplus/log около нуля.

    inverse клиппит нормированное u = (θ−lo)/(hi−lo) в [m, 1−m],
    m = 100·eps(dtype), поэтому round-trip у границ возвращает ближайшую
    допустимую точку, а φ остаётся конечным.
    """

    name = "scaled_sigmoid"

    def __init__(self, min: float, max: float, s: float = 1.0):
        if not min < max:
            raise ValueError(
                f"ScaledSigmoidTransform: require min < max, got min={min}, max={max}."
            )
        if s <= 0.0:
            raise ValueError("ScaledSigmoidTransform: scale 's' must be positive.")
        self.min = float(min)
        self.max = float(max)
        self.s = float(s)

    def _u(self, phi) -> tf.Tensor:
        return tf.sigmoid(_to_tensor(phi) / self.s)

    def forward(self, phi) -> tf.Tensor:
        return self.min + (self.max - self.min) * self._u(phi)

    def inverse(self, theta) -> tf.Tensor:
        span = self.max - self.min
        u = (_to_tensor(theta) - self.min) / span
        m = _margin()
        u = tf.clip_by_value(u, m, 1.0 - m)
        return self.s * (tf.math.log(u) - tf.math.log1p(-u))

    def jacobian(self, phi) -> tf.Tensor:
        u = self._u(phi)
        return (self.max - self.min) / self.s * u * (1.0 - u)


class LogitTransform(ScaledSigmoidTransform):
    """θ = lo + (hi − lo)·sigmoid(φ) — частный случай scaled_sigmoid, s = 1.

    См. предупреждение о насыщении sigmoid у границ в ScaledSigmoidTransform.
    """

    name = "logit"

    def __init__(self, min: float, max: float):
        super().__init__(min=min, max=max, s=1.0)


class SoftplusTransform(Transform):
    """
    θ = softplus(φ) + floor — неотрицательные параметры с достижимой
    окрестностью нуля.

    В отличие от sigmoid-образных преобразований, производная sigmoid(φ)
    около нижней границы затухает как exp(φ) (экспоненциально, но не
    двойственно-экспоненциально как u·(1−u) у границы интервала), поэтому
    градиентный поток около нуля сохраняется заметно дольше.

    inverse вычисляется в стабильной форме φ = y + log(−expm1(−y)),
    y = θ − floor, с клиппингом y снизу на min normal выбранного dtype.
    """

    name = "softplus"

    def __init__(self, floor: Optional[float] = None):
        if floor is not None and floor < 0.0:
            raise ValueError("SoftplusTransform: 'floor' must be non-negative.")
        self.floor = None if floor is None else float(floor)

    def _floor(self) -> tf.Tensor:
        floor = self.floor if self.floor is not None else _tiny()
        return tf.constant(floor, dtype=_dtype())

    def _y_min(self) -> tf.Tensor:
        return tf.constant(_tiny() * (1.0 + _margin()), dtype=_dtype())

    def forward(self, phi) -> tf.Tensor:
        return tf.nn.softplus(_to_tensor(phi)) + self._floor()

    def inverse(self, theta) -> tf.Tensor:
        y = tf.maximum(_to_tensor(theta) - self._floor(), self._y_min())
        return y + tf.math.log(-tf.math.expm1(-y))

    def jacobian(self, phi) -> tf.Tensor:
        return tf.sigmoid(_to_tensor(phi))


class BoundedLogTransform(Transform):
    """
    θ = lo + (hi − lo)·sigmoid(φ) — логарифмическая координата внутри
    заданного диапазона (без дополнительного масштаба s).

    .. warning::
        Как и у остальных sigmoid-образных преобразований, производная
        исчезает у границ интервала — см. ScaledSigmoidTransform.
    """

    name = "bounded_log"

    def __init__(self, min: float, max: float):
        if not min < max:
            raise ValueError(
                f"BoundedLogTransform: require min < max, got min={min}, max={max}."
            )
        self.min = float(min)
        self.max = float(max)

    def _u(self, phi) -> tf.Tensor:
        return tf.sigmoid(_to_tensor(phi))

    def forward(self, phi) -> tf.Tensor:
        return self.min + (self.max - self.min) * self._u(phi)

    def inverse(self, theta) -> tf.Tensor:
        span = self.max - self.min
        u = (_to_tensor(theta) - self.min) / span
        m = _margin()
        u = tf.clip_by_value(u, m, 1.0 - m)
        return tf.math.log(u) - tf.math.log1p(-u)

    def jacobian(self, phi) -> tf.Tensor:
        u = self._u(phi)
        return (self.max - self.min) * u * (1.0 - u)


TRANSFORM_REGISTRY: Dict[str, Type[Transform]] = {
    "identity": IdentityTransform,
    "affine": AffineTransform,
    "log": LogTransform,
    "logit": LogitTransform,
    "scaled_sigmoid": ScaledSigmoidTransform,
    "softplus": SoftplusTransform,
    "bounded_log": BoundedLogTransform,
}


def register_transform(kind: str, cls: Type[Transform]) -> None:
    """Регистрирует класс преобразования под строковым именем kind."""
    TRANSFORM_REGISTRY[kind] = cls


def make_transform(kind: str, **kwargs) -> Transform:
    """
    Фабрика преобразований по строковому kind.

    Raises:
        ValueError: при неизвестном kind.
    """
    cls = TRANSFORM_REGISTRY.get(kind)
    if cls is None:
        raise ValueError(
            f"Unknown transform kind '{kind}'. "
            f"Available kinds: {sorted(TRANSFORM_REGISTRY)}."
        )
    return cls(**kwargs)
