import tensorflow as tf
from typing import Optional

class MinMaxConstraint(tf.keras.constraints.Constraint):
    """Ограничение параметра на отрезок [min_val, max_val]."""

    def __init__(self, min_val: Optional[float], max_val: Optional[float]):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, w: tf.Tensor) -> tf.Tensor:
        if self.min_val is not None and self.max_val is not None:
            return tf.clip_by_value(w, self.min_val, self.max_val)
        elif self.min_val is not None:
            return tf.maximum(w, self.min_val)
        elif self.max_val is not None:
            return tf.minimum(w, self.max_val)
        return w

    def get_config(self) -> dict:
        return {
            'min_val': self.min_val,
            'max_val': self.max_val,
        }

    @classmethod
    def from_config(cls, config: dict) -> 'MinMaxConstraint':
        return cls(config['min_val'], config['max_val'])


class NonNegConstraint(tf.keras.constraints.Constraint):
    """Ограничение: неотрицательные значения через ReLU."""

    def __call__(self, w: tf.Tensor) -> tf.Tensor:
        return tf.nn.relu(w)

    def get_config(self) -> dict:
        return {}

    @classmethod
    def from_config(cls, config: dict) -> 'NonNegConstraint':
        return cls()


class UnitIntervalConstraint(tf.keras.constraints.Constraint):
    """Ограничение на отрезок [0, 1]. Эквивалентна MinMaxConstraint(0, 1)."""

    def __init__(self):
        self._inner = MinMaxConstraint(0.0, 1.0)

    def __call__(self, w: tf.Tensor) -> tf.Tensor:
        return self._inner(w)

    def get_config(self) -> dict:
        return self._inner.get_config()

    @classmethod
    def from_config(cls, config: dict) -> 'UnitIntervalConstraint':
        return cls()
