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
