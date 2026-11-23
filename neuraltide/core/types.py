from typing import Any

import tensorflow as tf

import neuraltide.config as _config

TensorType = tf.Tensor
StateList = list[TensorType]
ParamDict = dict[str, Any]


def get_pi() -> tf.Tensor:
    """Возвращает π с текущим глобальным dtype."""
    return tf.constant(3.14159265358979323846, dtype=_config.get_dtype())


def cast_if_needed(val: TensorType, dtype: tf.DType) -> TensorType:
    """Приводит тензор к dtype только если это необходимо."""
    if isinstance(val, tf.Tensor) and val.dtype == dtype:
        return val
    return tf.cast(val, dtype)
