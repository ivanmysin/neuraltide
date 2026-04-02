import tensorflow as tf
from typing import Dict, List, Tuple, Optional, Any, Type
import neuraltide.config as _config

def get_pi() -> tf.Tensor:
    """Возвращает π с текущим глобальным dtype."""
    return tf.constant(3.14159265358979323846, dtype=_config.get_dtype())

TensorType = tf.Tensor
StateList = List[TensorType]
ParamDict = Dict[str, Any]
