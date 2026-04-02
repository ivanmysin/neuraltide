import tensorflow as tf
from typing import Dict, Type

_DTYPE = tf.float32

def set_dtype(dtype: tf.DType) -> None:
    """Устанавливает глобальный тип данных. Должен вызываться до создания любых объектов."""
    global _DTYPE
    _DTYPE = dtype

def get_dtype() -> tf.DType:
    """Возвращает текущий глобальный тип данных."""
    return _DTYPE

POPULATION_REGISTRY: Dict[str, Type] = {}
SYNAPSE_REGISTRY: Dict[str, Type] = {}
INPUT_REGISTRY: Dict[str, Type] = {}

def register_population(name: str, cls: Type) -> None:
    """Регистрирует пользовательский класс PopulationModel для config-first API."""
    POPULATION_REGISTRY[name] = cls

def register_synapse(name: str, cls: Type) -> None:
    """Регистрирует пользовательский класс SynapseModel."""
    SYNAPSE_REGISTRY[name] = cls

def register_input(name: str, cls: Type) -> None:
    """Регистрирует пользовательский класс BaseInputGenerator."""
    INPUT_REGISTRY[name] = cls
