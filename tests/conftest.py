import pytest
import tensorflow as tf
import neuraltide

@pytest.fixture(autouse=True)
def reset_dtype():
    neuraltide.config.set_dtype(tf.float32)
    yield

@pytest.fixture
def dt():
    return 0.5

@pytest.fixture
def n_steps():
    return 100

@pytest.fixture
def small_izh_params():
    """Минимальные параметры IzhikevichMeanField для n_units=2."""
    return {
        'alpha':     {'value': [0.5, 0.5],    'trainable': False},
        'a':         {'value': [0.02, 0.02],  'trainable': False},
        'b':         {'value': [0.2, 0.2],    'trainable': False},
        'w_jump':    {'value': [0.1, 0.1],    'trainable': False},
        'dt_nondim': {'value': [0.01, 0.01],  'trainable': False},
        'Delta_eta': {'value': [0.5, 0.6],    'trainable': True,
                      'min': 0.01, 'max': 2.0},
        'I_ext':     {'value': [1.0, 1.2],    'trainable': True},
    }
