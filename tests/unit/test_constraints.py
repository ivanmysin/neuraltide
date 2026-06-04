import pytest
import tensorflow as tf
from neuraltide.constraints.param_constraints import (
    MinMaxConstraint,
)

class TestMinMaxConstraint:
    def test_clip_full_bounds(self):
        constraint = MinMaxConstraint(0, 1)
        w = tf.constant([-1.0, 0.5, 2.0])
        result = constraint(w)
        expected = [0.0, 0.5, 1.0]
        assert tf.reduce_all(tf.abs(result - expected) < 1e-6)

    def test_clip_only_min(self):
        constraint = MinMaxConstraint(0.0, None)
        w = tf.constant([-1.0, 0.5, 2.0])
        result = constraint(w)
        expected = [0.0, 0.5, 2.0]
        assert tf.reduce_all(tf.abs(result - expected) < 1e-6)

    def test_clip_only_max(self):
        constraint = MinMaxConstraint(None, 1.0)
        w = tf.constant([-1.0, 0.5, 2.0])
        result = constraint(w)
        expected = [-1.0, 0.5, 1.0]
        assert tf.reduce_all(tf.abs(result - expected) < 1e-6)

    def test_get_config(self):
        constraint = MinMaxConstraint(0.1, 0.9)
        config = constraint.get_config()
        assert config['min_val'] == 0.1
        assert config['max_val'] == 0.9

    def test_from_config(self):
        config = {'min_val': 0.0, 'max_val': 1.0}
        constraint = MinMaxConstraint.from_config(config)
        assert constraint.min_val == 0.0
        assert constraint.max_val == 1.0

