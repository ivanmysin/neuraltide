import pytest
import tensorflow as tf
from neuraltide.constraints.param_constraints import (
    MinMaxConstraint,
    NonNegConstraint,
    UnitIntervalConstraint,
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


class TestNonNegConstraint:
    def test_relu(self):
        constraint = NonNegConstraint()
        w = tf.constant([-2.0, -0.5, 0.0, 0.5, 2.0])
        result = constraint(w)
        expected = [0.0, 0.0, 0.0, 0.5, 2.0]
        assert tf.reduce_all(tf.abs(result - expected) < 1e-6)

    def test_get_config(self):
        constraint = NonNegConstraint()
        config = constraint.get_config()
        assert config == {}

    def test_from_config(self):
        constraint = NonNegConstraint.from_config({})
        assert isinstance(constraint, NonNegConstraint)


class TestUnitIntervalConstraint:
    def test_equivalence_to_minmax(self):
        unit = UnitIntervalConstraint()
        minmax = MinMaxConstraint(0, 1)
        w = tf.constant([-0.5, 0.3, 1.5])
        result_unit = unit(w)
        result_minmax = minmax(w)
        assert tf.reduce_all(tf.abs(result_unit - result_minmax) < 1e-6)

    def test_get_config(self):
        constraint = UnitIntervalConstraint()
        config = constraint.get_config()
        assert config['min_val'] == 0.0
        assert config['max_val'] == 1.0

    def test_from_config(self):
        constraint = UnitIntervalConstraint.from_config({})
        assert isinstance(constraint, UnitIntervalConstraint)
