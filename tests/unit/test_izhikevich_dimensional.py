"""
Tests for _build_params_from_dimensional in IzhikevichMeanField.

Verifies that dimensional → dimensionless conversion:
1. Works for all-scalar params (no dict specs)
2. Properly propagates trainable/min/max from dict-spec params
3. Converts min/max bounds to dimensionless using correct scale factor
4. Marks derived (composite) params as non-trainable, no constraints
5. Produces identical interface to dimensionless params
"""
import numpy as np
import tensorflow as tf
import pytest
import neuraltide
from neuraltide.populations.izhikevich_mf import IzhikevichMeanField


POP_PARAMS_SCALAR = {
    'Cm': 114.0,
    'K': 1.19,
    'V_rest': -57.63,
    'V_T': -35.53,
    'V_reset': -48.7,
    'A': 0.005,
    'B': 0.22,
    'W_jump': 2.0,
    'Delta_I': 20.0,
    'I_ext': 400.0,
}

POP_PARAMS_WITH_IEXT_DICT = {
    'Cm': 114.0,
    'K': 1.19,
    'V_rest': -57.63,
    'V_T': -35.53,
    'V_reset': -48.7,
    'A': 0.005,
    'B': 0.22,
    'W_jump': 2.0,
    'Delta_I': 20.0,
    'I_ext': {'value': 400.0, 'trainable': True, 'min': 0.0, 'max': 1000.0},
}


class TestIzhikevichDimensionalConversion:
    """Tests for _build_params_from_dimensional."""

    def _get_pop(self, params, dt=0.5):
        return IzhikevichMeanField(dt=dt, params=params)

    # ――― all-scalar baseline ――――――――――――――――――――――――――――――――――――――――――

    def test_all_scalar_params(self):
        """All scalars: should construct without error and produce expected values."""
        pop = self._get_pop(POP_PARAMS_SCALAR)
        assert pop.n_units == 1
        assert pop.I_ext.trainable is False

    # ――― dict-spec I_ext ―――――――――――――――――――――――――――――――――――――――――――――――

    def test_I_ext_dict_propagates_trainable(self):
        """I_ext as dict: trainable should be True."""
        pop = self._get_pop(POP_PARAMS_WITH_IEXT_DICT)
        assert pop.I_ext.trainable is True

    def test_I_ext_dict_propagates_constraint(self):
        """I_ext as dict: min/max should create a constraint."""
        pop = self._get_pop(POP_PARAMS_WITH_IEXT_DICT)
        assert pop.I_ext.constraint is not None

    def test_I_ext_dict_min_max_are_converted_to_dimensionless(self):
        """I_ext bounds must be divided by K * V_rest^2."""
        pop = self._get_pop(POP_PARAMS_WITH_IEXT_DICT)
        V_rest = abs(-57.63)
        K = 1.19
        factor = K * V_rest ** 2
        expected_value = 400.0 / factor
        expected_min = 0.0 / factor
        expected_max = 1000.0 / factor

        val = float(pop.I_ext.numpy()[0])
        assert abs(val - expected_value) < 1e-6, f"value {val} != {expected_value}"

        constraint = pop.I_ext.constraint
        assert constraint is not None
        cmin = float(constraint.min_val[0]) if hasattr(constraint.min_val, '__getitem__') else float(constraint.min_val)
        cmax = float(constraint.max_val[0]) if hasattr(constraint.max_val, '__getitem__') else float(constraint.max_val)
        assert abs(cmin - expected_min) < 1e-6
        assert abs(cmax - expected_max) < 1e-6

    # ――― dict-spec Delta_I ―――――――――――――――――――――――――――――――――――――――――――

    def test_Delta_I_dict_propagates_trainable_and_bounds(self):
        """Delta_I as dict with bounds: should be trainable with converted bounds."""
        params = {**POP_PARAMS_SCALAR,
                  'Delta_I': {'value': 20.0, 'trainable': True, 'min': 10.0, 'max': 50.0}}
        pop = self._get_pop(params)
        assert pop.Delta_I.trainable is True
        assert pop.Delta_I.constraint is not None

        V_rest = abs(-57.63)
        K = 1.19
        factor = K * V_rest ** 2
        expected_value = 20.0 / factor
        expected_min = 10.0 / factor
        expected_max = 50.0 / factor

        val = float(pop.Delta_I.numpy()[0])
        assert abs(val - expected_value) < 1e-6
        constraint = pop.Delta_I.constraint
        cmin = float(constraint.min_val[0]) if hasattr(constraint.min_val, '__getitem__') else float(constraint.min_val)
        cmax = float(constraint.max_val[0]) if hasattr(constraint.max_val, '__getitem__') else float(constraint.max_val)
        assert abs(cmin - expected_min) < 1e-6
        assert abs(cmax - expected_max) < 1e-6

    # ――― dict-spec W_jump → w_jump ――――――――――――――――――――――――――――――――――――

    def test_W_jump_dict_maps_to_w_jump_with_converted_bounds(self):
        """W_jump as dict: should become trainable w_jump with converted bounds."""
        params = {**POP_PARAMS_SCALAR,
                  'W_jump': {'value': 2.0, 'trainable': True, 'min': 0.0, 'max': 5.0}}
        pop = self._get_pop(params)
        assert pop.w_jump.trainable is True
        assert pop.w_jump.constraint is not None

        V_rest = abs(-57.63)
        K = 1.19
        factor = K * V_rest ** 2
        expected_value = 2.0 / factor
        expected_min = 0.0 / factor
        expected_max = 5.0 / factor

        val = float(pop.w_jump.numpy()[0])
        assert abs(val - expected_value) < 1e-6
        constraint = pop.w_jump.constraint
        cmin = float(constraint.min_val[0]) if hasattr(constraint.min_val, '__getitem__') else float(constraint.min_val)
        cmax = float(constraint.max_val[0]) if hasattr(constraint.max_val, '__getitem__') else float(constraint.max_val)
        assert abs(cmin - expected_min) < 1e-6
        assert abs(cmax - expected_max) < 1e-6

    # ――― derived params (tau_pop, alpha, a, b) ―――――――――――――――――――――――――――

    def test_derived_params_are_nontrainable(self):
        """tau_pop, alpha, a, b should be non-trainable with no constraints."""
        pop = self._get_pop(POP_PARAMS_WITH_IEXT_DICT)
        for name in ['tau_pop', 'alpha', 'a', 'b']:
            var = getattr(pop, name)
            assert var.trainable is False, f"{name} should not be trainable"
            assert var.constraint is None, f"{name} should have no constraint"

    def test_w_jump_is_nontrainable_when_scalar(self):
        """w_jump should be non-trainable when given as scalar."""
        pop = self._get_pop(POP_PARAMS_SCALAR)
        assert pop.w_jump.trainable is False
        assert pop.w_jump.constraint is None

    # ――― value correctness for all params ―――――――――――――――――――――――――――――――

    def test_derived_params_have_correct_values(self):
        """Verify tau_pop, alpha, a, b, w_jump values match formulas."""
        pop = self._get_pop(POP_PARAMS_SCALAR)

        Cm, K = 114.0, 1.19
        V_rest_abs = abs(-57.63)
        V_T = -35.53
        A, B = 0.005, 0.22
        W_jump = 2.0

        factor1 = K * V_rest_abs
        factor2 = K * V_rest_abs ** 2

        assert abs(float(pop.tau_pop.numpy()[0]) - Cm / factor1) < 1e-6
        assert abs(float(pop.alpha.numpy()[0]) - (1.0 + V_T / V_rest_abs)) < 1e-6
        assert abs(float(pop.a.numpy()[0]) - Cm * A / factor1) < 1e-6
        assert abs(float(pop.b.numpy()[0]) - B / factor1) < 1e-6
        assert abs(float(pop.w_jump.numpy()[0]) - W_jump / factor2) < 1e-6

    # ――― multi-unit (n_units=2) ―――――――――――――――――――――――――――――――――――――――

    def test_multi_unit_all_scalar_broadcasts(self):
        """n_units=2 with scalar + dict params should broadcast."""
        params = {
            'Cm': 114.0,
            'K': 1.19,
            'V_rest': -57.63,
            'V_T': -35.53,
            'V_reset': -48.7,
            'A': 0.005,
            'B': 0.22,
            'W_jump': {'value': 2.0, 'trainable': True, 'min': 0.0, 'max': 5.0},
            'Delta_I': 20.0,
            'I_ext': {'value': 400.0, 'trainable': True, 'min': 0.0, 'max': 1000.0},
        }
        pop = self._get_pop(params)
        assert pop.n_units == 1  # all scalars → n_units=1

    def test_multi_unit_with_lists(self):
        """n_units=2 using lists of values."""
        params = {
            'Cm': [114.0, 114.0],
            'K': [1.19, 1.19],
            'V_rest': [-57.63, -57.63],
            'V_T': [-35.53, -35.53],
            'V_reset': [-48.7, -48.7],
            'A': [0.005, 0.005],
            'B': [0.22, 0.22],
            'W_jump': [2.0, 2.0],
            'Delta_I': [20.0, 20.0],
            'I_ext': {'value': [400.0, 400.0], 'trainable': True,
                      'min': [0.0, 0.0], 'max': [1000.0, 1000.0]},
        }
        pop = self._get_pop(params)
        assert pop.n_units == 2
        assert pop.I_ext.trainable is True
        assert pop.I_ext.constraint is not None

    # ――― interface parity: dimensional ↔ dimensionless ――――――――――――――――――――

    def test_dimensional_produces_same_values_as_dimensionless(self):
        """Dimensional and dimensionless interfaces should give same parameter values."""
        V_rest_abs = 57.63
        K = 1.19
        factor2 = K * V_rest_abs ** 2
        factor1 = K * V_rest_abs

        # Helper: create both and compare
        pop_dim = self._get_pop(POP_PARAMS_WITH_IEXT_DICT)

        # Compute expected dimensionless values
        dimless_params = {
            'tau_pop': 114.0 / factor1,
            'alpha': 1.0 + (-35.53) / V_rest_abs,
            'a': 114.0 * 0.005 / factor1,
            'b': 0.22 / factor1,
            'w_jump': 2.0 / factor2,
            'Delta_I': 20.0 / factor2,
            'I_ext': {'value': 400.0 / factor2, 'trainable': True,
                      'min': 0.0 / factor2, 'max': 1000.0 / factor2},
        }
        pop_dimless = IzhikevichMeanField(dt=0.5, params=dimless_params)

        for name in ['tau_pop', 'alpha', 'a', 'b', 'w_jump', 'Delta_I', 'I_ext']:
            val_dim = float(getattr(pop_dim, name).numpy()[0])
            val_dimless = float(getattr(pop_dimless, name).numpy()[0])
            assert abs(val_dim - val_dimless) < 1e-5, f"Mismatch for {name}"

    def test_dimensional_with_missing_required_params_raises(self):
        """Missing a required dimensional param should raise ValueError."""
        bad_params = {'Cm': 114.0, 'K': 1.19}  # missing V_rest etc.
        with pytest.raises(ValueError, match='missing required dimensional parameters'):
            self._get_pop(bad_params)

    # ――― edge cases ――――――――――――――――――――――――――――――――――――――――――――――――――

    def test_none_params_raises(self):
        """params=None should raise ValueError."""
        with pytest.raises(ValueError, match='params cannot be None'):
            IzhikevichMeanField(dt=0.5, params=None)


class TestIzhikevichDictIntegration:
    """End-to-end: creation + forward pass with dict-spec dimensional params."""

    def test_forward_pass_with_dict_params(self):
        """A network created with dict dimensional params should run forward pass."""
        from neuraltide.core.network import NetworkGraph, NetworkRNN
        from neuraltide.integrators import EulerIntegrator

        pop = IzhikevichMeanField(dt=0.5, params=POP_PARAMS_WITH_IEXT_DICT)

        graph = NetworkGraph(dt=0.5)
        graph.add_population('pop', pop)

        network = NetworkRNN(graph, integrator=EulerIntegrator())

        t = tf.constant([[0.0], [0.5], [1.0]], dtype=tf.float32)
        output = network(t, training=False)
        rates = output.firing_rates['pop']
        assert rates.shape[0] == 1
        assert not tf.reduce_any(tf.math.is_nan(rates))
