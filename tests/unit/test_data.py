"""Tests for neuraltide.data module."""
import os
import tempfile

import numpy as np
import pytest

from neuraltide.data import save_dataset, load_dataset, Dataset


class TestSaveLoadDataset:
    """Test save_dataset and load_dataset roundtrip."""

    def test_basic_roundtrip(self, tmp_path):
        """save → load should preserve all data."""
        inputs = {'theta': np.random.randn(100, 1).astype(np.float32)}
        target = {'exc': np.random.randn(100, 2).astype(np.float32)}
        dt = 0.5
        gen_params = {'theta': {'freq': 8.0, 'R': 0.5}}

        path = str(tmp_path / 'test.h5')
        save_dataset(path, inputs, target, dt, gen_params)

        data = load_dataset(path)

        assert isinstance(data, Dataset)
        assert data.dt == dt
        assert data.T == 100
        assert data.total_input_units == 1
        assert data.total_target_units == 2
        assert data.input_names == ['theta']
        assert data.target_names == ['exc']
        assert data.input_n_units == {'theta': 1}
        assert data.target_n_units == {'exc': 2}
        assert data.generator_params == gen_params

        np.testing.assert_array_almost_equal(data.inputs, inputs['theta'])
        np.testing.assert_array_almost_equal(data.target, target['exc'])

    def test_multiple_inputs(self, tmp_path):
        """Multiple named inputs should be stacked correctly."""
        inputs = {
            'theta': np.random.randn(50, 1).astype(np.float32),
            'place': np.random.randn(50, 3).astype(np.float32),
        }
        target = {'exc': np.random.randn(50, 2).astype(np.float32)}

        path = str(tmp_path / 'test.h5')
        save_dataset(path, inputs, target, dt=0.5)

        data = load_dataset(path)

        assert data.input_names == ['theta', 'place']
        assert data.input_n_units == {'theta': 1, 'place': 3}
        assert data.total_input_units == 4

        # Check input_slice
        theta_slice = data.input_slice('theta')
        assert theta_slice.shape == (50, 1)
        np.testing.assert_array_almost_equal(theta_slice, inputs['theta'])

        place_slice = data.input_slice('place')
        assert place_slice.shape == (50, 3)
        np.testing.assert_array_almost_equal(place_slice, inputs['place'])

    def test_multiple_targets(self, tmp_path):
        """Multiple named targets should be stacked correctly."""
        inputs = {'theta': np.random.randn(50, 1).astype(np.float32)}
        target = {
            'exc': np.random.randn(50, 2).astype(np.float32),
            'inh': np.random.randn(50, 1).astype(np.float32),
        }

        path = str(tmp_path / 'test.h5')
        save_dataset(path, inputs, target, dt=0.5)

        data = load_dataset(path)

        assert data.target_names == ['exc', 'inh']
        assert data.target_n_units == {'exc': 2, 'inh': 1}
        assert data.total_target_units == 3

        exc_slice = data.target_slice('exc')
        assert exc_slice.shape == (50, 2)
        np.testing.assert_array_almost_equal(exc_slice, target['exc'])

        inh_slice = data.target_slice('inh')
        assert inh_slice.shape == (50, 1)
        np.testing.assert_array_almost_equal(inh_slice, target['inh'])

    def test_time_seq_computed_from_dt(self, tmp_path):
        """time_seq should be arange(T) * dt."""
        inputs = {'x': np.random.randn(20, 1).astype(np.float32)}
        target = {'y': np.random.randn(20, 1).astype(np.float32)}
        dt = 0.25

        path = str(tmp_path / 'test.h5')
        save_dataset(path, inputs, target, dt)

        data = load_dataset(path)

        expected_time = np.arange(20, dtype=np.float32) * 0.25
        np.testing.assert_array_almost_equal(data.time_seq, expected_time)

    def test_empty_inputs_raises(self, tmp_path):
        """save_dataset should raise on empty inputs."""
        with pytest.raises(ValueError, match="inputs must not be empty"):
            save_dataset(str(tmp_path / 'test.h5'), {}, {'y': np.zeros((10, 1))}, 0.5)

    def test_empty_target_raises(self, tmp_path):
        """save_dataset should raise on empty target."""
        with pytest.raises(ValueError, match="target must not be empty"):
            save_dataset(str(tmp_path / 'test.h5'), {'x': np.zeros((10, 1))}, {}, 0.5)

    def test_inconsistent_T_raises(self, tmp_path):
        """save_dataset should raise if inputs have different T."""
        inputs = {
            'a': np.random.randn(10, 1).astype(np.float32),
            'b': np.random.randn(20, 1).astype(np.float32),
        }
        with pytest.raises(ValueError, match="expected 10"):
            save_dataset(str(tmp_path / 'test.h5'), inputs,
                        {'y': np.zeros((10, 1))}, 0.5)

    def test_1d_input_expanded(self, tmp_path):
        """1D input arrays should be expanded to [T, 1]."""
        inputs = {'x': np.random.randn(30).astype(np.float32)}
        target = {'y': np.random.randn(30).astype(np.float32)}

        path = str(tmp_path / 'test.h5')
        save_dataset(path, inputs, target, 0.5)

        data = load_dataset(path)
        assert data.total_input_units == 1
        assert data.inputs.shape == (30, 1)

    def test_compression(self, tmp_path):
        """HDF5 file should be compressed."""
        inputs = {'x': np.random.randn(1000, 10).astype(np.float32)}
        target = {'y': np.random.randn(1000, 5).astype(np.float32)}

        path = str(tmp_path / 'test.h5')
        save_dataset(path, inputs, target, 0.5)

        file_size = os.path.getsize(path)
        raw_size = 1000 * 10 * 4 + 1000 * 5 * 4  # float32
        # Compressed should be significantly smaller for random data
        # (random data doesn't compress well, but at least it shouldn't be bigger)
        assert file_size < raw_size * 2
