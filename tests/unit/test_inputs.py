import numpy as np
import pytest
import tensorflow as tf

import neuraltide.config
from neuraltide.inputs import (
    ConstantRateGenerator,
    PlaceFieldGenerator,
    SinusoidalGenerator,
    VonMisesGenerator,
)
from neuraltide.populations.input_population import InputPopulation


@pytest.fixture
def t_seq():
    """Временная ось: 2 с с шагом 0.5 мс, shape [1, T, 1]."""
    dt = 0.5
    t = np.arange(0, 2000, dt, dtype=np.float32)
    return tf.constant(t[None, :, None])


class TestInputPopulation:
    def test_initial_state_shapes(self):
        gen = ConstantRateGenerator(dt=0.5, params={"rate": 7.0})
        pop = InputPopulation(gen)
        state = pop.get_initial_state()
        assert len(state) == 2
        assert state[0].shape == (1, 1)
        assert state[1].shape == (1, 0)

    def test_derivatives_empty(self):
        gen = ConstantRateGenerator(dt=0.5, params={"rate": 7.0})
        pop = InputPopulation(gen)
        assert pop.derivatives(pop.get_initial_state(), {}) == []

    def test_firing_rate_calls_generator(self):
        gen = ConstantRateGenerator(dt=0.5, params={"rate": 7.0})
        pop = InputPopulation(gen)
        rate = pop.get_firing_rate(pop.get_initial_state())
        assert rate.shape == (1, 1)
        assert np.allclose(rate.numpy(), 7.0)

    def test_firing_rate_shape_n_units(self):
        gen = ConstantRateGenerator(dt=0.5, params={"rate": [1.0, 2.0, 3.0]})
        pop = InputPopulation(gen)
        rate = pop.get_firing_rate(pop.get_initial_state())
        assert rate.shape == (1, 3)
        assert np.allclose(rate.numpy(), [[1.0, 2.0, 3.0]])

    def test_deprecated_warning(self):
        gen = ConstantRateGenerator(dt=0.5, params={"rate": 7.0})
        with pytest.warns(DeprecationWarning):
            InputPopulation(gen)


class TestConstantRateGenerator:
    def test_output_constant(self, t_seq):
        gen = ConstantRateGenerator(dt=0.5, params={"rate": [10.0, 20.0]})
        out = gen(t_seq)
        assert out.shape == (1, t_seq.shape[1], 2)
        assert np.allclose(out.numpy()[..., 0], 10.0)
        assert np.allclose(out.numpy()[..., 1], 20.0)

    def test_tf_function_compatible(self, t_seq):
        gen = ConstantRateGenerator(dt=0.5, params={"rate": 10.0})
        f = tf.function(lambda t: gen(t))
        out = f(t_seq)
        assert out.shape == (1, t_seq.shape[1], 1)


class TestSinusoidalGenerator:
    def test_amplitude_and_offset(self, t_seq):
        gen = SinusoidalGenerator(
            dt=0.5,
            params={
                "amplitude": 10.0,
                "freq": 8.0,
                "phase": 0.0,
                "offset": 12.0,
            },
        )
        out = gen(t_seq).numpy()
        assert out.shape == (1, t_seq.shape[1], 1)
        # relu не обрезает: offset > amplitude
        assert np.max(out) == pytest.approx(22.0, abs=0.2)
        assert np.min(out) == pytest.approx(2.0, abs=0.2)

    def test_frequency(self, t_seq):
        """Доминантная частота спектра равна freq."""
        freq = 8.0
        gen = SinusoidalGenerator(
            dt=0.5,
            params={
                "amplitude": 10.0,
                "freq": freq,
                "phase": 0.0,
                "offset": 12.0,
            },
        )
        out = gen(t_seq).numpy()[0, :, 0]
        spectrum = np.abs(np.fft.rfft(out - np.mean(out)))
        freqs = np.fft.rfftfreq(len(out), d=0.5 / 1000.0)
        dominant = freqs[np.argmax(spectrum)]
        assert dominant == pytest.approx(freq, abs=0.6)

    def test_tf_function_compatible(self, t_seq):
        gen = SinusoidalGenerator(
            dt=0.5,
            params={"amplitude": 1.0, "freq": 8.0, "phase": 0.0, "offset": 5.0},
        )
        f = tf.function(lambda t: gen(t))
        out = f(t_seq)
        assert out.shape == (1, t_seq.shape[1], 1)


class TestVonMisesGenerator:
    def test_nonnegative(self, t_seq):
        gen = VonMisesGenerator(
            dt=0.5,
            params={"mean_rate": 10.0, "R": 0.8, "freq": 8.0, "phase": 0.0},
        )
        out = gen(t_seq).numpy()
        assert np.all(out >= 0.0)

    def test_temporal_mean_equals_mean_rate(self, t_seq):
        """Среднее по времени равно mean_rate (нормировка через I0(kappa))."""
        gen = VonMisesGenerator(
            dt=0.5,
            params={"mean_rate": 10.0, "R": 0.8, "freq": 8.0, "phase": 0.0},
        )
        out = gen(t_seq).numpy()
        assert np.mean(out) == pytest.approx(10.0, rel=0.01)

    def test_dominant_frequency(self, t_seq):
        freq = 8.0
        gen = VonMisesGenerator(
            dt=0.5,
            params={"mean_rate": 10.0, "R": 0.8, "freq": freq, "phase": 0.0},
        )
        out = gen(t_seq).numpy()[0, :, 0]
        spectrum = np.abs(np.fft.rfft(out - np.mean(out)))
        freqs = np.fft.rfftfreq(len(out), d=0.5 / 1000.0)
        dominant = freqs[np.argmax(spectrum)]
        assert dominant == pytest.approx(freq, abs=0.6)

    def test_trainable_r_produces_gradient(self, t_seq):
        gen = VonMisesGenerator(
            dt=0.5,
            params={
                "mean_rate": 10.0,
                "R": {"value": 0.8, "trainable": True, "min": 0.0, "max": 0.99},
                "freq": 8.0,
                "phase": 0.0,
            },
        )
        with tf.GradientTape() as tape:
            out = gen(t_seq)
            loss = tf.reduce_mean(out)
        grads = tape.gradient(loss, gen.trainable_variables)
        assert len(grads) == 1
        assert grads[0] is not None
        assert np.all(np.isfinite(grads[0].numpy()))

    def test_tf_function_compatible(self, t_seq):
        gen = VonMisesGenerator(
            dt=0.5,
            params={"mean_rate": 10.0, "R": 0.8, "freq": 8.0, "phase": 0.0},
        )
        f = tf.function(lambda t: gen(t))
        out = f(t_seq)
        assert out.shape == (1, t_seq.shape[1], 1)


class TestPlaceFieldGenerator:
    @pytest.fixture
    def params(self):
        return {
            "center_x": [0.0],
            "center_y": [0.0],
            "radius": [20.0],
            "peak_rate": [5.0],
            "background_rate": [0.5],
            "theta_modulation_factor": [0.0],
            "precession_slope": [5.0],
            "precession_init_phase": [0.0],
            "phase_outside": [0.0],
            "R": [0.8],
            "freq": [8.0],
        }

    def test_center_temporal_mean_equals_peak(self, t_seq, params):
        """В центре поля средняя по времени частота равна peak_rate."""
        gen = PlaceFieldGenerator(dt=0.5, params=params)
        center = tf.constant(np.zeros((1, 1, 2), dtype=np.float32))
        out = gen(t_seq, extra_inputs=center).numpy()
        assert np.mean(out) == pytest.approx(5.0, rel=0.02)

    def test_docstring_formula(self, t_seq, params):
        """Проверка формулы rate = peak·spatial·θ_in + bg·(1−spatial)·(1−tmf+tmf·θ_out)."""
        from scipy.special import i0

        gen = PlaceFieldGenerator(dt=0.5, params=params)
        x, y = 7.5, -3.0
        t_ms = 123.4
        extra = tf.constant([[[x, y]]], dtype=tf.float32)
        t = tf.constant([[[t_ms]]], dtype=tf.float32)
        out = gen(t, extra_inputs=extra).numpy()[0, 0, 0]

        kappa = gen.kappa.numpy()[0]
        spatial = np.exp(-0.5 * ((x / 20.0) ** 2 + (y / 20.0) ** 2))
        theta_in = np.exp(
            kappa * np.cos(2 * np.pi * 8.0 * t_ms / 1000.0 - np.deg2rad(5.0) * x)
        ) / i0(kappa)
        expected = 5.0 * spatial * theta_in + 0.5 * (1 - spatial) * 1.0
        assert out == pytest.approx(expected, rel=1e-4)

    def test_tmf_zero_removes_outside_modulation(self, t_seq, params):
        """tmf=0: вне поля частота постоянна и равна background_rate."""
        params = {**params, "theta_modulation_factor": [0.0]}
        gen = PlaceFieldGenerator(dt=0.5, params=params)
        far = tf.constant(np.full((1, 1, 2), 500.0, dtype=np.float32))
        far = tf.tile(far, [1, int(t_seq.shape[1]), 1])
        out = gen(t_seq, extra_inputs=far).numpy()
        assert np.mean(out) == pytest.approx(0.5, rel=1e-3)
        assert np.std(out) == pytest.approx(0.0, abs=1e-5)

    def test_tmf_one_modulates_outside(self, t_seq, params):
        """tmf=1: вне поля частота тета-модулирована со средним bg."""
        params = {**params, "theta_modulation_factor": [1.0]}
        gen = PlaceFieldGenerator(dt=0.5, params=params)
        far = tf.constant(np.full((1, 1, 2), 500.0, dtype=np.float32))
        far = tf.tile(far, [1, int(t_seq.shape[1]), 1])
        out = gen(t_seq, extra_inputs=far).numpy()
        assert np.mean(out) == pytest.approx(0.5, rel=0.05)
        assert np.std(out) > 0.1

    def test_velocity_projection_symmetry(self, params):
        """4-колоночный вход: заход слева направо и зеркальный заход дают одинаковый отклик."""
        gen = PlaceFieldGenerator(dt=0.5, params=params)
        t = tf.constant([[[0.0]]], dtype=tf.float32)
        left = tf.constant([[[-10.0, 0.0, 1.0, 0.0]]], dtype=tf.float32)
        right = tf.constant([[[10.0, 0.0, -1.0, 0.0]]], dtype=tf.float32)
        r_left = gen(t, extra_inputs=left).numpy()[0, 0, 0]
        r_right = gen(t, extra_inputs=right).numpy()[0, 0, 0]
        assert r_left == pytest.approx(r_right, rel=1e-4)

    def test_no_position_uses_background(self, t_seq, params):
        gen = PlaceFieldGenerator(dt=0.5, params=params)
        out = gen(t_seq, extra_inputs=None).numpy()
        assert np.allclose(out, 0.5, atol=1e-5)

    def test_float64(self, params):
        neuraltide.config.set_dtype(tf.float64)
        try:
            gen = PlaceFieldGenerator(dt=0.5, params=params)
            t = tf.constant([[[0.0]]], dtype=tf.float64)
            out = gen(t, extra_inputs=None)
            assert out.dtype == tf.float64
        finally:
            neuraltide.config.set_dtype(tf.float32)

    def test_tf_function_compatible(self, t_seq, params):
        gen = PlaceFieldGenerator(dt=0.5, params=params)
        extra = tf.zeros((1, int(t_seq.shape[1]), 2), dtype=tf.float32)
        f = tf.function(lambda t, e: gen(t, extra_inputs=e))
        out = f(t_seq, extra)
        assert out.shape == (1, t_seq.shape[1], 1)
