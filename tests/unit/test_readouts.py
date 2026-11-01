"""
Тесты readout-слоёв: BandpassReadout фильтрует реально (не возвращает вход),
HemodynamicReadout сворачивает без смешивания каналов, float64-путь работает.
"""

import numpy as np
import pytest
import tensorflow as tf

import neuraltide.config
from neuraltide.training.readouts import BandpassReadout, HemodynamicReadout

DT = 0.5  # мс → fs = 2000 Гц
T = 2000  # 1 с


def _time():
    return np.arange(T, dtype=np.float32) * DT / 1000.0  # секунды


def _sine(freq_hz):
    return np.sin(2.0 * np.pi * freq_hz * _time())


class TestBandpassReadout:
    def test_in_band_signal_passes(self):
        readout = BandpassReadout(f_low=8.0, f_high=12.0, dt=DT, n_taps=51)
        x = _sine(10.0)[None, :, None]  # 10 Гц в полосе 8–12 Гц
        y = readout(tf.constant(x)).numpy()

        rms_in = float(np.sqrt(np.mean(x**2)))
        rms_out = float(np.sqrt(np.mean(y**2)))
        assert rms_out == pytest.approx(rms_in, rel=0.3)

        # корреляция с исходным сигналом высокая (ФИР в полосе почти без искажений)
        corr = np.corrcoef(x.ravel(), y.ravel())[0, 1]
        assert corr > 0.9

    def test_out_of_band_signal_suppressed(self):
        readout = BandpassReadout(f_low=8.0, f_high=12.0, dt=DT, n_taps=51)
        x = _sine(100.0)[None, :, None]  # 100 Гц вне полосы
        y = readout(tf.constant(x)).numpy()

        rms_in = float(np.sqrt(np.mean(x**2)))
        rms_out = float(np.sqrt(np.mean(y**2)))
        assert rms_out < 0.05 * rms_in

    def test_dc_suppressed(self):
        # узкая полоса у DC требует длинного фильтра (иначе осцилляции
        # Гиббса дают заметный отклик на нулевой частоте)
        readout = BandpassReadout(f_low=40.0, f_high=60.0, dt=DT, n_taps=1001)
        x = np.ones((1, T, 1), dtype=np.float32)
        y = readout(tf.constant(x)).numpy()
        # внутри траектории (вне краевого транзиента длины n_taps/2) DC ≈ 0
        interior = y[0, 600:-600, 0]
        assert np.max(np.abs(interior)) < 0.05

    def test_channels_not_mixed(self):
        readout = BandpassReadout(f_low=8.0, f_high=12.0, dt=DT, n_taps=51)
        x = np.zeros((1, T, 3), dtype=np.float32)
        x[0, :, 0] = _sine(10.0)
        x[0, :, 2] = _sine(10.0) * 0.5
        y = readout(tf.constant(x)).numpy()

        # канал 1 всё время ноль → на выходе тоже должен быть ~ноль
        assert np.max(np.abs(y[0, :, 1])) < 1e-4
        # каналы 0 и 2 прошли фильтр независимо
        assert np.sqrt(np.mean(y[0, :, 0] ** 2)) > 0.5
        assert np.sqrt(np.mean(y[0, :, 2] ** 2)) > 0.25
        # соотношение амплитуд каналов сохранилось (0.5), значит каналы
        # не перемешались
        ratio = np.sqrt(np.mean(y[0, :, 2] ** 2)) / np.sqrt(np.mean(y[0, :, 0] ** 2))
        assert ratio == pytest.approx(0.5, rel=0.05)

    def test_float64(self):
        neuraltide.config.set_dtype(tf.float64)
        readout = BandpassReadout(f_low=8.0, f_high=12.0, dt=DT, n_taps=51)
        x = tf.constant(_sine(10.0)[None, :, None], dtype=tf.float64)
        y = readout(x)
        assert y.dtype == tf.float64
        rms_out = float(tf.sqrt(tf.reduce_mean(y**2)))
        assert rms_out == pytest.approx(np.sqrt(np.mean(_sine(10.0) ** 2)), rel=0.3)


class TestHemodynamicReadout:
    def test_output_shape_and_finite(self):
        readout = HemodynamicReadout(dt=DT)
        x = np.stack([_sine(10.0), _sine(5.0)], axis=-1)[None, :, :]
        y = readout(tf.constant(x)).numpy()
        assert y.shape == x.shape
        assert np.all(np.isfinite(y))

    def test_channels_not_mixed(self):
        readout = HemodynamicReadout(dt=DT)
        x = np.zeros((1, T, 2), dtype=np.float32)
        x[0, :, 0] = _sine(10.0)
        y = readout(tf.constant(x)).numpy()
        # второй канал на входе нулевой → на выходе должен остаться нулевым
        assert np.max(np.abs(y[0, :, 1])) < 1e-5
        # первый канал реально свёрнут (энергия изменилась, но не занулилась)
        assert np.sqrt(np.mean(y[0, :, 0] ** 2)) > 1e-3

    def test_delta_response_matches_kernel(self):
        # отклик на единичный импульс совпадает с ядром HRF
        readout = HemodynamicReadout(dt=DT)
        # build() вызывается лениво — сначала прогон на фиктивном входе
        readout(tf.zeros((1, 8, 1)))
        kernel = readout._kernel.numpy()[:, 0, 0]
        k = len(kernel)
        x = np.zeros((1, 2 * k, 1), dtype=np.float32)
        x[0, k, 0] = 1.0
        y = readout(tf.constant(x)).numpy()[0, :, 0]
        seg = y[k : k + k]
        assert np.allclose(seg, kernel, atol=1e-5)

    def test_float64(self):
        neuraltide.config.set_dtype(tf.float64)
        readout = HemodynamicReadout(dt=DT)
        x = tf.constant(_sine(10.0)[None, :, None], dtype=tf.float64)
        y = readout(x)
        assert y.dtype == tf.float64
        assert np.all(np.isfinite(y.numpy()))
