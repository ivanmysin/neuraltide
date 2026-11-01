"""
Эквивалентность полной loss и per_step_loss, семантика masked mean.

Контракты (этап P0 дорожной карты):
- полная loss ≡ sum_t per_step_loss (аддитивная конвенция) для MSE/MSLE;
- маска работает как masked mean: sum(mask*err)/max(sum(mask), eps),
  масштаб маски значение не меняет;
- MSLELoss принимает mask и не падает с AttributeError в per_step_loss;
- CompositeLoss.components возвращает детерминированные ключи term_i:ClassName,
  а __call__ — сумму компонент.
"""

import numpy as np
import pytest
import tensorflow as tf

import neuraltide.config
from neuraltide.core.network import NetworkOutput
from neuraltide.training.losses import CompositeLoss, MSELoss, MSLELoss

T = 7
N = 3


def _rates(rng, scale=5.0):
    return scale * rng.standard_normal((1, T, N)).astype(np.float32)


def _targets(rng):
    return 5.0 + 2.0 * rng.standard_normal((1, T, N)).astype(np.float32)


def _predictions(rates) -> NetworkOutput:
    return NetworkOutput(
        firing_rates={"pop": tf.constant(rates)},
        stability_loss=tf.constant(0.0, dtype=neuraltide.config.get_dtype()),
        final_state=([], []),
    )


def _sum_per_step(loss, rates, target, mask=None):
    total = tf.constant(0.0, dtype=neuraltide.config.get_dtype())
    for t in range(T):
        rates_t = {"pop": rates[:, t, :]}
        target_t = {"pop": target[:, t, :]}
        mask_t = {"pop": mask[:, t, :]} if mask is not None else None
        total += loss.per_step_loss(rates_t, target_t, mask=mask_t)
    return total


class TestMSEEquivalence:
    def test_full_equals_sum_per_step_no_mask(self):
        rng = np.random.default_rng(0)
        rates, target = _rates(rng), _targets(rng)
        loss = MSELoss({"pop": tf.constant(target)})

        full = float(loss(_predictions(rates), None))
        stepped = float(_sum_per_step(loss, tf.constant(rates), tf.constant(target)))
        assert full == pytest.approx(stepped, rel=1e-6, abs=1e-7)

    def test_full_equals_sum_per_step_time_invariant_mask(self):
        rng = np.random.default_rng(1)
        rates, target = _rates(rng), _targets(rng)
        # веса постоянны во времени, но разные по каналам
        mask = np.tile(
            rng.uniform(0.2, 1.0, size=(1, 1, N)).astype(np.float32), (1, T, 1)
        )
        loss = MSELoss({"pop": tf.constant(target)}, mask={"pop": tf.constant(mask)})

        full = float(loss(_predictions(rates), None))
        stepped = float(_sum_per_step(loss, tf.constant(rates), tf.constant(target)))
        assert full == pytest.approx(stepped, rel=1e-6, abs=1e-7)

    def test_masked_mean_semantics(self):
        rng = np.random.default_rng(2)
        rates, target = _rates(rng), _targets(rng)
        mask = (rng.uniform(size=(1, T, 1)) > 0.4).astype(np.float32)
        loss = MSELoss({"pop": tf.constant(target)}, mask={"pop": tf.constant(mask)})

        err = (rates - target) ** 2
        expected = float(np.sum(mask * err) / np.sum(mask))
        full = float(loss(_predictions(rates), None))
        assert full == pytest.approx(expected, rel=1e-6)

    def test_mask_scale_invariant(self):
        rng = np.random.default_rng(3)
        rates, target = _rates(rng), _targets(rng)
        base = (rng.uniform(size=(1, T, 1)) > 0.5).astype(np.float32)
        loss_1 = MSELoss(
            {"pop": tf.constant(target)}, mask={"pop": tf.constant(1.0 * base)}
        )
        loss_7 = MSELoss(
            {"pop": tf.constant(target)}, mask={"pop": tf.constant(7.0 * base)}
        )

        full_1 = float(loss_1(_predictions(rates), None))
        full_7 = float(loss_7(_predictions(rates), None))
        assert full_1 == pytest.approx(full_7, rel=1e-6)
        # масштаб маски не меняет и нормировку относительно немаскированной
        unmasked = MSELoss({"pop": tf.constant(target)})(_predictions(rates), None)
        assert full_1 != pytest.approx(float(unmasked), rel=1e-3)

    def test_all_zero_mask_gives_zero_loss(self):
        rng = np.random.default_rng(4)
        rates, target = _rates(rng), _targets(rng)
        mask = np.zeros((1, T, 1), dtype=np.float32)
        loss = MSELoss({"pop": tf.constant(target)}, mask={"pop": tf.constant(mask)})
        full = float(loss(_predictions(rates), None))
        assert full == 0.0

    def test_masked_per_step_with_exact_slices(self):
        rng = np.random.default_rng(5)
        rates, target = _rates(rng), _targets(rng)
        mask = (rng.uniform(size=(1, T, 1)) > 0.3).astype(np.float32)
        loss = MSELoss({"pop": tf.constant(target)}, mask={"pop": tf.constant(mask)})

        full = float(loss(_predictions(rates), None))
        stepped = float(
            _sum_per_step(
                loss, tf.constant(rates), tf.constant(target), mask=tf.constant(mask)
            )
        )
        assert full == pytest.approx(stepped, rel=1e-6, abs=1e-7)


class TestMSLEEquivalence:
    def test_mask_no_attribute_error(self):
        rng = np.random.default_rng(6)
        rates, target = np.abs(_rates(rng)) + 0.5, np.abs(_targets(rng)) + 1.0
        mask = np.ones((1, T, 1), dtype=np.float32)
        loss = MSLELoss({"pop": tf.constant(target)}, mask={"pop": tf.constant(mask)})
        # раньше падало с AttributeError: 'MSLELoss' object has no attribute 'mask'
        full = float(loss(_predictions(rates), None))
        stepped = float(_sum_per_step(loss, tf.constant(rates), tf.constant(target)))
        assert full == pytest.approx(stepped, rel=1e-6, abs=1e-7)

    def test_masked_mean_semantics(self):
        rng = np.random.default_rng(7)
        rates, target = np.abs(_rates(rng)) + 0.5, np.abs(_targets(rng)) + 1.0
        mask = (rng.uniform(size=(1, T, 1)) > 0.4).astype(np.float32)
        eps = 1.0
        loss = MSLELoss(
            {"pop": tf.constant(target)}, eps=eps, mask={"pop": tf.constant(mask)}
        )

        err = (np.log(rates + eps) - np.log(target + eps)) ** 2
        expected = float(np.sum(mask * err) / np.sum(mask))
        full = float(loss(_predictions(rates), None))
        assert full == pytest.approx(expected, rel=1e-6)

    def test_full_equals_sum_per_step_masked_slices(self):
        rng = np.random.default_rng(8)
        rates, target = np.abs(_rates(rng)) + 0.5, np.abs(_targets(rng)) + 1.0
        mask = (rng.uniform(size=(1, T, 1)) > 0.5).astype(np.float32)
        loss = MSLELoss({"pop": tf.constant(target)}, mask={"pop": tf.constant(mask)})

        full = float(loss(_predictions(rates), None))
        stepped = float(
            _sum_per_step(
                loss, tf.constant(rates), tf.constant(target), mask=tf.constant(mask)
            )
        )
        assert full == pytest.approx(stepped, rel=1e-6, abs=1e-7)


class TestFloat64:
    def test_mse_equivalence_float64(self):
        neuraltide.config.set_dtype(tf.float64)
        rng = np.random.default_rng(9)
        rates = _rates(rng).astype(np.float64)
        target = _targets(rng).astype(np.float64)
        mask = (rng.uniform(size=(1, T, 1)) > 0.4).astype(np.float64)
        loss = MSELoss({"pop": tf.constant(target)}, mask={"pop": tf.constant(mask)})

        full = loss(_predictions(rates), None)
        assert full.dtype == tf.float64
        stepped = _sum_per_step(
            loss, tf.constant(rates), tf.constant(target), mask=tf.constant(mask)
        )
        assert float(full) == pytest.approx(float(stepped), rel=1e-10)


class TestCompositeComponents:
    def test_components_keys_and_sum(self):
        rng = np.random.default_rng(10)
        rates, target = np.abs(_rates(rng)) + 0.5, _targets(rng)
        target2 = np.abs(_targets(rng)) + 1.0
        mse = MSELoss({"pop": tf.constant(target)})
        msle = MSLELoss({"pop": tf.constant(target2)})
        composite = CompositeLoss([(0.5, mse), (2.0, msle)])

        preds = _predictions(rates)
        components = composite.components(preds, None)
        assert list(components.keys()) == ["term_0:MSELoss", "term_1:MSLELoss"]

        total = composite(preds, None)
        expected = 0.5 * mse(preds, None) + 2.0 * msle(preds, None)
        assert float(total) == pytest.approx(float(expected), rel=1e-6)

    def test_components_deterministic_keys(self):
        mse = MSELoss({"pop": tf.constant(np.zeros((1, T, N), np.float32))})
        composite = CompositeLoss([(1.0, mse), (3.0, mse)])
        preds = _predictions(np.ones((1, T, N), np.float32))
        keys_a = list(composite.components(preds, None).keys())
        keys_b = list(composite.components(preds, None).keys())
        assert keys_a == keys_b == ["term_0:MSELoss", "term_1:MSELoss"]
