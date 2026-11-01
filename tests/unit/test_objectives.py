"""
Тесты ResidualLoss / ObservationSpec: нормированные невязки этапа P0.

Покрываются:
- ручной пересчёт weighted L2 с sigma и mask (masked mean);
- инвариантность loss к масштабу маски;
- виды kind: l2/relative/msle/huber/pseudo_huber против эталонных формул;
- residual reconstruction: 0.5 * ||r||^2 == __call__ для всех видов;
- components / target property;
- аддитивность per_step_loss: полная loss == sum_t per_step_loss.
"""

import numpy as np
import pytest
import tensorflow as tf

import neuraltide.config
from neuraltide.core.network import NetworkOutput
from neuraltide.optimization import ObservationSpec, ResidualLoss

T = 6
N = 3


def _rates(rng, scale=5.0, offset=5.0):
    return (offset + scale * rng.standard_normal((1, T, N))).astype(np.float32)


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
        target_t = {name: value[:, t, :] for name, value in target.items()}
        mask_t = None
        if mask is not None:
            mask_t = {name: value[:, t, :] for name, value in mask.items()}
        total += loss.per_step_loss(rates_t, target_t, mask=mask_t)
    return float(total)


def _ref_loss(
    rates, target, kind, sigma=1.0, weight=1.0, mask=None, eps=1e-8, delta=1.0
):
    """Эталонный пересчёт вклада наблюдения на numpy."""
    e = rates - target
    if kind == "relative":
        e = e / (np.abs(target) + eps)
        psi = 0.5 * e**2
    elif kind == "msle":
        e = np.log1p(rates + eps) - np.log1p(target + eps)
        psi = 0.5 * e**2
    elif kind == "huber":
        a = np.abs(e)
        psi = np.where(a <= delta, 0.5 * e**2, delta * (a - 0.5 * delta))
    elif kind == "pseudo_huber":
        psi = delta**2 * (np.sqrt(1.0 + (e / delta) ** 2) - 1.0)
    else:  # l2
        psi = 0.5 * e**2
    m = np.ones_like(psi) if mask is None else mask
    norm = max(float(m.sum()), 1e-8)
    return weight * float((m * psi).sum()) / (norm * sigma**2)


class TestWeightedL2:
    def test_manual_weighted_l2_with_sigma_and_mask(self):
        rng = np.random.default_rng(0)
        rates, target = _rates(rng), _rates(rng, offset=6.0)
        sigma, weight = 2.0, 0.5
        mask = (rng.uniform(size=(1, T, N)) > 0.3).astype(np.float32)
        spec = ObservationSpec(
            name="obs",
            population="pop",
            target=target,
            sigma=sigma,
            weight=weight,
            mask=mask,
        )
        loss = ResidualLoss([spec])

        ref = _ref_loss(rates, target, "l2", sigma=sigma, weight=weight, mask=mask)
        value = float(loss(_predictions(rates), None))
        assert value == pytest.approx(ref, rel=1e-6)

    def test_mask_scale_invariance(self):
        rng = np.random.default_rng(1)
        rates, target = _rates(rng), _rates(rng, offset=6.0)
        mask = (rng.uniform(size=(1, T, N)) > 0.3).astype(np.float32)
        loss_a = ResidualLoss(
            [ObservationSpec(name="o", population="pop", target=target, mask=mask)]
        )
        loss_b = ResidualLoss(
            [
                ObservationSpec(
                    name="o", population="pop", target=target, mask=7.3 * mask
                )
            ]
        )
        pa, pb = _predictions(rates), _predictions(rates)
        assert float(loss_a(pa, None)) == pytest.approx(
            float(loss_b(pb, None)), rel=1e-6
        )

    def test_components_and_target(self):
        rng = np.random.default_rng(2)
        rates, target = _rates(rng), _rates(rng, offset=6.0)
        loss = ResidualLoss(
            [
                ObservationSpec(name="a", population="pop", target=target),
                ObservationSpec(name="b", population="pop", target=target, weight=2.0),
            ]
        )
        pred = _predictions(rates)
        comps = loss.components(pred, None)
        assert set(comps) == {"a", "b"}
        assert float(comps["b"]) == pytest.approx(2.0 * float(comps["a"]), rel=1e-6)
        assert float(loss(pred, None)) == pytest.approx(
            float(comps["a"]) + float(comps["b"]), rel=1e-6
        )
        assert set(loss.target) == {"a", "b"}
        assert loss.target["a"] is loss.observations[0].target


class TestKindsAgainstReference:
    @pytest.mark.parametrize(
        "kind", ["l2", "relative", "msle", "huber", "pseudo_huber"]
    )
    def test_matches_reference_formula(self, kind):
        rng = np.random.default_rng(3)
        # msle требует неотрицательных «частот»
        rates = np.abs(_rates(rng)) + 0.1
        target = np.abs(_rates(rng, offset=6.0)) + 0.1
        sigma, weight = 1.5, 0.7
        mask = (rng.uniform(size=(1, T, N)) > 0.4).astype(np.float32)
        spec = ObservationSpec(
            name="o",
            population="pop",
            target=target,
            kind=kind,
            sigma=sigma,
            weight=weight,
            mask=mask,
        )
        loss = ResidualLoss([spec])
        value = float(loss(_predictions(rates), None))
        ref = _ref_loss(rates, target, kind, sigma=sigma, weight=weight, mask=mask)
        assert value == pytest.approx(ref, rel=1e-5)

    def test_huber_delta_threshold(self):
        delta = 1.0
        rates = np.zeros((1, T, N), dtype=np.float32)
        # ошибки как внутри, так и за пределами квадратичной зоны
        target = np.linspace(-3.0, 3.0, T * N).reshape(1, T, N).astype(np.float32)
        loss = ResidualLoss(
            [
                ObservationSpec(
                    name="o", population="pop", target=target, kind="huber", delta=delta
                )
            ]
        )
        value = float(loss(_predictions(rates), None))
        ref = _ref_loss(rates, target, "huber", delta=delta)
        assert value == pytest.approx(ref, rel=1e-6)


class TestResidualReconstruction:
    @pytest.mark.parametrize(
        "kind", ["l2", "relative", "msle", "huber", "pseudo_huber"]
    )
    def test_half_norm_squared_equals_loss(self, kind):
        rng = np.random.default_rng(4)
        rates = np.abs(_rates(rng)) + 0.1
        target = np.abs(_rates(rng, offset=6.0)) + 0.1
        mask = (rng.uniform(size=(1, T, N)) > 0.4).astype(np.float32)
        spec = ObservationSpec(
            name="o",
            population="pop",
            target=target,
            kind=kind,
            sigma=1.7,
            weight=0.9,
            mask=mask,
        )
        loss = ResidualLoss([spec])
        pred = _predictions(rates)
        r = np.asarray(loss.residual(pred))
        assert 0.5 * float(r @ r) == pytest.approx(float(loss(pred, None)), rel=1e-5)

    def test_residual_concatenates_multiple_specs(self):
        rng = np.random.default_rng(5)
        rates, target = _rates(rng), _rates(rng, offset=6.0)
        loss = ResidualLoss(
            [
                ObservationSpec(name="a", population="pop", target=target),
                ObservationSpec(name="b", population="pop", target=target),
            ]
        )
        r = np.asarray(loss.residual(_predictions(rates)))
        assert r.shape == (2 * T * N,)

    def test_masked_out_elements_have_zero_residual(self):
        rng = np.random.default_rng(6)
        rates, target = _rates(rng), _rates(rng, offset=6.0)
        mask = np.zeros((1, T, N), dtype=np.float32)
        mask[0, 1, 2] = 1.0
        loss = ResidualLoss(
            [ObservationSpec(name="o", population="pop", target=target, mask=mask)]
        )
        r = np.asarray(loss.residual(_predictions(rates))).reshape(1, T, N)
        assert r[0, 1, 2] != 0.0
        assert float(np.abs(r).sum()) == pytest.approx(abs(r[0, 1, 2]), rel=1e-6)


class TestPerStepConsistency:
    def test_full_equals_sum_per_step_no_mask(self):
        rng = np.random.default_rng(7)
        rates, target = _rates(rng), _rates(rng, offset=6.0)
        loss = ResidualLoss(
            [ObservationSpec(name="o", population="pop", target=target)]
        )
        full = float(loss(_predictions(rates), None))
        stepped = _sum_per_step(loss, rates, loss.target)
        assert full == pytest.approx(stepped, rel=1e-6)

    def test_full_equals_sum_per_step_with_mask_slices(self):
        rng = np.random.default_rng(8)
        rates, target = _rates(rng), _rates(rng, offset=6.0)
        mask = (rng.uniform(size=(1, T, N)) > 0.4).astype(np.float32)
        loss = ResidualLoss(
            [ObservationSpec(name="o", population="pop", target=target, mask=mask)]
        )
        full = float(loss(_predictions(rates), None))
        stepped = _sum_per_step(loss, rates, loss.target, mask={"o": mask})
        assert full == pytest.approx(stepped, rel=1e-6)

    def test_full_equals_sum_per_step_time_invariant_mask(self):
        rng = np.random.default_rng(9)
        rates, target = _rates(rng), _rates(rng, offset=6.0)
        mask = np.tile(
            rng.uniform(0.2, 1.0, size=(1, 1, N)).astype(np.float32), (1, T, 1)
        )
        loss = ResidualLoss(
            [ObservationSpec(name="o", population="pop", target=target, mask=mask)]
        )
        full = float(loss(_predictions(rates), None))
        # без передачи срезов: точно для масок, постоянных во времени
        stepped = _sum_per_step(loss, rates, loss.target)
        assert full == pytest.approx(stepped, rel=1e-6)


class TestValidation:
    def test_unknown_kind(self):
        with pytest.raises(ValueError, match="unknown kind"):
            ObservationSpec(
                name="o", population="pop", target=np.zeros((1, T, N)), kind="cauchy"
            )

    def test_duplicate_names(self):
        target = np.zeros((1, T, N), dtype=np.float32)
        with pytest.raises(ValueError, match="unique"):
            ResidualLoss(
                [
                    ObservationSpec(name="o", population="pop", target=target),
                    ObservationSpec(name="o", population="pop", target=target),
                ]
            )

    def test_empty_observations(self):
        with pytest.raises(ValueError, match="at least one"):
            ResidualLoss([])

    def test_bad_sigma(self):
        with pytest.raises(ValueError, match="sigma"):
            ObservationSpec(
                name="o", population="pop", target=np.zeros((1, T, N)), sigma=0.0
            )

    def test_missing_population_key(self):
        target = np.zeros((1, T, N), dtype=np.float32)
        loss = ResidualLoss(
            [ObservationSpec(name="o", population="nope", target=target)]
        )
        with pytest.raises(KeyError, match="nope"):
            loss(_predictions(np.zeros((1, T, N), dtype=np.float32)), None)
