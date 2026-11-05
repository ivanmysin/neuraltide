"""
Тесты преобразований координат neuraltide.optimization.transforms.

Покрываются: round-trip φ ↔ θ около и вдали от границ для всех
видов преобразований, jacobian против центральных конечных разностей,
конечность и ненулевость градиентов около нижней границы у log/softplus,
работа в float32 (по умолчанию) и float64.
"""

import numpy as np
import pytest
import tensorflow as tf

import neuraltide.config
from neuraltide.optimization.transforms import (
    AffineTransform,
    BoundedLogTransform,
    IdentityTransform,
    LogTransform,
    LogitTransform,
    ScaledSigmoidTransform,
    SoftplusTransform,
    Transform,
    make_transform,
    register_transform,
)

F32 = np.float32


def _eps32() -> float:
    return float(np.finfo(F32).eps)


class TestContract:
    def test_base_class_raises_not_implemented(self):
        tr = Transform()
        phi = tf.constant([0.0])
        with pytest.raises(NotImplementedError):
            tr.forward(phi)
        with pytest.raises(NotImplementedError):
            tr.inverse(phi)
        with pytest.raises(NotImplementedError):
            tr.jacobian(phi)

    def test_names(self):
        assert IdentityTransform().name == "identity"
        assert AffineTransform().name == "affine"
        assert LogTransform().name == "log"
        assert LogitTransform(0.0, 1.0).name == "logit"
        assert ScaledSigmoidTransform(0.0, 1.0).name == "scaled_sigmoid"
        assert SoftplusTransform().name == "softplus"
        assert BoundedLogTransform(0.1, 1.0).name == "bounded_log"

    def test_scalar_and_tensor_inputs(self):
        tr = make_transform("log")
        scalar_out = tr.forward(0.5)
        assert np.isclose(float(scalar_out), np.exp(0.5), rtol=1e-6)
        phi = tf.constant([0.0, 1.0])
        assert tr.forward(phi).shape == (2,)
        assert tr.inverse(phi).shape == (2,)
        assert tr.jacobian(phi).shape == (2,)

    def test_output_dtype_follows_global_dtype(self):
        phi = tf.constant([0.0, 1.0])
        tr = make_transform("scaled_sigmoid", min=0.0, max=1.0, s=2.0)
        assert tr.forward(phi).dtype == tf.float32
        assert tr.inverse(phi).dtype == tf.float32
        assert tr.jacobian(phi).dtype == tf.float32


class TestFactory:
    def test_make_all_kinds(self):
        assert isinstance(make_transform("identity"), IdentityTransform)
        assert isinstance(make_transform("affine", a=2.0, b=1.0), AffineTransform)
        assert isinstance(make_transform("log"), LogTransform)
        assert isinstance(make_transform("logit", min=0.0, max=1.0), LogitTransform)
        assert isinstance(
            make_transform("scaled_sigmoid", min=0.0, max=1.0, s=2.0),
            ScaledSigmoidTransform,
        )
        assert isinstance(make_transform("softplus"), SoftplusTransform)
        assert isinstance(
            make_transform("bounded_log", min=0.1, max=1.0),
            BoundedLogTransform,
        )

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="Unknown transform kind"):
            make_transform("does_not_exist")

    def test_register_custom_transform(self):
        class DoubleTransform(IdentityTransform):
            name = "double_for_test"

        register_transform("double_for_test", DoubleTransform)
        tr = make_transform("double_for_test")
        assert isinstance(tr, DoubleTransform)
        assert tr.name == "double_for_test"

    def test_invalid_parameters_raise(self):
        with pytest.raises(ValueError, match="non-zero"):
            AffineTransform(a=0.0)
        with pytest.raises((TypeError, ValueError)):
            LogitTransform()
        with pytest.raises(ValueError, match="min < max"):
            ScaledSigmoidTransform(min=2.0, max=1.0)
        with pytest.raises(ValueError, match="scale"):
            ScaledSigmoidTransform(min=0.0, max=1.0, s=0.0)
        with pytest.raises(ValueError, match="min < max"):
            BoundedLogTransform(min=1.0, max=1.0)
        with pytest.raises(ValueError, match="floor"):
            LogTransform(floor=0.0)
        with pytest.raises(ValueError, match="floor"):
            SoftplusTransform(floor=-1.0)


ROUNDTRIP_CASES = [
    ("identity", {}, [-3.0, 0.0, 2.5]),
    ("affine", {"a": 2.0, "b": -1.0}, [-3.0, 0.0, 2.5]),
    ("log", {}, [-20.0, -2.0, 0.5, 3.0]),
    ("logit", {"min": 0.0, "max": 1.0}, [-8.0, -1.0, 0.0, 4.0]),
    (
        "scaled_sigmoid",
        {"min": -2.0, "max": 5.0, "s": 2.0},
        [-8.0, -1.0, 0.0, 6.0],
    ),
    ("softplus", {}, [-20.0, -5.0, 0.0, 3.0]),
    ("bounded_log", {"min": 0.1, "max": 3.0}, [-8.0, -1.0, 0.0, 6.0]),
]


class TestRoundTrip:
    @pytest.mark.parametrize("kind,kwargs,phis", ROUNDTRIP_CASES)
    def test_phi_theta_roundtrip(self, kind, kwargs, phis):
        """inverse(forward(phi)) ≈ phi и forward(inverse(forward(phi))) ≈ theta.

        Проверяется и около границ (очень отрицательные φ у log/softplus,
        |φ| большой у sigmoid-образных), и вдали от них.
        """
        tr = make_transform(kind, **kwargs)
        phi = tf.constant(phis, dtype=F32)
        theta = tr.forward(phi)
        phi2 = tr.inverse(theta)
        theta2 = tr.forward(phi2)
        assert theta.dtype == F32
        assert theta2.dtype == F32
        np.testing.assert_allclose(phi2.numpy(), phi.numpy(), rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(theta2.numpy(), theta.numpy(), rtol=1e-4, atol=1e-6)


class TestLogBoundary:
    def test_roundtrip_near_floor(self):
        tr = LogTransform()
        tiny = np.finfo(F32).tiny
        thetas = np.array(
            [tiny * 100, tiny * 1e10, 1e-30, 1e-10, 1e-5, 1.0],
            dtype=F32,
        )
        theta = tf.constant(thetas)
        rt = tr.forward(tr.inverse(theta)).numpy()
        np.testing.assert_allclose(rt, thetas, rtol=1e-5, atol=0.0)

    def test_inverse_clips_below_floor(self):
        tr = LogTransform()
        tiny = np.finfo(F32).tiny
        theta = tf.constant([-5.0, 0.0, tiny * 0.5], dtype=F32)
        phi = tr.inverse(theta)
        assert np.all(np.isfinite(phi.numpy()))
        restored = tr.forward(phi).numpy()
        assert np.all(np.isfinite(restored))
        assert np.all(restored >= tiny)

    def test_custom_floor(self):
        tr = LogTransform(floor=1e-3)
        phi = tr.inverse(tf.constant([1e-8, 0.0], dtype=F32))
        assert np.all(np.isfinite(phi.numpy()))
        assert np.allclose(tr.forward(phi).numpy(), 1e-3, rtol=1e-2)


class TestSigmoidBoundaries:
    @pytest.mark.parametrize(
        "cls,kwargs",
        [
            (LogitTransform, {"min": 0.0, "max": 1.0}),
            (ScaledSigmoidTransform, {"min": -2.0, "max": 5.0, "s": 2.0}),
            (BoundedLogTransform, {"min": 0.1, "max": 3.0}),
        ],
    )
    def test_roundtrip_near_bounds(self, cls, kwargs):
        tr = cls(**kwargs)
        lo, hi = kwargs["min"], kwargs["max"]
        span = hi - lo
        fracs = [1e-4, 1e-2, 0.5, 1.0 - 1e-2, 1.0 - 1e-4]
        thetas = np.array([lo + span * f for f in fracs], dtype=F32)
        theta = tf.constant(thetas)
        phi = tr.inverse(theta)
        assert np.all(np.isfinite(phi.numpy()))
        rt = tr.forward(phi).numpy()
        np.testing.assert_allclose(rt, thetas, rtol=1e-4, atol=span * 1e-5)

    @pytest.mark.parametrize(
        "cls,kwargs",
        [
            (LogitTransform, {"min": 0.0, "max": 1.0}),
            (ScaledSigmoidTransform, {"min": -2.0, "max": 5.0, "s": 2.0}),
            (BoundedLogTransform, {"min": 0.1, "max": 3.0}),
        ],
    )
    def test_inverse_clips_outside_interval(self, cls, kwargs):
        """θ на границах и за пределами: φ конечный, forward(φ) внутри (lo, hi)."""
        tr = cls(**kwargs)
        lo, hi = kwargs["min"], kwargs["max"]
        theta = tf.constant([lo, hi, lo - 1.0, hi + 1.0], dtype=F32)
        phi = tr.inverse(theta)
        assert np.all(np.isfinite(phi.numpy()))
        rt = tr.forward(phi).numpy()
        assert np.all(rt >= lo)
        assert np.all(rt <= hi)
        # строго внутри из-за клиппинга с margin
        assert np.all(rt > lo)
        assert np.all(rt < hi)


class TestSoftplusBoundary:
    def test_roundtrip_near_floor(self):
        tr = SoftplusTransform()
        tiny = np.finfo(F32).tiny
        # theta ~ 1e-3 и больше: ниже в f32 log1p-округление softplus
        # неизбежно теряет относительную точность (это предел представимости).
        thetas = np.array([tiny * 1e10, 1e-6, 1e-3, 0.1, 2.0, 50.0], dtype=F32)
        theta = tf.constant(thetas)
        phi = tr.inverse(theta)
        assert np.all(np.isfinite(phi.numpy()))
        rt = tr.forward(phi).numpy()
        np.testing.assert_allclose(rt, thetas, rtol=1e-3, atol=1e-8)

    def test_inverse_clips_below_floor(self):
        tr = SoftplusTransform()
        tiny = np.finfo(F32).tiny
        theta = tf.constant([-1.0, 0.0, tiny * 0.5], dtype=F32)
        phi = tr.inverse(theta)
        assert np.all(np.isfinite(phi.numpy()))
        restored = tr.forward(phi).numpy()
        assert np.all(np.isfinite(restored))
        assert np.all(restored >= tiny)


JACOBIAN_CASES = [
    ("identity", {}, [-1.0, 0.0, 1.5]),
    ("affine", {"a": 2.0, "b": -1.0}, [-1.0, 0.0, 1.5]),
    ("log", {}, [-3.0, 0.0, 1.5]),
    ("logit", {"min": 0.0, "max": 1.0}, [-1.0, 0.0, 1.5]),
    ("scaled_sigmoid", {"min": -2.0, "max": 5.0, "s": 2.0}, [-1.0, 0.0, 1.5]),
    ("softplus", {}, [-4.0, 0.0, 1.5]),
    ("bounded_log", {"min": 0.1, "max": 3.0}, [-1.0, 0.0, 1.5]),
]


class TestJacobian:
    @pytest.mark.parametrize("kind,kwargs,points", JACOBIAN_CASES)
    def test_jacobian_matches_central_finite_differences(self, kind, kwargs, points):
        tr = make_transform(kind, **kwargs)
        phi = tf.constant(points, dtype=F32)
        h = _eps32() ** (1.0 / 3.0)
        analytic = tr.jacobian(phi).numpy()
        numeric = (tr.forward(phi + h).numpy() - tr.forward(phi - h).numpy()) / (
            2.0 * h
        )
        assert analytic.shape == numeric.shape == (len(points),)
        np.testing.assert_allclose(analytic, numeric, rtol=1e-2, atol=1e-4)

    @pytest.mark.parametrize("kind,kwargs,points", JACOBIAN_CASES)
    def test_jacobian_matches_tape_gradient(self, kind, kwargs, points):
        tr = make_transform(kind, **kwargs)
        phi = tf.constant(points, dtype=F32)
        with tf.GradientTape() as tape:
            tape.watch(phi)
            theta = tr.forward(phi)
        tape_grad = tape.gradient(theta, phi).numpy()
        np.testing.assert_allclose(
            tr.jacobian(phi).numpy(), tape_grad, rtol=1e-5, atol=1e-7
        )


class TestGradientsNearBounds:
    @pytest.mark.parametrize("kind", ["log", "softplus"])
    def test_gradient_finite_and_nonzero_near_lower_bound(self, kind):
        """Около нижней границы градиент конечен и не исчезает (log/softplus)."""
        tr = make_transform(kind)
        phi = tf.constant([-15.0, -10.0], dtype=F32)
        with tf.GradientTape() as tape:
            tape.watch(phi)
            theta = tr.forward(phi)
        grad = tape.gradient(theta, phi)
        assert grad is not None
        assert np.all(np.isfinite(grad.numpy()))
        assert np.all(np.abs(grad.numpy()) > 0.0)
        assert theta.numpy().shape == (2,)

    def test_gradient_finite_and_nonzero_far_in_tail(self):
        """Глубоко в отрицательном хвосте (θ ≪ 1) градиент конечен
        и не обнуляется, пока exp представима в данном dtype (log/softplus)."""
        phi = tf.constant([-30.0, -20.0], dtype=F32)
        for kind in ("log", "softplus"):
            tr = make_transform(kind)
            with tf.GradientTape() as tape:
                tape.watch(phi)
                theta = tr.forward(phi)
            grad = tape.gradient(theta, phi)
            assert grad is not None
            assert np.all(np.isfinite(grad.numpy()))
            assert np.all(np.abs(grad.numpy()) > 0.0)
            assert theta.numpy().shape == (2,)


class TestFloat64:
    def test_roundtrip_and_jacobian_float64(self):
        neuraltide.config.set_dtype(tf.float64)
        tr = make_transform("scaled_sigmoid", min=0.0, max=1.0, s=2.0)
        phi = tf.constant([-1.0, 0.0, 1.5], dtype=tf.float64)
        theta = tr.forward(phi)
        assert theta.dtype == tf.float64
        phi2 = tr.inverse(theta)
        np.testing.assert_allclose(phi2.numpy(), phi.numpy(), rtol=1e-12, atol=1e-14)
        analytic = tr.jacobian(phi).numpy()
        h = float(np.finfo(np.float64).eps) ** (1.0 / 3.0)
        numeric = (tr.forward(phi + h).numpy() - tr.forward(phi - h).numpy()) / (
            2.0 * h
        )
        np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-9)

    def test_log_float64_floor_uses_dtype(self):
        neuraltide.config.set_dtype(tf.float64)
        tr = LogTransform()
        tiny64 = np.finfo(np.float64).tiny
        theta = tf.constant([tiny64 * 100, 1e-250], dtype=tf.float64)
        phi = tr.inverse(theta)
        assert np.all(np.isfinite(phi.numpy()))
        rt = tr.forward(phi).numpy()
        np.testing.assert_allclose(rt, theta.numpy(), rtol=1e-12, atol=0.0)
