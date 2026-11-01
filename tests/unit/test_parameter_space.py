"""
Тесты ParameterSpace: реестр оптимизируемых параметров сети.

Сеть: 2 популяции WilsonCowan (exc, 2 юнита; inh, 1 юнит) + 2 синапса
(StaticSynapse exc->inh, TsodyksMarkramSynapse inh->exc).

Покрываются: стабильные полные имена, shape/group, pack -> unpack
совпадает с исходными значениями, apply_to_model реально меняет
переменные, градиентный поток phi -> unpack -> theta через
GradientTape, project возвращает допустимые theta, поведение без
transforms == identity, предупреждения о неизвестных переменных.
"""

import numpy as np
import pytest
import tensorflow as tf

import neuraltide.config
from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.integrators import EulerIntegrator
from neuraltide.optimization import (
    LogTransform,
    LogitTransform,
    ParameterEntry,
    ParameterSpace,
    ScaledSigmoidTransform,
)
from neuraltide.populations import WilsonCowan
from neuraltide.synapses import StaticSynapse, TsodyksMarkramSynapse

EXPECTED_NAMES = [
    "exc/w",
    "exc/I_ext",
    "inh/w",
    "exc->inh/gsyn_max",
    "inh->exc/gsyn_max",
    "inh->exc/tau_f",
    "inh->exc/tau_d",
    "inh->exc/tau_r",
    "inh->exc/Uinc",
]


def _build_network() -> NetworkRNN:
    dt = 0.5
    exc = WilsonCowan(
        dt=dt,
        params={
            "tau": {"value": [10.0, 10.0], "trainable": False},
            "a": {"value": [1.0, 1.2], "trainable": False},
            "theta": {"value": [0.0, 0.0], "trainable": False},
            "w": {"value": [1.0, 0.8], "trainable": True, "min": 0.0, "max": 5.0},
            "I_ext": {"value": [1.0, 0.5], "trainable": True},
            "max_rate": {"value": [100.0, 100.0], "trainable": False},
        },
        name="exc",
    )
    inh = WilsonCowan(
        dt=dt,
        params={
            "tau": 10.0,
            "a": 1.0,
            "theta": 0.0,
            "w": {"value": 1.0, "trainable": True},
            "I_ext": 0.5,
            "max_rate": 100.0,
        },
        name="inh",
    )
    syn_static = StaticSynapse(
        n_pre=2,
        n_post=1,
        dt=dt,
        params={
            "gsyn_max": {
                "value": [[0.1], [0.2]],
                "trainable": True,
                "min": 0.0,
                "max": 1.0,
            },
            "pconn": {"value": [[1.0], [1.0]], "trainable": False},
            "e_r": 0.0,
        },
        name="exc->inh",
    )
    syn_stp = TsodyksMarkramSynapse(
        n_pre=1,
        n_post=2,
        dt=dt,
        params={
            "gsyn_max": {"value": [[0.1, 0.1]], "trainable": True},
            "tau_f": {"value": 20.0, "trainable": True, "min": 6.0, "max": 240.0},
            "tau_d": {"value": 5.0, "trainable": True, "min": 2.0, "max": 15.0},
            "tau_r": {"value": 200.0, "trainable": True, "min": 91.0, "max": 1300.0},
            "Uinc": {"value": 0.2, "trainable": True, "min": 0.04, "max": 0.7},
            "pconn": {"value": [[1.0, 1.0]], "trainable": False},
            "e_r": 0.0,
        },
        name="inh->exc",
    )
    graph = NetworkGraph(dt=dt)
    graph.add_population("exc", exc)
    graph.add_population("inh", inh)
    graph.add_synapse("exc->inh", syn_static, src="exc", tgt="inh")
    graph.add_synapse("inh->exc", syn_stp, src="inh", tgt="exc")
    return NetworkRNN(graph, EulerIntegrator())


@pytest.fixture
def network() -> NetworkRNN:
    return _build_network()


def _variables(space: ParameterSpace):
    return {name: e.variable for name, e in space.entries.items()}


class TestIntrospection:
    def test_names_are_stable_and_full(self, network):
        space = ParameterSpace.from_network(network)
        assert space.names == EXPECTED_NAMES

    def test_size(self, network):
        space = ParameterSpace.from_network(network)
        assert space.size == 2 + 2 + 1 + 2 + 2 + 2 + 2 + 2 + 2

    def test_shape(self, network):
        space = ParameterSpace.from_network(network)
        assert space.shape("exc/w") == tf.TensorShape([2])
        assert space.shape("exc/I_ext") == tf.TensorShape([2])
        assert space.shape("inh/w") == tf.TensorShape([1])
        assert space.shape("exc->inh/gsyn_max") == tf.TensorShape([2, 1])
        assert space.shape("inh->exc/tau_f") == tf.TensorShape([1, 2])

    def test_group(self, network):
        space = ParameterSpace.from_network(network)
        assert space.group("exc/w") == "population"
        assert space.group("exc/I_ext") == "population"
        assert space.group("inh/w") == "population"
        assert space.group("exc->inh/gsyn_max") == "synapse"
        assert space.group("inh->exc/Uinc") == "synapse"


class TestPackUnpack:
    def test_pack_unpack_roundtrip_matches_variables(self, network):
        """pack(None) -> unpack == текущие значения переменных (identity)."""
        space = ParameterSpace.from_network(network)
        phi = space.pack()
        assert phi.dtype == tf.float32
        assert phi.shape == (space.size,)
        theta = space.unpack(phi)
        assert set(theta.keys()) == set(EXPECTED_NAMES)
        for name, var in _variables(space).items():
            np.testing.assert_allclose(
                theta[name].numpy(), var.numpy(), rtol=1e-6, atol=1e-8
            )

    def test_pack_from_dict_uses_full_names(self, network):
        space = ParameterSpace.from_network(network)
        values = {name: np.full(space.shape(name), 1.25) for name in space.names}
        phi = space.pack(values)
        theta = space.unpack(phi)
        for name in space.names:
            np.testing.assert_allclose(
                theta[name].numpy(), np.full(space.shape(name), 1.25), atol=1e-6
            )

    def test_pack_from_dict_missing_key_raises(self, network):
        space = ParameterSpace.from_network(network)
        with pytest.raises(ValueError, match="missing key"):
            space.pack({"exc/w": [1.0, 1.0]})

    def test_pack_from_dict_wrong_shape_raises(self, network):
        space = ParameterSpace.from_network(network)
        values = {name: np.full(space.shape(name), 1.0) for name in space.names}
        values["exc/w"] = [1.0, 2.0, 3.0]
        with pytest.raises(ValueError, match="elements"):
            space.pack(values)

    def test_pack_from_dict_broadcasts_scalar(self, network):
        space = ParameterSpace.from_network(network)
        values = {name: 0.5 for name in space.names}
        theta = space.unpack(space.pack(values))
        for name in space.names:
            np.testing.assert_allclose(
                theta[name].numpy(), np.full(space.shape(name), 0.5), atol=1e-6
            )

    def test_unpack_rejects_wrong_size(self, network):
        space = ParameterSpace.from_network(network)
        with pytest.raises((ValueError, tf.errors.InvalidArgumentError)):
            space.unpack(tf.zeros([space.size + 1], dtype=tf.float32))


class TestApplyToModel:
    def test_apply_to_model_updates_variables(self, network):
        space = ParameterSpace.from_network(network)
        current = space.unpack(space.pack())
        new_values = {name: value * 1.5 + 0.01 for name, value in current.items()}
        space.apply_to_model(new_values)
        for name, var in _variables(space).items():
            np.testing.assert_allclose(
                var.numpy(), new_values[name].numpy(), rtol=1e-6, atol=1e-8
            )

    def test_apply_to_model_missing_key_raises(self, network):
        space = ParameterSpace.from_network(network)
        with pytest.raises(ValueError, match="missing key"):
            space.apply_to_model({"exc/w": [1.0, 1.0]})

    def test_pack_after_apply_reads_updated_values(self, network):
        """round-trip через apply_to_model не накапливает скрытое состояние."""
        space = ParameterSpace.from_network(network)
        theta = space.unpack(space.pack())
        theta["exc/w"] = tf.constant([3.0, 4.0])
        space.apply_to_model(theta)
        theta2 = space.unpack(space.pack())
        np.testing.assert_allclose(theta2["exc/w"].numpy(), [3.0, 4.0])


class TestGradientFlow:
    def test_gradient_phi_to_theta(self, network):
        space = ParameterSpace.from_network(network)
        phi = space.pack()
        with tf.GradientTape() as tape:
            tape.watch(phi)
            theta = space.unpack(phi)
            loss = tf.add_n([tf.reduce_sum(value * value) for value in theta.values()])
        grad = tape.gradient(loss, phi)
        assert grad is not None
        assert grad.shape == (space.size,)
        assert np.all(np.isfinite(grad.numpy()))
        assert np.linalg.norm(grad.numpy()) > 0.0

    def test_gradient_flows_through_transform(self, network):
        """С transform'ом градиент доходит до φ по forward-ветви."""
        space = ParameterSpace.from_network(
            network, transforms={"exc/w": LogTransform()}
        )
        phi = space.pack()
        with tf.GradientTape() as tape:
            tape.watch(phi)
            theta = space.unpack(phi)
            loss = tf.reduce_sum(theta["exc/w"] * theta["exc/w"])
        grad = tape.gradient(loss, phi)
        # первые две компоненты φ отвечают exc/w
        assert np.all(grad.numpy()[:2] != 0.0)


class TestTransforms:
    def test_explicit_transforms_roundtrip(self, network):
        """С явными transform'ами pack->unpack восстанавливает θ."""
        transforms = {
            "exc/w": ScaledSigmoidTransform(min=0.0, max=5.0),
            "inh->exc/Uinc": LogitTransform(min=0.04, max=0.7),
            "exc/I_ext": LogTransform(),
        }
        space = ParameterSpace.from_network(network, transforms=transforms)
        phi = space.pack()
        theta = space.unpack(phi)
        for name, var in _variables(space).items():
            np.testing.assert_allclose(
                theta[name].numpy(), var.numpy(), rtol=1e-5, atol=1e-6
            )

    def test_transform_changes_coordinates(self, network):
        """С transform'ом φ отличается от θ (координаты действительно
        преобразуются), а unpack возвращает исходные θ."""
        space_plain = ParameterSpace.from_network(network)
        space_log = ParameterSpace.from_network(
            network, transforms={"exc/I_ext": LogTransform()}
        )
        phi_plain = space_plain.pack()
        phi_log = space_log.pack()
        w_offset = list(space_plain.names).index("exc/I_ext")
        assert not np.allclose(
            phi_log.numpy()[w_offset : w_offset + 2],
            phi_plain.numpy()[w_offset : w_offset + 2],
        )
        theta = space_log.unpack(phi_log)
        np.testing.assert_allclose(theta["exc/I_ext"].numpy(), [1.0, 0.5], rtol=1e-6)

    def test_unknown_transform_name_warns(self, network):
        with pytest.warns(UserWarning, match="unknown parameters"):
            ParameterSpace.from_network(
                network, transforms={"nope/param": LogTransform()}
            )

    def test_unknown_trainable_variable_warns(self, network):
        extra = tf.Variable([0.1], dtype=tf.float32, name="extra_var")
        network._cached_trainable_vars.append(extra)
        with pytest.warns(UserWarning, match="not matched"):
            ParameterSpace.from_network(network)


class TestProject:
    def test_project_respects_physical_bounds(self, network):
        space = ParameterSpace.from_network(
            network, transforms={"exc/I_ext": LogTransform()}
        )
        phi0 = space.pack()
        phi_bad = phi0 + 20.0
        phi_proj = space.project(phi_bad)
        assert np.all(np.isfinite(phi_proj.numpy()))
        theta = space.unpack(phi_proj)
        w = theta["exc/w"].numpy()
        assert np.all(w >= 0.0) and np.all(w <= 5.0)
        iext = theta["exc/I_ext"].numpy()
        assert np.all(iext > 0.0) and np.all(np.isfinite(iext))
        gsyn = theta["exc->inh/gsyn_max"].numpy()
        assert np.all(gsyn >= 0.0) and np.all(gsyn <= 1.0)

    def test_project_downward_respects_bounds(self, network):
        space = ParameterSpace.from_network(network)
        phi_proj = space.project(space.pack() - 100.0)
        theta = space.unpack(phi_proj)
        w = theta["exc/w"].numpy()
        assert np.all(w >= 0.0)
        tf_var = theta["inh->exc/tau_f"].numpy()
        assert np.all(tf_var >= 6.0)

    def test_project_is_identity_inside_bounds(self, network):
        """Внутри допустимой области project не двигает точку."""
        space = ParameterSpace.from_network(network)
        phi = space.pack()
        phi_proj = space.project(phi)
        np.testing.assert_allclose(phi_proj.numpy(), phi.numpy(), atol=1e-6)


class TestAddEntry:
    def test_add_entry_manual_nuisance(self, network):
        space = ParameterSpace.from_network(network)
        values = space.unpack(space.pack())  # снимок до добавления записи
        size_before = space.size
        space.add_entry(
            ParameterEntry(
                full_name="noise/sigma",
                shape=(1,),
                transform=LogTransform(),
                min=1e-6,
                group="noise",
            )
        )
        assert "noise/sigma" in space.names
        assert space.size == size_before + 1
        values["noise/sigma"] = tf.constant([0.25])
        phi = space.pack(values)
        theta = space.unpack(phi)
        np.testing.assert_allclose(theta["noise/sigma"].numpy(), [0.25])

    def test_add_entry_duplicate_raises(self, network):
        space = ParameterSpace.from_network(network)
        with pytest.raises(ValueError, match="duplicate"):
            space.add_entry(ParameterEntry(full_name="exc/w", shape=(2,)))

    def test_pack_none_requires_variable(self, network):
        space = ParameterSpace.from_network(network)
        space.add_entry(ParameterEntry(full_name="x", shape=(1,)))
        with pytest.raises(ValueError, match=r"pack\(None\)"):
            space.pack(None)

    def test_apply_to_model_skips_manual_entries(self, network):
        space = ParameterSpace.from_network(network)
        theta = space.unpack(space.pack())  # снимок до добавления записи
        space.add_entry(ParameterEntry(full_name="x", shape=(1,)))
        theta["x"] = tf.constant([1.0])  # есть в dict, переменной нет
        space.apply_to_model(theta)  # не падает


class TestFloat64:
    def test_float64_roundtrip(self):
        neuraltide.config.set_dtype(tf.float64)
        network = _build_network()
        space = ParameterSpace.from_network(network)
        phi = space.pack()
        assert phi.dtype == tf.float64
        theta = space.unpack(phi)
        for name, var in _variables(space).items():
            np.testing.assert_allclose(
                theta[name].numpy(), var.numpy(), rtol=1e-12, atol=1e-14
            )
