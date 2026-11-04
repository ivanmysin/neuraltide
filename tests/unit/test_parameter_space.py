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
    IdentityTransform,
    LogTransform,
    LogitTransform,
    ParameterEntry,
    ParameterSpace,
    ScaledSigmoidTransform,
    StateLocator,
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


class TestInitialStateEntries:
    """P1: записи начального состояния 'x0/<pop>/<slot>'."""

    def test_creates_entries_for_all_population_slots(self, network):
        space = ParameterSpace.from_network(network)
        registered = space.add_initial_state_entries(network)
        assert registered == ["x0/exc/0", "x0/inh/0"]
        assert space.names == EXPECTED_NAMES + ["x0/exc/0", "x0/inh/0"]
        entry = space.entries["x0/exc/0"]
        assert entry.kind == "initial_state"
        assert entry.variable is None
        assert entry.group == "initial_state"
        assert entry.trainable is True
        assert entry.shape == tf.TensorShape([1, 2])
        assert entry.numel == 2
        assert entry.locator == StateLocator(population="exc", slot=0)
        assert space.entries["x0/inh/0"].shape == tf.TensorShape([1, 1])

    def test_include_filters_by_name_and_slot_tuple(self, network):
        space = ParameterSpace.from_network(network)
        space.add_initial_state_entries(network, include={"x0/inh/0"})
        assert space.tensor_entry_names == ["x0/inh/0"]
        space2 = ParameterSpace.from_network(network)
        space2.add_initial_state_entries(network, include={("exc", 0)})
        assert space2.tensor_entry_names == ["x0/exc/0"]

    def test_include_partial_match_registers_matched(self, network):
        """Частично совпавший include-набор регистрирует совпавшие имена."""
        space = ParameterSpace.from_network(network)
        registered = space.add_initial_state_entries(
            network, include={"x0/inh/0", ("exc", 0)}
        )
        assert registered == ["x0/exc/0", "x0/inh/0"]

    def test_include_unknown_name_raises(self, network):
        """Полностью неизвестный include — честный отказ до регистрации."""
        space = ParameterSpace.from_network(network)
        with pytest.raises(ValueError, match="include"):
            space.add_initial_state_entries(network, include={"x0/nope/0"})
        # реестр не мутирован частичной регистрацией
        assert space.tensor_entry_names == []

    def test_include_mixed_known_and_unknown_raises(self, network):
        space = ParameterSpace.from_network(network)
        with pytest.raises(ValueError, match="x0/nope/0"):
            space.add_initial_state_entries(network, include={"x0/exc/0", "x0/nope/0"})
        assert space.tensor_entry_names == []

    def test_include_unknown_item_raises(self, network):
        space = ParameterSpace.from_network(network)
        with pytest.raises(ValueError, match="include"):
            space.add_initial_state_entries(network, include={42})

    def test_pack_unpack_roundtrip_with_x0(self, network):
        space = ParameterSpace.from_network(network)
        space.add_initial_state_entries(network)
        values = {
            name: (
                entry.variable.numpy()
                if entry.variable is not None
                else np.full(entry.shape, 0.3)
            )
            for name, entry in space.entries.items()
        }
        phi = space.pack(values)
        assert phi.shape == (space.size,)
        theta = space.unpack(phi)
        for name, entry in space.entries.items():
            ref = (
                entry.variable.numpy()
                if entry.variable is not None
                else np.full(entry.shape, 0.3)
            )
            np.testing.assert_allclose(theta[name].numpy(), ref, rtol=1e-6, atol=1e-8)

    def test_pack_none_with_x0_requires_values(self, network):
        space = ParameterSpace.from_network(network)
        space.add_initial_state_entries(network)
        with pytest.raises(ValueError, match=r"pack\(None\)"):
            space.pack(None)

    def test_apply_to_model_skips_x0_entries(self, network):
        space = ParameterSpace.from_network(network)
        space.add_initial_state_entries(network)
        values = {
            name: (
                entry.variable.numpy()
                if entry.variable is not None
                else np.full(entry.shape, 0.1)
            )
            for name, entry in space.entries.items()
        }
        values["exc/w"] = np.array([3.0, 4.0])
        theta = space.unpack(space.pack(values))
        space.apply_to_model(theta)  # x0-записи пропускаются без ошибок
        np.testing.assert_allclose(
            space.entries["exc/w"].variable.numpy(), [3.0, 4.0], rtol=1e-6
        )

    def test_transforms_and_bounds_applied(self, network):
        space = ParameterSpace.from_network(network)
        space.add_initial_state_entries(
            network,
            transforms={"x0/exc/0": LogTransform()},
            bounds={"x0/exc/0": (0.01, 10.0)},
        )
        entry = space.entries["x0/exc/0"]
        assert isinstance(entry.transform, LogTransform)
        assert entry.min == pytest.approx(0.01)
        assert entry.max == pytest.approx(10.0)
        values = {
            name: (
                entry.variable.numpy()
                if entry.variable is not None
                else np.full(entry.shape, 1.0)
            )
            for name, entry in space.entries.items()
        }
        theta = space.unpack(space.pack(values))
        # log-transform: phi = log(1.0) = 0 → forward(0) = 1.0
        np.testing.assert_allclose(theta["x0/exc/0"].numpy(), [[1.0, 1.0]], rtol=1e-6)

    def test_unknown_transforms_and_bounds_warn(self, network):
        space = ParameterSpace.from_network(network)
        with pytest.warns(UserWarning, match="unknown"):
            space.add_initial_state_entries(
                network, transforms={"x0/nope/0": LogTransform()}
            )
        space2 = ParameterSpace.from_network(network)
        with pytest.warns(UserWarning, match="unknown"):
            space2.add_initial_state_entries(network, bounds={"x0/nope/0": (0.0, 1.0)})

    def test_duplicate_registration_raises(self, network):
        space = ParameterSpace.from_network(network)
        space.add_initial_state_entries(network)
        with pytest.raises(ValueError, match="duplicate"):
            space.add_initial_state_entries(network)

    def test_name_helpers_partition_entries(self, network):
        space = ParameterSpace.from_network(network)
        space.add_initial_state_entries(network)
        assert space.model_entry_names == EXPECTED_NAMES
        assert space.tensor_entry_names == ["x0/exc/0", "x0/inh/0"]

    def test_p0_entries_default_to_model_var(self, network):
        space = ParameterSpace.from_network(network)
        for entry in space.entries.values():
            assert entry.kind == "model_var"
            assert entry.locator is None
        assert space.tensor_entry_names == []

    def test_entry_kind_validation(self):
        with pytest.raises(ValueError, match="unknown kind"):
            ParameterEntry(full_name="x", shape=(1,), kind="nope")
        var = tf.Variable([1.0])
        with pytest.raises(ValueError, match="variable=None"):
            ParameterEntry(
                full_name="x",
                shape=(1,),
                kind="initial_state",
                variable=var,
                locator=StateLocator("exc", 0),
            )
        with pytest.raises(ValueError, match="StateLocator"):
            ParameterEntry(full_name="x", shape=(1,), kind="shooting_state")


class TestExperimentInitialStateEntries:
    """Пункт 2 docs/next_steps.md: эксперимент-специфичные x0
    'x0/<experiment>/<population>/<slot>' рядом с общими 'x0/<pop>/<slot>'."""

    def test_creates_experiment_entries_with_locator(self, network):
        space = ParameterSpace.from_network(network)
        registered = space.add_experiment_initial_state_entries(network, "exp1")
        assert registered == ["x0/exp1/exc/0", "x0/exp1/inh/0"]
        assert space.names == EXPECTED_NAMES + ["x0/exp1/exc/0", "x0/exp1/inh/0"]
        entry = space.entries["x0/exp1/exc/0"]
        assert entry.kind == "initial_state"
        assert entry.group == "initial_state"
        assert entry.trainable is True
        assert entry.variable is None
        assert entry.shape == tf.TensorShape([1, 2])
        assert entry.numel == 2
        assert entry.locator == StateLocator(
            population="exc", slot=0, experiment="exp1"
        )
        assert space.entries["x0/exp1/inh/0"].shape == tf.TensorShape([1, 1])

    def test_common_locator_defaults_to_none_experiment(self, network):
        """Классификация common vs exp — по полю locator.experiment."""
        space = ParameterSpace.from_network(network)
        space.add_initial_state_entries(network)
        entry = space.entries["x0/exc/0"]
        assert entry.locator == StateLocator("exc", 0)
        assert entry.locator.experiment is None

    def test_experiment_name_with_slash_raises(self, network):
        space = ParameterSpace.from_network(network)
        with pytest.raises(ValueError, match="must not contain"):
            space.add_experiment_initial_state_entries(network, "exp/1")

    def test_empty_experiment_name_raises(self, network):
        space = ParameterSpace.from_network(network)
        with pytest.raises(ValueError, match="non-empty"):
            space.add_experiment_initial_state_entries(network, "")

    @pytest.mark.parametrize("bad", [None, 42, ("exp1",)])
    def test_non_string_experiment_name_raises(self, network, bad):
        """None/нестроковый experiment — понятный TypeError до '/' in."""
        space = ParameterSpace.from_network(network)
        with pytest.raises(TypeError, match="must be a string"):
            space.add_experiment_initial_state_entries(network, bad)

    def test_include_full_exp_names(self, network):
        space = ParameterSpace.from_network(network)
        registered = space.add_experiment_initial_state_entries(
            network, "exp1", include={"x0/exp1/inh/0"}
        )
        assert registered == ["x0/exp1/inh/0"]
        assert space.tensor_entry_names == ["x0/exp1/inh/0"]

    def test_include_population_slot_tuples(self, network):
        space = ParameterSpace.from_network(network)
        registered = space.add_experiment_initial_state_entries(
            network, "exp1", include={("exc", 0)}
        )
        assert registered == ["x0/exp1/exc/0"]

    def test_include_experiment_population_slot_tuples(self, network):
        space = ParameterSpace.from_network(network)
        registered = space.add_experiment_initial_state_entries(
            network, "exp1", include={("exp1", "inh", 0)}
        )
        assert registered == ["x0/exp1/inh/0"]

    def test_include_tuple_with_other_experiment_raises(self, network):
        space = ParameterSpace.from_network(network)
        with pytest.raises(ValueError, match="experiment"):
            space.add_experiment_initial_state_entries(
                network, "exp1", include={("exp2", "exc", 0)}
            )

    def test_include_unknown_item_raises(self, network):
        space = ParameterSpace.from_network(network)
        with pytest.raises(ValueError, match="include"):
            space.add_experiment_initial_state_entries(network, "exp1", include={42})

    def test_include_partial_match_registers_matched(self, network):
        """Частично совпавший include-набор регистрирует совпавшие имена."""
        space = ParameterSpace.from_network(network)
        registered = space.add_experiment_initial_state_entries(
            network, "exp1", include={("exc", 0), ("exp1", "inh", 0)}
        )
        assert registered == ["x0/exp1/exc/0", "x0/exp1/inh/0"]

    def test_include_unknown_exp_name_raises(self, network):
        """Полностью неизвестный include — честный отказ до регистрации."""
        space = ParameterSpace.from_network(network)
        with pytest.raises(ValueError, match="include"):
            space.add_experiment_initial_state_entries(
                network, "exp1", include={"x0/exp1/nope/0"}
            )
        assert space.tensor_entry_names == []

    def test_include_mixed_known_and_unknown_raises(self, network):
        space = ParameterSpace.from_network(network)
        with pytest.raises(ValueError, match="x0/exp1/nope/0"):
            space.add_experiment_initial_state_entries(
                network, "exp1", include={("exc", 0), "x0/exp1/nope/0"}
            )
        assert space.tensor_entry_names == []

    def test_common_and_experiment_entries_coexist(self, network):
        space = ParameterSpace.from_network(network)
        common = space.add_initial_state_entries(network)
        exp = space.add_experiment_initial_state_entries(network, "exp1")
        assert common == ["x0/exc/0", "x0/inh/0"]
        assert exp == ["x0/exp1/exc/0", "x0/exp1/inh/0"]
        assert space.names == EXPECTED_NAMES + common + exp
        common_entry = space.entries["x0/exc/0"]
        exp_entry = space.entries["x0/exp1/exc/0"]
        assert common_entry.locator.experiment is None
        assert exp_entry.locator.experiment == "exp1"
        assert common_entry.locator.population == exp_entry.locator.population
        assert common_entry.locator.slot == exp_entry.locator.slot

    def test_two_experiments_coexist(self, network):
        space = ParameterSpace.from_network(network)
        space.add_experiment_initial_state_entries(network, "exp1")
        space.add_experiment_initial_state_entries(network, "exp2")
        assert space.tensor_entry_names == [
            "x0/exp1/exc/0",
            "x0/exp1/inh/0",
            "x0/exp2/exc/0",
            "x0/exp2/inh/0",
        ]

    def test_duplicate_experiment_registration_raises(self, network):
        space = ParameterSpace.from_network(network)
        space.add_experiment_initial_state_entries(network, "exp1")
        with pytest.raises(ValueError, match="duplicate"):
            space.add_experiment_initial_state_entries(network, "exp1")

    def test_transforms_and_bounds_applied_by_exp_name(self, network):
        space = ParameterSpace.from_network(network)
        space.add_experiment_initial_state_entries(
            network,
            "exp1",
            transforms={"x0/exp1/exc/0": LogTransform()},
            bounds={"x0/exp1/exc/0": (0.01, 10.0)},
        )
        entry = space.entries["x0/exp1/exc/0"]
        assert isinstance(entry.transform, LogTransform)
        assert entry.min == pytest.approx(0.01)
        assert entry.max == pytest.approx(10.0)
        other = space.entries["x0/exp1/inh/0"]
        assert isinstance(other.transform, IdentityTransform)
        assert other.min is None and other.max is None

    def test_pack_unpack_roundtrip_with_experiment_x0(self, network):
        space = ParameterSpace.from_network(network)
        space.add_experiment_initial_state_entries(network, "exp1")
        values = {
            name: (
                entry.variable.numpy()
                if entry.variable is not None
                else np.full(entry.shape, 0.3)
            )
            for name, entry in space.entries.items()
        }
        phi = space.pack(values)
        assert phi.shape == (space.size,)
        theta = space.unpack(phi)
        np.testing.assert_allclose(
            theta["x0/exp1/exc/0"].numpy(), [[0.3, 0.3]], rtol=1e-6, atol=1e-8
        )
        np.testing.assert_allclose(
            theta["x0/exp1/inh/0"].numpy(), [[0.3]], rtol=1e-6, atol=1e-8
        )

    def test_unknown_transforms_and_bounds_warn(self, network):
        space = ParameterSpace.from_network(network)
        with pytest.warns(UserWarning, match="unknown"):
            space.add_experiment_initial_state_entries(
                network, "exp1", transforms={"x0/exp1/nope/0": LogTransform()}
            )
        space2 = ParameterSpace.from_network(network)
        with pytest.warns(UserWarning, match="unknown"):
            space2.add_experiment_initial_state_entries(
                network, "exp1", bounds={"x0/exp1/nope/0": (0.0, 1.0)}
            )

    def test_entries_are_trainable_and_counted_in_phi(self, network):
        """C8: exp-записи trainable по умолчанию и входят в φ."""
        space = ParameterSpace.from_network(network)
        base_size = space.size
        space.add_experiment_initial_state_entries(network, "exp1")
        assert "x0/exp1/exc/0" in space.trainable_names
        assert "x0/exp1/inh/0" in space.trainable_names
        assert list(space.phi_entries)[-2:] == ["x0/exp1/exc/0", "x0/exp1/inh/0"]
        assert space.size == base_size + 3  # exc [1, 2] + inh [1, 1]
        values = {
            name: np.full(space.shape(name), 0.4) for name in space.trainable_names
        }
        phi = space.pack(values)
        assert phi.shape == (space.size,)
        theta = space.unpack(phi)
        np.testing.assert_allclose(
            theta["x0/exp1/exc/0"].numpy(), [[0.4, 0.4]], atol=1e-6
        )


class TestTrainableExclusion:
    """C8: trainable=False превращает запись в реестровую константу вне φ."""

    @staticmethod
    def _space_with_const(network):
        space = ParameterSpace.from_network(network)
        const_var = tf.Variable([0.7], dtype=tf.float32)
        space.add_entry(
            ParameterEntry(
                full_name="noise/offset",
                shape=(1,),
                trainable=False,
                group="noise",
                variable=const_var,
            )
        )
        return space, const_var

    def test_non_trainable_excluded_from_size_pack_unpack(self, network):
        space, _ = self._space_with_const(network)
        size_before = space.size
        # в реестре запись остаётся (интроспекция)
        assert "noise/offset" in space.names
        assert "noise/offset" in space.entries
        assert space.size == size_before
        phi = space.pack()
        assert phi.shape == (size_before,)
        theta = space.unpack(phi)
        assert "noise/offset" not in theta
        assert set(theta.keys()) == set(space.trainable_names)

    def test_pack_values_ignores_non_trainable_and_unknown_keys(self, network):
        space, _ = self._space_with_const(network)
        # ключа 'noise/offset' нет — pack(values) работает
        values = {
            name: np.full(space.shape(name), 1.25) for name in space.trainable_names
        }
        phi = space.pack(values)
        # лишние ключи (non-trainable имя + неизвестное имя) молча игнорируются
        values["noise/offset"] = [9.9]
        values["unknown/param"] = [1.0]
        phi_extra = space.pack(values)
        np.testing.assert_allclose(phi_extra.numpy(), phi.numpy())

    def test_pack_none_reads_only_trainable_variables(self, network):
        space, const_var = self._space_with_const(network)
        const_var.assign([42.0])
        phi = space.pack()  # pack(None) не трогает non-trainable переменную
        space_ref = ParameterSpace.from_network(network)
        np.testing.assert_allclose(phi.numpy(), space_ref.pack().numpy())
        assert "noise/offset" not in space.unpack(phi)

    def test_apply_to_model_keeps_non_trainable_variable(self, network):
        space, const_var = self._space_with_const(network)
        theta = space.unpack(space.pack())
        theta["exc/w"] = np.array([3.0, 4.0], dtype=np.float32)
        before = const_var.numpy().copy()
        # ключа 'noise/offset' в theta_dict нет — apply_to_model не падает
        space.apply_to_model(theta)
        np.testing.assert_allclose(const_var.numpy(), before)
        np.testing.assert_allclose(space.entries["exc/w"].variable.numpy(), [3.0, 4.0])

    def test_project_ignores_non_trainable_bounds(self, network):
        space, _ = self._space_with_const(network)
        space.add_entry(
            ParameterEntry(
                full_name="noise/clipped",
                shape=(1,),
                trainable=False,
                min=0.0,
                max=1.0,
            )
        )
        phi = space.pack()
        phi_proj = space.project(phi + 50.0)  # за пределами max=1.0 константы
        assert phi_proj.shape == (space.size,)  # длина φ не выросла
        theta = space.unpack(phi_proj)
        assert "noise/clipped" not in theta

    def test_from_network_spaces_all_trainable(self, network):
        """Автопространства from_network: поведение прежнее (trainable-only)."""
        space = ParameterSpace.from_network(network)
        assert space.trainable_names == EXPECTED_NAMES
        assert list(space.phi_entries.keys()) == EXPECTED_NAMES
        assert set(space.phi_entries.keys()) == set(space.entries.keys())
        assert space.size == 2 + 2 + 1 + 2 + 2 + 2 + 2 + 2 + 2

    def test_trainable_names_and_phi_entries_order_and_identity(self, network):
        space = ParameterSpace.from_network(network)
        space.add_entry(
            ParameterEntry(full_name="noise/sigma", shape=(1,), group="noise")
        )  # trainable по умолчанию
        space.add_entry(
            ParameterEntry(
                full_name="noise/offset", shape=(1,), trainable=False, group="noise"
            )
        )
        assert space.names == EXPECTED_NAMES + ["noise/sigma", "noise/offset"]
        assert space.trainable_names == EXPECTED_NAMES + ["noise/sigma"]
        assert list(space.phi_entries.keys()) == EXPECTED_NAMES + ["noise/sigma"]
        assert "noise/offset" not in space.phi_entries
        # view возвращает те же объекты записей, что и entries
        assert space.phi_entries["exc/w"] is space.entries["exc/w"]

    def test_all_non_trainable_space_has_zero_size(self):
        var = tf.Variable([1.0], dtype=tf.float32)
        space = ParameterSpace()
        space.add_entry(
            ParameterEntry(
                full_name="const/a", shape=(1,), trainable=False, variable=var
            )
        )
        space.add_entry(
            ParameterEntry(full_name="const/b", shape=(2,), trainable=False)
        )
        assert space.size == 0
        assert space.trainable_names == []
        assert dict(space.phi_entries) == {}
        assert space.names == ["const/a", "const/b"]
        phi_none = space.pack()  # pack(None): trainable переменных нет
        phi_vals = space.pack({"const/a": [1.0], "const/b": [2.0, 3.0]})
        for phi in (phi_none, phi_vals):
            assert phi.shape == (0,)
            assert phi.dtype == tf.float32  # глобальный dtype
        assert space.unpack(phi_none) == {}
        assert space.project(phi_none).shape == (0,)

    def test_all_non_trainable_pack_uses_global_dtype(self):
        neuraltide.config.set_dtype(tf.float64)
        try:
            space = ParameterSpace()
            space.add_entry(
                ParameterEntry(full_name="const/a", shape=(1,), trainable=False)
            )
            phi = space.pack({"const/a": [1.0]})
            assert phi.dtype == tf.float64
            assert phi.shape == (0,)
        finally:
            neuraltide.config.set_dtype(tf.float32)

    def test_duplicate_non_trainable_name_raises(self, network):
        space = ParameterSpace.from_network(network)
        space.add_entry(ParameterEntry(full_name="const/x", shape=(1,)))
        space.add_entry(
            ParameterEntry(full_name="const/y", shape=(1,), trainable=False)
        )
        # duplicate non-trainable против non-trainable
        with pytest.raises(ValueError, match="duplicate"):
            space.add_entry(
                ParameterEntry(full_name="const/y", shape=(1,), trainable=False)
            )
        # duplicate non-trainable против trainable-имени из from_network
        with pytest.raises(ValueError, match="duplicate"):
            space.add_entry(
                ParameterEntry(full_name="exc/w", shape=(2,), trainable=False)
            )
