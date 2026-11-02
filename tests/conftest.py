import numpy as np
import pytest
import tensorflow as tf
import neuraltide
from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.integrators import EulerIntegrator
from neuraltide.inputs import ConstantRateGenerator
from neuraltide.optimization import (
    CalibrationProblem,
    ObservationSpec,
    ParameterSpace,
    ResidualLoss,
)
from neuraltide.populations import WilsonCowan
from neuraltide.synapses import StaticSynapse

L0_DT = 0.5
L0_T = 60


@pytest.fixture(autouse=True)
def reset_dtype():
    neuraltide.config.set_dtype(tf.float32)
    yield


@pytest.fixture
def dt():
    return 0.5


@pytest.fixture
def n_steps():
    return 100


@pytest.fixture
def small_izh_params():
    """Минимальные параметры IzhikevichMeanField для n_units=2."""
    return {
        "alpha": {"value": [0.5, 0.5], "trainable": False},
        "a": {"value": [0.02, 0.02], "trainable": False},
        "b": {"value": [0.2, 0.2], "trainable": False},
        "w_jump": {"value": [0.1, 0.1], "trainable": False},
        "dt_nondim": {"value": [0.01, 0.01], "trainable": False},
        "Delta_eta": {"value": [0.5, 0.6], "trainable": True, "min": 0.01, "max": 2.0},
        "I_ext": {"value": [1.0, 1.2], "trainable": True},
    }


def _build_l0():
    """
    L0-сеть: WilsonCowan (1 юнит) + StaticSynapse от ConstantRate.

    Обучаемые параметры: exc/tau (min/max), exc/I_ext, in->exc/gsyn_max
    (min/max). Целевая траектория — прогон при параметрах конструктора.
    Возвращает (network, space, t_seq, inputs, loss); сеть строится в
    текущем глобальном dtype (см. autouse reset_dtype).
    """
    pop = WilsonCowan(
        dt=L0_DT,
        params={
            "tau": {"value": [8.0], "trainable": True, "min": 1.0, "max": 50.0},
            "a": 1.0,
            "theta": 0.0,
            "w": 0.2,
            "I_ext": {"value": [0.8], "trainable": True},
            "max_rate": 100.0,
        },
        name="exc",
    )
    syn = StaticSynapse(
        n_pre=1,
        n_post=1,
        dt=L0_DT,
        params={
            "gsyn_max": {
                "value": [[0.05]],
                "trainable": True,
                "min": 0.001,
                "max": 1.0,
            },
            "pconn": [[1.0]],
            "e_r": 0.0,
        },
        name="in->exc",
    )
    gen = ConstantRateGenerator(dt=L0_DT, params={"rate": 10.0})

    graph = NetworkGraph(dt=L0_DT)
    graph.declare_input("in", n_units=1)
    graph.add_population("exc", pop)
    graph.add_synapse("in->exc", syn, src="in", tgt="exc")
    network = NetworkRNN(graph, EulerIntegrator())

    t_seq = tf.constant((np.arange(L0_T, dtype=np.float32) * L0_DT)[None, :, None])
    inputs = graph.pack_inputs({"in": gen(t_seq)})

    target = network(t_seq, inputs=inputs).firing_rates["exc"]
    loss = ResidualLoss(
        [ObservationSpec(name="obs", population="exc", target=target, sigma=2.0)]
    )

    space = ParameterSpace.from_network(network)
    return network, space, t_seq, inputs, loss


@pytest.fixture
def l0_factory():
    """
    Фабрика L0-сети: возвращает callable -> (network, space, t_seq, inputs, loss).

    Позволяет тесту сначала установить dtype (например, float64), затем
    построить сеть уже в нём.
    """
    return _build_l0


@pytest.fixture
def l0_problem() -> CalibrationProblem:
    """Готовая L0 CalibrationProblem в текущем dtype (по умолчанию float32)."""
    network, space, t_seq, inputs, loss = _build_l0()
    return CalibrationProblem(network, space, t_seq, inputs=inputs, objective=loss)
