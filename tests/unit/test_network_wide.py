"""Контрактные тесты core-части network-wide интегрирования (этап В6).

Реальные интеграторы с step_network() реализуются на следующем этапе;
здесь используется фейковый duck-интегратор с supports_network_step=True,
записывающий вызовы RHS. Запись значений ведётся через прямые eager-вызовы
_step_fn(): внутри @tf.function (_scan_forward_states) Python-списки
заполнялись бы только в момент трейса, а не на каждой итерации цикла.
"""

import numpy as np
import pytest
import tensorflow as tf

import neuraltide.config
from neuraltide.core.network import (
    NetworkGraph,
    NetworkRNN,
    _JointNetworkRHS,
    _step_fn,
)
from neuraltide.integrators import AdaptiveRKIntegrator, EulerIntegrator
from neuraltide.integrators.base import BaseIntegrator
from neuraltide.inputs import ConstantRateGenerator
from neuraltide.populations import InputPopulation, WilsonCowan
from neuraltide.synapses import StaticSynapse

DT = 0.5

# Параметры малой сети (используются и в эталонных вычислениях).
WC_PARAMS = {
    "tau": 10.0,
    "a": 1.0,
    "theta": 0.0,
    "w": 0.2,
    "I_ext": 0.8,
    "max_rate": 100.0,
}
SYN_IN_PARAMS = {"gsyn_max": [[0.05]], "pconn": [[1.0]], "e_r": 5.0}
SYN_REC_PARAMS = {"gsyn_max": [[0.01]], "pconn": [[1.0]], "e_r": -10.0}


def _build_small_graph() -> NetworkGraph:
    """in(external) -> [StaticSynapse] -> exc(WilsonCowan) + рекуррент."""
    pop = WilsonCowan(dt=DT, params=WC_PARAMS, name="exc")
    syn_in = StaticSynapse(n_pre=1, n_post=1, dt=DT, params=SYN_IN_PARAMS)
    syn_rec = StaticSynapse(n_pre=1, n_post=1, dt=DT, params=SYN_REC_PARAMS)

    graph = NetworkGraph(dt=DT)
    graph.declare_input("in", n_units=1)
    graph.add_population("exc", pop)
    graph.add_synapse("in->exc", syn_in, src="in", tgt="exc")
    graph.add_synapse("exc->exc", syn_rec, src="exc", tgt="exc")
    return graph


def _euler_like_flat_update(states, derivs, blocks, dt):
    """
    Эйлеровское обновление плоского списка по блокам.

    Блок получает либо ровно n производных (n — число тензоров блока),
    либо ни одной (модели без динамики: InputPopulation, StaticSynapse) —
    тогда состояние блока не меняется.
    """
    new = []
    idx_s = 0
    idx_d = 0
    for n in blocks:
        if len(derivs) - idx_d >= n:
            for _ in range(n):
                new.append(states[idx_s] + dt * derivs[idx_d])
                idx_s += 1
                idx_d += 1
        else:
            for _ in range(n):
                new.append(states[idx_s])
                idx_s += 1
    return new


class _RecordingNetworkIntegrator(BaseIntegrator):
    """
    Фейковый интегратор с supports_network_step=True.

    step_network() вызывает RHS, записывает состояния/производные и делает
    один явный шаг Эйлера по плоским спискам. step()/step_synapse() бросают
    AssertionError — так тесты проверяют, что network-режим не обращается
    к split-API, и наоборот.
    """

    supports_network_step = True

    def __init__(self, graph: NetworkGraph):
        self._graph = graph
        self.calls = []
        self.step_network_invoked = False

    def step(self, *args, **kwargs):
        raise AssertionError("network-режим не должен вызывать step()")

    def step_synapse(self, *args, **kwargs):
        raise AssertionError("network-режим не должен вызывать step_synapse()")

    def step_network(self, rhs, pop_states, syn_states, dt):
        dpop, dsyn = rhs(pop_states, syn_states)
        self.step_network_invoked = True
        self.calls.append(
            {
                "pop_states": list(pop_states),
                "syn_states": list(syn_states),
                "dpop": list(dpop),
                "dsyn": list(dsyn),
            }
        )
        pop_blocks = [n for _, _, n in self._graph._pop_info_cache]
        syn_blocks = [n for _, _, n in self._graph._syn_info_cache]
        new_pop = _euler_like_flat_update(pop_states, dpop, pop_blocks, dt)
        new_syn = _euler_like_flat_update(syn_states, dsyn, syn_blocks, dt)
        return new_pop, new_syn


def _expected_dE(e, rate_in):
    """Эталон dE WilsonCowan для состояния e и входной частоты rate_in."""
    pre_rec = max(float(e), 0.0) * WC_PARAMS["max_rate"]
    i_in = SYN_IN_PARAMS["gsyn_max"][0][0] * (rate_in / 1000.0) * SYN_IN_PARAMS["e_r"]
    i_rec = (
        SYN_REC_PARAMS["gsyn_max"][0][0] * (pre_rec / 1000.0) * SYN_REC_PARAMS["e_r"]
    )
    i_syn = i_in + i_rec
    x = WC_PARAMS["w"] * float(e) + WC_PARAMS["I_ext"] + i_syn
    f = 1.0 / (1.0 + np.exp(-WC_PARAMS["a"] * (x - WC_PARAMS["theta"])))
    return (-float(e) + f) / WC_PARAMS["tau"]


class TestIntegrationModeValidation:
    def test_invalid_mode_rejected(self):
        graph = _build_small_graph()
        with pytest.raises(ValueError, match="integration_mode"):
            NetworkRNN(graph, EulerIntegrator(), integration_mode="joint")

    def test_network_mode_requires_integrator_support(self):
        graph = _build_small_graph()
        with pytest.raises(ValueError, match="supports_network_step") as exc_info:
            NetworkRNN(graph, AdaptiveRKIntegrator(), integration_mode="network")
        assert "AdaptiveRKIntegrator" in str(exc_info.value)

    def test_network_mode_with_capable_integrator_ok(self):
        graph = _build_small_graph()
        network = NetworkRNN(
            graph,
            _RecordingNetworkIntegrator(graph),
            integration_mode="network",
        )
        assert network.integration_mode == "network"


class TestIntegrationModeProperty:
    def test_default_is_split(self):
        graph = _build_small_graph()
        network = NetworkRNN(graph, EulerIntegrator())
        assert network.integration_mode == "split"

    def test_explicit_split(self):
        graph = _build_small_graph()
        network = NetworkRNN(graph, EulerIntegrator(), integration_mode="split")
        assert network.integration_mode == "split"

    def test_property_matches_constructor(self):
        graph = _build_small_graph()
        network = NetworkRNN(
            graph,
            _RecordingNetworkIntegrator(graph),
            integration_mode="network",
        )
        assert network.integration_mode == "network"


class TestNetworkModeRhs:
    """Eager-прогон _step_fn: RHS даёт ожидаемые производные, вход ZOH."""

    def test_rhs_derivatives_and_zoh(self):
        graph = _build_small_graph()
        integrator = _RecordingNetworkIntegrator(graph)
        network = NetworkRNN(graph, integrator, integration_mode="network")

        n_steps = 5
        # Различные значения на каждом шаге → проверяем zero-order hold.
        rates = np.array([[[10.0], [20.0], [30.0], [40.0], [50.0]]], dtype=np.float32)

        pop, syn = network.get_initial_state()
        stab = tf.zeros([1], dtype=neuraltide.config.get_dtype())
        e_prev = 0.0
        for k in range(n_steps):
            t = tf.constant([[k * DT]], dtype=neuraltide.config.get_dtype())
            input_rates = {"in": tf.constant(rates[:, k], dtype=tf.float32)}
            pop, syn, stab = _step_fn(
                (tuple(pop), tuple(syn), stab),
                t,
                input_rates,
                graph,
                integrator,
                integration_mode="network",
            )

            assert len(integrator.calls) == k + 1
            call = integrator.calls[k]

            # Согласованные плоские списки: одна популяция [1,1], без синапсов.
            assert len(call["pop_states"]) == 1
            assert call["pop_states"][0].shape == (1, 1)
            assert call["syn_states"] == []
            assert len(call["dpop"]) == 1
            assert call["dsyn"] == []

            e_k = float(call["pop_states"][0][0, 0])
            assert e_k == pytest.approx(e_prev, rel=1e-4)

            # ZOH: производные на шаге k зависят от входа именно шага k
            # (включая вклад рекуррентного синапса от e_k; post_v — нулевой
            # fallback, т.к. у WilsonCowan нет 'v_mean' в observables).
            expected = _expected_dE(e_k, float(rates[0, k, 0]))
            assert float(call["dpop"][0][0, 0]) == pytest.approx(expected, rel=1e-5)

            # Цепочка состояний как у Эйлера: e_{k+1} = e_k + dt * de_k.
            e_prev = e_k + DT * expected

        # Состояние за n_steps шагов совпадает с последним carry.
        assert float(pop[0][0, 0]) == pytest.approx(e_prev, rel=1e-4)
        # stability_error пробрасывается без изменений (нет оценки ошибки).
        assert float(stab[0]) == pytest.approx(0.0)

    def test_rhs_direct_call_external_vs_population_src(self):
        """_JointNetworkRHS напрямую: src external читает input_rates."""
        graph = _build_small_graph()
        rhs = _JointNetworkRHS(graph, {"in": tf.constant([[42.0]])})
        network = NetworkRNN(
            graph, _RecordingNetworkIntegrator(graph), integration_mode="network"
        )
        init_pop, init_syn = network.get_initial_state()

        dpop, dsyn = rhs(list(init_pop), list(init_syn))
        assert len(dpop) == 1 and len(dsyn) == 0

        # rate_in = 42 Гц при e = 0 → проверяем по эталону.
        assert float(dpop[0][0, 0]) == pytest.approx(_expected_dE(0.0, 42.0), rel=1e-5)


class TestNetworkModeEndToEnd:
    """Полный прогон через NetworkRNN.call (tf.function + tf.while_loop)."""

    def _run(self, graph, integrator, n_steps=4, dtype=tf.float32):
        network = NetworkRNN(graph, integrator, integration_mode="network")
        t_seq = tf.constant((np.arange(n_steps) * DT)[None, :, None], dtype=dtype)
        inputs = graph.pack_inputs({"in": tf.ones([1, n_steps, 1], dtype=dtype) * 25.0})
        return network(t_seq, inputs=inputs)

    def test_smoke_reaches_step_network(self):
        graph = _build_small_graph()
        integrator = _RecordingNetworkIntegrator(graph)
        out = self._run(graph, integrator)

        # step_network был вызван при трейсе (split-методы при этом
        # не вызывались — иначе бы бросили AssertionError).
        assert integrator.step_network_invoked
        assert out.firing_rates["exc"].shape == (1, 4, 1)
        assert np.all(np.isfinite(out.firing_rates["exc"].numpy()))

    @pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
    def test_dtype_smoke(self, dtype):
        neuraltide.config.set_dtype(dtype)
        graph = _build_small_graph()
        integrator = _RecordingNetworkIntegrator(graph)
        out = self._run(graph, integrator, n_steps=3, dtype=dtype)

        assert out.firing_rates["exc"].dtype == dtype
        assert out.firing_rates["exc"].shape == (1, 3, 1)
        assert out.final_state[0][0].dtype == dtype
        assert np.all(np.isfinite(out.firing_rates["exc"].numpy()))
        assert float(out.stability_loss) == pytest.approx(0.0)


class TestSplitModeUnchanged:
    def test_split_never_calls_step_network(self):
        graph = _build_small_graph()
        integrator = EulerIntegrator()

        def _boom(*args, **kwargs):
            raise AssertionError("step_network не должен вызываться в split")

        integrator.step_network = _boom  # type: ignore[attr-defined]
        network = NetworkRNN(graph, integrator)  # default 'split'

        n_steps = 4
        t_seq = tf.constant((np.arange(n_steps) * DT)[None, :, None], dtype=tf.float32)
        inputs = graph.pack_inputs({"in": tf.ones([1, n_steps, 1], dtype=tf.float32)})
        out = network(t_seq, inputs=inputs)

        assert network.integration_mode == "split"
        assert out.firing_rates["exc"].shape == (1, 4, 1)
        assert np.all(np.isfinite(out.firing_rates["exc"].numpy()))
        assert float(out.stability_loss) == pytest.approx(0.0)


class TestInputPopulationSmoke:
    """InputPopulation в network-режиме: derivatives пустые, состояние
    неизменно. Проверяется на прямом eager-вызове _step_fn (в полном
    scan firing_rates для InputPopulation не поддерживается — это
    существующее ограничение split-режима, здесь не меняется)."""

    def test_state_unchanged_and_empty_derivatives(self):
        gen = ConstantRateGenerator(dt=DT, params={"rate": 10.0})
        inp = InputPopulation(gen)
        pop = WilsonCowan(dt=DT, params=WC_PARAMS, name="exc")
        syn = StaticSynapse(n_pre=1, n_post=1, dt=DT, params=SYN_IN_PARAMS)

        graph = NetworkGraph(dt=DT)
        graph.add_population("ext", inp)
        graph.add_population("exc", pop)
        graph.add_synapse("ext->exc", syn, src="ext", tgt="exc")

        integrator = _RecordingNetworkIntegrator(graph)
        with pytest.warns(UserWarning, match="no incoming synapses"):
            network = NetworkRNN(graph, integrator, integration_mode="network")

        init_pop, init_syn = network.get_initial_state()
        assert len(init_pop) == 3  # ext: [t, extra] + exc: [E]

        t = tf.constant([[0.0]], dtype=neuraltide.config.get_dtype())
        new_pop, new_syn, new_stab = _step_fn(
            (
                tuple(init_pop),
                tuple(init_syn),
                tf.zeros([1], dtype=neuraltide.config.get_dtype()),
            ),
            t,
            None,  # внешних declare_input нет
            graph,
            integrator,
            integration_mode="network",
        )

        # Состояние InputPopulation неизменно, E популяции обновилось.
        assert len(new_pop) == 3
        for i in range(2):
            assert new_pop[i].shape == init_pop[i].shape
            if new_pop[i].shape.num_elements() > 0:
                diff = tf.reduce_max(tf.abs(new_pop[i] - init_pop[i]))
                assert float(diff) == 0.0
        assert float(new_pop[2][0, 0]) != 0.0

        # dpop содержит только блок WilsonCowan (ext дал 0 производных).
        call = integrator.calls[0]
        assert len(call["dpop"]) == 1
        assert call["dsyn"] == []
