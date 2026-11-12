"""
Adjoint state method for gradient computation.

This module provides an alternative to BPTT for computing gradients,
with reduced memory requirements for long sequences.

Memory strategy (O(1) Python objects w.r.t. T):
  - Forward pass: compiled tf.scan returns stacked [T, batch, ...] tensors
  - Backward pass: slices stacked tensors on-the-fly per step
  - No Python list-of-tuples for state trajectory; no pre-sliced target dicts

Two modes:
  - Discrete adjoint (default): uses tf.GradientTape per step (autodiff through integrator)
  - Analytical adjoint (use_analytical_adjoint=True): uses explicit
    adjoint_derivatives() + parameter_jacobian() — no GradientTape in the loop

Integration modes (network.integration_mode):
  - 'split' (default): отдельные шаги синапсов и популяций; оба adjoint-режима
    поддерживаются.
  - 'network': единый network-wide шаг интегратора (step_network). Discrete
    adjoint работает (дифференцирует тот же _step_fn, что и прямой прогон).
    Analytical adjoint НЕ поддерживается (NotImplementedError при создании
    solver'а). Ограничение stability-штрафа: step_network не аккумулирует
    per-step local error, поэтому градиенты StabilityPenalty в этом режиме
    тождественно нулевые (см. AdjointSolver._stability_gradients).

Loss convention: per_step_loss возвращает аддитивный вклад временного шага,
полная loss ≡ sum_t per_step_loss (нормировка 1/T внутри per_step_loss
потерь, а не в adjoint-циклах).
"""

import warnings
from typing import Dict, List, Tuple, Optional

import tensorflow as tf

import neuraltide
import neuraltide.config
from neuraltide.core.network import (
    NetworkRNN,
    NetworkOutput,
    _step_fn,
    unpack_state,
    get_firing_rates,
)
from neuraltide.core.types import TensorType, StateList
from neuraltide.integrators.base import BaseIntegrator
from neuraltide.populations.input_population import InputPopulation
from neuraltide.training.losses import BaseLoss, CompositeLoss


def _sanitise(name: str) -> str:
    return name.replace("/", "_").replace(":", "_").replace(".", "_")


def _get_named_vars(layer: tf.keras.layers.Layer) -> Dict[str, tf.Variable]:
    """Get trainable variables mapped by their short name from a Keras layer."""
    result = {}
    for v in layer.trainable_variables:
        raw = v.name
        if "/" in raw:
            raw = raw.rsplit("/", 1)[-1]
        if ":" in raw:
            raw = raw.rsplit(":", 1)[0]
        result[raw] = v
    return result


def _extract_masks(loss_fn: BaseLoss) -> Optional[Dict[str, TensorType]]:
    """Собирает маски термов потери в один dict {pop_name: mask_tensor}."""
    terms = loss_fn.terms if isinstance(loss_fn, CompositeLoss) else [(1.0, loss_fn)]
    masks: Dict[str, TensorType] = {}
    for _, sub in terms:
        m = getattr(sub, "mask", None)
        if m:
            masks.update(m)
    return masks or None


def _slice_mask_step(
    mask_dict: Optional[Dict[str, TensorType]], t: TensorType
) -> Optional[Dict[str, TensorType]]:
    """Срез масок на шаг t; маски без временной оси передаются как есть."""
    if mask_dict is None:
        return None
    out = {}
    for name, m in mask_dict.items():
        out[name] = m[:, t, :] if m.shape.rank == 3 else m
    return out


def _reduce_param_grad(grad: tf.Tensor, var: tf.Tensor) -> tf.Tensor:
    """Сворачивает батч-оси градиента параметра до ранга переменной.

    Состояния сети батчированы ([B, ...]), поэтому градиенты параметров из
    parameter_jacobian/compute_current_param_grad имеют ведущую батч-ось;
    переменная параметра общая (per-sample форма). При B=1 свёртка по оси
    размера 1 не меняет значений.
    """
    target_rank = var.shape.rank
    while (
        target_rank is not None
        and grad.shape.rank is not None
        and grad.shape.rank > target_rank
    ):
        grad = tf.reduce_sum(grad, axis=0)
    return grad


class AdjointSolver(tf.Module):
    """
    Solves adjoint state method for gradient computation.

    Inherits from tf.Module so that @tf.function methods correctly track
    trainable Variable reads (preventing them from being baked as constants).

    Args:
        network: NetworkRNN instance.
        integrator: integrator to use (defaults to network's integrator).
        use_analytical_adjoint: if True, use explicit adjoint_derivatives()
            and parameter_jacobian() instead of tf.GradientTape.

    Raises:
        NotImplementedError: if use_analytical_adjoint=True and the network
            uses integration_mode='network' (analytical adjoint is wired for
            split stepping only; use the discrete adjoint or split mode).

    The solver captures network.integration_mode (default 'split') and
    replays it in every _step_fn call of the backward loops, so discrete
    adjoint differentiates exactly the step used by the forward pass.
    """

    def __init__(
        self,
        network: NetworkRNN,
        integrator: Optional[BaseIntegrator] = None,
        use_analytical_adjoint: bool = False,
    ):
        super().__init__()
        self._network = network
        self._graph = network._graph
        self._integrator = integrator if integrator is not None else network._integrator
        self._use_analytical_adjoint = use_analytical_adjoint
        self._integration_mode = getattr(network, "integration_mode", "split")

        if use_analytical_adjoint and self._integration_mode == "network":
            raise NotImplementedError(
                "Analytical adjoint does not support integration_mode='network' "
                "(network-wide stepping via step_network); it is wired for "
                "split stepping only. Use the discrete adjoint "
                "(use_analytical_adjoint=False) or build the network with "
                "integration_mode='split'."
            )

        self._trainable = [v for v in network.trainable_variables if v.trainable]
        for v in self._trainable:
            setattr(self, "v_" + _sanitise(v.name), v)

        trainable_set = set(id(v) for v in self._trainable)
        self._trainable_to_grad_idx: Dict[int, int] = {}
        for i, v in enumerate(network.trainable_variables):
            if id(v) in trainable_set:
                self._trainable_to_grad_idx[id(v)] = len(self._trainable_to_grad_idx)

        if use_analytical_adjoint:
            self._build_analytical_index()

    def _build_analytical_index(self) -> None:
        """Precompute flat-list-to-component mappings for the analytical loop."""
        graph = self._graph

        self._ap_pop_names = []
        self._ap_pop_offsets = []
        self._ap_pop_sizes = []
        for name in graph.population_names:
            pop = graph._populations[name]
            offset = self._network._pop_state_offsets[name]
            size = len(pop.state_size)
            self._ap_pop_names.append(name)
            self._ap_pop_offsets.append(offset)
            self._ap_pop_sizes.append(size)

        self._as_syn_names = []
        self._as_syn_offsets = []
        self._as_syn_sizes = []
        self._as_syn_src_idx = []
        self._as_syn_tgt_idx = []
        for name in graph.synapse_names:
            entry = graph._synapses[name]
            offset = self._network._syn_state_offsets[name]
            size = len(entry.model.state_size)
            src_idx = (
                self._ap_pop_names.index(entry.src)
                if entry.src in graph._populations
                else -1
            )
            tgt_idx = self._ap_pop_names.index(entry.tgt)
            self._as_syn_names.append(name)
            self._as_syn_offsets.append(offset)
            self._as_syn_sizes.append(size)
            self._as_syn_src_idx.append(src_idx)
            self._as_syn_tgt_idx.append(tgt_idx)

        self._ap_n_dynamic = sum(
            1
            for name in graph.population_names
            if not isinstance(graph._populations[name], InputPopulation)
        )
        self._ap_dynamic_idx = [
            i
            for i, name in enumerate(self._ap_pop_names)
            if not isinstance(graph._populations[name], InputPopulation)
        ]

        self._aparam_info: List[Tuple[int, str, str, str]] = []
        for p_i, var in enumerate(self._trainable):
            found = False
            for pop_name in graph.population_names:
                pop = graph._populations[pop_name]
                for pname, pvar in _get_named_vars(pop).items():
                    if pvar is var:
                        self._aparam_info.append((p_i, "pop", pop_name, pname))
                        found = True
                        break
                if found:
                    break
            if not found:
                for syn_name in graph.synapse_names:
                    syn = graph._synapses[syn_name].model
                    for pname, pvar in _get_named_vars(syn).items():
                        if pvar is var:
                            self._aparam_info.append((p_i, "syn", syn_name, pname))
                            found = True
                            break
                    if found:
                        break
            if not found:
                self._aparam_info.append((p_i, "unknown", "", ""))

        self._aparam_rev: Dict[Tuple[str, str, str], int] = {}
        for p_i, ctype, cname, pname in self._aparam_info:
            if ctype != "unknown":
                self._aparam_rev[(ctype, cname, pname)] = p_i

    # ── Forward pass (compiled scan, O(1) Python objects) ────────────────────

    def forward_pass(
        self,
        t_sequence: TensorType,
        inputs: Optional[TensorType] = None,
        initial_state: Optional[Tuple[StateList, StateList]] = None,
    ) -> Tuple[
        NetworkOutput,
        Tuple[StateList, StateList],
        Tuple[StateList, StateList, List[TensorType], List[TensorType]],
    ]:
        if t_sequence.shape.rank == 2:
            t_sequence = t_sequence[:, :, tf.newaxis]

        if inputs is None:
            inputs = tf.zeros(
                [tf.shape(t_sequence)[0], tf.shape(t_sequence)[1], 0],
                dtype=neuraltide.config.get_dtype(),
            )

        batch_size = int(t_sequence.shape[0])

        if initial_state is None:
            init_pop, init_syn = self._network.get_initial_state(batch_size)
        else:
            init_pop, init_syn = initial_state

        (all_rates, stability_loss, final_pop, final_syn, pop_stacked, syn_stacked) = (
            self._network._scan_forward_states(
                t_sequence, tuple(init_pop), tuple(init_syn), inputs=inputs
            )
        )

        final_state = (list(final_pop), list(final_syn))

        output = NetworkOutput(
            firing_rates=all_rates,
            stability_loss=stability_loss,
            final_state=final_state,
        )
        return (
            output,
            final_state,
            (list(init_pop), list(init_syn), pop_stacked, syn_stacked),
        )

    def _build_state_full(
        self,
        init_pop: StateList,
        init_syn: StateList,
        pop_stacked: List[TensorType],
        syn_stacked: List[TensorType],
    ) -> Tuple[Tuple[TensorType, ...], Tuple[TensorType, ...]]:
        pop_full = tuple(
            tf.concat([[init_pop[i]], pop_stacked[i]], axis=0)
            for i in range(len(pop_stacked))
        )
        syn_full = tuple(
            tf.concat([[init_syn[i]], syn_stacked[i]], axis=0)
            for i in range(len(syn_stacked))
        )
        return pop_full, syn_full

    # ── Compiled forward-only step ───────────────────────────────────────────

    @tf.function
    def _forward_only(
        self,
        pop_tup: Tuple[TensorType, ...],
        syn_tup: Tuple[TensorType, ...],
        t_val: TensorType,
        input_rates_step: Dict[str, TensorType],
        target_dict: Dict[str, TensorType],
        mask_dict: Optional[Dict[str, TensorType]],
        loss_fn_obj: BaseLoss,
        dtype: tf.DType,
    ) -> Tuple[
        Tuple[TensorType, ...],
        Tuple[TensorType, ...],
        Dict[str, TensorType],
        TensorType,
    ]:
        # integration_mode захвачен из self при трейсе (Python-константа):
        # сигнатура tf.function не меняется, backward дифференцирует тот же
        # шаг, что и прямой прогон сети.
        new_pop, new_syn, _ = _step_fn(
            (pop_tup, syn_tup, tf.constant([0.0], dtype=dtype)),
            t_val,
            input_rates_step,
            self._graph,
            self._integrator,
            integration_mode=self._integration_mode,
        )
        npl = list(new_pop)
        nsl = list(new_syn)

        npd, _ = unpack_state(self._graph, npl, nsl)
        y_dict = get_firing_rates(self._graph, npd)

        # per_step_loss возвращает аддитивный вклад шага (полная loss ==
        # сумма вкладов); делить на T здесь не нужно — нормировка внутри
        # per_step_loss потерь.
        l_t1 = loss_fn_obj.per_step_loss(y_dict, target_dict, mask=mask_dict)

        return new_pop, new_syn, y_dict, l_t1

    # ── Synaptic context computation ─────────────────────────────────────────

    @tf.function
    def _compute_synaptic_context(
        self,
        pop_tup: Tuple[TensorType, ...],
        syn_tup: Tuple[TensorType, ...],
        input_rates_step: Dict[str, TensorType],
        dtype: tf.DType,
    ) -> Tuple[
        Dict[str, TensorType],
        Dict[str, TensorType],
        Dict[str, TensorType],
        Dict[str, TensorType],
    ]:
        """Compute I_syn, g_syn, pre_rates, post_vs from current state."""
        graph = self._graph
        pop_states_list = list(pop_tup)
        syn_states_list = list(syn_tup)

        pop_states_dict = {}
        idx = 0
        for name, pop, n in graph._pop_info_cache:
            pop_states_dict[name] = pop_states_list[idx : idx + n]
            idx += n

        syn_states_dict = {}
        idx = 0
        for name, entry, n in graph._syn_info_cache:
            syn_states_dict[name] = syn_states_list[idx : idx + n]
            idx += n

        pre_rates_dict: Dict[str, TensorType] = {}
        post_vs_dict: Dict[str, TensorType] = {}
        for name in graph.population_names:
            pop = graph._populations[name]
            state = pop_states_dict[name]
            pre_rates_dict[name] = pop.get_firing_rate(state)
            obs = pop.observables(state)
            post_vs_dict[name] = obs.get(
                "v_mean", tf.zeros([1, pop.n_units], dtype=dtype)
            )

        syn_I: Dict[str, TensorType] = {}
        syn_g: Dict[str, TensorType] = {}
        for name in graph.dynamic_population_names:
            n = graph._populations[name].n_units
            syn_I[name] = tf.zeros([1, n], dtype=dtype)
            syn_g[name] = tf.zeros([1, n], dtype=dtype)

        for syn_name, entry, _ in graph._syn_info_cache:
            syn_state = syn_states_dict[syn_name]
            # Get pre-synaptic firing rate
            if entry.src in graph._external_inputs:
                pre_rate = input_rates_step[entry.src]
            else:
                pre_rate = pre_rates_dict[entry.src]
            post_v = post_vs_dict[entry.tgt]
            current_dict = entry.model.compute_current(syn_state, pre_rate, post_v)
            syn_I[entry.tgt] = syn_I[entry.tgt] + current_dict["I_syn"]
            syn_g[entry.tgt] = syn_g[entry.tgt] + current_dict["g_syn"]

        return syn_I, syn_g, pre_rates_dict, post_vs_dict

    # ── Analytical backward loop ─────────────────────────────────────────────

    @tf.function
    def _analytical_backward_loop(
        self,
        pop_full: Tuple[TensorType, ...],
        syn_full: Tuple[TensorType, ...],
        t_sequence: TensorType,
        inputs_T: TensorType,
        target: Dict[str, TensorType],
        mask_dict: Optional[Dict[str, TensorType]],
        loss_fn_obj: BaseLoss,
        dtype: tf.DType,
    ) -> List[TensorType]:
        """
        Analytical adjoint backward pass inside a single tf.while_loop.

        Uses adjoint_derivatives() and parameter_jacobian() instead of
        tf.GradientTape for all state-to-state and state-to-param gradients.
        Only uses GradientTape for the scalar loss gradient (O(1)).
        """
        graph = self._graph
        dt_val = graph.dt

        lam_pop = tuple(tf.zeros_like(s[0]) for s in pop_full)
        lam_syn = tuple(tf.zeros_like(s[0]) for s in syn_full)
        dtheta_acc = tuple(tf.zeros_like(v) for v in self._trainable)

        T = int(t_sequence.shape[1])

        n_ap = len(self._ap_pop_names)
        n_as = len(self._as_syn_names)

        def cond(counter, lp, ls, da):
            return counter >= 0

        def body(counter, lp, ls, da):
            t = counter

            pop_t = tuple(s[t] for s in pop_full)
            syn_t = tuple(s[t] for s in syn_full)
            t_val = tf.squeeze(t_sequence[:, t, :], axis=1)

            # Slice inputs at this time step
            input_step = {}
            for name in graph._input_names:
                offset = graph._input_offsets[name]
                size = graph._external_inputs[name]
                input_step[name] = inputs_T[t, :, offset : offset + size]

            target_t = {name: t_tensor[:, t, :] for name, t_tensor in target.items()}
            mask_t = _slice_mask_step(mask_dict, t)

            syn_I, syn_g, pre_rates, post_vs = self._compute_synaptic_context(
                pop_t, syn_t, input_step, dtype
            )

            new_pop, new_syn, y_dict, l_t1 = self._forward_only(
                pop_t,
                syn_t,
                t_val,
                input_step,
                target_t,
                mask_t,
                loss_fn_obj,
                dtype,
            )

            new_lp = list(lp)
            new_ls = list(ls)
            new_da = list(da)

            cur_param_updates: List[Tuple[int, TensorType]] = []

            # Internal dynamics adjoint: populations
            for pi in range(n_ap):
                off = self._ap_pop_offsets[pi]
                sz = self._ap_pop_sizes[pi]
                name = self._ap_pop_names[pi]
                pop_model = graph._populations[name]

                ps = pop_t[off : off + sz]
                pl = list(new_lp[off : off + sz])

                total_syn = {
                    "I_syn": syn_I.get(name, tf.zeros_like(ps[0])),
                    "g_syn": syn_g.get(name, tf.zeros_like(ps[0])),
                }

                try:
                    dlam = pop_model.adjoint_derivatives(pl, ps, total_syn)
                    for k in range(sz):
                        new_lp[off + k] = new_lp[off + k] - dt_val * dlam[k]
                except NotImplementedError:
                    pass

            # Internal dynamics adjoint: synapses
            for si in range(n_as):
                off = self._as_syn_offsets[si]
                sz = self._as_syn_sizes[si]
                name = self._as_syn_names[si]
                entry = graph._synapses[name]

                ss = syn_t[off : off + sz]
                sl = list(new_ls[off : off + sz])

                try:
                    dlam = entry.model.adjoint_derivatives(
                        sl, ss, pre_rates[entry.src], post_vs[entry.tgt]
                    )
                    for k in range(sz):
                        new_ls[off + k] = new_ls[off + k] - dt_val * dlam[k]
                except NotImplementedError:
                    pass

            # Coupling: pop adjoint → I/g → synapse adjoint
            for si in range(n_as):
                off = self._as_syn_offsets[si]
                sz = self._as_syn_sizes[si]
                name = self._as_syn_names[si]
                entry = graph._synapses[name]
                tgt_name = entry.tgt
                tgt_pi = self._as_syn_tgt_idx[si]

                pop_model = graph._populations[tgt_name]
                syn_model = entry.model

                ps_off = self._ap_pop_offsets[tgt_pi]
                ps_sz = self._ap_pop_sizes[tgt_pi]
                ps = pop_t[ps_off : ps_off + ps_sz]
                pl = list(lp[ps_off : ps_off + ps_sz])

                total_syn = {
                    "I_syn": syn_I.get(tgt_name, tf.zeros_like(ps[0])),
                    "g_syn": syn_g.get(tgt_name, tf.zeros_like(ps[0])),
                }

                try:
                    lam_I, lam_g = pop_model.synaptic_coupling(pl, ps, total_syn)
                except NotImplementedError:
                    lam_I = tf.zeros_like(ps[0])
                    lam_g = tf.zeros_like(ps[0])

                try:
                    ss = syn_t[off : off + sz]
                    coupling = syn_model.compute_current_state_vjp(
                        lam_I, lam_g, ss, pre_rates[entry.src], post_vs[entry.tgt]
                    )
                    for k in range(len(coupling)):
                        new_ls[off + k] = new_ls[off + k] + dt_val * coupling[k]
                except NotImplementedError:
                    pass

                ss = syn_t[off : off + sz]
                try:
                    cur_param_grads = syn_model.compute_current_param_grad(
                        lam_I, lam_g, ss, pre_rates[entry.src], post_vs[entry.tgt]
                    )
                except NotImplementedError:
                    cur_param_grads = {}
                for pname, pgrad in cur_param_grads.items():
                    key = ("syn", name, pname)
                    p_idx_c = self._aparam_rev.get(key)
                    if p_idx_c is not None:
                        pgrad = _reduce_param_grad(pgrad, self._trainable[p_idx_c])
                        cur_param_updates.append((p_idx_c, dt_val * pgrad))

            # Loss coupling: ∂L/∂rate → ∂rate/∂pop_state
            if len(y_dict) > 0:
                y_list = [
                    y_dict[name]
                    for name in graph.dynamic_population_names
                    if name in y_dict
                ]
                if len(y_list) > 0:
                    with tf.GradientTape() as lt:
                        lt.watch(y_list)
                        l_check = loss_fn_obj.per_step_loss(
                            {
                                name: y_list[i]
                                for i, name in enumerate(graph.dynamic_population_names)
                                if name in y_dict
                            },
                            target_t,
                            mask=mask_t,
                        )
                    dl_dy = lt.gradient(l_check, y_list, unconnected_gradients="zero")

                    dy_idx = 0
                    for pi in self._ap_dynamic_idx:
                        name = self._ap_pop_names[pi]
                        if name not in y_dict:
                            continue
                        dl = dl_dy[dy_idx]
                        dy_idx += 1
                        if dl is None:
                            continue

                        pop_model = graph._populations[name]
                        off = self._ap_pop_offsets[pi]
                        sz = self._ap_pop_sizes[pi]

                        scaling = tf.cast(1.0 / (pop_model.dt * 1e-3), dtype)
                        new_lp[off] = new_lp[off] + dt_val * dl * tf.cast(
                            scaling, dl.dtype
                        )

            # Parameter gradients
            new_da = list(da)
            for p_i, comp_type, comp_name, param_name in self._aparam_info:
                if comp_type == "pop":
                    if param_name == "":
                        continue
                    pop_model = graph._populations[comp_name]
                    pi_idx = self._ap_pop_names.index(comp_name)
                    off = self._ap_pop_offsets[pi_idx]
                    sz = self._ap_pop_sizes[pi_idx]
                    ps = pop_t[off : off + sz]
                    pl = list(new_lp[off : off + sz])
                    total_syn = {
                        "I_syn": syn_I.get(comp_name, tf.zeros_like(ps[0])),
                        "g_syn": syn_g.get(comp_name, tf.zeros_like(ps[0])),
                    }
                    try:
                        jac = pop_model.parameter_jacobian(param_name, ps, total_syn)
                        grad = tf.zeros(tf.shape(jac)[1:], dtype=jac.dtype)
                        for k, lam in enumerate(pl):
                            grad = grad + tf.reduce_sum(lam * jac, axis=0)
                        new_da[p_i] = new_da[p_i] + dt_val * grad
                    except NotImplementedError:
                        pass

                elif comp_type == "syn":
                    if param_name == "":
                        continue
                    entry = graph._synapses[comp_name]
                    syn_model = entry.model
                    si_idx = self._as_syn_names.index(comp_name)
                    off = self._as_syn_offsets[si_idx]
                    sz = self._as_syn_sizes[si_idx]
                    ss = syn_t[off : off + sz]
                    sl = list(new_ls[off : off + sz])
                    try:
                        jac = syn_model.parameter_jacobian(
                            param_name, ss, pre_rates[entry.src], post_vs[entry.tgt]
                        )
                        grad = tf.zeros_like(jac)
                        for k, lam in enumerate(sl):
                            grad = grad + lam * jac
                        grad = _reduce_param_grad(grad, new_da[p_i])
                        new_da[p_i] = new_da[p_i] + dt_val * grad
                    except NotImplementedError:
                        pass

                elif comp_type == "unknown":
                    pop_list = list(pop_t)
                    syn_list = list(syn_t)
                    with tf.GradientTape() as pt:
                        pt.watch(pop_list)
                        pt.watch(syn_list)
                        new_pop2, new_syn2, y_dict2, l_t2 = self._forward_only(
                            pop_t,
                            syn_t,
                            t_val,
                            input_step,
                            target_t,
                            mask_t,
                            loss_fn_obj,
                            dtype,
                        )
                        proxy2 = sum(
                            tf.reduce_sum(p * lm) for p, lm in zip(new_pop2, new_lp)
                        ) + sum(
                            tf.reduce_sum(s * lm) for s, lm in zip(new_syn2, new_ls)
                        )
                        combined2 = proxy2 + l_t2
                    all_grads = pt.gradient(
                        combined2, self._trainable, unconnected_gradients="zero"
                    )
                    if all_grads[p_i] is not None:
                        new_da[p_i] = new_da[p_i] + all_grads[p_i]

            for p_idx_u, pgrad_u in cur_param_updates:
                new_da[p_idx_u] = new_da[p_idx_u] + pgrad_u

            return counter - 1, tuple(new_lp), tuple(new_ls), tuple(new_da)

        _, _, _, dtheta_acc = tf.while_loop(
            cond,
            body,
            (tf.constant(T - 1, dtype=tf.int32), lam_pop, lam_syn, dtheta_acc),
            parallel_iterations=1,
        )

        return list(dtheta_acc)

    # ── Compiled backward loop (discrete adjoint) ────────────────────────────

    @tf.function
    def _compiled_backward_loop(
        self,
        pop_full: Tuple[TensorType, ...],
        syn_full: Tuple[TensorType, ...],
        t_sequence: TensorType,
        inputs_T: TensorType,
        target: Dict[str, TensorType],
        mask_dict: Optional[Dict[str, TensorType]],
        loss_fn_obj: BaseLoss,
        dtype: tf.DType,
    ) -> List[TensorType]:
        n_pop = len(pop_full)
        n_syn = len(syn_full)

        lam_pop = tuple(tf.zeros_like(s[0]) for s in pop_full)
        lam_syn = tuple(tf.zeros_like(s[0]) for s in syn_full)
        dtheta_acc = tuple(tf.zeros_like(v) for v in self._trainable)

        T = int(t_sequence.shape[1])

        def cond(counter, lp, ls, da):
            return counter >= 0

        def body(counter, lp, ls, da):
            t = counter

            pop_t = tuple(s[t] for s in pop_full)
            syn_t = tuple(s[t] for s in syn_full)
            t_val = tf.squeeze(t_sequence[:, t, :], axis=1)

            # Slice inputs at this time step
            input_step = {}
            for name in self._graph._input_names:
                offset = self._graph._input_offsets[name]
                size = self._graph._external_inputs[name]
                input_step[name] = inputs_T[t, :, offset : offset + size]

            target_t = {name: t_tensor[:, t, :] for name, t_tensor in target.items()}
            mask_t = _slice_mask_step(mask_dict, t)

            pop_list = list(pop_t)
            syn_list = list(syn_t)
            all_src = pop_list + syn_list + self._trainable

            with tf.GradientTape() as tape:
                tape.watch(pop_list)
                tape.watch(syn_list)

                new_pop, new_syn, y_dict, l_t1 = self._forward_only(
                    pop_t,
                    syn_t,
                    t_val,
                    input_step,
                    target_t,
                    mask_t,
                    loss_fn_obj,
                    dtype,
                )

                npl = list(new_pop)
                nsl = list(new_syn)
                lpl = list(lp)
                lsl = list(ls)

                proxy = sum(tf.reduce_sum(p * lm) for p, lm in zip(npl, lpl)) + sum(
                    tf.reduce_sum(s * lm) for s, lm in zip(nsl, lsl)
                )
                combined = proxy + l_t1

            grads = tape.gradient(combined, all_src, unconnected_gradients="zero")

            new_lp = tuple(grads[:n_pop])
            new_ls = tuple(grads[n_pop : n_pop + n_syn])
            ds = grads[n_pop + n_syn :]

            new_da = tuple(a + g for a, g in zip(da, ds))

            return counter - 1, new_lp, new_ls, new_da

        _, _, _, dtheta_acc = tf.while_loop(
            cond,
            body,
            (tf.constant(T - 1, dtype=tf.int32), lam_pop, lam_syn, dtheta_acc),
            parallel_iterations=1,
        )

        return list(dtheta_acc)

    # ── Backward pass ────────────────────────────────────────────────────────

    def backward_pass(
        self,
        t_sequence: TensorType,
        inputs: TensorType,
        state_info: Tuple[StateList, StateList, List[TensorType], List[TensorType]],
        target: Dict[str, TensorType],
        loss_fn: BaseLoss,
    ) -> List[TensorType]:
        if t_sequence.shape.rank == 2:
            t_sequence = t_sequence[:, :, tf.newaxis]

        init_pop, init_syn, pop_stacked, syn_stacked = state_info
        dtype = neuraltide.config.get_dtype()
        variables = self._network.trainable_variables

        pop_full, syn_full = self._build_state_full(
            init_pop, init_syn, pop_stacked, syn_stacked
        )

        # Pre-transpose inputs: [batch, T, total] -> [T, batch, total]
        inputs_T = tf.transpose(inputs, [1, 0, 2])

        # Маски термов потери (если заданы) — adjoint использует точные
        # срезы на шаг t, чтобы оптимизировать ту же функцию, что и BPTT.
        mask_dict = _extract_masks(loss_fn)

        if self._use_analytical_adjoint:
            dtheta_step_list = self._analytical_backward_loop(
                pop_full,
                syn_full,
                t_sequence,
                inputs_T,
                target,
                mask_dict,
                loss_fn,
                dtype,
            )
        else:
            dtheta_step_list = self._compiled_backward_loop(
                pop_full,
                syn_full,
                t_sequence,
                inputs_T,
                target,
                mask_dict,
                loss_fn,
                dtype,
            )

        grad_list: List[TensorType] = list(dtheta_step_list)
        return [
            grad_list[self._trainable_to_grad_idx[id(v)]]
            if id(v) in self._trainable_to_grad_idx
            else tf.zeros_like(v)
            for v in variables
        ]

    # ── Stability gradients ──────────────────────────────────────────────────

    @tf.function
    def _stab_forward_step(
        self,
        pop_tup: Tuple[TensorType, ...],
        syn_tup: Tuple[TensorType, ...],
        stab_acc: TensorType,
        t_val: TensorType,
        dtype: tf.DType,
    ) -> Tuple[Tuple[TensorType, ...], Tuple[TensorType, ...], TensorType]:
        """Один шаг короткого прогона для градиентов stability-штрафа.

        Идёт через тот же integration_mode, что и основной adjoint. В режиме
        'network' _step_fn возвращает stab_acc неизменным (step_network не
        аккумулирует per-step local error), поэтому stability_loss такого
        прогона тождественно ноль и градиенты вырождаются в нули — ограничение
        документировано в _stability_gradients.
        """
        # Provide zero input rates for all external inputs
        input_step = {}
        for name in self._graph._input_names:
            n = self._graph._external_inputs[name]
            input_step[name] = tf.zeros([1, n], dtype=dtype)
        return _step_fn(
            (pop_tup, syn_tup, stab_acc),
            t_val,
            input_step,
            self._graph,
            self._integrator,
            integration_mode=self._integration_mode,
        )

    # ── Compute gradients ────────────────────────────────────────────────────

    def compute_gradients(
        self,
        t_sequence: TensorType,
        inputs: TensorType,
        target: Dict[str, TensorType],
        loss_fn: BaseLoss,
        initial_state: Optional[Tuple[StateList, StateList]] = None,
    ) -> Tuple[List[TensorType], List[tf.Variable], NetworkOutput]:
        """
        Вычисляет градиенты потери через adjoint-метод.

        Args:
            t_sequence: временная последовательность [batch, T, 1].
            inputs: упакованные внешние входы [batch, T, total_input_units].
            target: целевые траектории {name: [batch, T, n]}.
            loss_fn: потеря с per_step_loss (аддитивная конвенция:
                полная loss == sum_t per_step_loss).
            initial_state: опциональное начальное состояние
                (pop_states, syn_states). Передаётся в forward_pass.

        Returns:
            (grads, variables, output): градиенты по trainable_variables
            сети (до клиппинга), сами переменные и прямой прогон.

        Note:
            Градиенты StabilityPenalty вычисляются отдельным коротким
            прогоном и всегда стартуют из состояния по умолчанию сети
            (self._network.get_initial_state) — initial_state на них не
            влияет. Шаг этого прогона идёт через integration_mode сети;
            в режиме 'network' step_network не аккумулирует per-step local
            error, поэтому градиенты stability-штрафа тождественно нулевые
            (с предупреждением UserWarning, см. _stability_gradients).
        """
        output, _, state_info = self.forward_pass(
            t_sequence, inputs, initial_state=initial_state
        )

        variables = self._network.trainable_variables

        if isinstance(loss_fn, CompositeLoss):
            main_loss_obj, stab_terms = self._split_composite_loss(loss_fn)
        else:
            main_loss_obj = loss_fn
            stab_terms = []

        if main_loss_obj is not None:
            main_grads = self.backward_pass(
                t_sequence, inputs, state_info, target, main_loss_obj
            )
        else:
            main_grads = [tf.zeros_like(v) for v in variables]

        if stab_terms:
            total_stab_weight = sum(w for w, _ in stab_terms)
            stab_grads = self._stability_gradients()
            total_grads = [
                mg + sg * total_stab_weight for mg, sg in zip(main_grads, stab_grads)
            ]
        else:
            total_grads = main_grads

        return total_grads, variables, output

    @staticmethod
    def _split_composite_loss(
        loss_fn: CompositeLoss,
    ) -> Tuple[Optional[BaseLoss], List[Tuple[float, BaseLoss]]]:
        from neuraltide.training.losses import StabilityPenalty

        main_terms = [
            (w, loss)
            for w, loss in loss_fn.terms
            if not isinstance(loss, StabilityPenalty)
        ]
        stab_terms = [
            (w, loss) for w, loss in loss_fn.terms if isinstance(loss, StabilityPenalty)
        ]

        if not main_terms:
            return None, stab_terms

        if len(main_terms) == 1 and main_terms[0][0] == 1.0:
            return main_terms[0][1], stab_terms

        return CompositeLoss(main_terms), stab_terms

    def _stability_gradients(self) -> List[TensorType]:
        """
        Градиенты StabilityPenalty отдельным коротким прогоном.

        Прямой прогон идёт через integration_mode сети (тот же режим, что и
        в основном adjoint). Ограничение режима 'network': step_network не
        возвращает per-step local error, _step_fn пробрасывает stab_acc
        неизменным, поэтому stability_loss прогона тождественно ноль и все
        градиенты вырождаются в нули (штраф не даёт градиентного сигнала).
        Это честное поведение — фиктивный градиент не выдумывается; выдаётся
        UserWarning. В режиме 'split' поведение прежнее.
        """
        variables = self._network.trainable_variables
        dtype = neuraltide.config.get_dtype()

        if self._integration_mode == "network":
            warnings.warn(
                "integration_mode='network': step_network does not accumulate "
                "per-step local errors, so stability gradients are identically "
                "zero and the stability penalty contributes no gradient signal. "
                "Use integration_mode='split' for stability-penalty gradients.",
                UserWarning,
                stacklevel=2,
            )

        batch_size = 1
        init_pop, init_syn = self._network.get_initial_state(batch_size)
        pop_states = tuple(init_pop)
        syn_states = tuple(init_syn)
        stability_acc = tf.zeros([1], dtype=dtype)

        t_val = tf.constant(
            self._graph.dt,
            dtype=dtype,
        )

        with tf.GradientTape() as tape:
            for _ in range(10):
                pop_states, syn_states, stability_acc = self._stab_forward_step(
                    pop_states, syn_states, stability_acc, t_val, dtype
                )
            stability_loss = tf.reduce_mean(stability_acc)

        grads = tape.gradient(stability_loss, variables)

        return [
            g if g is not None else tf.zeros_like(v) for g, v in zip(grads, variables)
        ]
