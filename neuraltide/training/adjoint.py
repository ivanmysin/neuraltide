"""
Adjoint state method for gradient computation.

This module provides an alternative to BPTT for computing gradients,
with reduced memory requirements for long sequences.

Memory strategy (O(1) Python objects w.r.t. T):
  - Forward pass: compiled tf.scan returns stacked [T, batch, ...] tensors
  - Backward pass: slices stacked tensors on-the-fly per step
  - No Python list-of-tuples for state trajectory; no pre-sliced target dicts
"""
import tensorflow as tf
from typing import Dict, List, Tuple, Optional

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
from neuraltide.training.losses import BaseLoss, CompositeLoss


def _sanitise(name: str) -> str:
    return name.replace("/", "_").replace(":", "_").replace(".", "_")


class AdjointSolver(tf.Module):
    """
    Solves adjoint state method for gradient computation.

    Inherits from tf.Module so that @tf.function methods correctly track
    trainable Variable reads (preventing them from being baked as constants).
    """

    def __init__(
        self,
        network: NetworkRNN,
        integrator: Optional[BaseIntegrator] = None,
    ):
        super().__init__()
        self._network = network
        self._graph = network._graph
        self._integrator = integrator if integrator is not None else network._integrator

        self._trainable = [v for v in network.trainable_variables if v.trainable]
        for v in self._trainable:
            setattr(self, "v_" + _sanitise(v.name), v)

    # ── Forward pass (compiled scan, O(1) Python objects) ────────────────────

    def forward_pass(
        self,
        t_sequence: TensorType,
        initial_state: Optional[Tuple[StateList, StateList]] = None,
    ) -> Tuple[
        NetworkOutput,
        Tuple[StateList, StateList],
        Tuple[StateList, StateList, List[TensorType], List[TensorType]],
    ]:
        """
        Run forward pass using compiled tf.scan.

        Returns stacked state tensors (compact [T, batch, ...] layout)
        instead of Python list-of-tuples — O(1) Python overhead w.r.t. T.
        """
        if t_sequence.shape.rank == 2:
            t_sequence = t_sequence[:, :, tf.newaxis]

        batch_size = int(t_sequence.shape[0])

        if initial_state is None:
            init_pop, init_syn = self._network.get_initial_state(batch_size)
        else:
            init_pop, init_syn = initial_state

        (all_rates, stability_loss,
         final_pop, final_syn,
         pop_stacked, syn_stacked) = self._network._scan_forward_states(
            t_sequence, tuple(init_pop), tuple(init_syn))

        final_state = (list(final_pop), list(final_syn))

        output = NetworkOutput(
            firing_rates=all_rates,
            hidden_states=None,
            stability_loss=stability_loss,
            final_state=final_state,
        )
        return output, final_state, (list(init_pop), list(init_syn),
                                      pop_stacked, syn_stacked)

    def _build_state_full(
        self,
        init_pop: StateList,
        init_syn: StateList,
        pop_stacked: List[TensorType],
        syn_stacked: List[TensorType],
    ) -> Tuple[Tuple[TensorType, ...], Tuple[TensorType, ...]]:
        """Prepend initial state to stacked [T, ...] → [T+1, ...] for graph indexing."""
        pop_full = tuple(
            tf.concat([[init_pop[i]], pop_stacked[i]], axis=0)
            for i in range(len(pop_stacked))
        )
        syn_full = tuple(
            tf.concat([[init_syn[i]], syn_stacked[i]], axis=0)
            for i in range(len(syn_stacked))
        )
        return pop_full, syn_full

    # ── Compiled full backward loop ──────────────────────────────────────────

    @tf.function
    def _compiled_backward_loop(
        self,
        pop_full: Tuple[TensorType, ...],
        syn_full: Tuple[TensorType, ...],
        t_sequence: TensorType,
        target: Dict[str, TensorType],
        loss_fn_obj: BaseLoss,
        T_val: TensorType,
        dtype: tf.DType,
    ) -> List[TensorType]:
        """
        Full backward pass compiled into a single tf.while_loop graph.

        All slicing, adjoint steps, and gradient accumulation happen in graph
        mode — zero eager overhead, no Python/RAM↔VRAM transfers per step.
        """
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

            target_t = {name: t_tensor[:, t, :]
                        for name, t_tensor in target.items()}

            pop_list = list(pop_t)
            syn_list = list(syn_t)
            all_src = pop_list + syn_list + self._trainable

            with tf.GradientTape() as tape:
                tape.watch(pop_list)
                tape.watch(syn_list)

                new_pop, new_syn, y_dict, l_t1 = self._forward_only(
                    pop_t, syn_t, t_val, target_t,
                    loss_fn_obj, T_val, dtype,
                )

                npl = list(new_pop)
                nsl = list(new_syn)
                lpl = list(lp)
                lsl = list(ls)

                proxy = (
                    sum(tf.reduce_sum(p * lm) for p, lm in zip(npl, lpl))
                    + sum(tf.reduce_sum(s * lm) for s, lm in zip(nsl, lsl))
                )
                combined = proxy + l_t1

            grads = tape.gradient(combined, all_src, unconnected_gradients="zero")

            new_lp = tuple(grads[:n_pop])
            new_ls = tuple(grads[n_pop:n_pop + n_syn])
            ds = grads[n_pop + n_syn:]

            new_da = tuple(a + g for a, g in zip(da, ds))

            return counter - 1, new_lp, new_ls, new_da

        _, _, _, dtheta_acc = tf.while_loop(
            cond, body,
            (tf.constant(T - 1, dtype=tf.int32),
             lam_pop, lam_syn, dtheta_acc),
            parallel_iterations=1,
        )

        return list(dtheta_acc)

    # ── Compiled forward-only step ───────────────────────────────────────────

    @tf.function
    def _forward_only(
        self,
        pop_tup: Tuple[TensorType, ...],
        syn_tup: Tuple[TensorType, ...],
        t_val: TensorType,
        target_dict: Dict[str, TensorType],
        loss_fn_obj: BaseLoss,
        T_val: TensorType,
        dtype: tf.DType,
    ) -> Tuple[
        Tuple[TensorType, ...],
        Tuple[TensorType, ...],
        Dict[str, TensorType],
        TensorType,
    ]:
        new_pop, new_syn, _ = _step_fn(
            (pop_tup, syn_tup, tf.constant([0.0], dtype=dtype)),
            t_val, self._graph, self._integrator)
        npl = list(new_pop)
        nsl = list(new_syn)

        npd, _ = unpack_state(self._graph, npl, nsl)
        y_dict = get_firing_rates(self._graph, npd)

        l_t1 = loss_fn_obj.per_step_loss(y_dict, target_dict)
        T_f = tf.cast(T_val, dtype)
        l_t1 = tf.cond(T_val > 1, lambda: l_t1 / T_f, lambda: l_t1)

        return new_pop, new_syn, y_dict, l_t1

    @tf.function
    def _stab_forward_step(
        self,
        pop_tup: Tuple[TensorType, ...],
        syn_tup: Tuple[TensorType, ...],
        stab_acc: TensorType,
        t_val: TensorType,
        dtype: tf.DType,
    ) -> Tuple[Tuple[TensorType, ...], Tuple[TensorType, ...], TensorType]:
        return _step_fn(
            (pop_tup, syn_tup, stab_acc),
            t_val, self._graph, self._integrator)

    # ── Compiled full adjoint step (tape + forward + gradient) ───────────────

    @tf.function
    # ── Backward pass (compiled tf.while_loop) ──────────────────────────────

    def backward_pass(
        self,
        t_sequence: TensorType,
        state_info: Tuple[StateList, StateList, List[TensorType], List[TensorType]],
        target: Dict[str, TensorType],
        loss_fn: BaseLoss,
    ) -> List[TensorType]:
        """Discrete adjoint backward pass — fully compiled tf.while_loop.

        Zero eager overhead: state slicing, adjoint steps, and gradient
        accumulation all run in a single graph-mode tf.while_loop.
        """
        if t_sequence.shape.rank == 2:
            t_sequence = t_sequence[:, :, tf.newaxis]

        init_pop, init_syn, pop_stacked, syn_stacked = state_info
        T = int(t_sequence.shape[1])
        dtype = neuraltide.config.get_dtype()
        variables = self._network.trainable_variables

        pop_full, syn_full = self._build_state_full(
            init_pop, init_syn, pop_stacked, syn_stacked)

        dtheta_step_list = self._compiled_backward_loop(
            pop_full, syn_full, t_sequence, target,
            loss_fn, tf.constant(T, dtype=tf.int32), dtype,
        )

        grad_map = {v.name: g for v, g in zip(self._trainable, dtheta_step_list)}
        return [
            grad_map.get(v.name, tf.zeros_like(v))
            for v in variables
        ]

    # ── Compute gradients ────────────────────────────────────────────────────

    def compute_gradients(
        self,
        t_sequence: TensorType,
        target: Dict[str, TensorType],
        loss_fn: BaseLoss,
    ) -> Tuple[List[TensorType], List[tf.Variable], NetworkOutput]:
        output, _, state_info = self.forward_pass(t_sequence)

        variables = self._network.trainable_variables

        if isinstance(loss_fn, CompositeLoss):
            main_loss_obj, stab_terms = self._split_composite_loss(loss_fn)
        else:
            main_loss_obj = loss_fn
            stab_terms = []

        if main_loss_obj is not None:
            main_grads = self.backward_pass(
                t_sequence, state_info, target, main_loss_obj
            )
        else:
            main_grads = [tf.zeros_like(v) for v in variables]

        if stab_terms:
            total_stab_weight = sum(w for w, _ in stab_terms)
            stab_grads = self._stability_gradients()
            total_grads = [
                mg + sg * total_stab_weight
                for mg, sg in zip(main_grads, stab_grads)
            ]
        else:
            total_grads = main_grads

        return total_grads, variables, output

    @staticmethod
    def _split_composite_loss(
        loss_fn: CompositeLoss,
    ) -> Tuple[Optional[BaseLoss], List[Tuple[float, BaseLoss]]]:
        from neuraltide.training.losses import StabilityPenalty

        main_terms = [(w, l) for w, l in loss_fn.terms
                      if not isinstance(l, StabilityPenalty)]
        stab_terms = [(w, l) for w, l in loss_fn.terms
                      if isinstance(l, StabilityPenalty)]

        if not main_terms:
            return None, stab_terms

        if len(main_terms) == 1 and main_terms[0][0] == 1.0:
            return main_terms[0][1], stab_terms

        return CompositeLoss(main_terms), stab_terms

    def _stability_gradients(self) -> List[TensorType]:
        variables = self._network.trainable_variables
        dtype = neuraltide.config.get_dtype()

        batch_size = 1
        init_pop, init_syn = self._network.get_initial_state(batch_size)
        pop_states = tuple(init_pop)
        syn_states = tuple(init_syn)
        stability_acc = tf.zeros([1], dtype=dtype)

        t_val = tf.squeeze(tf.constant([[[0.05]]], dtype=dtype))

        with tf.GradientTape() as tape:
            for _ in range(10):
                pop_states, syn_states, stability_acc = self._stab_forward_step(
                    pop_states, syn_states, stability_acc, t_val, dtype)
            stability_loss = tf.reduce_mean(stability_acc)

        grads = tape.gradient(stability_loss, variables)

        return [
            g if g is not None else tf.zeros_like(v)
            for g, v in zip(grads, variables)
        ]
