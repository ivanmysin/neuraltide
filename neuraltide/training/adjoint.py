"""
Adjoint state method for gradient computation.

This module provides an alternative to BPTT for computing gradients,
with reduced memory requirements for long sequences.

Memory strategy:
  - Forward pass: compiled tf.scan (no Python list overhead)
  - Backward pass: compiled forward step + eager GradientTape
    (avoids ~2MB of compiled backward-graph memory)
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

    Memory: O(state_size) — no O(T) trajectory storage in forward pass.
    The compiled scan produces stacked states (T × state_size) internally,
    which are sliced on-the-fly during backward without extra copies.
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

    # ── Forward pass (compiled scan, no Python state storage) ────────────────

    def forward_pass(
        self,
        t_sequence: TensorType,
        initial_state: Optional[Tuple[StateList, StateList]] = None,
    ) -> Tuple[NetworkOutput, Tuple[StateList, StateList], List[Tuple[StateList, StateList]]]:
        """
        Run forward pass using compiled tf.scan.

        Uses NetworkRNN._scan_forward_states for graph-mode speed.
        Returns stacked state tensors (sliced on-the-fly in backward pass)
        avoiding per-step Python list allocation.
        """
        if t_sequence.shape.rank == 2:
            t_sequence = t_sequence[:, :, tf.newaxis]

        batch_size = int(t_sequence.shape[0])
        n_steps = int(t_sequence.shape[1])

        if initial_state is None:
            init_pop, init_syn = self._network.get_initial_state(batch_size)
        else:
            init_pop, init_syn = initial_state

        (all_rates, stability_loss,
         final_pop, final_syn,
         pop_stacked, syn_stacked) = self._network._scan_forward_states(
            t_sequence, tuple(init_pop), tuple(init_syn))

        # Build states_sequence by slicing stacked tensors at each time step.
        # list(zip(...)) uses tensor references from the stacked arrays — no copy.
        init_pop_list = list(init_pop)
        init_syn_list = list(init_syn)
        post_step_pop = [[s[t] for s in pop_stacked] for t in range(n_steps)]
        post_step_syn = [[s[t] for s in syn_stacked] for t in range(n_steps)]
        states_sequence = (
            [(init_pop_list, init_syn_list)]
            + list(zip(post_step_pop, post_step_syn))
        )

        final_state = (list(final_pop), list(final_syn))

        output = NetworkOutput(
            firing_rates=all_rates,
            hidden_states=None,
            stability_loss=stability_loss,
            final_state=final_state,
        )
        return output, final_state, states_sequence

    # ── Compiled forward-only step (no GradientTape inside) ──────────────────

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
        """
        One forward simulation step + output + loss (no gradient tape).

        Returns (new_pop, new_syn, y_dict, l_t1) for use by the outer tape.
        The outer GradientTape wraps the compiled graph, recording ops for
        differentiation without storing a compiled backward graph.
        """
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

    # ── Backward pass (eager tape + compiled forward) ────────────────────────

    def backward_pass(
        self,
        t_sequence: TensorType,
        states_sequence: List[Tuple[StateList, StateList]],
        target: Dict[str, TensorType],
        loss_fn: BaseLoss,
    ) -> List[TensorType]:
        """
        Discrete adjoint backward pass.

        Hybrid: compiled forward step (graph-mode speed) inside
        eager GradientTape (avoids ~2 MB of compiled backward-graph memory).

        Args:
            t_sequence: [batch, T, 1]
            states_sequence: list of T+1 (pop_states, syn_states) from forward_pass
            target: {pop_name: [batch, T, n_units]}
            loss_fn: must implement per_step_loss()

        Returns:
            List of gradients aligned with network.trainable_variables
        """
        if t_sequence.shape.rank == 2:
            t_sequence = t_sequence[:, :, tf.newaxis]

        T = len(states_sequence) - 1
        dtype = neuraltide.config.get_dtype()
        variables = self._network.trainable_variables

        pop_0, syn_0 = states_sequence[0]
        lam_pop = [tf.zeros_like(s) for s in pop_0]
        lam_syn = [tf.zeros_like(s) for s in syn_0]

        dL_dtheta_list = [tf.zeros_like(v) for v in self._trainable]

        target_slices = []
        for t_idx in range(T):
            step_target = {}
            for name, t_tensor in target.items():
                if t_idx < int(t_tensor.shape[1]):
                    step_target[name] = t_tensor[:, t_idx, :]
            target_slices.append(step_target)

        T_int = tf.constant(T, dtype=tf.int32)
        n_pop = len(pop_0)
        n_syn = len(syn_0)

        for t_idx in range(T - 1, -1, -1):
            pop_t = states_sequence[t_idx][0]
            syn_t = states_sequence[t_idx][1]
            t_val = t_sequence[:, t_idx:t_idx + 1, 0]
            t_sq = tf.squeeze(t_val, axis=1) if t_val.shape.rank > 1 else t_val

            all_src = pop_t + syn_t + self._trainable

            with tf.GradientTape() as tape:
                tape.watch(pop_t)
                tape.watch(syn_t)

                new_pop, new_syn, y_dict, l_t1 = self._forward_only(
                    tuple(pop_t), tuple(syn_t),
                    t_sq, target_slices[t_idx], loss_fn, T_int, dtype,
                )

                npl = list(new_pop)
                nsl = list(new_syn)

                proxy = (
                    sum(tf.reduce_sum(p * lm)
                        for p, lm in zip(npl, lam_pop))
                    + sum(tf.reduce_sum(s * lm)
                         for s, lm in zip(nsl, lam_syn))
                )
                combined = proxy + l_t1

            all_grads = tape.gradient(
                combined, all_src, unconnected_gradients="zero")

            lam_pop = all_grads[:n_pop]
            lam_syn = all_grads[n_pop:n_pop + n_syn]
            dtheta_step = all_grads[n_pop + n_syn:]

            for i in range(len(self._trainable)):
                dL_dtheta_list[i] = dL_dtheta_list[i] + dtheta_step[i]

        grad_map = {v.name: g for v, g in zip(self._trainable, dL_dtheta_list)}
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
        output, _, states_sequence = self.forward_pass(t_sequence)

        variables = self._network.trainable_variables

        if isinstance(loss_fn, CompositeLoss):
            main_loss_obj, stab_terms = self._split_composite_loss(loss_fn)
        else:
            main_loss_obj = loss_fn
            stab_terms = []

        if main_loss_obj is not None:
            main_grads = self.backward_pass(
                t_sequence, states_sequence, target, main_loss_obj
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

        batch_size = 1
        init_pop, init_syn = self._network.get_initial_state(batch_size)
        pop_states = list(init_pop)
        syn_states = list(init_syn)
        stability_acc = tf.zeros([1], dtype=neuraltide.config.get_dtype())

        t = tf.constant([[[0.05]]], dtype=neuraltide.config.get_dtype())

        with tf.GradientTape() as tape:
            for _ in range(10):
                new_pop, new_syn, stability_acc = _step_fn(
                    (tuple(pop_states), tuple(syn_states), stability_acc),
                    t, self._graph, self._integrator)
                pop_states = list(new_pop)
                syn_states = list(new_syn)
            stability_loss = tf.reduce_mean(stability_acc)

        grads = tape.gradient(stability_loss, variables)

        return [
            g if g is not None else tf.zeros_like(v)
            for g, v in zip(grads, variables)
        ]
