"""
Adjoint state method for gradient computation.

This module provides an alternative to BPTT for computing gradients,
with reduced memory requirements for long sequences.
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


class AdjointSolver:
    """
    Solves adjoint state method for gradient computation.
    
    The adjoint method computes gradients by:
    1. Forward pass: integrate the system and store states
    2. Backward pass: recompute forward and integrate adjoint equation
    
    This reduces peak memory compared to BPTT by not storing
    the full computational graph.
    """

    def __init__(
        self,
        network: NetworkRNN,
        integrator: Optional[BaseIntegrator] = None,
    ):
        """
        Initialize adjoint solver.
        
        Args:
            network: NetworkRNN to compute gradients for
            integrator: Integrator to use (defaults to network's integrator)
        """
        self._network = network
        self._graph = network._graph
        self._integrator = integrator if integrator is not None else network._integrator

    def forward_pass(
        self,
        t_sequence: TensorType,
        initial_state: Optional[Tuple[StateList, StateList]] = None,
    ) -> Tuple[NetworkOutput, Tuple[StateList, StateList], List[Tuple[StateList, StateList]]]:
        """
        Run forward pass and store states for backward pass.
        
        Uses eager execution for maximum gradient correctness (tf.function
        compilation risks capturing tf.Variable values as constants).
        
        Args:
            t_sequence: Tensor of shape [batch, T, 1] or [batch, T]
            initial_state: Optional initial (pop_states, syn_states)
        
        Returns:
            NetworkOutput, final_state, states_sequence
        """
        if t_sequence.shape.rank == 2:
            t_sequence = t_sequence[:, :, tf.newaxis]

        batch_size = int(t_sequence.shape[0])
        n_steps = int(t_sequence.shape[1])

        if initial_state is None:
            init_pop, init_syn = self._network.get_initial_state(batch_size)
        else:
            init_pop, init_syn = initial_state

        pop_states = list(init_pop)
        syn_states = list(init_syn)
        stability_acc = tf.zeros([1], dtype=neuraltide.config.get_dtype())

        pop_states_dict, syn_states_dict = unpack_state(
            self._graph, pop_states, syn_states
        )

        dyn_names = self._graph.dynamic_population_names
        all_rates: Dict[str, List[TensorType]] = {name: [] for name in dyn_names}

        states_sequence: List[Tuple[StateList, StateList]] = [
            (list(pop_states), list(syn_states))
        ]

        for step in range(n_steps):
            t = t_sequence[:, step:step + 1, 0]

            pop_states_tuple = tuple(pop_states)
            syn_states_tuple = tuple(syn_states)

            new_pop, new_syn, stability_acc = _step_fn(
                (pop_states_tuple, syn_states_tuple, stability_acc),
                t, self._graph, self._integrator)

            pop_states = list(new_pop)
            syn_states = list(new_syn)

            states_sequence.append((list(pop_states), list(syn_states)))

            pop_states_dict, syn_states_dict = unpack_state(
                self._graph, pop_states, syn_states
            )

            rates = get_firing_rates(self._graph, pop_states_dict)
            for name in dyn_names:
                all_rates[name].append(rates[name])

        for name in dyn_names:
            all_rates[name] = tf.stack(all_rates[name], axis=1)

        stability_loss = self._network._stability_penalty_weight * tf.reduce_mean(
            stability_acc
        )

        final_state = (list(pop_states), list(syn_states))

        output = NetworkOutput(
            firing_rates=all_rates,
            hidden_states=None,
            stability_loss=stability_loss,
            final_state=final_state,
        )
        return output, final_state, states_sequence

    def backward_pass(
        self,
        t_sequence: TensorType,
        states_sequence: List[Tuple[StateList, StateList]],
        target: Dict[str, TensorType],
        loss_fn: BaseLoss,
    ) -> List[TensorType]:
        """
        Discrete adjoint backward pass.

        Computes gradients dL/dθ by propagating the adjoint vector λ backwards
        through the stored state sequence without holding the full TF graph.

        Optimised: non-persistent GradientTape, combined (proxy + loss) scalar
        target, single tape.gradient call per step (eliminates 3× redundant
        backward passes vs naive persistent-tape implementation).

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
        trainable_vars = [v for v in variables if v.trainable]

        pop_0, syn_0 = states_sequence[0]
        lam_pop = [tf.zeros_like(s) for s in pop_0]
        lam_syn = [tf.zeros_like(s) for s in syn_0]

        dL_dtheta = {v.name: tf.zeros_like(v) for v in trainable_vars}

        target_slices = []
        for t_idx in range(T):
            step_target = {}
            for name, t_tensor in target.items():
                if t_idx < int(t_tensor.shape[1]):
                    step_target[name] = t_tensor[:, t_idx, :]
            target_slices.append(step_target)

        graph = self._graph
        integrator = self._integrator
        zero_scalar = tf.constant([0.0], dtype=dtype)
        n_pop = len(pop_0)
        n_syn = len(syn_0)

        for t_idx in range(T - 1, -1, -1):
            pop_t = states_sequence[t_idx][0]
            syn_t = states_sequence[t_idx][1]
            t_val = t_sequence[:, t_idx:t_idx + 1, 0]
            target_t1 = target_slices[t_idx]

            all_sources = pop_t + syn_t + trainable_vars

            with tf.GradientTape() as tape:
                tape.watch(pop_t)
                tape.watch(syn_t)

                new_pop, new_syn, _ = _step_fn(
                    (tuple(pop_t), tuple(syn_t), zero_scalar),
                    t_val, graph, integrator)
                new_pop_list = list(new_pop)
                new_syn_list = list(new_syn)

                new_pop_dict, _ = unpack_state(
                    graph, new_pop_list, new_syn_list)
                y_t1 = get_firing_rates(graph, new_pop_dict)

                l_t1 = loss_fn.per_step_loss(y_t1, target_t1)
                if T > 1:
                    l_t1 = l_t1 / tf.cast(T, dtype)

                proxy = (
                    sum(tf.reduce_sum(p * lam)
                        for p, lam in zip(new_pop_list, lam_pop))
                    + sum(tf.reduce_sum(s * lam)
                         for s, lam in zip(new_syn_list, lam_syn))
                )

                combined = proxy + l_t1

            all_grads = tape.gradient(
                combined, all_sources, unconnected_gradients='zero')

            lam_pop = all_grads[:n_pop]
            lam_syn = all_grads[n_pop:n_pop + n_syn]
            dtheta_step = all_grads[n_pop + n_syn:]

            for v, g in zip(trainable_vars, dtheta_step):
                dL_dtheta[v.name] = dL_dtheta[v.name] + g

        return [
            dL_dtheta.get(v.name, tf.zeros_like(v))
            for v in variables
        ]

    def compute_gradients(
        self,
        t_sequence: TensorType,
        target: Dict[str, TensorType],
        loss_fn: BaseLoss,
    ) -> Tuple[List[TensorType], List[tf.Variable], NetworkOutput]:
        """
        Compute gradients using the discrete adjoint backward pass.

        Runs one forward pass (stores states), then one backward pass
        (adjoint propagation). Does not store the TF computational graph
        across all T steps — peak memory is O(state_size), not O(T·graph).

        Args:
            t_sequence: Input sequence [batch, T, 1]
            target: Target firing rates {pop_name: [batch, T, n_units]}
            loss_fn: Loss function (must implement per_step_loss for adjoint)

        Returns:
            (gradients, trainable_variables, network_output)
        """
        output, _, states_sequence = self.forward_pass(t_sequence)

        variables = self._network.trainable_variables

        # Extract the primary (non-stability) loss object for adjoint
        if isinstance(loss_fn, CompositeLoss):
            main_loss_obj, stab_terms = self._split_composite_loss(loss_fn)
        else:
            main_loss_obj = loss_fn
            stab_terms = []

        # ── Main gradients via discrete adjoint backward pass ────────────────
        if main_loss_obj is not None:
            main_grads = self.backward_pass(
                t_sequence, states_sequence, target, main_loss_obj
            )
        else:
            main_grads = [tf.zeros_like(v) for v in variables]

        # ── Stability penalty gradients via separate GradientTape ────────────
        # StabilityPenalty depends on integrator local error (a function of
        # parameters only, not of the long trajectory), so a local tape suffices.
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
        """
        Separate CompositeLoss into adjoint-compatible terms and StabilityPenalty.

        Returns:
            (main_loss_obj, stab_terms)
            main_loss_obj: first non-stability BaseLoss (or None)
            stab_terms: list of (weight, StabilityPenalty) entries
        """
        from neuraltide.training.losses import StabilityPenalty

        main_terms = [(w, l) for w, l in loss_fn.terms
                      if not isinstance(l, StabilityPenalty)]
        stab_terms  = [(w, l) for w, l in loss_fn.terms
                       if isinstance(l, StabilityPenalty)]

        if not main_terms:
            return None, stab_terms

        # If there is exactly one main term with weight 1.0, use it directly
        if len(main_terms) == 1 and main_terms[0][0] == 1.0:
            return main_terms[0][1], stab_terms

        # Otherwise wrap remaining terms in a new CompositeLoss
        return CompositeLoss(main_terms), stab_terms

    def _stability_gradients(self) -> List[TensorType]:
        """
        Compute gradients for stability penalty.
        
        Stability penalty depends on local integrator error,
        so we compute it via separate GradientTape.
        """
        variables = self._network.trainable_variables

        batch_size = 1
        init_pop, init_syn = self._network.get_initial_state(batch_size)
        pop_states = list(init_pop)
        syn_states = list(init_syn)
        stability_acc = tf.zeros([1], dtype=neuraltide.config.get_dtype())

        t = tf.constant([[[0.05]]], dtype=neuraltide.config.get_dtype())

        with tf.GradientTape() as tape:
            for _ in range(10):
                pop_states_tuple = tuple(pop_states)
                syn_states_tuple = tuple(syn_states)

                new_pop, new_syn, stability_acc = _step_fn(
                    (pop_states_tuple, syn_states_tuple, stability_acc),
                    t, self._graph, self._integrator)

                pop_states = list(new_pop)
                syn_states = list(new_syn)

            stability_loss = tf.reduce_mean(stability_acc)

        grads = tape.gradient(stability_loss, variables)

        return [
            g if g is not None else tf.zeros_like(v)
            for g, v in zip(grads, variables)
        ]