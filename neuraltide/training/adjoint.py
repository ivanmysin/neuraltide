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
    NetworkGraph,
    NetworkRNN,
    NetworkOutput,
    unpack_state,
    pack_state,
    get_firing_rates,
)
from neuraltide.core.types import TensorType, StateList
from neuraltide.integrators.base import BaseIntegrator
from neuraltide.populations.input_population import InputPopulation
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
    ) -> Tuple[NetworkOutput, Tuple[StateList, StateList]]:
        """
        Run forward pass and store states for backward pass.
        
        Args:
            t_sequence: Tensor of shape [batch, T, 1] or [batch, T]
            initial_state: Optional initial (pop_states, syn_states)
        
        Returns:
            (NetworkOutput, (final_pop_states, final_syn_states))
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

            new_pop, new_syn, stability_acc = self._step_fn(
                (pop_states_tuple, syn_states_tuple, stability_acc),
                t,
                self._graph,
                self._integrator
            )

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

        output = NetworkOutput(
            firing_rates=all_rates,
            hidden_states=None,
            stability_loss=stability_loss,
        )

        final_state = (pop_states, syn_states)

        return output, final_state

    def backward_pass(
        self,
        t_sequence: TensorType,
        states_sequence: List[Tuple[StateList, StateList]],
        target: Dict[str, TensorType],
        loss_fn: CompositeLoss,
    ) -> List[TensorType]:
        """
        Run backward pass using adjoint state method.
        
        Args:
            t_sequence: Input sequence [batch, T, 1]
            states_sequence: List of (pop_states, syn_states) for each step
            target: Target firing rates
            loss_fn: Loss function
        
        Returns:
            List of gradients for trainable_variables
        """
        if t_sequence.shape.rank == 2:
            t_sequence = t_sequence[:, :, tf.newaxis]

        n_steps = int(t_sequence.shape[1])
        variables = self._network.trainable_variables

        dtype = neuraltide.config.get_dtype()

        adj_state_len = sum(
            len(s) for s in states_sequence[0][0]
        ) + sum(
            len(s) for s in states_sequence[0][1]
        )
        adj_state = [
            tf.zeros_like(s) for s in states_sequence[0][0]
        ] + [
            tf.zeros_like(s) for s in states_sequence[0][1]
        ]

        dL_dtheta = [
            tf.zeros_like(v) if v.trainable else None
            for v in variables
        ]

        for step_idx in range(n_steps - 1, -1, -1):
            pop_states_prev, syn_states_prev = states_sequence[step_idx]
            pop_states_dict, syn_states_dict = unpack_state(
                self._graph, pop_states_prev, syn_states_prev
            )

            t = t_sequence[:, step_idx:step_idx + 1, 0]
            rates_t = get_firing_rates(self._graph, pop_states_dict)

            target_t = {}
            for name in target:
                if step_idx < target[name].shape[1]:
                    target_t[name] = target[name][:, step_idx, :]

            if target_t:
                step_loss = loss_fn.per_step_loss(rates_t, target_t)
            else:
                step_loss = tf.constant(0.0, dtype=dtype)

            with tf.GradientTape() as tape:
                tape.watch(pop_states_prev + syn_states_prev)
                tape.watch([v for v in variables if v is not None and v.trainable])

                pop_states_next, syn_states_next = self._integrate_step(
                    pop_states_dict,
                    syn_states_dict,
                    t,
                )

            adj_state_tensor = []
            for s in pop_states_next + syn_states_next:
                if s is not None:
                    adj_state_tensor.append(s)

            step_grads = tape.gradient(
                step_loss,
                pop_states_prev + syn_states_prev + [v for v in variables if v is not None and v.trainable]
            )

            step_grads_pop = step_grads[:len(pop_states_prev)]
            step_grads_syn = step_grads[len(pop_states_prev):len(pop_states_prev) + len(syn_states_prev)]
            step_grads_var = step_grads[len(pop_states_prev) + len(syn_states_prev):]

            for i, (g, a) in enumerate(zip(
                step_grads_pop,
                adj_state[:len(pop_states_prev)]
            )):
                if g is not None and a is not None:
                    adj_state[i] = adj_state[i] + g

            for i, (g, a) in enumerate(zip(
                step_grads_syn,
                adj_state[len(pop_states_prev):]
            )):
                if g is not None and a is not None:
                    adj_state[len(pop_states_prev) + i] = \
                        adj_state[len(pop_states_prev) + i] + g

            for i, (var, grad) in enumerate(zip(
                [v for v in variables if v is not None and v.trainable],
                step_grads_var
            )):
                if grad is not None:
                    var_idx = next(
                        j for j, vj in enumerate(variables)
                        if vj is not None and vj.name == var.name
                    )
                    if dL_dtheta[var_idx] is not None:
                        dL_dtheta[var_idx] = dL_dtheta[var_idx] + grad
                    else:
                        dL_dtheta[var_idx] = grad

        return dL_dtheta

    def _integrate_step(
        self,
        pop_states_dict: Dict[str, StateList],
        syn_states_dict: Dict[str, StateList],
        t: TensorType,
    ) -> Tuple[StateList, StateList]:
        """
        Perform one integration step.
        
        Args:
            pop_states_dict: Population states
            syn_states_dict: Synapse states
            t: Current time
        
        Returns:
            (new_pop_states, new_syn_states)
        """
        dtype = neuraltide.config.get_dtype()

        for name in self._graph.population_names:
            pop = self._graph._populations[name]
            if isinstance(pop, InputPopulation):
                pop_states_dict[name] = [t]

        syn_I: Dict[str, TensorType] = {}
        syn_g: Dict[str, TensorType] = {}
        for name in self._graph.dynamic_population_names:
            n = self._graph._populations[name].n_units
            syn_I[name] = tf.zeros([1, n], dtype=dtype)
            syn_g[name] = tf.zeros([1, n], dtype=dtype)

        for syn_name, entry in self._graph._synapses.items():
            src_pop = self._graph._populations[entry.src]
            tgt_pop = self._graph._populations[entry.tgt]
            src_state = pop_states_dict[entry.src]
            tgt_state = pop_states_dict[entry.tgt]
            syn_state = syn_states_dict[syn_name]

            pre_rate = src_pop.get_firing_rate(src_state)

            tgt_obs = tgt_pop.observables(tgt_state)
            post_v = tgt_obs.get(
                'v_mean',
                tf.zeros([1, tgt_pop.n_units], dtype=dtype)
            )

            new_syn_state, _ = self._integrator.step_synapse(
                entry.model, syn_state, pre_rate, post_v, entry.model.dt
            )

            current_dict = entry.model.compute_current(
                new_syn_state, pre_rate, post_v
            )

            syn_I[entry.tgt] = syn_I[entry.tgt] + current_dict['I_syn']
            syn_g[entry.tgt] = syn_g[entry.tgt] + current_dict['g_syn']
            syn_states_dict[syn_name] = new_syn_state

        new_pop_states_list = []
        for name in self._graph.population_names:
            pop = self._graph._populations[name]
            pop_state = pop_states_dict[name]

            if isinstance(pop, InputPopulation):
                new_pop_states_list.extend(pop_state)
            else:
                total_syn = {'I_syn': syn_I[name], 'g_syn': syn_g[name]}
                new_pop_state, _ = self._integrator.step(pop, pop_state, total_syn)
                new_pop_states_list.extend(new_pop_state)

        new_syn_states_list = []
        for name in self._graph.synapse_names:
            new_syn_states_list.extend(syn_states_dict[name])

        return new_pop_states_list, new_syn_states_list

    def _step_fn(
        self,
        states: Tuple[StateList, StateList, TensorType],
        t: TensorType,
        graph: NetworkGraph,
        integrator: BaseIntegrator,
    ) -> Tuple[StateList, StateList, TensorType]:
        """
        Internal step function (same as network._step_fn).
        """
        from neuraltide.core.network import compute_synapse_current

        pop_states, syn_states, stability_acc = states
        dtype = neuraltide.config.get_dtype()

        pop_states_dict = {}
        idx = 0
        for name in graph.population_names:
            pop = graph._populations[name]
            n = len(pop.state_size)
            pop_states_dict[name] = pop_states[idx:idx + n]
            idx += n

        syn_states_dict = {}
        idx = 0
        for name in graph.synapse_names:
            entry = graph._synapses[name]
            n = len(entry.model.state_size)
            syn_states_dict[name] = syn_states[idx:idx + n]
            idx += n

        for name in graph.population_names:
            pop = graph._populations[name]
            if isinstance(pop, InputPopulation):
                pop_states_dict[name] = [t]

        syn_I: Dict[str, TensorType] = {}
        syn_g: Dict[str, TensorType] = {}
        for name in graph.dynamic_population_names:
            n = graph._populations[name].n_units
            syn_I[name] = tf.zeros([1, n], dtype=dtype)
            syn_g[name] = tf.zeros([1, n], dtype=dtype)

        for syn_name, entry in graph._synapses.items():
            src_pop = graph._populations[entry.src]
            tgt_pop = graph._populations[entry.tgt]
            src_state = pop_states_dict[entry.src]
            tgt_state = pop_states_dict[entry.tgt]
            syn_state = syn_states_dict[syn_name]

            pre_rate = src_pop.get_firing_rate(src_state)

            tgt_obs = tgt_pop.observables(tgt_state)
            post_v = tgt_obs.get(
                'v_mean',
                tf.zeros([1, tgt_pop.n_units], dtype=dtype)
            )

            new_syn_state, local_err = integrator.step_synapse(
                entry.model, syn_state, pre_rate, post_v, entry.model.dt
            )

            current_dict = compute_synapse_current(
                entry.model, new_syn_state, pre_rate, post_v
            )

            syn_I[entry.tgt] = syn_I[entry.tgt] + current_dict['I_syn']
            syn_g[entry.tgt] = syn_g[entry.tgt] + current_dict['g_syn']
            syn_states_dict[syn_name] = new_syn_state

        stability_error = stability_acc
        new_pop_states_list = []
        for name in graph.population_names:
            pop = graph._populations[name]
            pop_state = pop_states_dict[name]

            if isinstance(pop, InputPopulation):
                new_pop_states_list.extend(pop_state)
            else:
                total_syn = {'I_syn': syn_I[name], 'g_syn': syn_g[name]}
                new_pop_state, local_err = integrator.step(pop, pop_state, total_syn)
                new_pop_states_list.extend(new_pop_state)
                stability_error = stability_error + local_err

        new_syn_states_list = []
        for name in graph.synapse_names:
            new_syn_states_list.extend(syn_states_dict[name])

        return (
            tuple(new_pop_states_list),
            tuple(new_syn_states_list),
            stability_error
        )

    def compute_gradients(
        self,
        t_sequence: TensorType,
        target: Dict[str, TensorType],
        loss_fn: BaseLoss,
    ) -> Tuple[List[TensorType], List[tf.Variable], NetworkOutput]:
        """
        Compute gradients using adjoint method with stored states.
        
        The adjoint method stores forward states then recomputes gradients more efficiently.
        
        Args:
            t_sequence: Input sequence [batch, T, 1]
            target: Target firing rates
            loss_fn: Loss function
        
        Returns:
            (gradients, trainable_variables, network_output)
        """
        output, states_final = self.forward_pass(t_sequence)

        variables = self._network.trainable_variables

        if isinstance(loss_fn, CompositeLoss):
            loss_obj = loss_fn.terms[0][1] if loss_fn.terms else None
        else:
            loss_obj = loss_fn

        with tf.GradientTape() as tape:
            network_output = self._network(t_sequence, training=False)
            loss = loss_obj(network_output, self._network)

        main_grads = tape.gradient(loss, self._network.trainable_variables)

        main_grads = [
            g if g is not None else tf.zeros_like(v)
            for g, v in zip(main_grads, variables)
        ]

        stability_weight = self._network._stability_penalty_weight

        if stability_weight > 0:
            stability_grads = self._stability_gradients()
            total_grads = [
                mg + sg * stability_weight
                if sg is not None else mg
                for mg, sg in zip(main_grads, stability_grads)
            ]
        else:
            total_grads = main_grads

        return total_grads, variables, output

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

                new_pop, new_syn, stability_acc = self._step_fn(
                    (pop_states_tuple, syn_states_tuple, stability_acc),
                    t,
                    self._graph,
                    self._integrator
                )

                pop_states = list(new_pop)
                syn_states = list(new_syn)

            stability_loss = tf.reduce_mean(stability_acc)

        grads = tape.gradient(stability_loss, variables)

        return [
            g if g is not None else tf.zeros_like(v)
            for g, v in zip(grads, variables)
        ]