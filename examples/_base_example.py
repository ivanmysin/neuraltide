"""
Базовый шаблон для примеров.
Импорты, настройка matplotlib и общие утилиты.
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.integrators import RK4Integrator
from neuraltide.utils import seed_everything


def run_simulation(pop, syn_in, syn_rec=None, T=100, dt=0.5, seed=42):
    """
    Запускает симуляцию сети и возвращает историю наблюдаемых.

    Args:
        pop: PopulationModel — главная популяция
        syn_in: SynapseModel — синапс вход→популяция
        syn_rec: SynapseModel — рекуррентный синапс (опц)
        T: время симуляции (ms)
        dt: шаг интегрирования (ms)
        seed: random seed

    Returns:
        t_values, obs_history, pop_states_history
    """
    seed_everything(seed)

    graph = NetworkGraph(dt=dt)
    graph.add_population('main', pop)
    if syn_rec is not None:
        graph.add_synapse('rec', syn_rec, src='main', tgt='main')

    network = NetworkRNN(graph, integrator=RK4Integrator())

    t_values = np.arange(0, T, dt, dtype=np.float32)
    t_seq = tf.constant(t_values[None, :, None], dtype=tf.float32)

    output = network(t_seq, training=False)

    return t_values, output, network


def plot_population_dynamics(t_values, pop, pop_states_history, output, save_path):
    """
    Отрисовывает динамику переменных популяции.

    Args:
        t_values: массив времен
        pop: PopulationModel
        pop_states_history: список состояний
        output: NetworkOutput
        save_path: путь для сохранения
    """
    n_states = len(pop.state_size)
    n_units = pop.n_units

    fig, axes = plt.subplots(n_states + 1, 1, figsize=(10, 3 * (n_states + 1)))

    state_labels = ['r (firing rate)', 'v (membrane)', 'w (adaptation)'][:n_states]
    colors = ['tab:blue', 'tab:orange', 'tab:green'][:n_states]

    for i in range(n_states):
        state = pop_states_history[i]
        if isinstance(state, tf.Tensor):
            state = state.numpy()
        if state.ndim == 3:
            state = state[0]
        for u in range(min(n_units, 3)):
            axes[i].plot(t_values, state[:, u], label=f'Unit {u}', alpha=0.8)
        axes[i].set_ylabel(state_labels[i])
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3)

    rates = output.firing_rates['main'].numpy()[0]
    axes[-1].plot(t_values, rates, color='tab:blue', linewidth=1.5)
    axes[-1].set_ylabel('Firing Rate (Hz)')
    axes[-1].set_xlabel('Time (ms)')
    axes[-1].grid(True, alpha=0.3)
    axes[-1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Figure saved as {save_path}")
