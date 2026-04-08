import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from typing import Dict, Optional, List


def plot_loss_curve(
    loss_history: List[float],
    title: str = "Training Loss",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Строит кривую loss на протяжении обучения.

    Args:
        loss_history: список значений loss по эпохам.
        title: заголовок графика.
        ax: если передан, рисует на существующем subplot.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    else:
        fig = ax.figure

    epochs = np.arange(1, len(loss_history) + 1)
    ax.plot(epochs, loss_history, color='tab:blue', linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig


def plot_target_vs_prediction(
    output: 'NetworkOutput',
    target: Dict[str, tf.Tensor],
    pop_name: str,
    t_values: np.ndarray,
    time_before: float = 5.0,
    time_after: float = 5.0,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Сравнивает target и предсказанные firing rates до и после обучения.

    Args:
        output: NetworkOutput от сети.
        target: словарь target тензоров.
        pop_name: имя популяции.
        t_values: массив временных точек.
        time_before: показывать первые N секунд (до обучения).
        time_after: показывать последние N секунд (после обучения).
        ax: если передан, рисует на существующем subplot.
    """
    pred = output.firing_rates[pop_name]
    tgt = target[pop_name]

    pred_np = pred.numpy()[0]
    tgt_np = tgt.numpy()[0]
    n_units = pred_np.shape[-1]

    t_before_idx = min(int(time_before * 1000 / (t_values[1] - t_values[0])), len(t_values))
    t_after_idx = max(len(t_values) - int(time_after * 1000 / (t_values[1] - t_values[0])), 0)

    n_cols = min(n_units, 3)
    n_rows = 2

    if ax is None:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_units == 1:
            axes = axes.reshape(2, 1)
    else:
        axes = ax
        fig = ax.figure

    for unit in range(n_units):
        row = unit // n_cols
        col = unit % n_cols
        a = axes[row, col] if n_rows > 1 else axes[col]

        a.plot(t_values[:t_before_idx], tgt_np[:t_before_idx, unit],
               color='tab:green', linewidth=2, label='Target (begin)', alpha=0.8)
        a.plot(t_values[t_after_idx:], tgt_np[t_after_idx:, unit],
               color='tab:orange', linewidth=2, label='Target (end)', alpha=0.8)
        a.plot(t_values[:t_before_idx], pred_np[:t_before_idx, unit],
               color='tab:blue', linewidth=1.5, linestyle='--', label='Pred (begin)', alpha=0.8)
        a.plot(t_values[t_after_idx:], pred_np[t_after_idx:, unit],
               color='tab:red', linewidth=1.5, linestyle='--', label='Pred (end)', alpha=0.8)

        a.set_xlabel("Time (ms)")
        a.set_ylabel("Firing Rate (Hz)")
        a.set_title(f"Unit {unit}")
        a.legend(fontsize=7)
        a.grid(True, alpha=0.3)

    return fig


def plot_training_comparison(
    network,
    loss_fn,
    t_sequence: tf.Tensor,
    target: Dict[str, tf.Tensor],
    pop_name: str,
    t_values: np.ndarray,
    epochs: int,
    title: str = "Training Progress",
) -> plt.Figure:
    """
    Строит 3 панели: loss curve + сравнение в начале + сравнение в конце обучения.

    Args:
        network: обученная NetworkRNN.
        loss_fn: функция потерь.
        t_sequence: временная последовательность.
        target: target тензоры.
        pop_name: имя популяции.
        t_values: массив временных точек.
        epochs: общее число эпох.
    """
    pred_before = network(t_sequence, training=False)
    history = []

    optimizer = tf.keras.optimizers.Adam(1e-3)
    trainer = Trainer(network, loss_fn, optimizer)

    for epoch in range(epochs):
        step = trainer.train_step(t_sequence)
        history.append(float(step['loss']))

    pred_after = network(t_sequence, training=False)

    n_units = pred_before.firing_rates[pop_name].numpy().shape[-1]
    n_cols = min(n_units, 3)

    fig = plt.figure(figsize=(15, 4))
    axes = [
        plt.subplot(1, 3, 1),
        plt.subplot(1, 3, 2),
        plt.subplot(1, 3, 3),
    ]

    plot_loss_curve(history, title=f"Loss (final: {history[-1]:.2f})", ax=axes[0])

    _plot_comparison_panel(
        axes[1], pred_before, target, pop_name, t_values,
        label_prefix="Before", color='tab:blue'
    )
    _plot_comparison_panel(
        axes[2], pred_after, target, pop_name, t_values,
        label_prefix="After", color='tab:red'
    )

    axes[1].set_title("Before Training")
    axes[2].set_title("After Training")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


def _plot_comparison_panel(
    ax: plt.Axes,
    output: 'NetworkOutput',
    target: Dict[str, tf.Tensor],
    pop_name: str,
    t_values: np.ndarray,
    label_prefix: str,
    color: str,
) -> None:
    pred = output.firing_rates[pop_name].numpy()[0]
    tgt = target[pop_name].numpy()[0]
    n_units = pred.shape[-1]

    t_show = min(500, len(t_values))
    t_idx = slice(0, t_show)

    for unit in range(n_units):
        ax.plot(t_values[t_idx], tgt[t_idx, unit],
                color='tab:green', linewidth=2, label='Target' if unit == 0 else None, alpha=0.8)
        ax.plot(t_values[t_idx], pred[t_idx, unit],
                color=color, linewidth=1.5, linestyle='--',
                label=f'{label_prefix}' if unit == 0 else None, alpha=0.8)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Firing Rate (Hz)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
