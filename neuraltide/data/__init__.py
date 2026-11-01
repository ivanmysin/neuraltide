"""
Модуль для работы с данными: сохранение, загрузка, визуализация.

Формат: HDF5 (.h5)

Структура файла:
    data.h5
    ├── inputs              [T, total_input_units]
    ├── target              [T, n_target_units]
    ├── time_seq            [T]
    └── metadata
        ├── dt              float
        ├── input_names     JSON list
        ├── input_n_units   JSON dict
        ├── target_names    JSON list
        ├── target_n_units  JSON dict
        └── generator_params JSON dict
"""
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import numpy as np


@dataclass
class Dataset:
    """Контейнер для данных обучения."""
    inputs: np.ndarray          # [T, total_input_units]
    target: np.ndarray          # [T, total_target_units]
    time_seq: np.ndarray        # [T]
    dt: float
    input_names: List[str]
    input_n_units: Dict[str, int]
    target_names: List[str]
    target_n_units: Dict[str, int]
    generator_params: Dict[str, Any] = field(default_factory=dict)

    @property
    def T(self) -> int:
        return self.inputs.shape[0]

    @property
    def total_input_units(self) -> int:
        return self.inputs.shape[1]

    @property
    def total_target_units(self) -> int:
        return self.target.shape[1]

    def input_slice(self, name: str) -> np.ndarray:
        """Возвращает срез входа по имени: [T, n_units_i]."""
        offset = 0
        for n in self.input_names:
            if n == name:
                break
            offset += self.input_n_units[n]
        n = self.input_n_units[name]
        return self.inputs[:, offset:offset + n]

    def target_slice(self, name: str) -> np.ndarray:
        """Возвращает срез цели по имени: [T, n_units_i]."""
        offset = 0
        for n in self.target_names:
            if n == name:
                break
            offset += self.target_n_units[n]
        n = self.target_n_units[name]
        return self.target[:, offset:offset + n]


def save_dataset(
    path: str,
    inputs: Dict[str, np.ndarray],
    target: Dict[str, np.ndarray],
    dt: float,
    generator_params: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Сохраняет dataset в HDF5.

    Args:
        path: путь к .h5 файлу
        inputs: {name: array[T, n_units_i]} — firing rates входов
        target: {name: array[T, n_units_i]} — целевые firing rates
        dt: шаг интегрирования (мс)
        generator_params: параметры генераторов (для reproducibility)
    """
    import h5py

    if not inputs:
        raise ValueError("inputs must not be empty")
    if not target:
        raise ValueError("target must not be empty")

    # Определяем T из первого входа
    first_key = next(iter(inputs))
    T = inputs[first_key].shape[0]

    # Проверяем согласованность
    for name, arr in inputs.items():
        if arr.shape[0] != T:
            raise ValueError(
                f"Input '{name}' has T={arr.shape[0]}, expected {T}")
    for name, arr in target.items():
        if arr.shape[0] != T:
            raise ValueError(
                f"Target '{name}' has T={arr.shape[0]}, expected {T}")

    # Стекаем inputs и target в плоские тензоры
    input_names = list(inputs.keys())
    input_n_units = {}
    input_parts = []
    for name in input_names:
        arr = np.asarray(inputs[name], dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        input_n_units[name] = arr.shape[1]
        input_parts.append(arr)
    inputs_flat = np.concatenate(input_parts, axis=1)  # [T, total_input]

    target_names = list(target.keys())
    target_n_units = {}
    target_parts = []
    for name in target_names:
        arr = np.asarray(target[name], dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        target_n_units[name] = arr.shape[1]
        target_parts.append(arr)
    target_flat = np.concatenate(target_parts, axis=1)  # [T, total_target]

    time_seq = np.arange(T, dtype=np.float32) * dt

    if generator_params is None:
        generator_params = {}

    with h5py.File(path, 'w') as f:
        f.create_dataset('inputs', data=inputs_flat, compression='gzip')
        f.create_dataset('target', data=target_flat, compression='gzip')
        f.create_dataset('time_seq', data=time_seq)

        meta = f.create_group('metadata')
        meta.attrs['dt'] = dt
        meta.attrs['input_names'] = json.dumps(input_names)
        meta.attrs['input_n_units'] = json.dumps(input_n_units)
        meta.attrs['target_names'] = json.dumps(target_names)
        meta.attrs['target_n_units'] = json.dumps(target_n_units)
        meta.attrs['generator_params'] = json.dumps(generator_params)


def load_dataset(path: str) -> Dataset:
    """
    Загружает dataset из HDF5.

    Args:
        path: путь к .h5 файлу

    Returns:
        Dataset с inputs, target, metadata
    """
    import h5py

    with h5py.File(path, 'r') as f:
        inputs = f['inputs'][:]           # [T, total_input]
        target = f['target'][:]           # [T, total_target]
        time_seq = f['time_seq'][:]       # [T]

        meta = f['metadata']
        dt = float(meta.attrs['dt'])
        input_names = json.loads(meta.attrs['input_names'])
        input_n_units = json.loads(meta.attrs['input_n_units'])
        target_names = json.loads(meta.attrs['target_names'])
        target_n_units = json.loads(meta.attrs['target_n_units'])
        generator_params = json.loads(meta.attrs.get('generator_params', '{}'))

    return Dataset(
        inputs=inputs,
        target=target,
        time_seq=time_seq,
        dt=dt,
        input_names=input_names,
        input_n_units=input_n_units,
        target_names=target_names,
        target_n_units=target_n_units,
        generator_params=generator_params,
    )


def plot_dataset(
    data: Dataset,
    max_t: Optional[float] = None,
    figsize: tuple = (14, 8),
) -> None:
    """
    Строит графики для визуальной проверки данных.

    Args:
        data: Dataset
        max_t: максимальное время для отображения (мс). Если None — всё.
        figsize: размер фигуры
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    T = data.T
    t = data.time_seq

    if max_t is not None:
        mask = t <= max_t
        t_plot = t[mask]
        inputs_plot = data.inputs[mask]
        target_plot = data.target[mask]
    else:
        t_plot = t
        inputs_plot = data.inputs
        target_plot = data.target

    n_inputs = len(data.input_names)
    n_targets = len(data.target_names)
    n_rows = max(n_inputs, n_targets)

    fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    # Входы
    col = 0
    offset = 0
    for i, name in enumerate(data.input_names):
        n = data.input_n_units[name]
        ax = axes[i, col]
        for j in range(n):
            ax.plot(t_plot, inputs_plot[:, offset + j], label=f'{name}[{j}]', alpha=0.8)
        ax.set_title(f'Input: {name}')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Rate (Hz)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        offset += n
    for i in range(n_inputs, n_rows):
        axes[i, col].axis('off')

    # Цели
    col = 1
    offset = 0
    for i, name in enumerate(data.target_names):
        n = data.target_n_units[name]
        ax = axes[i, col]
        for j in range(n):
            ax.plot(t_plot, target_plot[:, offset + j], label=f'{name}[{j}]', alpha=0.8)
        ax.set_title(f'Target: {name}')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Rate (Hz)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        offset += n
    for i in range(n_targets, n_rows):
        axes[i, col].axis('off')

    fig.suptitle(f'Dataset: T={T}, dt={data.dt}ms, '
                 f'inputs={list(data.input_n_units.keys())}, '
                 f'targets={list(data.target_n_units.keys())}',
                 fontsize=12)
    plt.tight_layout()
    plt.show()
