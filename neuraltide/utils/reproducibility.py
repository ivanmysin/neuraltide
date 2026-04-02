import os
import json
import random
import numpy as np
import tensorflow as tf


def seed_everything(seed: int) -> None:
    """
    Фиксирует seed для всех источников случайности.

    Args:
        seed: значение seed.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def log_versions() -> dict:
    """
    Возвращает словарь с версиями библиотек.
    """
    import neuraltide
    return {
        'neuraltide': neuraltide.__version__,
        'tensorflow': tf.__version__,
        'numpy': np.__version__,
    }


def save_experiment_state(path: str, network, optimizer, history=None,
                         seed=None, extra_config=None) -> None:
    """
    Сохраняет полное состояние эксперимента.

    Структура:
        {path}/
        ├── checkpoint/
        │   ├── checkpoint
        │   └── ckpt-1.*
        ├── config.json
        ├── versions.json
        ├── training_history.json
        └── seeds.json
    """
    os.makedirs(path, exist_ok=True)

    checkpoint_dir = os.path.join(path, 'checkpoint')
    checkpoint = tf.train.Checkpoint(
        optimizer=optimizer,
        network=network,
    )
    checkpoint.save(file_prefix=os.path.join(checkpoint_dir, 'ckpt'))

    if extra_config is not None:
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(extra_config, f, indent=2)

    versions = log_versions()
    with open(os.path.join(path, 'versions.json'), 'w') as f:
        json.dump(versions, f, indent=2)

    if history is not None:
        history_data = {
            'loss_history': [float(x) for x in history.loss_history],
            'epochs': history.epochs,
        }
        with open(os.path.join(path, 'training_history.json'), 'w') as f:
            json.dump(history_data, f, indent=2)

    if seed is not None:
        seeds_data = {'seed': seed}
        with open(os.path.join(path, 'seeds.json'), 'w') as f:
            json.dump(seeds_data, f, indent=2)
