"""
Performance profiling utilities for NeuralTide.

Provides tf.profiler integration to identify bottlenecks in the simulation
and training pipeline.
"""

import os
import tempfile
import time
from contextlib import contextmanager
from typing import Optional

import tensorflow as tf


@contextmanager
def profile(logdir: Optional[str] = None):
    """
    Context manager that profiles TF ops within the block.

    Usage::

        with profile("logs/profile"):
            trainer.train_step(t_seq)

    Then view results with::

        tensorboard --logdir logs/profile

    Args:
        logdir: Directory for profiling traces.
                If None, uses a temporary directory.
    """
    if logdir is None:
        logdir = tempfile.mkdtemp(prefix="neuraltide_profile_")
    os.makedirs(logdir, exist_ok=True)
    options = tf.profiler.experimental.ProfilerOptions(
        host_tracer_level=2,
        python_tracer_level=1,
        device_tracer_level=1,
    )
    try:
        tf.profiler.experimental.start(logdir, options=options)
        yield logdir
    finally:
        try:
            tf.profiler.experimental.stop()
        except Exception:
            pass


def profile_step(trainer, t_sequence, initial_state=None, logdir=None):
    """
    Profile a single training step and return elapsed wall time.

    Args:
        trainer: Trainer instance.
        t_sequence: Time sequence tensor.
        initial_state: Optional initial state.
        logdir: Directory for profiling traces.

    Returns:
        (loss, elapsed_seconds, logdir)
    """
    with profile(logdir) as logdir:
        t0 = time.perf_counter()
        result = trainer.train_step(t_sequence, initial_state=initial_state)
        elapsed = time.perf_counter() - t0
    return result, elapsed, logdir
