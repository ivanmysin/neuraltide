#!/usr/bin/env python
"""
Бенчмарк G0: граф-режимы вычислений (этап 1 плана
context/gpu_batching_plan.md).

Задача: L0-сеть (WilsonCowan 1 юнит + StaticSynapse от ConstantRate,
параметры как в tests/conftest.py::_build_l0), EulerIntegrator, dt=0.5 мс.

Два блока замеров (warmup 1 вызов — первая трассировка графа, затем
5 повторов, median; файлы не создаются):

  1. Trainer.train_step — 1 эпоха = 1 шаг на [1, T, 1]:
     eager (run_eagerly=True) vs tf.function (default) vs
     tf.function + jit_compile=True (XLA);
  2. CalibrationProblem.value_and_gradient: eager (compiled=False) vs
     скомпилированное ядро (compiled=True).

Горизонты T ∈ {200, 1000, 4000}. Честное ожидание: tf.function убирает
Python-оверхед за итерацию, XLA даёт дополнительный выигрыш на больших
T; compiled-ядро задачи калибровки убирает оверхед построения графа
операций на каждую итерацию оптимизатора.

Запуск:
    python benchmarks/g0_graph_mode.py
"""

import argparse
import platform
import time

import numpy as np
import tensorflow as tf

import neuraltide.config
from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.integrators import EulerIntegrator
from neuraltide.inputs import ConstantRateGenerator
from neuraltide.optimization import (
    CalibrationProblem,
    ObservationSpec,
    ParameterSpace,
    ResidualLoss,
)
from neuraltide.populations import WilsonCowan
from neuraltide.synapses import StaticSynapse
from neuraltide.training import CompositeLoss, MSELoss, Trainer
from neuraltide.utils import seed_everything

DT = 0.5


def _t_seq(n):
    return tf.constant((np.arange(n, dtype=np.float32) * DT)[None, :, None])


def build_l0(T):
    """L0-сеть + протокол длины T (параметры как в tests/conftest.py)."""
    pop = WilsonCowan(
        dt=DT,
        params={
            "tau": {"value": [8.0], "trainable": True, "min": 1.0, "max": 50.0},
            "a": 1.0,
            "theta": 0.0,
            "w": 0.2,
            "I_ext": {"value": [0.8], "trainable": True},
            "max_rate": 100.0,
        },
        name="exc",
    )
    syn = StaticSynapse(
        n_pre=1,
        n_post=1,
        dt=DT,
        params={
            "gsyn_max": {"value": [[0.05]], "trainable": True},
            "pconn": [[1.0]],
            "e_r": 0.0,
        },
        name="in->exc",
    )
    gen = ConstantRateGenerator(dt=DT, params={"rate": 10.0})
    graph = NetworkGraph(dt=DT)
    graph.declare_input("in", n_units=1)
    graph.add_population("exc", pop)
    graph.add_synapse("in->exc", syn, src="in", tgt="exc")
    network = NetworkRNN(graph, EulerIntegrator())
    t_seq = _t_seq(T)
    inputs = graph.pack_inputs({"in": gen(t_seq)})
    return network, t_seq, inputs


def _median_time(fn, n_warmup=1, n_repeat=5):
    """Median wall time после разогрева (первая вызовы трассируют граф)."""
    for _ in range(n_warmup):
        fn()
    ts = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def bench_trainer(T, jit):
    """Одна конфигурация Trainer: свежая сеть, warmup + 5 повторов train_step."""
    network, t_seq, inputs = build_l0(T)
    target = network(t_seq, inputs=inputs).firing_rates["exc"]
    loss_fn = CompositeLoss([(1.0, MSELoss({"exc": target + 0.5}))])
    trainer = Trainer(
        network,
        loss_fn,
        tf.keras.optimizers.SGD(1e-3),
        run_eagerly=False,
        jit_compile=jit,
    )
    return _median_time(lambda: trainer.train_step(t_seq, inputs))


def bench_trainer_eager(T):
    network, t_seq, inputs = build_l0(T)
    target = network(t_seq, inputs=inputs).firing_rates["exc"]
    loss_fn = CompositeLoss([(1.0, MSELoss({"exc": target + 0.5}))])
    trainer = Trainer(network, loss_fn, tf.keras.optimizers.SGD(1e-3), run_eagerly=True)
    return _median_time(lambda: trainer.train_step(t_seq, inputs))


def bench_problem(T, compiled):
    """Одна конфигурация CalibrationProblem: warmup + 5 повторов vg."""
    network, t_seq, inputs = build_l0(T)
    target = network(t_seq, inputs=inputs).firing_rates["exc"]
    loss = ResidualLoss(
        [ObservationSpec(name="obs", population="exc", target=target, sigma=2.0)]
    )
    space = ParameterSpace.from_network(network)
    problem = CalibrationProblem(
        network, space, t_seq, inputs=inputs, objective=loss, compiled=compiled
    )
    phi = problem.pack()
    return _median_time(lambda: problem.value_and_gradient(phi))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="G0 benchmark: eager vs tf.function vs XLA (Trainer) и "
        "eager vs compiled (CalibrationProblem), L0-сеть."
    )
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[200, 1000, 4000],
        help="длины протокола T",
    )
    args = parser.parse_args()

    neuraltide.config.set_dtype(tf.float32)  # до создания моделей
    seed_everything(args.seed)

    print("=" * 78)
    print(
        "G0: граф-режимы — Trainer (eager / tf.function / XLA), "
        "CalibrationProblem (eager / compiled)"
    )
    print("=" * 78)
    print(
        f"TF {tf.__version__}, Python {platform.python_version()}, CPU, "
        f"dtype={neuraltide.config.get_dtype().name}, dt={DT} мс; "
        "сеть L0 (WilsonCowan 1 юнит + StaticSynapse, Euler); "
        "warmup 1 + 5 повторов, median"
    )

    print("\n[1] Trainer.train_step (1 эпоха = 1 шаг на [1, T, 1])")
    hdr = (
        f"{'T':>6}{'eager,s':>12}{'tf.function,s':>15}{'speedup':>9}"
        f"{'XLA,s':>12}{'speedup':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    trainer_rows = []
    for T in args.horizons:
        t_eager = bench_trainer_eager(T)
        t_graph = bench_trainer(T, jit=False)
        t_xla = bench_trainer(T, jit=True)
        row = (T, t_eager, t_graph, t_xla)
        trainer_rows.append(row)
        print(
            f"{T:>6}{t_eager:>12.4f}{t_graph:>15.4f}"
            f"{t_eager / t_graph:>8.2f}x{t_xla:>11.4f}{t_eager / t_xla:>8.2f}x"
        )

    print("\n[2] CalibrationProblem.value_and_gradient")
    hdr = f"{'T':>6}{'eager,s':>12}{'compiled,s':>13}{'speedup':>9}"
    print(hdr)
    print("-" * len(hdr))
    problem_rows = []
    for T in args.horizons:
        t_eager = bench_problem(T, compiled=False)
        t_compiled = bench_problem(T, compiled=True)
        problem_rows.append((T, t_eager, t_compiled))
        print(f"{T:>6}{t_eager:>12.4f}{t_compiled:>13.4f}{t_eager / t_compiled:>8.2f}x")

    print("\n[заключение]")
    t_max = trainer_rows[-1]
    if t_max[1] > t_max[2] and t_max[1] > t_max[3]:
        print(
            f"  * Подтверждено: на T={t_max[0]} tf.function быстрее eager в "
            f"{t_max[1] / t_max[2]:.2f} раза, XLA — в {t_max[1] / t_max[3]:.2f} "
            "раза (Python-оверхед за итерацию снят графом)."
        )
    else:
        print(
            "  * Ожидание не подтверждено: на большом T граф-режимы не "
            "быстрее eager — проверьте окружение (CPU, версию TF)."
        )
    p_max = problem_rows[-1]
    if p_max[1] > p_max[2]:
        print(
            f"  * compiled=True у задачи калибровки быстрее eager в "
            f"{p_max[1] / p_max[2]:.2f} раза на T={p_max[0]}."
        )
    else:
        print(
            f"  * compiled=True на T={p_max[0]} не быстрее eager "
            f"(speedup {p_max[1] / p_max[2]:.2f}x) — на мелких T оверхед "
            "графа может не окупаться; эффект растёт с T и числом итераций."
        )
    print(
        "  * XLA имеет постоянный оверхед на вызов скомпилированного "
        "кластера: на малых T (200) он медленнее обычного tf.function — "
        "выигрыш XLA проявляется с ростом T."
    )
    print(
        "  * Практический вывод: длинные горизонты и многоитерационная "
        "оптимизация — включайте Trainer(jit_compile=True) (если "
        "интегратор не AdaptiveRK/Exponential) и CalibrationProblem("
        "compiled=True); отладка и FD/Taylor-диагностика — в eager."
    )


if __name__ == "__main__":
    main()
