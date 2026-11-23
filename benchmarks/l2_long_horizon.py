#!/usr/bin/env python
"""
Бенчмарк L2: длинный горизонт и режимы градиента.

Задача (дорожная карта, раздел 9.1): L1-подобная малая сеть с медленной
динамикой (tau_pop = 60 мс — медленная адаптация) на длинном горизонте
(по умолчанию 1500 шагов = 750 мс). Сравниваются три режима производных P1:

  1. plain BPTT — problem.value_and_gradient (эталон дискретной модели);
  2. checkpointed — value_and_gradient_checkpointed: сегментный прогон с
     tf.recompute_grad / ручным fallback, градиент должен совпасть с plain;
  3. truncated_bptt — явно приближённый режим с обрывом градиента на границе
     окна; честность оценивается обязательным free-run после обучения.

Метрики (stdout, файлы не создаются):
  - относительная ошибка градиента checkpointed vs plain;
  - loss и wall time каждого режима, n_forward / n_gradient;
  - peak memory (best-effort, tracemalloc — нижняя оценка);
  - для truncated BPTT: final_loss (окно) vs free_run_loss (полный прогон).

Запуск:
    python benchmarks/l2_long_horizon.py
    python benchmarks/l2_long_horizon.py --help
    python benchmarks/l2_long_horizon.py --steps 800 --horizon 50 --epochs 2
"""

import argparse
import time

import numpy as np
import tensorflow as tf

import neuraltide.config
from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.inputs import SinusoidalGenerator
from neuraltide.integrators import EulerIntegrator
from neuraltide.optimization import (
    CalibrationProblem,
    LogTransform,
    ObservationSpec,
    ParameterSpace,
    ResidualLoss,
    truncated_bptt,
    value_and_gradient_checkpointed,
)
from neuraltide.populations import WilsonCowan
from neuraltide.synapses import StaticSynapse
from neuraltide.utils import seed_everything

DT = 0.5
TAU_SLOW = 60.0  # мс — медленная адаптация

THETA_STAR = {
    "exc/tau": TAU_SLOW,
    "exc/I_ext": 0.8,
    "in->exc/gsyn_max": 30.0,
}
THETA_START = {
    "exc/tau": 25.0,
    "exc/I_ext": 0.2,
    "in->exc/gsyn_max": 120.0,
}


def build_problem(steps: int):
    """Медленная L0-сеть на длинном горизонте; цель — прогон при theta*."""
    pop = WilsonCowan(
        dt=DT,
        params={
            "tau": {
                "value": [THETA_STAR["exc/tau"]],
                "trainable": True,
                "min": 1.0,
                "max": 200.0,
            },
            "a": 1.0,
            "theta": 0.0,
            "w": 0.2,
            "I_ext": {"value": [THETA_STAR["exc/I_ext"]], "trainable": True},
            "max_rate": 100.0,
        },
        name="exc",
    )
    syn = StaticSynapse(
        n_pre=1,
        n_post=1,
        dt=DT,
        params={
            "gsyn_max": {
                "value": [[THETA_STAR["in->exc/gsyn_max"]]],
                "trainable": True,
                "min": 0.1,
                "max": 1000.0,
            },
            "pconn": [[1.0]],
            "e_r": 1.0,
        },
        name="in->exc",
    )
    gen = SinusoidalGenerator(
        dt=DT,
        params={"amplitude": 8.0, "freq": 8.0, "phase": 0.0, "offset": 10.0},
    )
    graph = NetworkGraph(dt=DT)
    graph.declare_input("in", n_units=1)
    graph.add_population("exc", pop)
    graph.add_synapse("in->exc", syn, src="in", tgt="exc")
    network = NetworkRNN(graph, EulerIntegrator())

    t_seq = tf.constant((np.arange(steps, dtype=np.float64) * DT)[None, :, None])
    inputs = graph.pack_inputs({"in": gen(t_seq)})
    target = network(t_seq, inputs=inputs).firing_rates["exc"]
    space = ParameterSpace.from_network(
        network,
        transforms={
            "exc/tau": LogTransform(),
            "in->exc/gsyn_max": LogTransform(),
        },
    )
    objective = ResidualLoss(
        [ObservationSpec(name="exc", population="exc", target=target, sigma=2.0)]
    )
    problem = CalibrationProblem(
        network, space, t_seq, inputs=inputs, objective=objective
    )
    return problem, space


def rel_err(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="L2 benchmark: long-horizon gradient modes."
    )
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--steps", type=int, default=1500, help="число временных шагов горизонта"
    )
    parser.add_argument(
        "--segment", type=int, default=100, help="длина сегмента checkpointed-режима"
    )
    parser.add_argument(
        "--horizon", type=int, default=100, help="длина окна truncated BPTT"
    )
    parser.add_argument("--epochs", type=int, default=3, help="эпохи truncated BPTT")
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="learning rate Keras-оптимизатора truncated BPTT",
    )
    args = parser.parse_args()

    neuraltide.config.set_dtype(tf.float64)  # до создания моделей
    seed_everything(args.seed)

    problem, space = build_problem(args.steps)
    space.apply_to_model(THETA_START)
    phi0 = np.asarray(problem.pack(), dtype=np.float64)

    print("=" * 60)
    print("L2 long horizon: slow adaptation, gradient modes")
    print("=" * 60)
    print(
        f"steps: {args.steps} (T={args.steps * DT:.0f} ms), "
        f"tau*={THETA_STAR['exc/tau']} ms, params: {space.names}"
    )

    # --- Эталон: plain BPTT ---
    t0 = time.perf_counter()
    ev_plain, grad_plain = problem.value_and_gradient(phi0)
    wall_plain = time.perf_counter() - t0
    print("\n[plain BPTT]")
    print(f"  loss: {ev_plain.loss:.6g}   wall: {wall_plain:.2f}s")
    print(f"  n_forward: {problem.n_forward}  n_gradient: {problem.n_gradient}")

    # --- Checkpointed BPTT (точность vs plain — критерий приёмки) ---
    fwd0 = problem.n_forward
    t0 = time.perf_counter()
    ckpt = value_and_gradient_checkpointed(problem, phi0, args.segment)
    wall_ckpt = time.perf_counter() - t0
    grad_err = rel_err(ckpt.grad, grad_plain)
    print("\n[checkpointed BPTT]")
    print(f"  segments: {ckpt.segments} (len {args.segment})")
    print(f"  loss: {ckpt.loss:.6g}   wall: {wall_ckpt:.2f}s")
    print(f"  grad rel err vs plain: {grad_err:.3e}")
    print(f"  n_forward (delta): {problem.n_forward - fwd0}")
    mem = ckpt.peak_memory
    print(
        f"  peak memory (tracemalloc, lower bound): {mem / 1e6:.1f} MB"
        if mem is not None
        else "  n/a"
    )

    # --- Truncated BPTT (приближённый; free-run — обязательная проверка) ---
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    fwd0 = problem.n_forward
    t0 = time.perf_counter()
    tbptt = truncated_bptt(
        problem,
        phi0,
        optimizer,
        horizon=args.horizon,
        epochs=args.epochs,
    )
    wall_tbptt = time.perf_counter() - t0
    print("\n[truncated BPTT]")
    print(f"  window horizon: {args.horizon}, epochs: {args.epochs}")
    print(f"  final_loss (last window): {tbptt.final_loss:.6g}")
    print(f"  free_run_loss (full run): {tbptt.free_run_loss:.6g}")
    print(f"  wall: {wall_tbptt:.2f}s   n_forward (delta): {problem.n_forward - fwd0}")
    theta_fit = space.unpack(np.asarray(problem.pack(), dtype=np.float64))
    tau_err = abs(
        float(tf.reshape(theta_fit["exc/tau"], [-1])[0]) - THETA_STAR["exc/tau"]
    )
    print(f"  tau recovery after tbptt: |err|={tau_err:.3e}")

    print("\n[summary]")
    print(f"  ckpt grad rel err: {grad_err:.3e} (критерий приёмки: ~0)")
    print(
        f"  tbptt free_run vs plain loss: {tbptt.free_run_loss:.6g} vs "
        f"{ev_plain.loss:.6g} (стартовая точка общая)"
    )
    print(
        f"  total wall: {wall_plain + wall_ckpt + wall_tbptt:.2f}s, "
        f"total n_forward: {problem.n_forward}"
    )


if __name__ == "__main__":
    main()
