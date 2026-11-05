#!/usr/bin/env python
"""
Бенчмарк L1: мультистарт на малой E–I сети (10–30 параметров в оригинале
лестницы; здесь 6 для приемлемого времени на CPU).

Сеть: две популяции WilsonCowan (exc, inh), внешний вход на exc,
синапсы in->exc (возбуждающий), exc->inh (возбуждающий), inh->exc
(тормозный, e_r < 0). Целевая траектория — обе частоты разрядов при
известных theta*. Мультистарт P1: детерминированный Sobol-дизайн ->
screening -> локальная доводка top-k через least_squares (TRF).

Метрики (stdout, файлы не создаются):
  - доля стартов, прошедших screening, и доля дошедших до доводки;
  - доля доведённых стартов, достигших порога потери (--loss-tol);
  - число отдельных кластеров решений (greedy dedup);
  - n_forward (общий счётчик задачи), wall time;
  - max ошибка восстановления theta по лучшему старту.

Запуск:
    python benchmarks/l1_multistart_ei.py
    python benchmarks/l1_multistart_ei.py --help
    python benchmarks/l1_multistart_ei.py --n-starts 16 --refine-top-k 4
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
    multistart,
)
from neuraltide.populations import WilsonCowan
from neuraltide.synapses import StaticSynapse
from neuraltide.utils import seed_everything

DT = 0.5
T = 120  # мс

THETA_STAR = {
    "exc/tau": 8.0,
    "exc/I_ext": 0.8,
    "inh/tau": 12.0,
    "in->exc/gsyn_max": 30.0,
    "exc->inh/gsyn_max": 20.0,
    "inh->exc/gsyn_max": 10.0,
}


def build_problem():
    """E–I сеть и задача калибровки; цель — прогон при theta*."""
    exc = WilsonCowan(
        dt=DT,
        params={
            "tau": {
                "value": [THETA_STAR["exc/tau"]],
                "trainable": True,
                "min": 1.0,
                "max": 50.0,
            },
            "a": 1.0,
            "theta": 0.0,
            "w": 0.2,
            "I_ext": {"value": [THETA_STAR["exc/I_ext"]], "trainable": True},
            "max_rate": 100.0,
        },
        name="exc",
    )
    inh = WilsonCowan(
        dt=DT,
        params={
            "tau": {
                "value": [THETA_STAR["inh/tau"]],
                "trainable": True,
                "min": 1.0,
                "max": 50.0,
            },
            "a": 1.0,
            "theta": 0.0,
            "w": 0.1,
            "I_ext": 0.3,
            "max_rate": 100.0,
        },
        name="inh",
    )
    syn_in = StaticSynapse(
        n_pre=1,
        n_post=1,
        dt=DT,
        params={
            "gsyn_max": {
                "value": [[THETA_STAR["in->exc/gsyn_max"]]],
                "trainable": True,
                "min": 1.0,
                "max": 200.0,
            },
            "pconn": [[1.0]],
            "e_r": 1.0,
        },
        name="in->exc",
    )
    syn_ei = StaticSynapse(
        n_pre=1,
        n_post=1,
        dt=DT,
        params={
            "gsyn_max": {
                "value": [[THETA_STAR["exc->inh/gsyn_max"]]],
                "trainable": True,
                "min": 1.0,
                "max": 200.0,
            },
            "pconn": [[1.0]],
            "e_r": 1.0,
        },
        name="exc->inh",
    )
    syn_ie = StaticSynapse(
        n_pre=1,
        n_post=1,
        dt=DT,
        params={
            "gsyn_max": {
                "value": [[THETA_STAR["inh->exc/gsyn_max"]]],
                "trainable": True,
                "min": 1.0,
                "max": 200.0,
            },
            "pconn": [[1.0]],
            "e_r": -0.5,  # тормозный
        },
        name="inh->exc",
    )
    gen = SinusoidalGenerator(
        dt=DT,
        params={"amplitude": 8.0, "freq": 8.0, "phase": 0.0, "offset": 10.0},
    )
    graph = NetworkGraph(dt=DT)
    graph.declare_input("in", n_units=1)
    graph.add_population("exc", exc)
    graph.add_population("inh", inh)
    graph.add_synapse("in->exc", syn_in, src="in", tgt="exc")
    graph.add_synapse("exc->inh", syn_ei, src="exc", tgt="inh")
    graph.add_synapse("inh->exc", syn_ie, src="inh", tgt="exc")
    network = NetworkRNN(graph, EulerIntegrator())

    t_seq = tf.constant((np.arange(T, dtype=np.float64) * DT)[None, :, None])
    inputs = graph.pack_inputs({"in": gen(t_seq)})
    out = network(t_seq, inputs=inputs)
    observations = [
        ObservationSpec(
            name="exc", population="exc", target=out.firing_rates["exc"], sigma=2.0
        ),
        ObservationSpec(
            name="inh", population="inh", target=out.firing_rates["inh"], sigma=2.0
        ),
    ]
    space = ParameterSpace.from_network(
        network,
        transforms={
            name: LogTransform()
            for name in ("in->exc/gsyn_max", "exc->inh/gsyn_max", "inh->exc/gsyn_max")
        },
    )
    problem = CalibrationProblem(
        network,
        space,
        t_seq,
        inputs=inputs,
        objective=ResidualLoss(observations),
    )
    return problem, space


def theta_max_err(space, phi) -> float:
    theta = space.unpack(np.asarray(phi, dtype=np.float64))
    return max(
        abs(float(tf.reshape(theta[name], [-1])[0]) - star)
        for name, star in THETA_STAR.items()
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="L1 benchmark: multistart on a small E-I network."
    )
    parser.add_argument(
        "--seed", type=int, default=11, help="seed дизайна начальных точек"
    )
    parser.add_argument(
        "--n-starts", type=int, default=16, help="число стартов мультистарта"
    )
    parser.add_argument(
        "--refine-top-k",
        type=int,
        default=4,
        help="сколько лучших стартов доводить локально",
    )
    parser.add_argument(
        "--loss-tol",
        type=float,
        default=1e-6,
        help="порог потери для 'достигнуто' на доводке",
    )
    parser.add_argument(
        "--method",
        default="sobol",
        choices=["sobol", "lhs", "random"],
        help="метод дизайна начальных точек",
    )
    args = parser.parse_args()

    neuraltide.config.set_dtype(tf.float64)  # до создания моделей
    seed_everything(args.seed)

    problem, space = build_problem()
    print("=" * 60)
    print(f"L1 multistart E-I: 2 populations, {len(THETA_STAR)} trainable params")
    print("=" * 60)
    print(f"theta*: {THETA_STAR}")

    t_begin = time.perf_counter()
    ms = multistart(
        problem,
        n_starts=args.n_starts,
        method=args.method,
        refine_top_k=args.refine_top_k,
        local="least_squares",
        seed=args.seed,
    )
    wall = time.perf_counter() - t_begin

    screened = [s for s in ms.solutions if s.screened_loss is not None]
    refined_ok = [s for s in ms.solutions if s.refined and np.isfinite(s.loss)]
    reached = [s for s in refined_ok if s.loss <= args.loss_tol]

    print("\n[multistart metrics]")
    print(f"  starts:                {ms.n_starts}")
    print(
        f"  screened:              {len(screened)} ({len(screened) / ms.n_starts:.0%})"
    )
    print(f"  refined:               {ms.n_refined}")
    print(f"  refined & ok:          {len(refined_ok)}")
    print(
        f"  reached loss<={args.loss_tol:g}: {len(reached)} "
        f"({len(reached) / max(1, ms.n_refined):.0%} of refined)"
    )
    print(f"  failed starts:         {ms.n_failed}")
    print(f"  clusters (dedup):      {ms.clustering['n_clusters']}")
    print(f"  best loss:             {ms.base.loss_best:.6g}")
    print(f"  theta err (best):      {theta_max_err(space, ms.base.phi_best):.3e}")
    print(f"  n_forward (problem):   {problem.n_forward}")
    print(f"  wall time:             {wall:.2f}s")
    losses = sorted(float(s.loss) for s in ms.solutions if np.isfinite(s.loss))
    print(
        f"  solution loss quartiles: {losses[0]:.3g} / "
        f"{losses[len(losses) // 2]:.3g} / {losses[-1]:.3g}"
    )


if __name__ == "__main__":
    main()
