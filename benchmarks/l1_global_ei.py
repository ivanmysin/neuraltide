#!/usr/bin/env python
"""
Бенчмарк L1 (P2): глобальный поиск на малой E–I сети.

Сравнение трёх стратегий на одной и той же задаче калибровки
(6 обучаемых параметров, все с конечными границами — обязательное
условие для DE/CMA-ES; у WilsonCowan exc/I_ext min/max заданы явно):

  1. multistart (P1, baseline): Sobol-дизайн -> screening -> доводка
     top-k через least_squares (TRF);
  2. DE (global_search, scipy differential_evolution, бюджет max_evals)
     -> доводка refine_top_k через TRF;
  3. CMA-ES (global_search, optional extra 'global') -> доводка TRF;
     пропускается с подсказкой по установке, если пакет cma отсутствует.

Каждый метод получает заново построенную задачу (build_problem) и один
seed, чтобы прогоны были независимы и сопоставимы по счётчикам.

Метрики (stdout, файлы не создаются):
  - best loss и признак достижения порога --loss-tol;
  - доля дошедших до доводки кандидатов, достигших порога;
  - n_forward (счётчик задачи), wall time, ошибка восстановления theta.

Запуск:
    python benchmarks/l1_global_ei.py
    python benchmarks/l1_global_ei.py --help
    python benchmarks/l1_global_ei.py --max-evals 2000 --seed 3
"""

import argparse
import importlib.util
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
from neuraltide.optimization.global_search import global_search
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
    """
    E–I сеть и задача калибровки; цель — прогон при theta*.

    Отличие от l1_multistart_ei.py: exc/I_ext обучаем с конечными
    min/max — без этого DE/CMA-ES (bounds обязательны) не применимы.
    """
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
            "I_ext": {
                "value": [THETA_STAR["exc/I_ext"]],
                "trainable": True,
                "min": 0.0,
                "max": 5.0,
            },
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


def _metrics(result, problem, space, loss_tol, refined, n_refined, wall):
    """Единый словарь метрик для сравнения методов."""
    best = float(result.base.loss_best)
    reached_refined = [c for c in refined if np.isfinite(c.loss) and c.loss <= loss_tol]
    return {
        "best_loss": best,
        "reached_tol": best <= loss_tol,
        "frac_to_tol": (len(reached_refined) / max(1, n_refined)),
        "n_refined": n_refined,
        "n_forward": int(problem.n_forward),
        "wall": wall,
        "theta_err": theta_max_err(space, result.base.phi_best),
    }


def run_multistart_baseline(args):
    problem, space = build_problem()
    t_begin = time.perf_counter()
    result = multistart(
        problem,
        n_starts=args.n_starts,
        method="sobol",
        refine_top_k=args.refine_top_k,
        local="least_squares",
        seed=args.seed,
    )
    wall = time.perf_counter() - t_begin
    refined = [s for s in result.solutions if s.refined and np.isfinite(s.loss)]
    metrics = _metrics(
        result, problem, space, args.loss_tol, refined, len(refined), wall
    )
    metrics["extra"] = (
        f"starts={result.n_starts} clusters={result.clustering['n_clusters']} "
        f"failed={result.n_failed}"
    )
    return metrics


def run_global(method, args, options):
    problem, space = build_problem()
    t_begin = time.perf_counter()
    result = global_search(
        problem,
        method=method,
        max_evals=args.max_evals,
        seed=args.seed,
        refine_top_k=args.refine_top_k,
        local="least_squares",
        options=options,
    )
    wall = time.perf_counter() - t_begin
    refined = [c for c in result.candidates if c.refined and np.isfinite(c.loss)]
    metrics = _metrics(
        result, problem, space, args.loss_tol, refined, len(refined), wall
    )
    metrics["extra"] = (
        f"global_evals={result.n_evals} "
        f"refined_ok={sum(1 for c in refined if c.status == 'ok')} "
        f"message={result.message!r}"
    )
    return metrics


def print_metrics(name, m) -> None:
    print(f"\n[{name}]")
    print(f"  best loss:          {m['best_loss']:.6g}")
    print(f"  reached tol:        {m['reached_tol']}")
    print(f"  frac to tol:        {m['frac_to_tol']:.2f} ({m['n_refined']} refined)")
    print(f"  theta err (best):   {m['theta_err']:.3e}")
    print(f"  n_forward:          {m['n_forward']}")
    print(f"  wall time:          {m['wall']:.2f}s")
    print(f"  {m['extra']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="L1 benchmark: global search (multistart vs DE vs CMA-ES)."
    )
    parser.add_argument("--seed", type=int, default=11, help="seed дизайнов и методов")
    parser.add_argument(
        "--max-evals",
        type=int,
        default=1500,
        help="бюджет max_evals глобальной фазы DE/CMA-ES",
    )
    parser.add_argument(
        "--loss-tol",
        type=float,
        default=1e-6,
        help="порог потери для 'достигнуто'",
    )
    parser.add_argument(
        "--refine-top-k",
        type=int,
        default=3,
        help="сколько лучших кандидатов доводить локально (TRF)",
    )
    parser.add_argument(
        "--n-starts",
        type=int,
        default=12,
        help="число стартов baseline-мультистарта",
    )
    parser.add_argument(
        "--popsize",
        type=int,
        default=12,
        help="popsize DE / popsize CMA-ES",
    )
    parser.add_argument(
        "--sigma0",
        type=float,
        default=None,
        help="начальная sigma CMA-ES (None — авто от границ)",
    )
    parser.add_argument(
        "--no-multistart", action="store_true", help="пропустить baseline"
    )
    parser.add_argument("--no-de", action="store_true", help="пропустить DE")
    parser.add_argument("--no-cma", action="store_true", help="пропустить CMA-ES")
    args = parser.parse_args()

    neuraltide.config.set_dtype(tf.float64)  # до создания моделей
    print("=" * 64)
    print(f"L1 global search E-I: 2 populations, {len(THETA_STAR)} trainable")
    print("=" * 64)
    print(f"theta*: {THETA_STAR}")
    print(f"seed={args.seed} max_evals={args.max_evals} loss_tol={args.loss_tol}")

    table = []

    if not args.no_multistart:
        seed_everything(args.seed)
        metrics = run_multistart_baseline(args)
        print_metrics("multistart baseline (P1)", metrics)
        table.append(("multistart", metrics))

    if not args.no_de:
        seed_everything(args.seed)
        metrics = run_global(
            "de", args, {"popsize": args.popsize, "polish": False, "tol": 0.01}
        )
        print_metrics(f"DE (max_evals={args.max_evals})", metrics)
        table.append(("de", metrics))

    cma_missing = importlib.util.find_spec("cma") is None
    if not args.no_cma and not cma_missing:
        options = {"popsize": args.popsize}
        if args.sigma0 is not None:
            options["sigma0"] = args.sigma0
        seed_everything(args.seed)
        metrics = run_global("cma", args, options)
        print_metrics(f"CMA-ES (max_evals={args.max_evals})", metrics)
        table.append(("cma-es", metrics))
    elif not args.no_cma and cma_missing:
        print(
            "\n[CMA-ES] пропущен: пакет 'cma' не установлен; "
            "установите extra: pip install 'neuraltide[global]'"
        )

    if table:
        print("\n[comparison]")
        header = (
            f"  {'method':<12} {'best loss':>12} {'reached':>8} "
            f"{'frac':>6} {'n_forward':>10} {'wall':>8} {'theta err':>10}"
        )
        print(header)
        print("  " + "-" * (len(header) - 2))
        for name, m in table:
            print(
                f"  {name:<12} {m['best_loss']:>12.4g} "
                f"{str(m['reached_tol']):>8} {m['frac_to_tol']:>6.2f} "
                f"{m['n_forward']:>10} {m['wall']:>7.1f}s "
                f"{m['theta_err']:>10.2e}"
            )


if __name__ == "__main__":
    main()
