#!/usr/bin/env python
"""
Бенчмарк L3 (P2): масштабирование пакетной оценки кандидатов
(SerialPool vs ProcessPool).

Одинаковые кандидаты оцениваются дважды через neuraltide.optimization
.make_evaluator:

  1. n_workers=1 — SerialPool (родительский процесс);
  2. n_workers=2 — ProcessPool (spawn-процессы, каждый держит свою
     копию задачи; TF-граф строится в worker'е один раз).

Метрики (stdout, файлы не создаются):
  - эквивалентность потерь: max |loss_serial - loss_process| (один seed
    -> одинаковые per-candidate сиды -> потери совпадают);
  - wall time серийного и процессного прогонов по батчам;
  - speedup = t_serial / t_process суммарно и на «стабилизировавшихся»
    батчах (после первого — в нём амортизируется spawn + импорт TF и
    построение графа в worker'ах).

Честное ожидание: на мелких сетях и коротких горизонтах оценка кандидата
стоит миллисекунды, и процессный режим ПРОИГРЫВАЕТ серийному из-за
стартовых издержек (spawn-процессы заново импортируют TF). Выигрыш
появляется, когда одна оценка дорога (длинный горизонт, жёсткие
интеграторы с substepping, крупная сеть) и кандидатов много — увеличивайте
--t/--n-candidates. В пределе speedup стремится к числу ядер, но не выше
него; для CPU-bound TF-прогонов реальный потолок ниже из-за конкуренции
за intra-op потоки (по умолчанию каждый TF в worker'е видит все ядра).

Замечание о границах этапа: candidate-batch внутри прогона (векторная
оценка ансамбля состояний одним проходом NetworkRNN) — вне P2; здесь
параллелится только «один кандидат = один прогон».

Запуск:
    python benchmarks/l3_parallel_scaling.py
    python benchmarks/l3_parallel_scaling.py --help
    python benchmarks/l3_parallel_scaling.py --n-candidates 16 --t 1000
    python benchmarks/l3_parallel_scaling.py --no-process
"""

import argparse
import time

import numpy as np

DT = 0.5
# theta* для синтетической E–I сети (постоянные времени обучаемы — та же
# постановка, что в example_09; сеть малая, поэтому для заметного эффекта
# масштабируйте горизонт --t и число кандидатов).
THETA_STAR = {
    "exc/tau": 8.0,
    "inh/tau": 12.0,
    "in->exc/gsyn_max": 30.0,
    "inh->exc/gsyn_max": 10.0,
}


def build_problem(t_steps: int = 300):
    """
    Фабрика уровня модуля (picklable) — строит задачу калибровки E–I сети.

    Вызывается в родителе (SerialPool) и в каждом spawn-worker'е
    (ProcessPool); TF-граф строится при вызове внутри того процесса,
    который её оценивает.
    """
    import tensorflow as tf

    import neuraltide.config

    neuraltide.config.set_dtype(tf.float64)
    from neuraltide.core.network import NetworkGraph, NetworkRNN
    from neuraltide.inputs import SinusoidalGenerator
    from neuraltide.integrators import EulerIntegrator
    from neuraltide.optimization import (
        CalibrationProblem,
        ObservationSpec,
        ParameterSpace,
        ResidualLoss,
    )
    from neuraltide.populations import WilsonCowan
    from neuraltide.synapses import StaticSynapse

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
            "I_ext": 0.8,
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
            "gsyn_max": [[THETA_STAR["in->exc/gsyn_max"]]],
            "pconn": [[1.0]],
            "e_r": 1.0,
        },
        name="in->exc",
    )
    syn_ie = StaticSynapse(
        n_pre=1,
        n_post=1,
        dt=DT,
        params={
            "gsyn_max": [[THETA_STAR["inh->exc/gsyn_max"]]],
            "pconn": [[1.0]],
            "e_r": -0.5,
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
    graph.add_synapse("inh->exc", syn_ie, src="inh", tgt="exc")
    network = NetworkRNN(graph, EulerIntegrator())

    t_seq = tf.constant((np.arange(t_steps, dtype=np.float64) * DT)[None, :, None])
    inputs = graph.pack_inputs({"in": gen(t_seq)})
    target = network(t_seq, inputs=inputs).firing_rates
    observations = [
        ObservationSpec(name=name, population=name, target=target[name], sigma=1.0)
        for name in ("exc", "inh")
    ]
    return CalibrationProblem(
        network,
        ParameterSpace.from_network(network),
        t_seq,
        inputs=inputs,
        objective=ResidualLoss(observations),
    )


def make_candidates(phi0: np.ndarray, n: int, seed: int) -> list:
    """Детерминированные кандидаты в окрестности phi0 (часть — заведомо плохие)."""
    rng = np.random.default_rng(seed)
    scale = 0.1 * np.abs(phi0) + 0.05
    return [
        np.asarray(phi0 + rng.standard_normal(phi0.size) * scale, dtype=np.float64)
        for _ in range(n)
    ]


def run_batches(evaluate, candidates, batches: int) -> tuple:
    """Оценивает candidates batches раз; возвращает (losses_первого_батча, [wall_i])."""
    walls = []
    losses = None
    for _ in range(batches):
        t0 = time.perf_counter()
        losses = list(evaluate(candidates))
        walls.append(time.perf_counter() - t0)
    return losses, walls


def main() -> None:
    parser = argparse.ArgumentParser(
        description="L3 benchmark: candidate evaluation scaling "
        "(SerialPool vs ProcessPool)."
    )
    parser.add_argument(
        "--seed", type=int, default=11, help="сиды кандидатов и методов"
    )
    parser.add_argument(
        "--n-candidates", type=int, default=8, help="число кандидатов в батче"
    )
    parser.add_argument(
        "--batches",
        type=int,
        default=3,
        help="число повторных батчей (амортизация пула; первый батч — cold start)",
    )
    parser.add_argument(
        "--t",
        type=int,
        default=300,
        help="горизонт прогона в шагах dt (стоимость одной оценки)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="число worker-процессов ProcessPool (>= 2 включает процессный режим)",
    )
    parser.add_argument(
        "--no-process",
        action="store_true",
        help="пропустить процессный режим (только SerialPool)",
    )
    args = parser.parse_args()

    # Родитель строит задачу один раз — для phi0 и эталонных потерь.
    problem = build_problem(t_steps=args.t)
    phi0 = np.asarray(problem.pack(), dtype=np.float64)
    candidates = make_candidates(phi0, args.n_candidates, args.seed)

    print("=" * 64)
    print(
        f"L3 parallel scaling: {args.n_candidates} candidates x "
        f"{args.batches} batches, T={args.t} steps (dt={DT} ms)"
    )
    print(f"phi size {problem.size}, workers={args.workers}, seed={args.seed}")
    print("=" * 64)

    from neuraltide.optimization import make_evaluator

    serial_eval = make_evaluator(build_problem, n_workers=1, seed=args.seed)
    s_losses, s_walls = run_batches(serial_eval, candidates, args.batches)
    serial_eval.close()
    print("\n[SerialPool n_workers=1]")
    for i, w in enumerate(s_walls):
        print(f"  batch {i}: {w:.3f}s")
    print(f"  total: {sum(s_walls):.3f}s")

    table = [("serial(1w)", s_walls)]

    if not args.no_process and int(args.workers) >= 2:
        try:
            proc_eval = make_evaluator(
                build_problem, n_workers=args.workers, seed=args.seed
            )
            try:
                p_losses, p_walls = run_batches(proc_eval, candidates, args.batches)
            finally:
                proc_eval.close()
        except Exception as exc:  # process-safe: не роняем бенчмарк
            print(
                f"\n[ProcessPool n_workers={args.workers}] пропущен: "
                f"{type(exc).__name__}: {exc}"
            )
        else:
            print(f"\n[ProcessPool n_workers={args.workers}]")
            for i, w in enumerate(p_walls):
                print(f"  batch {i}: {w:.3f}s")
            print(f"  total: {sum(p_walls):.3f}s")

            dev = max(abs(float(a) - float(b)) for a, b in zip(s_losses, p_losses))
            print("\n[equivalence]")
            print(f"  max |loss_serial - loss_process|: {dev:.3e}")
            print(f"  losses match: {dev <= 1e-9}")

            speed_total = sum(s_walls) / max(sum(p_walls), 1e-12)
            steady_s = sum(s_walls[1:]) / max(len(s_walls) - 1, 1)
            steady_p = sum(p_walls[1:]) / max(len(p_walls) - 1, 1)
            speed_steady = steady_s / max(steady_p, 1e-12)
            print("\n[speedup]")
            print(f"  total (cold start included): {speed_total:.2f}x")
            print(f"  steady-state (batch>=1):     {speed_steady:.2f}x")
            table.append((f"process({args.workers}w)", p_walls))

            print("\n[вывод]")
            print(
                "  Суммарно (с cold start) процессный режим на этом бюджете "
                f"{'проигрывает' if speed_total < 1.0 else 'выигрывает'} "
                f"({speed_total:.2f}x): первый батч оплачивает spawn-процессы, "
                "повторный импорт TF и построение графа в каждом worker'е."
            )
            if speed_steady < 1.0:
                print(
                    "  Даже на стабилизировавшихся батчах ускорения нет: одна"
                    " оценка слишком дешёвая, и накладные расходы (pickle"
                    " задач/результатов, планирование пула, конкуренция за"
                    " CPU между TF-процессами) не окупаются. Увеличьте"
                    " --t/--n-candidates: speedup растёт с ростом стоимости"
                    " одной оценки и в пределе стремится к числу ядер (но не"
                    " превышает его)."
                )
            else:
                print(
                    "  На стабилизировавшихся батчах ускорение "
                    f"{speed_steady:.2f}x — выигрыш реален, когда пул живёт"
                    " между батчами (make_evaluator) и одна оценка дорогая."
                    " Потолок — число ядер; для CPU-bound TF-прогонов предел"
                    " ниже из-за конкуренции за intra-op потоки (каждый TF в"
                    " worker'е по умолчанию видит все ядра). Тензорный"
                    " candidate-batch (векторная оценка ансамбля одним"
                    " прогоном NetworkRNN) — вне P2; здесь параллелится"
                    " только «один кандидат = один прогон»."
                )

    if len(table) > 1:
        print("\n[comparison] wall per batch, s")
        header = f"  {'mode':<18}" + "".join(
            f"  {'b' + str(i):>8}" for i in range(args.batches)
        )
        print(header)
        print("  " + "-" * (len(header) - 2))
        for name, walls in table:
            print(f"  {name:<18}" + "".join(f"  {w:>8.3f}" for w in walls))

    print(
        "\n[note] candidate-batch (batch>1 через NetworkRNN в одном прогоне)"
        " вне этапа P2 — см. docs/optimization_roadmap.md."
    )


if __name__ == "__main__":
    main()
