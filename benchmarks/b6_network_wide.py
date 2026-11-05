#!/usr/bin/env python
"""
Бенчмарк В6: network-wide интегрирование связанной сети
(integration_mode='network').

Задача: E–E сеть Уилсона–Коуэна с рекуррентным синапсом Цодыкса–Маркрама
(автономная, e_r != 0 — синапс реально влияет на динамику). Сравниваются
два режима интегрирования одной и той же сети RK4:

  'split' (по умолчанию): синапсы и популяции шагают отдельно — ток
      синапса берётся по обновлённому состоянию, частота пресинаптического
      юнита — по состоянию на начало шага;
  'network': единый шаг RK4 по всей связанной сети (step_network поверх
      _JointNetworkRHS) — все компоненты согласованы внутри шага.

Метрики (stdout, файлы не создаются):
  - порядок сходимости по dt (референс dt/64);
  - ошибка |E(T) - E_ref| при рабочем шаге --dt;
  - wall time прогона после разогрева (первая вызов трасcирует граф).

Честное ожидание: network-wide RK4 сохраняет порядок 4 на связанной сети,
а split-RK4 падает до ~2 из-за расщепления. Плата за network-wide —
более дорогой шаг (полный joint RHS на каждой стадии RK4 вместо отдельных
шагов синапсов и популяций), поэтому wall time выше; выигрыш в ошибке при
том же dt — на порядки, и он растёт с горизонтом и шагом.

Запуск:
    python benchmarks/b6_network_wide.py
    python benchmarks/b6_network_wide.py --help
    python benchmarks/b6_network_wide.py --dt 0.25 --t-end 40
"""

import argparse
import time

import numpy as np
import tensorflow as tf

import neuraltide.config
from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.integrators import RK4Integrator
from neuraltide.populations import WilsonCowan
from neuraltide.synapses import TsodyksMarkramSynapse
from neuraltide.utils import seed_everything


def _t_seq(n, dt):
    return tf.constant((np.arange(n, dtype=np.float64) * dt)[None, :, None])


def build_network(dt, mode):
    """E–E сеть Уилсона–Коуэна с рекуррентным синапсом Цодыкса–Маркрама."""
    pop = WilsonCowan(
        dt=dt,
        params={
            "tau": [8.0],
            "a": 1.0,
            "theta": 0.0,
            "w": 0.2,
            "I_ext": [0.8],
            "max_rate": 100.0,
        },
        name="exc",
    )
    syn = TsodyksMarkramSynapse(
        n_pre=1,
        n_post=1,
        dt=dt,
        params={
            "gsyn_max": [[0.5]],
            "tau_f": [[500.0]],
            "tau_d": [[200.0]],
            "tau_r": [[50.0]],
            "Uinc": [[0.2]],
            "pconn": [[1.0]],
            "e_r": 1.0,
        },
        name="exc->exc",
    )
    graph = NetworkGraph(dt=dt)
    graph.add_population("exc", pop)
    graph.add_synapse("exc->exc", syn, src="exc", tgt="exc")
    return NetworkRNN(graph, RK4Integrator(), integration_mode=mode)


def run_once(dt, mode, t_end):
    """Прогон сети: возвращает (E в момент t_end, wall time после разогрева)."""
    network = build_network(dt, mode)
    t_seq = _t_seq(int(round(t_end / dt)), dt)
    network(t_seq)  # разогрев: трассировка @tf.function
    t0 = time.perf_counter()
    out = network(t_seq)
    wall = time.perf_counter() - t0
    return float(np.asarray(out.final_state[0][0]).ravel()[0]), wall


def main() -> None:
    parser = argparse.ArgumentParser(
        description="B6 benchmark: RK4 split vs network-wide на связанной TM-сети."
    )
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--dt", type=float, default=0.5, help="рабочий шаг, мс")
    parser.add_argument("--t-end", type=float, default=20.0, help="горизонт, мс")
    parser.add_argument(
        "--ref-div",
        type=int,
        default=64,
        help="делитель референсного шага (ref dt = --dt / ref_div)",
    )
    args = parser.parse_args()

    neuraltide.config.set_dtype(tf.float64)  # до создания моделей
    seed_everything(args.seed)

    dt = args.dt
    t_end = args.t_end
    ref_dt = dt / args.ref_div
    dts = (dt, dt / 2, dt / 4)

    print("=" * 72)
    print("B6: RK4 split vs network-wide на связанной TM-сети (E–E, STP)")
    print("=" * 72)
    print(
        f"dt={dt} мс, горизонт T={t_end:g} мс, референс dt={ref_dt:g} мс "
        f"(network-wide, тот же seed)"
    )

    results = {}
    for mode in ("split", "network"):
        e_ref, _ = run_once(ref_dt, mode, t_end)
        errs = []
        walls = []
        for d in dts:
            e, w = run_once(d, mode, t_end)
            errs.append(abs(e - e_ref))
            walls.append(w)
        orders = [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
        results[mode] = {
            "e_ref": e_ref,
            "errs": errs,
            "orders": orders,
            "wall": walls[0],  # wall при рабочем шаге --dt
        }

    print()
    print(f"{'mode':<10}{'order(dt->dt/2)':>17}{'order(dt/2->dt/4)':>19}")
    print("-" * 46)
    for mode in ("split", "network"):
        r = results[mode]
        print(f"{mode:<10}{r['orders'][0]:>17.2f}{r['orders'][1]:>19.2f}")

    print()
    print(
        f"{'mode':<10}{'err@dt':>14}{'err@dt/2':>14}{'err@dt/4':>14}{'wall,s@dt':>12}"
    )
    print("-" * 64)
    for mode in ("split", "network"):
        r = results[mode]
        print(
            f"{mode:<10}{r['errs'][0]:>14.3e}{r['errs'][1]:>14.3e}"
            f"{r['errs'][2]:>14.3e}{r['wall']:>12.3f}"
        )

    split, net = results["split"], results["network"]
    order_split = split["orders"][-1]
    order_net = net["orders"][-1]
    err_ratio = (
        split["errs"][0] / net["errs"][0] if net["errs"][0] > 0 else float("inf")
    )
    wall_ratio = net["wall"] / split["wall"] if split["wall"] > 0 else float("inf")

    print()
    print("[заключение]")
    if 1.5 < order_split < 2.6 and order_net > 3.5:
        print(
            f"  * Подтверждено: split-RK4 на связанной сети даёт порядок "
            f"{order_split:.2f} (~2, расщеплённый шаг), network-wide — "
            f"{order_net:.2f} (> 3.5, согласованный шаг)."
        )
    else:
        print(
            f"  * Ожидание не подтверждено: split order={order_split:.2f} "
            f"(~2), network order={order_net:.2f} (> 3.5) — проверьте сеть/шаг."
        )
    print(
        f"  * При том же dt={dt:g} мс network-wide точнее split в "
        f"~{err_ratio:.1f} раза по |E(T) - E_ref|."
    )
    if wall_ratio > 1.05:
        print(
            f"  * Цена: network-wide шаг дороже в ~{wall_ratio:.2f} раза по "
            "wall time (полный joint RHS на каждой стадии RK4 вместо "
            "отдельных шагов синапсов и популяций)."
        )
    elif wall_ratio < 0.95:
        print(
            f"  * Wall time: network-wide ~{wall_ratio:.2f}× от split — на "
            "мелкой сети доминирует tf/Python overhead, а не число вызовов "
            "правой части; на крупных сетях полный joint RHS на каждой "
            "стадии RK4 ожидаемо дороже отдельных шагов split."
        )
    else:
        print("  * Wall time режимов практически совпадает на этой сети.")
    print(
        "  * Практический вывод: network-wide выигрывает, когда важна "
        "ошибка при заданном dt (длинный горизонт, жёсткая связность, "
        "высокая точность); split остаётся дешевле, когда порядка 2 "
        "достаточно."
    )


if __name__ == "__main__":
    main()
