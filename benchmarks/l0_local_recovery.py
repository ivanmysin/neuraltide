#!/usr/bin/env python
"""
Бенчмарк L0: точное локальное восстановление параметров малой сети.

Задача (дорожная карта, раздел 9.1): одна популяция WilsonCowan, один
внешний вход через StaticSynapse, 3 оптимизируемых параметра
(exc/tau, exc/I_ext, in->exc/gsyn_max). Целевая траектория снята с известных
theta* (опционально зашумлена), стартовая точка сдвинута в 2.5-4 раза.
Восстановление выполняется двумя локальными адаптерами P1:
least_squares (TRF, честный residual-интерфейс) и minimize_lbfgsb
(скалярная потеря + градиент).

Метрики (печатаются в stdout, файлы не создаются):
  - loss до/после и success по каждому методу;
  - max ошибка восстановления theta (по всем записям ParameterSpace);
  - n_forward / n_gradient / wall time (дельта счётчиков задачи);
  - число координат, прижатых к границе (bounds_active).

Запуск:
    python benchmarks/l0_local_recovery.py                 # полный прогон
    python benchmarks/l0_local_recovery.py --help          # справка
    python benchmarks/l0_local_recovery.py --noise-sigma 1.0 --seed 3
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
    least_squares,
    minimize_lbfgsb,
)
from neuraltide.populations import WilsonCowan
from neuraltide.synapses import StaticSynapse
from neuraltide.utils import seed_everything

DT = 0.5
T = 80  # мс

THETA_STAR = {
    "exc/tau": 8.0,
    "exc/I_ext": 0.8,
    "in->exc/gsyn_max": 30.0,
}
# Стартовая точка: сдвиг в 2.5-4 раза от theta*.
THETA_START = {
    "exc/tau": 20.0,
    "exc/I_ext": 0.2,
    "in->exc/gsyn_max": 120.0,
}


def build_problem(noise_sigma: float, seed: int):
    """L0-сеть и задача калибровки; цель — прогон при theta* (+ шум)."""
    pop = WilsonCowan(
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
            "e_r": 1.0,  # иначе (e_r - v_mean) = 0 и вход молчит
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

    t_seq = tf.constant((np.arange(T, dtype=np.float64) * DT)[None, :, None])
    inputs = graph.pack_inputs({"in": gen(t_seq)})
    target = network(t_seq, inputs=inputs).firing_rates["exc"]
    if noise_sigma > 0.0:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, noise_sigma, size=np.asarray(target).shape)
        target = tf.constant(np.asarray(target) + noise, dtype=target.dtype)

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


def theta_max_err(space, phi) -> float:
    theta = space.unpack(np.asarray(phi, dtype=np.float64))
    return max(
        abs(float(tf.reshape(theta[name], [-1])[0]) - star)
        for name, star in THETA_STAR.items()
    )


def report(method: str, res, space) -> None:
    err = theta_max_err(space, res.base.phi_best)
    print(f"[{method}]")
    print(f"  success:    {res.success} (status {res.status})")
    print(f"  loss:       {res.base.loss_best:.6g}")
    print(f"  theta err:  {err:.3e} (max over params)")
    print(f"  n_forward:  {res.n_forward}   n_gradient: {res.n_gradient}")
    print(f"  wall time:  {res.wall_time:.2f}s")
    if res.bounds_active:
        print(f"  bounds active: {res.bounds_active}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="L0 benchmark: exact local recovery on a tiny network."
    )
    parser.add_argument("--seed", type=int, default=0, help="seed шума цели")
    parser.add_argument(
        "--noise-sigma",
        type=float,
        default=0.0,
        help="стандартное отклонение гауссова шума цели (Гц), по умолчанию 0",
    )
    parser.add_argument(
        "--max-nfev",
        type=int,
        default=None,
        help="бюджет оценок невязок для TRF (None — без ограничения)",
    )
    args = parser.parse_args()

    neuraltide.config.set_dtype(tf.float64)  # до создания моделей
    seed_everything(args.seed)
    t_begin = time.perf_counter()

    problem, space = build_problem(args.noise_sigma, args.seed)
    space.apply_to_model(THETA_START)

    print("=" * 60)
    print("L0 local recovery: 1 population, 3 trainable params")
    print("=" * 60)
    print(f"theta*: {THETA_STAR}")
    print(f"start:  {THETA_START}  (noise sigma={args.noise_sigma})")
    ev0 = problem.evaluate()
    print(f"loss before: {ev0.loss:.6g}\n")

    res_trf = least_squares(problem, problem.pack(), max_nfev=args.max_nfev)
    report("least_squares TRF", res_trf, space)

    # Сброс к той же стартовой точке для честного сравнения методов.
    space.apply_to_model(THETA_START)
    res_bfgs = minimize_lbfgsb(problem, problem.pack())
    report("L-BFGS-B", res_bfgs, space)

    fwd = problem.n_forward
    print(
        f"[summary] total n_forward: {fwd}, wall: {time.perf_counter() - t_begin:.2f}s"
    )
    ok = res_trf.success and res_bfgs.success
    print(
        f"[summary] both succeeded: {ok}; "
        f"TRF theta err: {theta_max_err(space, res_trf.base.phi_best):.3e}, "
        f"L-BFGS-B theta err: {theta_max_err(space, res_bfgs.base.phi_best):.3e}"
    )


if __name__ == "__main__":
    main()
