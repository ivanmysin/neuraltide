#!/usr/bin/env python
"""
Бенчмарк L2: P2-интеграторы на малой жёсткой задаче.

Задача: линейная система y' = A y, A = [[-1/eps, 1], [0, -1]] —
классическая жёсткая модель: быстрая компонента λ_fast ≈ -1/eps,
медленная λ_slow = -1 (stiffness ratio ≈ 1/eps). При dt=0.5, eps=1e-3
явный шаг по быстрой компоненте z = -500 далеко за пределами области
устойчивости RK4 (|z| <= 2.78).

Сравниваются: RK4 (явный эталон), AdaptiveRK 3(2) ( Bogacki–Shampine,
PI-контроллер), IMEX-Euler (популяция даёт derivatives_split +
linear_operator), Rosenbrock ROS2 (JVP-якобиан), ExponentialEuler
(JVP-якобиан, augmented expm). Плюс честный контроль: IMEX на той же
системе БЕЗ хуков разделения (fallback == явный Эйлер → нестабилен).

Метрики (stdout, файлы не создаются):
  - устойчивость (finite, max|y|), относительная ошибка к exact = expm(A·T)·y0;
  - wall time прогона; суммарное число субшагов; накопленный local_error.

Запуск:
    python benchmarks/l2_stiff_integrators.py
    python benchmarks/l2_stiff_integrators.py --help
    python benchmarks/l2_stiff_integrators.py --eps 1e-2 --dt 0.1 --steps 50
"""

import argparse
import time

import numpy as np
import tensorflow as tf
from scipy.linalg import expm

import neuraltide.config
from neuraltide.core.base import PopulationModel
from neuraltide.integrators import (
    AdaptiveRKIntegrator,
    ExponentialEulerIntegrator,
    IMEXEulerIntegrator,
    RK4Integrator,
    RosenbrockIntegrator,
    classify_failure,
)
from neuraltide.utils import seed_everything


class StiffLinearPopulation(PopulationModel):
    """y' = A y, A = [[-1/eps, 1], [0, -1]]; n_units=1, d=2.

    Для IMEX: жёсткая часть = быстрая диагональ (-1/eps)·y1, линейный
    оператор A_stiff = diag(-1/eps, 0); нежёсткая — остаток.
    split_enabled=False имитирует популяцию без переопределённых хуков
    (честный fallback IMEX на явный Эйлер).
    """

    def __init__(self, eps: float, dt: float, split_enabled: bool = True):
        super().__init__(n_units=1, dt=dt, name="stiff_linear")
        dtype = neuraltide.config.get_dtype()
        self.eps = float(eps)
        self.split_enabled = bool(split_enabled)
        self.a_full = tf.constant([[-1.0 / self.eps, 1.0], [0.0, -1.0]], dtype=dtype)
        self.a_stiff = tf.constant([[-1.0 / self.eps, 0.0], [0.0, 0.0]], dtype=dtype)
        self.state_size = [tf.TensorShape([1, 1]), tf.TensorShape([1, 1])]

    def get_initial_state(self, batch_size: int = 1):
        dtype = neuraltide.config.get_dtype()
        return [tf.ones([1, 1], dtype=dtype), tf.ones([1, 1], dtype=dtype)]

    def _matvec(self, matrix, state):
        flat = tf.stack([state[0][0], state[1][0]])
        out = matrix @ flat
        return [out[0][tf.newaxis, :], out[1][tf.newaxis, :]]

    def derivatives(self, state, total_synaptic_input):
        return self._matvec(self.a_full, state)

    def derivatives_split(self, state, total_synaptic_input):
        f = self.derivatives(state, total_synaptic_input)
        if not self.split_enabled:
            return super().derivatives_split(state, total_synaptic_input)
        f_stiff = self._matvec(self.a_stiff, state)
        return ([f[0] - f_stiff[0], f[1] - f_stiff[1]], f_stiff)

    def linear_operator(self, state, total_synaptic_input):
        if not self.split_enabled:
            return None
        dtype = neuraltide.config.get_dtype()
        return self.a_stiff[tf.newaxis, :, :], tf.zeros([1, 2], dtype=dtype)

    def get_firing_rate(self, state):
        return state[0]

    @property
    def parameter_spec(self):
        return {}


def _zeros_inp():
    dtype = neuraltide.config.get_dtype()
    return {
        "I_syn": tf.zeros([1, 1], dtype=dtype),
        "g_syn": tf.zeros([1, 1], dtype=dtype),
    }


def run_method(integrator, eps, dt, steps, split_enabled=True):
    """Прогон в eager через прямые вызовы step(); метрики — как в тестах."""
    pop = StiffLinearPopulation(eps=eps, dt=dt, split_enabled=split_enabled)
    inp = _zeros_inp()
    local_error_sum = 0.0
    n_substeps_total = 0

    # Разогрев (трассировка/кэш не участвуют в замере), затем сброс.
    warm_state = pop.get_initial_state()
    warm_state, _ = integrator.step(pop, warm_state, inp)
    state = pop.get_initial_state()

    t0 = time.perf_counter()
    for _ in range(steps):
        state, le = integrator.step(pop, state, inp)
        local_error_sum += float(le[0])
        if getattr(integrator, "last_n_substeps", None) is not None:
            n_substeps_total += integrator.last_n_substeps
    wall = time.perf_counter() - t0

    y = np.array([float(state[0][0, 0]), float(state[1][0, 0])])
    failure = classify_failure(list(state))
    if n_substeps_total == 0:
        n_substeps_total = getattr(integrator, "n_substeps", 1) * steps
    return {
        "y": y,
        "wall": wall,
        "local_error": local_error_sum,
        "n_substeps": n_substeps_total,
        "finite": failure["status"] == "ok",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="L2 benchmark: P2 integrators on a small stiff problem."
    )
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--eps", type=float, default=1e-3, help="параметр жёсткости")
    parser.add_argument("--dt", type=float, default=0.5, help="шаг сети, мс")
    parser.add_argument("--steps", type=int, default=10, help="число шагов сети")
    parser.add_argument("--rtol", type=float, default=1e-3, help="допуск AdaptiveRK")
    parser.add_argument("--atol", type=float, default=1e-7, help="допуск AdaptiveRK")
    parser.add_argument(
        "--substeps",
        type=int,
        default=10,
        help="субшаги IMEX/Rosenbrock/Exponential (h = dt/substeps)",
    )
    args = parser.parse_args()

    neuraltide.config.set_dtype(tf.float64)  # до создания моделей
    seed_everything(args.seed)

    dt = args.dt
    steps = args.steps
    total_time = dt * steps
    exact = expm(
        np.array([[-1.0 / args.eps, 1.0], [0.0, -1.0]]) * total_time
    ) @ np.ones(2)
    norm_exact = np.linalg.norm(exact)

    print("=" * 72)
    print("L2 stiff integrators: linear stiff system y' = A y")
    print("=" * 72)
    print(
        f"A = [[-1/eps, 1], [0, -1]], eps={args.eps:g}  (stiffness ratio "
        f"~{1.0 / args.eps:g})"
    )
    print(
        f"dt={dt} ms, steps={steps} (T={total_time:g} ms), "
        f"explicit Euler/RK4 limit dt <= {2.0 * args.eps:g} ms"
    )
    print(f"exact y(T) = [{exact[0]:.6g}, {exact[1]:.6g}]")

    configs = [
        ("RK4", RK4Integrator(), True),
        ("AdaptiveRK 3(2)", AdaptiveRKIntegrator(rtol=args.rtol, atol=args.atol), True),
        ("IMEX-Euler", IMEXEulerIntegrator(n_substeps=args.substeps), True),
        (
            "IMEX (no split -> Euler)",
            IMEXEulerIntegrator(n_substeps=args.substeps),
            False,
        ),
        ("Rosenbrock ROS2", RosenbrockIntegrator(n_substeps=args.substeps), True),
        (
            "ExponentialEuler",
            ExponentialEulerIntegrator(n_substeps=args.substeps),
            True,
        ),
    ]

    print()
    print(f"{'method':<28}{'stable':>8}{'rel_err':>19}{'wall,s':>9}{'substeps':>10}")
    print("-" * 74)
    results = {}
    for name, integrator, split in configs:
        r = run_method(integrator, args.eps, dt, steps, split_enabled=split)
        results[name] = r
        if r["finite"]:
            rel_err = float(np.linalg.norm(r["y"] - exact) / norm_exact)
            rel_err_str = f"{rel_err:.3e}"
        else:
            rel_err = float("inf")
            rel_err_str = "unstable/overflow"
        print(
            f"{name:<28}{str(r['finite']):>8}{rel_err_str:>19}"
            f"{r['wall']:>9.3f}{r['n_substeps']:>10}"
        )

    print()
    print("[local_error (stability indicator, суммарный)]")
    for name, r in results.items():
        print(f"  {name:<27}{r['local_error']:.4g}")

    # --- Честное заключение ---
    print()
    print("[заключение]")
    rk4 = results["RK4"]
    imex = results["IMEX-Euler"]
    ros = results["Rosenbrock ROS2"]
    exp = results["ExponentialEuler"]
    ark = results["AdaptiveRK 3(2)"]

    if not rk4["finite"]:
        print(
            f"  * RK4 нестабилен: |lambda_fast*dt| = {1.0 / args.eps * dt:g} "
            f">>= 2.78 — явные схемы здесь требуют dt <= {2.0 * args.eps:g} мс."
        )
    if imex["finite"] and ros["finite"] and exp["finite"]:
        print(
            "  * IMEX / Rosenbrock / Exponential устойчивы при том же dt за "
            f"фиксированное число субшагов (h = {dt / args.substeps:g} мс): "
            "жёсткая компонента обработана неявно/экспоненциально, а не "
            "явным шагом."
        )
    if exp["finite"]:
        exp_rel = float(np.linalg.norm(exp["y"] - exact) / norm_exact)
        print(
            f"  * ExponentialEuler — точен до {exp_rel:.1e} на этой линейной "
            "задаче (линеаризация совпадает с системой); на нелинейных задачах "
            "точность порядка h^2 и требует мелких субшагов, как у Rosenbrock."
        )
    if ark["finite"]:
        ark_rel = float(np.linalg.norm(ark["y"] - exact) / norm_exact)
        print(
            f"  * AdaptiveRK устойчив без производных и хуков (число субшагов "
            f"автоматически, rel_err={ark_rel:.1e}), но платит за каждый "
            "субшаг 4 вычислениями правой части — выигрыш у него против "
            "неявных методов только когда производные/JVP дороги."
        )
    print(
        "  * IMEX дешевле Rosenbrock/Exponential (нет JVP), но требует от "
        "популяции корректного derivatives_split + linear_operator; без них "
        "метод честно вырождается в явный Эйлер (строка 'IMEX (no split)')."
    )
    print(
        "  * Все wall time — индикативны (eager, один прогон после разогрева; "
        "задача размером 2 состояния, доминирует Python/tf overhead)."
    )


if __name__ == "__main__":
    main()
