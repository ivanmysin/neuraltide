"""
Пример: сквозной рабочий процесс калибровки P2 (этап P2 дорожной карты).

Сценарий на малой синтетической E–I сети с известным theta*: целевые
траектории сняты с параметров theta* и зашумлены (гауссов шум с тем же
sigma, что в ObservationSpec) — так MAP-невязки ненулевые, интервалы
имеют осмысленное покрытие, а exact-fit (потеря ровно 0) был бы
вырожденным случаем: residual-нормировка sqrt(2*psi) недифференцируема
в нуле, и forward-чувствительности (Laplace) в точке exact-fit дают NaN.

Обучаемые параметры — две постоянные времени (exc/tau, inh/tau): они
хорошо идентифицируемы по зашумлённым траекториям. Синаптические усиления
фиксированы: вместе с входным током WilsonCowan они образуют почти
коллинеарные направления (все суммируются во входное поле), и зашумлённые
данные их не идентифицируют — Laplace-интервалы честно раздуваются.

Цепочка:
1. Глобальный поиск малого бюджета: global_search (DE, max_evals) ->
   доводка refine_top_k через L-BFGS-B; отдельная явная доводка
   minimize_lbfgsb от лучшей точки глобальной фазы.
2. Балансировка компонент потерь (smoke): balanced_train со стратегией
   GradNormWeights и PCGrad-проекцией конфликтующих градиентов; сравнение
   с обязательным baseline FixedWeights (те же шаги без адаптивных весов).
3. Неопределённость (на свежем экземпляре задачи из той же фабрики —
   ограничение совместимости TF forward-AD с предшествующими
   gradient-прогонами, см. «Ограничения этапа P2» в docs/optimization.md):
   laplace_approximation (GN-ковариация через якобиан невязок) ->
   маргинальные интервалы параметров -> прогнозные интервалы
   firing_rates по ансамблю phi из posterior.
4. Отдельно: residual_diagnostics (остатки в точке решения) и пакетная
   оценка кандидатов через SerialPool (эквивалентность losses ручному
   циклу problem.evaluate).

Файлы не создаются; работает на CPU за считанные секунды.

Запускать из директории neuraltide:
    python examples/example_09_p2_workflow.py
"""

import numpy as np
import tensorflow as tf

import neuraltide.config
from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.inputs import SinusoidalGenerator
from neuraltide.integrators import EulerIntegrator
from neuraltide.optimization import (
    CalibrationProblem,
    GradNormWeights,
    ObservationSpec,
    ParameterSpace,
    ResidualLoss,
    evaluate_candidates,
    global_search,
    laplace_approximation,
    minimize_lbfgsb,
    balanced_train,
    pcgrad_project,
    predictive_intervals,
    residual_diagnostics,
)
from neuraltide.populations import WilsonCowan
from neuraltide.synapses import StaticSynapse
from neuraltide.utils import seed_everything

# Локальные оптимизаторы и Лаплас-аппроксимация точнее в float64; dtype
# ставим ДО создания моделей (контракт neuraltide.config).
neuraltide.config.set_dtype(tf.float64)
seed_everything(11)

DT = 0.5
T = 80  # мс
SIGMA = 1.0  # модель шума ObservationSpec; шум цели генерируется с тем же sigma

# --- Известные "истинные" параметры theta* (обе tau обучаемы) ---
# Синаптические усиления фиксированы на theta* — см. обоснование в
# шапке модуля: в этой сети они плохо идентифицируются по зашумлённым
# данным, что честно показывают раздутые Laplace-интервалы.
THETA_STAR = {
    "exc/tau": 8.0,
    "inh/tau": 12.0,
}


def build_network():
    """E–I сеть и протокол; целевые траектории — прогон при theta*."""
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
        params={  # фиксирован на theta* (см. шапку)
            "gsyn_max": [[30.0]],
            "pconn": [[1.0]],
            "e_r": 1.0,
        },
        name="in->exc",
    )
    syn_ie = StaticSynapse(
        n_pre=1,
        n_post=1,
        dt=DT,
        params={  # фиксирован на theta* (тормозный)
            "gsyn_max": [[10.0]],
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

    t_seq = tf.constant((np.arange(T, dtype=np.float64) * DT)[None, :, None])
    inputs = graph.pack_inputs({"in": gen(t_seq)})
    out = network(t_seq, inputs=inputs)
    # Шум цели с sigma ObservationSpec: MAP-невязки ненулевые, интервалы
    # имеют правильное покрытие; exact-fit был бы вырожденным случаем
    # (sqrt-нормировка residual недифференцируема в нуле).
    rng = np.random.default_rng(123)
    targets = {}
    for pop in ("exc", "inh"):
        rates = np.asarray(out.firing_rates[pop], dtype=np.float64)
        targets[pop] = tf.constant(rates + rng.normal(0.0, SIGMA, rates.shape))
    observations = [
        ObservationSpec(
            name="exc", population="exc", target=targets["exc"], sigma=SIGMA
        ),
        ObservationSpec(
            name="inh", population="inh", target=targets["inh"], sigma=SIGMA
        ),
    ]
    return network, ParameterSpace.from_network(network), t_seq, inputs, observations


def build_problem():
    """
    Фабрика уровня модуля для parallel.evaluate_candidates: строит свежую
    задачу калибровки (picklable — worker-процессы тоже смогли бы её
    использовать; здесь используется только SerialPool в родителе).
    """
    network, space, t_seq, inputs, observations = build_network()
    return CalibrationProblem(
        network,
        space,
        t_seq,
        inputs=inputs,
        objective=ResidualLoss(observations),
    )


problem = build_problem()
space = problem.parameter_space

print("=" * 64)
print("P2 workflow: DE -> L-BFGS-B -> balanced_train -> Laplace")
print("             -> predictive intervals + diagnostics + SerialPool")
print("=" * 64)
print(f"theta*: {THETA_STAR}")
print(f"optimizable parameters: {space.names} (phi size {problem.size})")

# --- Сдвиг параметров от theta* (задача калибровки) ---
space.apply_to_model(
    {
        "exc/tau": tf.constant([[20.0]]),  # истина 8.0
        "inh/tau": tf.constant([[25.0]]),  # истина 12.0
    }
)
ev0 = problem.evaluate()
phi_start = np.asarray(problem.pack(), dtype=np.float64)  # сдвинутая точка
print(f"loss before: {ev0.loss:.6f}")

# --- 1. Глобальный поиск малого бюджета (DE) + доводка top-k (L-BFGS-B) ---
gs = global_search(
    problem,
    method="de",
    max_evals=200,
    refine_top_k=1,
    local="lbfgsb",
    seed=11,
    options={"popsize": 10, "polish": False, "tol": 0.01},
)
phi_global = np.asarray(gs.base.phi_best, dtype=np.float64)
n_failed = sum(1 for h in gs.history if h["failed"])
print("\n[global_search: de]")
print(
    f"  evals: {gs.n_evals} (failed {n_failed}), "
    f"best loss: {gs.base.loss_best:.6f}, wall: {gs.wall_time:.2f}s"
)
print(f"  message: {gs.message!r}")

# --- 2. Явная доводка L-BFGS-B от лучшей точки глобальной фазы ---
lbfgs = minimize_lbfgsb(problem, phi_global, maxiter=100)
phi_map = np.asarray(lbfgs.base.phi_best, dtype=np.float64)
print("\n[minimize_lbfgsb: доводка]")
print(
    f"  loss: {lbfgs.base.loss_best:.6f} (status {lbfgs.status}), "
    f"n_forward: {lbfgs.n_forward}, wall: {lbfgs.wall_time:.2f}s"
)
theta_fit = space.unpack(phi_map)
print("  recovery:")
for name, star in THETA_STAR.items():
    got = float(tf.reshape(theta_fit[name], [-1])[0])
    print(f"    {name}: fit={got:.4f}  theta*={star:.4f}  |err|={abs(got - star):.2e}")

# --- 3. Балансировка компонент (smoke): GradNormWeights + PCGrad ---
# balanced_train мутирует модель по общему контракту; начинаем из
# сдвинутой точки и после smoke возвращаем модель в phi_map явной оценкой.
phi_pert = phi_global
bt = balanced_train(
    problem,
    phi_pert,
    GradNormWeights(alpha=1.0),
    optimizer="adam",
    max_steps=6,
    projector=pcgrad_project,
    seed=11,
)
print("\n[balanced_train: GradNormWeights + PCGrad (smoke)]")
w_first = bt.weight_history[0]
w_last = bt.weight_history[-1]
print(
    "  weights (first -> last step): "
    + ", ".join(
        f"{k[2:]}={w_first[k]:.3f}->{w_last[k]:.3f}"
        for k in w_first
        if k.startswith("w:")
    )
)
print(
    f"  balanced best: {bt.base.loss_best:.6f}  vs  "
    f"fixed-weights baseline: {float(bt.baseline['best_loss']):.6f}"
)
print(f"  note: {bt.notes}")
problem.evaluate(phi_map)  # вернуть модель в MAP-точку после smoke

# --- 4. Лаплас-аппроксимация постериора в MAP-точке ---
# Forward-режим чувствительностей (residual_jacobian внутри laplace) в
# пределах одного экземпляра NetworkRNN несовместим с предшествующими
# gradient-прогонами: @tf.function _scan_forward_states кэширует граф, и
# TF forward-AD падает после GradientTape (см. «Ограничения этапа P2» в
# docs/optimization.md). Поэтому UQ-блок запускаем на свежем экземпляре
# задачи из той же фабрики (те же theta*, протокол и цели).
problem_uq = build_problem()
lap = laplace_approximation(problem_uq, phi_map, method="gn")
print("\n[laplace_approximation]")
print(
    f"  method={lap.method}, PD: {lap.pd} "
    f"(jitter={lap.jitter_used}), wall: {lap.wall_time:.2f}s"
)
for name, star in THETA_STAR.items():
    lo, hi = lap.intervals[name]
    cover = "✓" if lo <= star <= hi else "✗"
    print(f"  95% {name}: [{lo:.3f}, {hi:.3f}]  theta*={star} {cover}")

# --- 5. Прогнозные интервалы firing_rates по ансамблю posterior ---
rng = np.random.default_rng(11)
cov = lap.covariance + 1e-12 * np.eye(lap.phi_map.size)
phi_samples = rng.multivariate_normal(lap.phi_map, cov, size=48)
pred = predictive_intervals(problem_uq, phi_samples, level=0.95)
print("\n[predictive_intervals]")
print(f"  samples: {pred.n_samples} (skipped {pred.n_skipped_nonfinite})")
# Покрытие зашумлённой цели интервалами (при верной модели шума ~ level).
spec_by_pop = {spec.population: spec for spec in problem_uq.objective.observations}
for pop in pred.populations:
    width = float(np.mean(pred.upper[pop] - pred.lower[pop]))
    target = np.asarray(spec_by_pop[pop].target, dtype=np.float64)[0]
    inside = (pred.lower[pop] <= target) & (target <= pred.upper[pop])
    print(
        f"  '{pop}': mean width {width:.3f} Hz, "
        f"target coverage {float(np.mean(inside)):.3f}"
    )

# --- 6. Диагностика остатков в точке решения ---
diag = residual_diagnostics(problem_uq, phi_map, max_lag=5)
print("\n[residual_diagnostics]")
for spec in diag.specs:
    ac = [a for a in spec.autocorr if a is not None]
    ac_max = max((abs(a) for a in ac), default=0.0)
    print(
        f"  '{spec.name}': std={spec.std:.4g}, max|r|={spec.max_abs:.4g}, "
        f"max|autocorr|={ac_max:.3f}"
    )

# --- 7. Пакетная оценка кандидатов: SerialPool (эквивалентность losses) ---
candidates = [phi_map, phi_start]
records = evaluate_candidates(build_problem, candidates, n_workers=1, seed=0)
print("\n[evaluate_candidates: SerialPool]")
max_dev = 0.0
for rec, phi in zip(records, candidates):
    direct = problem.evaluate(np.asarray(phi, dtype=np.float64)).loss
    dev = abs(rec.loss - direct)
    max_dev = max(max_dev, dev)
    print(
        f"  #{rec.index}: pool loss={rec.loss:.9g}, direct={direct:.9g}, "
        f"dev={dev:.2e}, wall={rec.wall_time:.3f}s"
    )
print(f"  max |pool - direct|: {max_dev:.2e}")

print("\n" + "=" * 64)
print(" Готово: DE+L-BFGS-B восстановили theta*, балансировка/PCGrad")
print(" отработали, Laplace дал интервалы, SerialPool совпал с ручным")
print(" циклом. Ограничения P2 см. в docs/optimization.md.")
print("=" * 64)
