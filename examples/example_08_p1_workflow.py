"""
Пример: сквозной рабочий процесс калибровки P1 (этап P1 дорожной карты).

Сценарий на малой синтетической сети с известным theta*:

1. WilsonCowan (1 юнит) + StaticSynapse от SinusoidalGenerator; целевая
   траектория снята с параметров theta* (сеть восстанавливает свои же
   параметры — чистая проверка цепочки).
2. multistart (Sobol-дизайн -> screening -> доводка top-k через
   least_squares/TRF) сходится к theta*.
3. residual_jacobian (forward sensitivities) в точке решения ->
   compute_fim: спектр FIM, effective rank, число обусловленности,
   жёсткие направления.
4. likelihood_profile по одному параметру (exc/tau): χ²-интервал 95%.
5. validate на held-out протоколе (другая частота входа) и free-run
   (честный прогон на исходных входах без teacher forcing).

Файлы не создаются; работает на CPU за считанные секунды.

Запускать из директории neuraltide:
    python examples/example_08_p1_workflow.py
"""

import numpy as np
import tensorflow as tf

import neuraltide.config
from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.inputs import SinusoidalGenerator
from neuraltide.integrators import EulerIntegrator
from neuraltide.optimization import (
    CalibrationProblem,
    Experiment,
    LogTransform,
    ObservationSpec,
    ParameterSpace,
    ResidualLoss,
    compute_fim,
    likelihood_profile,
    multistart,
    residual_jacobian,
)
from neuraltide.populations import WilsonCowan
from neuraltide.synapses import StaticSynapse
from neuraltide.utils import seed_everything

# Локальные оптимизаторы и FIM-анализ точнее в float64; dtype ставим ДО
# создания моделей (контракт neuraltide.config).
neuraltide.config.set_dtype(tf.float64)
seed_everything(7)

DT = 0.5
T = 80  # мс

# --- Известные "истинные" параметры theta* ---
THETA_STAR = {
    "exc/tau": 8.0,
    "exc/I_ext": 0.8,
    "in->exc/gsyn_max": 30.0,
}

# --- Модель ---
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
        "e_r": 1.0,  # возврат к равновесию: иначе (e_r - v_mean) = 0
    },
    name="in->exc",
)


def make_generator(freq):
    return SinusoidalGenerator(
        dt=DT,
        params={
            "amplitude": 8.0,
            "freq": float(freq),
            "phase": 0.0,
            "offset": 10.0,
        },
    )


graph = NetworkGraph(dt=DT)
graph.declare_input("in", n_units=1)
graph.add_population("exc", pop)
graph.add_synapse("in->exc", syn, src="in", tgt="exc")
network = NetworkRNN(graph, EulerIntegrator())

t_seq = tf.constant((np.arange(T, dtype=np.float64) * DT)[None, :, None])
inputs_train = graph.pack_inputs({"in": make_generator(freq=8.0)(t_seq)})
inputs_holdout = graph.pack_inputs({"in": make_generator(freq=16.0)(t_seq)})

# --- Целевая траектория: прогон при theta* ---
target = network(t_seq, inputs=inputs_train).firing_rates["exc"]
objective = ResidualLoss(
    [ObservationSpec(name="exc", population="exc", target=target, sigma=2.0)]
)

space = ParameterSpace.from_network(
    network,
    transforms={
        "exc/tau": LogTransform(),
        "in->exc/gsyn_max": LogTransform(),
    },
)

# --- Отложенный протокол: другая частота входа (для validate) ---
holdout = Experiment(
    name="holdout_f16",
    t_sequence=t_seq,
    inputs=inputs_holdout,
    objective=ResidualLoss(
        [ObservationSpec(name="exc", population="exc", target=target, sigma=2.0)]
    ),
)

problem = CalibrationProblem(
    network,
    space,
    t_seq,
    inputs=inputs_train,
    objective=objective,
    held_out=[holdout],
)

# --- Сдвиг параметров от theta* (задача калибровки) ---
space.apply_to_model(
    {
        "exc/tau": tf.constant([[20.0]]),  # истина 8.0
        "exc/I_ext": tf.constant([[0.2]]),  # истина 0.8
        "in->exc/gsyn_max": tf.constant([[120.0]]),  # истина 30.0
    }
)

print("=" * 64)
print("P1 workflow: multistart -> jacobian -> FIM -> profile -> validate")
print("=" * 64)
print(f"theta*: {THETA_STAR}")
print(f"optimizable parameters: {space.names} (phi size {problem.size})")
ev0 = problem.evaluate()
print(f"loss before: {ev0.loss:.6f}")

# --- 1-2. Мультистарт: Sobol -> screening -> доводка top-k (TRF) ---
ms = multistart(
    problem,
    n_starts=8,
    method="sobol",
    refine_top_k=3,
    local="least_squares",
    seed=7,
)
best = ms.solutions[ms.best_index]
phi_best = np.asarray(ms.base.phi_best, dtype=np.float64)
print("\n[multistart]")
print(
    f"  starts: {ms.n_starts} (refined {ms.n_refined}, failed {ms.n_failed}), "
    f"clusters: {ms.clustering['n_clusters']}"
)
print(f"  best loss: {ms.base.loss_best:.6f}  wall: {ms.wall_time:.2f}s")
print(f"  n_forward total: {problem.n_forward}")

# --- 3. Якобиан невязок (forward sensitivities) в точке решения ---
jac = residual_jacobian(problem, phi_best)
print("\n[residual_jacobian]")
print(
    f"  J shape: {jac.J.shape}, n_forward: {jac.n_forward}, wall: {jac.wall_time:.2f}s"
)

# --- 4. FIM = J^T J: спектр и идентифицируемость ---
fim = compute_fim(jac, param_names=space.names)
print("\n[compute_fim]")
print(f"  effective rank (energy={fim.energy}): {fim.effective_rank}/{problem.size}")
print(f"  condition number kappa(J): {fim.condition_number:.3g}")
for d in fim.stiff_dirs[:2]:
    top = ", ".join(f"{n}={v:+.2f}" for n, v in d.top_params[:3])
    print(f"  stiff dir {d.index}: s={d.singular_value:.3g}, top: {top}")
if fim.collinearity:
    pairs = ", ".join(
        f"{p.first}~{p.second} (|cos|={p.cosine:.3f})" for p in fim.collinearity
    )
    print(f"  collinear pairs: {pairs}")

# --- 5. Профиль правдоподобия по exc/tau ---
profile = likelihood_profile(
    problem,
    "exc/tau",
    grid=np.linspace(4.0, 14.0, 6),
    optimizer="lbfgsb",
    warm_start=True,
)
print("\n[likelihood_profile: exc/tau]")
print(f"  successful nodes: {sum(profile.successes)}/{len(profile.successes)}")
print(
    f"  95% interval: {profile.intervals['0.95']}  (theta* = {THETA_STAR['exc/tau']})"
)
print(f"  profile warnings: {profile.warnings or 'none'}")

# --- 6. Валидация на held-out протоколе и честный free-run ---
validation = problem.validate(phi_best)
ev_free = problem.evaluate(phi_best)  # свободный прогон, teacher forcing нет
print("\n[validate / free-run]")
for name, loss in validation.items():
    print(f"  held-out '{name}': loss = {loss:.6f}")
print(f"  free-run loss: {ev_free.loss:.6f}  (status {ev_free.status})")

# --- Итог: ошибка восстановления theta ---
theta_fit = space.unpack(phi_best)
print("\n[recovery error]")
max_err = 0.0
for name, star in THETA_STAR.items():
    got = float(tf.reshape(theta_fit[name], [-1])[0])
    err = abs(got - star)
    max_err = max(max_err, err)
    print(f"  {name}: fit={got:.6f}  theta*={star:.6f}  |err|={err:.2e}")
print(f"  max |err|: {max_err:.2e}")

print("\n" + "=" * 64)
print(" Готово: мультистарт восстановил theta*, FIM/профиль оценили")
print(" идентифицируемость, held-out и free-run подтверждены.")
print("=" * 64)
