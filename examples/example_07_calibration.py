"""
Пример: калибровка малой сети через CalibrationProblem (этап P0).

Сценарий:
1. Собирается малая сеть (WilsonCowan + вход через StaticSynapse) и с неё
   снимается целевая траектория частоты разрядов.
2. Параметры сети сдвигаются от «истинных» значений.
3. Параметры объявляются оптимизируемыми через ParameterSpace с
   преобразованиями координат (log — для положительных tau/gsyn_max).
4. CalibrationProblem проверяется taylor_test'ом (верификация градиента),
   затем выполняется несколько ручных шагов Adam по вектору phi.
5. Результат сохраняется в объект OptimizationResult (в памяти, без файлов).

Запускать из директории neuraltide:
    python examples/example_07_calibration.py
"""

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
    OptimizationResult,
    ParameterSpace,
    ResidualLoss,
    taylor_test,
)
from neuraltide.populations import WilsonCowan
from neuraltide.synapses import StaticSynapse
from neuraltide.utils import seed_everything

seed_everything(42)

DT = 0.5
T = 80  # мс

# --- Модель ---
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
        "gsyn_max": {"value": [[0.05]], "trainable": True, "min": 0.001, "max": 1.0},
        "pconn": [[1.0]],
        "e_r": 0.0,
    },
    name="in->exc",
)
gen = SinusoidalGenerator(
    dt=DT,
    params={
        "amplitude": 8.0,
        "freq": 8.0,
        "phase": 0.0,
        "offset": 10.0,
    },
)

graph = NetworkGraph(dt=DT)
graph.declare_input("in", n_units=gen.n_units)
graph.add_population("exc", pop)
graph.add_synapse("in->exc", syn, src="in", tgt="exc")
network = NetworkRNN(graph, EulerIntegrator())

t_seq = tf.constant((np.arange(T, dtype=np.float32) * DT)[None, :, None])
inputs = graph.pack_inputs({"in": gen(t_seq)})

# --- Целевая траектория: прогон при «истинных» параметрах ---
target = network(t_seq, inputs=inputs).firing_rates["exc"]

# --- Сдвиг параметров от истинных значений (задача калибровки) ---
theta_start = {
    "exc/tau": tf.constant([[16.0]]),  # истина 8.0
    "exc/I_ext": tf.constant([[0.2]]),  # истина 0.8
    "in->exc/gsyn_max": tf.constant([[0.15]]),  # истина 0.05
}

space = ParameterSpace.from_network(
    network,
    transforms={
        "exc/tau": LogTransform(),
        "in->exc/gsyn_max": LogTransform(),
    },
)
space.apply_to_model(theta_start)

objective = ResidualLoss(
    [ObservationSpec(name="exc", population="exc", target=target, sigma=2.0)]
)
problem = CalibrationProblem(network, space, t_seq, inputs=inputs, objective=objective)

print("=" * 60)
print("CalibrationProblem (P0)")
print("=" * 60)
print(f"optimizable parameters: {space.names}")
print(f"phi size: {problem.size}")

# --- Оценка до оптимизации ---
ev0 = problem.evaluate()
print(f"\nloss before: {ev0.loss:.6f}  (components: {ev0.components})")
print(f"status: {ev0.status}, n_forward: {ev0.n_forward}")


# --- Верификация градиента: направленный Taylor test ---
# (в точке вдали от минимума — иначе R0 вырождается в O(h^2))
def value_and_grad(phi_np):
    ev, grad = problem.value_and_gradient(phi_np)
    return ev.loss, np.asarray(grad)


phi0 = problem.pack()
report = taylor_test(
    value_and_grad,
    np.asarray(phi0) + 0.25,
    # float32: ограничиваем сетку окном без округления (см. taylor_test)
    h_values=np.logspace(-1, -3, 7),
)
print("\nTaylor test (value_and_gradient):")
print(f"  slope R0 median: {report['slope_R0_median']:.3f} (ожидается ~1)")
print(f"  slope R1 median: {report['slope_R1_median']:.3f} (ожидается ~2)")

# --- Несколько ручных шагов Adam по phi ---
phi_var = tf.Variable(phi0)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.02)
n_steps = 30
for step in range(n_steps):
    ev, grad = problem.value_and_gradient(phi_var)
    optimizer.apply_gradients([(tf.convert_to_tensor(grad), phi_var)])
    if step % 5 == 0 or step == n_steps - 1:
        print(f"  step {step:2d}: loss = {ev.loss:.6f}")

# --- Оценка после оптимизации ---
ev1 = problem.evaluate(phi_var)
print(f"\nloss after:  {ev1.loss:.6f}")
print(f"decreased: {ev1.loss < ev0.loss}")

theta_fit = space.unpack(problem.pack())
print("\nfitted physical values:")
for name in space.names:
    print(f"  {name}: {float(theta_fit[name].numpy().ravel()[0]):.4f}")

# --- Единый формат результата (в памяти; файлы не создаются) ---
result = OptimizationResult(
    phi_best=np.asarray(problem.pack()),
    theta_best=theta_fit,
    loss_best=ev1.loss,
    history=[{"loss": ev0.loss}, {"loss": ev1.loss}],
    seed=42,
    config={"dt": DT, "optimizer": "Adam", "steps": n_steps},
    dtype=str(neuraltide.config.get_dtype().name),
)
data = result.to_dict()
print(f"\nOptimizationResult keys: {sorted(data)}")
print(f"loss_best: {data['loss_best']:.6f}")

print("\n" + "=" * 60)
print(" Готово: value_and_gradient проверен Taylor test'ом,")
print(" ручные шаги Adam уменьшили loss.")
print("=" * 60)
