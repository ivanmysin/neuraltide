"""
Пример: общий и эксперимент-специфичный начальный вектор x0 в
мультиэкспериментной калибровке (MultiExperimentProblem).

Сценарий на малой синтетической сети (WilsonCowan 1 юнит + StaticSynapse
от SinusoidalGenerator):

1. Два эксперимента с ОДИНАКОВЫМИ входами и динамикой, но разным
   начальным состоянием: 'control' стартует из сетевого дефолта (E=0),
   'stim' — из смещённого состояния (E=0.8). Цели сняты с прогонов при
   theta* — сеть восстанавливает свои же данные.
2. В ParameterSpace регистрируются общие компоненты начального состояния
   'x0/<pop>/<slot>' (участвуют во всех экспериментах) и
   эксперимент-специфичные 'x0/<experiment>/<pop>/<slot>' (только у
   эксперимента с совпадающим именем, с приоритетом над общими).
3. Траектории двух экспериментов при одном и том же phi различаются:
   у 'stim' exp-запись перезаписывает общий слот начального состояния.
4. Градиент по общему x0 — сумма вкладов экспериментов; градиент по
   exp-specific x0 идёт только через свой эксперимент (проверяется
   сравнением с одноэкспериментными задачами).

Файлы не создаются; работает на CPU за считанные секунды.

Запускать из директории neuraltide:
    python examples/example_10_multiexperiment_x0.py
"""

import numpy as np
import tensorflow as tf

import neuraltide.config
from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.inputs import SinusoidalGenerator
from neuraltide.integrators import EulerIntegrator
from neuraltide.optimization import (
    Experiment,
    MultiExperimentProblem,
    ObservationSpec,
    ParameterSpace,
    ResidualLoss,
)
from neuraltide.populations import WilsonCowan
from neuraltide.synapses import StaticSynapse
from neuraltide.utils import seed_everything

# float64 для чистоты сравнения градиентов; dtype ставим ДО создания моделей.
neuraltide.config.set_dtype(tf.float64)
seed_everything(11)

DT = 0.5
T = 60  # мс
E_STIM_TRUE = 0.8  # "истинное" начальное состояние эксперимента 'stim'


# --- Модель (одна популяция, один вход) ---
pop = WilsonCowan(
    dt=DT,
    params={
        "tau": {"value": [8.0], "trainable": True, "min": 1.0, "max": 50.0},
        "a": 1.0,
        "theta": 0.0,
        "w": 0.2,
        "I_ext": 0.8,
        "max_rate": 100.0,
    },
    name="exc",
)
syn = StaticSynapse(
    n_pre=1,
    n_post=1,
    dt=DT,
    params={"gsyn_max": [[30.0]], "pconn": [[1.0]], "e_r": 1.0},
    name="in->exc",
)

graph = NetworkGraph(dt=DT)
graph.declare_input("in", n_units=1)
graph.add_population("exc", pop)
graph.add_synapse("in->exc", syn, src="in", tgt="exc")
network = NetworkRNN(graph, EulerIntegrator())

t_seq = tf.constant((np.arange(T, dtype=np.float64) * DT)[None, :, None])
inputs = graph.pack_inputs(
    {
        "in": SinusoidalGenerator(
            dt=DT,
            params={
                "amplitude": 8.0,
                "freq": 8.0,
                "phase": 0.0,
                "offset": 10.0,
            },
        )(t_seq)
    }
)

# --- Целевые траектории: одинаковая динамика, разные начальные состояния ---
target_control = network(t_seq, inputs=inputs).firing_rates["exc"]

init_pop, init_syn = network.get_initial_state(batch_size=1)
init_pop[0] = tf.constant([[E_STIM_TRUE]], dtype=tf.float64)
target_stim = network(
    t_seq, inputs=inputs, initial_state=(init_pop, init_syn)
).firing_rates["exc"]


def make_objective(target):
    return ResidualLoss(
        [ObservationSpec(name="exc", population="exc", target=target, sigma=1.0)]
    )


exp_control = Experiment(
    name="control",
    t_sequence=t_seq,
    inputs=inputs,
    objective=make_objective(target_control),
)
exp_stim = Experiment(
    name="stim",
    t_sequence=t_seq,
    inputs=inputs,
    objective=make_objective(target_stim),
)


# --- Пространство: общий x0 (все эксперименты) + exp-specific x0 ('stim') ---
def build_space(with_exp_entries):
    space = ParameterSpace.from_network(network)
    space.add_initial_state_entries(network)  # x0/exc/0 (common)
    if with_exp_entries:
        # x0/stim/exc/0 — только у эксперимента с name='stim'
        space.add_experiment_initial_state_entries(network, "stim")
    return space


space = build_space(with_exp_entries=True)
print("trainable-имена (координаты phi):", space.trainable_names)

problem = MultiExperimentProblem(network, space, experiments=[exp_control, exp_stim])

# Начальное приближение: общий x0 угадываем неверно (E=0.2), exp-specific —
# близко к истине; tau смещён от theta*=8.
phi0 = space.pack(
    {
        "exc/tau": tf.constant([[20.0]]),
        "x0/exc/0": tf.constant([[0.2]]),
        "x0/stim/exc/0": tf.constant([[0.75]]),
    }
)


def grad_by_name(sp, grad):
    """Раскладка плоского градиента по именам записей."""
    out = {}
    pos = 0
    for name in sp.trainable_names:
        numel = sp.entries[name].numel
        out[name] = float(np.asarray(grad[pos : pos + numel]).reshape(-1)[0])
        pos += numel
    return out


# --- 1. Различие траекторий при одном phi (exp-specific x0 имеет приоритет) ---
out_control = problem.simulate(phi0, experiment="control")
out_stim = problem.simulate(phi0, experiment="stim")
rate_c = np.asarray(out_control.firing_rates["exc"]).reshape(-1)
rate_s = np.asarray(out_stim.firing_rates["exc"]).reshape(-1)
print("\nПри одном и том же phi:")
print(f"  первый шаг 'control' (общий x0=0.2):   rate={rate_c[0]:.4f}")
print(f"  первый шаг 'stim'    (exp x0=0.75):    rate={rate_s[0]:.4f}")
print(f"  max |rate_control - rate_stim| = {np.max(np.abs(rate_c - rate_s)):.4f}")

# --- 2. Градиенты: общий x0 — сумма по экспериментам, exp x0 — только 'stim' ---
ev, grad = problem.value_and_gradient(phi0)
g_total = grad_by_name(space, grad)

# Одноэкспериментные задачи на отдельных пространствах для проверки
# разложения градиента.
space_c = build_space(with_exp_entries=False)
prob_c = MultiExperimentProblem(network, space_c, experiments=[exp_control])
_, grad_c = prob_c.value_and_gradient(
    space_c.pack({"exc/tau": tf.constant([[20.0]]), "x0/exc/0": tf.constant([[0.2]])})
)
g_c = grad_by_name(space_c, grad_c)

space_s = build_space(with_exp_entries=True)
prob_s = MultiExperimentProblem(network, space_s, experiments=[exp_stim])
phi0_s = space_s.pack(
    {
        "exc/tau": tf.constant([[20.0]]),
        "x0/exc/0": tf.constant([[0.2]]),
        "x0/stim/exc/0": tf.constant([[0.75]]),
    }
)
_, grad_s = prob_s.value_and_gradient(phi0_s)
g_s = grad_by_name(space_s, grad_s)

loss_c = prob_c.evaluate(
    space_c.pack({"exc/tau": tf.constant([[20.0]]), "x0/exc/0": tf.constant([[0.2]])})
).loss
loss_s = prob_s.evaluate(phi0_s).loss
print(
    f"\nloss(control+stim) = {ev.loss:.6f}  "
    f"(control={loss_c:.6f} + stim={loss_s:.6f} = {loss_c + loss_s:.6f})"
)
print("\nГрадиенты по phi (совместная задача):")
for name, value in g_total.items():
    print(f"  dloss/d[{name}] = {value:+.6f}")
print("\nПроверка разложения (одноэкспериментные задачи):")
print(
    f"  common x0: {g_total['x0/exc/0']:+.6f}  vs  "
    f"control {g_c['x0/exc/0']:+.6f} + stim {g_s['x0/exc/0']:+.6f} "
    f"= {g_c['x0/exc/0'] + g_s['x0/exc/0']:+.6f}"
)
print(
    "  (вклад 'stim' в общий x0 равен нулю: exp-запись перезаписала"
    " общий слот начального состояния)"
)
print(
    f"  exp x0 'stim': {g_total['x0/stim/exc/0']:+.6f}  vs  "
    f"только stim {g_s['x0/stim/exc/0']:+.6f} "
    f"(у 'control' этой записи нет — вклад не смешивается)"
)
