# Первые шаги

Это руководство — входная точка в NeuralTide. Здесь описано, для кого
пакет, как его установить, и за пять минут собирается первая сеть с
первой калибровкой параметров по данным. Дальше по цепочке идут
[Построение сети](building_networks.md), [Запуск симуляций](running_simulations.md),
[Калибровка по данным](calibration.md) и
[Расширенная калибровка](advanced_calibration.md).

## Для кого этот пакет

NeuralTide — библиотека для дифференцируемого моделирования и обучения
нейронных сетей, собранных из популяционных (mean-field) моделей. Она
решает три класса задач:

- **Моделирование динамики сетей** из популяций (Izhikevich mean-field,
  Wilson–Cowan, Fokker–Planck/Брюнель) с реалистичной синаптической
  динамикой (Tsodyks–Markram, NMDA, двойная экспонента) и внешними
  входами (синусоидальные, тета-ритмичные, поля места).
- **Подгонка параметров под экспериментальные данные** — от простого
  «подвинуть веса синапсов, чтобы траектория совпала с записью» до
  мультиэкспериментной калибровки с оценкой идентифицируемости и
  неопределённости параметров.
- **Обучение через градиенты.** Все вычисления — в TensorFlow, поэтому
  градиенты потерь по параметрам доступны напрямую (BPTT или
  adjoint-метод), а поверх них работают SciPy-оптимизаторы, мультистарт
  и методы глобального поиска.

Типичный пользователь — вычислительный нейробиолог, которому нужна
сеть из mean-field популяций с заданной топологией, или исследователь,
калибрующий такую сеть по снятым firing rate. Знание TensorFlow
полезно, но не обязательно: модель собирается декларативно из готовых
компонентов.

## Установка и требования

**Когда использовать.** Стандартная установка из PyPI подходит для
большинства пользователей; установка из исходников — для разработчиков
и тех, кому нужна конкретная версия из репозитория.

Требования: Python 3.12+, TensorFlow 2.x, NumPy. Пакет пока не опубликован
на PyPI, поэтому установка выполняется из исходников репозитория (после
публикации на PyPI станет доступна команда `pip install neuraltide`):

```bash
git clone https://github.com/ivanmysin/neuraltide.git
cd neuraltide
pip install -e "."
```

Для разработки используйте `pip install -e ".[dev]"`.

Опциональные зависимости:

- **h5py** — для сохранения/загрузки данных в HDF5 (`neuraltide.data`);
- **matplotlib** — только для визуализации.

SciPy — обязательная зависимость (ставится автоматически с пакетом):
его используют оптимизаторы и стратегии калибровки (`minimize_lbfgsb`,
`least_squares`, `multistart`, `global_search`, QMC-стартовый
дизайн) и спектрально-статистические вычисления (FIM, профили
правдоподобия, Laplace/bootstrap через `scipy.linalg`/`scipy.stats`).
Генераторы входов (`VonMisesGenerator`, `PlaceFieldGenerator` и остальные)
scipy не используют — им хватает TensorFlow (`tf.math.bessel_i0` и др.).

**Типичные ошибки и ограничения:**

- TensorFlow 2.x с поддержкой вашей версии Python — проверяйте
  совместимость до установки; Python 3.12+ обязателен.
- Без SciPy недоступны локальные оптимизаторы, стартовые дизайны и
  спектральные вычисления слоя калибровки (генераторы входов работают и
  так) — в штатной установке scipy уже включён в зависимости.

Для численно чувствительной работы (калибровка, жёсткие системы)
рекомендуется глобальный тип `float64` — он включается одной строкой до
создания любых моделей (подробности — в
[Запуске симуляций](running_simulations.md#тип-данных-float64-и-воспроизводимость)):

```python
import tensorflow as tf
import neuraltide.config

neuraltide.config.set_dtype(tf.float64)  # строго до создания моделей
```

## Ключевые понятия

Сеть в NeuralTide собирается из четырёх видов объектов:

| Объект | Роль | Основные классы |
|--------|------|-----------------|
| Популяция | Динамика mean-field, выдаёт firing rate | `IzhikevichMeanField`, `WilsonCowan`, `BrunelFokkerPlanck` |
| Синапс | Преобразует пресинаптическую частоту в ток постсинаптической популяции | `StaticSynapse`, `TsodyksMarkramSynapse`, `NMDASynapse`, `SecondOrderSynapse`, `CompositeSynapse` |
| Генератор входа | Предвычисляет частоты внешнего сигнала | `SinusoidalGenerator`, `ConstantRateGenerator`, `VonMisesGenerator`, `PlaceFieldGenerator` |
| Граф сети | Топология: кто с кем связан | `NetworkGraph`, `NetworkRNN` |

`NetworkRNN` — это Keras-слой, который прогоняет собранный граф по
временно́й сетке и возвращает firing rates всех популяций. Поверх
прогона строится задача калибровки: `ParameterSpace` (какие параметры
оптимизируем и в каких границах) + `ObservationSpec`/`ResidualLoss`
(что сравниваем с данными) + `CalibrationProblem` (единый интерфейс для
всех оптимизаторов).

## Пять минут: от сети до первой калибровки

Ниже — полный сквозной пример на малой сети: целевая траектория снята с
самой сети, затем параметры намеренно сдвинуты, и калибровка
восстанавливает их. Такой «самовосстановительный» сценарий — стандартный
способ проверить, что вся цепочка настроена верно, прежде чем брать
реальные данные.

```python
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
    minimize_lbfgsb,
    taylor_test,
)
from neuraltide.populations import WilsonCowan
from neuraltide.synapses import StaticSynapse
from neuraltide.utils import seed_everything

neuraltide.config.set_dtype(tf.float64)   # до создания моделей
seed_everything(42)

DT, T = 0.5, 80  # шаг и длительность, мс

# 1. Компоненты
pop = WilsonCowan(dt=DT, params={
    "tau":  {"value": [8.0],  "trainable": True, "min": 1.0, "max": 50.0},
    "a": 1.0, "theta": 0.0, "w": 0.2,
    "I_ext": {"value": [0.8], "trainable": True},
    "max_rate": 100.0,
})
syn = StaticSynapse(n_pre=1, n_post=1, dt=DT, params={
    "gsyn_max": {"value": [[30.0]], "trainable": True, "min": 0.1, "max": 1000.0},
    "pconn": [[1.0]],
    "e_r": 1.0,
})
gen = SinusoidalGenerator(dt=DT, params={
    "amplitude": 8.0, "freq": 8.0, "phase": 0.0, "offset": 10.0,
})

# 2. Топология
graph = NetworkGraph(dt=DT)
graph.declare_input("in", n_units=gen.n_units)
graph.add_population("exc", pop)
graph.add_synapse("in->exc", syn, src="in", tgt="exc")
network = NetworkRNN(graph, EulerIntegrator())

t_seq = tf.constant((np.arange(T, dtype=np.float64) * DT)[None, :, None])
inputs = graph.pack_inputs({"in": gen(t_seq)})

# 3. Цель: прогон при «истинных» параметрах
target = network(t_seq, inputs=inputs).firing_rates["exc"]

# 4. Пространство оптимизируемых параметров и сдвиг от истины
space = ParameterSpace.from_network(network, transforms={
    "exc/tau": LogTransform(),
    "in->exc/gsyn_max": LogTransform(),
})
space.apply_to_model({
    "exc/tau": tf.constant([[20.0]]),            # истина 8.0
    "exc/I_ext": tf.constant([[0.2]]),           # истина 0.8
    "in->exc/gsyn_max": tf.constant([[120.0]]),  # истина 30.0
})

objective = ResidualLoss(
    [ObservationSpec(name="exc", population="exc", target=target, sigma=2.0)]
)
problem = CalibrationProblem(network, space, t_seq, inputs=inputs, objective=objective)
print(f"loss до: {problem.evaluate().loss:.6f}")

# 5. Проверка градиента, затем локальная оптимизация
def value_and_grad(phi_np):
    ev, grad = problem.value_and_gradient(phi_np)
    return ev.loss, np.asarray(grad)

report = taylor_test(value_and_grad, np.asarray(problem.pack()) + 0.25)
print("Taylor test:", report["slope_R0_median"], report["slope_R1_median"])

result = minimize_lbfgsb(problem, maxiter=200)
print(f"loss после: {result.base.loss_best:.6f}")
theta_fit = space.unpack(result.base.phi_best)
for name in space.names:
    print(f"  {name}: {float(tf.reshape(theta_fit[name], [-1])[0]):.4f}")
```

Ожидаемый результат: `loss до` заметно больше нуля, наклоны Taylor test'а
близки к 1 и 2, `loss после` — на порядки меньше, а восстановленные
`exc/tau ≈ 8` и `in->exc/gsyn_max ≈ 30`. Если наклоны далеки от 1/2 —
градиент не прошёл проверку, и оптимизировать рано (см.
[Калибровку](calibration.md#верификация-задачи-до-оптимизации)).

Что здесь происходит по шагам:

1. Популяция, синапс и генератор создаются отдельно; параметры,
   отмеченные `trainable: True`, попадут в оптимизацию.
2. `NetworkGraph` связывает компоненты: внешний вход `in` подаётся на
   синапс `in->exc`, синапс — на популяцию `exc`.
3. `pack_inputs` упаковывает предвычисленные частоты входа в один
   тензор, который `NetworkRNN` разбирает по объявленным входам.
4. `ParameterSpace.from_network` собирает плоский вектор φ из
   обучаемых параметров; `LogTransform` переводит положительные величины
   в неограниченные координаты (экспонента гарантирует `θ > 0`).
5. `CalibrationProblem` — единая точка входа: `evaluate` (потеря),
   `value_and_gradient` (потеря и ∇ по φ), `residual` (вектор невязок).
6. `minimize_lbfgsb` — обёртка над SciPy L-BFGS-B с границами из
   `ParameterSpace`.

## Куда дальше

| Раздел | Что даёт |
|--------|----------|
| [Построение сети](building_networks.md) | Выбор популяций/синапсов/входов, формат параметров, `pconn`, рецепты сетей |
| [Запуск симуляций](running_simulations.md) | Выбор интегратора, режимы split/network-wide, начальные состояния, проверка корректности прогона |
| [Калибровка по данным](calibration.md) | Полный рецепт: от данных и целевой функции до оптимизаторов, мультистарта и multiple shooting |
| [Расширенная калибровка](advanced_calibration.md) | Идентифицируемость, неопределённость параметров, робастные потери, ограничения на прогнозы, параллельная оценка |

## Полезные ссылки

- [Примеры](../examples.md) — подробные разборы готовых сценариев;
  исходники — в директории `examples/` репозитория
  (`example_01_single_population.py` — одна популяция с обучением,
  `example_07_calibration.py` — первая калибровка);
- [API Reference](../api_reference.md) — полный справочник по классам и
  функциям;
- [Решение проблем](../troubleshooting.md) — чек-лист первого запуска и
  типичные ошибки;
- [Архитектура](../architecture.md) — устройство библиотеки;
- [README](../../README.md) — краткое описание и быстрый старт.
