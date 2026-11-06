# NeuralTide

NeuralTide — Python-библиотека для дифференцируемого моделирования популяционных
нейронных сетей и калибровки их по экспериментальным данным. Вся динамика
вычисляется на TensorFlow, поэтому по любому прогону сети доступны точные
градиенты — а значит, параметры моделей можно подгонять под данные теми же
инструментами, что и в глубоком обучении, но с физически осмысленными
переменными (проводимости, постоянные времени, шум, связность).

## Какие задачи решает

Построение количественных моделей нейронной динамики — это не только
«запустить симуляцию». На практике приходится отвечать на вопросы:

- **Дифференцируемое моделирование.** Как получить градиенты по параметрам
  сквозь жёсткие ОДУ, синапсы с короткопериодической пластичностью и
  Fokker–Planck-операторы? NeuralTide даёт градиенты тремя способами —
  BPTT, analytical adjoint и forward-чувствительности — с встроенными
  проверками корректности (taylor/dot-product тесты).
- **Калибровка по данным.** Как подогнать параметры сети под записанные
  траектории (частоты разрядов, LFP-прокси, полевые потенциалы)? Пакет
  включает линз-градиентные оптимизаторы (L-BFGS-B, TRF/LM поверх SciPy),
  мультистарт с Sobol-дизайном, глобальный безградиентный поиск
  (DE, CMA-ES, SPSA, basinhopping, NSGA-II), multiple shooting и
  continuation-подходы с честным teacher forcing.
- **Идентифицируемость.** Какие параметры вообще можно восстановить из
  данных, а какие вырождены? Модуль идентифицируемости считает FIM,
  строит профили правдоподобия и помогает выбрать подмножество параметров.
- **Неопределённость.** Насколько точны оценки? Доступны Laplace-приближение,
  bootstrap и предиктивные интервалы прогонов.
- **Воспроизводимые постановки.** Сиды, логирование версий, сохранение
  состояния эксперимента, HDF5-датасеты и декларативный конфиг-API позволяют
  фиксировать постановку целиком и делиться ею.

## Основные возможности

### Модели

- **Популяции**: `IzhikevichMeanField` (mean-field квадратичного интегрирующего
  нейрона), `WilsonCowan` (rate-модель), `BrunelFokkerPlanck`
  (классическая модель A из Brunel, 2000, на сетке по мембранному
  потенциалу), `FokkerPlanckPopulation` как расширяемая основа для своих
  FP-моделей. Число юнитов выводится из параметров автоматически.
- **Синапсы**: `TsodyksMarkram` (короткопериодическая пластичность),
  `NMDA` с Mg²⁺-блоком, `SecondOrderSynapse` (двойная экспонента),
  `StaticSynapse`, `Composite` (составные проводимости, напр. AMPA+NMDA).
- **Входные генераторы**: `SinusoidalGenerator`, `ConstantRateGenerator`,
  `VonMisesGenerator` (обучаемый R), `PlaceFieldGenerator` (клетки места,
  фазовая прецессия). Внешние входы предвычисляются и подаются в сеть через
  `NetworkGraph.declare_input()` + `graph.pack_inputs()`.

### Интегрирование

- Явные интеграторы `EulerIntegrator`, `HeunIntegrator`, `RK4Integrator`.
- Жёсткие и адаптивные: `AdaptiveRKIntegrator` (BS23+PI с контролем шага),
  `IMEXEulerIntegrator`, `RosenbrockIntegrator` (ROS2),
  `ExponentialEulerIntegrator`.
- Режим сетевого интегрирования `NetworkRNN(integration_mode='network')`:
  единый шаг по всей связанной системе (`Euler/Heun/RK4/Rosenbrock`) —
  RK4 сохраняет ~4-й порядок на связанной сети, где шаг по популяциям
  по очереди (split-режим) ограничен 2-м.
- `stiffness_report()` — диагностика жёсткости по спектру якобиана
  (по популяциям или по сети целиком).

### Калибровка и оптимизация (`neuraltide.optimization`)

- `CalibrationProblem` / `ParameterSpace` — единый реестр оптимизируемых
  величин с дифференцируемыми преобразованиями координат
  (identity/affine/log/logit/scaled_sigmoid/softplus/bounded_log), включая
  общие и эксперимент-специфичные начальные состояния.
- `minimize_lbfgsb`, `least_squares` (TRF/LM) — адаптеры поверх SciPy.
- `multistart` (Sobol/LHS/random + доводка top-k),
  `global_search` (методы `de`, `cma`, `spsa`, `basinhopping`, `nsga2`).
- `MultipleShootingProblem` — сегменты с независимыми начальными
  состояниями и continuity-штрафом.
- Continuation-стратегии: рост горизонта, укрупнение шага dt,
  teacher forcing с обязательным контрольным свободным прогоном.
- `MultiExperimentProblem` — совместная калибровка по нескольким
  экспериментам в одной GradientTape.
- Мультиэкспериментные и робастные сценарии: балансировка компонент потерь
  (весовые стратегии, PCGrad/CAGrad), ограничения на прогнозы
  (augmented Lagrangian, epsilon-constraint), пакетная оценка кандидатов
  в параллельных процессах.

### Производительность и вычислительные режимы

- Скан симуляции и train step исполняются через `tf.function` с
  закреплёнными сигнатурами; опциональная XLA-компиляция —
  `NetworkRNN(jit_compile=True)` / `Trainer(jit_compile=True)`, у задач
  калибровки — флаг `compiled=True` (граф прогона+градиента). См.
  [docs/optimization.md](docs/optimization.md#компилируемый-режим-tffunction).
- На GPU рекомендуется float32: float64 работает, но заметно медленнее.

### Идентифицируемость и неопределённость

- `compute_fim` (спектр, effective rank, обусловленность),
  `likelihood_profile` (χ²-интервалы), `select_subset` (выбор подмножества
  параметров с оценкой смещения).
- UQ: `residual_diagnostics`, `laplace_approximation` (интервалы параметров),
  bootstrap (`bootstrap_experiments`, `residual_bootstrap`),
  `predictive_intervals` для траекторий.

### Данные, конфигурация, воспроизводимость

- HDF5-датасеты: `neuraltide.data.save_dataset` / `load_dataset` /
  `plot_dataset` для целевых траекторий и наблюдений.
- Конфиг-first API: `build_network_from_config` собирает сеть из
  декларативного описания (реестры классов подхватываются автоматически
  при импорте).
- `seed_everything`, `log_versions`, `save_experiment_state` —
  фиксация сидов, версий и состояния прогона.

## Установка

Пакет пока не опубликован на PyPI, поэтому установка выполняется из
исходников репозитория (после публикации на PyPI станет доступна
команда `pip install neuraltide`):

```bash
git clone https://github.com/ivanmysin/neuraltide.git
cd neuraltide
pip install -e "."
```

Для разработки используйте `pip install -e ".[dev]"`.

Дополнительные extras:

- `.[dev]` — pytest, pytest-cov, black, ruff (разработка);
- `.[global]` — CMA-ES для `global_search(method='cma')`;
- `.[pareto]` — pymoo для многокритериального поиска (`nsga2`);
- `.[vis]` — rich-вывод сводки модели (`print_summary`).

Зависимости: Python >= 3.12, TensorFlow >= 2.16, NumPy >= 1.26, SciPy >= 1.15
(обязательна: локальные оптимизаторы, стратегии калибровки и спектральные
вычисления; генераторы входов её не используют).

## Быстрый старт

```python
import numpy as np
import tensorflow as tf
from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.populations import IzhikevichMeanField
from neuraltide.synapses import StaticSynapse
from neuraltide.inputs import SinusoidalGenerator
from neuraltide.integrators import RK4Integrator
from neuraltide.utils import print_summary

# Создание генератора (n_units выводится автоматически)
gen = SinusoidalGenerator(
    dt=0.5,
    params={
        'amplitude': 10.0,
        'freq': 8.0,
        'phase': 0.0,
        'offset': 5.0,
    }
)

# Создание популяции (безразмерные параметры, n_units выводится из params)
pop = IzhikevichMeanField(
    dt=0.5,
    params={
        'tau_pop': [1.0],
        'alpha': [0.5],
        'a': [0.02],
        'b': [0.2],
        'w_jump': [0.1],
        'Delta_I': [0.5],
        'I_ext': [1.0],
    }
)

# Создание синапса
syn = StaticSynapse(
    n_pre=1,
    n_post=1,
    dt=0.5,
    params={
        'gsyn_max': 0.1,
        'pconn': [[1.0]],
        'e_r': 0.0,
    }
)

# Построение сети
graph = NetworkGraph(dt=0.5)
graph.declare_input('input', n_units=gen.n_units)   # внешний вход
graph.add_population('exc', pop)
graph.add_synapse('input->exc', syn, src='input', tgt='exc')

network = NetworkRNN(graph, integrator=RK4Integrator())
print_summary(network)

# Симуляция: входные частоты предвычисляются и упаковываются
t_seq = tf.constant(np.arange(0, 1000, 0.5)[None, :, None], dtype=tf.float32)
input_rates = gen(t_seq)                            # [1, T, n_units]
inputs = graph.pack_inputs({'input': input_rates})

output = network(t_seq, inputs=inputs)
print(f"Output shape: {output.firing_rates['exc'].shape}")
```

## Примеры

Подробные примеры — в директории `examples/`:

- `example_01_single_population.py` — одна популяция `IzhikevichMeanField`
  с тета-входом через синапс Tsodyks–Markram;
- `example_02_exc_inh_nmda.py` — E/I-сеть с NMDA-синапсами и частотно
  селективным readout;
- `example_03_custom_population.py` — собственная популяционная модель
  по минимальному контракту;
- `example_04_fokker_planck.py` — своя Fokker–Planck-модель через
  наследование;
- `example_05_lfp_proxy.py` — LFP-прокси как взвешенная сумма
  тета-фильтрованной активности;
- `example_06_brunel_fp.py` — `BrunelFokkerPlanck`: сходимость
  к стационару против аналитики;
- `example_07_calibration.py` — постановка задачи калибровки
  (`CalibrationProblem`, преобразования параметров, проверка градиента);
- `example_10_multiexperiment_x0.py` — совместная калибровка по нескольким
  экспериментам с общими и эксперимент-специфичными начальными состояниями;
- сквозные рабочие процессы калибровки — от сборки задачи до
  идентифицируемости и оценки неопределённости — в `example_08_*.py`
  и `example_09_*.py`.

Тематические примеры: `example_place_field_phase_precession.py`
(клетки места с фазовой прецессией), `example_pop_wilson_cowan.py`
(rate-модель), `example_pop_izhikevich.py`, `example_syn_nmda.py`,
`example_syn_tsodyks_markram.py`, `example_second_order_synapse.py`,
`example_syn_composite.py`, `example_von_mises.py`,
`example_multi_input_to_population.py`, `example_mutual_inhibition.py`,
`example_gradient_method.py`, `example_compare_gradients.py`,
`example_adjoint_direct.py`, `example_initial_state.py`.

См. также [docs/examples.md](docs/examples.md) для подробных описаний.

## Мануал и решение проблем

Пошаговый мануал и справочные материалы — в `docs/`:

| Раздел | Описание |
|--------|----------|
| [docs/guide/getting_started.md](docs/guide/getting_started.md) | Установка и первый прогон |
| [docs/guide/building_networks.md](docs/guide/building_networks.md) | Построение сети: популяции, синапсы, входы |
| [docs/guide/running_simulations.md](docs/guide/running_simulations.md) | Запуск симуляций и выбор интегратора |
| [docs/guide/calibration.md](docs/guide/calibration.md) | Калибровка по данным: постановка задачи и локальные оптимизаторы |
| [docs/guide/advanced_calibration.md](docs/guide/advanced_calibration.md) | Глобальный поиск, multiple shooting, continuation |
| [docs/optimization.md](docs/optimization.md) | Справочник слоя оптимизации: режимы вычислений (eager / tf.function / XLA), ограничения |
| [docs/troubleshooting.md](docs/troubleshooting.md) | Решение проблем: жёсткость, NaN, сходимость |
| [docs/examples.md](docs/examples.md) | Каталог примеров с описаниями |
| [docs/api_reference.md](docs/api_reference.md) | Полный API-справочник |

## Ограничения

- По умолчанию вычисления — в eager-режиме TensorFlow; опциональны
  компилируемые режимы (`tf.function`, XLA) — см.
  [docs/optimization.md](docs/optimization.md#компилируемый-режим-tffunction).
- Пакетные размеры (batch) больше 1 в текущей последовательной схеме
  `tf.while_loop` не поддерживаются.
- Требуется Python >= 3.12.
- Шаг по популяциям по очереди (split-режим, по умолчанию) ограничен
  2-м порядком на связанной сети; для высокого порядка используйте
  `integration_mode='network'` (см. мануал по симуляциям).

## Лицензия

MIT
