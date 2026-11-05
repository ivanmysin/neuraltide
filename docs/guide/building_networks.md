# Построение сети

Этот раздел — про выбор компонентов и сборку топологии: какие бывают
популяции, синапсы и входные генераторы, как задавать параметры и
связность, как объявлять внешние входы и как описать сеть декларативно
через конфигурацию. За запуском собранной сети — в
[Запуск симуляций](running_simulations.md), за подгонкой параметров — в
[Калибровке](calibration.md).

## Из чего собирается сеть

Сеть — это направленный граф: **внешние входы** и **популяции** — узлы,
**синапсы** — рёбра. Популяция вычисляет firing rate по своей динамике;
синапс преобразует пресинаптическую частоту в синаптический ток;
генератор входа предвычисляет частоты внешнего сигнала, которые затем
подаются на синапсы от имени объявленного входа. Сборка выполняется
через `NetworkGraph`:

```python
from neuraltide.core.network import NetworkGraph

graph = NetworkGraph(dt=0.5)                        # общий шаг, мс
graph.declare_input("theta", n_units=gen.n_units)   # внешний вход
graph.add_population("exc", pop)                    # популяция
graph.add_synapse("theta->exc", syn, src="theta", tgt="exc")
graph.validate()                                    # проверка топологии
```

Имена, заданные здесь, — не ярлыки: они определяют адреса параметров в
задаче калибровки (`exc/Delta_I`, `in->exc/gsyn_max`, ...) и ключи
словаря `output.firing_rates`.

## Выбор популяции

**Когда использовать какую модель:**

| Популяция | Динамика | Когда использовать |
|-----------|----------|--------------------|
| `IzhikevichMeanField` | Mean-field адаптивная квадратичная (Izhikevich) | Основной выбор для возбуждающих/тормозных популяций с богатой внутренней динамикой (спайки, адаптация) |
| `WilsonCowan` | Rate-модель `τ·dE/dt = −E + F(w·E + I_ext + I_syn)` | Компактные rate-сети, E–I баланс, быстрые эксперименты; торможение — только через входящие синапсы |
| `BrunelFokkerPlanck` | Уравнение Фоккера–Планка для популяции ЛИФ-нейронов (Brunel 2000, модель A) | Одиночная популяция ЛИФ с точной статистикой пороговых событий; связь с сетью — через `mu_0 + I_syn` |

**Минимальные примеры:**

```python
from neuraltide.populations import IzhikevichMeanField, WilsonCowan

# Izhikevich mean-field, безразмерные параметры; n_units выводится
# из длины параметров (здесь — 2 юнита)
pop = IzhikevichMeanField(dt=0.5, params={
    "tau_pop": [1.0, 1.0], "alpha": [0.5, 0.5], "a": [0.02, 0.02],
    "b": [0.2, 0.2], "w_jump": [0.1, 0.1],
    "Delta_I": [0.5, 0.6], "I_ext": [0.1, 0.2],
})

# То же в размерных единицах (мВ, нФ, ...):
# params = {"V_rest": ..., "V_T": ..., "Cm": ..., "K": ..., "A": ...,
#           "B": ..., "W_jump": ..., "Delta_I": ..., "I_ext": ...}
# Пересчёт: IzhikevichMeanField.dimensionless_to_dimensional(...)

# Wilson–Cowan: обязательные параметры tau, a, theta, w, I_ext, max_rate
rate_pop = WilsonCowan(dt=0.5, params={
    "tau": 8.0, "a": 1.0, "theta": 0.0,
    "w": 0.2, "I_ext": 0.8, "max_rate": 100.0,
})
```

Для `BrunelFokkerPlanck` параметры — `tau_m`, `v_theta`, `v_reset`,
`tau_rp`, `mu_0`, `sigma_ext`; объект всегда одноюнитный (`n_units=1`),
состояние `[P, p_r, nu_buf]`. У этой модели есть особые правила
интегрирования — см. [Запуск симуляций](running_simulations.md#brunelfokkerplanck).

Базовый класс `FokkerPlanckPopulation` — абстрактный: свою FP-модель
наследуют от него и задают начальное распределение напряжений (готовый
способ — `self._gaussian_initial_state(center, sigma)`), см.
`examples/example_04_fokker_planck.py`.

**Типичные ошибки и ограничения:**

- `n_units` нельзя передать конструктором: он выводится из максимальной
  длины параметров. Несогласованные длины массивов — источник ошибок
  формы; проверяйте `pop.n_units`.
- У `WilsonCowan` все шесть параметров обязательны; отсутствующий `w`
  или `max_rate` — ошибка конструирования.
- Собственной тормозной динамики у `WilsonCowan` нет: торможение
  вводится синапсом с отрицательным `e_r` от тормозной популяции.

## Выбор синапса

**Когда использовать какую модель:**

| Синапс | Динамика | Когда использовать |
|--------|----------|--------------------|
| `StaticSynapse` | Мгновенный/первого порядка без пластичности | Простые связи, внешние входы, отладка топологии |
| `TsodyksMarkramSynapse` | Краткосрочная пластичность (депрессия/фасилитация) | Реалистичные AMPA/GABA связи с парой `tau_f`, `tau_d`, `tau_r`, `Uinc` |
| `NMDASynapse` | Двойная экспонента с Mg²⁺-блоком | Медленные NMDA-компоненты; потенциалозависимое проведение |
| `SecondOrderSynapse` | Непластичный фильтр 2-го порядка (двойная экспонента) | Когда нужна двойная экспонента без Mg²⁺-блока; поддерживает adjoint-метод |
| `CompositeSynapse` | Сумма нескольких компонентов | Одинаково названная связь из AMPA+STP и NMDA и т.п. |

**Минимальный пример:**

```python
from neuraltide.synapses import (
    StaticSynapse, TsodyksMarkramSynapse, NMDASynapse,
    SecondOrderSynapse, CompositeSynapse,
)

# Статический синапс 1 -> 2
syn = StaticSynapse(n_pre=1, n_post=2, dt=0.5, params={
    "gsyn_max": [[0.1, 0.1]], "pconn": [[1.0, 1.0]], "e_r": 0.0,
})

# Tsodyks–Markram (STP)
stp = TsodyksMarkramSynapse(n_pre=2, n_post=2, dt=0.5, params={
    "gsyn_max": 0.05, "tau_f": 20.0, "tau_d": 5.0, "tau_r": 200.0,
    "Uinc": 0.2, "pconn": [[1, 1], [1, 1]], "e_r": 0.0,
})

# NMDA с Mg2+-блоком
nmda = NMDASynapse(n_pre=2, n_post=2, dt=0.5, params={
    "gsyn_max_nmda": 0.05, "tau1_nmda": 2.0, "tau2_nmda": 100.0,
    "Mgb": 1.0, "av_nmda": 0.062,
    "pconn_nmda": [[1, 1], [1, 1]], "e_r_nmda": 0.0, "v_ref": 1.0,
})
```

Знак тока задаётся `e_r`: для возбуждающих связей `e_r` выше рабочего
потенциала постсинаптической популяции (в примерах часто `0.0` или
`1.0`), для тормозных — ниже (например, `−70.0` или `−0.5`).

**Типичные ошибки и ограничения:**

- Размерности жёстко связаны: `pconn` — матрица `[n_pre, n_post]`,
  несовпадение с `n_pre`/`n_post` даст ошибку формы в первом же прогоне.
- `NMDASynapse` и `TsodyksMarkramSynapse` используют собственные имена
  параметров (`gsyn_max_nmda`, `pconn_nmda`, ...) — не смешивайте их с
  именами `StaticSynapse`.
- Потенциалозависимые синапсы (NMDA) чувствительны к выбору режима
  интегрирования: в расщеплённом режиме они видят `v_mean` на границе
  шага, в согласованном — актуальный на каждой стадии (см.
  [split vs network-wide](running_simulations.md#режимы-интегрирования-split-и-network-wide)).

## Входные генераторы

**Когда использовать какой генератор:**

| Генератор | Сигнал | Когда использовать |
|-----------|--------|--------------------|
| `ConstantRateGenerator` | Постоянная частота `rate` | Тонический вход, смещение |
| `SinusoidalGenerator` | Синусоида `amplitude`, `freq`, `phase`, `offset` | Периодическая стимуляция |
| `VonMisesGenerator` | Периодический «тета-подобный» сигнал с концентрацией `R` | Тета-ритмичные входы; `R` может быть обучаемым |
| `PlaceFieldGenerator` | Частоты клеток места с тета-модуляцией и фазовой прецессией | Пространственная навигация; требует координат траектории |

**Минимальный пример:**

```python
from neuraltide.inputs import (
    ConstantRateGenerator, SinusoidalGenerator,
    VonMisesGenerator, PlaceFieldGenerator,
)

gen = SinusoidalGenerator(dt=0.5, params={
    "amplitude": 10.0, "freq": 8.0, "phase": 0.0, "offset": 5.0,
})
theta = VonMisesGenerator(dt=0.5, params={
    "mean_rate": 20.0, "R": 0.5, "freq": 8.0, "phase": 0.0,
})

# Клетки места: координаты (x, y) в см передаются при вызове
pf = PlaceFieldGenerator(dt=0.5, params={
    "center_x": [40.0, -50.0], "center_y": [30.0, 40.0],
    "radius": [35.0, 40.0], "peak_rate": [25.0, 30.0],
    "background_rate": [2.0, 3.0], "theta_modulation_factor": 0.0,
    "precession_slope": [30.0, 35.0], "precession_init_phase": [0.0, 90.0],
    "phase_outside": 0.0, "R": 0.6, "freq": 8.0,
})
rates = pf(t_seq, extra_inputs=extra)   # extra: [batch, T, n_cols], n_cols >= 2
```

Все генераторы принимают `t` формы `[batch, T, 1]`, `[batch, T]` или
`[T]` и возвращают `[batch, T, n_units]`.

**Типичные ошибки и ограничения:**

- `PlaceFieldGenerator` без координат (`extra_inputs=None`) выдаёт чисто
  фоновую частоту — пространственная компонента обнуляется.
- Входы не «вшиваются» в граф динамики: генератор предвычисляет частоты,
  дальше они подаются как обычный вход (см. следующий раздел). Старый
  класс `InputPopulation` объявлен устаревшим — используйте
  `declare_input()`.

## Формат params: скаляры, массивы и обучаемые величины

**Когда использовать.** Единый формат описания параметров действует для
всех компонентов — популяций, синапсов и генераторов. Он задаёт и
стартовые значения, и состав вектора оптимизации.

Каждый параметр любого компонента задаётся одним из двух способов:

- **сырое значение** — скаляр или список/матрица: параметр фиксирован,
  без ограничений;
- **словарь** `{"value": ..., "trainable": bool, "min": ..., "max": ...}` —
  `value` обязателен; `trainable=True` включает параметр в оптимизацию,
  `min`/`max` задают физические границы (через `MinMaxConstraint`).

**Минимальный пример:**

```python
pop = WilsonCowan(dt=0.5, params={
    "tau": {"value": [8.0], "trainable": True, "min": 1.0, "max": 50.0},
    "a": 1.0,                     # фиксированный скаляр
    "theta": 0.0,
    "w": 0.2,
    "I_ext": {"value": [0.8], "trainable": True},   # без явных границ
    "max_rate": 100.0,
})
```

Длина векторных параметров определяет `n_units` популяции; для
синапсов размеры явные — `n_pre`, `n_post` — и все матричные параметры
(`pconn`, `gsyn_max`) должны иметь согласованные формы.

**Типичные ошибки и ограничения:**

- Границы в пользовательском словаре называются `min`/`max`, а конструктор
  внутреннего ограничения — `MinMaxConstraint(min_val=..., max_val=...)`;
  ручную сборку ограничений выполняйте через `neuraltide.constraints`.
- `trainable: True` без `min`/`max` допустимо, но для устойчивой
  калибровки границы почти всегда нужны — иначе оптимизатор может уйти
  в нефизичную область.
- Для положительных по смыслу параметров (`tau`, `gsyn_max`, ...) в
  калибровке используйте `LogTransform`/`BoundedLogTransform` (см.
  [Калибровку](calibration.md#пространство-параметров-и-преобразования)) —
  границы тогда соблюдаются структурно.

## Связность: pconn и матрицы синапсов

**Когда использовать.** Каждый синапс описывает связь одной пары
«пре → пост»; связность внутри пары задаётся матрицей `pconn`, а
множественные синаптические механизмы между теми же популяциями —
несколькими синапсами или `CompositeSynapse`.

Полная связность между популяциями задаётся матрицей вероятностей
связи `pconn` формы `[n_pre, n_post]`: элемент `(i, j)` — доля связей от
юнита `i` пресинаптической популяции к юниту `j` постсинаптической.
Единичная матрица — полная связь, нули — отсутствие связей, промежуточные
значения — частичная (при этом `gsyn_max` обычно понимается как усиление
единичной связи, масштабированное `pconn`).

**Минимальный пример:**

```python
# 4 пресинаптических юнита -> 2 постсинаптических, полная связь
syn = TsodyksMarkramSynapse(n_pre=4, n_post=2, dt=0.5, params={
    "gsyn_max": 0.08,
    "tau_f": 20.0, "tau_d": 5.0, "tau_r": 200.0, "Uinc": 0.2,
    "pconn": [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
    "e_r": 0.0,
})
```

**Типичные ошибки и ограничения:**

- Скаляр `pconn` допускается как сокращение для однородной связи, но при
  `n_pre > 1, n_post > 1` явная матрица защищает от ошибок раскладки.
- Тормозная связь делается не знаком `pconn`, а отрицательным `e_r`.
- Число юнитов популяций в паре «источник → цель» должно совпадать с
  `n_pre`/`n_post` синапса.

## Внешние входы: declare_input и pack_inputs

**Когда использовать.** Любой сигнал извне (стимуляция, тета-генератор,
координаты животного) объявляется именованным входом графа. Частоты
входа предвычисляются генератором, упаковываются в один тензор и подаются
в `NetworkRNN.call(t_seq, inputs=...)`. Такой приём убирает генератор из
динамики сети и делает вход полностью контролируемым — в том числе при
калибровке (входы можно заменять, смешивать с данными и т.д.).

**Минимальный пример:**

```python
graph = NetworkGraph(dt=0.5)
graph.declare_input("theta", n_units=gen.n_units)   # объявление входа
graph.add_population("exc", pop)
graph.add_synapse("theta->exc", syn, src="theta", tgt="exc")

network = NetworkRNN(graph, integrator=RK4Integrator())

t_seq = tf.constant((np.arange(200, dtype=np.float32) * 0.5)[None, :, None])
inputs = graph.pack_inputs({"theta": gen(t_seq)})   # упаковка частот

output = network(t_seq, inputs=inputs)
```

Порядок ключей в `pack_inputs` может быть любым — упаковка идёт по
раскладке `graph.input_offsets`. Если входов несколько, каждому — свой
слой в тензоре `inputs` формы `[batch, T, total_input_units]`.

**Типичные ошибки и ограничения:**

- Если входы объявлены, вызов без `inputs` — `ValueError`: сеть требует
  полный тензор входов.
- Размерность `n_units` в `declare_input` должна совпадать с фактической
  размерностью упакованных частот.
- Устаревший `InputPopulation` внутри полного прогона не поддержан —
  только `declare_input()` + `pack_inputs()`.

## Топология: NetworkGraph

**Когда использовать.** Ручная сборка через `NetworkGraph` — основной
способ, дающий полный контроль: произвольное число популяций, входов и
синапсов, включая рекуррентные и композитные связи.

**Минимальный пример** — минимальная связная сеть с двумя популяциями:

```python
graph = NetworkGraph(dt=0.5)
graph.declare_input("in", n_units=1)
graph.add_population("exc", exc)
graph.add_population("inh", inh)
graph.add_synapse("in->exc",  s_in,  src="in",  tgt="exc")
graph.add_synapse("exc->inh", s_ei,  src="exc", tgt="inh")
graph.add_synapse("inh->exc", s_ie,  src="inh", tgt="exc")   # рекуррентное торможение
graph.validate()
```

Полезные свойства графа: `population_names`, `synapse_names`,
`input_names`, `input_offsets`, `total_input_units`.

**Типичные ошибки и ограничения:**

- `src`/`tgt` — имена уже добавленных узлов; опечатка в имени всплывёт
  в `validate()` или в первом прогоне.
- Одна популяция внутри графа — один объект; дважды добавить узел с тем
  же именем нельзя.
- Полный список методов — в [API Reference](../api_reference.md#neuraltidecorenetwork)
  и [Сети](../network.md).

## Config-first API: сборка из конфигурации

**Когда использовать.** Если сеть описывается декларативно — например,
хранится в файле или генерируется — используйте схему конфигурации:
`NetworkConfig` с `PopulationConfig`/`SynapseConfig`/`InputConfig` и
фабрику `build_network_from_config`. Классы подбираются из реестров
(`POPULATION_REGISTRY`, `SYNAPSE_REGISTRY`, `INPUT_REGISTRY`), поэтому
работают и пользовательские компоненты, зарегистрированные через
`neuraltide.config.register_*`.

**Минимальный пример:**

```python
from neuraltide.config.schema import (
    InputConfig, NetworkConfig, PopulationConfig, SynapseConfig,
    build_network_from_config,
)

config = NetworkConfig(
    dt=0.5,
    integrator="rk4",          # 'euler' | 'heun' | 'rk4'
    inputs=[
        InputConfig(
            name="in",
            generator_class="SinusoidalGenerator",
            params={"dt": 0.5, "params": {
                "amplitude": 8.0, "freq": 8.0, "phase": 0.0, "offset": 10.0,
            }},
        ),
    ],
    populations=[
        PopulationConfig(
            name="exc", model_class="WilsonCowan", dt=0.5,
            params={"tau": 8.0, "a": 1.0, "theta": 0.0,
                    "w": 0.2, "I_ext": 0.8, "max_rate": 100.0},
        ),
    ],
    synapses=[
        SynapseConfig(
            name="in->exc", synapse_class="StaticSynapse",
            src="in", tgt="exc", dt=0.5,
            params={"n_pre": 1, "n_post": 1,
                    "gsyn_max": 30.0, "pconn": [[1.0]], "e_r": 1.0},
        ),
    ],
)

network, generators = build_network_from_config(config)
```

**Типичные ошибки и ограничения:**

- Фабрика возвращает только `(network, generators)` и не отдаёт наружу
  построенный `NetworkGraph`. Чтобы упаковать входы для прогона,
  соберите ту же раскладку входов на собственном графе (порядок объявлений
  должен совпадать с `config.inputs`):

  ```python
  graph = NetworkGraph(dt=0.5)
  for name, gen in generators.items():
      graph.declare_input(name, n_units=gen.n_units)
  inputs = graph.pack_inputs({name: gen(t_seq) for name, gen in generators.items()})
  ```

  Если упаковка входов — центральная часть сценария, удобнее собирать
  сеть вручную через `NetworkGraph`.
- Имена классов — строки из реестров; неизвестное имя — `ValueError`.
- Параметры генератора вкладываются как `{"dt": ..., "params": {...}}`;
  параметры синапса должны включать `n_pre`/`n_post`.
- Подробности схемы — в [Конфигурации](../config.md).

## Рецепты

**Когда использовать.** Готовые сочетания компонентов для типовых
задач — берите ближайший как основу и подставляйте свои параметры.
Каждый рецепт — минимальный рабочий каркас топологии.

### Рецепт 1. Одна популяция с тета-входом

`IzhikevichMeanField` (2 юнита) + тета-вход через синапс Цодыкса–Маркрама
+ рекуррентный STP-синапс. Полный обучающий сценарий на этой топологии —
в `examples/example_01_single_population.py`.

```python
pop = IzhikevichMeanField(dt=0.5, params={
    "tau_pop": [1.0, 1.0], "alpha": [0.5, 0.5], "a": [0.02, 0.02],
    "b": [0.2, 0.2], "w_jump": [0.1, 0.1],
    "Delta_I": [0.5, 0.6], "I_ext": [0.1, 0.2],
})
gen = VonMisesGenerator(dt=0.5, params={
    "mean_rate": 20.0, "R": 0.5, "freq": 8.0, "phase": 0.0,
})
syn_in = TsodyksMarkramSynapse(n_pre=1, n_post=2, dt=0.5, params={
    "gsyn_max": [[0.1, 0.1]], "tau_f": 20.0, "tau_d": 5.0,
    "tau_r": 200.0, "Uinc": 0.2, "pconn": [[1.0, 1.0]], "e_r": 1.0,
})
syn_rec = TsodyksMarkramSynapse(n_pre=2, n_post=2, dt=0.5, params={
    "gsyn_max": 0.05, "tau_f": 20.0, "tau_d": 5.0,
    "tau_r": 200.0, "Uinc": 0.2, "pconn": [[1, 1], [1, 1]], "e_r": 1.0,
})

graph = NetworkGraph(dt=0.5)
graph.declare_input("theta", n_units=gen.n_units)
graph.add_population("exc", pop)
graph.add_synapse("theta->exc", syn_in, src="theta", tgt="exc")
graph.add_synapse("exc->exc", syn_rec, src="exc", tgt="exc")
```

### Рецепт 2. Сеть E–I с балансом возбуждения и торможения

Две популяции `WilsonCowan`: возбуждающая получает вход, тормозная
тормозит возбуждающую. Торможение — синапс с отрицательным `e_r`.

```python
exc = WilsonCowan(dt=0.5, params={
    "tau": 8.0, "a": 1.0, "theta": 0.0, "w": 0.2,
    "I_ext": 0.8, "max_rate": 100.0,
})
inh = WilsonCowan(dt=0.5, params={
    "tau": 12.0, "a": 1.0, "theta": 0.0, "w": 0.1,
    "I_ext": 0.3, "max_rate": 100.0,
})
s_in = StaticSynapse(n_pre=1, n_post=1, dt=0.5, params={
    "gsyn_max": [[30.0]], "pconn": [[1.0]], "e_r": 1.0,
})
s_ei = StaticSynapse(n_pre=1, n_post=1, dt=0.5, params={
    "gsyn_max": [[10.0]], "pconn": [[1.0]], "e_r": 1.0,
})
s_ie = StaticSynapse(n_pre=1, n_post=1, dt=0.5, params={
    "gsyn_max": [[10.0]], "pconn": [[1.0]], "e_r": -0.5,
})

graph = NetworkGraph(dt=0.5)
graph.declare_input("in", n_units=1)
graph.add_population("exc", exc)
graph.add_population("inh", inh)
graph.add_synapse("in->exc",  s_in,  src="in",  tgt="exc")
graph.add_synapse("exc->inh", s_ei,  src="exc", tgt="inh")
graph.add_synapse("inh->exc", s_ie,  src="inh", tgt="exc")
```

### Рецепт 3. Связь с краткосрочной пластичностью и NMDA (Composite)

Рекуррентная связь `exc → exc` из двух параллельных компонентов:
AMPA-подобный `TsodyksMarkramSynapse` и `NMDASynapse`, объединённых
`CompositeSynapse`. Похожая топология (с популяциями `IzhikevichMeanField`)
разобрана в `examples/example_02_exc_inh_nmda.py`.

```python
ampa = TsodyksMarkramSynapse(n_pre=4, n_post=4, dt=0.5, params={
    "gsyn_max": 0.05, "tau_f": 20.0, "tau_d": 5.0, "tau_r": 200.0,
    "Uinc": 0.2, "pconn": [[1] * 4] * 4, "e_r": 0.0,
})
nmda = NMDASynapse(n_pre=4, n_post=4, dt=0.5, params={
    "gsyn_max_nmda": 0.05, "tau1_nmda": 2.0, "tau2_nmda": 100.0,
    "Mgb": 1.0, "av_nmda": 0.062,
    "pconn_nmda": [[1] * 4] * 4, "e_r_nmda": 0.0, "v_ref": 1.0,
})
rec = CompositeSynapse(
    n_pre=4, n_post=4, dt=0.5,
    components=[("ampa", ampa), ("nmda", nmda)],
)

graph = NetworkGraph(dt=0.5)
graph.declare_input("in", n_units=1)
graph.add_population("exc", exc)   # IzhikevichMeanField с 4 юнитами
graph.add_synapse("in->exc", syn_in, src="in", tgt="exc")
graph.add_synapse("exc->exc", rec, src="exc", tgt="exc")
```

### Рецепт 4. Вход «клетки места» (PlaceField)

Координаты траектории `(x, y)` в см передаются генератору через
`extra_inputs`, частоты упаковываются как обычный вход. Подробности —
в [Генераторах входов](../inputs.md#placefieldgenerator) и
`examples/example_place_field_phase_precession.py`.

```python
gen = PlaceFieldGenerator(dt=0.5, params={
    "center_x": [40.0, -50.0], "center_y": [30.0, 40.0],
    "radius": [35.0, 40.0], "peak_rate": [25.0, 30.0],
    "background_rate": [2.0, 3.0], "theta_modulation_factor": 0.0,
    "precession_slope": [30.0, 35.0], "precession_init_phase": [0.0, 90.0],
    "phase_outside": 0.0, "R": 0.6, "freq": 8.0,
})

graph = NetworkGraph(dt=0.5)
graph.declare_input("place", n_units=gen.n_units)
graph.add_population("readout", pop)   # например, IzhikevichMeanField, 2 юнита
graph.add_synapse("place->readout", syn, src="place", tgt="readout")

network = NetworkRNN(graph, integrator=RK4Integrator())

t_seq = tf.constant(np.arange(0, 5000, 0.5, dtype=np.float32)[None, :, None])
n_steps = t_seq.shape[1]
pos_x = 70.0 * np.cos(2 * np.pi * np.arange(n_steps) / n_steps * 2)
pos_y = 70.0 * np.sin(2 * np.pi * np.arange(n_steps) / n_steps * 2)
extra = tf.constant(np.stack([pos_x, pos_y], axis=-1)[None, :, :])

rates = gen(t_seq, extra_inputs=extra)
inputs = graph.pack_inputs({"place": rates})
output = network(t_seq, inputs=inputs)
```

## Полезные ссылки

- [Популяции](../populations.md), [Синапсы](../synapses.md),
  [Генераторы входов](../inputs.md) — детальные описания каждого класса;
- [Сеть](../network.md) — `NetworkGraph` и `NetworkRNN`;
- [Примеры](../examples.md) — разборы `example_01`–`example_06` и
  синаптических примеров; исходники — в `examples/`;
- [API Reference](../api_reference.md) — сигнатуры конструкторов;
- [Решение проблем](../troubleshooting.md) — если сеть не собирается или
  прогон ведёт себя странно.
