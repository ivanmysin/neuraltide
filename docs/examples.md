# Примеры использования

Все примеры — исполняемые скрипты в каталоге [`examples/`](../examples/),
запускаются из корня репозитория:

```bash
python examples/example_simple.py
```

Разделы сгруппированы по задачам: от построения и симуляции сети до
обучения и калибровки параметров.

---

## Базовая симуляция сетей

| Пример | Файл | Что показывает |
|--------|------|----------------|
| Простая сеть | [`example_simple.py`](../examples/example_simple.py) | Базовая схема «вход → синапс → популяция»: синусоидальный генератор, `IzhikevichMeanField`, обучаемый `TsodyksMarkramSynapse`; сеть обучается воспроизводить заданную траекторию firing rate |
| Одна популяция с тета-входом | [`example_01_single_population.py`](../examples/example_01_single_population.py) | `IzhikevichMeanField` (n_units=2) с тета-ритмическим входом через обучаемый синапс; воспроизведение целевой firing rate |
| Сеть E–I с NMDA | [`example_02_exc_inh_nmda.py`](../examples/example_02_exc_inh_nmda.py) | Две популяции (exc + inh), синапсы AMPA+NMDA / AMPA / GABA_A, частичные наблюдения (target только для exc), `BandpassReadout` — loss в тета-диапазоне |
| Несколько входов на популяции | [`example_multi_input_to_population.py`](../examples/example_multi_input_to_population.py) | Два `VonMisesGenerator` (30/40 Гц) → две популяции через синапс с диагональной матрицей связей |
| Управление начальным состоянием | [`example_initial_state.py`](../examples/example_initial_state.py) | Пользовательские начальные условия (r, v, w), продолжение симуляции через `final_state`, сброс к нулевому состоянию |

Общий паттерн построения сети (все примеры выше):

```python
graph = NetworkGraph(dt=0.5)
graph.declare_input('input', n_units=gen.n_units)
graph.add_population('exc', pop)
graph.add_synapse('input->exc', syn, src='input', tgt='exc')

network = NetworkRNN(graph, integrator=RK4Integrator())

t_seq = tf.constant(t_values[None, :, None])       # [1, T, 1]
inputs = graph.pack_inputs({'input': gen(t_seq)})  # предвычисленные частоты
output = network(t_seq, inputs=inputs)             # output.firing_rates['exc']
```

### Продолжение прогонов

Прогоны стыкуются через `initial_state`: конечное состояние одного прогона
(`output.final_state`) передаётся в начальное следующего. Сброс к состоянию
по умолчанию — через `network.get_initial_state()`:

```python
output1 = network(t_seq, inputs=inputs)                          # первый прогон
output2 = network(t_seq, inputs=inputs,
                  initial_state=output1.final_state)             # продолжение

init_pop, init_syn = network.get_initial_state(batch_size=1)     # сброс
output3 = network(t_seq, inputs=inputs,
                  initial_state=(init_pop, init_syn))
```

Подробности — в [руководстве по запуску симуляций](guide/running_simulations.md#начальные-состояния-и-продолжение-прогонов).

### Построение сети из конфигурации

Сеть можно собрать декларативно через `NetworkConfig` (классы
регистрируются при импорте `neuraltide.populations` / `neuraltide.synapses`
/ `neuraltide.inputs`):

```python
from neuraltide.config.schema import (
    NetworkConfig, PopulationConfig, SynapseConfig, InputConfig,
    build_network_from_config,
)

config = NetworkConfig(
    dt=0.5,
    integrator='rk4',
    inputs=[
        InputConfig(
            name='input',
            generator_class='SinusoidalGenerator',
            params={'dt': 0.5, 'params': {
                'amplitude': 10.0, 'freq': 8.0, 'phase': 0.0, 'offset': 5.0}},
        ),
    ],
    populations=[
        PopulationConfig(
            name='exc',
            model_class='IzhikevichMeanField',
            dt=0.5,
            params={
                'tau_pop': [1.0], 'alpha': [0.5], 'a': [0.02],
                'b': [0.2], 'w_jump': [0.1], 'Delta_I': [0.5], 'I_ext': [1.0],
            },
        ),
    ],
    synapses=[
        SynapseConfig(
            name='input->exc',
            synapse_class='StaticSynapse',
            src='input', tgt='exc', dt=0.5,
            params={'n_pre': 1, 'n_post': 1, 'gsyn_max': 0.1,
                    'pconn': [[1.0]], 'e_r': 0.0},
        ),
    ],
)

network, generators = build_network_from_config(config)

t_seq = tf.constant(np.arange(0, 1000, 0.5)[None, :, None])
inputs = graph.pack_inputs({name: gen(t_seq) for name, gen in generators.items()})
output = network(t_seq, inputs=inputs)
```

---

## Популяционные модели

| Пример | Файл | Что показывает |
|--------|------|----------------|
| Izhikevich mean-field | [`example_pop_izhikevich.py`](../examples/example_pop_izhikevich.py) | Безразмерные параметры (рекомендуется) и размерные (обратная совместимость), векторизация по юнитам, обучаемые параметры с ограничениями |
| Wilson-Cowan | [`example_pop_wilson_cowan.py`](../examples/example_pop_wilson_cowan.py) | Запуск активности постоянным током, установление в фиксированную точку, две независимые популяции с разными параметрами |
| Пользовательская популяция | [`example_03_custom_population.py`](../examples/example_03_custom_population.py) | Как наследовать `PopulationModel` (минимальная rate-модель `τ·dr/dt = −r + tanh(I_ext + I_syn)`) и подключить её к сети |
| Fokker-Planck | [`example_04_fokker_planck.py`](../examples/example_04_fokker_planck.py) | Пользовательская FP-модель с процессом Орнштейна–Уленбека |
| Brunel Fokker-Planck | [`example_06_brunel_fp.py`](../examples/example_06_brunel_fp.py) | `BrunelFokkerPlanck`: сходимость к стационарной частоте, сравнение с аналитическим решением Brunel (2000), сохранение полной вероятности ∫P dV + p_r = 1 |

---

## Синапсы и входные генераторы

| Пример | Файл | Что показывает |
|--------|------|----------------|
| StaticSynapse | [`example_syn_static.py`](../examples/example_syn_static.py) | Синапс без внутренней динамики: g_syn пропорциональна входной частоте, мгновенный отклик |
| Tsodyks-Markram | [`example_syn_tsodyks_markram.py`](../examples/example_syn_tsodyks_markram.py) | Кратковременная пластичность: фракции R/U, аккумулятор A, ток и проводимость |
| Tsodyks-Markram (численно) | [`example_syn_tsodyks_markram_numeric.py`](../examples/example_syn_tsodyks_markram_numeric.py) | То же через интегратор (RK4) — синапсы интегрируются численно, аналогично популяциям |
| NMDA | [`example_syn_nmda.py`](../examples/example_syn_nmda.py) | Двойная экспонента, магниевый блок от постсинаптического потенциала |
| Composite (AMPA + NMDA) | [`example_syn_composite.py`](../examples/example_syn_composite.py) | `CompositeSynapse`: быстрая AMPA-компонента + медленная NMDA, суммарный ток |
| Синапс второго порядка | [`example_second_order_synapse.py`](../examples/example_second_order_synapse.py) | `SecondOrderSynapse`: фильтр второго порядка (двойная экспонента), состояния [g_s, dg_s] |
| VonMises | [`example_von_mises.py`](../examples/example_von_mises.py), [`example_von_mises_with_synapse.py`](../examples/example_von_mises_with_synapse.py) | Тета-модулированный генератор и его связка с TM-синапсом (динамика R, U, A и ток на популяцию) |

### PlaceFieldGenerator — прямой вызов с фазовой прецессией

Полный пример: [`example_place_field_phase_precession.py`](../examples/example_place_field_phase_precession.py)
— прямой вызов генератора с координатами `(x, y)` в см через `extra_inputs`,
зашумлённая круговая траектория, диагностический вывод параметров и три
графика (rate vs theta, фазовая прецессия, карта арены).

```python
gen = PlaceFieldGenerator(dt=dt, params={
    'center_x': [40.0, -50.0, ...], 'center_y': [30.0, 40.0, ...],
    'radius': [35.0, 40.0, ...], 'peak_rate': [25.0, 30.0, ...],
    'background_rate': [2.0]*6,
    'theta_modulation_factor': 0.0,
    'precession_slope': [30.0, ...], 'precession_init_phase': [0.0, ...],
    'phase_outside': 0.0, 'R': 0.6, 'freq': 8.0,
})

# extra_inputs: [batch, T, 2] — (x, y) в см, или [batch, T, 4] — (x, y, vx, vy)
rates = gen(t_seq, extra_inputs=extra)   # [1, T, n_place_cells]
```

Поток данных:

```
pos_x, pos_y, vx, vy (см, см/мс)  ──→  extra_inputs [batch, T, 4]
                              │
    t [batch, T, 1]  ────────┤
                              │
                    PlaceFieldGenerator.call(t, extra_inputs=extra)
                              │
                    rates [batch, T, n_units]  (Hz)
                              │
              dphi = -slope * ((px-cx)*vx + (py-cy)*vy) / |v|
              (различает заход в поле и выход)
                              │
                    graph.pack_inputs({'place': rates})
                              │
                    inputs [batch, T, total_input_units]
                              │
                    NetworkRNN.call(t_seq, inputs=inputs)
                              │
                    output.firing_rates['readout']
```

Примечания:

- `inputs` опционален (по умолчанию `None`). При `None` и нулевом `total_input_units` генераторы не используются.
- `inputs` передаётся в `NetworkRNN.call()` как единый тензор, упакованный через `graph.pack_inputs()`.
- Генераторы `SinusoidalGenerator`, `ConstantRateGenerator`, `VonMisesGenerator` игнорируют `extra_inputs`.

---

## Обучение

| Пример | Файл | Что показывает |
|--------|------|----------------|
| Обучение через Trainer | [`example_simple.py`](../examples/example_simple.py) | `Trainer` + `CompositeLoss([MSELoss, StabilityPenalty])`, воспроизведение целевой траектории |
| Сравнение градиентных методов | [`example_gradient_method.py`](../examples/example_gradient_method.py), [`example_compare_gradients.py`](../examples/example_compare_gradients.py) | Autograd (BPTT) против adjoint: равенство градиентов, замеры времени и памяти |
| Прямой AdjointSolver | [`example_adjoint_direct.py`](../examples/example_adjoint_direct.py) | Использование `AdjointSolver` без `Trainer`, градиенты для длинных последовательностей |
| Adjoint + штраф за стабильность | [`example_adjoint_with_stability.py`](../examples/example_adjoint_with_stability.py) | Adjoint-обучение с `StabilityPenalty` на сети, склонной к нестабильности |
| Взаимное ингибирование | [`example_mutual_inhibition.py`](../examples/example_mutual_inhibition.py) | Две fast-spiking популяции с взаимным торможением через TM-синапсы (депрессия); оптимизация проводимостей и внешних токов; экспорт результатов |
| Демо нестабильности | [`example_instability_demo.py`](../examples/example_instability_demo.py) | Сеть с высоким `I_ext` и `stability_penalty_weight` — как работает штраф за численную нестабильность |
| LFP-прокси | [`example_05_lfp_proxy.py`](../examples/example_05_lfp_proxy.py) | LFP-подобный сигнал как взвешенная сумма частот популяций, отфильтрованная в тета-диапазоне (`BandpassReadout`), сравнение с целью через `MSELoss` |

Краткий паттерн обучения:

```python
from neuraltide.training import Trainer, CompositeLoss, MSELoss, StabilityPenalty

loss_fn = CompositeLoss([
    (1.0, MSELoss(target)),      # основная ошибка
    (1e-3, StabilityPenalty()),  # штраф за нестабильность
])
trainer = Trainer(network, loss_fn, optimizer=tf.keras.optimizers.Adam(1e-3))
history = trainer.fit(t_seq, inputs=inputs, epochs=500, verbose=1)

# Экспорт результатов
trainer.export_results('results.json')
trainer.export_results('results.csv', format='csv')
```

Больше о градиентных методах (BPTT/adjoint), колбэках и ограничениях —
в [`docs/training.md`](training.md).

---

## Калибровка и оптимизация

Сквозные сценарии слоя `neuraltide.optimization`. Ограничения и полное
описание API — в [`docs/optimization.md`](optimization.md).

### Калибровка: постановка задачи и верификация градиентов

Полный пример: [`example_07_calibration.py`](../examples/example_07_calibration.py)

Сборка `ParameterSpace` из сети с преобразованиями координат (log — для
положительных параметров), постановка задачи через `ObservationSpec`/
`ResidualLoss` и `CalibrationProblem`, верификация градиента `taylor_test`ом
и ручные шаги Adam по вектору φ с сохранением результата в
`OptimizationResult`.

### Сквозной рабочий процесс калибровки

Полный пример: [`example_08_p1_workflow.py`](../examples/example_08_p1_workflow.py)

Практическая цепочка на малой синтетической сети (WilsonCowan +
StaticSynapse от SinusoidalGenerator) с известным θ*:

1. **multistart** — детерминированный Sobol-дизайн стартов, screening и
   локальная доводка top-k через `least_squares` (TRF);
2. **residual_jacobian** — якобиан невязок в точке решения (forward
   sensitivities, по столбцам через unit-JVP);
3. **compute_fim** — FIM = JᵀJ: спектр, effective rank, число
   обусловленности, жёсткие направления и коллинеарные пары параметров;
4. **likelihood_profile** — одномерный профиль правдоподобия по `exc/tau` с
   χ²-интервалом 95 % (warm start от соседнего узла);
5. **validate / free-run** — отложенный протокол (другая частота входа) и
   честный свободный прогон без teacher forcing.

```python
ms = multistart(problem, n_starts=8, method="sobol", refine_top_k=3,
                local="least_squares", seed=7)
phi_best = np.asarray(ms.base.phi_best)

jac = residual_jacobian(problem, phi_best)
fim = compute_fim(jac, param_names=space.names)

profile = likelihood_profile(problem, "exc/tau",
                             grid=np.linspace(4.0, 14.0, 6))
print(profile.intervals["0.95"])          # 95% интервал по chi2(1)

validation = problem.validate(phi_best)   # held-out протокол
ev_free = problem.evaluate(phi_best)      # честный free-run
```

### Глобальный поиск, балансировка и неопределённость

Полный пример: [`example_09_p2_workflow.py`](../examples/example_09_p2_workflow.py)

На малой синтетической E–I сети (WilsonCowan × 2 + StaticSynapse от
SinusoidalGenerator) с известным θ* и зашумлёнными целями (шум с тем же
sigma, что в ObservationSpec — тогда интервалы UQ имеют статистический
смысл):

1. **global_search** — DE малого бюджета (max_evals) с доводкой
   refine_top_k через L-BFGS-B;
2. **minimize_lbfgsb** — явная доводка от лучшей точки глобальной фазы,
   восстановление θ*;
3. **balanced_train** — балансировка компонент потерь (GradNormWeights) с
   PCGrad-проекцией конфликтующих градиентов и обязательным сравнением с
   baseline FixedWeights;
4. **laplace_approximation** — GN-ковариация постериора через якобиан
   невязок, маргинальные 95 %-интервалы параметров (на свежем экземпляре
   задачи: TF forward-AD несовместим с предшествующими gradient-прогонами
   в пределах одного NetworkRNN);
5. **predictive_intervals** — прогнозные интервалы firing_rates по
   ансамблю φ из posterior, покрытие зашумлённой цели;
6. Отдельно: **residual_diagnostics** (остатки в MAP-точке) и
   **evaluate_candidates** с SerialPool (эквивалентность потерь ручному
   циклу problem.evaluate).

```python
gs = global_search(problem, method="de", max_evals=200, refine_top_k=1,
                   local="lbfgsb", seed=11)
phi_map = minimize_lbfgsb(problem, gs.base.phi_best, maxiter=100).base.phi_best

bt = balanced_train(problem, gs.base.phi_best, GradNormWeights(),
                    optimizer="adam", max_steps=6, projector=pcgrad_project)

# UQ — на свежем экземпляре задачи (ограничение TF forward-AD, см. выше)
problem_uq = build_problem()
lap = laplace_approximation(problem_uq, phi_map, method="gn")
samples = rng.multivariate_normal(lap.phi_map, lap.covariance, size=48)
pred = predictive_intervals(problem_uq, samples, level=0.95)
diag = residual_diagnostics(problem_uq, phi_map)
```

### Мультиэкспериментная калибровка с общим и эксперимент-специфичным x0

Полный пример: [`example_10_multiexperiment_x0.py`](../examples/example_10_multiexperiment_x0.py)

`MultiExperimentProblem` с двумя экспериментами на одной сети
(WilsonCowan + StaticSynapse от SinusoidalGenerator, одинаковые входы и
динамика, разные начальные состояния):

1. **Регистрация начального состояния**: общие записи
   `x0/<pop>/<slot>` через `add_initial_state_entries` (участвуют во всех
   экспериментах) и эксперимент-специфичные `x0/<experiment>/<pop>/<slot>`
   через `add_experiment_initial_state_entries` (только у эксперимента с
   совпадающим именем, с приоритетом над общими);
2. **Различие траекторий**: при одном и том же φ прогоны `simulate(...,
   experiment="control")` и `simulate(..., experiment="stim")` стартуют из
   разных начальных состояний — exp-запись перезаписывает общий слот;
3. **Разложение градиентов**: градиент по общему `x0` — сумма вкладов
   экспериментов (вклад перезаписанного эксперимента равен нулю), градиент
   по exp-specific `x0` идёт только через свой эксперимент — проверяется
   сравнением с одноэкспериментными задачами.

```python
space = ParameterSpace.from_network(network)
space.add_initial_state_entries(network)                     # общий x0
space.add_experiment_initial_state_entries(network, "stim")  # exp-specific x0

problem = MultiExperimentProblem(network, space,
                                 experiments=[exp_control, exp_stim])
ev, grad = problem.value_and_gradient(phi0)
```

Подробности семантики (приоритет exp → common → default, предупреждения
для orphan-записей и внеэкспериментного контекста, ограничения
sensitivities/UQ и shooting) — в [`docs/optimization.md`](optimization.md).
