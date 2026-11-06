# Решение проблем

Эта страница помогает быстро локализовать типичные проблемы: NaN в прогоне,
расходящуюся траекторию, неудачную калибровку, странные интервалы
неопределённости. Она опирается только на реальный функционал пакета — все
упомянутые функции и классы можно найти в [справочнике API](api_reference.md).

Как пользоваться страницей:

1. Пройдите [чек-лист первого запуска](#первый-запуск-чек-лист) — большинство
   проблем с градиентами и точностью видно сразу.
2. Найдите свой симптом в [таблице](#симптом--вероятная-причина--как-проверить--как-решить).
3. Проверьте, не упираетесь ли вы в [ограничения платформы](#ограничения-платформы).
4. Если ничего не помогло — соберите [минимальный воспроизводимый пример](#если-ничего-не-помогло).

## Первый запуск: чек-лист

Выполните эти шаги до начала серьёзной калибровки. Они занимают минуты и
экономят часы отладки.

1. **Точность вычислений (dtype).** По умолчанию пакет работает в `float32`.
   Для калибровки и анализа чувствительности настоятельно рекомендуется
   `float64`: вызовите `neuraltide.config.set_dtype(tf.float64)` **до создания
   любых объектов модели** (популяций, синапсов, `NetworkGraph`). Детали — в
   [Конфигурации](config.md). Перекрёстную проверку согласованности dtype даёт
   `neuraltide.optimization.diagnostics.dtype_consistency_check`.
2. **Воспроизводимость (seed).** Зафиксируйте все источники случайности через
   `neuraltide.utils.reproducibility.seed_everything(seed)`. Там же есть
   `log_versions()` — сохраняйте его вывод рядом с результатами. В
   стохастических процедурах передавайте seed явно: `sample_phi0s(seed=...)`,
   `multistart(seed=...)`, `global_search(seed=...)`, `evaluate_candidates(seed=...)`.
   Локальные оптимизаторы SciPy (`minimize_lbfgsb`, `least_squares`)
   детерминированы сами по себе: при том же `phi0` и тех же `options` повторный
   запуск даст тот же результат.
3. **Сходимость по шагу интегрирования.** Прогоните задачу на сетке
   `[dt, dt/2, dt/4]` через
   `neuraltide.optimization.diagnostics.dt_convergence`: потеря, норма градиента
   и сводка выходов должны перестать меняться между уточнениями. Если потеря
   «плавает» на мелком шаге — проблема не в dt, а в постановке (см. таблицу).
   Учтите: для `BrunelFokkerPlanck` эта функция неинформативна из-за внутреннего
   CFL-субшагивания (см. «Ограничения платформы»).
4. **Корректность градиентов.** Запустите
   `neuraltide.optimization.diagnostics.taylor_test` — он должен показать
   первый порядок вплоть до численного пола. Если градиенты подозрительны,
   смотрите плато в `finite_difference_scan` и согласованность прямого/сопряжённого
   операторов в `dot_product_test`.
5. **Согласованность методов градиента.** Если планируете adjoint-метод,
   сравните его с BPTT через
   `neuraltide.optimization.diagnostics.compare_bptt_adjoint`:
   `cosine_similarity` градиентов должна быть близка к 1. Учтите: аналитический
   adjoint поддерживают не все модели (например, `IzhikevichMeanField` и
   встроенные синапсы — да), а в режиме `integration_mode='network'` он не
   работает вовсе (см. «Ограничения платформы»).
6. **Контроль каждого прогона.** Оберните прогоны в
   `neuraltide.integrators.stiffness.classify_failure`: он возвращает
   `{'status': 'ok' | 'nan' | 'diverged', 'max_abs': float}` и превращает
   молчаливую порчу результата в явный статус. Это же используется при
   скрининге стартов в `multistart`.

## Симптом → вероятная причина → как проверить → как решить

| Симптом | Вероятная причина | Как проверить | Как решить |
|---|---|---|---|
| NaN/Inf в выходах, потере или градиентах | Переполнение в прямом прогоне; шаг dt слишком велик; неустойчивость правой части | Включите `neuraltide.config.set_debug_numerics(True)` и `maybe_check_numerics` покажет первый плохой тензор; статус прогона — `classify_failure` (ожидается `nan`) | Уменьшите dt; перейдите на `float64` через `set_dtype`; проверьте спектр через `stiffness_report` (мода с Re λ ≥ 0 означает неустойчивость); для неявного шага — `AdaptiveRKIntegrator`, `RosenbrockIntegrator` |
| Траектория взрывается (огромные значения состояния) | Явный Эйлер/Хойн неустойчив на текущем шаге; жёсткая динамика | `classify_failure` → `status='diverged'`; `stiffness_report` → поля `explicit_dt_limit` и `euler_stable` | Снизьте dt до `explicit_dt_limit`; используйте `AdaptiveRKIntegrator` (BS23+PI) или `RosenbrockIntegrator` — оба устойчивы на жёстких задачах; для жёстких синапсов — `IMEXEulerIntegrator` или `ExponentialEulerIntegrator` |
| Жёсткость: нужен крошечный шаг, прогон нестабилен | В спектре якобиана есть моды с сильно разными масштабами | `stiffness_report(network_or_problem, scope='populations')` для отдельных популяций и `scope='network'` для спектра всей связанной сети; смотрите `stiffness_ratio` и `explicit_dt_limit` | Неявные/адаптивные интеграторы: `RosenbrockIntegrator`, `IMEXEulerIntegrator`, `ExponentialEulerIntegrator`, `AdaptiveRKIntegrator`; в network-режиме связанной сети устойчив `RosenbrockIntegrator` |
| RK4 не даёт 4-й порядок на связанной сети | Режим по умолчанию `integration_mode='split'` расщепляет шаг популяций и синапсов — порядок не выше 2 даже у `RK4Integrator` | Сравните ошибку на сетке dt для `NetworkRNN(integration_mode='split')` и `'network'` | Используйте `NetworkRNN(integration_mode='network')` с `RK4Integrator` — там шаг `step_network` единый и порядок ~4; для жёстких связанных задач в этом режиме — `RosenbrockIntegrator` |
| loss не падает при обучении или калибровке | Плохой масштаб параметров; не тот dt; слишком большой шаг обучения; старт вне бассейна притяжения | `taylor_test` и `finite_difference_scan` (градиенты вообще правильные?); `dt_convergence` (потеря стабильна по dt?); прогоните `multistart` со скринингом | Перемасштабируйте параметры преобразованиями `transforms` (`log`, `affine`); снизьте шаг оптимизатора `tf.keras.optimizers`; начните с других точек через `sample_phi0s`; сгладьте цель `smooth_objective` |
| `ABNORMAL_TERMINATION_IN_LNSRCH` от L-BFGS-B | Эффективный `ftol` меньше точности `float32`-координат φ | Проверьте, не было ли `UserWarning` про `ftol < 1e-7` при `minimize_lbfgsb`; фактический ftol зафиксирован в `config['options']['ftol']` | Задайте `ftol >= 1e-7` для float32-φ или перейдите на `float64` (`set_dtype`) — тогда auto-ftol ≈ 2.2e-9 безопасен |
| Градиенты исчезают у границ параметров | Трансформы `log`/`softplus`/`scaled_sigmoid` насыщаются: производная dθ/dφ стремится к нулю у min/max | Поле `bounds_active` в `LocalOptimizerResult` — какие координаты прижаты к границе; сверьте `min`/`max` в записях `ParameterEntry` | Ослабьте или уберите границы; замените трансформ на `affine`/`identity`; стартуйте изнутри интервала, а не на кромке |
| `least_squares` сам перешёл на `'2-point'` | Размер φ превысил `jacobian_max_params` (по умолчанию 200); задача не полный `CalibrationProblem`; либо на этом экземпляре уже были gradient-прогоны (forward-AD несовместим с ними) | `UserWarning` при вызове; фактический режим записан в `config['jacobian']` (`'forward'`, `'2-point'`, `'3-point'`, `'callable'`) | Для forward-режима передайте `jacobian=None` на свежем экземпляре задачи (без предшествующих BPTT-прогонов); либо оставьте `'2-point'`/`'3-point'`, либо передайте свой `callable(x) -> [m, n]` |
| Мультистарт даёт разные решения | Цель мультимодальна; старты попадают в разные бассейны притяжения | Кластеризация решений в `MultistartResult` (решения внутри кластера совпадают с точностью `dedup_rtol`); сравните `loss_best` по стартам | Это нормально для невыпуклых задач — берите лучший кластер; повысьте `n_starts`, включите `screen_refine_steps`; для осознанного глобального поиска используйте `global_search` (методы `'de'`, `'cma'`, `'spsa'`, `'basinhopping'`) |
| FIM вырожден (малые собственные значения) | Параметры не идентифицируемы: избыточны или линейно зависимы по данным | `compute_fim` → `FIMResult`: смотрите `effective_rank`, `condition_number_fim`, `stiff_dirs`, пары `CollinearityPair` | Исключите/зафиксируйте сильно скоррелированные параметры (`select_subset` — доводка подмножества с оценкой смещения); расширьте протокол стимуляции; проверьте границы через `likelihood_profile` |
| Профильный интервал слишком узкий или «схлопнулся» | Сетка профиля не покрывает допустимую область; минимум на краю сетки; локальная несходимость оптимизатора в узлах | Предупреждения в `ProfileResult` от `likelihood_profile` («минимум профиля на границе сетки», «нет успешных узлов», немонотонность) | Расширьте сетку (больше `n_points`, шире диапазон вокруг текущего θ); проверьте, что внутренний оптимизатор узлов сходится (передайте `optimizer='lbfgsb'` или `'trf'`); сопоставьте с `bootstrap_experiments` |
| Shooting: разрывы между сегментами не сходятся | Вес continuity-штрафа слишком мал; слишком мало узлов; состояние синапсов исключено из склейки | Поле разрывов в `MultipleShootingProblem`; предупреждение при `include_syn_states=False` с синапсами, имеющими состояние | Повышайте `continuity_weight` постепенно; увеличьте число узлов `ShootingConfig`; оставьте `include_syn_states=True`; по окончании обязательно выполните `finalize()` / `to_single_shooting_problem()` и сверьтесь с `SingleShootingCheck` |
| Teacher forcing хорош, свободный прогон плох | Модель подогналась под подмешанные целевые частоты и не воспроизводит динамику сама | `free_run_check` в `ScheduleResult` от `ContinuationScheduler.run` — обязательная итоговая оценка на исходной задаче без подмешивания | Финальная стадия `teacher_forcing_schedule` обязана иметь `alpha=0` (иначе расписание не построится); наращивайте горизонт постепенно через `horizon_growth`; не оценивайте результат по стадиям с teacher forcing |
| Held-out хуже train | Переобучение на обучающих протоколах; цель нестабильна по времени | Включите `validate=True` в `minimize_lbfgsb`/`least_squares`/`global_search` — метрики на held-out появятся в поле `validation`; сравните `validation_delta` из `likelihood_profile` | Сузьте подмножество обучаемых параметров (`select_subset`); упростите цель через `smooth_objective`; сбалансируйте вклад окон (`AdaptiveWindowFocus`, `balanced_train`); добавьте held-out протоколы в `problem.held_out` |
| Интервалы неопределённости огромные | Слабая идентифицируемость (см. FIM); гетероскедастичные невязки; гауссово приближение неточно в хвостах | Сначала `compute_fim` и `residual_diagnostics` (оконная статистика невязок, автокорреляция); сравните `laplace_approximation` с `bootstrap_experiments` и `parameter_ensemble` по нескольким прогонам | Уточните постановку (меньше параметров, богаче протокол); используйте `predictive_intervals` по ансамблю вместо только Лапласа; проверьте устойчивость выводов через `noise_sensitivity` |
| `BrunelFokkerPlanck` не сходится к аналитике Brunel | Слишком грубая сетка по V; неверный интегратор (буфер рефрактерного ν сдвигается неточно); ожидание точности там, где модель её не даёт | Статистика частоты на стационаре против формулы (21) из Brunel (2000); проверьте, что внешний интегратор — `EulerIntegrator` или `HeunIntegrator` | Увеличьте `grid_size` (около порога ~0.2% при 250 узлах; сверхпороговый режим ~4% при 500 — это свойство дискретизации, уточняйте сетку); не пользуйтесь `dt_convergence` по внешнему dt — внутреннее CFL-субшагивание делает её неинформативной |
| Нехватка памяти (OOM) на длинном горизонте | Прямой прогон хранит все шаги для BPTT; длина горизонта T слишком велика | Оцените пиковую память через `CheckpointedGradient.peak_memory` в `value_and_gradient_checkpointed` | Режьте градиент по сегментам: `truncated_bptt` (с честной итоговой оценкой `free_run_loss`) или `value_and_gradient_checkpointed` с `segment_length`; наращивайте горизонт через `horizon_growth`; экономия памяти в eager-режиме ограничена — держите сегменты короткими |
| Параллельный пул медленнее последовательного | Задачи дешёвые — накладные расходы на spawn-процессы и сериализацию больше выигрыша; CPU перегружен потоками TF | Сравните время `evaluate_candidates(..., n_workers=1)` (это `SerialPool`) с `n_workers>=2` (`ProcessPool`); убедитесь, что `build_problem` — picklable-фабрика уровня модуля | Параллельность окупается только на дорогих оценках (длинные прогоны); оставьте `SerialPool` для дешёвых; в worker-процессах потоки TF ограничены параметром `tf_threads` (по умолчанию 1 — защита от oversubscription) |
| `stability_loss == 0` неожиданно | Либо штраф не включён, либо режим прогона его не оценивает | Проверьте `stability_penalty_weight` в конструкторе `NetworkRNN` (по умолчанию 0.0); проверьте `integration_mode` | Задайте `stability_penalty_weight > 0`; учтите, что в режиме `integration_mode='network'` штраф стабильности не оценивается (единый шаг `step_network` не возвращает локальную ошибку) — оценивайте точность через `dt_convergence` и `stiffness_report` |
| Предупреждение про «осиротевшие» записи начального состояния (orphan x0) | Записи `x0/<experiment>/<pop>/<slot>`, добавленные через `ParameterSpace.add_experiment_initial_state_entries`, используются вне многоэкспериментной задачи | Текст `UserWarning`: записи существуют, но не материализованы | Создавайте такие записи только для `MultiExperimentProblem` (приоритет: эксперимент → общие → умолчание); в обычной `CalibrationProblem` используйте общие записи `x0/<pop>/<slot>` через `add_initial_state_entries` и `StateLocator` |
| Итерации оптимизатора медленные, CPU недогружен | Python-оверхед: граф операций строится заново на каждую итерацию eager-прогона | Сравните `Trainer(run_eagerly=True)` с режимом по умолчанию (`tf.function`) и `CalibrationProblem(..., compiled=True)` с `compiled=False` на одном φ; замеры — `benchmarks/g0_graph_mode.py` | Длинные T и много итераций — включайте `compiled=True` / `Trainer(jit_compile=True)`; если после включения появилось предупреждение TF о retracing — между вызовами меняются формы или объекты (см. view-хелперы: каждый view строит новое ядро) |
| `InvalidArgumentError: XLA compilation requires a fixed tensor list size` | `Trainer(jit_compile=True)` с интегратором, чей градиент не компилируется под XLA (`AdaptiveRKIntegrator`, `ExponentialEulerIntegrator`) | Текст ошибки; исчезает при `jit_compile=False` | Снимите `jit_compile` (граф без XLA остаётся) или смените интегратор; прямой прогон под XLA у всех встроенных интеграторов работает |
| Ошибка/некорректный результат при batch > 1 | Последовательный цикл `NetworkRNN` поддерживает только ведущий batch-размер 1 | Форма состояний `[1, n_units]` из `get_initial_state(batch_size=1)`; все состояния имеют ведущую единицу | Прогоняйте образцы по одному (цикл по batch с общими параметрами); для пакетной оценки кандидатов используйте `evaluate_candidates`/`make_evaluator` — пакетирование там реализовано на уровне задач, а не тензоров |

## Ограничения платформы

Эти ограничения заложены в архитектуру пакета. Они не являются ошибками, но
важно знать о них до проектирования сценария.

- **Eager по умолчанию.** Слой калибровки (`neuraltide.optimization`)
  по умолчанию работает в eager-режиме TensorFlow; компилируемый путь —
  опция `compiled=True` у `CalibrationProblem`/`MultiExperimentProblem`
  (граф прогона+градиента без XLA, см. «Компилируемый режим» в
  [optimization.md](optimization.md)). Экономия памяти при обратном проходе
  при этом ограничена (см. строку про OOM в таблице).
- **Batch равен 1.** Все состояния популяций и синапсов имеют форму
  `[1, n_units]`; `NetworkRNN` прогоняет временную последовательность
  последовательно. Множественные образцы прогоняются по одному; пакетная
  оценка кандидатов реализована через `evaluate_candidates`, а не тензорный
  batch.
- **`InputPopulation` устарел.** Класс помечен deprecated и при создании
  выдаёт `DeprecationWarning`. Внешние входы подключайте через
  `NetworkGraph.declare_input()` и предвычисленные входы `graph.pack_inputs()`,
  передаваемые в `NetworkRNN.call(t_seq, inputs=...)`.
- **`BrunelFokkerPlanck` — только split-режим с `EulerIntegrator`/`HeunIntegrator`.**
  Сдвиг буфера рефрактерного ν(t − τ_rp) точен именно для этой пары; внутри
  `derivatives()` работает CFL-субшагивание, из-за чего сходимость по внешнему
  dt через `dt_convergence` неинформативна. У начального распределения базового
  `FokkerPlanckPopulation` нет умолчания: подкласс обязан задать его сам
  (например, через `_gaussian_initial_state(center, sigma)`).
- **Forward-AD несовместим с предшествующими gradient-прогонами.** На одном
  экземпляре `NetworkRNN` нельзя сочетать BPTT и forward-чувствительности
  (`residual_jvp`, `residual_jacobian`, forward-режим `least_squares`): TF
  не позволяет forward-AD после gradient-прохода на том же состоянии. Запускайте
  их на свежем экземпляре задачи; `least_squares` честно падает на `'2-point'`
  с `UserWarning`. Требование «свежего экземпляра» относится и к
  неопределённости (UQ) после обучающих прогонов.
- **Лапласово приближение — гауссово.** `laplace_approximation` строит
  гауссову аппроксимацию постериора в точке MAP: она не улавливает асимметрию,
  скошенность и мультимодальность. Для проверки хвостов используйте
  `bootstrap_experiments`, `parameter_ensemble` по нескольким прогонам и
  `predictive_intervals`.
- **Дополнительные ограничения режимов.** В режиме `integration_mode='network'`
  не оценивается `stability_loss` (всегда 0) и не поддерживается аналитический
  adjoint (`AdjointSolver` с `use_analytical_adjoint=True` поднимет
  `NotImplementedError`) — там доступен только дискретный adjoint/BPTT; сами
  интеграторы `AdaptiveRKIntegrator`, `IMEXEulerIntegrator` и
  `ExponentialEulerIntegrator` в network-режиме не работают (поддерживают
  `EulerIntegrator`, `HeunIntegrator`, `RK4Integrator`, `RosenbrockIntegrator`).
  Балансировка компонент цели (`balanced_train`, `pcgrad_project`,
  `cagrad_project`) работает только с BPTT-градиентами.

## Если ничего не помогло

### Как собрать минимальный воспроизводимый пример

1. Сократите сеть до одной популяции без синапсов, горизонт — несколько сотен
   шагов, цель — простая (`ResidualLoss` по одной наблюдаемой).
2. Зафиксируйте окружение: `seed_everything(seed)` в первой строке, явный
   `phi0` в `problem.pack()`/аргументах оптимизатора, сохранённые `options`
   вызова (`maxiter`, `ftol`, `gtol`, `jacobian`).
3. Воспроизведите на нём: если проблема исчезла — добавляйте элементы
   оригинальной постановки по одному, пока она не вернётся. Последний
   добавленный элемент и есть виновник.

### Какие диагностики приложить к обращению

- Версии: вывод `neuraltide.utils.reproducibility.log_versions()` и dtype
  (`neuraltide.config.get_dtype()`).
- Статус прогонов: вывод `classify_failure` для проблемной траектории.
- Спектр и шаг: сводка `stiffness_report` (оба scope, если сеть связанная) —
  поля `stiffness_ratio`, `explicit_dt_limit`, `euler_stable`, `dt_network`.
- Сходимость по dt: таблица `dt_convergence`.
- Градиенты: результат `taylor_test` (и `finite_difference_scan`/`dot_product_test`,
  если есть сомнения), при adjoint — `compare_bptt_adjoint` с `cosine_similarity`.
- Запуск оптимизатора: `config` из результата (`config['jacobian']`,
  `config['options']['ftol']`), предупреждения `UserWarning` дословно.
- Невязки: сводка `residual_diagnostics` (окна, автокорреляция).
- Счётчики обращений: `n_forward`/`n_gradient` задачи до и после прогона —
  они показывают, сколько прогонов реально выполнилось.

Больше контекста по отдельным подсистемам: [интеграторы и режимы прогона](integrators.md),
[оптимизация и калибровка](optimization.md), [конфигурация и dtype](config.md),
[популяционные модели](populations.md), [обучение через Trainer](training.md),
[готовые сценарии](examples.md).
