# Оптимизация и калибровка

Модуль `neuraltide.optimization` — слой калибровки поверх моделей сети. Он
реализует этап **P0** дорожной карты (`docs/optimization_roadmap.md`):
корректную постановку задачи и проверяемые градиенты, на которые опираются
все будущие методы оптимизации. `Trainer` остаётся простым совместимым API
градиентного обучения; новые оптимизаторы (SciPy LM/TRF, L-BFGS-B,
мультистарт, глобальный поиск) будут работать поверх общего
`CalibrationProblem`.

## Преобразования координат параметров (`transforms`)

Физические параметры модели θ связаны с координатами оптимизатора φ
дифференцируемым поэлементным преобразованием θ = forward(φ):

| Преобразование | Назначение |
|---|---|
| `identity` | текущее поведение params-формата без изменений |
| `affine` | центрирование и масштабирование (θ = a·φ + b) |
| `log` | положительные параметры, меняющиеся по порядкам величины (θ = exp φ) |
| `logit`, `scaled_sigmoid` | параметры ограниченного интервала |
| `softplus` | неотрицательные параметры с достижимой окрестностью нуля |
| `bounded_log` | логарифмическая координата внутри диапазона |

Фабрика `make_transform(kind, **kwargs)` и регистр `TRANSFORM_REGISTRY`.
Обратное преобразование `inverse(θ)` клиппит θ в допустимую область с
безопасным численным margin; `jacobian(φ)` возвращает dθ/dφ.

Предупреждение: sigmoid-образные преобразования насыщаются у границ
интервала (производная → 0), поэтому это не универсальный способ задания
ограничений — настоящие bounds (L-BFGS-B/TRF) появятся в P1.

## Реестр параметров (`ParameterSpace`)

`ParameterSpace.from_network(network, transforms={full_name: Transform})`
сканирует обучаемые переменные популяций и синапсов графа и строит плоский
вектор φ единый для всех оптимизаторов:

- `pack(values=None)` — θ → φ; `None` — текущие значения переменных модели;
- `unpack(phi)` — φ → θ (дифференцируемо, GradientTape-совместимо);
- `apply_to_model(theta_dict)` — запись θ в переменные модели;
- `project(phi)` — проекция φ в допустимую область (через физические
  min/max из params-формата).

Стабильные полные имена имеют вид `<owner>/<param>`, например
`exc/Delta_I` или `inh->exc/tau_f`. Параметры входных генераторов и
readout-слои автоматически не охватываются; nuisance-параметры (сдвиги,
масштабы, параметры шума) регистрируются вручную через `add_entry`.

## Наблюдения и невязки (`objectives`)

`ObservationSpec` описывает одно наблюдение: популяция, целевая траектория
`[batch, T, n_units]`, модель шума `sigma`, вес, маска, readout и вид
штрафа `kind`. Поддерживаются виды:

- `l2` — 0.5·e², e = pred − target;
- `relative` — 0.5·(e / (|target| + eps))²;
- `msle` — 0.5·(log1p(pred + eps) − log1p(target + eps))²;
- `huber` и `pseudo_huber` — порог `delta`, устойчивость к выбросам.

Нормировка — masked mean: `weight · sum(mask · psi) / (N · sigma²)`,
где N = max(sum(mask), eps) либо число элементов траектории. Маска масштаб
не меняет. `ResidualLoss` агрегирует набор спецификаций и разделяет два
интерфейса:

- скалярный: `__call__`, `components()` (вклады по именам), `target`
  (совместимо с `Trainer`), аддитивный `per_step_loss`;
- векторный: `residual(predictions)` — честные невязки r, такие что
  `0.5 · ||r||² == loss` с учётом sigma/weight/mask/kind. Это постановка
  для nonlinear least squares (Gauss–Newton / LM / TRF в P1).

## Задача калибровки (`CalibrationProblem`)

Инкапсулирует сеть, ParameterSpace, протокол (`t_sequence`, `inputs`,
`initial_state`) и objective (callable, `BaseLoss`/`CompositeLoss`;
`None` допускается с последующим присваиванием `problem.objective`):

- `simulate(phi=None)` — прямой прогон;
- `evaluate(phi=None)` — `Evaluation`: loss, компоненты, невязки,
  счётчики `n_forward`/`n_gradient`, wall time, dtype, dt;
- `value_and_gradient(phi=None)` — `(Evaluation, grad)` одним прогоном;
- `residual(phi=None)` — вектор невязок для GN/LM/TRF;
- `bounds_phi()` — φ-границы из физических min/max (заготовка для
  SciPy-адаптеров).

Контракт мутации модели: `phi=None` модель не изменяет; `phi != None`
явно переводит модель в состояние θ = unpack(phi) перед прогоном и
оставляет её в нём. Для сохранения/восстановления состояния:
`saved = problem.pack()` и
`space.apply_to_model(space.unpack(saved))`.

Режимы градиента (`grad_mode`): `'assign'` (по умолчанию) и `'chain'`. В
текущем eager TensorFlow градиент через `variable.assign` недоступен, поэтому
оба режима вычисляют dloss/dφ как dloss/dθ по переменным модели, умноженный
на dθ/dφ (jacobians transforms); результаты совпадают, что контролируется
taylor_test'ом.

## Результаты (`results`)

- `Evaluation` — результат одной оценки (см. выше), `to_dict()` с
  numpy → list для JSON;
- `OptimizationResult` — единый формат результата оптимизации: `phi_best`,
  физические `theta_best`, `loss_best`, история, seed, конфигурация,
  версии пакетов, dtype; `to_dict()`/`from_dict()`/`save()`/`load()`
  (JSON, numpy → list; сериализация самой модели не входит в контракт).

## Диагностика градиентов (`diagnostics`)

Пользовательские численные проверки (см. также `tests/unit/test_diagnostics.py`):

- `finite_difference_scan` — центральные конечные разности со сканированием h;
- `taylor_test` — направленный тест порядков 1 и 2 до численного предела;
- `dot_product_test` — сопряжённость прямого и транспонированного операторов;
- `dt_convergence` — сходимость потери/градиента при уточнении dt;
- `dtype_consistency_check` — согласованность float32/float64 через config;
- `compare_bptt_adjoint` — равенство потери и градиентов BPTT и adjoint.

## Что дальше (этапы P1+)

SciPy L-BFGS-B и `least_squares` (TRF/LM) поверх `value_and_gradient`/
`residual`, LHS/Sobol-мультистарт и гибридные стратегии, checkpointed/truncated
BPTT и forward sensitivities, обучаемые начальные состояния и несколько
протоколов, multiple shooting, FIM/профили правдоподобия и отложенная
валидация — следующие этапы дорожной карты.

Сквозной пример: `examples/example_07_calibration.py`.
