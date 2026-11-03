"""
Параллельная оценка кандидатов для глобального поиска и ансамблей (этап P2,
ранг 17 дорожной карты docs/optimization_roadmap.md).

Задача — одна и та же операция problem.evaluate(phi) (или иной метод, см.
fn) над множеством независимых точек φ: screening стартов мультистарта,
популяции CMA-ES/DE, bootstrap-ансамбли. Существующий контракт NetworkRNN
(batch=1, eager) не меняется: параллельность достигается независимыми
процессами, каждый со своей копией модели.

Публичный API:

- CandidateTask — один кандидат: index (позиция в выдаче вызывающей
  стороны), phi (вектор φ), seed (per-candidate seed или None);
- EvalRecord — результат оценки одного кандидата: index, loss,
  components, wall_time, n_forward, error;
- SerialPool — последовательный evaluator (n_workers=1), эквивалентен
  прямому циклу ``bp = build_problem(); [bp.evaluate(phi) for phi]``;
- ProcessPool — multiprocessing.get_context('spawn'), n_workers
  процессов, TF-граф строится лениво внутри worker'а;
- evaluate_candidates(build_problem, candidates, *, fn='evaluate',
  n_workers=1, seed=None, ordered=True) -> List[EvalRecord] — разовый
  прогон списка кандидатов (сырые массивы или CandidateTask);
- make_evaluator(build_problem, n_workers=1, seed=None) ->
  Callable[[Sequence[np.ndarray]], Sequence[float]] — переиспользуемый
  evaluator потерь для инъекции в global_search; у возвращаемого
  callable есть атрибут close(), освобождающий worker-процессы.

Требования к build_problem
--------------------------
Фабрика обязана быть picklable функцией уровня модуля (top-level def, не
лямбда и не замыкание): она сериализуется pickle и передаётся в
spawn-процессы, которые импортируют её модуль заново. Лямбды/замыкания
отклоняются ValueError с понятным сообщением ещё в родительском процессе.
Фабрика вызывается без аргументов и должна вернуть объект с методом fn
(по умолчанию evaluate(phi), duck-typed как CalibrationProblem:
возвращает объект с атрибутами loss и components).

Семантика выполнения
--------------------
SerialPool строит задачу лениво один раз и далее выполняет прямой цикл
вычислений в родительском процессе — контракт мутации модели совпадает с
ручным циклом bp.evaluate(phi). ProcessPool держит n_workers процессов
('spawn'; родительская модель внешне не мутируется, у каждого worker'а
своя копия), задача в каждом процессе строится лениво при первой задаче
и переиспользуется далее. Результаты ordered=True возвращаются в порядке
подачи кандидатов; ordered=False — в порядке завершения (imap_unordered).

Сиды и детерминизм
------------------
seed (int или None) разворачивается в per-candidate сиды через
np.random.SeedSequence(seed).generate_state(n) — детерминированно.
Перед оценкой кандидата worker выставляет np.random.seed(task.seed),
что делает воспроизводимыми цели со стохастикой на np.random. Одинаковый
seed -> одинаковый набор потерь; явный CandidateTask.seed имеет приоритет
над производными.

Политика ошибок
---------------
Исключение внутри worker'а (включая построение задачи фабрикой)
превращается в EvalRecord с loss=inf, components={} и текстовым error
('<ТипИсключения>: <сообщение>'); остальные кандидаты вычисляются, пул
выживает. BaseException (KeyboardInterrupt/SystemExit) не перехватывается.
Нефинитная потеря (nan) без исключения возвращается как есть — отбор
таких кандидатов остаётся за вызывающей стороной (см. screening в
strategies.multistart, который отбрасывает нефинитные screened_loss).

Ограничения: eager-режим, batch=1 (наследуются от CalibrationProblem);
нагрузка предполагается «много дорогих прогонов сети», поэтому overhead
spawn + повторный импорт TensorFlow в worker'ах амортизируется размером
задачи. scipy workers не используются.
"""

import multiprocessing
import pickle
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np

__all__ = [
    "CandidateTask",
    "EvalRecord",
    "SerialPool",
    "ProcessPool",
    "evaluate_candidates",
    "make_evaluator",
]

FAILED_LOSS = float("inf")

CandidateLike = Union[np.ndarray, "CandidateTask"]


# ---------------------------------------------------------------------------
# DTO
# ---------------------------------------------------------------------------


@dataclass
class CandidateTask:
    """
    Один кандидат на оценку.

    Атрибуты:
        index: позиция кандидата в выдаче вызывающей стороны — сохраняется
            в EvalRecord и определяет порядок при ordered=True;
        phi: вектор φ (приводится к 1-D float64 numpy);
        seed: per-candidate seed (или None — без явного сидирования).
    """

    index: int
    phi: np.ndarray
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self.index = int(self.index)
        self.phi = np.asarray(self.phi, dtype=np.float64).reshape(-1)
        if self.seed is not None:
            self.seed = int(self.seed)


@dataclass
class EvalRecord:
    """
    Результат оценки одного кандидата.

    Атрибуты:
        index: индекс кандидата (из CandidateTask);
        loss: скалярная потеря; inf при исключении в worker'е;
        components: взвешенные компоненты потери {имя: float} ({}, если
            результат их не предоставляет или кандидат упал);
        wall_time: длительность оценки этого кандидата, секунды;
        n_forward: число прямых прогонов сети, потребованное этой
            оценкой (дельта счётчика n_forward задачи в worker'е);
        error: None при успехе, иначе '<ТипИсключения>: <сообщение>'.
    """

    index: int
    loss: float
    components: Dict[str, float] = field(default_factory=dict)
    wall_time: float = 0.0
    n_forward: int = 0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Хелперы
# ---------------------------------------------------------------------------


def _check_picklable_factory(build_problem: Callable[[], Any]) -> None:
    """
    Проверяет, что build_problem можно передать в spawn-процесс pickle'ом.

    Лямбды и замыкания не сериализуются стандартным pickle — ловим это
    до запуска процессов и поднимаем понятный ValueError.
    """
    if not callable(build_problem):
        raise ValueError(
            "parallel: build_problem должна быть callable-фабрикой без "
            "аргументов, возвращающей объект задачи калибровки; получен "
            f"невызываемый объект {type(build_problem).__name__}."
        )
    try:
        pickle.dumps(build_problem)
    except Exception as exc:
        raise ValueError(
            "parallel: build_problem должна быть picklable фабрикой "
            "уровня модуля (например, top-level def build_problem(): ...), "
            "чтобы её можно было передать в worker-процессы multiprocessing "
            "'spawn'. Лямбды и замыкания не поддерживаются. Ошибка pickle: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _derived_seeds(seed: Optional[int], n: int) -> List[Optional[int]]:
    """
    Детерминированные per-candidate сиды из np.random.SeedSequence(seed).

    seed=None даёт [None] * n — явное сидирование отключено. Схема
    стабильна между процессами и версиями numpy (контракт SeedSequence).
    """
    if seed is None:
        return [None] * n
    state = np.random.SeedSequence(int(seed)).generate_state(n, dtype=np.uint32)
    return [int(s) for s in state]


def _normalize_candidates(
    candidates: Sequence[CandidateLike], seed: Optional[int]
) -> List[CandidateTask]:
    """
    Приводит сырые массивы к CandidateTask, сохраняя явные задачи как есть.

    Сырым кандидатам присваивается index = позиции в последовательности
    и per-candidate seed из seed; у CandidateTask собственные index/seed
    имеют приоритет.
    """
    items = list(candidates)
    seeds = _derived_seeds(seed, len(items))
    tasks: List[CandidateTask] = []
    for i, item in enumerate(items):
        if isinstance(item, CandidateTask):
            tasks.append(item)
        else:
            tasks.append(CandidateTask(index=i, phi=np.asarray(item), seed=seeds[i]))
    return tasks


def _count_forward_delta(problem: Any, counter_before: Optional[int]) -> int:
    """Дельта счётчика n_forward задачи; 0, если счётчика нет."""
    if counter_before is None:
        return 0
    try:
        after = int(getattr(problem, "n_forward", counter_before))
        return max(0, after - int(counter_before))
    except (TypeError, ValueError):
        return 0


def _evaluate_one(problem: Any, fn: str, task: CandidateTask) -> EvalRecord:
    """
    Оценка одного кандидата (общая для SerialPool и worker-процесса).

    Исключение метода превращается в EvalRecord с loss=inf и error;
    wall_time и n_forward фиксируются в обоих исходах. Результат метода
    — duck-typed: объект с .loss/.components (Evaluation) либо скаляр.
    """
    if task.seed is not None:
        np.random.seed(task.seed)
    counter_before = getattr(problem, "n_forward", None)
    t0 = time.perf_counter()
    try:
        method = getattr(problem, fn)
        result = method(task.phi)
        loss = float(getattr(result, "loss", result))
        raw_components = getattr(result, "components", None)
        components = (
            {}
            if raw_components is None
            else {str(k): float(v) for k, v in dict(raw_components).items()}
        )
    except Exception as exc:  # noqa: BLE001 — ошибки worker'ов безопасны
        return EvalRecord(
            index=task.index,
            loss=FAILED_LOSS,
            components={},
            wall_time=time.perf_counter() - t0,
            n_forward=_count_forward_delta(problem, counter_before),
            error=f"{type(exc).__name__}: {exc}",
        )
    return EvalRecord(
        index=task.index,
        loss=loss,
        components=components,
        wall_time=time.perf_counter() - t0,
        n_forward=_count_forward_delta(problem, counter_before),
        error=None,
    )


# ---------------------------------------------------------------------------
# Worker-процесс ProcessPool (module-level — требование pickle/spawn)
# ---------------------------------------------------------------------------

_WORKER: Dict[str, Any] = {}


def _init_worker(build_problem: Callable[[], Any], fn: str) -> None:
    """Initializer процесса: запоминает фабрику, задача строится лениво."""
    _WORKER["build_problem"] = build_problem
    _WORKER["fn"] = fn
    _WORKER["problem"] = None


def _worker_evaluate(task: CandidateTask) -> EvalRecord:
    """
    Тело одной задачи в worker-процессе.

    Проблема строится лениво при первой задаче и переиспользуется
    далее (TF-граф не перестраивается на кандидата). Исключение фабрики
    превращается в error-record — пул выживает.
    """
    problem = _WORKER.get("problem")
    if problem is None:
        try:
            problem = _WORKER["build_problem"]()
        except Exception as exc:  # noqa: BLE001 — см. политику ошибок
            return EvalRecord(
                index=task.index,
                loss=FAILED_LOSS,
                components={},
                wall_time=0.0,
                n_forward=0,
                error=f"{type(exc).__name__}: {exc}",
            )
        _WORKER["problem"] = problem
    return _evaluate_one(problem, _WORKER["fn"], task)


# ---------------------------------------------------------------------------
# Пулы
# ---------------------------------------------------------------------------


class SerialPool:
    """
    Последовательный evaluator в родительском процессе.

    Задача строится лениво при первом map() и переиспользуется; далее
    выполняется прямой цикл problem.evaluate(phi) — те же losses и тот
    же контракт мутации модели, что при ручном цикле. Порядок записей
    совпадает с порядком задач. ordered принимается для единого
    интерфейса с ProcessPool и игнорируется.
    """

    def __init__(self, build_problem: Callable[[], Any], fn: str = "evaluate"):
        _check_picklable_factory(build_problem)
        self._build_problem = build_problem
        self._fn = str(fn)
        self._problem: Any = None

    @property
    def problem(self) -> Any:
        """Построенная задача (None до первого map)."""
        return self._problem

    def _ensure_problem(self) -> Any:
        if self._problem is None:
            self._problem = self._build_problem()
        return self._problem

    def map(
        self, tasks: Sequence[CandidateTask], *, ordered: bool = True
    ) -> List[EvalRecord]:
        problem = self._ensure_problem()
        return [_evaluate_one(problem, self._fn, task) for task in tasks]

    def close(self) -> None:
        """Совместимость с ProcessPool; освобождать нечего."""

    def __enter__(self) -> "SerialPool":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


class ProcessPool:
    """
    Пул worker-процессов multiprocessing с контекстом 'spawn'.

    Процессы создаются лениво при первом непустом map(); каждый держит
    свою копию задачи (TF-граф строится внутри worker'а один раз).
    Родительская модель внешне не мутируется. Результаты ordered=True
    возвращаются в порядке подачи задач, ordered=False — в порядке
    завершения (все EvalRecord.index сохраняются, выборка полная).

    close() завершает worker'ов по завершении; terminate() — немедленно.
    Повторный map() после close() пересоздаёт процессы.
    """

    def __init__(
        self,
        build_problem: Callable[[], Any],
        fn: str = "evaluate",
        n_workers: int = 2,
    ):
        _check_picklable_factory(build_problem)
        n_workers = int(n_workers)
        if n_workers < 1:
            raise ValueError(
                f"ProcessPool: n_workers должен быть >= 1, получен {n_workers}."
            )
        self._build_problem = build_problem
        self._fn = str(fn)
        self.n_workers = n_workers
        self._pool: Optional[Any] = None

    def _ensure_pool(self) -> Any:
        if self._pool is None:
            context = multiprocessing.get_context("spawn")
            self._pool = context.Pool(
                processes=self.n_workers,
                initializer=_init_worker,
                initargs=(self._build_problem, self._fn),
            )
        return self._pool

    def map(
        self, tasks: Sequence[CandidateTask], *, ordered: bool = True
    ) -> List[EvalRecord]:
        task_list = list(tasks)
        if not task_list:
            return []
        pool = self._ensure_pool()
        if ordered:
            return list(pool.imap(_worker_evaluate, task_list))
        return list(pool.imap_unordered(_worker_evaluate, task_list))

    def close(self) -> None:
        """Штатное завершение worker-процессов."""
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def terminate(self) -> None:
        """Немедленная остановка worker-процессов."""
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None

    def __enter__(self) -> "ProcessPool":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Публичные точки входа
# ---------------------------------------------------------------------------


def evaluate_candidates(
    build_problem: Callable[[], Any],
    candidates: Sequence[CandidateLike],
    *,
    fn: str = "evaluate",
    n_workers: int = 1,
    seed: Optional[int] = None,
    ordered: bool = True,
) -> List[EvalRecord]:
    """
    Оценивает список кандидатов и возвращает EvalRecord на каждого.

    Args:
        build_problem: picklable фабрика уровня модуля (см. модульную
            документацию); вызывается без аргументов.
        candidates: последовательность векторов φ (np.ndarray) и/или
            CandidateTask; сырым кандидатам присваиваются index=позиция и
            per-candidate сиды из seed.
        fn: имя метода задачи ('evaluate' по умолчанию); вызывается как
            getattr(problem, fn)(phi), результат — duck-typed Evaluation.
        n_workers: 1 — SerialPool (родительский процесс); >= 2 —
            ProcessPool с n_workers spawn-процессами.
        seed: основа per-candidate сидов через np.random.SeedSequence;
            None — без явного сидирования.
        ordered: True — записи в порядке подачи кандидатов; False —
            в порядке завершения.

    Returns:
        Список EvalRecord, выровненный с candidates (ordered=True).

    Политика ошибок — см. модульную документацию: исключение кандидата
    даёт loss=inf и error, остальные кандидаты вычисляются.
    """
    tasks = _normalize_candidates(candidates, seed)
    if int(n_workers) <= 1:
        pool: Union[SerialPool, ProcessPool] = SerialPool(build_problem, fn=fn)
    else:
        pool = ProcessPool(build_problem, fn=fn, n_workers=n_workers)
    try:
        if not tasks:
            return []
        return pool.map(tasks, ordered=ordered)
    finally:
        pool.close()


def make_evaluator(
    build_problem: Callable[[], Any],
    n_workers: int = 1,
    seed: Optional[int] = None,
    *,
    fn: str = "evaluate",
) -> Callable[[Sequence[np.ndarray]], Sequence[float]]:
    """
    Переиспользуемый evaluator потерь для инъекции в global_search.

    Возвращает callable ``evaluate(candidates) -> [loss, ...]``; при
    n_workers >= 2 worker-процессы создаются один раз и живут между
    вызовами (TF-граф не перестраивается). Одинаковый seed даёт
    одинаковые потери на одних и тех же кандидатах. У callable есть
    атрибут close() — штатное завершение процессов (дублируется
    при сборке garbage collector'ом через финализатор multiprocessing,
    но явный close() предсказуемее).
    """

    def _evaluate(candidates: Sequence[np.ndarray]) -> List[float]:
        tasks = _normalize_candidates(candidates, seed)
        if not tasks:
            return []
        return [float(record.loss) for record in pool.map(tasks)]

    if int(n_workers) <= 1:
        pool: Union[SerialPool, ProcessPool] = SerialPool(build_problem, fn=fn)
    else:
        pool = ProcessPool(build_problem, fn=fn, n_workers=n_workers)
    _evaluate.close = pool.close  # type: ignore[attr-defined]
    return _evaluate
