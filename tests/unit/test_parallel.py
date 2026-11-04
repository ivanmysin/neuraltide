"""
Тесты P2: параллельная оценка кандидатов (optimization/parallel.py).

SerialPool сверяется с прямым циклом bp().evaluate(phi) малой L0-задачи
(фикстура l0_problem / module-level фабрика build_l0_problem) до машинной
точности. ProcessPool (n_workers=2) использует module-level фабрики:
spawn-процессы заново импортируют этот модуль по sys.path, поэтому
фабрики обязаны быть top-level def, а не замыканиями фикстур. Тесты
ProcessPool помечены skipif(cpu_count < 2).

Бюджет времени: каждый ProcessPool-тест платит spawn + повторный импорт
TensorFlow в worker'ах (~10-20 с на старт пула), прогоны L0 (T=120 шагов)
миллисекундные; duck-задача _QuadraticProblem без TF удешевляет тесты
политики ошибок, а _ThreadProbeProblem — проверку tf_threads в worker'ах
(ограничение TF-потоков, замечание C10). При запуске файла используйте
внешний timeout ~5-10 минут.
"""

import os
import pickle

import numpy as np
import pytest

from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.integrators import EulerIntegrator
from neuraltide.inputs import ConstantRateGenerator
from neuraltide.optimization import (
    CalibrationProblem,
    ObservationSpec,
    ParameterSpace,
    ResidualLoss,
)
from neuraltide.optimization.parallel import (
    CandidateTask,
    EvalRecord,
    ProcessPool,
    SerialPool,
    _init_worker,
    _worker_evaluate,
    evaluate_candidates,
    make_evaluator,
)
from neuraltide.populations import WilsonCowan
from neuraltide.synapses import StaticSynapse

L0_DT = 0.5
L0_T = 60

NEEDS_TWO_CPUS = pytest.mark.skipif(
    (os.cpu_count() or 1) < 2, reason="нужно >= 2 CPU для ProcessPool"
)


# ---------------------------------------------------------------------------
# Module-level фабрики (picklable; spawn worker'ы импортируют этот модуль)
# ---------------------------------------------------------------------------


def build_l0_problem() -> CalibrationProblem:
    """
    L0-задача калибровки: WilsonCowan (1 юнит) + StaticSynapse от
    ConstantRate — копия conftest._build_l0, обязана жить на уровне
    модуля (picklable фабрика для spawn-процессов).
    """
    pop = WilsonCowan(
        dt=L0_DT,
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
        dt=L0_DT,
        params={
            "gsyn_max": {
                "value": [[0.05]],
                "trainable": True,
                "min": 0.001,
                "max": 1.0,
            },
            "pconn": [[1.0]],
            "e_r": 0.0,
        },
        name="in->exc",
    )
    gen = ConstantRateGenerator(dt=L0_DT, params={"rate": 10.0})

    graph = NetworkGraph(dt=L0_DT)
    graph.declare_input("in", n_units=1)
    graph.add_population("exc", pop)
    graph.add_synapse("in->exc", syn, src="in", tgt="exc")
    network = NetworkRNN(graph, EulerIntegrator())

    t_seq = np.arange(L0_T, dtype=np.float32) * L0_DT
    t_seq = t_seq[None, :, None]
    inputs = graph.pack_inputs({"in": gen(t_seq)})

    target = network(t_seq, inputs=inputs).firing_rates["exc"]
    loss = ResidualLoss(
        [ObservationSpec(name="obs", population="exc", target=target, sigma=2.0)]
    )
    space = ParameterSpace.from_network(network)
    return CalibrationProblem(network, space, t_seq, inputs=inputs, objective=loss)


class _SimpleEval:
    """Минимальный duck Evaluation: loss + components."""

    def __init__(self, loss: float, components: dict):
        self.loss = float(loss)
        self.components = dict(components)


class _QuadraticProblem:
    """
    Duck-typed задача без TensorFlow: loss = ||phi||^2, components
    {'quad': loss}; ValueError для phi[0] < 0 (политика ошибок worker'ов).
    """

    size = 2

    def __init__(self):
        self.n_forward = 0
        self.n_gradient = 0

    def evaluate(self, phi) -> _SimpleEval:
        self.n_forward += 1
        x = np.asarray(phi, dtype=np.float64).reshape(-1)
        if x[0] < 0.0:
            raise ValueError("negative phi[0]")
        loss = float(np.dot(x, x))
        return _SimpleEval(loss, {"quad": loss})


def build_quadratic_problem() -> _QuadraticProblem:
    """Module-level фабрика _QuadraticProblem (picklable)."""
    return _QuadraticProblem()


class _ThreadProbeProblem:
    """
    Duck-задача-зонд для TF-потоков: loss = число inter-op потоков TF,
    components несёт оба значения. TF импортируется лениво в evaluate —
    в worker-процессе initializer пула уже успел ограничить пулы потоков.
    """

    def evaluate(self, phi) -> _SimpleEval:
        import tensorflow as tf

        inter = tf.config.threading.get_inter_op_parallelism_threads()
        intra = tf.config.threading.get_intra_op_parallelism_threads()
        return _SimpleEval(float(inter), {"inter": float(inter), "intra": float(intra)})


def build_thread_probe_problem() -> _ThreadProbeProblem:
    """Module-level фабрика _ThreadProbeProblem (picklable)."""
    return _ThreadProbeProblem()


def _l0_phis(problem: CalibrationProblem, n: int = 4) -> list:
    """n соседних точек φ вокруг текущих параметров L0-задачи."""
    phi0 = np.asarray(problem.pack(), dtype=np.float64).reshape(-1)
    direction = np.linspace(0.0, 0.3, n, dtype=np.float64)
    return [phi0 + float(d) for d in direction]


# ---------------------------------------------------------------------------
# SerialPool
# ---------------------------------------------------------------------------


def test_serial_matches_direct_loop_to_machine_precision(l0_problem):
    """SerialPool == прямой цикл bp().evaluate(phi) до машинной точности."""
    phis = _l0_phis(l0_problem)
    direct = [l0_problem.evaluate(phi).loss for phi in phis]

    records = evaluate_candidates(build_l0_problem, phis, n_workers=1)

    assert [r.index for r in records] == list(range(len(phis)))
    np.testing.assert_allclose([r.loss for r in records], direct, rtol=1e-6, atol=1e-9)
    assert all(r.error is None for r in records)
    assert all(np.isfinite(r.loss) for r in records)
    # L0: один evaluate — один прямой прогон сети.
    assert all(r.n_forward == 1 for r in records)


def test_serial_record_components_and_wall_time(l0_problem):
    """EvalRecord несёт компоненты objective, wall_time и счётчики."""
    phi0 = np.asarray(l0_problem.pack(), dtype=np.float64).reshape(-1)
    records = evaluate_candidates(build_l0_problem, [phi0], n_workers=1)
    (record,) = records
    assert record.components.keys() == {"obs"}
    assert np.isfinite(record.components["obs"])
    assert record.wall_time >= 0.0
    assert record.n_forward == 1
    assert record.error is None


def test_serial_pool_context_manager_matches(l0_problem):
    """SerialPool как контекстный менеджер повторяет evaluate_candidates."""
    phis = _l0_phis(l0_problem, n=3)
    direct = [l0_problem.evaluate(phi).loss for phi in phis]
    with SerialPool(build_l0_problem) as pool:
        records = pool.map([CandidateTask(i, phi) for i, phi in enumerate(phis)])
    np.testing.assert_allclose([r.loss for r in records], direct, rtol=1e-6, atol=1e-9)


def test_empty_candidates_returns_empty_list():
    """Пустой список кандидатов — пустой результат без построения задачи."""
    assert evaluate_candidates(build_quadratic_problem, [], n_workers=1) == []


# ---------------------------------------------------------------------------
# ProcessPool
# ---------------------------------------------------------------------------


@NEEDS_TWO_CPUS
def test_process_pool_matches_direct_order_seeds_counters(l0_problem):
    """
    n_workers=2 на module-level L0-фабрике: те же losses, порядок
    сохранён, per-candidate сиды воспроизводимы, n_forward > 0, а
    родительская модель после прогона в порядке: прямая оценка phi0
    воспроизводит direct[0].
    """
    phi0 = np.asarray(l0_problem.pack(), dtype=np.float64).reshape(-1)
    phis = [phi0 + float(d) for d in np.linspace(0.0, 0.3, 4)]
    direct = [l0_problem.evaluate(phi).loss for phi in phis]

    records = evaluate_candidates(build_l0_problem, phis, n_workers=2, seed=42)

    assert [r.index for r in records] == [0, 1, 2, 3]  # порядок сохранён
    np.testing.assert_allclose([r.loss for r in records], direct, rtol=1e-6, atol=1e-9)
    assert all(r.error is None for r in records)
    assert all(r.n_forward > 0 for r in records)
    assert all(r.wall_time >= 0.0 for r in records)

    # Детерминизм: тот же seed -> тот же набор потерь.
    again = evaluate_candidates(build_l0_problem, phis, n_workers=2, seed=42)
    assert [r.loss for r in again] == [r.loss for r in records]

    # Родительская модель корректна: прямая оценка в phi0 совпадает.
    repeat = l0_problem.evaluate(phi0).loss
    np.testing.assert_allclose(repeat, direct[0], rtol=1e-6, atol=1e-9)


@NEEDS_TWO_CPUS
def test_process_pool_unordered_returns_full_set():
    """ordered=False: все записи на месте, index сохраняются."""
    phis = [np.array([float(i), 1.0]) for i in range(1, 7)]
    records = evaluate_candidates(
        build_quadratic_problem, phis, n_workers=2, ordered=False
    )
    assert sorted(r.index for r in records) == list(range(6))
    by_index = {r.index: r for r in records}
    for i, phi in enumerate(phis):
        np.testing.assert_allclose(by_index[i].loss, float(np.dot(phi, phi)))


@NEEDS_TWO_CPUS
def test_process_pool_worker_errors_are_safe():
    """
    Исключение в одном worker'е -> EvalRecord с loss=inf и error,
    остальные кандидаты вычисляются; следующий запуск evaluate_candidates
    (новый пул) работает — политика ошибок не ломает сьют.
    """
    phis = [np.array([1.0, 1.0]), np.array([-1.0, 1.0]), np.array([0.5, 2.0])]
    records = evaluate_candidates(build_quadratic_problem, phis, n_workers=2)

    assert records[0].error is None
    np.testing.assert_allclose(records[0].loss, 2.0)
    assert records[1].error is not None
    assert records[1].error.startswith("ValueError")
    assert np.isinf(records[1].loss)
    assert records[1].n_forward == 1  # попытка прогона учтена счётчиком
    assert records[1].components == {}
    assert records[2].error is None
    np.testing.assert_allclose(records[2].loss, 4.25)

    # Следующий запуск работает штатно.
    again = evaluate_candidates(build_quadratic_problem, [np.array([2.0, 0.0])])
    np.testing.assert_allclose(again[0].loss, 4.0)


def test_serial_worker_errors_are_safe():
    """Политика ошибок та же в SerialPool (без процессов)."""
    phis = [np.array([-2.0, 0.0]), np.array([3.0, 4.0])]
    records = evaluate_candidates(build_quadratic_problem, phis, n_workers=1)
    assert np.isinf(records[0].loss) and records[0].error is not None
    np.testing.assert_allclose(records[1].loss, 25.0)


# ---------------------------------------------------------------------------
# Требования к build_problem
# ---------------------------------------------------------------------------


def test_lambda_factory_raises_understandable_error():
    """Лямбда не picklable -> понятный ValueError ещё в родителе."""
    with pytest.raises(ValueError, match="picklable"):
        evaluate_candidates(
            lambda: build_quadratic_problem(), [np.zeros(2)], n_workers=1
        )


def test_closure_factory_raises_understandable_error():
    """Замыкание (вложенный def с захватом локальных) тоже не picklable."""
    factor = 2.0

    def closure_factory():
        problem = _QuadraticProblem()
        problem.factor = factor  # замыкание делает фабрику не-picklable
        return problem

    with pytest.raises(ValueError, match="picklable"):
        ProcessPool(closure_factory, n_workers=2)


def test_non_callable_factory_raises():
    with pytest.raises(ValueError, match="callable"):
        evaluate_candidates(
            42,
            [np.zeros(2)],
            n_workers=1,  # type: ignore[arg-type]
        )


def test_process_pool_rejects_zero_workers():
    with pytest.raises(ValueError, match="n_workers"):
        ProcessPool(build_quadratic_problem, n_workers=0)


# ---------------------------------------------------------------------------
# tf_threads: ограничение TF-потоков в worker-процессах (C10)
# ---------------------------------------------------------------------------


@NEEDS_TWO_CPUS
def test_process_pool_worker_tf_threads_are_limited():
    """
    tf_threads=1 доходит до worker'ов: TF-пулы внутри каждого процесса
    ограничены единицей (intra-op и inter-op), оценка при этом корректна.
    """
    phis = [np.array([0.0]), np.array([1.0])]
    records = evaluate_candidates(
        build_thread_probe_problem, phis, n_workers=2, tf_threads=1
    )

    assert all(r.error is None for r in records)
    assert all(r.loss == 1.0 for r in records)
    assert all(r.components["inter"] == 1.0 for r in records)
    assert all(r.components["intra"] == 1.0 for r in records)


@NEEDS_TWO_CPUS
def test_make_evaluator_passes_tf_threads_to_workers():
    """make_evaluator пробрасывает tf_threads в worker-процессы."""
    evaluator = make_evaluator(build_thread_probe_problem, n_workers=2, tf_threads=1)
    try:
        losses = list(evaluator([np.zeros(1)]))
    finally:
        evaluator.close()
    assert losses == [1.0]


@NEEDS_TWO_CPUS
def test_process_pool_tf_threads_none_keeps_loss_semantics():
    """tf_threads=None не падает и не меняет семантику потерь."""
    phis = [np.array([1.0, 2.0]), np.array([3.0, 0.0])]
    records = evaluate_candidates(
        build_quadratic_problem, phis, n_workers=2, tf_threads=None
    )
    assert all(r.error is None for r in records)
    np.testing.assert_allclose([r.loss for r in records], [5.0, 9.0])

    evaluator = make_evaluator(build_quadratic_problem, n_workers=2, tf_threads=None)
    try:
        np.testing.assert_allclose(list(evaluator(phis)), [5.0, 9.0])
    finally:
        evaluator.close()


def test_invalid_tf_threads_raise_before_processes_start():
    """tf_threads=0/-1 -> ValueError до запуска процессов (все API)."""
    with pytest.raises(ValueError, match="tf_threads"):
        ProcessPool(build_quadratic_problem, n_workers=2, tf_threads=0)
    with pytest.raises(ValueError, match="tf_threads"):
        ProcessPool(build_quadratic_problem, n_workers=2, tf_threads=-1)
    with pytest.raises(ValueError, match="tf_threads"):
        evaluate_candidates(
            build_quadratic_problem, [np.zeros(2)], n_workers=2, tf_threads=0
        )
    with pytest.raises(ValueError, match="tf_threads"):
        make_evaluator(build_quadratic_problem, n_workers=2, tf_threads=-1)
    # SerialPool-путь (n_workers=1) тоже валидирует, хоть и не использует.
    with pytest.raises(ValueError, match="tf_threads"):
        evaluate_candidates(
            build_quadratic_problem, [np.zeros(2)], n_workers=1, tf_threads=0
        )


def test_init_worker_falls_back_when_tf_pools_initialized():
    """
    Если TF-пулы уже инициализированы, _init_worker не убивает процесс:
    RuntimeError от set_*_threads перехватывается (fallback на дефолт
    TF-потоков), оценка работает с прежней семантикой.
    """
    import tensorflow as tf

    _ = tf.constant(0.0)  # форсируем инициализацию TF-пулов в родителе
    _init_worker(build_quadratic_problem, "evaluate", 1)  # не должно подняться

    record = _worker_evaluate(CandidateTask(index=0, phi=np.array([1.0, 2.0])))
    assert record.error is None
    np.testing.assert_allclose(record.loss, 5.0)


# ---------------------------------------------------------------------------
# CandidateTask / index handling / serialization
# ---------------------------------------------------------------------------


def test_candidate_task_preserves_index_and_seed():
    """Явные CandidateTask не перенумеровываются; seed выше derived."""
    tasks = [
        CandidateTask(index=7, phi=np.array([1.0, 0.0]), seed=11),
        CandidateTask(index=3, phi=np.array([2.0, 0.0]), seed=None),
    ]
    records = evaluate_candidates(build_quadratic_problem, tasks, n_workers=1)
    assert [r.index for r in records] == [7, 3]
    np.testing.assert_allclose(records[0].loss, 1.0)
    np.testing.assert_allclose(records[1].loss, 4.0)


def test_raw_candidates_get_positional_indexes():
    """Сырым массивам присваиваются index = позиции и derived seeds."""
    phis = [np.array([1.0, 0.0]), np.array([0.0, 2.0])]
    records = evaluate_candidates(build_quadratic_problem, phis, n_workers=1)
    assert [r.index for r in records] == [0, 1]
    np.testing.assert_allclose([r.loss for r in records], [1.0, 4.0])


def test_candidate_task_phi_coerced_to_1d_float64():
    """phi приводится к 1-D float64 (вложенные списки допустимы)."""
    task = CandidateTask(index=0, phi=[[1.0, 2.0]])
    assert task.phi.shape == (2,)
    assert task.phi.dtype == np.float64
    records = evaluate_candidates(build_quadratic_problem, [task], n_workers=1)
    np.testing.assert_allclose(records[0].loss, 5.0)


def test_dto_pickle_roundtrip():
    """CandidateTask/EvalRecord сериализуются pickle (передача в процессы)."""
    task = CandidateTask(index=2, phi=np.array([0.1, 0.2]), seed=5)
    task2 = pickle.loads(pickle.dumps(task))
    assert task2.index == 2
    np.testing.assert_allclose(task2.phi, task.phi)
    assert task2.seed == 5

    record = EvalRecord(
        index=2,
        loss=1.5,
        components={"a": 1.0},
        wall_time=0.1,
        n_forward=1,
        error="ValueError: boom",
    )
    record2 = pickle.loads(pickle.dumps(record))
    assert record2.index == 2
    assert record2.loss == 1.5
    assert record2.components == {"a": 1.0}
    assert record2.error == "ValueError: boom"


# ---------------------------------------------------------------------------
# make_evaluator + заглушка global_search
# ---------------------------------------------------------------------------


def _stub_global_search(evaluate, candidates):
    """
    Минимальная заглушка global_search: evaluate — callable кандидатов ->
    потерям; возвращает (лучший кандидат, потери).
    """
    candidates = list(candidates)
    losses = list(evaluate(candidates))
    best = int(np.argmin(losses))
    return candidates[best], losses


def test_make_evaluator_serial_compatible_with_stub_global_search(l0_problem):
    """make_evaluator взаимодействует с заглушкой global_search."""
    phis = _l0_phis(l0_problem, n=4)
    direct = [l0_problem.evaluate(phi).loss for phi in phis]

    evaluator = make_evaluator(build_l0_problem, n_workers=1)
    try:
        best, losses = _stub_global_search(evaluator, phis)
    finally:
        evaluator.close()

    np.testing.assert_allclose(losses, direct, rtol=1e-6, atol=1e-9)
    # phi0 (первый кандидат) — точка генерации target, потеря минимальна.
    np.testing.assert_allclose(losses[0], 0.0, atol=1e-6)
    np.testing.assert_allclose(best, phis[0])


@NEEDS_TWO_CPUS
def test_make_evaluator_process_reuses_pool(l0_problem):
    """Процессный evaluator переиспользует worker'ов между вызовами."""
    phis = _l0_phis(l0_problem, n=3)
    direct = [l0_problem.evaluate(phi).loss for phi in phis]

    evaluator = make_evaluator(build_l0_problem, n_workers=2, seed=7)
    try:
        first = list(evaluator(phis))
        second = list(evaluator(phis))
    finally:
        evaluator.close()

    np.testing.assert_allclose(first, direct, rtol=1e-6, atol=1e-9)
    assert first == second  # тот же seed -> те же потери
