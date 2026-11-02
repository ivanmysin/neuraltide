"""
Тесты sensitivities (этап P1): forward JVP и якобиан невязок, checkpointed
градиент, truncated BPTT.

Сеть — L0-фикстура conftest (WilsonCowan 1 юнит + StaticSynapse от
ConstantRate; 3 trainable-параметра). Все проверки чувствительностей —
в float64 (dtype устанавливается до построения сети через l0_factory).

Покрываются:
- JVP против центральных конечных разностей (rel < 1e-3);
- J.T @ r == grad из value_and_gradient (allclose, float64);
- residual_jacobian: DTO, shape, счётчики; guard max_params;
- residual_jvp: TypeError при objective без residual-интерфейса;
- tensor-backed запись начального состояния (x0) в JVP/якобиане;
- checkpointed grad == plain BPTT (allclose 1e-6), число сегментов,
  счётчики; принудительный custom_gradient-backend;
- truncated_bptt: снижение free_run_loss, записанные final_loss /
  free_run_loss, изменение поведения при warm_steps > 0.
"""

import numpy as np
import pytest
import tensorflow as tf

import neuraltide.config
import neuraltide.optimization.sensitivities as sensitivities
from neuraltide.optimization import CalibrationProblem, JacobianResult
from neuraltide.optimization.sensitivities import (
    CheckpointedGradient,
    TruncatedResult,
    residual_jacobian,
    residual_jvp,
    truncated_bptt,
    value_and_gradient_checkpointed,
)

DT = 0.5


def _build_f64(l0_factory) -> CalibrationProblem:
    """L0-задача в float64."""
    neuraltide.config.set_dtype(tf.float64)
    network, space, t_seq, inputs, loss = l0_factory()
    return CalibrationProblem(network, space, t_seq, inputs=inputs, objective=loss)


def _perturbed_phi(problem: CalibrationProblem, seed: int = 0) -> np.ndarray:
    """Точка вдали от истинных параметров (невязки ненулевые)."""
    phi0 = np.asarray(problem.pack(), dtype=np.float64)
    rng = np.random.default_rng(seed)
    scale = 0.05 * np.abs(phi0) + 1e-3
    return phi0 + rng.standard_normal(problem.size) * scale


@pytest.fixture
def problem_f64(l0_factory) -> CalibrationProblem:
    return _build_f64(l0_factory)


@pytest.fixture
def x0_problem_f64(l0_factory) -> CalibrationProblem:
    """L0-задача + tensor-backed запись начального состояния x0/exc/0."""
    neuraltide.config.set_dtype(tf.float64)
    network, space, t_seq, inputs, loss = l0_factory()
    space.add_initial_state_entries(network, include=[("exc", 0)])
    return CalibrationProblem(network, space, t_seq, inputs=inputs, objective=loss)


class TestResidualJVP:
    def test_shapes_and_dtype(self, problem_f64):
        phi = _perturbed_phi(problem_f64)
        v = np.array([1.0, -0.5, 2.0], dtype=np.float64)
        r, jv = residual_jvp(problem_f64, phi, v)
        assert r.ndim == 1 and jv.ndim == 1
        assert r.shape == jv.shape
        assert r.dtype == np.float64 and jv.dtype == np.float64
        assert np.linalg.norm(r) > 0.0

    def test_jvp_matches_finite_differences(self, problem_f64):
        phi = _perturbed_phi(problem_f64, seed=1)
        rng = np.random.default_rng(2)
        v = rng.standard_normal(problem_f64.size)
        v /= np.linalg.norm(v)
        _, jv = residual_jvp(problem_f64, phi, v)

        h = 1e-5
        r_plus = problem_f64.residual(phi + h * v)
        r_minus = problem_f64.residual(phi - h * v)
        fd = (r_plus - r_minus) / (2.0 * h)
        rel = np.linalg.norm(jv - fd) / max(np.linalg.norm(fd), 1e-12)
        assert rel < 1e-3

    def test_bad_v_size(self, problem_f64):
        phi = _perturbed_phi(problem_f64)
        with pytest.raises(ValueError, match="elements"):
            residual_jvp(problem_f64, phi, np.zeros(problem_f64.size + 1))

    def test_typeerror_without_residual_interface(self, problem_f64):
        phi = _perturbed_phi(problem_f64)

        scalar_problem = CalibrationProblem(
            problem_f64.network,
            problem_f64.parameter_space,
            problem_f64.t_sequence,
            inputs=problem_f64.inputs,
            objective=lambda predictions, network: tf.reduce_sum(
                predictions.firing_rates["exc"]
            ),
        )
        with pytest.raises(TypeError, match="residual"):
            residual_jvp(scalar_problem, phi, np.zeros(problem_f64.size))


class TestResidualJacobian:
    def test_jacobian_matches_finite_differences(self, problem_f64):
        phi = _perturbed_phi(problem_f64, seed=3)
        result = residual_jacobian(problem_f64, phi)
        J, r = result.J, result.r

        h = 1e-5
        J_fd = np.zeros_like(J)
        for i in range(problem_f64.size):
            e = np.zeros(problem_f64.size)
            e[i] = h
            J_fd[:, i] = (
                problem_f64.residual(phi + e) - problem_f64.residual(phi - e)
            ) / (2.0 * h)
        rel = np.linalg.norm(J - J_fd) / max(np.linalg.norm(J_fd), 1e-12)
        assert rel < 1e-3
        assert np.allclose(r, problem_f64.residual(phi), atol=1e-12)

    def test_jtj_equals_value_and_gradient(self, problem_f64):
        """Jᵀ r == grad потери 0.5·||r||² (float64, allclose)."""
        phi = _perturbed_phi(problem_f64, seed=4)
        result = residual_jacobian(problem_f64, phi)
        eval_result, grad = problem_f64.value_and_gradient(phi)
        grad = np.asarray(grad, dtype=np.float64)

        jtj = result.J.T @ result.r
        assert np.allclose(jtj, grad, rtol=1e-6, atol=1e-8)
        # Потеря из DTO согласована с evaluate: loss == 0.5·||r||².
        assert result.loss == pytest.approx(0.5 * float(result.r @ result.r))
        assert eval_result.loss == pytest.approx(result.loss, rel=1e-12)

    def test_dto_shapes_and_counters(self, problem_f64):
        phi = _perturbed_phi(problem_f64)
        n_forward0 = problem_f64.n_forward
        result = residual_jacobian(problem_f64, phi)

        assert isinstance(result, JacobianResult)
        n = problem_f64.size
        assert result.J.shape == (result.r.shape[0], n)
        assert result.phi.shape == (n,)
        assert np.allclose(result.phi, phi, atol=1e-12)
        assert result.method == "forward"
        assert result.wall_time >= 0.0
        # Каждый unit-JVP: 1 оценка невязок + 1 JVP-прогон.
        assert result.n_forward == problem_f64.n_forward - n_forward0
        assert result.n_forward == 2 * n
        # Сериализация DTO.
        restored = JacobianResult.from_dict(result.to_dict())
        assert restored.J.shape == result.J.shape
        assert np.allclose(restored.J, result.J, atol=1e-15)

    def test_max_params_guard(self, problem_f64):
        phi = _perturbed_phi(problem_f64)
        with pytest.raises(ValueError, match="2-point"):
            residual_jacobian(problem_f64, phi, max_params=problem_f64.size - 1)
        # force=True снимает guard.
        result = residual_jacobian(
            problem_f64, phi, max_params=problem_f64.size - 1, force=True
        )
        assert result.J.shape[1] == problem_f64.size


def _x0_base_phi(problem: CalibrationProblem) -> np.ndarray:
    """
    Базовый вектор φ для пространства с записью x0 (pack(None) невозможен:
    у tensor-записи нет переменной — явные значения).
    """
    values = {
        "exc/tau": [8.0],
        "exc/I_ext": [0.8],
        "in->exc/gsyn_max": [[0.05]],
        "x0/exc/0": [0.0],
    }
    return np.asarray(problem.parameter_space.pack(values), dtype=np.float64)


class TestTensorBackedEntries:
    def test_x0_jvp_matches_finite_differences(self, x0_problem_f64):
        problem = x0_problem_f64
        rng = np.random.default_rng(5)
        phi = _x0_base_phi(problem) + rng.standard_normal(problem.size) * 0.05
        # Отодвигаем x0-координату от значения по умолчанию (0.0).
        phi[-1] = 0.4
        i_x0 = problem.size - 1
        e = np.zeros(problem.size)
        e[i_x0] = 1.0
        _, jv = residual_jvp(problem, phi, e)

        h = 1e-5
        fd = (problem.residual(phi + h * e) - problem.residual(phi - h * e)) / (2.0 * h)
        rel = np.linalg.norm(jv - fd) / max(np.linalg.norm(fd), 1e-12)
        assert rel < 1e-3

    def test_x0_jtj_equals_gradient(self, x0_problem_f64):
        problem = x0_problem_f64
        rng = np.random.default_rng(6)
        phi = _x0_base_phi(problem) + rng.standard_normal(problem.size) * 0.05
        phi[-1] = 0.4
        result = residual_jacobian(problem, phi)
        _, grad = problem.value_and_gradient(phi)
        grad = np.asarray(grad, dtype=np.float64)
        assert np.allclose(result.J.T @ result.r, grad, rtol=1e-6, atol=1e-8)


class TestCheckpointedGradient:
    def test_grad_matches_plain_bptt(self, problem_f64):
        phi = _perturbed_phi(problem_f64, seed=7)
        eval_plain, grad_plain = problem_f64.value_and_gradient(phi)
        grad_plain = np.asarray(grad_plain, dtype=np.float64)

        segment_length = 17
        result = value_and_gradient_checkpointed(problem_f64, phi, segment_length)

        assert isinstance(result, CheckpointedGradient)
        T = int(problem_f64.t_sequence.shape[1])
        assert result.segments == -(-T // segment_length)
        assert result.loss == pytest.approx(eval_plain.loss, rel=1e-12)
        assert np.allclose(result.grad, grad_plain, rtol=1e-6, atol=1e-8)
        assert result.wall_time >= 0.0
        assert result.peak_memory is None or result.peak_memory > 0.0

    def test_counters_and_single_segment(self, problem_f64):
        phi = _perturbed_phi(problem_f64)
        n_forward0 = problem_f64.n_forward
        n_gradient0 = problem_f64.n_gradient
        result = value_and_gradient_checkpointed(
            problem_f64, phi, segment_length=10_000
        )
        assert result.segments == 1
        # Прямые сегменты + пересчёт при обратном проходе.
        assert problem_f64.n_forward - n_forward0 == 2 * result.segments
        assert problem_f64.n_gradient - n_gradient0 == 1

    def test_forced_custom_gradient_backend(self, problem_f64, monkeypatch):
        """Fallback tf.custom_gradient даёт тот же градиент."""
        phi = _perturbed_phi(problem_f64, seed=8)
        eval_plain, grad_plain = problem_f64.value_and_gradient(phi)
        grad_plain = np.asarray(grad_plain, dtype=np.float64)

        monkeypatch.setattr(
            sensitivities, "_select_checkpoint_backend", lambda: "custom_gradient"
        )
        result = value_and_gradient_checkpointed(problem_f64, phi, 13)
        assert result.loss == pytest.approx(eval_plain.loss, rel=1e-12)
        assert np.allclose(result.grad, grad_plain, rtol=1e-6, atol=1e-8)

    def test_bad_segment_length(self, problem_f64):
        with pytest.raises(ValueError, match="segment_length"):
            value_and_gradient_checkpointed(problem_f64, _perturbed_phi(problem_f64), 0)


class TestTruncatedBPTT:
    def test_reduces_loss_and_records(self, problem_f64):
        phi0 = _perturbed_phi(problem_f64, seed=9)
        loss0 = problem_f64.evaluate(phi0).loss

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.02)
        result = truncated_bptt(problem_f64, phi0, optimizer, horizon=20, epochs=3)

        assert isinstance(result, TruncatedResult)
        assert len(result.history) == 3
        assert result.history[0]["epoch"] == 0.0
        assert result.final_loss > 0.0
        assert result.free_run_loss > 0.0
        assert np.isfinite(result.final_loss)
        assert np.isfinite(result.free_run_loss)
        assert result.notes
        # Обучение снижает потерю полного прогона.
        assert result.free_run_loss < loss0

    def test_warm_steps_changes_behavior(self, l0_factory):
        """Перекрытие окон меняет траекторию обучения (и результат)."""
        problem_a = _build_f64(l0_factory)
        phi0 = _perturbed_phi(problem_a, seed=10)
        loss0 = problem_a.evaluate(phi0).loss

        res_a = truncated_bptt(
            problem_a,
            phi0,
            tf.keras.optimizers.Adam(learning_rate=0.02),
            horizon=20,
            epochs=2,
            warm_steps=0,
        )
        problem_b = _build_f64(l0_factory)
        res_b = truncated_bptt(
            problem_b,
            phi0.copy(),
            tf.keras.optimizers.Adam(learning_rate=0.02),
            horizon=20,
            epochs=2,
            warm_steps=8,
        )
        losses_a = [h["loss"] for h in res_a.history]
        losses_b = [h["loss"] for h in res_b.history]
        assert not np.allclose(losses_a, losses_b, rtol=1e-6, atol=1e-9)
        assert res_a.free_run_loss != pytest.approx(res_b.free_run_loss, rel=1e-6)
        # Оба режима при этом сходятся (loss снижается относительно старта).
        assert res_a.free_run_loss < loss0
        assert res_b.free_run_loss < loss0

    def test_validation(self, problem_f64):
        optimizer = tf.keras.optimizers.SGD(learning_rate=1.0)
        phi0 = _perturbed_phi(problem_f64)
        with pytest.raises(ValueError, match="horizon"):
            truncated_bptt(problem_f64, phi0, optimizer, horizon=0, epochs=1)
        with pytest.raises(ValueError, match="epochs"):
            truncated_bptt(problem_f64, phi0, optimizer, horizon=5, epochs=0)
        with pytest.raises(ValueError, match="warm_steps"):
            truncated_bptt(
                problem_f64, phi0, optimizer, horizon=5, epochs=1, warm_steps=5
            )
        with pytest.raises(TypeError, match="apply_gradients"):
            truncated_bptt(problem_f64, phi0, object(), horizon=5, epochs=1)
