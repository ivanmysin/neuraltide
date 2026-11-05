"""
Тесты P2-интеграторов: AdaptiveRK 3(2), IMEX-Euler, Rosenbrock ROS2,
Exponential Euler, stiffness_report/classify_failure и BPTT-градиенты.

Фейковые популяции намеренно минимальны: задают известную динамику
(линейные системы с аналитическим спектром/решением), чтобы проверять
порядок сходимости, устойчивость на жёстких задачах и спектр отчёта
по аналитике.
"""

import warnings

import numpy as np
import pytest
import tensorflow as tf
from scipy.linalg import expm

import neuraltide.config
from neuraltide.core.base import PopulationModel
from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.inputs import ConstantRateGenerator
from neuraltide.integrators import (
    AdaptiveRKIntegrator,
    EulerIntegrator,
    ExponentialEulerIntegrator,
    IMEXEulerIntegrator,
    RK4Integrator,
    RosenbrockIntegrator,
    classify_failure,
    stiffness_report,
)
from neuraltide.optimization import CalibrationProblem
from neuraltide.populations import WilsonCowan
from neuraltide.synapses import StaticSynapse

DT = 0.5


def _zeros_inp(n_units=1):
    dtype = neuraltide.config.get_dtype()
    return {
        "I_syn": tf.zeros([1, n_units], dtype=dtype),
        "g_syn": tf.zeros([1, n_units], dtype=dtype),
    }


class Linear2DPopulation(PopulationModel):
    """Линейная система y' = A y (n_units=1, d=2) с заданной матрицей A.

    initial="zeros" — стационарная траектория (для тестов спектра);
    "ones" — ненулевые начальные условия (для тестов устойчивости).
    """

    def __init__(self, a_matrix, dt=DT, initial="zeros"):
        super().__init__(n_units=1, dt=dt, name="lin2d")
        dtype = neuraltide.config.get_dtype()
        self.a_matrix = tf.constant(a_matrix, dtype=dtype)
        if isinstance(initial, str):
            if initial == "zeros":
                self._initial = [0.0, 0.0]
            elif initial == "ones":
                self._initial = [1.0, 1.0]
            else:
                raise ValueError(f"unknown initial '{initial}'")
        else:
            self._initial = [float(v) for v in initial]
        self.state_size = [tf.TensorShape([1, 1]), tf.TensorShape([1, 1])]

    def get_initial_state(self, batch_size=1):
        dtype = neuraltide.config.get_dtype()
        return [
            tf.constant([[self._initial[0]]], dtype=dtype),
            tf.constant([[self._initial[1]]], dtype=dtype),
        ]

    def derivatives(self, state, total_synaptic_input):
        flat = tf.stack([state[0][0], state[1][0]])
        out = self.a_matrix @ flat
        return [out[0][tf.newaxis, :], out[1][tf.newaxis, :]]

    def get_firing_rate(self, state):
        return state[0]

    @property
    def parameter_spec(self):
        return {}


class StiffSplitPopulation(Linear2DPopulation):
    """Линейная система с разделением: жёсткая диагональная часть + остаток."""

    def derivatives_split(self, state, total_synaptic_input):
        f = self.derivatives(state, total_synaptic_input)
        stiff = [-1000.0 * state[0], tf.zeros_like(state[1])]
        return ([f[0] - stiff[0], f[1] - stiff[1]], stiff)

    def linear_operator(self, state, total_synaptic_input):
        dtype = neuraltide.config.get_dtype()
        a_stiff = tf.constant([[-1000.0, 0.0], [0.0, 0.0]], dtype=dtype)
        b = tf.zeros([1, 2], dtype=dtype)
        return a_stiff[tf.newaxis, :, :], b


class DecayPopulation(PopulationModel):
    """Скалярное затухание y' = lam·y, y(0) = 1 (n_units=1, d=1)."""

    def __init__(self, lam=-1.0, dt=DT):
        super().__init__(n_units=1, dt=dt, name="decay")
        self.lam = float(lam)
        self.state_size = [tf.TensorShape([1, 1])]

    def get_initial_state(self, batch_size=1):
        dtype = neuraltide.config.get_dtype()
        return [tf.ones([1, 1], dtype=dtype)]

    def derivatives(self, state, total_synaptic_input):
        return [self.lam * state[0]]

    def get_firing_rate(self, state):
        return state[0]

    @property
    def parameter_spec(self):
        return {}


class NonlinearPopulation(PopulationModel):
    """y' = y − y³, y(0) = 2 → сходится к 1 (нелинейность для адаптации)."""

    def __init__(self, dt=1.0):
        super().__init__(n_units=1, dt=dt, name="nonlin")
        self.state_size = [tf.TensorShape([1, 1])]

    def get_initial_state(self, batch_size=1):
        dtype = neuraltide.config.get_dtype()
        return [tf.constant([[2.0]], dtype=dtype)]

    def derivatives(self, state, total_synaptic_input):
        return [state[0] - state[0] ** 3]

    def get_firing_rate(self, state):
        return state[0]

    @property
    def parameter_spec(self):
        return {}


# --------------------------------------------------------------------- #
# Хуки разделения (core/base.py)
# --------------------------------------------------------------------- #


class TestDerivativesSplitFallback:
    def test_default_split_sums_to_derivatives(self):
        """По умолчанию f_nonstiff + f_stiff == derivatives, f_stiff == 0."""
        neuraltide.config.set_dtype(tf.float64)
        pop = Linear2DPopulation([[-3.0, 1.0], [0.5, -1.0]])
        state = pop.get_initial_state()
        f_ns, f_s = pop.derivatives_split(state, _zeros_inp())
        f = pop.derivatives(state, _zeros_inp())
        assert len(f_ns) == len(f) == len(f_s)
        for i in range(len(f)):
            np.testing.assert_allclose(
                (f_ns[i] + f_s[i]).numpy(), f[i].numpy(), rtol=1e-12
            )
            np.testing.assert_array_equal(f_s[i].numpy(), 0.0)

    def test_linear_operator_default_is_none(self):
        neuraltide.config.set_dtype(tf.float64)
        pop = Linear2DPopulation([[-3.0, 1.0], [0.5, -1.0]])
        assert pop.linear_operator(pop.get_initial_state(), _zeros_inp()) is None

    def test_wilson_cowan_uses_default_hooks(self):
        """Встроенные популяции не переопределяют хуки — fallback корректен."""
        neuraltide.config.set_dtype(tf.float64)
        pop = WilsonCowan(
            dt=DT,
            params={
                "tau": 10.0,
                "a": 1.0,
                "theta": 0.0,
                "w": 0.5,
                "I_ext": 0.8,
                "max_rate": 100.0,
            },
        )
        state = pop.get_initial_state()
        f_ns, f_s = pop.derivatives_split(state, _zeros_inp())
        f = pop.derivatives(state, _zeros_inp())
        np.testing.assert_allclose((f_ns[0] + f_s[0]).numpy(), f[0].numpy(), rtol=1e-12)
        np.testing.assert_array_equal(f_s[0].numpy(), 0.0)


# --------------------------------------------------------------------- #
# Adaptive RK 3(2)
# --------------------------------------------------------------------- #


class TestAdaptiveRK:
    def test_fixed_mode_third_order(self):
        """Фиксированный субшаг: глобальная ошибка ∝ h³ (ratio ≈ 8)."""
        neuraltide.config.set_dtype(tf.float64)

        def run(h0):
            pop = DecayPopulation(lam=-1.0, dt=1.0)
            integ = AdaptiveRKIntegrator(adapt=False, h0=h0)
            state = pop.get_initial_state()
            for _ in range(5):
                state, _ = integ.step(pop, state, _zeros_inp())
            return abs(float(state[0][0, 0]) - np.exp(-5.0))

        e_fine = run(0.125)
        e_coarse = run(0.25)
        ratio = e_coarse / e_fine
        assert 6.5 < ratio < 9.5, f"expected ~8, got {ratio}"
        assert e_fine < 1e-5

    def test_adaptive_mode_converges_with_substeps(self):
        """Адаптивный режим на нелинейной задаче: точность ~ допуску,
        число субшагов фиксируется в статистике."""
        neuraltide.config.set_dtype(tf.float64)
        pop = NonlinearPopulation(dt=1.0)
        integ = AdaptiveRKIntegrator(rtol=1e-7, atol=1e-9)
        state = pop.get_initial_state()
        for _ in range(5):
            state, local_err = integ.step(pop, state, _zeros_inp())

        # Референс: RK4 с мелким шагом.
        ref_pop = NonlinearPopulation(dt=0.01)
        ref_state = ref_pop.get_initial_state()
        ref_integ = RK4Integrator()
        for _ in range(500):
            ref_state, _ = ref_integ.step(ref_pop, ref_state, _zeros_inp())

        err = abs(float(state[0][0, 0]) - float(ref_state[0][0, 0]))
        assert err < 1e-4, f"adaptive error {err:.3g}"
        assert integ.last_n_substeps is not None
        assert integ.last_n_substeps >= 1
        assert float(local_err[0]) >= 0.0

    def test_local_error_accumulated(self):
        """local_error ненулевой на нелинейной динамике (как у RK4)."""
        neuraltide.config.set_dtype(tf.float64)
        pop = NonlinearPopulation(dt=0.5)
        integ = AdaptiveRKIntegrator(rtol=1e-8, atol=1e-10)
        _, local_err = integ.step(pop, pop.get_initial_state(), _zeros_inp())
        assert float(local_err[0]) > 0.0

    def test_step_synapse_delegates_explicit_euler(self):
        """Синапсы в P2 явные: step_synapse работает (делегирует Эйлеру)."""
        neuraltide.config.set_dtype(tf.float64)
        syn = StaticSynapse(
            n_pre=1,
            n_post=1,
            dt=DT,
            params={"gsyn_max": 0.1, "pconn": [[1.0]], "e_r": 0.0},
        )
        integ = AdaptiveRKIntegrator()
        state = syn.get_initial_state()
        pre = tf.constant([[10.0]], dtype=tf.float64)
        post_v = tf.zeros([1, 1], dtype=tf.float64)
        new_state, err = integ.step_synapse(syn, state, pre, post_v, DT)
        assert len(new_state) == len(state)
        assert float(err[0]) == 0.0


# --------------------------------------------------------------------- #
# Rosenbrock ROS2 на жёсткой задаче
# --------------------------------------------------------------------- #

STIFF_A = [[-1000.0, 0.0], [0.0, -1.0]]


class TestRosenbrockStiff:
    def test_rk4_diverges_rosenbrock_stable(self):
        """При dt, где RK4 нестабилен (|z_fast| = 500 >> 2.78), ROS2 устойчив."""
        neuraltide.config.set_dtype(tf.float64)
        pop = Linear2DPopulation(STIFF_A, dt=0.5, initial="ones")
        state0 = pop.get_initial_state()

        rk4_state, _ = RK4Integrator().step(pop, state0, _zeros_inp())
        ros_state, _ = RosenbrockIntegrator().step(pop, state0, _zeros_inp())

        rk4_fast = abs(float(rk4_state[0][0, 0]))
        ros_fast = abs(float(ros_state[0][0, 0]))
        assert rk4_fast > 1e3, f"RK4 expected unstable, got {rk4_fast}"
        assert ros_fast < 1.0, f"ROS2 expected stable, got {ros_fast}"
        # Медленная компонента: ROS2 с одним субшагом близок к e^{-0.5}.
        ros_slow = float(ros_state[1][0, 0])
        assert abs(ros_slow - np.exp(-0.5)) < 5e-3

    def test_rosenbrock_accuracy_with_substeps(self):
        """Субшаги поднимают точность до референса scipy.linalg.expm."""
        neuraltide.config.set_dtype(tf.float64)
        pop = Linear2DPopulation(STIFF_A, dt=0.5, initial="ones")
        exact = expm(np.asarray(STIFF_A) * 0.5) @ np.ones(2)

        integ = RosenbrockIntegrator(n_substeps=8)
        state, _ = integ.step(pop, pop.get_initial_state(), _zeros_inp())
        numerical = np.array([float(state[0][0, 0]), float(state[1][0, 0])])
        rel = np.linalg.norm(numerical - exact) / np.linalg.norm(exact)
        assert rel < 1e-3, f"rel err {rel:.3g}"

    def test_rosenbrock_on_wilson_cowan(self):
        """d=1 у WilsonCowan: метод работает на встроенной популяции."""
        neuraltide.config.set_dtype(tf.float64)
        pop = WilsonCowan(
            dt=DT,
            params={
                "tau": 10.0,
                "a": 1.0,
                "theta": 0.0,
                "w": 0.5,
                "I_ext": 0.8,
                "max_rate": 100.0,
            },
        )
        state, err = RosenbrockIntegrator().step(
            pop, pop.get_initial_state(), _zeros_inp()
        )
        assert np.isfinite(float(state[0][0, 0]))
        assert float(err[0]) >= 0.0


# --------------------------------------------------------------------- #
# Exponential Euler
# --------------------------------------------------------------------- #


class TestExponentialEuler:
    def test_exact_on_linear_system(self):
        """На линейной системе шаг точен до машинного нуля при любом h."""
        neuraltide.config.set_dtype(tf.float64)
        pop = Linear2DPopulation(STIFF_A, dt=0.5, initial="ones")
        exact = expm(np.asarray(STIFF_A) * 0.5) @ np.ones(2)
        state, _ = ExponentialEulerIntegrator().step(
            pop, pop.get_initial_state(), _zeros_inp()
        )
        numerical = np.array([float(state[0][0, 0]), float(state[1][0, 0])])
        np.testing.assert_allclose(numerical, exact, rtol=1e-8, atol=1e-12)

    def test_nonlinear_accuracy(self):
        """Нелинейная задача: сходимость к RK4-референсу."""
        neuraltide.config.set_dtype(tf.float64)
        ref_pop = NonlinearPopulation(dt=0.01)
        ref_state = ref_pop.get_initial_state()
        ref_integ = RK4Integrator()
        for _ in range(500):
            ref_state, _ = ref_integ.step(ref_pop, ref_state, _zeros_inp())

        pop = NonlinearPopulation(dt=0.5)
        state = pop.get_initial_state()
        integ = ExponentialEulerIntegrator(n_substeps=5)
        for _ in range(10):
            state, _ = integ.step(pop, state, _zeros_inp())
        err = abs(float(state[0][0, 0]) - float(ref_state[0][0, 0]))
        assert err < 1e-4, f"exp euler error {err:.3g}"

    def test_large_block_raises(self):
        """d+1 > max_block_size: документированная ошибка по умолчанию."""
        neuraltide.config.set_dtype(tf.float64)
        pop = Linear2DPopulation(STIFF_A, dt=0.5, initial="ones")
        integ = ExponentialEulerIntegrator(max_block_size=1)
        with pytest.raises(ValueError, match="max_block_size"):
            integ.step(pop, pop.get_initial_state(), _zeros_inp())

    def test_large_block_explicit_fallback(self):
        """on_large_block='euler' — честный fallback с warning."""
        neuraltide.config.set_dtype(tf.float64)
        pop = Linear2DPopulation(STIFF_A, dt=0.5, initial="ones")
        integ = ExponentialEulerIntegrator(max_block_size=1, on_large_block="euler")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            state, _ = integ.step(pop, pop.get_initial_state(), _zeros_inp())
        assert any("Falling back to explicit Euler" in str(w.message) for w in caught)
        # Fallback совпал с явным Эйлером.
        euler_state, _ = EulerIntegrator().step(
            pop, pop.get_initial_state(), _zeros_inp()
        )
        np.testing.assert_allclose(state[0].numpy(), euler_state[0].numpy(), rtol=1e-12)


# --------------------------------------------------------------------- #
# IMEX-Euler
# --------------------------------------------------------------------- #


class TestIMEXEuler:
    def test_stable_on_stiff_split(self):
        """Жёсткая часть неявно: устойчив при λ·h = −500, где явный Эйлер
        расходится."""
        neuraltide.config.set_dtype(tf.float64)
        pop = StiffSplitPopulation(STIFF_A, dt=0.5, initial="ones")
        state, _ = IMEXEulerIntegrator().step(
            pop, pop.get_initial_state(), _zeros_inp()
        )
        fast = abs(float(state[0][0, 0]))
        slow = float(state[1][0, 0])
        assert fast < 1.0, f"IMEX fast component {fast}"
        # Медленная компонента — явный Эйлер на остатке: 1 − 0.5 = 0.5.
        assert abs(slow - 0.5) < 1e-12

    def test_zero_split_falls_back_to_explicit_euler(self):
        """Нулевой split + linear_operator None == явный Эйлер + warning."""
        neuraltide.config.set_dtype(tf.float64)
        pop = DecayPopulation(lam=-2.0, dt=0.25)
        integ = IMEXEulerIntegrator()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            state_imex, _ = integ.step(pop, pop.get_initial_state(), _zeros_inp())
        assert any("explicit Euler" in str(w.message) for w in caught)

        euler_state, _ = EulerIntegrator().step(
            pop, pop.get_initial_state(), _zeros_inp()
        )
        np.testing.assert_allclose(
            state_imex[0].numpy(), euler_state[0].numpy(), rtol=1e-12
        )

    def test_no_operator_nonzero_stiff_warns(self):
        """f_stiff != 0 без linear_operator: жёсткая часть явно + warning."""
        neuraltide.config.set_dtype(tf.float64)

        class SplitOnly(Linear2DPopulation):
            def derivatives_split(self, state, total_synaptic_input):
                f = self.derivatives(state, total_synaptic_input)
                stiff = [-1000.0 * state[0], tf.zeros_like(state[1])]
                return ([f[0] - stiff[0], f[1] - stiff[1]], stiff)

        pop = SplitOnly(STIFF_A, dt=0.5, initial="ones")
        integ = IMEXEulerIntegrator()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            integ.step(pop, pop.get_initial_state(), _zeros_inp())
        assert any("treated explicitly" in str(w.message) for w in caught)


# --------------------------------------------------------------------- #
# stiffness_report / classify_failure
# --------------------------------------------------------------------- #

KNOWN_A = [[-3.0, 1.0], [0.5, -1.0]]


def _build_known_network():
    graph = NetworkGraph(dt=DT)
    graph.add_population("x", Linear2DPopulation(KNOWN_A, dt=DT))
    return NetworkRNN(graph, EulerIntegrator())


@pytest.mark.filterwarnings("ignore:Population .* has no incoming synapses")
class TestStiffnessReport:
    @pytest.mark.parametrize("method", ["jvp", "fd"])
    def test_spectrum_matches_analytics(self, method):
        """Спектр якобиана == собственные значения A с точностью 2%."""
        neuraltide.config.set_dtype(tf.float64)
        network = _build_known_network()
        report = stiffness_report(network, t_points=4, method=method)

        lam_max = report["lambda_max"]
        lam_max_exact = float(np.max(np.abs(np.linalg.eigvals(KNOWN_A))))
        assert abs(lam_max - lam_max_exact) / lam_max_exact < 0.02

        limit = report["explicit_dt_limit"]
        limit_exact = 2.0 / lam_max_exact
        assert abs(limit - limit_exact) / limit_exact < 0.02

        ratio = report["stiffness_ratio"]
        ev = np.linalg.eigvals(KNOWN_A)
        ratio_exact = float(np.max(np.abs(ev.real)) / np.min(np.abs(ev.real)))
        assert abs(ratio - ratio_exact) / ratio_exact < 0.02
        assert report["t_points"] == 4
        assert set(report["populations"].keys()) == {"x"}

    def test_euler_stable_flag(self):
        neuraltide.config.set_dtype(tf.float64)
        network = _build_known_network()
        report = stiffness_report(network, t_points=2)
        assert report["dt_network"] == DT
        assert report["euler_stable"] == (DT <= report["explicit_dt_limit"])

    def test_explicit_dt_limit_complex_spectrum(self):
        """Лимит Эйлера для комплексной пары λ = −1 ± 10i: покомпонентная
        формула 2|Re λ|/|λ|² ≈ 0.0198 (старая 2/|λ| давала бы ~0.199 —
        завышение лимита в 10 раз)."""
        neuraltide.config.set_dtype(tf.float64)
        a_complex = [[-1.0, -10.0], [10.0, -1.0]]
        graph = NetworkGraph(dt=DT)
        graph.add_population("osc", Linear2DPopulation(a_complex, dt=DT))
        network = NetworkRNN(graph, EulerIntegrator())

        report = stiffness_report(network, t_points=2)

        ev = np.linalg.eigvals(np.asarray(a_complex))
        assert np.max(np.abs(ev.imag)) > 0.0  # контроль: спектр комплексный
        limit = report["explicit_dt_limit"]
        limit_exact = 2.0 * abs(ev[0].real) / abs(ev[0]) ** 2
        assert limit == pytest.approx(limit_exact, rel=1e-3)
        assert limit < 0.02  # старая формула 2/|λ| дала бы ~0.199
        assert report["euler_stable"] is False  # dt=0.5 >> 0.0198

    def test_explicit_dt_limit_unstable_mode_is_zero(self):
        """Мода с Re λ >= 0: устойчивого шага Эйлера нет — лимит 0.0."""
        neuraltide.config.set_dtype(tf.float64)
        graph = NetworkGraph(dt=DT)
        graph.add_population(
            "unstable", Linear2DPopulation([[1.0, 0.0], [0.0, -4.0]], dt=DT)
        )
        network = NetworkRNN(graph, EulerIntegrator())

        report = stiffness_report(network, t_points=2)

        assert report["explicit_dt_limit"] == 0.0
        assert report["euler_stable"] is False

    def test_zero_input_per_population_shapes(self):
        """Две динамические популяции разной n_units (3 и 1): нулевой
        синаптический вход строится под каждую популяцию, отчёт не падает
        и даёт корректную форму спектра."""
        neuraltide.config.set_dtype(tf.float64)

        def _wc(n):
            return WilsonCowan(
                dt=DT,
                params={
                    "tau": [10.0] * n,
                    "a": 1.0,
                    "theta": [0.0] * n,
                    "w": [0.5] * n,
                    "I_ext": [0.8] * n,
                    "max_rate": [100.0] * n,
                },
            )

        graph = NetworkGraph(dt=DT)
        graph.add_population("big", _wc(3))
        graph.add_population("small", _wc(1))
        network = NetworkRNN(graph, EulerIntegrator())

        report = stiffness_report(network, t_points=3)

        assert set(report["populations"].keys()) == {"big", "small"}
        ev_big = report["populations"]["big"]["eigenvalues"]
        ev_small = report["populations"]["small"]["eigenvalues"]
        assert ev_big.shape == (3, 3, 1)
        assert ev_small.shape == (3, 1, 1)
        assert np.all(np.isfinite(ev_big.real))
        assert np.all(np.isfinite(ev_small.real))
        # Аналитика WilsonCowan при нулевом I_syn: λ = (−1 + w·F′)/τ,
        # F′ = a·F(1−F), F = σ(a·(I_ext − θ)) = σ(0.8).
        sig = 1.0 / (1.0 + np.exp(-0.8))
        lam = (-1.0 + 0.5 * sig * (1.0 - sig)) / 10.0
        np.testing.assert_allclose(ev_big, lam, rtol=2e-2)
        np.testing.assert_allclose(ev_small, lam, rtol=2e-2)
        assert report["explicit_dt_limit"] == pytest.approx(2.0 / abs(lam), rel=2e-2)

    def test_problem_path_and_phi_restore(self, l0_factory):
        """CalibrationProblem: отчёт по протоколу задачи; phi применяется
        и восстанавливается."""
        neuraltide.config.set_dtype(tf.float64)
        network, space, t_seq, inputs, loss = l0_factory()
        problem = CalibrationProblem(
            network, space, t_seq, inputs=inputs, objective=loss
        )
        phi0 = np.asarray(problem.pack(), dtype=np.float64)
        report = stiffness_report(problem, phi=phi0, t_points=3)
        assert report["t_points"] == 3
        assert "exc" in report["populations"]
        ev = report["populations"]["exc"]["eigenvalues"]
        # WilsonCowan при нулевом I_syn: λ = (−1 + w·F')/τ, Re λ < 0.
        assert np.all(np.isfinite(ev.real))
        assert np.all(ev.real < 0.0)
        phi_restored = np.asarray(problem.pack(), dtype=np.float64)
        np.testing.assert_allclose(phi_restored, phi0, rtol=1e-12)


class TestClassifyFailure:
    @pytest.mark.filterwarnings("ignore:Population .* has no incoming synapses")
    def test_ok_and_nan(self):
        neuraltide.config.set_dtype(tf.float64)
        network = _build_known_network()
        t_seq = tf.constant((np.arange(4) * DT)[None, :, None], dtype=tf.float64)
        output = network(t_seq)
        result = classify_failure(output)
        assert result["status"] == "ok"
        assert result["max_abs"] == pytest.approx(0.0)

        bad = classify_failure([tf.constant([[np.nan]])])
        assert bad["status"] == "nan"

        big = classify_failure([tf.constant([[1e9]])])
        assert big["status"] == "diverged"

    def test_unknown_type_raises(self):
        with pytest.raises(TypeError):
            classify_failure(42.0)


# --------------------------------------------------------------------- #
# BPTT-градиент против конечных разностей
# --------------------------------------------------------------------- #


def _build_grad_network():
    """Малая сеть L0-подобной структуры в float64."""
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
        params={"gsyn_max": 0.05, "pconn": [[1.0]], "e_r": 0.0},
        name="in->exc",
    )
    gen = ConstantRateGenerator(dt=DT, params={"rate": 10.0})
    graph = NetworkGraph(dt=DT)
    graph.declare_input("in", n_units=1)
    graph.add_population("exc", pop)
    graph.add_synapse("in->exc", syn, src="in", tgt="exc")
    return NetworkRNN(graph, EulerIntegrator()), graph, gen, pop


def _loss_for(network, graph, gen, pop, integrator):
    network._integrator = integrator
    n_steps = 20
    t_seq = tf.constant(
        (np.arange(n_steps, dtype=np.float64) * DT)[None, :, None],
        dtype=tf.float64,
    )
    inputs = graph.pack_inputs({"in": gen(t_seq)})
    rates = network(t_seq, inputs=inputs).firing_rates["exc"]
    return tf.reduce_sum(rates**2)


class TestBPTTGradients:
    @pytest.mark.parametrize(
        "integrator",
        [
            AdaptiveRKIntegrator(rtol=1e-7, atol=1e-9),
            IMEXEulerIntegrator(),
            RosenbrockIntegrator(),
            ExponentialEulerIntegrator(),
        ],
        ids=["adaptive_rk", "imex", "rosenbrock", "exponential"],
    )
    def test_gradient_matches_finite_differences(self, integrator):
        """AD-градиент через развёртку дифференцируемых шагов совпадает
        с центральной разностью (finite-difference scan по eps)."""
        neuraltide.config.set_dtype(tf.float64)
        network, graph, gen, pop = _build_grad_network()

        with tf.GradientTape() as tape:
            loss = _loss_for(network, graph, gen, pop, integrator)
        grad_ad = float(tape.gradient(loss, pop.tau)[0])

        # FD-scan: два шага разности, оба должны совпасть с AD.
        tau0 = float(pop.tau[0])
        estimates = []
        for eps in (3e-5, 1e-4):
            pop.tau.assign([tau0 + eps])
            loss_plus = float(_loss_for(network, graph, gen, pop, integrator))
            pop.tau.assign([tau0 - eps])
            loss_minus = float(_loss_for(network, graph, gen, pop, integrator))
            estimates.append((loss_plus - loss_minus) / (2.0 * eps))
        pop.tau.assign([tau0])

        assert not np.isnan(grad_ad)
        for est in estimates:
            rel = abs(est - grad_ad) / max(abs(grad_ad), 1e-12)
            assert rel < 1e-3, (
                f"integrator={type(integrator).__name__}: AD {grad_ad:.8g} "
                f"vs FD {est:.8g} (rel {rel:.3g})"
            )
        # Самосогласованность FD-scan: оценки на разных eps близки.
        rel_fd = abs(estimates[0] - estimates[1]) / max(abs(estimates[1]), 1e-12)
        assert rel_fd < 1e-2, f"FD estimates inconsistent: {estimates}"
