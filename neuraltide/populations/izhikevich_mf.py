import tensorflow as tf

import neuraltide
import neuraltide.config
from neuraltide.core.base import PopulationModel
from neuraltide.core.types import TensorType, StateList, Dict


class IzhikevichMeanField(PopulationModel):
    """
    Популяционная модель Ижикевича (Montbrio-Pazo-Roxe) в безразмерной форме.

    Размерные уравнения (уравнение 8 в статье):
        C dV_i/dt = K(V_i - V_R)(V_i - V_T) - W_i + η_i + I_syn
        dW_i/dt = A(B(V_i - V_R) - W_i)
        если V_i > V_peak: V_i <- V_reset, W_i <- W_i + W_jump

    Безразмерные уравнения (уравнения 1-3 в статье):
        τ_pop dν/dt = Δη/π + 2ν⟨v⟩ - (α + g_total)·ν
        τ_pop d⟨v⟩/dt = ⟨v⟩² - α·⟨v⟩ - ⟨w⟩ + η̄ + I_syn - (π·ν)²
        τ_pop d⟨w⟩/dt = a·(b·⟨v⟩ - ⟨w⟩) + w_jump·ν

    Перевод размерных параметров в безразмерные (таблица 3 в статье):
        v = 1 + V/|V_R|
        w = W/(K|V_R|²)
        α = 1 + V_T/|V_R|
        a = C·A/(K·|V_R|)
        b = B/(K|V_R|)
        w_jump = W_jump/(K|V_R|²)
        η̄ = Ī/(K|V_R|²)
        Δη = ΔĪ/(K|V_R|²)
        τ_pop = C/(K·|V_R|)
        g_syn = G_syn/(K|V_R|)
        e_r = 1 + E_r/|V_R|

    Состояние: [r, v, w], каждое shape [1, n_units].
    Здесь r соответствует ν (частота разрядов), v — ⟨v⟩ (безразмерный потенциал),
    w — ⟨w⟩ (безразмерный ток адаптации).
    """

    def __init__(
        self,
        n_units: int,
        dt: float,
        V_R: float,
        V_T: float,
        V_peak: float,
        V_reset: float,
        C: float,
        K: float,
        A: float,
        B: float,
        W_jump: float,
        E_r: float,
        Delta_I: float,
        I_ext: float,
        eta_bar: float = 0.0,
        use_dimensionless: bool = False,
        **kwargs,
    ):
        """
        Параметры:
            n_units: число нейронов в популяции
            dt: шаг интегрирования [мс]
            V_R: потенциал покоя [мВ]
            V_T: пороговый потенциал [мВ]
            V_peak: пиковый потенциал [мВ]
            V_reset: потенциал сброса [мВ]
            C: мембранная ёмкость [пФ]
            K: параметр масштабирования [нСм/мВ]
            A: коэффициент скорости изменения адаптации [мс]
            B: чувствительность восстановления к мембранному потенциалу [нСм]
            W_jump: прирост тока адаптации после потенциала действия [пА]
            E_r: равновесный потенциал синапса [мВ] (для справки)
            Delta_I: разброс внешнего тока (半宽度 распределения Лоренца) [пА]
            I_ext: внешний ток (среднее распределения Лоренца) [пА]
            eta_bar: безразмерный внешний ток (по умолчанию 0)
            use_dimensionless: если True, параметры уже безразмерные
        """
        super().__init__(n_units=n_units, dt=dt, **kwargs)

        self.use_dimensionless = use_dimensionless

        if use_dimensionless:
            params = {
                'tau_pop': C / (K * abs(V_R)),
                'alpha': V_T / abs(V_R) + 1.0,
                'a': C * A / (K * abs(V_R)),
                'b': B / (K * abs(V_R)),
                'w_jump': W_jump / (K * abs(V_R)**2),
                'Delta_eta': {'value': Delta_I, 'trainable': False},
                'eta_bar': eta_bar,
                'e_r': E_r / abs(V_R) + 1.0,
                'v_rest': 1.0,
                'v_peak': 1.0 + V_peak / abs(V_R),
                'v_reset': 1.0 + V_reset / abs(V_R),
            }
        else:
            V_R_abs = abs(V_R)
            params = {
                'tau_pop': C / (K * V_R_abs),
                'alpha': 1.0 + V_T / V_R_abs,
                'a': C * A / (K * V_R_abs),
                'b': B / (K * V_R_abs),
                'w_jump': W_jump / (K * V_R_abs**2),
                'Delta_eta': {'value': Delta_I / (K * V_R_abs**2), 'trainable': False},
                'eta_bar': I_ext / (K * V_R_abs**2),
                'e_r': 1.0 + E_r / V_R_abs,
                'v_rest': 1.0,
                'v_peak': 1.0 + V_peak / V_R_abs,
                'v_reset': 1.0 + V_reset / V_R_abs,
            }

        self.tau_pop = self._make_param(params, 'tau_pop')
        self.alpha = self._make_param(params, 'alpha')
        self.a = self._make_param(params, 'a')
        self.b = self._make_param(params, 'b')
        self.w_jump = self._make_param(params, 'w_jump')
        self.Delta_eta = self._make_param(params, 'Delta_eta')
        self.eta_bar = self._make_param(params, 'eta_bar')
        self.e_r = self._make_param(params, 'e_r')
        self.v_rest = self._make_param(params, 'v_rest')
        self.v_peak = self._make_param(params, 'v_peak')
        self.v_reset = self._make_param(params, 'v_reset')

        self.state_size = [
            tf.TensorShape([1, n_units]),
            tf.TensorShape([1, n_units]),
            tf.TensorShape([1, n_units]),
        ]

    def get_initial_state(self, batch_size: int = 1) -> StateList:
        dtype = neuraltide.config.get_dtype()
        v0 = tf.zeros([1, self.n_units], dtype=dtype) + 1.0
        return [
            tf.zeros([1, self.n_units], dtype=dtype),
            v0,
            tf.zeros([1, self.n_units], dtype=dtype),
        ]

    def derivatives(
        self,
        state: StateList,
        total_synaptic_input: Dict[str, TensorType],
    ) -> StateList:
        r, v, w = state
        g_syn_tot = total_synaptic_input['g_syn']
        I_syn = total_synaptic_input['I_syn']

        dtype = neuraltide.config.get_dtype()
        PI = 3.141592653589793

        tau_pop = tf.cast(self.tau_pop, dtype)
        alpha = tf.cast(self.alpha, dtype)
        Delta_eta = tf.cast(self.Delta_eta, dtype)
        a = tf.cast(self.a, dtype)
        b = tf.cast(self.b, dtype)
        w_jump = tf.cast(self.w_jump, dtype)
        eta_bar = tf.cast(self.eta_bar, dtype)
        PI_tensor = tf.constant(PI, dtype=dtype)

        drdt = (Delta_eta / PI_tensor + 2.0 * r * v - (alpha + g_syn_tot) * r) / tau_pop
        dvdt = (v ** 2 - alpha * v - w + eta_bar + I_syn - (PI_tensor * r) ** 2) / tau_pop
        dwdt = (a * (b * v - w) + w_jump * r) / tau_pop

        return [drdt, dvdt, dwdt]

    def get_firing_rate(self, state: StateList) -> TensorType:
        r = state[0]
        return tf.nn.relu(r)

    def observables(self, state: StateList) -> Dict[str, TensorType]:
        r, v, w = state
        return {
            'firing_rate': self.get_firing_rate(state),
            'v_dimensionless': v,
            'w_dimensionless': w,
        }

    @property
    def parameter_spec(self) -> Dict[str, Dict[str, any]]:
        return {
            'tau_pop': {
                'shape': (self.n_units,),
                'trainable': False,
                'constraint': None,
                'units': 'ms',
            },
            'alpha': {
                'shape': (self.n_units,),
                'trainable': False,
                'constraint': None,
                'units': 'dimensionless',
            },
            'a': {
                'shape': (self.n_units,),
                'trainable': False,
                'constraint': None,
                'units': 'dimensionless',
            },
            'b': {
                'shape': (self.n_units,),
                'trainable': False,
                'constraint': None,
                'units': 'dimensionless',
            },
            'w_jump': {
                'shape': (self.n_units,),
                'trainable': False,
                'constraint': None,
                'units': 'dimensionless',
            },
            'Delta_eta': {
                'shape': (self.n_units,),
                'trainable': self.Delta_eta.trainable,
                'constraint': 'MinMaxConstraint' if hasattr(self.Delta_eta, 'constraint') and self.Delta_eta.constraint is not None else None,
                'units': 'dimensionless',
            },
            'eta_bar': {
                'shape': (self.n_units,),
                'trainable': self.eta_bar.trainable,
                'constraint': None,
                'units': 'dimensionless',
            },
        }

    @staticmethod
    def dimensional_to_dimensionless(
        V_R: float,
        V_T: float,
        V_peak: float,
        V_reset: float,
        C: float,
        K: float,
        A: float,
        B: float,
        W_jump: float,
        Delta_I: float,
        I_ext: float,
        E_r: float,
    ) -> dict:
        """
        Переводит размерные параметры нейрона в безразмерные.

        Формулы из таблицы 3 статьи:
            α = 1 + V_T/|V_R|
            a = C·A/(K·|V_R|)
            b = B/(K·|V_R|)
            w_jump = W_jump/(K·|V_R|²)
            Δη = ΔI/(K·|V_R|²)
            η̄ = I_ext/(K·|V_R|²)
            e_r = 1 + E_r/|V_R|
            τ_pop = C/(K·|V_R|)

        Returns:
            dict с ключами: tau_pop, alpha, a, b, w_jump, Delta_eta, eta_bar, e_r
        """
        V_R_abs = abs(V_R)
        return {
            'tau_pop': C / (K * V_R_abs),
            'alpha': 1.0 + V_T / V_R_abs,
            'a': C * A / (K * V_R_abs),
            'b': B / (K * V_R_abs),
            'w_jump': W_jump / (K * V_R_abs**2),
            'Delta_eta': Delta_I / (K * V_R_abs**2),
            'eta_bar': I_ext / (K * V_R_abs**2),
            'e_r': 1.0 + E_r / V_R_abs,
        }

    @staticmethod
    def dimensionless_to_dimensional(
        tau_pop: float,
        alpha: float,
        a: float,
        b: float,
        w_jump: float,
        Delta_eta: float,
        eta_bar: float,
        V_R: float,
        E_r: float,
    ) -> dict:
        """
        Переводит безразмерные параметры обратно в размерные.

        Обратные формулы:
            C = τ_pop·K·|V_R|
            V_T = (α - 1)·|V_R|
            A = a·K·|V_R|/C
            B = b·K·|V_R|
            W_jump = w_jump·K·|V_R|²
            ΔI = Δη·K·|V_R|²
            I_ext = η̄·K·|V_R|²

        Returns:
            dict с ключами: C, V_T, A, B, W_jump, Delta_I, I_ext
        """
        V_R_abs = abs(V_R)
        K_val = None  # Недоопределено без K
        return {
            'V_T': (alpha - 1.0) * V_R_abs,
            'W_jump': w_jump * K_val * V_R_abs**2 if K_val else None,
            'Delta_I': Delta_eta * K_val * V_R_abs**2 if K_val else None,
            'I_ext': eta_bar * K_val * V_R_abs**2 if K_val else None,
        }
