import tensorflow as tf

import neuraltide
import neuraltide.config
from neuraltide.core.base import PopulationModel
from neuraltide.core.types import TensorType, StateList, Dict


class WilsonCowan(PopulationModel):
    """
    Модель Wilson-Cowan.

    Уравнения:
        τ_E * dE/dt = -E + F_E(w_EE*E - w_IE*I + I_ext,E + I_syn,E)
        τ_I * dI/dt = -I + F_I(w_EI*E - w_II*I + I_ext,I + I_syn,I)
        F(x) = 1 / (1 + exp(-a_coeff*(x - theta)))

    Состояние: [E, I], каждое shape [1, n_units].
    """

    def __init__(self, n_units: int, dt: float, params: dict, **kwargs):
        super().__init__(n_units=n_units, dt=dt, **kwargs)

        self.tau_E = self._make_param(params, 'tau_E')
        self.tau_I = self._make_param(params, 'tau_I')
        self.a_E = self._make_param(params, 'a_E')
        self.a_I = self._make_param(params, 'a_I')
        self.theta_E = self._make_param(params, 'theta_E')
        self.theta_I = self._make_param(params, 'theta_I')
        self.w_EE = self._make_param(params, 'w_EE')
        self.w_IE = self._make_param(params, 'w_IE')
        self.w_EI = self._make_param(params, 'w_EI')
        self.w_II = self._make_param(params, 'w_II')
        self.I_ext_E = self._make_param(params, 'I_ext_E')
        self.I_ext_I = self._make_param(params, 'I_ext_I')
        self.max_rate = self._make_param(params, 'max_rate')

        self.state_size = [
            tf.TensorShape([1, n_units]),
            tf.TensorShape([1, n_units]),
        ]

    def _sigmoid(self, x: TensorType, a: TensorType, theta: TensorType) -> TensorType:
        return 1.0 / (1.0 + tf.exp(-a * (x - theta)))

    def get_initial_state(self, batch_size: int = 1) -> StateList:
        dtype = neuraltide.config.get_dtype()
        return [
            tf.zeros([1, self.n_units], dtype=dtype),
            tf.zeros([1, self.n_units], dtype=dtype),
        ]

    def derivatives(
        self,
        state: StateList,
        total_synaptic_input: Dict[str, TensorType],
    ) -> StateList:
        E, I = state
        I_syn = total_synaptic_input['I_syn']

        dtype = neuraltide.config.get_dtype()
        tau_E = tf.cast(self.tau_E, dtype)
        tau_I = tf.cast(self.tau_I, dtype)
        a_E = tf.cast(self.a_E, dtype)
        a_I = tf.cast(self.a_I, dtype)
        theta_E = tf.cast(self.theta_E, dtype)
        theta_I = tf.cast(self.theta_I, dtype)
        w_EE = tf.cast(self.w_EE, dtype)
        w_IE = tf.cast(self.w_IE, dtype)
        w_EI = tf.cast(self.w_EI, dtype)
        w_II = tf.cast(self.w_II, dtype)
        I_ext_E = tf.cast(self.I_ext_E, dtype)
        I_ext_I = tf.cast(self.I_ext_I, dtype)

        x_E = w_EE * E - w_IE * I + I_ext_E + I_syn
        x_I = w_EI * E - w_II * I + I_ext_I

        F_E = self._sigmoid(x_E, a_E, theta_E)
        F_I = self._sigmoid(x_I, a_I, theta_I)

        dEdt = (-E + F_E) / tau_E
        dIdt = (-I + F_I) / tau_I

        return [dEdt, dIdt]

    def get_firing_rate(self, state: StateList) -> TensorType:
        E = state[0]
        max_rate = tf.cast(self.max_rate, neuraltide.config.get_dtype())
        return tf.nn.relu(E) * max_rate

    def observables(self, state: StateList) -> Dict[str, TensorType]:
        return {'firing_rate': self.get_firing_rate(state)}

    @property
    def parameter_spec(self) -> Dict[str, Dict[str, any]]:
        return {
            'tau_E': {
                'shape': (self.n_units,),
                'trainable': False,
                'constraint': None,
                'units': 'ms',
            },
            'tau_I': {
                'shape': (self.n_units,),
                'trainable': False,
                'constraint': None,
                'units': 'ms',
            },
            'a_E': {
                'shape': (self.n_units,),
                'trainable': False,
                'constraint': None,
                'units': 'dimensionless',
            },
            'a_I': {
                'shape': (self.n_units,),
                'trainable': False,
                'constraint': None,
                'units': 'dimensionless',
            },
            'theta_E': {
                'shape': (self.n_units,),
                'trainable': False,
                'constraint': None,
                'units': 'mV',
            },
            'theta_I': {
                'shape': (self.n_units,),
                'trainable': False,
                'constraint': None,
                'units': 'mV',
            },
            'w_EE': {
                'shape': (self.n_units,),
                'trainable': False,
                'constraint': None,
                'units': 'dimensionless',
            },
            'w_IE': {
                'shape': (self.n_units,),
                'trainable': False,
                'constraint': None,
                'units': 'dimensionless',
            },
            'w_EI': {
                'shape': (self.n_units,),
                'trainable': False,
                'constraint': None,
                'units': 'dimensionless',
            },
            'w_II': {
                'shape': (self.n_units,),
                'trainable': False,
                'constraint': None,
                'units': 'dimensionless',
            },
            'I_ext_E': {
                'shape': (self.n_units,),
                'trainable': False,
                'constraint': None,
                'units': 'mV',
            },
            'I_ext_I': {
                'shape': (self.n_units,),
                'trainable': False,
                'constraint': None,
                'units': 'mV',
            },
            'max_rate': {
                'shape': (self.n_units,),
                'trainable': False,
                'constraint': None,
                'units': 'Hz',
            },
        }
