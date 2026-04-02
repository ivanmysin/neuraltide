import tensorflow as tf

import neuraltide
import neuraltide.config
from neuraltide.core.base import PopulationModel
from neuraltide.core.types import TensorType, StateList, Dict


class IzhikevichMeanField(PopulationModel):
    """
    Популяционная модель Ижикевича (Montbrio-Pazo-Roxe).

    Уравнения:
        dr/dt = Δ_η/π + 2*r*v - (α + g_syn_tot)*r
        dv/dt = v² - α*v - w + I_ext + I_syn - (π*r)²
        dw/dt = a*(b*v - w) + w_jump*r

    Состояние: [r, v, w], каждое shape [1, n_units].
    """

    def __init__(self, n_units: int, dt: float, params: dict, **kwargs):
        super().__init__(n_units=n_units, dt=dt, **kwargs)

        self.alpha = self._make_param(params, 'alpha')
        self.a = self._make_param(params, 'a')
        self.b = self._make_param(params, 'b')
        self.w_jump = self._make_param(params, 'w_jump')
        self.dt_nondim = self._make_param(params, 'dt_nondim')
        self.Delta_eta = self._make_param(params, 'Delta_eta')
        self.I_ext = self._make_param(params, 'I_ext')

        self.state_size = [
            tf.TensorShape([1, n_units]),
            tf.TensorShape([1, n_units]),
            tf.TensorShape([1, n_units]),
        ]

    def get_initial_state(self, batch_size: int = 1) -> StateList:
        dtype = neuraltide.config.get_dtype()
        return [
            tf.zeros([1, self.n_units], dtype=dtype),
            tf.zeros([1, self.n_units], dtype=dtype),
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

        PI = neuraltide.config.get_dtype().__class__.PI if hasattr(neuraltide.config.get_dtype().__class__, 'PI') else 3.141592653589793

        alpha = tf.cast(self.alpha, neuraltide.config.get_dtype())
        Delta_eta = tf.cast(self.Delta_eta, neuraltide.config.get_dtype())
        a = tf.cast(self.a, neuraltide.config.get_dtype())
        b = tf.cast(self.b, neuraltide.config.get_dtype())
        w_jump = tf.cast(self.w_jump, neuraltide.config.get_dtype())
        I_ext = tf.cast(self.I_ext, neuraltide.config.get_dtype())
        PI_tensor = tf.constant(PI, dtype=neuraltide.config.get_dtype())

        drdt = Delta_eta / PI_tensor + 2.0 * r * v - (alpha + g_syn_tot) * r
        dvdt = v ** 2 - alpha * v - w + I_ext + I_syn - (PI_tensor * r) ** 2
        dwdt = a * (b * v - w) + w_jump * r

        return [drdt, dvdt, dwdt]

    def get_firing_rate(self, state: StateList) -> TensorType:
        r = state[0]
        dt_nondim = tf.cast(self.dt_nondim, neuraltide.config.get_dtype())
        dt = tf.constant(self.dt, dtype=neuraltide.config.get_dtype())
        return tf.nn.relu(r) * dt_nondim / (dt * 1e-3)

    def observables(self, state: StateList) -> Dict[str, TensorType]:
        r, v, w = state
        return {
            'firing_rate': self.get_firing_rate(state),
            'v_mean': v,
            'w_mean': w,
        }

    @property
    def parameter_spec(self) -> Dict[str, Dict[str, any]]:
        return {
            'alpha': {
                'shape': (self.n_units,),
                'trainable': False,
                'constraint': None,
                'units': 'ms^-1',
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
                'units': 'mV/ms',
            },
            'dt_nondim': {
                'shape': (self.n_units,),
                'trainable': False,
                'constraint': None,
                'units': 'ms',
            },
            'Delta_eta': {
                'shape': (self.n_units,),
                'trainable': self.Delta_eta.trainable,
                'constraint': 'MinMaxConstraint' if hasattr(self.Delta_eta, 'constraint') and self.Delta_eta.constraint is not None else None,
                'units': 'Hz',
            },
            'I_ext': {
                'shape': (self.n_units,),
                'trainable': self.I_ext.trainable,
                'constraint': None,
                'units': 'mV/ms',
            },
        }
