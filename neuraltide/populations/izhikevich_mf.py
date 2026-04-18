"""
Izhikevich mean-field population model.

Implements the Montbrio-Pazo-Roxin (next-generation mean-field) model:
    tau_pop * d_nu/dt = Delta_I/pi + 2*nu*<v> - (alpha + g_syn_tot)*nu
    tau_pop * d<v>/dt = <v>^2 - alpha*<v> - <w> + I_ext + I_syn - (pi*nu)^2
    tau_pop * d<w>/dt = a*(b*<v> - <w>) + w_jump*nu

State: [r, v, w], each shape [1, n_units].
r corresponds to nu (dimensionless firing rate),
v corresponds to <v> (dimensionless mean membrane potential, relative to V_rest),
w corresponds to <w> (dimensionless mean adaptation current).

Note on dimensionless variables:
    All state variables (r, v, w) and parameters (tau_pop, alpha, etc.) are
    dimensionless. The membrane potential <v> is measured relative to V_rest:
        <v>_dimensionless = (<v>_mV - V_rest) / |V_rest|
    Therefore, the dimensionless rest state is v=0 (not v=1).
"""
import numpy as np
import tensorflow as tf
from typing import Any, Dict, List, Optional, Union

import neuraltide
import neuraltide.config
from neuraltide.core.base import PopulationModel
from neuraltide.core.types import TensorType, StateList


class IzhikevichMeanField(PopulationModel):
    """
    Mean-field Izhikevich population model.

    Supports two modes of operation:
    1. Dimensionless params: pass a `params` dict with dimensionless parameters.
       All values are per-unit (list of n_units) or scalar (broadcasts to n_units).
       All parameters non-trainable by default; user can mark any as trainable.
       n_units is inferred from the length of parameter arrays (must be consistent).

    2. Dimensional params: pass dimensional neuron parameters (V_rest, V_T, etc.).
       They are converted to dimensionless internally using the formulas
       from Chen & Campbell 2022. Dimensional parameters can be scalars
       (broadcasts to n_units) or vectors (one per unit).

    Parameters (dimensionless, via params dict):
        tau_pop: Population time constant [ms]. Typical: [1.0, ...]
        alpha: Threshold parameter [dimensionless]. Typical: [0.5, ...]
        a: Adaptation rate [dimensionless]. Typical: [0.02, ...]
        b: Adaptation coupling [dimensionless]. Typical: [0.2, ...]
        w_jump: Adaptation jump [dimensionless]. Typical: [0.1, ...]
        Delta_I: Lorentzian current spread [dimensionless]. Typical: [0.5, ...]
        I_ext: External current [dimensionless]. Typical: [1.0, ...]

    State variables:
        The state variables (r, v, w) are all dimensionless:
        - r: dimensionless firing rate
        - v: dimensionless mean membrane potential, measured relative to V_rest.
             In dimensional terms: v_dim = (<v>_mV - V_rest) / |V_rest|
             Therefore, the rest state corresponds to v = 0 (not v = 1).
        - w: dimensionless mean adaptation current

    Initial state:
        get_initial_state() returns [r=0, v=0, w=0], corresponding to the
        rest state where <v> = V_rest in dimensional units.

    Example (dimensionless, recommended):
        pop = IzhikevichMeanField(
            dt=0.5,
            params={
                'tau_pop':   {'value': [1.0, 1.0, 1.0, 1.0], 'trainable': False},
                'alpha':     {'value': [0.5, 0.5, 0.5, 0.5], 'trainable': False},
                'a':         {'value': [0.02, 0.02, 0.02, 0.02], 'trainable': False},
                'b':         {'value': [0.2, 0.2, 0.2, 0.2], 'trainable': False},
                'w_jump':    {'value': [0.1, 0.1, 0.1, 0.1], 'trainable': False},
                'Delta_I':   {'value': [0.5, 0.5, 0.5, 0.5], 'trainable': True,
                               'min': 0.01, 'max': 2.0},
                'I_ext':     {'value': [1.0, 1.0, 1.0, 1.0], 'trainable': True},
            }
        )

    Example (dimensional, for backward compatibility):
        pop = IzhikevichMeanField(
            dt=0.5,
            params={
                'V_rest': -57.6, 'V_T': -35.5, 'V_peak': 21.7, 'V_reset': -48.7,
                'Cm': 114.0, 'K': 1.194, 'A': 0.0046, 'B': 0.2157, 'W_jump': 2.0,
                'Delta_I': 20.0, 'I_ext': 120.0
            }
        )
    """

    def __init__(
        self,
        dt: float,
        params: Optional[Dict[str, Any]] = None,
        name: str = "izhikevich_mf",
        **kwargs,
    ):
        """
        Args:
            dt: Integration time step [ms].
            params: Dictionary of parameters. Each parameter can be:
                - A scalar (broadcasts to n_units)
                - A list of n_units values (one per unit)
                - A dict with 'value', optional 'trainable' (default False),
                  and optional 'min'/'max' constraints.
                All parameters must have consistent dimensions: either 1 or n_units.
                n_units is inferred from the maximum dimension of all parameters.
            name: Layer name.
        """
        if params is None:
            raise ValueError("params cannot be None")

        self.name = name
        self._use_dimensional = 'Cm' in params

        self._infer_n_units_from_raw_params(params)

        if self._use_dimensional:
            self._params = self._build_params_from_dimensional(params)
        else:
            self._params = params

        self._validate_params()
        self._validate_param_dimensions()

        super().__init__(n_units=self.n_units, dt=dt, name=name, **kwargs)

        self.tau_pop = self._make_param(self._params, 'tau_pop')
        self.alpha = self._make_param(self._params, 'alpha')
        self.a = self._make_param(self._params, 'a')
        self.b = self._make_param(self._params, 'b')
        self.w_jump = self._make_param(self._params, 'w_jump')
        self.Delta_I = self._make_param(self._params, 'Delta_I')
        self.I_ext = self._make_param(self._params, 'I_ext')
        self.v_max = tf.constant(10.0, dtype=neuraltide.config.get_dtype())

        dtype = neuraltide.config.get_dtype()
        self.PI = tf.constant(3.141592653589793, dtype=dtype)

        self.state_size = [
            tf.TensorShape([1, self.n_units]),
            tf.TensorShape([1, self.n_units]),
            tf.TensorShape([1, self.n_units]),
        ]

    def _infer_n_units_from_raw_params(self, params: Dict[str, Any]) -> None:
        """Infer n_units from raw input params before any conversion."""
        max_len = 1
        for key, spec in params.items():
            if isinstance(spec, dict):
                value = spec.get('value', None)
            else:
                value = spec

            if value is None:
                continue

            if isinstance(value, (list, tuple)):
                max_len = max(max_len, len(value))
            elif isinstance(value, np.ndarray):
                if value.ndim == 1:
                    max_len = max(max_len, len(value))

        self.n_units = max_len

    def _infer_n_units(self) -> None:
        """Infer n_units from the dimensions of parameters."""
        max_len = 1
        for key, spec in self._params.items():
            if isinstance(spec, dict):
                value = spec.get('value', None)
            else:
                value = spec

            if value is None:
                continue

            if isinstance(value, (list, tuple)):
                max_len = max(max_len, len(value))
            elif isinstance(value, np.ndarray):
                if value.ndim == 1:
                    max_len = max(max_len, len(value))

        self.n_units = max_len

    def _validate_param_dimensions(self) -> None:
        """Validate that all parameters have consistent dimensions (1 or n_units)."""
        for key, spec in self._params.items():
            if isinstance(spec, dict):
                value = spec.get('value', None)
            else:
                value = spec

            if value is None:
                continue

            if isinstance(value, (list, tuple)):
                if len(value) != 1 and len(value) != self.n_units:
                    raise ValueError(
                        f"IzhikevichMeanField '{self.name}': parameter '{key}' "
                        f"has length {len(value)}, expected 1 or {self.n_units}."
                    )
            elif isinstance(value, np.ndarray):
                if value.ndim == 1 and len(value) != 1 and len(value) != self.n_units:
                    raise ValueError(
                        f"IzhikevichMeanField '{self.name}': parameter '{key}' "
                        f"has length {len(value)}, expected 1 or {self.n_units}."
                    )

    def _validate_params(self) -> None:
        """Validate that all required parameters are present."""
        required = ['tau_pop', 'alpha', 'a', 'b', 'w_jump', 'Delta_I', 'I_ext']
        for name in required:
            if name not in self._params:
                raise ValueError(
                    f"IzhikevichMeanField '{self.name}': "
                    f"required parameter '{name}' not found in params."
                )

    @staticmethod
    def _compute_dimensionless_from_dimensional(
        params: Dict[str, Any],
        n_units: int,
    ) -> Dict[str, List[float]]:
        """
        Compute dimensionless parameters from dimensional ones (pure Python).

        Handles scalar/vector forms with broadcasting to n_units.
        V_peak and V_reset are accepted but ignored (not used in mean-field model).

        Args:
            params: dict with keys V_rest, V_T, Cm, K, A, B, W_jump, Delta_I, I_ext
            n_units: number of units for broadcasting

        Returns:
            dict with keys tau_pop, alpha, a, b, w_jump, Delta_I, I_ext
            Each value is a list of length n_units.
        """
        def process_param(value, param_name):
            """Convert scalar/list/np.ndarray to list of length n_units."""
            if isinstance(value, (list, tuple)):
                if len(value) == n_units:
                    return [float(v) for v in value]
                elif len(value) == 1:
                    return [float(value[0]) for _ in range(n_units)]
                else:
                    raise ValueError(
                        f"Parameter '{param_name}' has length {len(value)}, "
                        f"expected 1 or {n_units}"
                    )
            elif isinstance(value, np.ndarray):
                if value.ndim == 0:
                    return [float(value.item()) for _ in range(n_units)]
                elif len(value) == n_units:
                    return [float(v) for v in value]
                elif len(value) == 1:
                    return [float(value[0]) for _ in range(n_units)]
                else:
                    raise ValueError(
                        f"Parameter '{param_name}' has length {len(value)}, "
                        f"expected 1 or {n_units}"
                    )
            else:
                return [float(value) for _ in range(n_units)]

        V_rest = process_param(params['V_rest'], 'V_rest')
        V_T = process_param(params['V_T'], 'V_T')
        Cm = process_param(params['Cm'], 'Cm')
        K = process_param(params['K'], 'K')
        A = process_param(params['A'], 'A')
        B = process_param(params['B'], 'B')
        W_jump = process_param(params['W_jump'], 'W_jump')
        Delta_I = process_param(params['Delta_I'], 'Delta_I')
        I_ext = process_param(params['I_ext'], 'I_ext')

        tau_pop = [Cm[i] / (K[i] * abs(V_rest[i])) for i in range(n_units)]
        alpha = [1.0 + V_T[i] / abs(V_rest[i]) for i in range(n_units)]
        a = [Cm[i] * A[i] / (K[i] * abs(V_rest[i])) for i in range(n_units)]
        b = [B[i] / (K[i] * abs(V_rest[i])) for i in range(n_units)]
        w_jump = [W_jump[i] / (K[i] * abs(V_rest[i])**2) for i in range(n_units)]
        Delta_I_dimless = [Delta_I[i] / (K[i] * abs(V_rest[i])**2) for i in range(n_units)]
        I_ext_dimless = [I_ext[i] / (K[i] * abs(V_rest[i])**2) for i in range(n_units)]

        return {
            'tau_pop': tau_pop,
            'alpha': alpha,
            'a': a,
            'b': b,
            'w_jump': w_jump,
            'Delta_I': Delta_I_dimless,
            'I_ext': I_ext_dimless,
        }

    def _build_params_from_dimensional(
        self, params,
    ) -> Dict[str, Any]:
        """Convert dimensional parameters to dimensionless."""
        missing_keys = [key for key in ['V_rest', 'V_T', 'Cm', 'K', 'A', 'B', 'W_jump', 'Delta_I', 'I_ext'] if key not in params.keys()]
        if missing_keys:
            raise ValueError(
                f"IzhikevichMeanField '{self.name}': missing required dimensional parameters: {missing_keys}"
            )

        dimless = self._compute_dimensionless_from_dimensional(params, self.n_units)

        dtype = neuraltide.config.get_dtype()
        result = {}
        for key, value in dimless.items():
            result[key] = tf.constant(value, dtype=dtype)

        return result

    def _to_array(self, value: Union[float, List[float]]) -> tf.Tensor:
        """Convert scalar or list to tf.Tensor, broadcasting to n_units."""
        dtype = neuraltide.config.get_dtype()
        if isinstance(value, (int, float)):
            return tf.fill([self.n_units], tf.constant(float(value), dtype=dtype))
        elif isinstance(value, (list, tuple)):
            arr = tf.constant(value, dtype=dtype)
            if int(arr.shape[0]) == self.n_units:
                return arr
            elif int(arr.shape[0]) == 1:
                return tf.broadcast_to(arr, [self.n_units])
            else:
                raise ValueError(
                    f"IzhikevichMeanField '{self.name}': parameter length {int(arr.shape[0])} "
                    f"does not match n_units={self.n_units}."
                )
        elif isinstance(value, np.ndarray):
            if value.ndim == 0:
                return tf.fill([self.n_units], tf.constant(float(value), dtype=dtype))
            elif int(value.shape[0]) == self.n_units:
                return tf.constant(value, dtype=dtype)
            elif int(value.shape[0]) == 1:
                return tf.broadcast_to(tf.constant(value, dtype=dtype), [self.n_units])
            else:
                raise ValueError(
                    f"IzhikevichMeanField '{self.name}': parameter length {int(value.shape[0])} "
                    f"does not match n_units={self.n_units}."
                )
        else:
            raise TypeError(
                f"IzhikevichMeanField '{self.name}': expected scalar or list, "
                f"got {type(value).__name__}."
            )

    def get_initial_state(self, batch_size: int = 1) -> StateList:
        """
        Return initial state: r=0, v=0 (rest), w=0.

        The rest state is v=0 because <v> is dimensionless and measured
        relative to V_rest. In dimensional terms, v=0 corresponds to
        <v> = V_rest (the resting potential).
        """
        dtype = neuraltide.config.get_dtype()
        v0 = tf.zeros([1, self.n_units], dtype=dtype)
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
        """
        Compute derivatives of the mean-field Izhikevich equations.

        All inputs are vectorized with shape [1, n_units].
        """
        r = state[0]
        v = state[1]
        w = state[2]

        g_syn_tot = total_synaptic_input['g_syn']
        I_syn = total_synaptic_input['I_syn']

        drdt = (self.Delta_I / self.PI + 2.0 * r * v - (self.alpha + g_syn_tot) * r) / self.tau_pop
        dvdt = (v**2 / (1 + (v/self.v_max)**2) - self.alpha * v - w + self.I_ext + I_syn - (self.PI * r) ** 2) / self.tau_pop
        dwdt = (self.a * (self.b * v - w) + self.w_jump * r) / self.tau_pop

        return [drdt, dvdt, dwdt]

    def get_firing_rate(self, state: StateList) -> TensorType:
        """
        Extract firing rate from state.

        Returns the dimensionless rate r. The caller should scale to Hz
        if needed using: rate_Hz = r / (dt * 1e-3) where dt is in ms.
        """
        r = state[0]
        return tf.nn.relu(r)

    def observables(self, state: StateList) -> Dict[str, TensorType]:
        """
        Return observable variables.

        Includes:
            - firing_rate: dimensionless (should be scaled by 1/(dt*1e-3) for Hz)
            - v_mean: dimensionless mean membrane potential (used by NMDASynapse)
            - w_mean: dimensionless mean adaptation current
        """
        r, v, w = state
        return {
            'firing_rate': self.get_firing_rate(state),
            'v_mean': v,
            'w_mean': w,
        }

    @property
    def parameter_spec(self) -> Dict[str, Dict[str, Any]]:
        """Specification of all parameters for summary and serialization."""
        return {
            'tau_pop': {
                'shape': (self.n_units,),
                'trainable': self.tau_pop.trainable,
                'constraint': self._get_constraint_name(self.tau_pop),
                'units': 'ms',
            },
            'alpha': {
                'shape': (self.n_units,),
                'trainable': self.alpha.trainable,
                'constraint': self._get_constraint_name(self.alpha),
                'units': 'dimensionless',
            },
            'a': {
                'shape': (self.n_units,),
                'trainable': self.a.trainable,
                'constraint': self._get_constraint_name(self.a),
                'units': 'dimensionless',
            },
            'b': {
                'shape': (self.n_units,),
                'trainable': self.b.trainable,
                'constraint': self._get_constraint_name(self.b),
                'units': 'dimensionless',
            },
            'w_jump': {
                'shape': (self.n_units,),
                'trainable': self.w_jump.trainable,
                'constraint': self._get_constraint_name(self.w_jump),
                'units': 'dimensionless',
            },
            'Delta_I': {
                'shape': (self.n_units,),
                'trainable': self.Delta_I.trainable,
                'constraint': self._get_constraint_name(self.Delta_I),
                'units': 'dimensionless',
            },
            'I_ext': {
                'shape': (self.n_units,),
                'trainable': self.I_ext.trainable,
                'constraint': self._get_constraint_name(self.I_ext),
                'units': 'dimensionless',
            },
        }

    def _get_constraint_name(self, var: tf.Variable) -> Optional[str]:
        """Get the name of the constraint applied to a variable."""
        if var.constraint is not None:
            return var.constraint.__class__.__name__
        return None

    @staticmethod
    def dimensionless_to_dimensional(
        tau_pop: float,
        alpha: float,
        a: float,
        b: float,
        w_jump: float,
        Delta_I: float,
        I_ext: float,
        V_rest: float,
        K: float,
    ) -> Dict[str, float]:
        """
        Convert dimensionless parameters back to dimensional.

        Inverse formulas:
            Cm = tau_pop * K * |V_rest|
            V_T = (alpha - 1) * |V_rest|
            A = a * K * |V_rest| / Cm
            B = b * K * |V_rest|
            W_jump = w_jump * K * |V_rest|^2
            Delta_I = Delta_I * K * |V_rest|^2
            I_ext = I_ext * K * |V_rest|^2

        Args:
            tau_pop: Population time constant [ms]
            alpha: Threshold parameter [dimensionless]
            a: Adaptation rate [dimensionless]
            b: Adaptation coupling [dimensionless]
            w_jump: Adaptation jump [dimensionless]
            Delta_I: Lorentzian spread [dimensionless]
            I_ext: External current [dimensionless]
            V_rest: Resting potential [mV]
            K: Scaling parameter [nS/mV]

        Returns:
            Dictionary with dimensional parameters:
            V_T, Cm, A, B, W_jump, Delta_I, I_ext
        """
        V_rest_abs = abs(V_rest)
        Cm = tau_pop * K * V_rest_abs
        return {
            'V_T': (alpha - 1.0) * V_rest_abs,
            'Cm': Cm,
            'A': a * K * V_rest_abs / Cm,
            'B': b * K * V_rest_abs,
            'W_jump': w_jump * K * V_rest_abs**2,
            'Delta_I': Delta_I * K * V_rest_abs**2,
            'I_ext': I_ext * K * V_rest_abs**2,
        }
