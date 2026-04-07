"""
Izhikevich mean-field population model.

Implements the Montbrio-Pazo-Roxin (next-generation mean-field) model:
    tau_pop * d_nu/dt = Delta_I/pi + 2*nu*<v> - (alpha + g_syn_tot)*nu
    tau_pop * d<v>/dt = <v>^2 - alpha*<v> - <w> + I_ext + I_syn - (pi*nu)^2
    tau_pop * d<w>/dt = a*(b*<v> - <w>) + w_jump*nu

State: [r, v, w], each shape [1, n_units].
r corresponds to nu (dimensionless firing rate),
v corresponds to <v> (dimensionless mean membrane potential),
w corresponds to <w> (dimensionless mean adaptation current).
"""
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

    2. Dimensional params: pass dimensional neuron parameters (V_R, V_T, etc.).
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

    Example (dimensionless, recommended):
        pop = IzhikevichMeanField(
            n_units=4, dt=0.5,
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
            n_units=1, dt=0.5,
            V_R=-57.6, V_T=-35.5, V_peak=21.7, V_reset=-48.7,
            C=114.0, K=1.194, A=0.0046, B=0.2157, W_jump=2.0,
            Delta_I=20.0, I_ext=120.0
        )
    """

    def __init__(
        self,
        n_units: int,
        dt: float,
        params: Optional[Dict[str, Any]] = None,
        V_R: Optional[Union[float, List[float]]] = None,
        V_T: Optional[Union[float, List[float]]] = None,
        V_peak: Optional[Union[float, List[float]]] = None,
        V_reset: Optional[Union[float, List[float]]] = None,
        C: Optional[Union[float, List[float]]] = None,
        K: Optional[Union[float, List[float]]] = None,
        A: Optional[Union[float, List[float]]] = None,
        B: Optional[Union[float, List[float]]] = None,
        W_jump: Optional[Union[float, List[float]]] = None,
        Delta_I: Optional[Union[float, List[float]]] = None,
        I_ext: Optional[Union[float, List[float]]] = None,
        name: str = "izhikevich_mf",
        **kwargs,
    ):
        """
        Args:
            n_units: Number of independent populations.
            dt: Integration time step [ms].
            params: Dictionary of dimensionless parameters. Each parameter can be:
                - A scalar (broadcasts to n_units)
                - A list of n_units values (one per unit)
                - A dict with 'value', optional 'trainable' (default False),
                  and optional 'min'/'max' constraints.
            V_R, V_T, V_peak, V_reset, C, K, A, B, W_jump, Delta_I, I_ext:
                Dimensional neuron parameters (see convert_dimensional_to_dimensionless).
                Can be scalars (broadcasts to n_units) or vectors (one per unit).
                Used only if params is None. For backward compatibility.
            name: Layer name.
        """
        super().__init__(n_units=n_units, dt=dt, name=name, **kwargs)

        if params is not None:
            self._use_dimensional = False
            self._params = params
        else:
            self._use_dimensional = True
            self._params = self._build_params_from_dimensional(
                V_R, V_T, V_peak, V_reset, C, K, A, B, W_jump, Delta_I, I_ext
            )

        self._validate_params()

        self.tau_pop = self._make_param(self._params, 'tau_pop')
        self.alpha = self._make_param(self._params, 'alpha')
        self.a = self._make_param(self._params, 'a')
        self.b = self._make_param(self._params, 'b')
        self.w_jump = self._make_param(self._params, 'w_jump')
        self.Delta_I = self._make_param(self._params, 'Delta_I')
        self.I_ext = self._make_param(self._params, 'I_ext')

        self.state_size = [
            tf.TensorShape([1, n_units]),
            tf.TensorShape([1, n_units]),
            tf.TensorShape([1, n_units]),
        ]

    def _validate_params(self) -> None:
        """Validate that all required parameters are present."""
        required = ['tau_pop', 'alpha', 'a', 'b', 'w_jump', 'Delta_I', 'I_ext']
        for name in required:
            if name not in self._params:
                raise ValueError(
                    f"IzhikevichMeanField '{self.name}': "
                    f"required parameter '{name}' not found in params."
                )

    def _build_params_from_dimensional(
        self,
        V_R: Optional[Union[float, List[float]]],
        V_T: Optional[Union[float, List[float]]],
        V_peak: Optional[Union[float, List[float]]],
        V_reset: Optional[Union[float, List[float]]],
        C: Optional[Union[float, List[float]]],
        K: Optional[Union[float, List[float]]],
        A: Optional[Union[float, List[float]]],
        B: Optional[Union[float, List[float]]],
        W_jump: Optional[Union[float, List[float]]],
        Delta_I: Optional[Union[float, List[float]]],
        I_ext: Optional[Union[float, List[float]]],
    ) -> Dict[str, Any]:
        """Convert dimensional parameters to dimensionless."""
        if None in [V_R, V_T, V_peak, V_reset, C, K, A, B, W_jump, Delta_I, I_ext]:
            raise ValueError(
                f"IzhikevichMeanField '{self.name}': when using dimensional parameters, "
                f"all of V_R, V_T, V_peak, V_reset, C, K, A, B, W_jump, Delta_I, I_ext "
                f"must be provided."
            )

        dtype = neuraltide.config.get_dtype()

        V_R_arr = self._to_array(V_R)
        V_T_arr = self._to_array(V_T)
        V_peak_arr = self._to_array(V_peak)
        V_reset_arr = self._to_array(V_reset)
        C_arr = self._to_array(C)
        K_arr = self._to_array(K)
        A_arr = self._to_array(A)
        B_arr = self._to_array(B)
        W_jump_arr = self._to_array(W_jump)
        Delta_I_arr = self._to_array(Delta_I)
        I_ext_arr = self._to_array(I_ext)

        V_R_arr = tf.cast(V_R_arr, dtype)
        V_T_arr = tf.cast(V_T_arr, dtype)
        V_peak_arr = tf.cast(V_peak_arr, dtype)
        V_reset_arr = tf.cast(V_reset_arr, dtype)
        C_arr = tf.cast(C_arr, dtype)
        K_arr = tf.cast(K_arr, dtype)
        A_arr = tf.cast(A_arr, dtype)
        B_arr = tf.cast(B_arr, dtype)
        W_jump_arr = tf.cast(W_jump_arr, dtype)
        Delta_I_arr = tf.cast(Delta_I_arr, dtype)
        I_ext_arr = tf.cast(I_ext_arr, dtype)

        V_R_abs = tf.abs(V_R_arr)
        tau_pop = C_arr / (K_arr * V_R_abs)
        alpha = 1.0 + V_T_arr / V_R_abs
        a = C_arr * A_arr / (K_arr * V_R_abs)
        b = B_arr / (K_arr * V_R_abs)
        w_jump = W_jump_arr / (K_arr * V_R_abs**2)
        Delta_I_dimless = Delta_I_arr / (K_arr * V_R_abs**2)
        I_ext_dimless = I_ext_arr / (K_arr * V_R_abs**2)

        return {
            'tau_pop': tau_pop,
            'alpha': alpha,
            'a': a,
            'b': b,
            'w_jump': w_jump,
            'Delta_I': Delta_I_dimless,
            'I_ext': I_ext_dimless,
        }

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
        else:
            raise TypeError(
                f"IzhikevichMeanField '{self.name}': expected scalar or list, "
                f"got {type(value).__name__}."
            )

    def get_initial_state(self, batch_size: int = 1) -> StateList:
        """Return initial state: r=0, v=1 (rest), w=0."""
        dtype = neuraltide.config.get_dtype()
        v0 = tf.ones([1, self.n_units], dtype=dtype)
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

        dtype = neuraltide.config.get_dtype()
        PI = tf.constant(3.141592653589793, dtype=dtype)

        tau_pop = tf.cast(self.tau_pop, dtype)
        alpha = tf.cast(self.alpha, dtype)
        Delta_I = tf.cast(self.Delta_I, dtype)
        a = tf.cast(self.a, dtype)
        b = tf.cast(self.b, dtype)
        w_jump = tf.cast(self.w_jump, dtype)
        I_ext = tf.cast(self.I_ext, dtype)

        drdt = (Delta_I / PI + 2.0 * r * v - (alpha + g_syn_tot) * r) / tau_pop
        dvdt = (v ** 2 - alpha * v - w + I_ext + I_syn - (PI * r) ** 2) / tau_pop
        dwdt = (a * (b * v - w) + w_jump * r) / tau_pop

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
    def dimensional_to_dimensionless(
        V_R: Union[float, List[float]],
        V_T: Union[float, List[float]],
        V_peak: Union[float, List[float]],
        V_reset: Union[float, List[float]],
        C: Union[float, List[float]],
        K: Union[float, List[float]],
        A: Union[float, List[float]],
        B: Union[float, List[float]],
        W_jump: Union[float, List[float]],
        Delta_I: Union[float, List[float]],
        I_ext: Union[float, List[float]],
    ) -> Dict[str, Union[float, List[float]]]:
        """
        Convert dimensional neuron parameters to dimensionless.

        Formulas from Chen & Campbell 2022:
            tau_pop = C / (K * |V_R|)
            alpha = 1 + V_T / |V_R|
            a = C * A / (K * |V_R|)
            b = B / (K * |V_R|)
            w_jump = W_jump / (K * |V_R|^2)
            Delta_I_dimless = Delta_I / (K * |V_R|^2)
            I_ext_dimless = I_ext / (K * |V_R|^2)

        Args:
            V_R: Resting potential [mV]
            V_T: Threshold potential [mV]
            V_peak: Peak potential [mV]
            V_reset: Reset potential [mV]
            C: Membrane capacitance [pF]
            K: Scaling parameter [nS/mV]
            A: Adaptation rate [1/ms]
            B: Adaptation coupling [nS]
            W_jump: Adaptation jump [pA]
            Delta_I: Lorentzian current spread [pA]
            I_ext: External current [pA]

        Returns:
            Dictionary with dimensionless parameters:
            tau_pop, alpha, a, b, w_jump, Delta_I, I_ext
        """
        def to_float(x):
            return float(x) if not isinstance(x, list) else x

        V_R = to_float(V_R)
        V_T = to_float(V_T)
        V_peak = to_float(V_peak)
        V_reset = to_float(V_reset)
        C = to_float(C)
        K = to_float(K)
        A = to_float(A)
        B = to_float(B)
        W_jump = to_float(W_jump)
        Delta_I = to_float(Delta_I)
        I_ext = to_float(I_ext)

        is_vector = isinstance(V_R, list)

        if is_vector:
            V_R_arr = V_R
            V_T_arr = V_T
            V_peak_arr = V_peak
            V_reset_arr = V_reset
            C_arr = C
            K_arr = K
            A_arr = A
            B_arr = B
            W_jump_arr = W_jump
            Delta_I_arr = Delta_I
            I_ext_arr = I_ext

            tau_pop = [C_arr[i] / (K_arr[i] * abs(V_R_arr[i])) for i in range(len(V_R_arr))]
            alpha = [1.0 + V_T_arr[i] / abs(V_R_arr[i]) for i in range(len(V_R_arr))]
            a = [C_arr[i] * A_arr[i] / (K_arr[i] * abs(V_R_arr[i])) for i in range(len(V_R_arr))]
            b = [B_arr[i] / (K_arr[i] * abs(V_R_arr[i])) for i in range(len(V_R_arr))]
            w_jump = [W_jump_arr[i] / (K_arr[i] * abs(V_R_arr[i])**2) for i in range(len(V_R_arr))]
            Delta_I_dimless = [Delta_I_arr[i] / (K_arr[i] * abs(V_R_arr[i])**2) for i in range(len(V_R_arr))]
            I_ext_dimless = [I_ext_arr[i] / (K_arr[i] * abs(V_R_arr[i])**2) for i in range(len(V_R_arr))]
        else:
            V_R_abs = abs(V_R)
            tau_pop = C / (K * V_R_abs)
            alpha = 1.0 + V_T / V_R_abs
            a = C * A / (K * V_R_abs)
            b = B / (K * V_R_abs)
            w_jump = W_jump / (K * V_R_abs**2)
            Delta_I_dimless = Delta_I / (K * V_R_abs**2)
            I_ext_dimless = I_ext / (K * V_R_abs**2)

        return {
            'tau_pop': tau_pop,
            'alpha': alpha,
            'a': a,
            'b': b,
            'w_jump': w_jump,
            'Delta_I': Delta_I_dimless,
            'I_ext': I_ext_dimless,
        }

    @staticmethod
    def dimensionless_to_dimensional(
        tau_pop: float,
        alpha: float,
        a: float,
        b: float,
        w_jump: float,
        Delta_I: float,
        I_ext: float,
        V_R: float,
        K: float,
    ) -> Dict[str, float]:
        """
        Convert dimensionless parameters back to dimensional.

        Inverse formulas:
            C = tau_pop * K * |V_R|
            V_T = (alpha - 1) * |V_R|
            A = a * K * |V_R| / C
            B = b * K * |V_R|
            W_jump = w_jump * K * |V_R|^2
            Delta_I = Delta_I * K * |V_R|^2
            I_ext = I_ext * K * |V_R|^2

        Args:
            tau_pop: Population time constant [ms]
            alpha: Threshold parameter [dimensionless]
            a: Adaptation rate [dimensionless]
            b: Adaptation coupling [dimensionless]
            w_jump: Adaptation jump [dimensionless]
            Delta_I: Lorentzian spread [dimensionless]
            I_ext: External current [dimensionless]
            V_R: Resting potential [mV]
            K: Scaling parameter [nS/mV]

        Returns:
            Dictionary with dimensional parameters:
            V_T, C, A, B, W_jump, Delta_I, I_ext
        """
        V_R_abs = abs(V_R)
        C = tau_pop * K * V_R_abs
        return {
            'V_T': (alpha - 1.0) * V_R_abs,
            'C': C,
            'A': a * K * V_R_abs / C,
            'B': b * K * V_R_abs,
            'W_jump': w_jump * K * V_R_abs**2,
            'Delta_I': Delta_I * K * V_R_abs**2,
            'I_ext': I_ext * K * V_R_abs**2,
        }
