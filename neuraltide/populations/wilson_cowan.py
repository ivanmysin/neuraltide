import tensorflow as tf

import neuraltide
import neuraltide.config
from neuraltide.core.base import (
    PopulationModel,
    _get_constraint_name,
    _infer_n_units_from_params,
    _validate_param_dimensions,
)
from neuraltide.core.types import TensorType, StateList, Dict


class WilsonCowan(PopulationModel):
    """
    Модель Вильсона–Коуэна в форме одной популяции (rate model).

    Одна популяция описывается одной переменной состояния E —
    нормированной активностью (0..1). Торможение учитывается через
    входящие синаптические токи (в том числе от отдельных тормозных
    популяций с отрицательным весом).

    Уравнение:
        τ · dE/dt = −E + F(w·E + I_ext + I_syn)
        F(x) = 1 / (1 + exp(−a·(x − θ)))

    Состояние: [E], shape [1, n_units] — по одной независимой
    популяции на юнит, выход — частота разрядов relu(E)·max_rate,
    как у среднеполевой модели Ижикевича.

    Параметры (params, все обязательны):
        tau:       постоянная времени (мс). Скаляр или [n_units].
        a:         крутизна сигмоиды (безразмерная). Скаляр или [n_units].
        theta:     порог сигмоиды (мВ). Скаляр или [n_units].
        w:         рекуррентный вес (безразмерный). Скаляр или [n_units].
        I_ext:     внешний ток (мВ). Скаляр или [n_units].
        max_rate:  масштаб частоты разрядов (Гц). Скаляр или [n_units].

    n_units выводится из максимальной длины параметров (как у
    IzhikevichMeanField) и не передаётся явно.

    Пример:
        pop = WilsonCowan(
            dt=0.5,
            params={
                'tau':      {'value': [10.0, 10.0], 'trainable': False},
                'a':        {'value': [1.0, 1.2],   'trainable': False},
                'theta':    {'value': [0.0, 0.0],   'trainable': False},
                'w':        {'value': [1.0, 0.8],   'trainable': True},
                'I_ext':    {'value': [1.0, 0.5],   'trainable': True},
                'max_rate': {'value': [100.0, 100.0], 'trainable': False},
            },
        )
        # n_units=2
    """

    def __init__(self, dt: float, params: dict, name: str = "wilson_cowan", **kwargs):
        self.name = name
        self._params = params
        self.n_units = _infer_n_units_from_params(params)
        self._validate_params()
        self._validate_param_dimensions()

        super().__init__(n_units=self.n_units, dt=dt, name=name, **kwargs)

        self.tau = self._make_param(self._params, "tau")
        self.a = self._make_param(self._params, "a")
        self.theta = self._make_param(self._params, "theta")
        self.w = self._make_param(self._params, "w")
        self.I_ext = self._make_param(self._params, "I_ext")
        self.max_rate = self._make_param(self._params, "max_rate")

        self.state_size = [
            tf.TensorShape([1, self.n_units]),
        ]

    def _validate_params(self) -> None:
        """Проверяет наличие всех обязательных параметров."""
        required = ["tau", "a", "theta", "w", "I_ext", "max_rate"]
        for name in required:
            if name not in self._params:
                raise ValueError(
                    f"WilsonCowan: required parameter '{name}' not found in params."
                )

    def _validate_param_dimensions(self) -> None:
        """Проверяет согласованность размерностей параметров (1 или n_units)."""
        _validate_param_dimensions(
            self._params,
            self.n_units,
            f"WilsonCowan '{self.name}'",
        )

    def get_initial_state(self, batch_size: int = 1) -> StateList:
        dtype = neuraltide.config.get_dtype()
        return [tf.zeros([1, self.n_units], dtype=dtype)]

    def derivatives(
        self,
        state: StateList,
        total_synaptic_input: Dict[str, TensorType],
    ) -> StateList:
        E = state[0]
        I_syn = total_synaptic_input["I_syn"]

        dtype = neuraltide.config.get_dtype()
        tau = tf.cast(self.tau, dtype)
        a = tf.cast(self.a, dtype)
        theta = tf.cast(self.theta, dtype)
        w = tf.cast(self.w, dtype)
        I_ext = tf.cast(self.I_ext, dtype)

        x = w * E + I_ext + I_syn
        F = 1.0 / (1.0 + tf.exp(-a * (x - theta)))
        dEdt = (-E + F) / tau
        return [dEdt]

    def get_firing_rate(self, state: StateList) -> TensorType:
        E = state[0]
        max_rate = tf.cast(self.max_rate, neuraltide.config.get_dtype())
        return tf.nn.relu(E) * max_rate

    def observables(self, state: StateList) -> Dict[str, TensorType]:
        return {"firing_rate": self.get_firing_rate(state)}

    @property
    def parameter_spec(self) -> Dict[str, Dict[str, any]]:
        return {
            "tau": {
                "shape": (self.n_units,),
                "trainable": self.tau.trainable,
                "constraint": self._get_constraint_name(self.tau),
                "units": "ms",
            },
            "a": {
                "shape": (self.n_units,),
                "trainable": self.a.trainable,
                "constraint": self._get_constraint_name(self.a),
                "units": "dimensionless",
            },
            "theta": {
                "shape": (self.n_units,),
                "trainable": self.theta.trainable,
                "constraint": self._get_constraint_name(self.theta),
                "units": "mV",
            },
            "w": {
                "shape": (self.n_units,),
                "trainable": self.w.trainable,
                "constraint": self._get_constraint_name(self.w),
                "units": "dimensionless",
            },
            "I_ext": {
                "shape": (self.n_units,),
                "trainable": self.I_ext.trainable,
                "constraint": self._get_constraint_name(self.I_ext),
                "units": "mV",
            },
            "max_rate": {
                "shape": (self.n_units,),
                "trainable": self.max_rate.trainable,
                "constraint": self._get_constraint_name(self.max_rate),
                "units": "Hz",
            },
        }

    def _get_constraint_name(self, var: tf.Variable) -> str:
        return _get_constraint_name(var)
