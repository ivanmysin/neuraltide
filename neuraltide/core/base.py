import tensorflow as tf
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any

import neuraltide.config
from neuraltide.constraints import MinMaxConstraint
from neuraltide.core.types import TensorType, StateList


class PopulationModel(tf.keras.layers.Layer):
    """
    Абстрактный базовый класс для популяционной модели.

    Семантика n_units:
        n_units — число независимых популяций одного типа внутри одного объекта.
        Например, IzhikevichMeanField(n_units=4) описывает 4 независимые
        популяции нейронов Ижикевича с (потенциально) разными параметрами.
        Все n_units популяций обрабатываются одной батчированной матричной
        операцией — один kernel launch на объект.

    Параметры:
        Все параметры регистрируются через self._make_param(params, name).
        Обучаемость, начальное значение и ограничения задаются через
        словарь params при создании экземпляра.

    Контракт подкласса:
        1. Установить self.state_size в __init__ до завершения super().__init__().
        2. Реализовать get_initial_state().
        3. Реализовать derivatives().
        4. Реализовать get_firing_rate().
        5. Опционально переопределить observables() для доступа к v_mean и т.д.
        6. Реализовать parameter_spec (property).

    Args:
        n_units (int): число независимых популяций данного типа.
        dt (float): шаг интегрирования в мс.
        name (str): имя слоя Keras.
    """

    def __init__(self, n_units: int, dt: float,
                 name: str = "population", **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_units = n_units
        self.dt = dt

    def _make_param(self, params: dict, name: str) -> tf.Variable:
        """
        Регистрирует параметр популяции через add_weight().

        Форматы params[name]:

            Без словаря (только значение, trainable=False):
                params[name] = 0.5
                params[name] = [0.5, 0.6, 0.4, 0.7]

            Словарь с полным контролем:
                params[name] = {
                    'value':     0.5,           # скаляр или список длины n_units
                    'trainable': True,           # по умолчанию False
                    'min':       0.01,           # нижняя граница (опционально)
                    'max':       2.0,            # верхняя граница (опционально)
                }

        Правила broadcast:
            Скаляр → tf.fill([n_units], scalar)
            Список длины n_units → используется as-is
            Список другой длины → ValueError

        Returns:
            tf.Variable, зарегистрированная как вес слоя.

        Raises:
            ValueError: если name не найден в params.
            ValueError: если длина списка не совпадает с n_units.
        """
        spec = params.get(name)
        if spec is None:
            raise ValueError(
                f"PopulationModel '{self.name}': "
                f"parameter '{name}' not found in params."
            )
        if not isinstance(spec, dict):
            spec = {'value': spec, 'trainable': False}

        raw = spec['value']
        train = spec.get('trainable', False)
        lo = spec.get('min', None)
        hi = spec.get('max', None)

        if train:
            constraint = MinMaxConstraint(min_val=lo, max_val=hi) if (lo is not None or hi is not None) else None
        else:
            constraint = None

        value = tf.constant(raw, dtype=neuraltide.config.get_dtype())
        if value.shape.rank == 0:
            value = tf.fill([self.n_units], value)
        else:
            if value.shape[0] != self.n_units:
                raise ValueError(
                    f"PopulationModel '{self.name}': parameter '{name}' "
                    f"has length {value.shape[0]}, expected {self.n_units}."
                )

        return self.add_weight(
            shape=(self.n_units,),
            initializer=tf.keras.initializers.Constant(value.numpy()),
            trainable=train,
            constraint=constraint,
            dtype=neuraltide.config.get_dtype(),
            name=name,
        )

    @abstractmethod
    def get_initial_state(self, batch_size: int = 1) -> StateList:
        """
        Возвращает список тензоров начального состояния.

        Каждый тензор совместим с соответствующим элементом self.state_size.
        Стандартная инициализация — нулевые тензоры.

        Returns:
            list of tf.Tensor. Для большинства моделей:
                [tf.zeros([1, n_units]), ...]  — по одному тензору на переменную состояния.
        """
        raise NotImplementedError

    @abstractmethod
    def derivatives(
        self,
        state: StateList,
        total_synaptic_input: Dict[str, TensorType],
    ) -> StateList:
        """
        Вычисляет производные состояния популяции.

        Все внешние воздействия поступают через синапсы и содержатся
        в total_synaptic_input. Прямого аргумента external_input нет —
        внешние входы подключаются через InputPopulation и синапсы.

        Args:
            state: текущее состояние — список тензоров.
            total_synaptic_input: словарь агрегированных синаптических сигналов:
                {
                    'I_syn': tf.Tensor [1, n_units],  # суммарный синаптический ток
                    'g_syn': tf.Tensor [1, n_units],  # суммарная проводимость
                }

        Returns:
            list of tf.Tensor — производные состояния, той же структуры что state.
        """
        raise NotImplementedError

    @abstractmethod
    def get_firing_rate(self, state: StateList) -> TensorType:
        """
        Извлекает частоту разрядов из состояния популяции.

        Returns:
            tf.Tensor, shape = [1, n_units], в единицах Гц (спайков/с).
            Значения неотрицательны.
        """
        raise NotImplementedError

    def observables(self, state: StateList) -> Dict[str, TensorType]:
        """
        Возвращает словарь наблюдаемых переменных состояния.

        Обязательный ключ: 'firing_rate'.
        Опциональные ключи (если модель поддерживает):
            'v_mean'  — средний мембранный потенциал, shape [1, n_units].
            'w_mean'  — среднее адаптационное переменное.
            'lfp_proxy' — прокси LFP.

        По умолчанию возвращает только {'firing_rate': get_firing_rate(state)}.
        """
        return {'firing_rate': self.get_firing_rate(state)}

    @property
    @abstractmethod
    def parameter_spec(self) -> Dict[str, Dict[str, Any]]:
        """
        Спецификация параметров модели для summary и сериализации.

        Returns:
            {
                'param_name': {
                    'shape':      tuple,
                    'trainable':  bool,
                    'constraint': str or None,
                    'units':      str,
                },
                ...
            }
        """
        raise NotImplementedError

    def adjoint_derivatives(
        self,
        adjoint_state: StateList,
        state: StateList,
        total_synaptic_input: Dict[str, TensorType],
    ) -> StateList:
        """
        Вычисляет производные сопряжённого состояния назад во времени.

        Сопряжённая система ОДУ для adjoint state method:
            dλ/dt = -J^T @ λ

        где λ — сопряжённое состояние (adjoint_state),
        J = ∂derivatives/∂state — якобиан производных по состоянию.

        Args:
            adjoint_state: текущее сопряжённое состояние — список тензоров.
            state: текущее состояние популяции — список тензоров.
            total_synaptic_input: {'I_syn': [...], 'g_syn': [...]}.

        Returns:
            list of tf.Tensor — производные сопряжённого состояния dλ/dt.
        """
        raise NotImplementedError

    def parameter_jacobian(
        self,
        param_name: str,
        state: StateList,
        total_synaptic_input: Dict[str, TensorType],
    ) -> TensorType:
        """
        Вычисляет ∂derivatives/∂param — якобиан производных по параметру.

        Used in adjoint method for computing parameter gradients:
            ∂L/∂param = ∫ adjoint_state^T @ (∂derivatives/∂param) dt

        Args:
            param_name: имя параметра.
            state: текущее состояние популяции.
            total_synaptic_input: {'I_syn': [...], 'g_syn': [...]}.

        Returns:
            Tensor той же формы что parameter.
        """
        raise NotImplementedError


class SynapseModel(tf.keras.layers.Layer):
    """
    Абстрактный базовый класс для синаптической модели.

    Синапс описывает проекцию от n_pre пресинаптических популяций
    к n_post постсинаптическим популяциям.

    Матрица параметров:
        Все весовые матрицы имеют форму [n_pre, n_post].
        Элемент [i, j] описывает связь от пресинаптической популяции i
        к постсинаптической популяции j.

    Маскирование через pconn:
        pconn [n_pre, n_post] — матрица вероятностей/масок соединений,
        trainable=False.

    Args:
        n_pre (int): число пресинаптических популяций.
        n_post (int): число постсинаптических популяций.
        dt (float): шаг интегрирования в мс.
        name (str): имя слоя Keras.
    """

    def __init__(self, n_pre: int, n_post: int, dt: float,
                 name: str = "synapse", **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_pre = n_pre
        self.n_post = n_post
        self.dt = dt

    def _make_param(self, params: dict, name: str) -> tf.Variable:
        """
        Регистрирует параметр синапса через add_weight().

        Форматы params[name] и правила broadcast до [n_pre, n_post]:

            Скаляр:
                {'value': 0.1, 'trainable': True}
                → tf.fill([n_pre, n_post], 0.1)

            Вектор длины n_pre:
                {'value': [0.1, 0.2, 0.15, 0.1], 'trainable': True}
                → reshape [n_pre, 1] → broadcast [n_pre, n_post]

            Вектор длины n_post (только если n_pre != n_post):
                {'value': [0.1, 0.2, 0.15], 'trainable': True}
                → reshape [1, n_post] → broadcast [n_pre, n_post]

            Матрица [n_pre, n_post]:
                → используется as-is.

        Raises:
            ValueError: если name не найден в params.
            ValueError: если форма значения несовместима с [n_pre, n_post].
        """
        spec = params.get(name)
        if spec is None:
            raise ValueError(
                f"SynapseModel '{self.name}': "
                f"parameter '{name}' not found in params."
            )
        if not isinstance(spec, dict):
            spec = {'value': spec, 'trainable': False}

        raw = spec['value']
        train = spec.get('trainable', False)
        lo = spec.get('min', None)
        hi = spec.get('max', None)

        value = tf.constant(raw, dtype=neuraltide.config.get_dtype())
        value = self._broadcast_to_matrix(value, name)

        return self.add_weight(
            shape=(self.n_pre, self.n_post),
            initializer=tf.keras.initializers.Constant(value.numpy()),
            trainable=train,
            constraint=None,
            dtype=neuraltide.config.get_dtype(),
            name=name,
        )

    def _broadcast_to_matrix(self, value: TensorType, name: str) -> TensorType:
        """
        Приводит тензор произвольной формы к [n_pre, n_post].

        Raises:
            ValueError: при несовместимой форме.
        """
        rank = value.shape.rank

        if rank == 0:
            return tf.broadcast_to(
                tf.reshape(value, [1, 1]),
                [self.n_pre, self.n_post]
            )

        elif rank == 1:
            length = int(value.shape[0])
            if length == self.n_pre:
                return tf.broadcast_to(
                    tf.reshape(value, [self.n_pre, 1]),
                    [self.n_pre, self.n_post]
                )
            elif length == self.n_post and self.n_pre != self.n_post:
                return tf.broadcast_to(
                    tf.reshape(value, [1, self.n_post]),
                    [self.n_pre, self.n_post]
                )
            else:
                raise ValueError(
                    f"SynapseModel '{self.name}': parameter '{name}' "
                    f"has length {length}, expected {self.n_pre} (n_pre) "
                    f"or {self.n_post} (n_post)."
                )

        elif rank == 2:
            if value.shape != (self.n_pre, self.n_post):
                raise ValueError(
                    f"SynapseModel '{self.name}': parameter '{name}' "
                    f"has shape {value.shape}, "
                    f"expected [{self.n_pre}, {self.n_post}]."
                )
            return value

        else:
            raise ValueError(
                f"SynapseModel '{self.name}': parameter '{name}' "
                f"has unsupported rank {rank}. "
                f"Expected scalar, 1D vector, or 2D matrix."
            )

    @abstractmethod
    def get_initial_state(self, batch_size: int = 1) -> StateList:
        """Возвращает список тензоров начального состояния синапса."""
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        pre_firing_rate: TensorType,
        post_voltage: TensorType,
        state: StateList,
        dt: float,
    ) -> Tuple[Dict[str, TensorType], StateList]:
        """
        Вычисляет новое состояние синапса и возвращает синаптический ток.

        Args:
            pre_firing_rate: частота пресинаптических популяций, shape = [1, n_pre], в Гц.
            post_voltage: средний потенциал постсинаптических популяций, shape = [1, n_post].
            state: текущее внутреннее состояние синапса.
            dt: шаг интегрирования в мс.

        Returns:
            Tuple:
                current_dict: {
                    'I_syn': tf.Tensor [1, n_post],
                    'g_syn': tf.Tensor [1, n_post],
                }
                new_state: list of tf.Tensor — новое состояние синапса.
        """
        raise NotImplementedError

    @abstractmethod
    def derivatives(
        self,
        state: StateList,
        pre_firing_rate: TensorType,
        post_voltage: TensorType,
    ) -> StateList:
        """
        Вычисляет производные состояния синапса.

        Args:
            state: текущее состояние синапса — список тензоров.
            pre_firing_rate: частота пресинаптических популяций, shape = [1, n_pre], в Гц.
            post_voltage: средний потенциал постсинаптических популяций, shape = [1, n_post].

        Returns:
            list of tf.Tensor — производные состояния, той же структуры что state.
        """
        raise NotImplementedError

    def compute_current(
        self,
        state: StateList,
        pre_firing_rate: TensorType,
        post_voltage: TensorType,
    ) -> Dict[str, TensorType]:
        """
        Вычисляет синаптический ток и проводимость по текущему состоянию.

        Этот метод используется после численного интегрирования для получения
        токов на основе обновлённого состояния.

        Args:
            state: текущее состояние синапса.
            pre_firing_rate: частота пресинаптических популяций, shape = [1, n_pre], в Гц.
            post_voltage: средний потенциал постсинаптических популяций, shape = [1, n_post].

        Returns:
            dict с ключами 'I_syn' и 'g_syn'.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def parameter_spec(self) -> Dict[str, Dict[str, Any]]:
        """Аналогично PopulationModel.parameter_spec."""
        raise NotImplementedError

    def adjoint_forward(
        self,
        adjoint_current: Dict[str, TensorType],
        pre_firing_rate: TensorType,
        post_voltage: TensorType,
        state: StateList,
    ) -> Tuple[Dict[str, TensorType], StateList]:
        """
        Вычисляет сопряжённое состояние синапса назад во времени.

        Adjoint method для синапсов:
            dλ_syn/dt = -J_syn^T @ λ_syn

        где J_syn = ∂forward_output/∂syn_state.

        Args:
            adjoint_current: словарь {'I_syn': ..., 'g_syn': ...}.
            pre_firing_rate: частота пресинаптических популяций.
            post_voltage: средний потенциал постсинаптических популяций.
            state: текущее состояние синапса.

        Returns:
            (new_adjoint_dict, new_adjoint_state):
                new_adjoint_dict: {'I_syn': ..., 'g_syn': ...} для входа в популяцию.
                new_adjoint_state: новое сопряжённое состояние синапса.
        """
        raise NotImplementedError


class BaseInputGenerator(tf.keras.layers.Layer):
    """
    Базовый класс для генератора входных сигналов.

    Генератор оборачивается в InputPopulation и подключается к
    динамическим популяциям через полноценные синапсы (SynapseModel).

    Семантика n_units:
        n_units — число независимых входных каналов одного типа.
        Например, VonMisesGenerator с n_units=4 описывает 4 независимых
        входных сигнала с (потенциально) разными параметрами.
        Все n_units каналов обрабатываются одной векторизованной операцией.

    Параметры:
        Все параметры регистрируются через self._make_param(params, name).
        Формат аналогичен PopulationModel:
            - скаляр (broadcast к n_units)
            - список длины n_units
            - словарь {'value': ..., 'trainable': bool, 'min':, 'max':}

    Args:
        params: словарь параметров генератора.
        dt: шаг интегрирования в мс.
        name: имя слоя Keras.
    """

    def __init__(self, params: Dict[str, Any], dt: float,
                 name: str = "input_generator", **kwargs):
        super().__init__(name=name, **kwargs)
        self.dt = dt
        self._params = params

        self._infer_n_units_from_params()
        self._validate_param_dimensions()

        self.n_units = self._n_units

    def _infer_n_units_from_params(self) -> None:
        """Определяет n_units из размерности параметров."""
        import numpy as np
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

        self._n_units = max_len

    def _validate_param_dimensions(self) -> None:
        """Проверяет согласованность размерностей параметров."""
        import numpy as np
        for key, spec in self._params.items():
            if isinstance(spec, dict):
                value = spec.get('value', None)
            else:
                value = spec

            if value is None:
                continue

            if isinstance(value, (list, tuple)):
                if len(value) != 1 and len(value) != self._n_units:
                    raise ValueError(
                        f"BaseInputGenerator '{self.name}': parameter '{key}' "
                        f"has length {len(value)}, expected 1 or {self._n_units}."
                    )
            elif isinstance(value, np.ndarray):
                if value.ndim == 1 and len(value) != 1 and len(value) != self._n_units:
                    raise ValueError(
                        f"BaseInputGenerator '{self.name}': parameter '{key}' "
                        f"has length {len(value)}, expected 1 or {self._n_units}."
                    )

    def _make_param(self, params: dict, name: str) -> tf.Variable:
        """
        Регистрирует параметр генератора через add_weight().

        Форматы params[name]:
            Без словаря (только значение, trainable=False):
                params[name] = 0.5
                params[name] = [0.5, 0.6, 0.4, 0.7]

            Словарь с полным контролем:
                params[name] = {
                    'value':     0.5,
                    'trainable': True,
                    'min':       0.0,
                    'max':       10.0,
                }

        Returns:
            tf.Variable, зарегистрированная как вес слоя.
        """
        spec = params.get(name)
        if spec is None:
            raise ValueError(
                f"BaseInputGenerator '{self.name}': "
                f"parameter '{name}' not found in params."
            )
        if not isinstance(spec, dict):
            spec = {'value': spec, 'trainable': False}

        raw = spec['value']
        train = spec.get('trainable', False)
        lo = spec.get('min', None)
        hi = spec.get('max', None)

        from neuraltide.constraints import MinMaxConstraint
        constraint = MinMaxConstraint(min_val=lo, max_val=hi) if (lo is not None or hi is not None) else None

        value = tf.constant(raw, dtype=neuraltide.config.get_dtype())
        if value.shape.rank == 0:
            value = tf.fill([self._n_units], value)
        else:
            if int(value.shape[0]) != self._n_units:
                raise ValueError(
                    f"BaseInputGenerator '{self.name}': parameter '{name}' "
                    f"has length {int(value.shape[0])}, expected {self._n_units}."
                )

        return self.add_weight(
            shape=(self._n_units,),
            initializer=tf.keras.initializers.Constant(value.numpy()),
            trainable=train,
            constraint=constraint,
            dtype=neuraltide.config.get_dtype(),
            name=name,
        )

    @property
    def parameter_spec(self) -> Dict[str, Dict[str, Any]]:
        """
        Спецификация параметров генератора для summary и сериализации.
        """
        raise NotImplementedError
