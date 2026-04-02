import pytest
import tensorflow as tf

import neuraltide
import neuraltide.config
from neuraltide.core.base import PopulationModel, SynapseModel, BaseInputGenerator


class TestAbstractClasses:
    def test_population_model_abstract_methods(self):
        """PopulationModel requires implementing abstract methods."""
        pop = PopulationModel(n_units=2, dt=0.5)
        with pytest.raises(NotImplementedError):
            pop.get_initial_state()

    def test_synapse_model_abstract_methods(self):
        """SynapseModel requires implementing abstract methods."""
        syn = SynapseModel(n_pre=2, n_post=2, dt=0.5)
        with pytest.raises(NotImplementedError):
            syn.get_initial_state()

    def test_base_input_generator_is_layer(self):
        """BaseInputGenerator is a Keras Layer."""
        assert issubclass(BaseInputGenerator, tf.keras.layers.Layer)


class MinimalPopulation(PopulationModel):
    """Минимальная конкретная реализация PopulationModel для тестов."""

    def __init__(self, n_units, dt, **kwargs):
        super().__init__(n_units=n_units, dt=dt, **kwargs)
        self.state_size = [tf.TensorShape([1, n_units])]

    def get_initial_state(self, batch_size=1):
        return [tf.zeros([1, self.n_units], dtype=neuraltide.config.get_dtype())]

    def derivatives(self, state, total_synaptic_input):
        return state

    def get_firing_rate(self, state):
        return state[0]

    @property
    def parameter_spec(self):
        return {}


class TestMinimalConcretePopulation:
    def test_isinstance_check(self):
        pop = MinimalPopulation(n_units=2, dt=0.5)
        assert isinstance(pop, PopulationModel)

    def test_make_param_broadcast_scalar(self):
        """_make_param broadcast скаляра до [n_units]."""
        pop = MinimalPopulation(n_units=4, dt=0.5)
        param = pop._make_param({'value': 1.5, 'trainable': False}, 'value')
        assert param.shape == (4,)
        values = param.numpy()
        assert all(v == 1.5 for v in values)


class MinimalSynapse(SynapseModel):
    """Минимальная конкретная реализация SynapseModel для тестов."""

    def __init__(self, n_pre, n_post, dt, **kwargs):
        super().__init__(n_pre=n_pre, n_post=n_post, dt=dt, **kwargs)
        self.state_size = []

    def get_initial_state(self, batch_size=1):
        return []

    def forward(self, pre_firing_rate, post_voltage, state, dt):
        return (
            {
                'I_syn': tf.zeros([1, self.n_post]),
                'g_syn': tf.zeros([1, self.n_post])
            },
            []
        )

    @property
    def parameter_spec(self):
        return {}


class TestMinimalConcreteSynapse:
    def test_isinstance_check(self):
        syn = MinimalSynapse(n_pre=2, n_post=3, dt=0.5)
        assert isinstance(syn, SynapseModel)

    def test_broadcast_to_matrix_scalar(self):
        """_broadcast_to_matrix скаляр → [n_pre, n_post]."""
        syn = MinimalSynapse(n_pre=2, n_post=3, dt=0.5)
        value = tf.constant(0.5)
        result = syn._broadcast_to_matrix(value, 'test')
        assert result.shape == (2, 3)

    def test_broadcast_to_matrix_vector_n_pre(self):
        """_broadcast_to_matrix вектор [n_pre] → [n_pre, n_post]."""
        syn = MinimalSynapse(n_pre=2, n_post=3, dt=0.5)
        value = tf.constant([0.1, 0.2])
        result = syn._broadcast_to_matrix(value, 'test')
        assert result.shape == (2, 3)

    def test_broadcast_to_matrix_vector_n_post_different(self):
        """_broadcast_to_matrix вектор [n_post] при n_pre≠n_post → [n_pre, n_post]."""
        syn = MinimalSynapse(n_pre=2, n_post=3, dt=0.5)
        value = tf.constant([0.1, 0.2, 0.3])
        result = syn._broadcast_to_matrix(value, 'test')
        assert result.shape == (2, 3)

    def test_broadcast_to_matrix_vector_same_n(self):
        """Вектор длины n при n_pre==n_post трактуется как [n_pre]."""
        syn = MinimalSynapse(n_pre=3, n_post=3, dt=0.5)
        value = tf.constant([0.1, 0.2, 0.3])
        result = syn._broadcast_to_matrix(value, 'test')
        assert result.shape == (3, 3)

    def test_broadcast_to_matrix_matrix(self):
        """_broadcast_to_matrix матрица [n_pre, n_post] → без изменений."""
        syn = MinimalSynapse(n_pre=2, n_post=3, dt=0.5)
        value = tf.constant([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        result = syn._broadcast_to_matrix(value, 'test')
        assert result.shape == (2, 3)

    def test_broadcast_to_matrix_wrong_shape(self):
        """Неправильная форма матрицы → ValueError."""
        syn = MinimalSynapse(n_pre=2, n_post=3, dt=0.5)
        value = tf.constant([[0.1, 0.2], [0.3, 0.4]])
        with pytest.raises(ValueError):
            syn._broadcast_to_matrix(value, 'test')


class MinimalInput(BaseInputGenerator):
    """Минимальная конкретная реализация BaseInputGenerator для тестов."""

    def __init__(self, n_outputs, **kwargs):
        super().__init__(n_outputs=n_outputs, **kwargs)

    def call(self, t):
        return tf.ones([1, self.n_outputs])


class TestMinimalConcreteInput:
    def test_isinstance_check(self):
        inp = MinimalInput(n_outputs=2)
        assert isinstance(inp, BaseInputGenerator)
