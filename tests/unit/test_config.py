import pytest
import tensorflow as tf
import neuraltide
import neuraltide.config as config
from neuraltide.core.base import PopulationModel

class TestDtypeConfig:
    def test_default_dtype(self):
        neuraltide.config.set_dtype(tf.float32)
        assert config.get_dtype() == tf.float32

    def test_set_dtype_float64(self):
        neuraltide.config.set_dtype(tf.float64)
        assert config.get_dtype() == tf.float64
        neuraltide.config.set_dtype(tf.float32)

    def test_set_dtype_changes_global(self):
        neuraltide.config.set_dtype(tf.float64)
        assert config.get_dtype() == tf.float64
        neuraltide.config.set_dtype(tf.float32)


class TestRegistries:
    def test_register_population(self):
        class DummyPopulation(PopulationModel):
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

        config.register_population('DummyPopulation', DummyPopulation)
        assert 'DummyPopulation' in config.POPULATION_REGISTRY
        assert config.POPULATION_REGISTRY['DummyPopulation'] == DummyPopulation

    def test_register_synapse(self):
        from neuraltide.core.base import SynapseModel

        class DummySynapse(SynapseModel):
            def __init__(self, n_pre, n_post, dt, **kwargs):
                super().__init__(n_pre=n_pre, n_post=n_post, dt=dt, **kwargs)
                self.state_size = []

            def get_initial_state(self, batch_size=1):
                return []

            def forward(self, pre_firing_rate, post_voltage, state, dt):
                return ({'I_syn': tf.zeros([1, self.n_post]),
                         'g_syn': tf.zeros([1, self.n_post])}, [])

            @property
            def parameter_spec(self):
                return {}

        config.register_synapse('DummySynapse', DummySynapse)
        assert 'DummySynapse' in config.SYNAPSE_REGISTRY
        assert config.SYNAPSE_REGISTRY['DummySynapse'] == DummySynapse

    def test_register_input(self):
        from neuraltide.inputs import BaseInputGenerator

        class DummyInput(BaseInputGenerator):
            def __init__(self, n_outputs, **kwargs):
                super().__init__(n_outputs=n_outputs, **kwargs)

            def call(self, t):
                return tf.ones([1, self.n_outputs])

        config.register_input('DummyInput', DummyInput)
        assert 'DummyInput' in config.INPUT_REGISTRY
        assert config.INPUT_REGISTRY['DummyInput'] == DummyInput
