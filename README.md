# NeuralTide

NeuralTide is a Python package for differentiable modeling and training of population neural networks.

## Key Features

- **Population models**: Wilson-Cowan, Izhikevich mean-field, Fokker-Planck
- **Synaptic dynamics**: Tsodyks-Markram (STP), NMDA with Mg block, static synapses
- **Differentiable**: All computations via TensorFlow ops for automatic gradient
- **Extensible**: User inherits from base classes and implements minimal contract
- **BPTT training**: Backpropagation through time via explicit step-by-step integrator

## Installation

```bash
pip install neuraltide
```

Or from source:

```bash
pip install tensorflow>=2.16 numpy>=1.26 scipy>=1.12
```

## Quick Start

```python
import numpy as np
import tensorflow as tf
from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.populations import IzhikevichMeanField
from neuraltide.synapses import TsodyksMarkramSynapse
from neuraltide.inputs import VonMisesGenerator
from neuraltide.integrators import RK4Integrator
from neuraltide.training import Trainer, CompositeLoss, MSELoss

# Create population
pop = IzhikevichMeanField(n_units=2, dt=0.5, params={
    'alpha':     {'value': [0.5, 0.5],   'trainable': False},
    'a':         {'value': [0.02, 0.02], 'trainable': False},
    'b':         {'value': [0.2, 0.2],   'trainable': False},
    'w_jump':    {'value': [0.1, 0.1],   'trainable': False},
    'dt_nondim': {'value': [0.01, 0.01], 'trainable': False},
    'Delta_eta': {'value': [0.5, 0.6],   'trainable': True, 'min': 0.01, 'max': 2.0},
    'I_ext':     {'value': [1.0, 1.2],   'trainable': True},
})

# Create input generator
gen = VonMisesGenerator(params=[
    {'MeanFiringRate': 20.0, 'R': 0.5, 'ThetaFreq': 8.0, 'ThetaPhase': 0.0},
])

# Create synapse
syn = TsodyksMarkramSynapse(n_pre=1, n_post=2, dt=0.5, params={
    'gsyn_max': {'value': [[0.1, 0.1]], 'trainable': True},
    'tau_f':    {'value': 20.0,  'trainable': True, 'min': 6.0,  'max': 240.0},
    'tau_d':    {'value': 5.0,   'trainable': True, 'min': 2.0,  'max': 15.0},
    'tau_r':    {'value': 200.0, 'trainable': True, 'min': 91.0, 'max': 1300.0},
    'Uinc':     {'value': 0.2,   'trainable': True, 'min': 0.04, 'max': 0.7},
    'pconn':    {'value': [[1.0, 1.0]], 'trainable': False},
    'e_r':      {'value': 0.0,   'trainable': False},
})

# Build network
graph = NetworkGraph(dt=0.5)
graph.add_input_population('theta', gen)
graph.add_population('exc', pop)
graph.add_synapse('theta->exc', syn, src='theta', tgt='exc')

network = NetworkRNN(graph, integrator=RK4Integrator())

# Training
t_seq = tf.constant(np.arange(2000)[None, :, None], dtype=tf.float32) * 0.5
target = {'exc': tf.constant(np.random.randn(1, 2000, 2), dtype=tf.float32)}

trainer = Trainer(network, CompositeLoss([(1.0, MSELoss(target))]),
                  optimizer=tf.keras.optimizers.Adam(1e-3))
history = trainer.fit(t_seq, epochs=50)
```

## Examples

See the `examples/` directory for more detailed examples:
- `example_01_single_population.py` - Single IzhikevichMeanField with theta input
- `example_02_exc_inh_nmda.py` - Excitatory-inhibitory network with AMPA+NMDA
- `example_03_custom_population.py` - Custom user-defined population model
- `example_04_fokker_planck.py` - Fokker-Planck population model
- `example_05_lfp_proxy.py` - LFP proxy readout

## Requirements

- Python >= 3.12
- TensorFlow >= 2.16
- NumPy >= 1.26
- SciPy >= 1.12

## License

MIT
