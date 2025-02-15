# HybridSuperQubits 🌀⚡

[![PyPI Version](https://img.shields.io/pypi/v/HybridSuperQubits)](https://pypi.org/project/HybridSuperQubits/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/hybridsuperqubits/badge/?version=latest)](https://hybridsuperqubits.readthedocs.io/en/latest/?badge=latest)

A Python framework for simulating hybrid semiconductor-superconductor quantum circuits.

## Key Features ✨
- **Hybrid Circuit Simulation** 🔬  
  Unified framework for semiconductor-superconductor systems.

- **Advanced Noise Analysis** 📉  
  * Capacitive losses (```t1_capacitive```).
  * Inductive losses (```t1_inductive```).
  * Flux noise (```tphi_1_over_f_flux```).
  * Coherence Quantum Phase Slip (```tphi_CQPS```).
- **Professional Visualization** 📊  
  * Wavefunction plotting (```plot_wavefunction```).
  * Matrix element analysis (```plot_matelem_vs_paramvals```).
  * Spectrum vs parameter sweeps.
- **SC-Qubits Compatibility** 🔄  
  API-inspired interface for users familiar with scqubits

## Installation ⚙️
```bash
pip install HybridSuperQubits
```

## Basic Usage 🚀
### Supported Qubit Types
1. Andreev Pair Qubit (Andreev):
  Semiconductor nanowire-based protected qubit

1. Gatemon (Gatemon)
  Gate-tunable transmon-like qubit

1. Gatemonium (Gatemonium)
  Strongly charge-sensitive gatemon variant

1. Fermionic-Bosonic Qubit (Ferbo)
  Hybrid light-matter qubit (shown in example below)

### Initialize a hybrid qubit
```python
from HybridSuperQubits import Andreev, Gatemon, Gatemonium, Ferbo

# Fermionic-Bosonic Qubit (Ferbo)
qubit = Ferbo(
    Ec=1.2,          # Charging energy [GHz]
    El=0.8,          # Inductive energy [GHz]
    Gamma=5.0,       # Coupling strength [GHz]
    delta_Gamma=0.1, # Asymmetric coupling [GHz]
    er=0.05,         # Fermi detuning [GHz]
    phase=0.3,       # External phase (2 pi Φ/Φ₀)
    dimension=100    # Hilbert space dimension
)

# Andreev Pair Qubit
andreev_qubit = Andreev(
    EJ=15.0,        # Josephson energy [GHz]
    EC=0.5,         # Charging energy [GHz]
    delta=0.1,      # Superconducting gap [GHz]
    ng=0.0,         # Charge offset
    dimension=50
)

# Gatemonium
gatemonium = Gatemonium(
    EJ=10.0,        # Josephson energy [GHz]
    EC=1.2,         # Charging energy [GHz]
    ng=0.0,         # Charge offset
    dimension=100
)
```

## Documentation 📚

Full API reference and theory background available at:
[hybridsuperqubits.readthedocs.io](https://hybridsuperqubits.readthedocs.io/en/latest/?badge=latest)

## Contributing 🤝

We welcome contributions! Please see:

[CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines

## License

This project is licensed under the MIT License. However, it includes portions of code derived from 
[scqubits](https://github.com/scqubits/scqubits), which is licensed under the BSD 3-Clause License.

For more details, please refer to the [`LICENSE`](./LICENSE) file.

