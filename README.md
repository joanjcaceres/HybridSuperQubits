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

## 🚀 Installation

Follow these steps to install or contribute to the HybridSuperQubits library.

### For End Users

If you only plan to *use* `HybridSuperQubits` (i.e., you do not need to modify or extend its functionality), you can install it directly from PyPI:

1. **(Optional) Create a Virtual Environment**

   **Conda example**:
   ```bash
   conda create --name hsq_env python=3.10
   conda activate hsq_env
   ```

   **Or using `venv`**:
   ```bash
   python3 -m venv hsq_env
   source hsq_env/bin/activate  # macOS/Linux
   hsq_env/Scripts/activate     # Windows
   ```

2. **Install HybridSuperQubits via pip**:
   ```bash
   pip install hybridsuperqubits
   ```

   To upgrade:
   ```bash
   pip install --upgrade hybridsuperqubits
   ```

   **Note (macOS M1/M2/M3):** If you run into issues with `SciPy` on Apple Silicon, install OpenBLAS first:

   > ```bash
   > brew install openblas
   > pip install scipy
   > ```

### For Contributors or Developers

If you want to *contribute* to the project or modify the code:

1. **Fork and Clone** the repository from GitHub (see detailed steps in [`CONTRIBUTING.md`](CONTRIBUTING.md)).
2. Create a **new branch** for your feature or bug fix.
3. (Optional) Set up a **development environment**:
   ```bash
   conda create --name hsq_dev python=3.10
   conda activate hsq_dev
   ```
4. **Install in Editable Mode**:
   ```bash
   pip install -e .
   ```
   This lets you test changes locally without reinstalling the package each time.

5. **Run Tests** (if applicable):
   ```bash
   pytest tests/
   ```

For more details on contributing guidelines, code style, testing, and pull requests, please read our 
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## Basic Usage 🚀
### Supported Qubit Types
1. Andreev
2. Gatemon
3. Gatemonium
4. Fermionic bosonic qubit

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

