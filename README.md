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

HybridSuperQubits can be installed from PyPI. However, **SciPy** is not included among the mandatory dependencies so that each user can install it in the most suitable way (especially relevant for macOS Apple Silicon).

Choose one of the following:

### 1. Quick Installation (includes SciPy)

If you do **not** need to control how SciPy is installed and are fine with default pip wheels:

    pip install "HybridSuperQubits[scipy]"

> **Note for Apple Silicon (M1/M2/M3)**: If SciPy builds from source or runs slowly, 
> see the Apple Silicon notes below.

### 2. Manual / Optimized SciPy Installation

If you prefer to handle SciPy yourself (e.g., via conda or compiling from source):

1. Create or activate a Python environment:

   Using conda:

       conda create -n hsq_env python=3.10
       conda activate hsq_env

   Using venv:

       python3 -m venv hsq_env
       source hsq_env/bin/activate
       # or on Windows:
       hsq_env\Scripts\activate

2. Install SciPy by your chosen method:

- With conda:

      conda install scipy

 - With pip (possibly with Homebrew for libraries):

       brew install openblas gcc
       pip install --upgrade pip setuptools wheel
       pip install scipy

1. Install HybridSuperQubits (without SciPy extra):

       pip install HybridSuperQubits

### Apple Silicon (M1/M2/M3) Notes

- Make sure you are using a **native** Python build (not under Rosetta).
- If SciPy compiles from source and is extremely slow, it might not be linking to Accelerate or OpenBLAS.
- **Conda-forge** or **mambaforge** often provides optimized SciPy builds for Apple Silicon.

### Virtual Environments (Recommended)

Regardless of your method, installing into an isolated environment prevents dependency conflicts:

    conda create -n hsq_env python=3.10
    conda activate hsq_env
    pip install "HybridSuperQubits[scipy]"

Or if you installed SciPy separately:

    conda create -n hsq_env python=3.10
    conda activate hsq_env
    conda install scipy
    pip install HybridSuperQubits

### Upgrading

To update:

    pip install --upgrade "HybridSuperQubits[scipy]"

(Or just `HybridSuperQubits` if you're managing SciPy yourself.)

---

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

