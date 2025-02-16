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

---

## 🚀 Installation

HybridSuperQubits is available on [PyPI](https://pypi.org/project/HybridSuperQubits/).  
**SciPy** is kept as an **optional** dependency to let users install it optimally (especially on Apple Silicon).

### 1. Quick Installation (includes SciPy)

If you do **not** need to manage SciPy installation yourself, simply:

    pip install "HybridSuperQubits[scipy]"

> **Apple Silicon (M1/M2/M3)**: If SciPy compiles from source or runs slowly, check the notes below.

### 2. Manual / Optimized SciPy Installation

If you prefer to install SciPy independently (for example, via conda or building from source):

1. **(Optional) Create or activate a Python environment**:

- **Conda example**:
    
        conda create -n hsq_env python=3.10
        conda activate hsq_env

- **venv example**:
    
        python3 -m venv hsq_env
        source hsq_env/bin/activate     # macOS/Linux
        hsq_env\Scripts\activate        # Windows

2. **Install SciPy** by your preferred approach:

- **Conda**:
     
         conda install scipy

- **pip + Homebrew** (if compiling from source):
     
         brew install openblas gcc
         pip install --upgrade pip setuptools wheel
         pip install scipy

3. **Install HybridSuperQubits** (without `[scipy]`):

        pip install HybridSuperQubits

### Apple Silicon Notes (M1/M2/M3)

- Use a **native** Python build (not under Rosetta).
- If SciPy or HybridSuperQubits tries to **compile from source** and you do not get a precompiled wheel, you may need OpenBLAS and environment variables:
  
      conda install -c conda-forge openblas
      export LDFLAGS="-L/opt/homebrew/opt/openblas/lib"
      export CFLAGS="-I/opt/homebrew/opt/openblas/include"
      export BLAS=~/opt/homebrew/opt/openblas/lib
      pip install HybridSuperQubits
  
- Installing SciPy via **conda-forge** or **mambaforge** typically provides optimized builds automatically.

### Upgrading

To upgrade HybridSuperQubits:

    pip install --upgrade "HybridSuperQubits[scipy]"

(Or just `HybridSuperQubits` if handling SciPy separately.)

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

