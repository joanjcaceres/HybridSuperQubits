# HybridSuperQubits 🌀⚡

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14826349.svg)](https://doi.org/10.5281/zenodo.14826349)
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

### Recommended Installation (Apple Silicon M1/M2/M3)

For **optimal performance** on Apple Silicon Macs, install scientific dependencies via conda-forge:

```bash
# Option 1: Use the provided environment file
conda env create -f environment.yml
conda activate hybridsuperqubits

# Option 2: Manual conda installation
conda create -n hybridsuperqubits python>=3.9
conda activate hybridsuperqubits
conda install -c conda-forge numpy scipy matplotlib qutip scqubits
pip install -e . --no-deps
```

📖 **Detailed Apple Silicon guide**: [INSTALL_APPLE_SILICON.md](INSTALL_APPLE_SILICON.md)

### Quick Installation (All Platforms)

```bash
pip install HybridSuperQubits[full]
```

⚠️ **Note**: On Apple Silicon, this may install unoptimized versions of scientific libraries.

### Minimal Installation

```bash
pip install HybridSuperQubits
# Then manually install: numpy, scipy, matplotlib, qutip, scqubits
```

### Development Installation

```bash
git clone https://github.com/joanjcaceres/HybridSuperQubits.git
cd HybridSuperQubits
pip install -e .[full]
```

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

## 📖 Citation

If you use this software in your research, please cite it using the following BibTeX entry

```bibtex
@software{joan_j_caceres_2025_15315785,
  author       = {Joan J. Cáceres},
  title        = {joanjcaceres/HybridSuperQubits},
  month        = may,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {v0.8.2},
  doi          = {10.5281/zenodo.15315785},
  url          = {https://doi.org/10.5281/zenodo.15315785},
}
```

or using the Citation tool at [HybridSuperQubits' Zenodo](https://zenodo.org/records/15315785)
