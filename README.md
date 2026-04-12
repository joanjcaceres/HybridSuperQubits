# HybridSuperQubits

[![arXiv](https://img.shields.io/badge/arXiv-2604.01145-b31b1b.svg)](https://arxiv.org/abs/2604.01145)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15773124.svg)](https://doi.org/10.5281/zenodo.15773124)
[![PyPI Version](https://img.shields.io/pypi/v/HybridSuperQubits)](https://pypi.org/project/HybridSuperQubits/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/hybridsuperqubits/badge/?version=latest)](https://hybridsuperqubits.readthedocs.io/en/latest/?badge=latest)

**HybridSuperQubits** is a Python framework for simulating hybrid semiconductor-superconductor quantum circuits. It provides numerical tools for diagonalizing circuit Hamiltonians, computing energy spectra, analyzing noise susceptibility, and visualizing wavefunctions across several qubit architectures.

---

## Overview

Superconducting quantum circuits based on semiconductor weak links are a growing family of devices where gate-tunable Andreev bound states replace or complement conventional Josephson tunnel junctions. HybridSuperQubits provides a unified numerical framework for studying these systems -- from simple gate-tunable junctions (gatemon) to circuits that hybridize fermionic and bosonic degrees of freedom (FerBo).

The package covers a range of hybrid qubit architectures:

- **Charge-basis qubits** -- Andreev pair qubit, gatemon -- where the junction potential depends on the transmission of a semiconductor weak link.
- **Flux-basis qubits** -- fluxonium, gatemonium -- where a large inductance shunts the junction, providing phase delocalization and anharmonicity.
- **Hybrid-basis qubits** -- FerBo -- where the Hilbert space is a tensor product of a bosonic (LC) mode and a fermionic (Andreev) degree of freedom.
- **Metamaterial elements** -- Josephson junction arrays, resonators -- for modeling coupling environments and readout cavities.

All qubit classes share a common interface for Hamiltonian diagonalization, parameter sweeps, matrix element computation, noise analysis (T1, T_phi), and wavefunction visualization.

> This package was used in:
> **FerBo: a noise resilient qubit hybridizing Andreev and fluxonium states**
> J. J. Caceres, D. Sanz Marco, J. Ortuzar, E. Flurin, C. Urbina, H. Pothier, M. F. Goffman, F. J. Matute-Canadas, A. Levy Yeyati
> [arXiv:2604.01145](https://arxiv.org/abs/2604.01145) (2026)

### Supported qubit types

| Qubit | Description | Basis |
|-------|-------------|-------|
| **Ferbo** | Fermionic-bosonic hybrid qubit (Andreev + LC oscillator) | Hybrid (Fock x Andreev) |
| **Andreev** | Andreev pair qubit in a superconducting weak link | Charge |
| **Fluxonium** | Inductive-dominated superconducting qubit | Fock |
| **Gatemon** | Gate-tunable Josephson junction qubit | Charge |
| **Gatemonium** | Gate-tunable junction shunted by an inductance | Fock |
| **JJA** | Josephson junction array (metamaterial modes) | Analytical |
| **Resonator** | Quantum LC resonator / cavity | Fock |

---

## Installation

### Quick install (all platforms)

```bash
pip install HybridSuperQubits[full]
```

### Recommended for Apple Silicon (M1/M2/M3/M4)

For optimal performance with accelerated scientific libraries:

```bash
conda env create -f environment.yml
conda activate hybridsuperqubits
```

Or manually:

```bash
conda create -n hybridsuperqubits python>=3.10
conda activate hybridsuperqubits
conda install -c conda-forge numpy scipy matplotlib qutip scqubits
pip install -e . --no-deps
```

See [INSTALL_APPLE_SILICON.md](INSTALL_APPLE_SILICON.md) for a detailed guide.

### Development install

```bash
git clone https://github.com/joanjcaceres/HybridSuperQubits.git
cd HybridSuperQubits
pip install -e .[full]
```

---

## Quick start

### Fluxonium

```python
import numpy as np
from HybridSuperQubits import Fluxonium

qubit = Fluxonium(
    Ec=1.0,            # Charging energy [GHz]
    El=0.5,            # Inductive energy [GHz]
    Ej=4.0,            # Josephson energy [GHz]
    phase=np.pi,       # External flux (2pi Phi/Phi_0)
    dimension=100      # Fock space dimension
)

# Eigenvalues and eigenstates
evals, evecs = qubit.eigensys(evals_count=6)

# Energy spectrum vs external flux
spectrum = qubit.get_spectrum_vs_paramvals(
    param_name="phase",
    param_vals=np.linspace(0, 2 * np.pi, 101),
    evals_count=6
)
```

### Andreev pair qubit

```python
from HybridSuperQubits import Andreev

qubit = Andreev(
    Ec=0.5,            # Charging energy [GHz]
    Gamma=15.0,        # Andreev coupling [GHz]
    delta_Gamma=0.1,   # Coupling asymmetry [GHz]
    er=0.0,            # Resonant level detuning [GHz]
    phase=0.0,         # External phase
    ng=0.0,            # Charge offset
    n_cut=25           # Charge basis cutoff
)
```

### Gatemonium

```python
from HybridSuperQubits import Gatemonium

qubit = Gatemonium(
    Ec=1.2,            # Charging energy [GHz]
    El=0.5,            # Inductive energy [GHz]
    Ej=8.0,            # Josephson energy [GHz]
    T=np.array([0.9]), # Transmission coefficient(s)
    phase=0.0,         # External flux
    dimension=100      # Hilbert space dimension
)
```

### FerBo (Fermionic-Bosonic) qubit

```python
from HybridSuperQubits import Ferbo

qubit = Ferbo(
    Ec=1.2,            # Charging energy [GHz]
    El=0.8,            # Inductive energy [GHz]
    Gamma=5.0,         # Andreev coupling strength [GHz]
    delta_Gamma=0.1,   # Coupling asymmetry [GHz]
    er=0.05,           # Resonant level detuning [GHz]
    phase=0.3,         # External flux (2pi Phi/Phi_0)
    dimension=100      # Hilbert space dimension
)

# Wavefunction visualization in the Andreev basis
qubit.plot_wavefunction(which=[0, 1], andreev_basis="Andreev")
```

### Noise analysis

```python
# T1 from capacitive losses
t1_cap = qubit.t1_capacitive(i=0, j=1, Q_cap=1e6, T=0.02)

# T1 from inductive losses
t1_ind = qubit.t1_inductive(i=0, j=1, Q_ind=500e6, T=0.02)

# Dephasing from 1/f flux noise
tphi = qubit.tphi_1_over_f_flux(A_noise=1e-6)

# Matrix elements for transition analysis
mel_table = qubit.matrixelement_table(
    operator=qubit.n_operator(),
    evecs=evecs,
    evals_count=4
)
```

### Parameter sweeps and visualization

```python
import numpy as np

# Spectrum vs flux
spectrum = qubit.get_spectrum_vs_paramvals(
    param_name="phase",
    param_vals=np.linspace(0, 2 * np.pi, 201),
    evals_count=6
)

# Matrix elements vs parameter
matelems = qubit.get_matelements_vs_paramvals(
    operator=qubit.n_operator(),
    param_name="phase",
    param_vals=np.linspace(0, 2 * np.pi, 101),
    evals_count=4
)

# Save / load results
spectrum.filewrite("spectrum_data.hdf5")
```

---

## Key features

- **7 qubit types** with a shared interface: Ferbo, Andreev, Fluxonium, Gatemon, Gatemonium, JJA, Resonator
- **Hamiltonian construction and diagonalization** using `scipy.linalg.eigh`
- **Parameter sweeps** over any circuit parameter (flux, charge offset, energies, transmission) with `get_spectrum_vs_paramvals`
- **Noise analysis**: capacitive and inductive T1, 1/f flux dephasing, flux bias line losses
- **Matrix element computation** for arbitrary operators across energy levels
- **Wavefunction visualization** with support for hybrid (Andreev + Fock) bases
- **Wigner function** computation via QuTiP integration
- **First and second Hamiltonian derivatives** with respect to all circuit parameters
- **HDF5 storage** for spectrum data, matrix elements, and coherence times
- **Unit conversions** between circuit parameters (inductance/capacitance) and energy scales

---

## Documentation

Full API reference and theory background:
**[hybridsuperqubits.readthedocs.io](https://hybridsuperqubits.readthedocs.io)**

---

## Citation

If you use HybridSuperQubits in your research, please cite the software:

**Software:**

```bibtex
@software{caceres2025hybridsuperqubits,
  author    = {Joan J. C{\'a}ceres},
  title     = {HybridSuperQubits},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.15773124},
  url       = {https://github.com/joanjcaceres/HybridSuperQubits}
}
```

If you use the FerBo qubit specifically, please also cite the paper:

```bibtex
@article{caceres2026ferbo,
  title     = {FerBo: a noise resilient qubit hybridizing Andreev and fluxonium states},
  author    = {Caceres, J. J. and Sanz Marco, D. and Ortuzar, J. and Flurin, E.
               and Urbina, C. and Pothier, H. and Goffman, M. F.
               and Matute-Ca{\~n}adas, F. J. and Levy Yeyati, A.},
  year      = {2026},
  eprint    = {2604.01145},
  archivePrefix = {arXiv},
  primaryClass  = {cond-mat.mes-hall},
  url       = {https://arxiv.org/abs/2604.01145}
}
```

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License. This project includes portions of code derived from
[scqubits](https://github.com/scqubits/scqubits) (BSD 3-Clause License).
See [LICENSE](./LICENSE) for details.
