# HybridSuperQubits 🌀⚡

[![PyPI Version](https://img.shields.io/pypi/v/HybridSuperQubits)](https://pypi.org/project/HybridSuperQubits/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/hybridsuperqubits/badge/?version=latest)](https://hybridsuperqubits.readthedocs.io/en/latest/?badge=latest)

A Python framework for simulating hybrid semiconductor-superconductor quantum circuits, developed at Quantronics Group, CEA Paris-Saclay.

## Key Features ✨
- **Hybrid Qubit Modeling** 🔬  
  Simulate novel qubit architectures combining semiconductor and superconducting elements
- **Advanced Noise Analysis** 📉  
  Calculate decoherence times (T1, Tφ) from various noise sources
- **Professional Visualization** 📊  
  Built-in plotting tools for wavefunctions, energy spectra, and matrix elements
- **SC-Qubits Compatibility** 🔄  
  API-inspired interface for users familiar with scqubits

## Installation ⚙️
```bash
pip install HybridSuperQubits
```

## Basic Usage 🚀

```python
from HybridSuperQubits.ferbo import Ferbo

# Initialize a hybrid qubit
qubit = Ferbo(
    Ec=1.2,    # Charging energy (GHz)
    El=0.8,    # Inductive energy (GHz)
    Gamma=5.0, # Coupling strength (GHz)
    delta_Gamma = 0.1 # Coupling difference (GHz)
    dimension=100, # Dimension of the Fock space
    phase=0.3  # External flux (Φ₀)
)

# Calculate energy spectrum
evals = qubit.eigenvals(evals_count=5)
print(f"Energy levels: {evals} GHz")

# Plot wavefunction visualization
fig, ax = qubit.plot_wavefunction(which=0, plot_potential=True)
plt.show()
```

## Key Capabilities 🔍

### Spectrum Analysis

```python
# Calculate spectrum vs external flux
flux_values = np.linspace(0, 1, 100)
spectrum_data = qubit.get_spectrum_vs_paramvals('phase', flux_values)

# Plot energy spectrum
fig, ax = qubit.plot_evals_vs_paramvals(spectrum_data=spectrum_data)
```

### Decoherence Calculations
```python
# Calculate T1 times from various noise sources
noise_channels = ['capacitive', 'inductive', 'flux_bias_line']
t1_data = qubit.get_t1_vs_paramvals(noise_channels, 'phase', flux_values)

# Plot T1 results
fig, ax = qubit.plot_t1_vs_paramvals(noise_channels, spectrum_data=t1_data)
```

## Contributing 🤝

We welcome contributions! Please see:

[CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines

## License

This project is licensed under the MIT License. However, it includes portions of code derived from 
[scqubits](https://github.com/scqubits/scqubits), which is licensed under the BSD 3-Clause License.

For more details, please refer to the [`LICENSE`](./LICENSE) file.

---

Developed with ❤️ at Quantronics Group, CEA Paris-Saclay.

