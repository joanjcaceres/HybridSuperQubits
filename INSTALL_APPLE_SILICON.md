# Installation Guide for Apple Silicon (M1/M2/M3) Macs

## Recommended Installation Method (Apple Silicon)

For optimal performance on Apple Silicon Macs, we recommend using conda to install scientific dependencies:

### Step 1: Create conda environment
```bash
# Create environment from environment.yml
conda env create -f environment.yml

# Or create manually
conda create -n hybridsuperqubits python>=3.9
conda activate hybridsuperqubits
```

### Step 2: Install scientific dependencies via conda-forge
```bash
conda install -c conda-forge numpy scipy matplotlib qutip scqubits
```

### Step 3: Install HybridSuperQubits
```bash
# Install HybridSuperQubits in development mode
pip install -e . --no-deps
```

## Alternative Installation Methods

### Method 1: Using pip with all dependencies
```bash
pip install -e .[full]
```
⚠️ **Warning**: This will install scipy from PyPI, which may not be optimized for Apple Silicon.

### Method 2: Minimal installation
```bash
# Install only core dependencies, then manually install scientific packages
pip install -e .
conda install -c conda-forge numpy scipy matplotlib qutip scqubits
```

## Troubleshooting

### If you accidentally installed scipy from pip:
```bash
# Uninstall pip version
pip uninstall scipy

# Reinstall from conda-forge
conda install -c conda-forge scipy
```

### Verify optimized installation:
```bash
python -c "import numpy as np; np.show_config()"
```
Look for `openblas` configuration and `arm64` in the machine information.

## Performance Notes

- **conda-forge packages** are compiled with optimized BLAS/LAPACK libraries for Apple Silicon
- **pip packages** may use generic builds that don't leverage Apple's Accelerate framework
- JAX from conda-forge includes optimized XLA compilation for Apple Silicon
