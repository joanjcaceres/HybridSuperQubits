Installation
============

Requirements
------------

- Python >= 3.10, < 3.14
- Core: ``matplotlib``, ``tqdm``, ``h5py``
- Scientific (recommended): ``numpy >= 1.26``, ``scipy >= 1.5``
- Quantum: ``qutip >= 5.1``, ``scqubits >= 4.3``

Quick install
-------------

Install the full package with all dependencies:

.. code-block:: bash

   pip install HybridSuperQubits[full]

This installs NumPy, SciPy, QuTiP, and scqubits alongside the core package.

Minimal install
---------------

If you manage scientific dependencies separately (e.g., via conda):

.. code-block:: bash

   pip install HybridSuperQubits

Then install ``numpy``, ``scipy``, ``matplotlib``, ``qutip``, and ``scqubits`` through your preferred package manager.

Apple Silicon (M1/M2/M3/M4)
----------------------------

For optimal performance on Apple Silicon Macs, scientific libraries should be installed
through conda-forge to get native ARM builds:

.. code-block:: bash

   # Option 1: Use the provided environment file
   conda env create -f environment.yml
   conda activate hybridsuperqubits

   # Option 2: Manual setup
   conda create -n hybridsuperqubits python>=3.10
   conda activate hybridsuperqubits
   conda install -c conda-forge numpy scipy matplotlib qutip scqubits
   pip install -e . --no-deps

.. note::

   Using ``pip install`` on Apple Silicon may pull in non-optimized x86 wheels
   for NumPy and SciPy, resulting in significantly slower performance.

See `INSTALL_APPLE_SILICON.md <https://github.com/joanjcaceres/HybridSuperQubits/blob/main/INSTALL_APPLE_SILICON.md>`_ for a detailed guide.

Development install
-------------------

Clone the repository and install in editable mode:

.. code-block:: bash

   git clone https://github.com/joanjcaceres/HybridSuperQubits.git
   cd HybridSuperQubits
   pip install -e .[full]

To run tests:

.. code-block:: bash

   pytest tests/

Optional dependency groups
--------------------------

The package provides several extras for selective installation:

.. code-block:: bash

   # Full installation (all optional dependencies)
   pip install HybridSuperQubits[full]

   # Core scientific computing only (NumPy + SciPy)
   pip install HybridSuperQubits[scientific]

   # QuTiP ecosystem (QuTiP + scqubits)
   pip install HybridSuperQubits[qutip]
