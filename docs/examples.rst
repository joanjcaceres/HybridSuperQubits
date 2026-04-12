Examples
========

The ``examples/`` directory in the repository contains Jupyter notebooks
demonstrating various use cases.

Available notebooks
-------------------

**Fit frequencies JJA.ipynb**
   Fitting resonance frequencies of a Josephson junction array to experimental data
   using the :py:class:`~HybridSuperQubits.jja.JJA` class.

**fit_fluxonium.ipynb**
   Extracting circuit parameters (E_C, E_L, E_J) by fitting a fluxonium spectrum
   to measured transition frequencies.

**representation.ipynb**
   Wavefunction analysis and state representation for the FerBo qubit, including
   ballistic and Andreev bases.

**solving_rotation_ferbo.ipynb**
   Basis rotation techniques for the FerBo qubit and the effect of the Andreev
   transformation on the Andreev sector.

**circuit.ipynb**
   Hybrid circuit simulation example.

Running the examples
--------------------

.. code-block:: bash

   cd examples/
   jupyter notebook

Or open any ``.ipynb`` file in JupyterLab, VS Code, or your preferred notebook
environment.

.. note::

   The examples require the full set of dependencies. Install with
   ``pip install HybridSuperQubits[full]`` or use the conda environment.
