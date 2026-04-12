Quick start
===========

This guide walks through the basic workflow: creating a qubit, computing its
spectrum, and analyzing noise properties.

Creating a FerBo qubit
----------------------

.. code-block:: python

   import numpy as np
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

Computing eigenvalues and eigenstates
--------------------------------------

.. code-block:: python

   # First 6 eigenvalues only
   evals = qubit.eigenvals(evals_count=6)

   # Eigenvalues and eigenvectors
   evals, evecs = qubit.eigensys(evals_count=6)

   print(f"Transition frequency E01 = {evals[1] - evals[0]:.4f} GHz")

Energy spectrum vs flux
-----------------------

Sweep the external flux and compute the energy levels:

.. code-block:: python

   flux_values = np.linspace(0, 2 * np.pi, 201)

   spectrum = qubit.get_spectrum_vs_paramvals(
       param_name="phase",
       param_vals=flux_values,
       evals_count=6,
       subtract_ground=True
   )

The returned :py:class:`~HybridSuperQubits.storage.SpectrumData` object holds
the eigenvalues, eigenstates, and parameter values. Use ``subtract_ground=True``
to reference energies to the ground state.

Wavefunction visualization
--------------------------

Plot the ground and first excited state wavefunctions:

.. code-block:: python

   # Ballistic Andreev basis
   qubit.plot_wavefunction(which=[0, 1], andreev_basis="ballistic")

   # Andreev (rotated) basis
   qubit.plot_wavefunction(which=[0, 1], andreev_basis="Andreev")

Matrix elements
---------------

Compute transition matrix elements for the charge operator:

.. code-block:: python

   evals, evecs = qubit.eigensys(evals_count=4)

   mel = qubit.matrixelement_table(
       operator=qubit.n_operator(),
       evecs=evecs,
       evals_count=4
   )

   print(f"|<0|n|1>|^2 = {abs(mel[0, 1])**2:.6e}")

Noise analysis
--------------

Compute coherence times for the 0-1 transition:

.. code-block:: python

   # Capacitive T1
   t1_cap = qubit.t1_capacitive(i=0, j=1, Q_cap=1e6, T=0.02)

   # Inductive T1
   t1_ind = qubit.t1_inductive(i=0, j=1, Q_ind=500e6, T=0.02)

   # 1/f flux dephasing
   tphi = qubit.tphi_1_over_f_flux(A_noise=1e-6)

   print(f"T1 (capacitive) = {t1_cap:.2e} s")
   print(f"T1 (inductive)  = {t1_ind:.2e} s")
   print(f"T_phi (flux)    = {tphi:.2e} s")

Saving and loading results
--------------------------

Spectrum data can be saved to HDF5 files:

.. code-block:: python

   # Save
   spectrum.filewrite("my_spectrum.hdf5")

   # Load
   from HybridSuperQubits.storage import SpectrumData
   loaded = SpectrumData.read("my_spectrum.hdf5")

Other qubit types
-----------------

All qubit classes share the same interface inherited from
:py:class:`~HybridSuperQubits.qubit_base.QubitBase`:

.. code-block:: python

   from HybridSuperQubits import Fluxonium, Gatemon, Gatemonium, Andreev

   # Fluxonium
   flux_qubit = Fluxonium(Ec=1.0, El=0.5, Ej=4.0, phase=np.pi, dimension=100)

   # Gatemon
   gatemon = Gatemon(Ec=0.5, Delta=44.0, T=0.95, ng=0.0, n_cut=30)

   # Gatemonium (supports multiple channels)
   gatemonium = Gatemonium(
       Ec=1.2, El=0.5, Ej=8.0,
       T=np.array([0.9, 0.85]),
       phase=0.0, dimension=100
   )

   # Andreev pair qubit
   andreev = Andreev(
       Ec=0.5, Gamma=15.0, delta_Gamma=0.1,
       er=0.0, phase=0.0, ng=0.0, n_cut=25
   )

   # All support the same methods:
   evals = flux_qubit.eigenvals(evals_count=6)
   spectrum = gatemon.get_spectrum_vs_paramvals("ng", np.linspace(0, 1, 101), 6)

Unit conversions
----------------

Convert between circuit parameters and energy scales:

.. code-block:: python

   from HybridSuperQubits.utilities import L_to_El, El_to_L, C_to_Ec, Ec_to_C

   # Inductance (H) <-> Inductive energy (GHz)
   El = L_to_El(100e-9)        # 100 nH -> El in GHz
   L = El_to_L(0.5)            # 0.5 GHz -> L in Henries

   # Capacitance (F) <-> Charging energy (GHz)
   Ec = C_to_Ec(10e-15)        # 10 fF -> Ec in GHz
   C = Ec_to_C(1.0)            # 1.0 GHz -> C in Farads
