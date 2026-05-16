Noise
=====

Standalone pure functions for decoherence calculations: spectral densities,
T1 relaxation, and Tphi dephasing. Each function takes eigenvalues, operator
matrix elements, and scalar parameters — no qubit object is required — so
they can be applied to any user-supplied Hamiltonian.

The corresponding methods on :py:class:`~HybridSuperQubits.qubit_base.QubitBase`
(``t1_capacitive``, ``t1_inductive``, ``t1_flux_bias_line``, ``tphi_1_over_f``,
``tphi_CQPS``) are thin wrappers that delegate to these functions.

.. automodule:: HybridSuperQubits.noise
   :members:
   :undoc-members:
   :show-inheritance:
