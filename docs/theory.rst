Theory background
=================

This page provides a brief overview of the physics behind the qubit types implemented
in HybridSuperQubits. For a detailed treatment, see the companion paper:

   J. J. Caceres *et al.*, "FerBo: a noise resilient qubit hybridizing Andreev and
   fluxonium states", `arXiv:2604.01145 <https://arxiv.org/abs/2604.01145>`_ (2026).

The FerBo qubit
---------------

The **FerBo** (Fermionic-Bosonic) qubit is a superconducting circuit consisting of
a large inductance, a small capacitor, and a highly transmitting Josephson weak link
arranged in parallel. It can be viewed as a fluxonium qubit in which the standard
tunnel junction is replaced by a semiconductor weak link hosting Andreev bound states.

The circuit Hamiltonian is:

.. math::

   \hat{H} = 4E_C \hat{n}^2 + \tfrac{1}{2} E_L \hat{\varphi}^2 + H_{\mathrm{WL}}(\hat{\varphi} - \varphi_{\mathrm{ext}})

where :math:`E_C` is the charging energy, :math:`E_L` is the inductive energy,
:math:`\hat{n}` is the charge number operator, :math:`\hat{\varphi}` is the phase
operator, and :math:`\varphi_{\mathrm{ext}} = 2\pi \Phi / \Phi_0` is the external flux.

The weak link term is:

.. math::

   H_{\mathrm{WL}}(\hat{\varphi}) = \Gamma \cos(\hat{\varphi}/2)\,\hat{\sigma}_x
   - \delta\Gamma \sin(\hat{\varphi}/2)\,\hat{\sigma}_y
   + \varepsilon_r \,\hat{\sigma}_z

Here :math:`\Gamma` is the coupling to superconducting leads, :math:`\delta\Gamma`
is the coupling asymmetry, and :math:`\varepsilon_r` is the resonant level energy
(Fermi detuning). The Pauli matrices :math:`\hat{\sigma}_{x,y,z}` act on the
two-dimensional Andreev subspace spanned by the states :math:`|{-}\rangle` and
:math:`|{+}\rangle`.

The full Hilbert space is a tensor product of the Fock space (bosonic LC mode,
dimension :math:`N`) and the Andreev subspace (dimension 2), giving a total dimension
of :math:`2N`.

Protection mechanism
~~~~~~~~~~~~~~~~~~~~

The FerBo qubit achieves simultaneous protection against both dominant noise channels:

**Relaxation protection.**
In the high-impedance regime (:math:`Z/R_Q \gg 1`), the ground state
:math:`|0\rangle` and first excited state :math:`|1\rangle` have support in
*different* Andreev sectors (:math:`|{-}\rangle` and :math:`|{+}\rangle`
respectively). Because the charge operator :math:`\hat{n}` does not couple
different Andreev sectors, the matrix element :math:`|\langle 0|\hat{n}|1\rangle|^2`
is exponentially suppressed, protecting against charge-induced relaxation.

**Dephasing protection.**
As in the heavy fluxonium, the wavefunctions delocalize across the phase variable
:math:`\varphi` when the impedance is large. This delocalization reduces the
sensitivity to flux noise, suppressing the curvature
:math:`\partial^2 E_{01}/\partial \varphi_{\mathrm{ext}}^2`.

The protected regime is bounded by :math:`Z/R_Q \gtrsim 2E_C / (\pi \varepsilon_r)`.

Berry phase correction
~~~~~~~~~~~~~~~~~~~~~~

In the adiabatic representation (where the Andreev basis rotates with the phase),
a scalar Berry phase correction arises. This is implemented as
:py:meth:`~HybridSuperQubits.ferbo.Ferbo._berry_contribution` and can be toggled
via the ``include_berry`` flag in :py:meth:`~HybridSuperQubits.ferbo.Ferbo.potential`.

Flux grouping schemes
~~~~~~~~~~~~~~~~~~~~~

The FerBo Hamiltonian supports two flux grouping modes controlled by the
``flux_grouping`` parameter:

- ``"ABS"``: External flux is absorbed into the weak link term
  :math:`H_{\mathrm{WL}}(\hat{\varphi} - \varphi_{\mathrm{ext}})`.
- ``"EL"``: External flux appears in the inductive term
  :math:`\tfrac{1}{2}E_L(\hat{\varphi} - \varphi_{\mathrm{ext}})^2`.

Both are physically equivalent but may offer numerical advantages in different
parameter regimes.


Andreev pair qubit
------------------

The Andreev pair qubit describes pure Andreev bound states in a superconducting weak
link, discretized in the charge basis. The Hamiltonian contains:

- A charging term :math:`4E_C (\hat{n} - n_g)^2`
- The Josephson resonance level (JRL) potential from the weak link

The charge basis is a half-charge basis with states :math:`|n\rangle` where
:math:`n` runs from :math:`-n_{\mathrm{cut}}` to :math:`n_{\mathrm{cut}}`.


Fluxonium
---------

The fluxonium qubit is an LC circuit shunted by a Josephson junction:

.. math::

   \hat{H} = 4E_C \hat{n}^2
   + \tfrac{1}{2}E_L \hat{\varphi}^2
   - E_J \cos(\hat{\varphi} - \varphi_{\mathrm{ext}})

The large inductance :math:`E_L` lifts the ground-state degeneracy of the transmon,
producing a highly anharmonic spectrum. In the heavy regime (:math:`E_L \ll E_C`),
the wavefunctions delocalize in phase space, providing protection against flux noise.


Gatemon
-------

The gatemon replaces the tunnel junction with a gate-tunable semiconductor weak link.
The junction potential depends on the transmission coefficient :math:`\tau`, which is
controlled by a gate voltage:

.. math::

   E_{\mathrm{ABS}}(\varphi) = -\Delta \sqrt{1 - \tau \sin^2(\varphi/2)}

The Hamiltonian is discretized in the charge basis with a Fourier-expanded ABS potential.


Gatemonium
----------

The gatemonium combines a gate-tunable weak link with an inductive shunt, analogous
to the fluxonium but with a semiconductor junction. It supports multiple transmission
channels, each contributing independently to the junction potential.

The Hamiltonian:

.. math::

   \hat{H} = 4E_C \hat{n}^2
   + \tfrac{1}{2}E_L \hat{\varphi}^2
   + \sum_i V_{\mathrm{ABS}}^{(i)}(\hat{\varphi} - \varphi_{\mathrm{ext}})

where the sum runs over the transmission channels with individual transmissions
:math:`\tau_i`.


Josephson junction array (JJA)
------------------------------

The JJA module models an array of :math:`N` Josephson junctions as a metamaterial.
It computes:

- Plasma frequency :math:`\omega_p = 1/\sqrt{L_J C_J}`
- Resonance mode frequencies from the dispersion relation
- Group velocity
- Kerr nonlinearity matrix


Resonator
---------

A quantum LC resonator with the simple harmonic oscillator Hamiltonian:

.. math::

   \hat{H} = \hbar \omega \left(\hat{a}^\dagger \hat{a} + \tfrac{1}{2}\right)

Provides creation/annihilation operators, position/momentum (voltage/current)
operators, and zero-point fluctuation scales.


Noise channels
--------------

All qubits inheriting from ``QubitBase`` have access to built-in noise analysis
methods:

- **Capacitive losses** (:math:`T_1`): dielectric loss in the capacitor,
  parameterized by quality factor :math:`Q_{\mathrm{cap}}`.
- **Inductive losses** (:math:`T_1`): resistive losses in the inductor,
  parameterized by :math:`Q_{\mathrm{ind}}`.
- **Flux bias line coupling** (:math:`T_1`): coupling to an external flux line
  with mutual inductance :math:`M` and impedance :math:`Z`.
- **1/f flux noise** (:math:`T_\varphi`): dephasing from low-frequency flux
  fluctuations with amplitude :math:`A_\Phi`.

These are computed from Fermi's golden rule using the matrix elements of the
relevant noise operators between eigenstates.
