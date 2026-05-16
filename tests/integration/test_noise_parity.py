"""Parity tests: the QubitBase wrappers must produce numerically identical
results to direct calls of the new standalone ``HybridSuperQubits.noise``
functions.

This is the guardrail for the Phase A refactor of issue #17 — it pins down
that backward compatibility is exact, not "close enough".
"""

import numpy as np
import pytest

from HybridSuperQubits import noise


# ---------------------------------------------------------------------------
# Fluxonium parity
# ---------------------------------------------------------------------------


class TestFluxoniumNoiseParity:
    def test_t1_capacitive(self, fluxonium):
        evals, evecs = fluxonium.eigensys(evals_count=4)
        matelems = fluxonium.matrixelement_table(
            "n_operator", evecs=evecs, evals_count=4,
        )

        wrapper = fluxonium.t1_capacitive(
            i=1, j=0, esys=(evals, evecs), matrix_elements=matelems,
        )
        direct = noise.t1_capacitive(
            evals=evals, n_op_matelems=matelems, Ec=fluxonium.Ec,
        )
        assert wrapper == pytest.approx(direct, rel=1e-12)

    def test_t1_inductive(self, fluxonium):
        evals, evecs = fluxonium.eigensys(evals_count=4)
        matelems = fluxonium.matrixelement_table(
            "phase_operator", evecs=evecs, evals_count=4,
        )

        wrapper = fluxonium.t1_inductive(
            i=1, j=0, esys=(evals, evecs), matrix_elements=matelems,
        )
        direct = noise.t1_inductive(
            evals=evals, phase_op_matelems=matelems, El=fluxonium.El,
        )
        assert wrapper == pytest.approx(direct, rel=1e-12)

    def test_t1_flux_bias_line(self, fluxonium):
        evals, evecs = fluxonium.eigensys(evals_count=4)
        matelems = fluxonium.matrixelement_table(
            "d_hamiltonian_d_phase", evecs=evecs, evals_count=4,
        )

        wrapper = fluxonium.t1_flux_bias_line(
            i=1, j=0, esys=(evals, evecs), matrix_elements=matelems,
        )
        direct = noise.t1_flux_bias_line(
            evals=evals, dH_dphase_matelems=matelems,
        )
        assert wrapper == pytest.approx(direct, rel=1e-12)

    def test_tphi_1_over_f_first_order(self, fluxonium):
        evals, evecs = fluxonium.eigensys()
        dH = fluxonium.matrixelement_table("d_hamiltonian_d_phase", evecs=evecs)

        wrapper = fluxonium.tphi_1_over_f(
            A_noise=1e-6, noise_op="d_hamiltonian_d_phase", esys=(evals, evecs),
        )
        direct = noise.tphi_1_over_f(
            evals=evals, dH_dlambda_matelems=dH, A_noise=1e-6,
        )
        assert np.allclose(wrapper, direct, rtol=1e-12, atol=0)

    def test_tphi_cqps(self, fluxonium):
        evals, evecs = fluxonium.eigensys()
        disp = fluxonium.matrixelement_table("displacement_operator", evecs=evecs)

        wrapper = fluxonium.tphi_CQPS(esys=(evals, evecs))
        direct = noise.tphi_CQPS(
            evals=evals, displacement_op_matelems=disp, El=fluxonium.El,
        )
        # tphi_CQPS uses np.inf where rate is 0; use equal_nan-style compare.
        assert np.all(
            (np.isinf(wrapper) & np.isinf(direct))
            | np.isclose(wrapper, direct, rtol=1e-12)
        )


# ---------------------------------------------------------------------------
# Ferbo parity (one representative qubit beyond Fluxonium)
# ---------------------------------------------------------------------------


class TestFerboNoiseParity:
    def test_t1_capacitive(self, ferbo):
        evals, evecs = ferbo.eigensys(evals_count=4)
        matelems = ferbo.matrixelement_table(
            "n_operator", evecs=evecs, evals_count=4,
        )
        wrapper = ferbo.t1_capacitive(
            i=1, j=0, esys=(evals, evecs), matrix_elements=matelems,
        )
        direct = noise.t1_capacitive(
            evals=evals, n_op_matelems=matelems, Ec=ferbo.Ec,
        )
        assert wrapper == pytest.approx(direct, rel=1e-12)

    def test_tphi_1_over_f_second_order(self, ferbo):
        """Exercises the 2nd-order branch — the trickiest delegated path."""
        evals, evecs = ferbo.eigensys()
        dH = ferbo.matrixelement_table("d_hamiltonian_d_phase", evecs=evecs)
        d2H_bare = ferbo.d2_hamiltonian_d_phase2()

        wrapper = ferbo.tphi_1_over_f(
            A_noise=1e-6,
            noise_op=["d_hamiltonian_d_phase", "d2_hamiltonian_d_phase2"],
            esys=(evals, evecs),
        )
        direct = noise.tphi_1_over_f(
            evals=evals, dH_dlambda_matelems=dH, A_noise=1e-6,
            d2H_dlambda2_op=d2H_bare,
        )
        assert np.allclose(wrapper, direct, rtol=1e-12, atol=0)
