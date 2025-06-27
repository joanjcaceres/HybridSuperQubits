"""
Integration tests for parameter sweeps and spectrum calculations.
"""

from typing import Any

import numpy as np

from HybridSuperQubits import Ferbo, Fluxonium


class TestParameterSweeps:
    """Test parameter sweep functionality across different qubits."""

    def test_fluxonium_spectrum_vs_phase(self, fluxonium: Fluxonium):
        """Test Fluxonium spectrum calculation vs phase parameter."""
        phase_vals = np.linspace(0, 0.5, 5)
        evals_count = 3

        spectrum_data = fluxonium.get_spectrum_vs_paramvals(
            param_name="phase",
            param_vals=phase_vals,
            evals_count=evals_count,
            show_progress=False,
        )

        assert spectrum_data.energy_table.shape == (len(phase_vals), evals_count)
        assert spectrum_data.param_name == "phase"
        assert np.array_equal(spectrum_data.param_vals, phase_vals)

        # Check that energies are real and increasing
        for i in range(len(phase_vals)):
            energies = spectrum_data.energy_table[i, :]
            assert np.all(np.isreal(energies))
            assert np.all(np.diff(energies) >= 0)

    def test_fluxonium_spectrum_vs_ej(self, fluxonium: Fluxonium):
        """Test Fluxonium spectrum calculation vs Ej parameter."""
        ej_vals = np.linspace(1.0, 3.0, 4)
        evals_count = 4

        spectrum_data = fluxonium.get_spectrum_vs_paramvals(
            param_name="Ej",
            param_vals=ej_vals,
            evals_count=evals_count,
            show_progress=False,
        )

        assert spectrum_data.energy_table.shape == (len(ej_vals), evals_count)

        # Generally, higher Ej should lead to lower ground state energy
        ground_state_energies = spectrum_data.energy_table[:, 0]
        assert ground_state_energies[-1] <= ground_state_energies[0]

    def test_ferbo_spectrum_vs_gamma(self, ferbo: Ferbo):
        """Test Ferbo spectrum calculation vs Gamma parameter."""
        gamma_vals = np.linspace(1.0, 2.0, 3)
        evals_count = 3

        spectrum_data = ferbo.get_spectrum_vs_paramvals(
            param_name="Gamma",
            param_vals=gamma_vals,
            evals_count=evals_count,
            show_progress=False,
        )

        assert spectrum_data.energy_table.shape == (len(gamma_vals), evals_count)
        assert spectrum_data.param_name == "Gamma"

        # Check that energies are real and increasing for each parameter value
        for i in range(len(gamma_vals)):
            energies = spectrum_data.energy_table[i, :]
            assert np.all(np.isreal(energies))
            assert np.all(np.diff(energies) >= 0)

    def test_spectrum_subtract_ground(self, fluxonium: Fluxonium):
        """Test spectrum calculation with ground state subtraction."""
        phase_vals = np.linspace(0, 0.3, 3)
        evals_count = 4

        spectrum_data = fluxonium.get_spectrum_vs_paramvals(
            param_name="phase",
            param_vals=phase_vals,
            evals_count=evals_count,
            subtract_ground=True,
            show_progress=False,
        )

        # Ground state energies should be approximately zero
        ground_state_energies = spectrum_data.energy_table[:, 0]
        assert np.allclose(ground_state_energies, 0, atol=1e-10)

        # Excited state energies should be positive
        for i in range(len(phase_vals)):
            excited_energies = spectrum_data.energy_table[i, 1:]
            assert np.all(excited_energies >= 0)


class TestMatrixElements:
    """Test matrix element calculations."""

    def test_fluxonium_matrix_elements(self, fluxonium: Fluxonium):
        """Test matrix element calculation for Fluxonium operators."""
        evals_count = 3
        operators = ["n_operator", "phase_operator"]
        phase_vals = np.linspace(0, 0.2, 3)

        spectrum_data = fluxonium.get_matelements_vs_paramvals(
            operators=operators,
            param_name="phase",
            param_vals=phase_vals,
            evals_count=evals_count,
            show_progress=False,
        )

        # Check that matrix element tables exist for all operators
        for op in operators:
            assert op in spectrum_data.matrixelem_table
            me_table = spectrum_data.matrixelem_table[op]
            expected_shape = (len(phase_vals), evals_count, evals_count)
            assert me_table.shape == expected_shape

    def test_matrix_elements_hermiticity(self, fluxonium: Fluxonium):
        """Test that matrix elements preserve operator hermiticity."""
        evals_count = 3
        evals, evecs = fluxonium.eigensys(evals_count)

        # Test n_operator matrix elements
        me_n = fluxonium.matrixelement_table("n_operator", evecs, evals_count)
        assert np.allclose(me_n, me_n.conj().T, atol=1e-10)

        # Test phase_operator matrix elements
        me_phase = fluxonium.matrixelement_table("phase_operator", evecs, evals_count)
        assert np.allclose(me_phase, me_phase.conj().T, atol=1e-10)


class TestConsistencyChecks:
    """Test consistency between different calculation methods."""

    def test_eigenvals_vs_eigensys_consistency(self, fluxonium: Fluxonium):
        """Test that eigenvals and eigensys give consistent eigenvalues."""
        evals_count = 5

        evals_only = fluxonium.eigenvals(evals_count)
        evals_from_sys, _ = fluxonium.eigensys(evals_count)

        assert np.allclose(evals_only, evals_from_sys, atol=1e-12)

    def test_hamiltonian_eigenvalue_consistency(self, fluxonium: Fluxonium):
        """Test that Hamiltonian eigenvalues match eigensys results."""
        evals_count = 3
        H = fluxonium.hamiltonian()
        evals_from_sys, evecs = fluxonium.eigensys(evals_count)

        # Check that H @ evec = eval * evec for each eigenvector
        for i in range(evals_count):
            Hv = H @ evecs[:, i]
            ev = evals_from_sys[i] * evecs[:, i]
            assert np.allclose(Hv, ev, atol=1e-10)

    def test_parameter_restoration(self, fluxonium: Fluxonium):
        """Test that parameter sweeps restore original parameter values."""
        original_phase = fluxonium.phase
        original_ej = fluxonium.Ej

        # Run parameter sweep
        phase_vals = np.linspace(0, 0.5, 4)
        fluxonium.get_spectrum_vs_paramvals(
            param_name="phase",
            param_vals=phase_vals,
            evals_count=3,
            show_progress=False,
        )

        # Check that original values are restored
        assert abs(fluxonium.phase - original_phase) < 1e-12
        assert abs(fluxonium.Ej - original_ej) < 1e-12


class TestFluxGroupingConsistency:
    """Test consistency between different flux grouping methods."""

    def test_fluxonium_flux_groupings_at_zero_phase(
        self, fluxonium_params: dict[str, Any]
    ):
        """Test that both flux groupings give same result at zero phase."""
        # Set phase to zero
        fluxonium_params["phase"] = 0.0

        fluxonium_el = Fluxonium(**{**fluxonium_params, "flux_grouping": "EL"})
        fluxonium_ej = Fluxonium(**{**fluxonium_params, "flux_grouping": "EJ"})

        evals_el = fluxonium_el.eigenvals(evals_count=5)
        evals_ej = fluxonium_ej.eigenvals(evals_count=5)

        # At zero phase, both groupings should give identical results
        assert np.allclose(evals_el, evals_ej, atol=1e-10)

    def test_ferbo_flux_groupings_consistency(self, ferbo_params: dict[str, Any]):
        """Test that Ferbo works with different flux groupings."""
        # Test ABS grouping
        ferbo_abs = Ferbo(**{**ferbo_params, "flux_grouping": "ABS"})
        evals_abs = ferbo_abs.eigenvals(evals_count=4)

        # Test EL grouping
        ferbo_el = Ferbo(**{**ferbo_params, "flux_grouping": "EL"})
        evals_el = ferbo_el.eigenvals(evals_count=4)

        # Both should produce real eigenvalues
        assert np.allclose(evals_abs.imag, 0, atol=1e-10)
        assert np.allclose(evals_el.imag, 0, atol=1e-10)

        # And both should be in increasing order
        assert np.all(np.diff(evals_abs) >= 0)
        assert np.all(np.diff(evals_el) >= 0)
