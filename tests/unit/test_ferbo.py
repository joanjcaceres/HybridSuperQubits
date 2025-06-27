"""
Unit tests for the Ferbo qubit class.
"""

from typing import Any

import numpy as np
import pytest

from HybridSuperQubits import Ferbo


class TestFerboInitialization:
    """Test Ferbo initialization and basic properties."""

    def test_initialization_with_valid_params(self, ferbo_params: dict[str, Any]):
        """Test that Ferbo initializes correctly with valid parameters."""
        qubit = Ferbo(**ferbo_params)

        assert qubit.Ec == ferbo_params["Ec"]
        assert qubit.El == ferbo_params["El"]
        assert qubit.Ej == ferbo_params["Ej"]
        assert qubit.Gamma == ferbo_params["Gamma"]
        assert qubit.delta_Gamma == ferbo_params["delta_Gamma"]
        assert qubit.er == ferbo_params["er"]
        assert qubit.phase == ferbo_params["phase"]
        assert qubit.dimension == ferbo_params["dimension"]
        assert qubit.flux_grouping == ferbo_params["flux_grouping"]

    def test_invalid_flux_grouping_raises_error(self, ferbo_params: dict[str, Any]):
        """Test that invalid flux grouping raises ValueError."""
        ferbo_params["flux_grouping"] = "INVALID"

        with pytest.raises(ValueError, match="Invalid flux grouping"):
            Ferbo(**ferbo_params)

    def test_dimension_is_even(self, ferbo_params: dict[str, Any]):
        """Test that dimension is automatically made even."""
        ferbo_params["dimension"] = 201  # Odd number
        qubit = Ferbo(**ferbo_params)

        assert qubit.dimension == 200  # Should be made even

    def test_phase_zpf_property(self, ferbo: Ferbo):
        """Test phase zero-point fluctuation calculation."""
        expected = (2 * ferbo.Ec / ferbo.El) ** 0.25
        assert abs(ferbo.phase_zpf - expected) < 1e-12

    def test_n_zpf_property(self, ferbo: Ferbo):
        """Test charge zero-point fluctuation calculation."""
        expected = 0.5 * (ferbo.El / 2 / ferbo.Ec) ** 0.25
        assert abs(ferbo.n_zpf - expected) < 1e-12

    def test_lc_energy_property(self, ferbo: Ferbo):
        """Test LC energy calculation."""
        expected = np.sqrt(8 * ferbo.Ec * ferbo.El)
        assert abs(ferbo.lc_energy - expected) < 1e-12

    def test_transparency_property(self, ferbo: Ferbo):
        """Test transparency calculation."""
        expected = (ferbo.Gamma**2 - ferbo.delta_Gamma**2) / (
            ferbo.Gamma**2 + ferbo.er**2
        )
        assert abs(ferbo.transparency - expected) < 1e-12


class TestFerboOperators:
    """Test Ferbo operator methods."""

    def test_n_operator_shape(self, ferbo: Ferbo):
        """Test that n_operator returns correct shape."""
        n_op = ferbo.n_operator()
        expected_shape = (ferbo.dimension, ferbo.dimension)
        assert n_op.shape == expected_shape

    def test_phase_operator_shape(self, ferbo: Ferbo):
        """Test that phase_operator returns correct shape."""
        phase_op = ferbo.phase_operator()
        expected_shape = (ferbo.dimension, ferbo.dimension)
        assert phase_op.shape == expected_shape

    def test_jrl_potential_shape(self, ferbo: Ferbo):
        """Test that JRL potential returns correct shape."""
        jrl_pot = ferbo.jrl_potential()
        expected_shape = (ferbo.dimension, ferbo.dimension)
        assert jrl_pot.shape == expected_shape

    def test_jrl_potential_is_hermitian(self, ferbo: Ferbo, tolerance: float):
        """Test that JRL potential is Hermitian."""
        jrl_pot = ferbo.jrl_potential()
        assert np.allclose(jrl_pot, jrl_pot.conj().T, atol=tolerance)


class TestFerboHamiltonian:
    """Test Ferbo Hamiltonian methods."""

    def test_hamiltonian_shape(self, ferbo: Ferbo):
        """Test that Hamiltonian returns correct shape."""
        H = ferbo.hamiltonian()
        expected_shape = (ferbo.dimension, ferbo.dimension)
        assert H.shape == expected_shape

    def test_hamiltonian_is_hermitian(self, ferbo: Ferbo, tolerance: float):
        """Test that Hamiltonian is Hermitian."""
        H = ferbo.hamiltonian()
        assert np.allclose(H, H.conj().T, atol=tolerance)

    def test_hamiltonian_real_eigenvalues(self, ferbo: Ferbo, tolerance: float):
        """Test that Hamiltonian has real eigenvalues."""
        evals = ferbo.eigenvals(evals_count=5)
        assert np.allclose(evals.imag, 0, atol=tolerance)

    def test_hamiltonian_increasing_eigenvalues(self, ferbo: Ferbo):
        """Test that eigenvalues are in increasing order."""
        evals = ferbo.eigenvals(evals_count=8)
        assert np.all(np.diff(evals) >= 0)


class TestFerboDerivatives:
    """Test Ferbo derivative methods."""

    def test_d_hamiltonian_d_EC_shape(self, ferbo: Ferbo):
        """Test derivative with respect to EC has correct shape."""
        dH_dEC = ferbo.d_hamiltonian_d_EC()
        expected_shape = (ferbo.dimension, ferbo.dimension)
        assert dH_dEC.shape == expected_shape

    def test_d_hamiltonian_d_EL_shape(self, ferbo: Ferbo):
        """Test derivative with respect to EL has correct shape."""
        dH_dEL = ferbo.d_hamiltonian_d_EL()
        expected_shape = (ferbo.dimension, ferbo.dimension)
        assert dH_dEL.shape == expected_shape

    def test_d_hamiltonian_d_EJ_shape(self, ferbo: Ferbo):
        """Test derivative with respect to EJ has correct shape."""
        dH_dEJ = ferbo.d_hamiltonian_d_EJ()
        expected_shape = (ferbo.dimension, ferbo.dimension)
        assert dH_dEJ.shape == expected_shape

    def test_d_hamiltonian_d_Gamma_shape(self, ferbo: Ferbo):
        """Test derivative with respect to Gamma has correct shape."""
        dH_dGamma = ferbo.d_hamiltonian_d_Gamma()
        expected_shape = (ferbo.dimension, ferbo.dimension)
        assert dH_dGamma.shape == expected_shape

    def test_d_hamiltonian_d_er_shape(self, ferbo: Ferbo):
        """Test derivative with respect to er has correct shape."""
        dH_der = ferbo.d_hamiltonian_d_er()
        expected_shape = (ferbo.dimension, ferbo.dimension)
        assert dH_der.shape == expected_shape

    def test_d_hamiltonian_d_deltaGamma_shape(self, ferbo: Ferbo):
        """Test derivative with respect to delta_Gamma has correct shape."""
        dH_ddG = ferbo.d_hamiltonian_d_deltaGamma()
        expected_shape = (ferbo.dimension, ferbo.dimension)
        assert dH_ddG.shape == expected_shape

    def test_d_hamiltonian_d_phase_shape(self, ferbo: Ferbo):
        """Test derivative with respect to phase has correct shape."""
        dH_dphase = ferbo.d_hamiltonian_d_phase()
        expected_shape = (ferbo.dimension, ferbo.dimension)
        assert dH_dphase.shape == expected_shape


class TestFerboEigensystem:
    """Test Ferbo eigensystem calculations."""

    def test_eigensys_returns_correct_shapes(self, ferbo: Ferbo):
        """Test that eigensys returns arrays of correct shapes."""
        evals_count = 4
        evals, evecs = ferbo.eigensys(evals_count)

        assert evals.shape == (evals_count,)
        assert evecs.shape == (ferbo.dimension, evals_count)

    def test_eigenvals_returns_correct_shape(self, ferbo: Ferbo):
        """Test that eigenvals returns array of correct shape."""
        evals_count = 6
        evals = ferbo.eigenvals(evals_count)

        assert evals.shape == (evals_count,)

    def test_eigenvectors_are_normalized(self, ferbo: Ferbo, tolerance: float):
        """Test that eigenvectors are normalized."""
        evals_count = 3
        _, evecs = ferbo.eigensys(evals_count)

        for i in range(evals_count):
            norm = np.linalg.norm(evecs[:, i])
            assert abs(norm - 1.0) < tolerance

    def test_eigenvectors_are_orthogonal(self, ferbo: Ferbo, tolerance: float):
        """Test that eigenvectors are orthogonal."""
        evals_count = 4
        _, evecs = ferbo.eigensys(evals_count)

        # Check orthogonality
        overlap_matrix = evecs.conj().T @ evecs
        expected = np.eye(evals_count)

        assert np.allclose(overlap_matrix, expected, atol=tolerance)


class TestFerboSpecialMethods:
    """Test Ferbo special methods like reduced density matrix and wavefunction."""

    def test_reduced_density_matrix_shape(self, ferbo: Ferbo):
        """Test that reduced density matrix has correct shape."""
        rho_reduced = ferbo.reduced_density_matrix(which=0, subsys=0)
        expected_shape = (ferbo.dimension // 2, ferbo.dimension // 2)
        assert rho_reduced.shape == expected_shape

    def test_reduced_density_matrix_is_hermitian(self, ferbo: Ferbo, tolerance: float):
        """Test that reduced density matrix is Hermitian."""
        rho_reduced = ferbo.reduced_density_matrix(which=0, subsys=0)
        assert np.allclose(rho_reduced, rho_reduced.conj().T, atol=tolerance)

    def test_reduced_density_matrix_trace_positive(self, ferbo: Ferbo):
        """Test that reduced density matrix has positive trace."""
        rho_reduced = ferbo.reduced_density_matrix(which=0, subsys=0)
        trace = np.trace(rho_reduced)
        assert trace.real > 0
        assert abs(trace.imag) < 1e-10

    def test_wavefunction_returns_dict(self, ferbo: Ferbo):
        """Test that wavefunction returns a dictionary with expected keys."""
        phi_grid = np.linspace(-2 * np.pi, 2 * np.pi, 50)
        wf_data = ferbo.wavefunction(which=0, phi_grid=phi_grid)

        assert isinstance(wf_data, dict)
        assert "basis_labels" in wf_data
        assert "amplitudes" in wf_data
        assert "energy" in wf_data

    def test_wavefunction_amplitudes_shape(self, ferbo: Ferbo):
        """Test that wavefunction amplitudes have correct shape."""
        phi_grid = np.linspace(-2 * np.pi, 2 * np.pi, 50)
        wf_data = ferbo.wavefunction(which=0, phi_grid=phi_grid)

        # Should return amplitudes for both Andreev states
        assert wf_data["amplitudes"].shape == (2, len(phi_grid))
