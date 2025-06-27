"""
Unit tests for the Fluxonium qubit class.
"""

from typing import Any

import numpy as np
import pytest

from HybridSuperQubits import Fluxonium


class TestFluxoniumInitialization:
    """Test Fluxonium initialization and basic properties."""

    def test_initialization_with_valid_params(self, fluxonium_params: dict[str, Any]):
        """Test that Fluxonium initializes correctly with valid parameters."""
        qubit = Fluxonium(**fluxonium_params)

        assert qubit.Ec == fluxonium_params["Ec"]
        assert qubit.El == fluxonium_params["El"]
        assert qubit.Ej == fluxonium_params["Ej"]
        assert qubit.phase == fluxonium_params["phase"]
        assert qubit.dimension == fluxonium_params["dimension"]
        assert qubit.flux_grouping == fluxonium_params["flux_grouping"]

    def test_invalid_flux_grouping_raises_error(self, fluxonium_params: dict[str, Any]):
        """Test that invalid flux grouping raises ValueError."""
        fluxonium_params["flux_grouping"] = "INVALID"

        with pytest.raises(ValueError, match="Invalid flux grouping"):
            Fluxonium(**fluxonium_params)

    def test_phase_zpf_property(self, fluxonium: Fluxonium):
        """Test phase zero-point fluctuation calculation."""
        expected = (2 * fluxonium.Ec / fluxonium.El) ** 0.25
        assert abs(fluxonium.phase_zpf - expected) < 1e-12

    def test_n_zpf_property(self, fluxonium: Fluxonium):
        """Test charge zero-point fluctuation calculation."""
        expected = 0.5 * (fluxonium.El / 2 / fluxonium.Ec) ** 0.25
        assert abs(fluxonium.n_zpf - expected) < 1e-12

    def test_phi_osc_method(self, fluxonium: Fluxonium):
        """Test oscillator length calculation."""
        expected = (8.0 * fluxonium.Ec / fluxonium.El) ** 0.25
        assert abs(fluxonium.phi_osc() - expected) < 1e-12


class TestFluxoniumOperators:
    """Test Fluxonium operator methods."""

    def test_n_operator_shape(self, fluxonium: Fluxonium):
        """Test that n_operator returns correct shape."""
        n_op = fluxonium.n_operator()
        expected_shape = (fluxonium.dimension, fluxonium.dimension)
        assert n_op.shape == expected_shape

    def test_n_operator_is_hermitian(self, fluxonium: Fluxonium, tolerance: float):
        """Test that n_operator is Hermitian."""
        n_op = fluxonium.n_operator()
        assert np.allclose(n_op, n_op.conj().T, atol=tolerance)

    def test_phase_operator_shape(self, fluxonium: Fluxonium):
        """Test that phase_operator returns correct shape."""
        phase_op = fluxonium.phase_operator()
        expected_shape = (fluxonium.dimension, fluxonium.dimension)
        assert phase_op.shape == expected_shape

    def test_phase_operator_is_hermitian(self, fluxonium: Fluxonium, tolerance: float):
        """Test that phase_operator is Hermitian."""
        phase_op = fluxonium.phase_operator()
        assert np.allclose(phase_op, phase_op.conj().T, atol=tolerance)

    def test_commutation_relation(self, fluxonium: Fluxonium, loose_tolerance: float):
        """Test canonical commutation relation [n, φ] = -2i * n_zpf * phase_zpf.

        Note: In truncated Hilbert space, the relation may not hold exactly
        at the boundaries, so we test it for the interior part.
        """
        n_op = fluxonium.n_operator()
        phase_op = fluxonium.phase_operator()

        commutator = n_op @ phase_op - phase_op @ n_op
        expected = (
            -2j * fluxonium.n_zpf * fluxonium.phase_zpf * np.eye(fluxonium.dimension)
        )

        # Test the commutation relation for the bulk (avoid boundary effects)
        bulk_size = min(fluxonium.dimension - 5, 50)  # Test interior states
        assert np.allclose(
            commutator[:bulk_size, :bulk_size],
            expected[:bulk_size, :bulk_size],
            atol=loose_tolerance,
        )


class TestFluxoniumHamiltonian:
    """Test Fluxonium Hamiltonian methods."""

    def test_hamiltonian_shape(self, fluxonium: Fluxonium):
        """Test that Hamiltonian returns correct shape."""
        H = fluxonium.hamiltonian()
        expected_shape = (fluxonium.dimension, fluxonium.dimension)
        assert H.shape == expected_shape

    def test_hamiltonian_is_hermitian(self, fluxonium: Fluxonium, tolerance: float):
        """Test that Hamiltonian is Hermitian."""
        H = fluxonium.hamiltonian()
        assert np.allclose(H, H.conj().T, atol=tolerance)

    def test_hamiltonian_real_eigenvalues(self, fluxonium: Fluxonium, tolerance: float):
        """Test that Hamiltonian has real eigenvalues."""
        evals = fluxonium.eigenvals(evals_count=5)
        assert np.allclose(evals.imag, 0, atol=tolerance)

    def test_hamiltonian_increasing_eigenvalues(self, fluxonium: Fluxonium):
        """Test that eigenvalues are in increasing order."""
        evals = fluxonium.eigenvals(evals_count=10)
        assert np.all(np.diff(evals) >= 0)

    def test_hamiltonian_both_flux_groupings(
        self, fluxonium: Fluxonium, fluxonium_ej: Fluxonium
    ):
        """Test that both flux groupings produce valid Hamiltonians."""
        H_el = fluxonium.hamiltonian()
        H_ej = fluxonium_ej.hamiltonian()

        # Both should be Hermitian
        assert np.allclose(H_el, H_el.conj().T, atol=1e-10)
        assert np.allclose(H_ej, H_ej.conj().T, atol=1e-10)

        # Both should have real eigenvalues
        evals_el = fluxonium.eigenvals(evals_count=3)
        evals_ej = fluxonium_ej.eigenvals(evals_count=3)

        assert np.allclose(evals_el.imag, 0, atol=1e-10)
        assert np.allclose(evals_ej.imag, 0, atol=1e-10)


class TestFluxoniumDerivatives:
    """Test Fluxonium derivative methods."""

    def test_d_hamiltonian_d_EL_shape(self, fluxonium: Fluxonium):
        """Test derivative with respect to EL has correct shape."""
        dH_dEL = fluxonium.d_hamiltonian_d_EL()
        expected_shape = (fluxonium.dimension, fluxonium.dimension)
        assert dH_dEL.shape == expected_shape

    def test_d_hamiltonian_d_ng_shape(self, fluxonium: Fluxonium):
        """Test derivative with respect to ng has correct shape."""
        dH_dng = fluxonium.d_hamiltonian_d_ng()
        expected_shape = (fluxonium.dimension, fluxonium.dimension)
        assert dH_dng.shape == expected_shape

    def test_d_hamiltonian_d_phase_shape(self, fluxonium: Fluxonium):
        """Test derivative with respect to phase has correct shape."""
        dH_dphase = fluxonium.d_hamiltonian_d_phase()
        expected_shape = (fluxonium.dimension, fluxonium.dimension)
        assert dH_dphase.shape == expected_shape

    def test_d2_hamiltonian_d_ng2_shape(self, fluxonium: Fluxonium):
        """Test second derivative with respect to ng has correct shape."""
        d2H_dng2 = fluxonium.d2_hamiltonian_d_ng2()
        expected_shape = (fluxonium.dimension, fluxonium.dimension)
        assert d2H_dng2.shape == expected_shape

    def test_d2_hamiltonian_d_phase2_shape(self, fluxonium: Fluxonium):
        """Test second derivative with respect to phase has correct shape."""
        d2H_dphase2 = fluxonium.d2_hamiltonian_d_phase2()
        expected_shape = (fluxonium.dimension, fluxonium.dimension)
        assert d2H_dphase2.shape == expected_shape


class TestFluxoniumEigensystem:
    """Test Fluxonium eigensystem calculations."""

    def test_eigensys_returns_correct_shapes(self, fluxonium: Fluxonium):
        """Test that eigensys returns arrays of correct shapes."""
        evals_count = 5
        evals, evecs = fluxonium.eigensys(evals_count)

        assert evals.shape == (evals_count,)
        assert evecs.shape == (fluxonium.dimension, evals_count)

    def test_eigenvals_returns_correct_shape(self, fluxonium: Fluxonium):
        """Test that eigenvals returns array of correct shape."""
        evals_count = 8
        evals = fluxonium.eigenvals(evals_count)

        assert evals.shape == (evals_count,)

    def test_eigenvectors_are_normalized(self, fluxonium: Fluxonium, tolerance: float):
        """Test that eigenvectors are normalized."""
        evals_count = 3
        _, evecs = fluxonium.eigensys(evals_count)

        for i in range(evals_count):
            norm = np.linalg.norm(evecs[:, i])
            assert abs(norm - 1.0) < tolerance

    def test_eigenvectors_are_orthogonal(self, fluxonium: Fluxonium, tolerance: float):
        """Test that eigenvectors are orthogonal."""
        evals_count = 4
        _, evecs = fluxonium.eigensys(evals_count)

        # Check orthogonality
        overlap_matrix = evecs.conj().T @ evecs
        expected = np.eye(evals_count)

        assert np.allclose(overlap_matrix, expected, atol=tolerance)


class TestFluxoniumPotential:
    """Test Fluxonium potential method."""

    def test_potential_single_value(self, fluxonium: Fluxonium):
        """Test potential calculation for single phi value."""
        phi = 0.5
        V = fluxonium.potential(phi)

        assert isinstance(V, np.ndarray)
        assert V.shape == (1,)
        assert np.isfinite(V[0])

    def test_potential_array_input(self, fluxonium: Fluxonium):
        """Test potential calculation for array of phi values."""
        phi_array = np.linspace(-np.pi, np.pi, 10)
        V = fluxonium.potential(phi_array)

        assert V.shape == phi_array.shape
        assert np.all(np.isfinite(V))

    def test_potential_continuity(self, fluxonium: Fluxonium):
        """Test that potential is continuous."""
        phi_array = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
        V = fluxonium.potential(phi_array)

        # Check that there are no sudden jumps
        dV = np.diff(V)
        max_jump = np.max(np.abs(dV))

        # For a smooth potential, the maximum jump should be reasonable
        assert max_jump < 10.0  # Arbitrary threshold for continuity
