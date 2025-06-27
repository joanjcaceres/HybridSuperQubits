"""
Benchmark tests for performance measurement.
"""

from typing import Any

import numpy as np
import pytest

from HybridSuperQubits import Ferbo, Fluxonium


class TestFluxoniumBenchmarks:
    """Benchmark tests for Fluxonium qubit performance."""

    def test_hamiltonian_calculation_speed(self, fluxonium: Fluxonium, benchmark):
        """Benchmark Hamiltonian calculation speed."""
        result = benchmark(fluxonium.hamiltonian)

        # Verify the result is valid
        assert result.shape == (fluxonium.dimension, fluxonium.dimension)
        assert np.allclose(result, result.conj().T, atol=1e-10)  # Hermitian

    def test_eigenvalue_calculation_speed(self, fluxonium: Fluxonium, benchmark):
        """Benchmark eigenvalue calculation speed."""
        evals_count = 10
        result = benchmark(fluxonium.eigenvals, evals_count)

        # Verify the result is valid
        assert result.shape == (evals_count,)
        assert np.allclose(result.imag, 0, atol=1e-10)  # Real eigenvalues
        assert np.all(np.diff(result) >= 0)  # Increasing order

    def test_eigensystem_calculation_speed(self, fluxonium: Fluxonium, benchmark):
        """Benchmark full eigensystem calculation speed."""
        evals_count = 8
        evals, evecs = benchmark(fluxonium.eigensys, evals_count)

        # Verify the results are valid
        assert evals.shape == (evals_count,)
        assert evecs.shape == (fluxonium.dimension, evals_count)
        assert np.allclose(evals.imag, 0, atol=1e-10)

    def test_parameter_sweep_speed(self, fluxonium: Fluxonium, benchmark):
        """Benchmark parameter sweep calculation speed."""
        phase_vals = np.linspace(0, 0.5, 10)
        evals_count = 5

        result = benchmark(
            fluxonium.get_spectrum_vs_paramvals,
            param_name="phase",
            param_vals=phase_vals,
            evals_count=evals_count,
            show_progress=False,
        )

        # Verify the result is valid
        assert result.energy_table.shape == (len(phase_vals), evals_count)

    def test_matrix_elements_speed(self, fluxonium: Fluxonium, benchmark):
        """Benchmark matrix elements calculation speed."""
        operators = ["n_operator", "phase_operator"]
        phase_vals = np.linspace(0, 0.3, 5)
        evals_count = 4

        result = benchmark(
            fluxonium.get_matelements_vs_paramvals,
            operators=operators,
            param_name="phase",
            param_vals=phase_vals,
            evals_count=evals_count,
            show_progress=False,
        )

        # Verify the result is valid
        for op in operators:
            assert op in result.matrixelem_table
            expected_shape = (len(phase_vals), evals_count, evals_count)
            assert result.matrixelem_table[op].shape == expected_shape


class TestFerboBenchmarks:
    """Benchmark tests for Ferbo qubit performance."""

    def test_hamiltonian_calculation_speed(self, ferbo: Ferbo, benchmark):
        """Benchmark Hamiltonian calculation speed."""
        result = benchmark(ferbo.hamiltonian)

        # Verify the result is valid
        assert result.shape == (ferbo.dimension, ferbo.dimension)
        assert np.allclose(result, result.conj().T, atol=1e-10)  # Hermitian

    def test_eigenvalue_calculation_speed(self, ferbo: Ferbo, benchmark):
        """Benchmark eigenvalue calculation speed."""
        evals_count = 8
        result = benchmark(ferbo.eigenvals, evals_count)

        # Verify the result is valid
        assert result.shape == (evals_count,)
        assert np.allclose(result.imag, 0, atol=1e-10)  # Real eigenvalues

    def test_jrl_potential_speed(self, ferbo: Ferbo, benchmark):
        """Benchmark JRL potential calculation speed."""
        result = benchmark(ferbo.jrl_potential)

        # Verify the result is valid
        assert result.shape == (ferbo.dimension, ferbo.dimension)
        assert np.allclose(result, result.conj().T, atol=1e-10)  # Hermitian

    def test_reduced_density_matrix_speed(self, ferbo: Ferbo, benchmark):
        """Benchmark reduced density matrix calculation speed."""
        result = benchmark(ferbo.reduced_density_matrix, which=0, subsys=0)

        # Verify the result is valid
        expected_shape = (ferbo.dimension // 2, ferbo.dimension // 2)
        assert result.shape == expected_shape
        assert np.allclose(result, result.conj().T, atol=1e-10)  # Hermitian


class TestOperatorBenchmarks:
    """Benchmark tests for operator calculations."""

    def test_n_operator_speed(self, fluxonium: Fluxonium, benchmark):
        """Benchmark n_operator calculation speed."""
        result = benchmark(fluxonium.n_operator)

        # Verify the result is valid
        expected_shape = (fluxonium.dimension, fluxonium.dimension)
        assert result.shape == expected_shape
        assert np.allclose(result, result.conj().T, atol=1e-10)  # Hermitian

    def test_phase_operator_speed(self, fluxonium: Fluxonium, benchmark):
        """Benchmark phase_operator calculation speed."""
        result = benchmark(fluxonium.phase_operator)

        # Verify the result is valid
        expected_shape = (fluxonium.dimension, fluxonium.dimension)
        assert result.shape == expected_shape
        assert np.allclose(result, result.conj().T, atol=1e-10)  # Hermitian


class TestScalingBenchmarks:
    """Benchmark tests for scaling with system size."""

    @pytest.mark.parametrize("dimension", [20, 30, 40, 50])
    def test_fluxonium_scaling_with_dimension(
        self, fluxonium_params: dict[str, Any], benchmark, dimension
    ):
        """Test how Fluxonium performance scales with dimension."""
        fluxonium_params["dimension"] = dimension
        qubit = Fluxonium(**fluxonium_params)

        def calculate_spectrum():
            return qubit.eigenvals(evals_count=5)

        result = benchmark(calculate_spectrum)

        # Verify the result is valid
        assert result.shape == (5,)
        assert np.allclose(result.imag, 0, atol=1e-10)

    @pytest.mark.parametrize("evals_count", [5, 10, 15, 20])
    def test_eigenvalue_count_scaling(
        self, fluxonium: Fluxonium, benchmark, evals_count
    ):
        """Test how eigenvalue calculation scales with number of eigenvalues requested."""

        def calculate_eigenvals():
            return fluxonium.eigenvals(evals_count)

        result = benchmark(calculate_eigenvals)

        # Verify the result is valid
        assert result.shape == (evals_count,)
        assert np.allclose(result.imag, 0, atol=1e-10)

    @pytest.mark.parametrize("sweep_length", [5, 10, 20])
    def test_parameter_sweep_scaling(
        self, fluxonium: Fluxonium, benchmark, sweep_length
    ):
        """Test how parameter sweep scales with number of parameter values."""
        phase_vals = np.linspace(0, 0.5, sweep_length)
        evals_count = 3

        def calculate_sweep():
            return fluxonium.get_spectrum_vs_paramvals(
                param_name="phase",
                param_vals=phase_vals,
                evals_count=evals_count,
                show_progress=False,
            )

        result = benchmark(calculate_sweep)

        # Verify the result is valid
        assert result.energy_table.shape == (sweep_length, evals_count)


class TestMemoryUsage:
    """Tests for memory efficiency."""

    def test_fluxonium_memory_efficiency(self, fluxonium_params: dict[str, Any]):
        """Test that Fluxonium doesn't use excessive memory."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Create multiple qubit instances
        qubits = []
        for i in range(10):
            params = fluxonium_params.copy()
            params["phase"] = i * 0.1
            qubits.append(Fluxonium(**params))

        # Calculate spectra
        for qubit in qubits:
            qubit.eigenvals(evals_count=5)

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 100 MB)
        assert memory_increase < 100 * 1024 * 1024

    def test_large_dimension_memory(self, fluxonium_params: dict[str, Any]):
        """Test memory usage with larger dimensions."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Create qubit with larger dimension
        fluxonium_params["dimension"] = 100
        qubit = Fluxonium(**fluxonium_params)

        # Calculate Hamiltonian and eigenvalues
        H = qubit.hamiltonian()
        evals = qubit.eigenvals(evals_count=10)

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable for the dimension
        # 100x100 complex matrix ≈ 160 KB, so total should be much less than 10 MB
        assert memory_increase < 10 * 1024 * 1024

        # Verify results are still valid
        assert H.shape == (100, 100)
        assert evals.shape == (10,)
        assert np.allclose(evals.imag, 0, atol=1e-10)
