#!/usr/bin/env python
"""
Generate updated regression values with realistic physical parameters.

This utility script regenerates reference values for regression tests after
significant changes to the physics implementation or when updating to more
realistic parameter ranges. The generated values are manually copied into
the regression test files.

Usage:
    python generate_regression_values.py

Note: This script was used to update regression values after removing JAX
dependencies and switching to realistic physical parameters.
"""

import numpy as np

from HybridSuperQubits import Ferbo, Fluxonium


def generate_regression_values():
    """Generate all regression values with realistic parameters."""

    print("Generating regression values with realistic parameters...")
    print("=" * 60)

    # 1. Fluxonium EL standard params (realistic values)
    print("\n1. Fluxonium EL standard params:")
    flux_el_params = {
        "Ec": 2.0,  # 2 GHz - realistic
        "El": 0.15,  # 0.15 GHz - realistic
        "Ej": 5.0,  # 5 GHz - realistic
        "phase": 0.25,  # π/2 phase bias
        "dimension": 100,
        "flux_grouping": "EL",
    }

    flux_el = Fluxonium(**flux_el_params)
    evals_el = flux_el.eigenvals(evals_count=5)
    print(f"   Parameters: {flux_el_params}")
    print(f"   Eigenvalues: {evals_el}")
    print(f"   Array format: {list(evals_el)}")

    # 2. Fluxonium EJ standard params (same physical params, different grouping)
    print("\n2. Fluxonium EJ standard params:")
    flux_ej_params = {
        "Ec": 2.0,
        "El": 0.15,
        "Ej": 5.0,
        "phase": 0.25,
        "dimension": 100,
        "flux_grouping": "EJ",
    }

    flux_ej = Fluxonium(**flux_ej_params)
    evals_ej = flux_ej.eigenvals(evals_count=5)
    print(f"   Parameters: {flux_ej_params}")
    print(f"   Eigenvalues: {evals_ej}")
    print(f"   Array format: {list(evals_ej)}")

    # 3. Fluxonium zero phase (symmetric point)
    print("\n3. Fluxonium zero phase:")
    flux_zero_params = {
        "Ec": 3.0,  # Higher Ec
        "El": 0.2,  # Slightly higher El
        "Ej": 8.0,  # Higher Ej
        "phase": 0.0,  # Zero phase - symmetric
        "dimension": 100,
        "flux_grouping": "EL",
    }

    flux_zero = Fluxonium(**flux_zero_params)
    evals_zero = flux_zero.eigenvals(evals_count=4)
    print(f"   Parameters: {flux_zero_params}")
    print(f"   Eigenvalues: {evals_zero}")
    print(f"   Array format: {list(evals_zero)}")

    # 4. Fluxonium spectrum vs phase
    print("\n4. Fluxonium spectrum vs phase:")
    flux_sweep_params = {
        "Ec": 1.5,
        "El": 0.12,
        "Ej": 4.0,
        "phase": 0.0,  # Will be varied
        "dimension": 100,  # Increased for better accuracy
        "flux_grouping": "EL",
    }

    flux_sweep = Fluxonium(**flux_sweep_params)
    phase_vals = np.linspace(0, 0.5, 3)  # [0, 0.25, 0.5]

    spectrum_data = flux_sweep.get_spectrum_vs_paramvals(
        param_name="phase",
        param_vals=phase_vals,
        evals_count=3,
        show_progress=False,
    )

    print(f"   Parameters: {flux_sweep_params}")
    print(f"   Phase values: {phase_vals}")
    print(f"   Spectrum shape: {spectrum_data.energy_table.shape}")
    print("   Spectrum data:")
    for i, phase in enumerate(phase_vals):
        print(f"     Phase {phase:.3f}: {spectrum_data.energy_table[i]}")
    print("   Array format:")
    print(f"   {spectrum_data.energy_table.tolist()}")

    # 5. Ferbo ABS standard params (realistic values)
    print("\n5. Ferbo ABS standard params:")
    ferbo_abs_params = {
        "Ec": 3.0,  # 3 GHz - realistic for Ferbo
        "El": 0.15,  # 0.15 GHz - realistic
        "Ej": 6.0,  # 6 GHz - realistic
        "Gamma": 1.8,  # Andreev level coupling
        "delta_Gamma": 0.2,  # Asymmetry
        "er": 0.4,  # Environmental resistance
        "phase": 0.3,  # Phase bias
        "dimension": 200,
        "flux_grouping": "ABS",
    }

    ferbo_abs = Ferbo(**ferbo_abs_params)
    evals_ferbo_abs = ferbo_abs.eigenvals(evals_count=4)
    print(f"   Parameters: {ferbo_abs_params}")
    print(f"   Eigenvalues: {evals_ferbo_abs}")
    print(f"   Array format: {list(evals_ferbo_abs)}")

    # 6. Ferbo EL standard params
    print("\n6. Ferbo EL standard params:")
    ferbo_el_params = {
        "Ec": 3.0,
        "El": 0.15,
        "Ej": 6.0,
        "Gamma": 1.8,
        "delta_Gamma": 0.2,
        "er": 0.4,
        "phase": 0.3,
        "dimension": 200,
        "flux_grouping": "EL",
    }

    ferbo_el = Ferbo(**ferbo_el_params)
    evals_ferbo_el = ferbo_el.eigenvals(evals_count=4)
    print(f"   Parameters: {ferbo_el_params}")
    print(f"   Eigenvalues: {evals_ferbo_el}")
    print(f"   Array format: {list(evals_ferbo_el)}")

    # 7. Ferbo transparency
    print("\n7. Ferbo transparency:")
    ferbo_trans_params = {
        "Ec": 2.5,
        "El": 0.18,
        "Ej": 7.0,
        "Gamma": 2.2,
        "delta_Gamma": 0.15,
        "er": 0.35,
        "phase": 0.0,  # Zero phase for transparency calc
        "dimension": 200,
        "flux_grouping": "ABS",
    }

    ferbo_trans = Ferbo(**ferbo_trans_params)
    transparency = ferbo_trans.transparency
    print(f"   Parameters: {ferbo_trans_params}")
    print(f"   Transparency: {transparency}")

    # 8. Fluxonium operators test
    print("\n8. Fluxonium operators test:")
    flux_op_params = {
        "Ec": 2.0,
        "El": 0.18,
        "Ej": 4.5,
        "phase": 0.0,
        "dimension": 100,
        "flux_grouping": "EL",
    }

    flux_op = Fluxonium(**flux_op_params)
    print(f"   Parameters: {flux_op_params}")
    print(f"   n_zpf: {flux_op.n_zpf}")
    print(f"   phase_zpf: {flux_op.phase_zpf}")

    # Test operators
    n_op = flux_op.n_operator()
    phase_op = flux_op.phase_operator()

    n_evals = np.sort(np.linalg.eigvals(n_op).real)
    phase_evals = np.sort(np.linalg.eigvals(phase_op).real)

    print(f"   n_operator range: [{n_evals[0]:.3f}, {n_evals[-1]:.3f}]")
    print(f"   phase_operator range: [{phase_evals[0]:.3f}, {phase_evals[-1]:.3f}]")

    # 9. Commutation relations
    print("\n9. Commutation relations:")
    flux_comm_params = {
        "Ec": 1.8,
        "El": 0.16,
        "Ej": 5.5,
        "phase": 0.1,
        "dimension": 12,  # Small dimension for commutator test
        "flux_grouping": "EL",
    }

    flux_comm = Fluxonium(**flux_comm_params)
    n_op_comm = flux_comm.n_operator()
    phase_op_comm = flux_comm.phase_operator()

    # Calculate commutator [n, φ]
    commutator = n_op_comm @ phase_op_comm - phase_op_comm @ n_op_comm
    expected = 1j * np.eye(flux_comm.dimension)

    print(f"   Parameters: {flux_comm_params}")
    print(
        f"   Commutator vs expected (max diff): {np.max(np.abs(commutator - expected))}"
    )
    print(
        f"   Should be close to zero: {np.allclose(commutator, expected, atol=1e-10)}"
    )

    # 10. Potential values test
    print("\n10. Potential values test:")
    flux_pot_params = {
        "Ec": 2.2,
        "El": 0.14,
        "Ej": 6.5,
        "phase": 0.25,
        "dimension": 100,
        "flux_grouping": "EL",
    }

    flux_pot = Fluxonium(**flux_pot_params)
    phi_values = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    V = flux_pot.potential(phi_values)

    print(f"   Parameters: {flux_pot_params}")
    print(f"   Phi values: {phi_values}")
    print(f"   Potential values: {V}")
    print(f"   Array format: {list(V)}")

    # 11. Potential periodicity test
    print("\n11. Potential periodicity test:")
    flux_per_params = {
        "Ec": 1.9,
        "El": 0.13,
        "Ej": 5.8,
        "phase": 0.0,
        "dimension": 100,
        "flux_grouping": "EJ",
    }

    flux_per = Fluxonium(**flux_per_params)
    phi_base = np.linspace(0, 2 * np.pi, 100)
    V_base = flux_per.potential(phi_base)

    phi_shifted = phi_base + 2 * np.pi
    V_shifted = flux_per.potential(phi_shifted)

    max_diff = np.max(np.abs(V_base - V_shifted))
    print(f"   Parameters: {flux_per_params}")
    print(f"   Max difference V(φ) vs V(φ + 2π): {max_diff}")
    print(f"   Is periodic (tol=1e-12): {np.allclose(V_base, V_shifted, atol=1e-12)}")

    print("\n" + "=" * 60)
    print("All regression values generated with realistic parameters!")
    print("Use these values to update the test_regression.py file.")


if __name__ == "__main__":
    generate_regression_values()
