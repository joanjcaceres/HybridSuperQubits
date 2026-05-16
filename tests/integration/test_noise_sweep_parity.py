"""Parity tests for Phase B sweep refactor.

For each ``get_t1_*_vs_paramvals`` / ``get_tphi_*_vs_paramvals`` method:
call it on a real Fluxonium and confirm the resulting t1/tphi table is
identical (numerically, not just close) to what we get by:

1. Building the spectrum + matrix elements via the existing helpers
2. Calling the corresponding standalone function in ``HybridSuperQubits.noise``

This guarantees the wrapper layer adds no physics — only orchestration.
"""

import numpy as np
import pytest

from HybridSuperQubits import noise


@pytest.fixture
def param_vals():
    # Small flux sweep around the avoided crossing — gives non-trivial T1/Tphi.
    return np.linspace(0.1, np.pi - 0.1, 4)


# ---------------------------------------------------------------------------
# T1 sweep parity
# ---------------------------------------------------------------------------


class TestT1SweepParity:
    def test_capacitive(self, fluxonium, param_vals):
        sd = fluxonium.get_t1_capacitive_vs_paramvals(
            "phase", param_vals, evals_count=4,
        )
        Q_cap_fun = noise.default_Q_cap

        def spectral_density(omega, T):
            return noise.S_capacitive(omega, T, fluxonium.Ec, Q_cap_fun)

        expected = noise.t1_table_from_spectral_density(
            evals_table=sd.energy_table,
            matelems_table=sd.matrixelem_table["n_operator"],
            spectral_density=spectral_density,
            T=0.015,
        )
        np.testing.assert_array_equal(sd.t1_table["capacitive"], expected)

    def test_inductive(self, fluxonium, param_vals):
        sd = fluxonium.get_t1_inductive_vs_paramvals(
            "phase", param_vals, evals_count=4,
        )
        Q_ind_fun = noise.default_Q_ind_factory(0.015)

        def spectral_density(omega, T):
            return noise.S_inductive(omega, T, fluxonium.El, Q_ind_fun)

        expected = noise.t1_table_from_spectral_density(
            evals_table=sd.energy_table,
            matelems_table=sd.matrixelem_table["phase_operator"],
            spectral_density=spectral_density,
            T=0.015,
        )
        np.testing.assert_array_equal(sd.t1_table["inductive"], expected)

    def test_charge_impedance(self, fluxonium, param_vals):
        sd = fluxonium.get_t1_charge_impedance_vs_paramvals(
            "phase", param_vals, evals_count=4,
        )

        def spectral_density(omega, T):
            return noise.S_charge_impedance(omega, T, 50.0)

        expected = noise.t1_table_from_spectral_density(
            evals_table=sd.energy_table,
            matelems_table=sd.matrixelem_table["n_operator"],
            spectral_density=spectral_density,
            T=0.015,
        )
        np.testing.assert_array_equal(sd.t1_table["charge_impedance"], expected)

    def test_flux_bias_line(self, fluxonium, param_vals):
        sd = fluxonium.get_t1_flux_bias_line_vs_paramvals(
            "phase", param_vals, evals_count=4,
        )

        def spectral_density(omega, T):
            return noise.S_flux_bias_line(omega, T, 2500.0, 50.0)

        expected = noise.t1_table_from_spectral_density(
            evals_table=sd.energy_table,
            matelems_table=sd.matrixelem_table["d_hamiltonian_d_phase"],
            spectral_density=spectral_density,
            T=0.015,
        )
        np.testing.assert_array_equal(sd.t1_table["flux_bias_line"], expected)

    def test_one_over_f_flux(self, fluxonium, param_vals):
        sd = fluxonium.get_t1_1_over_f_flux_vs_paramvals(
            "phase", param_vals, evals_count=4,
        )

        def spectral_density(omega, T):
            return noise.S_one_over_f_flux(omega, T, 1e-6)

        expected = noise.t1_table_from_spectral_density(
            evals_table=sd.energy_table,
            matelems_table=sd.matrixelem_table["d_hamiltonian_d_phase"],
            spectral_density=spectral_density,
            T=0.015,
        )
        np.testing.assert_array_equal(sd.t1_table["flux_noise"], expected)


# ---------------------------------------------------------------------------
# Tphi sweep parity
# ---------------------------------------------------------------------------


class TestTphiSweepParity:
    def test_flux_first_order(self, fluxonium, param_vals):
        """Tphi 1/f flux without the 2nd-order operator."""
        sd = fluxonium._get_tphi_1_over_f_vs_paramvals(
            param_name="phase",
            param_vals=param_vals,
            A_noise=1e-6,
            noise_channel="flux_first_order_only",
            noise_operators=["d_hamiltonian_d_phase"],
            evals_count=4,
        )
        dE = np.diagonal(
            sd.matrixelem_table["d_hamiltonian_d_phase"], axis1=1, axis2=2,
        )
        expected = noise.tphi_1_over_f_table(
            dE_d_lambda_table=dE,
            A_noise=1e-6,
            d2E_d_lambda2_table=None,
        )
        np.testing.assert_array_equal(
            sd.tphi_table["flux_first_order_only"], expected,
        )

    def test_flux_with_second_order(self, fluxonium, param_vals):
        """Tphi 1/f flux WITH 2nd-order operator — the full default path.

        Must use ``evals_count=fluxonium.dimension`` because
        ``get_d2E_d_param_vs_paramvals`` always works at full dimension
        (it silently recomputes if given fewer evals).
        """
        sd = fluxonium.get_tphi_flux_vs_paramvals(
            "phase", param_vals, A_noise=1e-6, evals_count=fluxonium.dimension,
        )
        dE = np.diagonal(
            sd.matrixelem_table["d_hamiltonian_d_phase"], axis1=1, axis2=2,
        )
        d2E = sd.d2E_table["d2E_d_phase2"]
        expected = noise.tphi_1_over_f_table(
            dE_d_lambda_table=dE,
            A_noise=1e-6,
            d2E_d_lambda2_table=d2E,
        )
        np.testing.assert_array_equal(sd.tphi_table["flux_noise"], expected)

    def test_cqps(self, fluxonium, param_vals):
        sd = fluxonium.get_tphi_CQPS_vs_paramvals(
            "phase", param_vals, evals_count=4,
        )
        El_values = np.full(len(param_vals), fluxonium.El)
        expected = noise.tphi_cqps_table(
            displacement_op_matelems_table=sd.matrixelem_table[
                "displacement_operator"
            ],
            El_values=El_values,
            fp=17e9,
            z=0.05,
        )
        # CQPS uses np.inf in rate-zero entries; allow exact match including inf.
        np.testing.assert_array_equal(sd.tphi_table["CQPS"], expected)
