"""Standalone noise and decoherence calculations.

Pure functions taking eigenvalues, operator matrix elements, and scalar
parameters — no qubit object required. The class methods on
:class:`HybridSuperQubits.qubit_base.QubitBase` are thin wrappers that
resolve state from ``self`` and delegate to these functions.

Two layers:

* **Single-shot** functions (``t1_capacitive``, ``t1_inductive``, ``t1_flux_bias_line``,
  ``tphi_1_over_f``, ``tphi_CQPS``) take eigenvalues/matrix-elements at one
  parameter value and return a single rate or one (n_eval, n_eval) Tphi matrix.
* **Sweep / table** functions (``t1_table_from_spectral_density``,
  ``tphi_1_over_f_table``, ``tphi_cqps_table``) consume the full
  ``(n_param, n_eval[, n_eval])`` tables produced by a parameter sweep and
  return a ``t1_table`` / ``tphi_table`` of the same shape.

Both layers share the same channel-specific spectral densities (``S_capacitive``,
``S_inductive``, ``S_flux_bias_line``, ``S_charge_impedance``, ``S_one_over_f_flux``,
``S_critical_current``, ``S_andreev``) — a single source of truth for the
physics.
"""

from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np
from scipy.constants import e, h, hbar, k
from scipy.special import k0

__all__ = [
    "R_Q",
    # spectral densities
    "S_capacitive",
    "S_inductive",
    "S_flux_bias_line",
    "S_charge_impedance",
    "S_one_over_f_flux",
    "S_critical_current",
    "S_andreev",
    # default Q factories
    "default_Q_cap",
    "default_Q_ind_factory",
    # helpers
    "transition_omega",
    # single-shot
    "t1_from_spectral_density",
    "t1_capacitive",
    "t1_inductive",
    "t1_flux_bias_line",
    "tphi_1_over_f",
    "tphi_CQPS",
    # sweep / table
    "t1_table_from_spectral_density",
    "tphi_1_over_f_table",
    "tphi_cqps_table",
]


R_Q = h / (2 * e) ** 2  # Superconducting resistance quantum


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def transition_omega(evals: np.ndarray, i: int, j: int) -> float:
    """Angular frequency (rad/s) of the |i> <-> |j> transition.

    Eigenvalues are assumed to be in GHz (the library-wide convention).
    """
    return 2 * np.pi * (evals[i] - evals[j]) * 1e9


def _resolve_q_factor(
    q_factor: Optional[Union[float, Callable]],
    default: Callable[[np.ndarray], np.ndarray],
) -> Callable[[np.ndarray], np.ndarray]:
    """Coerce a quality-factor argument into a callable Q(omega)."""
    if q_factor is None:
        return default
    if callable(q_factor):
        return q_factor

    def _const(omega: np.ndarray, _value: float = q_factor) -> np.ndarray:
        return np.full_like(np.asarray(omega, dtype=float), _value)

    return _const


# ---------------------------------------------------------------------------
# Default quality-factor models
# ---------------------------------------------------------------------------


def default_Q_cap(omega: np.ndarray) -> np.ndarray:
    """Default capacitive quality factor: ``(1/3e-5) (2π·6e9 / |ω|)**0.7``."""
    return 1 / 3e-5 * (2 * np.pi * 6e9 / np.abs(omega)) ** 0.7


def default_Q_ind_factory(T: float) -> Callable[[np.ndarray], np.ndarray]:
    """Build the temperature-dependent default ``Q_ind(omega)``.

    Q_ind = 500e6 · q_ind(ω) / q_ind(ω_ref) with q_ind(ω) = 1 / (k0(x) sinh(x))
    and ω_ref = 2π · 0.5 GHz.
    """
    Q_ind_ref = 500e6
    omega_ref = 2 * np.pi * 0.5e9

    def q_ind(omega: np.ndarray) -> np.ndarray:
        x = (hbar * np.abs(omega)) / (2 * k * T)
        return 1 / (k0(x) * np.sinh(x))

    def Q_ind(omega: np.ndarray) -> np.ndarray:
        return Q_ind_ref * q_ind(omega) / q_ind(omega_ref)

    return Q_ind


# ---------------------------------------------------------------------------
# Spectral densities — single source of truth, shared by single-shot and
# sweep functions. Formulas follow the sweep-path conventions (the
# physically consistent ones).
# ---------------------------------------------------------------------------


def S_capacitive(
    omega: np.ndarray, T: float, Ec: float,
    Q_cap: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    x = hbar * omega / (k * T)
    return (
        16 * Ec / Q_cap(omega)
        * 1 / np.tanh(np.abs(x) / 2)
        / (1 + np.exp(-x))
    )


def S_inductive(
    omega: np.ndarray, T: float, El: float,
    Q_ind: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    x = hbar * omega / (k * T)
    return (
        2 * El / Q_ind(omega)
        * 1 / np.tanh(np.abs(x) / 2)
        / (1 + np.exp(-x))
    )


def S_flux_bias_line(
    omega: np.ndarray, T: float, M: float, Z: float,
) -> np.ndarray:
    x = hbar * omega / (k * T)
    return (
        4 * np.pi**2 * M**2 * omega * 1e9 * h / Z
        * (1 + 1 / np.tanh(x / 2))
    )


def S_charge_impedance(
    omega: np.ndarray, T: float, Z: float,
) -> np.ndarray:
    x = hbar * omega / (k * T)
    return (
        omega / 1e9 / R_Q * Z
        * (1 + 1 / np.tanh(np.abs(x) / 2))
        / (1 + np.exp(-x))
    )


def S_one_over_f_flux(
    omega: np.ndarray, T: float, A_noise: float,
) -> np.ndarray:
    del T  # not used; kept for dispatcher signature symmetry
    return 2 * np.pi * A_noise**2 / np.abs(omega)


def S_critical_current(
    omega: np.ndarray, T: float, A_noise: float, El: float, N: int,
) -> np.ndarray:
    del T
    return 2 * np.pi * (A_noise * El / np.sqrt(N)) ** 2 / omega * 1e9


def S_andreev(
    omega: np.ndarray, T: float, R: float,
) -> np.ndarray:
    x = hbar * omega / (k * T)
    return R * omega / 1e9 / 4 / R_Q * (1 + 1 / np.tanh(x / 2))


# ---------------------------------------------------------------------------
# Single-shot T1
# ---------------------------------------------------------------------------


def t1_from_spectral_density(
    evals: np.ndarray,
    matrix_elements: np.ndarray,
    spectral_density: Callable[[float, float], np.ndarray],
    T: float,
    i: int = 1,
    j: int = 0,
    get_rate: bool = False,
) -> float:
    """Generic T1 for one transition.

    rate = 2π · |M_ij|² · S(ω_ij, T) · 1e9, T1 = 1/rate.
    """
    omega = transition_omega(evals, i, j)
    s = spectral_density(omega, T)
    matrix_element = np.abs(matrix_elements[i, j])
    rate = 2 * np.pi * matrix_element**2 * s * 1e9
    return float(rate if get_rate else 1 / rate)


def t1_capacitive(
    evals: np.ndarray,
    n_op_matelems: np.ndarray,
    Ec: float,
    T: float = 0.015,
    Q_cap: Optional[Union[float, Callable]] = None,
    i: int = 1,
    j: int = 0,
    get_rate: bool = False,
) -> float:
    """T1 from capacitive (charge) noise."""
    Q_cap_fun = _resolve_q_factor(Q_cap, default_Q_cap)

    def sd(omega, temp):
        return S_capacitive(omega, temp, Ec, Q_cap_fun)

    return t1_from_spectral_density(
        evals, n_op_matelems, sd, T, i=i, j=j, get_rate=get_rate,
    )


def t1_inductive(
    evals: np.ndarray,
    phase_op_matelems: np.ndarray,
    El: float,
    T: float = 0.015,
    Q_ind: Optional[Union[float, Callable]] = None,
    i: int = 1,
    j: int = 0,
    get_rate: bool = False,
) -> float:
    """T1 from inductive (flux) noise."""
    Q_ind_fun = _resolve_q_factor(Q_ind, default_Q_ind_factory(T))

    def sd(omega, temp):
        return S_inductive(omega, temp, El, Q_ind_fun)

    return t1_from_spectral_density(
        evals, phase_op_matelems, sd, T, i=i, j=j, get_rate=get_rate,
    )


def t1_flux_bias_line(
    evals: np.ndarray,
    dH_dphase_matelems: np.ndarray,
    M: float = 2500,
    Z: float = 50,
    T: float = 0.015,
    i: int = 1,
    j: int = 0,
    get_rate: bool = False,
) -> float:
    """T1 from flux-bias-line noise."""

    def sd(omega, temp):
        return S_flux_bias_line(omega, temp, M, Z)

    return t1_from_spectral_density(
        evals, dH_dphase_matelems, sd, T, i=i, j=j, get_rate=get_rate,
    )


# ---------------------------------------------------------------------------
# Single-shot Tphi
# ---------------------------------------------------------------------------


def tphi_1_over_f(
    evals: np.ndarray,
    dH_dlambda_matelems: np.ndarray,
    A_noise: float,
    d2H_dlambda2_op: Optional[np.ndarray] = None,
    omega_ir: float = 2 * np.pi,
    omega_uv: float = 3 * 2 * np.pi * 1e9,
    t_exp: float = 10e-6,
    get_rate: bool = False,
) -> np.ndarray:
    """1/f dephasing rate (or Tphi) matrix from a noise operator.

    Parameters
    ----------
    evals
        Eigenvalues in GHz.
    dH_dlambda_matelems
        First-derivative operator in the eigenbasis.
    A_noise
        Noise amplitude.
    d2H_dlambda2_op
        Optional second-derivative operator in the *original* basis (its
        diagonal is taken internally). NOTE: this is an operator-based
        approximation to d²E/dλ². The sweep-path
        :func:`tphi_1_over_f_table` accepts an already-computed *numerical*
        d²E table (via ``QubitBase.get_d2E_d_param_vs_paramvals``), which is
        the physically preferred route when a parameter sweep is available.
    omega_ir, omega_uv, t_exp
        Noise-spectrum cutoffs and experiment duration.
    """
    dH_d_lambda = dH_dlambda_matelems
    dE_d_lambda = np.real(np.diagonal(dH_d_lambda))
    dEij_d_lambda = dE_d_lambda[:, np.newaxis] - dE_d_lambda[np.newaxis, :]

    rate_1st = (
        dEij_d_lambda * A_noise
        * np.sqrt(2 * np.abs(np.log(omega_ir * t_exp)))
    )

    if d2H_dlambda2_op is not None:
        d2H_d_lambda2 = np.diagonal(d2H_dlambda2_op)
        E_diff = evals[:, np.newaxis] - evals[np.newaxis, :]
        E_diff = np.where(E_diff == 0, np.inf, E_diff)

        dH_sq = np.abs(dH_d_lambda) ** 2
        d2E_correction = 2 * np.sum(dH_sq / E_diff)

        d2E_d_lambda2 = d2H_d_lambda2 + d2E_correction
        d2Eij_d_lambda2 = (
            d2E_d_lambda2[:, np.newaxis] + d2E_d_lambda2[np.newaxis, :]
        )

        rate_2nd = (
            np.abs(d2Eij_d_lambda2) * A_noise**2
            * np.sqrt(
                2 * np.log(omega_uv / omega_ir) ** 2
                + 2 * np.log(omega_ir * t_exp) ** 2
            )
        )
    else:
        rate_2nd = 0

    rate = np.sqrt(rate_1st**2 + rate_2nd**2)
    rate = np.where(rate == 0, 1e-12, rate)
    rate *= 2 * np.pi * 1e9

    return np.asarray(rate if get_rate else 1 / rate)


def tphi_CQPS(
    evals: np.ndarray,
    displacement_op_matelems: np.ndarray,
    El: float,
    fp: float = 17e9,
    z: float = 0.05,
    get_rate: bool = False,
) -> np.ndarray:
    """Coherent Quantum Phase Slip dephasing for a single parameter value."""
    del evals  # not used by the CQPS formula

    phase_slip_frequency = (
        4 * np.sqrt(2) / np.pi * fp / np.sqrt(z) * np.exp(-4 / np.pi / z)
    )
    diag = np.diagonal(displacement_op_matelems)
    structure_factor = diag[:, np.newaxis] - diag[np.newaxis, :]

    N_junctions = fp / 2 / np.pi / (El * 1e9) / z

    rate = (
        np.pi * np.sqrt(N_junctions) * phase_slip_frequency
        * np.abs(structure_factor)
    )
    rate = np.where(rate == 0, np.inf, rate)
    return np.asarray(rate if get_rate else 1 / rate)


# ---------------------------------------------------------------------------
# Sweep / table-level T1
# ---------------------------------------------------------------------------


def t1_table_from_spectral_density(
    evals_table: np.ndarray,
    matelems_table: np.ndarray,
    spectral_density: Callable[[np.ndarray, float], np.ndarray],
    T: float,
    min_freq_cutoff_ghz: float = 1e-9,
    max_freq_cutoff_ghz: float = 80.0,
) -> np.ndarray:
    """T1 table from sweep-level eigenvalues and operator matrix elements.

    Parameters
    ----------
    evals_table
        Shape ``(n_param, n_eval)``. Eigenvalues in GHz.
    matelems_table
        Shape ``(n_param, n_eval, n_eval)``. Operator matrix elements in the
        eigenbasis at every parameter value.
    spectral_density
        Callable ``S(omega, T) -> array`` evaluated on the full transition
        frequency grid (``omega`` in rad/s).
    T
        Temperature in K.
    min_freq_cutoff_ghz, max_freq_cutoff_ghz
        Transition-frequency band (GHz). Transitions outside the band have
        their T1 set to NaN.
    """
    transitions = (
        evals_table[:, :, np.newaxis] - evals_table[:, np.newaxis, :]
    )
    transitions = np.where(
        np.abs(transitions) < min_freq_cutoff_ghz, np.nan, transitions,
    )
    transitions = np.where(
        np.abs(transitions) > max_freq_cutoff_ghz, np.nan, transitions,
    )

    omega = 2 * np.pi * transitions * 1e9
    s = spectral_density(omega, T)

    rate = 2 * np.pi * np.abs(matelems_table) ** 2 * s
    rate *= 1e9
    rate = np.where(rate == 0, np.nan, rate)
    t1_table = 1 / rate

    for idx in range(t1_table.shape[0]):
        np.fill_diagonal(t1_table[idx], np.nan)
    return t1_table


# ---------------------------------------------------------------------------
# Sweep / table-level Tphi
# ---------------------------------------------------------------------------


def tphi_1_over_f_table(
    dE_d_lambda_table: np.ndarray,
    A_noise: float,
    d2E_d_lambda2_table: Optional[np.ndarray] = None,
    omega_ir: float = 2 * np.pi,
    omega_uv: float = 3 * 2 * np.pi * 1e9,
    t_exp: float = 10e-6,
) -> np.ndarray:
    """1/f dephasing table from per-parameter energy derivatives.

    Unlike the single-shot :func:`tphi_1_over_f` (which can fall back to an
    operator-diagonal approximation for d²E/dλ²), this table function
    consumes the *numerical* d²E/dλ² from a finite-difference sweep
    (computed upstream by ``QubitBase.get_d2E_d_param_vs_paramvals``).

    Parameters
    ----------
    dE_d_lambda_table
        Shape ``(n_param, n_eval)``. The diagonal of the dH/dλ matrix
        elements in the eigenbasis at every parameter value.
    A_noise
        Noise amplitude.
    d2E_d_lambda2_table
        Shape ``(n_param, n_eval)``. If supplied, adds the 2nd-order term.
    """
    dEij = (
        dE_d_lambda_table[:, :, np.newaxis]
        - dE_d_lambda_table[:, np.newaxis, :]
    )
    rate_1st = (
        np.abs(dEij) * A_noise
        * np.sqrt(2 * np.abs(np.log(omega_ir * t_exp)))
    )

    if d2E_d_lambda2_table is not None:
        d2Eij = (
            d2E_d_lambda2_table[:, :, np.newaxis]
            - d2E_d_lambda2_table[:, np.newaxis, :]
        )
        rate_2nd = (
            np.abs(d2Eij) * A_noise**2
            * np.sqrt(
                2 * np.log(omega_uv / omega_ir) ** 2
                + 2 * np.log(omega_ir * t_exp) ** 2
            )
        )
    else:
        rate_2nd = 0

    rate = np.sqrt(rate_1st**2 + rate_2nd**2)
    rate = np.where(rate == 0, 1e-12, rate)
    rate *= 2 * np.pi * 1e9
    tphi_table = 1 / rate

    for idx in range(tphi_table.shape[0]):
        np.fill_diagonal(tphi_table[idx], np.nan)
    return tphi_table


def tphi_cqps_table(
    displacement_op_matelems_table: np.ndarray,
    El_values: np.ndarray,
    fp: float = 17e9,
    z: float = 0.05,
) -> np.ndarray:
    """CQPS dephasing table for a parameter sweep.

    Parameters
    ----------
    displacement_op_matelems_table
        Shape ``(n_param, n_eval, n_eval)``. Displacement operator matrix
        elements at every parameter value.
    El_values
        Shape ``(n_param,)``. Inductive energy at each parameter value (for
        sweeps of ``param_name != "El"`` this is a constant array).
    """
    diag = np.diagonal(displacement_op_matelems_table, axis1=1, axis2=2)
    structure_factor = diag[:, :, np.newaxis] - diag[:, np.newaxis, :]

    phase_slip_frequency = (
        4 * np.sqrt(2) / np.pi * fp / np.sqrt(z) * np.exp(-4 / np.pi / z)
    )
    N_junctions = fp / 2 / np.pi / (El_values * 1e9) / z

    rate = (
        np.pi
        * np.sqrt(N_junctions)[:, np.newaxis, np.newaxis]
        * phase_slip_frequency
        * np.abs(structure_factor)
    )
    rate = np.where(rate == 0, np.inf, rate)
    return 1 / rate
