"""Standalone noise and decoherence calculations.

Pure functions taking eigenvalues, operator matrix elements, and scalar
parameters — no qubit object required. The class methods on
:class:`HybridSuperQubits.qubit_base.QubitBase` are thin wrappers that
resolve state from ``self`` and delegate to these functions.

This is Phase A of the functional refactor proposed in issue #17.
"""

from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np
from scipy.constants import e, h, hbar, k
from scipy.special import k0

__all__ = [
    "transition_omega",
    "t1_from_spectral_density",
    "t1_capacitive",
    "t1_inductive",
    "t1_flux_bias_line",
    "tphi_1_over_f",
    "tphi_CQPS",
]


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
# Spectral densities (private)
# ---------------------------------------------------------------------------


def _S_capacitive(
    omega: np.ndarray, T: float, Ec: float,
    Q_cap: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    x = hbar * omega / (k * T)
    return (
        8 * Ec / Q_cap(omega)
        * 1 / np.tanh(np.abs(x) / 2)
        / (1 + np.exp(-x))
    )


def _S_inductive(
    omega: np.ndarray, T: float, El: float,
    Q_ind: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    x = hbar * omega / (k * T)
    return (
        2 * El / Q_ind(omega)
        * 1 / np.tanh(np.abs(x) / 2)
        / (1 + np.exp(-x))
    )


def _S_flux_bias_line(
    omega: np.ndarray, T: float, M: float, Z: float,
) -> np.ndarray:
    x = hbar * omega / (k * T)
    return (
        4 * np.pi**2 * M**2 * np.abs(omega) * 1e9 * h / Z
        * (1 + 1 / np.tanh(np.abs(x)) / 2)
        / (1 + np.exp(-x))
    )


def _default_Q_cap(omega: np.ndarray) -> np.ndarray:
    return 1e6 * (2 * np.pi * 6e9 / np.abs(omega)) ** 0.7


def _default_Q_ind_factory(T: float) -> Callable[[np.ndarray], np.ndarray]:
    """Build the default inductive Q(omega) at temperature T.

    Reproduces the original behavior in ``QubitBase.t1_inductive``: the
    reference value uses ``h * 0.5e9`` (not ``hbar``), preserved here
    verbatim.
    """
    def Q_ind(omega: np.ndarray) -> np.ndarray:
        return 500e6 * (
            k0(h * 0.5e9 / (2 * k * T))
            * np.sinh(h * 0.5e9 / (2 * k * T))
            / (
                k0(hbar * np.abs(omega) / (2 * k * T))
                * np.sinh(hbar * np.abs(omega) / (2 * k * T))
            )
        )
    return Q_ind


# ---------------------------------------------------------------------------
# Generic T1
# ---------------------------------------------------------------------------


def t1_from_spectral_density(
    evals: np.ndarray,
    matrix_elements: np.ndarray,
    spectral_density: Callable[[float, float], np.ndarray],
    T: float,
    i: int = 1,
    j: int = 0,
    total: bool = True,
    get_rate: bool = False,
) -> float:
    """Generic T1 formula given a spectral density and operator matrix elements.

    Parameters
    ----------
    evals
        Eigenvalues in GHz.
    matrix_elements
        Operator matrix in the eigenbasis. Only ``matrix_elements[i, j]`` is
        used.
    spectral_density
        Callable ``S(omega, T) -> array``. ``omega`` is in rad/s, ``T`` in K.
    T
        Temperature in K.
    i, j
        Transition indices.
    total
        If ``True``, sum the spectral density at ``+omega`` and ``-omega``
        (relaxation + excitation).
    get_rate
        If ``True``, return the rate (1/s) instead of T1 (s).
    """
    omega = transition_omega(evals, i, j)
    s = (
        spectral_density(omega, T) + spectral_density(-omega, T)
        if total
        else spectral_density(omega, T)
    )
    matrix_element = np.abs(matrix_elements[i, j])
    rate = 2 * np.pi * matrix_element**2 * s * 1e9
    return float(rate if get_rate else 1 / rate)


# ---------------------------------------------------------------------------
# Channel-specific T1 functions
# ---------------------------------------------------------------------------


def t1_capacitive(
    evals: np.ndarray,
    n_op_matelems: np.ndarray,
    Ec: float,
    T: float = 0.015,
    Q_cap: Optional[Union[float, Callable]] = None,
    i: int = 1,
    j: int = 0,
    total: bool = True,
    get_rate: bool = False,
) -> float:
    """T1 from capacitive (charge) noise.

    Parameters
    ----------
    evals
        Eigenvalues in GHz.
    n_op_matelems
        Number operator matrix elements in the eigenbasis.
    Ec
        Charging energy in GHz.
    T
        Temperature in K.
    Q_cap
        Capacitive quality factor — a float, a callable ``Q(omega)``, or
        ``None`` for the default ``1e6 * (2*pi*6e9 / |omega|)**0.7``.
    """
    Q_cap_fun = _resolve_q_factor(Q_cap, _default_Q_cap)

    def sd(omega, temp):
        return _S_capacitive(omega, temp, Ec, Q_cap_fun)

    return t1_from_spectral_density(
        evals, n_op_matelems, sd, T, i=i, j=j, total=total, get_rate=get_rate,
    )


def t1_inductive(
    evals: np.ndarray,
    phase_op_matelems: np.ndarray,
    El: float,
    T: float = 0.015,
    Q_ind: Optional[Union[float, Callable]] = None,
    i: int = 1,
    j: int = 0,
    total: bool = True,
    get_rate: bool = False,
) -> float:
    """T1 from inductive (flux) noise."""
    Q_ind_fun = _resolve_q_factor(Q_ind, _default_Q_ind_factory(T))

    def sd(omega, temp):
        return _S_inductive(omega, temp, El, Q_ind_fun)

    return t1_from_spectral_density(
        evals, phase_op_matelems, sd, T, i=i, j=j, total=total, get_rate=get_rate,
    )


def t1_flux_bias_line(
    evals: np.ndarray,
    dH_dphase_matelems: np.ndarray,
    M: float = 2500,
    Z: float = 50,
    T: float = 0.015,
    i: int = 1,
    j: int = 0,
    total: bool = True,
    get_rate: bool = False,
) -> float:
    """T1 from flux-bias-line noise."""

    def sd(omega, temp):
        return _S_flux_bias_line(omega, temp, M, Z)

    return t1_from_spectral_density(
        evals, dH_dphase_matelems, sd, T, i=i, j=j, total=total, get_rate=get_rate,
    )


# ---------------------------------------------------------------------------
# Tphi
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
        First-derivative operator in the eigenbasis. Both its diagonal
        (1st-order term) and off-diagonal elements (2nd-order correction)
        are used.
    A_noise
        Noise amplitude.
    d2H_dlambda2_op
        Optional second-derivative operator in the *original* basis (its
        diagonal is taken internally). Preserves the original API behavior
        of ``QubitBase.tphi_1_over_f``: the bare diagonal is used for the
        2nd-order term, with a perturbative correction from
        ``dH_dlambda_matelems``.
    omega_ir, omega_uv, t_exp
        Noise-spectrum cutoffs and experiment duration.
    get_rate
        If ``True``, return rates (rad/s); otherwise Tphi (s).
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
    """Coherent Quantum Phase Slip dephasing.

    Parameters
    ----------
    evals
        Eigenvalues in GHz (unused by the formula but kept for signature
        symmetry with the other Tphi function).
    displacement_op_matelems
        Displacement operator in the eigenbasis.
    El
        Inductive energy in GHz.
    fp
        Plasma frequency (Hz).
    z
        Normalized impedance ``Z / R_Q``.
    """
    del evals  # not used; kept for API symmetry

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


# Re-export constants used by callers building custom spectral densities.
__all__ += ["R_Q"]
R_Q = h / (2 * e) ** 2
