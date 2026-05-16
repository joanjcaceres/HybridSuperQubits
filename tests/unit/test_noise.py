"""Unit tests for the standalone noise functions in HybridSuperQubits.noise.

These tests use small synthetic inputs (no qubit object) — the headline
goal of the functional refactor (issue #17) is that decoherence formulas
should be callable directly with primitives.
"""

import numpy as np
import pytest

from HybridSuperQubits import noise


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def two_level_evals() -> np.ndarray:
    """Eigenvalues in GHz for a fictitious 2-level system."""
    return np.array([0.0, 5.0])


@pytest.fixture
def two_level_matelems() -> np.ndarray:
    """Hermitian off-diagonal matrix elements with |M_10| = 1."""
    return np.array([[0.0, 1.0j], [-1.0j, 0.0]])


@pytest.fixture
def small_evals() -> np.ndarray:
    return np.array([0.0, 4.5, 7.2, 10.1])


@pytest.fixture
def small_matelems(small_evals) -> np.ndarray:
    """Hermitian operator with non-trivial diagonal and off-diagonal pattern."""
    n = len(small_evals)
    rng = np.random.default_rng(0)
    a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    return (a + a.conj().T) / 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_transition_omega_sign_and_units(two_level_evals):
    omega_10 = noise.transition_omega(two_level_evals, 1, 0)
    omega_01 = noise.transition_omega(two_level_evals, 0, 1)
    assert omega_10 == pytest.approx(2 * np.pi * 5.0 * 1e9)
    assert omega_01 == -omega_10


# ---------------------------------------------------------------------------
# T1
# ---------------------------------------------------------------------------


def test_t1_capacitive_returns_finite_positive(two_level_evals, two_level_matelems):
    t1 = noise.t1_capacitive(
        evals=two_level_evals, n_op_matelems=two_level_matelems, Ec=1.0,
    )
    assert np.isfinite(t1) and t1 > 0


def test_t1_capacitive_get_rate_is_reciprocal(two_level_evals, two_level_matelems):
    t1 = noise.t1_capacitive(
        evals=two_level_evals, n_op_matelems=two_level_matelems, Ec=1.0,
    )
    rate = noise.t1_capacitive(
        evals=two_level_evals, n_op_matelems=two_level_matelems, Ec=1.0,
        get_rate=True,
    )
    assert rate == pytest.approx(1 / t1)


def test_t1_capacitive_scales_with_matrix_element(two_level_evals):
    base = np.array([[0.0, 1.0j], [-1.0j, 0.0]])
    big = base * 2.0
    t1_base = noise.t1_capacitive(evals=two_level_evals, n_op_matelems=base, Ec=1.0)
    t1_big = noise.t1_capacitive(evals=two_level_evals, n_op_matelems=big, Ec=1.0)
    # |M|^2 grows by 4, so rate grows by 4 -> T1 shrinks by 4.
    assert t1_base / t1_big == pytest.approx(4.0, rel=1e-10)


def test_t1_capacitive_constant_Q(two_level_evals, two_level_matelems):
    """Passing a float Q_cap should produce a finite result identical to passing
    a callable that returns the same constant."""
    t1_const = noise.t1_capacitive(
        evals=two_level_evals, n_op_matelems=two_level_matelems, Ec=1.0,
        Q_cap=1e6,
    )
    t1_lambda = noise.t1_capacitive(
        evals=two_level_evals, n_op_matelems=two_level_matelems, Ec=1.0,
        Q_cap=lambda omega: np.full_like(omega, 1e6),
    )
    assert t1_const == pytest.approx(t1_lambda, rel=1e-12)


def test_t1_inductive_returns_finite_positive(two_level_evals, two_level_matelems):
    t1 = noise.t1_inductive(
        evals=two_level_evals, phase_op_matelems=two_level_matelems, El=0.5,
    )
    assert np.isfinite(t1) and t1 > 0


def test_t1_flux_bias_line_returns_finite_positive(two_level_evals, two_level_matelems):
    t1 = noise.t1_flux_bias_line(
        evals=two_level_evals, dH_dphase_matelems=two_level_matelems,
    )
    assert np.isfinite(t1) and t1 > 0


def test_t1_generic_matches_capacitive(two_level_evals, two_level_matelems):
    """``t1_from_spectral_density`` should reproduce ``t1_capacitive`` when fed
    the matching closure."""
    Ec = 1.0
    Q_cap = noise._default_Q_cap

    def sd(omega, T):
        return noise._S_capacitive(omega, T, Ec, Q_cap)

    t1_generic = noise.t1_from_spectral_density(
        evals=two_level_evals, matrix_elements=two_level_matelems,
        spectral_density=sd, T=0.015,
    )
    t1_specific = noise.t1_capacitive(
        evals=two_level_evals, n_op_matelems=two_level_matelems, Ec=Ec,
    )
    assert t1_generic == pytest.approx(t1_specific, rel=1e-12)


# ---------------------------------------------------------------------------
# Tphi
# ---------------------------------------------------------------------------


def test_tphi_1_over_f_returns_array(small_evals, small_matelems):
    out = noise.tphi_1_over_f(
        evals=small_evals, dH_dlambda_matelems=small_matelems, A_noise=1e-6,
    )
    assert out.shape == (len(small_evals), len(small_evals))
    # Off-diagonal Tphi should be positive and finite for non-zero derivatives.
    assert np.isfinite(out[1, 0]) and out[1, 0] > 0


def test_tphi_1_over_f_get_rate_is_reciprocal(small_evals, small_matelems):
    rate = noise.tphi_1_over_f(
        evals=small_evals, dH_dlambda_matelems=small_matelems, A_noise=1e-6,
        get_rate=True,
    )
    tphi = noise.tphi_1_over_f(
        evals=small_evals, dH_dlambda_matelems=small_matelems, A_noise=1e-6,
    )
    # Allow tiny FP slack from the epsilon floor.
    assert np.allclose(rate, 1 / tphi, rtol=1e-10)


def test_tphi_1_over_f_zero_derivative_gives_infinite_tphi(small_evals):
    """A noise operator with no diagonal coupling and no 2nd-order term should
    give Tphi == infinity (rate clamped to the 1e-12 epsilon floor)."""
    n = len(small_evals)
    zero_op = np.zeros((n, n), dtype=complex)
    tphi = noise.tphi_1_over_f(
        evals=small_evals, dH_dlambda_matelems=zero_op, A_noise=1e-6,
    )
    # Diagonal entries are i==j so rate_1st == 0; result hits the eps floor.
    expected = 1 / (1e-12 * 2 * np.pi * 1e9)
    assert tphi[0, 0] == pytest.approx(expected, rel=1e-10)


def test_tphi_1_over_f_second_order_branch_runs(small_evals, small_matelems):
    """Including the 2nd-order operator should change the output."""
    n = len(small_evals)
    d2_op = np.diag(np.arange(1.0, n + 1.0))
    tphi_1 = noise.tphi_1_over_f(
        evals=small_evals, dH_dlambda_matelems=small_matelems, A_noise=1e-6,
    )
    tphi_2 = noise.tphi_1_over_f(
        evals=small_evals, dH_dlambda_matelems=small_matelems, A_noise=1e-6,
        d2H_dlambda2_op=d2_op,
    )
    # 2nd-order term adds rate, so Tphi shrinks (or stays equal at most places).
    assert np.all(tphi_2 <= tphi_1 + 1e-20)
    assert not np.allclose(tphi_1, tphi_2)


def test_tphi_cqps_returns_finite(small_evals, small_matelems):
    out = noise.tphi_CQPS(
        evals=small_evals, displacement_op_matelems=small_matelems, El=0.5,
    )
    assert out.shape == (len(small_evals), len(small_evals))
    # Off-diagonal entries should be finite positive (or inf where structure
    # factor is zero).
    off = out[1, 0]
    assert (np.isfinite(off) and off > 0) or np.isinf(off)


def test_tphi_cqps_equal_diagonals_zero_rate(small_evals):
    """If the displacement operator diagonal is uniform, the structure factor
    vanishes and the rate is identically zero. The Tphi (1/rate) branch hits
    the ``np.where(rate==0, np.inf, rate)`` clamp before inversion, so Tphi
    returns 0 — exact behavior preserved from the original implementation."""
    n = len(small_evals)
    uniform_diag_op = np.eye(n, dtype=complex) * 0.5
    rate = noise.tphi_CQPS(
        evals=small_evals, displacement_op_matelems=uniform_diag_op, El=0.5,
        get_rate=True,
    )
    tphi = noise.tphi_CQPS(
        evals=small_evals, displacement_op_matelems=uniform_diag_op, El=0.5,
    )
    assert np.all(np.isinf(rate))
    assert np.all(tphi == 0.0)
