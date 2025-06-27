"""
Shared fixtures and test utilities for HybridSuperQubits tests.
"""

from typing import Any

import pytest

from HybridSuperQubits import Ferbo, Fluxonium


@pytest.fixture
def fluxonium_params() -> dict[str, Any]:
    """Standard Fluxonium parameters for testing."""
    return {
        "Ec": 0.5,
        "El": 1.0,
        "Ej": 2.0,
        "phase": 0.25,
        "dimension": 100,
        "flux_grouping": "EL",
    }


@pytest.fixture
def fluxonium_params_ej() -> dict[str, Any]:
    """Fluxonium parameters with EJ flux grouping for testing."""
    return {
        "Ec": 0.5,
        "El": 1.0,
        "Ej": 2.0,
        "phase": 0.25,
        "dimension": 100,
        "flux_grouping": "EJ",
    }


@pytest.fixture
def ferbo_params() -> dict[str, Any]:
    """Standard Ferbo parameters for testing."""
    return {
        "Ec": 8,
        "El": 0.05,
        "Ej": 2.0,
        "Gamma": 1.5,
        "delta_Gamma": 0.1,
        "er": 0.3,
        "phase": 0.4,
        "dimension": 200,
        "flux_grouping": "ABS",
    }


@pytest.fixture
def fluxonium(fluxonium_params) -> Fluxonium:
    """Fluxonium instance for testing."""
    return Fluxonium(**fluxonium_params)


@pytest.fixture
def fluxonium_ej(fluxonium_params_ej) -> Fluxonium:
    """Fluxonium instance with EJ flux grouping for testing."""
    return Fluxonium(**fluxonium_params_ej)


@pytest.fixture
def ferbo(ferbo_params) -> Ferbo:
    """Ferbo instance for testing."""
    return Ferbo(**ferbo_params)


@pytest.fixture
def tolerance() -> float:
    """Standard numerical tolerance for tests."""
    return 1e-10


@pytest.fixture
def loose_tolerance() -> float:
    """Loose numerical tolerance for tests."""
    return 1e-6
