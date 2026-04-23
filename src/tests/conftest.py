"""
Shared fixtures for the test suite.
"""
import pytest
import numpy as np
from market_data.yield_curve import FlatCurve, InterpolatedCurve
from models.numerical import MonteCarlo


@pytest.fixture
def flat_curve():
    """A flat yield curve at 5%."""
    return FlatCurve(0.05)


@pytest.fixture
def interp_curve():
    """An interpolated curve with 3 tenor points."""
    tenors = np.array([0.5, 1.0, 2.0])
    rates = np.array([0.04, 0.05, 0.06])
    return InterpolatedCurve(tenors, rates)


@pytest.fixture
def mc_engine(flat_curve):
    """A seeded Monte Carlo engine with 252 steps and 10_000 paths."""
    return MonteCarlo(n=252, M=10_000, curve=flat_curve, seed=42)
