"""
Tests for market_data.yield_curve — FlatCurve, InterpolatedCurve, YieldCurve base.
"""
import pytest
import numpy as np
from market_data.yield_curve import FlatCurve, InterpolatedCurve


# ── FlatCurve ──────────────────────────────────────────────────────

class TestFlatCurve:

    def test_discount_factor_at_zero(self, flat_curve):
        assert flat_curve.discount_factor(0) == pytest.approx(1.0)

    def test_discount_factor_positive_time(self, flat_curve):
        expected = np.exp(-0.05 * 1.0)
        assert flat_curve.discount_factor(1.0) == pytest.approx(expected)

    def test_discount_factor_decreases_with_time(self, flat_curve):
        assert flat_curve.discount_factor(2.0) < flat_curve.discount_factor(1.0)

    def test_zero_rate_is_constant(self, flat_curve):
        assert flat_curve.zero_rate(0.5) == pytest.approx(0.05)
        assert flat_curve.zero_rate(5.0) == pytest.approx(0.05)

    def test_forward_rate_is_constant(self, flat_curve):
        assert flat_curve.forward_rate(0.5, 1.5) == pytest.approx(0.05)

    def test_instantaneous_forward_is_constant(self, flat_curve):
        assert flat_curve.instantaneous_forward(1.0) == pytest.approx(0.05)

    def test_shift_creates_new_curve(self, flat_curve):
        shifted = flat_curve.shift(0.01)
        assert shifted.r == pytest.approx(0.06)
        assert flat_curve.r == pytest.approx(0.05)  # original unchanged

    def test_shift_affects_discount_factor(self, flat_curve):
        shifted = flat_curve.shift(0.01)
        assert shifted.discount_factor(1.0) == pytest.approx(np.exp(-0.06))


# ── InterpolatedCurve ─────────────────────────────────────────────

class TestInterpolatedCurve:

    def test_discount_factor_at_zero(self, interp_curve):
        assert interp_curve.discount_factor(0) == pytest.approx(1.0, abs=1e-10)

    def test_discount_factor_at_knot_point(self, interp_curve):
        # At tenor 1.0, zero rate is 0.05
        expected = np.exp(-0.05 * 1.0)
        assert interp_curve.discount_factor(1.0) == pytest.approx(expected, rel=1e-4)

    def test_zero_rate_at_knot_point(self, interp_curve):
        assert interp_curve.zero_rate(1.0) == pytest.approx(0.05, rel=1e-4)

    def test_shift_preserves_shape(self, interp_curve):
        shifted = interp_curve.shift(0.01)
        assert shifted.zero_rate(1.0) == pytest.approx(0.06, rel=1e-4)

    def test_interpolation_between_knots(self, interp_curve):
        # Rate at 0.75 should be between 0.04 and 0.05
        r = interp_curve.zero_rate(0.75)
        assert 0.03 < r < 0.06


# ── YieldCurve base class ────────────────────────────────────────

class TestYieldCurveBase:

    def test_zero_rate_raises_on_non_positive_time(self, interp_curve):
        with pytest.raises(ValueError, match="Time must be positive"):
            interp_curve.zero_rate(0)

    def test_zero_rate_raises_on_negative_time(self, interp_curve):
        with pytest.raises(ValueError, match="Time must be positive"):
            interp_curve.zero_rate(-1)
