"""
Yield curve module providing term structure abstractions for discounting
and rate calculations. Includes flat and interpolated curve implementations.
"""

import numpy as np
from abc import ABC, abstractmethod
from scipy.interpolate import CubicSpline

class YieldCurve(ABC):
    """Abstract base class for yield curve implementations.
    
    Defines the interface for discount factors, zero rates, forward rates,
    and parallel shifts. All concrete curves must implement discount_factor
    and shift methods.
    """

    @abstractmethod
    def discount_factor(self, t: float) -> float:
        """Return the discount factor D(0, t) for time t.

        Args:
            t: Time to maturity in years.

        Returns:
            Present value of 1 unit of currency received at time t.
        """
        pass

    @abstractmethod
    def shift(self, dr: float) -> 'YieldCurve':
        """Return a new curve with all rates shifted by dr.

        Args:
            dr: Parallel shift in rate (e.g. 0.0001 for 1 basis point).

        Returns:
            A new YieldCurve instance with shifted rates.
        """
        pass

    def zero_rate(self, t: float) -> float:
        """Compute the continuously compounded zero rate for maturity t.

        Derived from the discount factor: r(t) = -ln(D(0,t)) / t.

        Args:
            t: Time to maturity in years. Must be positive.

        Returns:
            Continuously compounded zero rate for maturity t.

        Raises:
            ValueError: If t is zero or negative.
        """
        if t <= 0:
            raise ValueError("Time must be positive")
        else:
            r = -np.log(self.discount_factor(t)) / t

        return r

    def forward_rate(self, t1: float, t2: float) -> float:
        """Compute the forward rate between times t1 and t2.

        Derived from discount factors: f(t1,t2) = -ln(D(0,t2)/D(0,t1)) / (t2-t1).

        Args:
            t1: Start time in years.
            t2: End time in years. Must be greater than t1.

        Returns:
            Continuously compounded forward rate between t1 and t2.
        """
        fr = -np.log(self.discount_factor(t2) / self.discount_factor(t1)) / (t2 - t1)

        return fr

    def instantaneous_forward(self, t: float) -> float:
        """Compute the instantaneous forward rate at time t.

        Approximated as forward_rate(t, t + epsilon) with epsilon = 1e-6.

        Args:
            t: Time point in years.

        Returns:
            Instantaneous forward rate at time t.
        """
        inst_f = self.forward_rate(t, t + 1e-6)

        return inst_f
    
class FlatCurve(YieldCurve):
    """Constant rate yield curve where all rates equal a single value r.
    
    All discount factors, zero rates, and forward rates are derived from
    the single constant rate. Useful for simplified pricing or as a
    fallback when only a scalar rate is provided.
    """

    def __init__(self, r: float):
        """Initialise a flat yield curve with constant rate.

        Args:
            r: Constant continuously compounded annual rate.
        """
        self.r = r

    def shift(self, dr: float):
        """Return a new FlatCurve with rate shifted by dr.

        Args:
            dr: Parallel shift to apply to the constant rate.

        Returns:
            A new FlatCurve with rate r + dr.
        """
        return FlatCurve(self.r + dr)

    def discount_factor(self, t: float) -> float:
        """Return exp(-r * t) for the constant rate r.

        Args:
            t: Time to maturity in years.

        Returns:
            Discount factor D(0, t) = exp(-r * t).
        """
        df = np.exp(-self.r * t)

        return df

    def zero_rate(self, t: float) -> float:
        """Return the constant rate r for any maturity.

        Args:
            t: Time to maturity in years (unused, rate is constant).

        Returns:
            The constant rate r.
        """
        zr = self.r

        return zr

    def forward_rate(self, t1: float, t2: float) -> float:
        """Return the constant rate r for any forward period.

        Args:
            t1: Start time in years (unused, rate is constant).
            t2: End time in years (unused, rate is constant).

        Returns:
            The constant rate r.
        """
        fr = self.r

        return fr

    def instantaneous_forward(self, t: float) -> float:
        """Return the constant rate r as the instantaneous forward rate.

        Args:
            t: Time point in years (unused, rate is constant).

        Returns:
            The constant rate r.
        """
        inst_f = self.r

        return inst_f

class InterpolatedCurve(YieldCurve):
    """Term-structure-aware yield curve using cubic spline interpolation.
    
    Fits a cubic spline through log-discount-factors at the given tenor
    points, anchored at (0, 0). Supports arbitrary term structures and
    smooth interpolation between knot points.
    """

    def __init__(self, tenors, zero_rates):
        """Initialise an interpolated curve from tenor-rate pairs.

        Builds a cubic spline on log-discount-factors for smooth
        interpolation of discount factors at arbitrary maturities.

        Args:
            tenors: Array of maturities in years (e.g. [0.25, 0.5, 1, 2, 5]).
            zero_rates: Array of continuously compounded zero rates at each tenor.
        """
        self.og_tenors = tenors
        self.og_zero_rates = zero_rates
        self.og_log_df = -self.og_zero_rates * self.og_tenors
        self.tenors = np.concatenate(([0], self.og_tenors))
        self.log_df = np.concatenate(([0], self.og_log_df))
        self._spline = CubicSpline(self.tenors, self.log_df)

    def shift(self, dr: float):
        """Return a new InterpolatedCurve with all zero rates shifted by dr.

        Args:
            dr: Parallel shift to apply to all zero rates.

        Returns:
            A new InterpolatedCurve with shifted rates and re-fitted spline.
        """
        return InterpolatedCurve(self.og_tenors, self.og_zero_rates + dr)

    def discount_factor(self, t: float):
        """Return the discount factor at time t via spline interpolation.

        Evaluates exp(spline(t)) where the spline is fitted on log-discount-factors.

        Args:
            t: Time to maturity in years.

        Returns:
            Interpolated discount factor D(0, t).
        """
        log_df = self._spline(t)

        return np.exp(log_df)