"""
Base instrument classes for single-asset and multi-asset derivatives.
Provides the common interface (spot, strike, maturity, vol, curve) used
by all payoff classes, pricers, and risk engines.
"""

import numpy as np
import copy
from pyqfin.market_data.yield_curve import YieldCurve, FlatCurve

class Instrument:
    """Base class for single-asset derivative instruments.
    
    Stores market parameters (spot, strike, vol, maturity) and a yield
    curve for discounting. Accepts either a YieldCurve object or a scalar
    rate (which is wrapped in a FlatCurve). All single-asset payoff
    classes inherit from this.
    """

    def __init__(self, S, K, T, vol, curve = None ,r=None, option_type=None, all_returns=None):
        """Initialise a single-asset instrument.

        Either curve or r must be provided. If both are given, curve
        takes precedence.

        Args:
            S: Current spot price of the underlying asset.
            K: Strike price of the derivative.
            T: Time to maturity in years.
            vol: Annualised volatility of the underlying (e.g. 0.20 for 20%).
            curve: YieldCurve instance for discounting. Takes priority over r.
            r: Constant risk-free rate. Wrapped in a FlatCurve if provided.
            option_type: 'c' for call, 'p' for put. None for non-directional instruments.
            all_returns: Simulated price paths, shape (n_steps+1, n_paths). Set by the pricer.

        Raises:
            ValueError: If neither curve nor r is provided.
        """
        self.S = S
        self.K = K
        if curve is not None:
            self.curve = curve
        elif r is not None:
            self.curve = FlatCurve(r)
        else:
            raise ValueError("Curve or rate must be provided")
        self.T = T
        self.vol = vol
        self.all_returns = all_returns
        self.option_type = option_type
    
    @property
    def r(self):
        """Risk-free rate derived from the yield curve at maturity T.

        Returns:
            Continuously compounded zero rate r(T) from the instrument's curve.
        """
        return self.curve.zero_rate(self.T)

    def copy_with(self, **kwargs):
        """Create a shallow copy of the instrument with overridden attributes.

        Used by the Greeks engine to create bumped instruments for
        finite-difference calculations without mutating the original.

        Args:
            **kwargs: Attribute names and their new values (e.g. S=101, vol=0.21).

        Returns:
            A new Instrument copy with the specified attributes replaced.
        """
        clone = copy.copy(self)

        for key, value in kwargs.items():
            setattr(clone, key, value)

        return clone

    def intrinsic_value(self, S):
        """Compute the intrinsic (exercise) value of the option at given spot prices.

        Used by the BinomialTree for terminal payoffs and early-exercise checks.

        Args:
            S: Spot price(s) — scalar or numpy array of asset prices.

        Returns:
            Intrinsic value(s): max(S - K, 0) for calls, max(K - S, 0) for puts.
        """
        if self.option_type == "c":
            payoff = np.maximum(S - self.K, 0)
        else:
            payoff = np.maximum(self.K - S, 0)

        return payoff

class MultiAssetInstrument:
    """Base class for multi-asset derivative instruments.
    
    Stores arrays of spot prices and volatilities, a correlation matrix,
    and a yield curve. Validates that the correlation matrix is symmetric,
    positive semi-definite, and has unit diagonal. All multi-asset payoff
    classes (Basket, Rainbow, Spread) inherit from this.
    """

    def __init__(self, S_arr: np.ndarray, vol_arr: np.ndarray, corr_matrix: np.ndarray, T: float, K: float, option_type: str, weights: np.ndarray = None, r: float = None, curve: YieldCurve = None, all_returns: np.ndarray = None):
        """Initialise a multi-asset instrument with validation.

        Validates that the correlation matrix is positive semi-definite,
        symmetric, has diagonal entries equal to 1, and that all input
        arrays have consistent lengths.

        Args:
            S_arr: Array of spot prices for each underlying asset.
            vol_arr: Array of annualised volatilities for each asset.
            corr_matrix: Correlation matrix (k x k) between the assets.
            T: Time to maturity in years.
            K: Strike price of the derivative.
            option_type: 'c' for call, 'p' for put.
            weights: Portfolio weights for each asset. Defaults to equal weights.
            r: Constant risk-free rate. Wrapped in a FlatCurve if no curve given.
            curve: YieldCurve instance for discounting. Takes priority over r.
            all_returns: Simulated price paths, shape (k, n_steps+1, n_paths). Set by the pricer.

        Raises:
            ValueError: If volatilities are non-positive, correlation matrix is
                invalid, neither curve nor r is given, or array lengths mismatch.
        """
        if np.all(vol_arr > 0):
            self.vol_arr = vol_arr
        else:
            raise ValueError("Volatility must be positive")

        if np.linalg.eigvalsh(corr_matrix).min() >= 0:
            self.corr_matrix = corr_matrix
        else:
            raise ValueError("Correlation matrix is not positive semi-definite")
        
        if np.allclose(np.diag(corr_matrix), 1):
            self.corr_matrix = corr_matrix
        else:
            raise ValueError("Correlation matrix diagonal is not equal to 1")
        
        if np.allclose(corr_matrix, corr_matrix.T):
            self.corr_matrix = corr_matrix
        else:
            raise ValueError("Correlation matrix is not symmetric")
            
        if curve is not None:
            self.curve = curve
        elif r is not None:
            self.curve = FlatCurve(r)
        else:
            raise ValueError("Curve or rate must be provided")
        
        if weights is None:
            self.weights = np.ones(len(S_arr)) / len(S_arr)
        else:
            self.weights = weights

        if len(S_arr) != len(vol_arr) or len(S_arr) != len(corr_matrix) or len(S_arr) != len(self.weights):
            raise ValueError("Arrays must have the same length")

        self.S_arr = S_arr
        self.T = T
        self.K = K
        self.option_type = option_type
        self.all_returns = all_returns

    def copy_with(self, **kwargs):
        """Create a shallow copy of the instrument with overridden attributes.

        Used by MultiAssetGreeks to create bumped instruments for
        finite-difference calculations without mutating the original.

        Args:
            **kwargs: Attribute names and their new values (e.g. S_arr=new_spots).

        Returns:
            A new MultiAssetInstrument copy with the specified attributes replaced.
        """
        clone = copy.copy(self)

        for key, value in kwargs.items():
            setattr(clone, key, value)

        return clone