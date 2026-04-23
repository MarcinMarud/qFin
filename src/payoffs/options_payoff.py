"""
Option payoff definitions for single-asset and multi-asset derivatives.
Each class defines a get_payoff() method that computes the terminal payoff
from simulated price paths set by the Monte Carlo pricer.
"""

from payoffs.instruments import Instrument, MultiAssetInstrument
import numpy as np

class VanillaOptions(Instrument):
    """European vanilla option with standard call/put payoff at maturity.
    
    Payoff: max(S_T - K, 0) for calls, max(K - S_T, 0) for puts.
    """

    def __init__(self, S, K, T, vol, curve=None, r=None, option_type=None, all_returns=None):
        """Initialise a vanilla European option.

        Args:
            S: Current spot price of the underlying.
            K: Strike price.
            T: Time to maturity in years.
            vol: Annualised volatility.
            curve: YieldCurve for discounting. Takes priority over r.
            r: Constant risk-free rate (used if curve is None).
            option_type: 'c' for call, 'p' for put.
            all_returns: Simulated paths, shape (n_steps+1, n_paths). Set by pricer.
        """
        super().__init__(S, K, T, vol, curve, r, option_type, all_returns)

    def get_payoff(self):
        """Compute the vanilla payoff from the terminal simulated prices.

        Uses the last row of all_returns as the terminal spot prices S_T.

        Returns:
            Array of payoffs across all Monte Carlo paths, shape (n_paths,).
        """
        if self.option_type == "c":
            payoff = np.maximum(0, self.all_returns[-1] - self.K)
        else:
            payoff = np.maximum(0, self.K - self.all_returns[-1])

        return payoff

class AsianOptions(Instrument):
    """Asian option whose payoff depends on the average price over the path.
    
    Payoff: max(avg(S) - K, 0) for calls, max(K - avg(S), 0) for puts,
    where the average is taken over all time steps excluding t=0.
    """

    def __init__(self, S, K, T, vol, curve=None, r=None, option_type=None, all_returns=None):
        """Initialise an Asian option.

        Args:
            S: Current spot price of the underlying.
            K: Strike price.
            T: Time to maturity in years.
            vol: Annualised volatility.
            curve: YieldCurve for discounting. Takes priority over r.
            r: Constant risk-free rate (used if curve is None).
            option_type: 'c' for call, 'p' for put.
            all_returns: Simulated paths, shape (n_steps+1, n_paths). Set by pricer.
        """
        super().__init__(S, K, T, vol, curve, r, option_type, all_returns)

    def get_payoff(self):
        """Compute the Asian payoff using the arithmetic mean of the path.

        Averages all_returns[1:] (excluding the initial spot at index 0)
        across the time axis for each Monte Carlo path.

        Returns:
            Array of payoffs across all Monte Carlo paths, shape (n_paths,).
        """
        if self.option_type == "c":
            payoff = np.maximum(0, np.mean(self.all_returns[1:], axis=0) - self.K)
        else:
            payoff = np.maximum(0, self.K - np.mean(self.all_returns[1:], axis=0))

        return payoff
    

class BarrierOptions(Instrument):
    """Barrier option whose payoff depends on whether the price path
    breaches a specified barrier level during the life of the option.
    
    Supports knock-in and knock-out variants with up or down barriers.
    """

    def __init__(self, S, K, T, vol, barrier_price, barrier_kind, barrier_direction, curve=None, r=None, option_type=None, all_returns=None):
        """Initialise a barrier option.

        Args:
            S: Current spot price of the underlying.
            K: Strike price.
            T: Time to maturity in years.
            vol: Annualised volatility.
            barrier_price: The barrier level that triggers activation/deactivation.
            barrier_kind: 'knock-out' (option dies if breached) or 'knock-in' (option activates if breached).
            barrier_direction: 'up' (barrier above spot) or 'down' (barrier below spot).
            curve: YieldCurve for discounting. Takes priority over r.
            r: Constant risk-free rate (used if curve is None).
            option_type: 'c' for call, 'p' for put.
            all_returns: Simulated paths, shape (n_steps+1, n_paths). Set by pricer.
        """
        super().__init__(S, K, T, vol, curve, r, option_type, all_returns)
        self.barrier_price = barrier_price
        self.barrier_direction = barrier_direction
        self.barrier_kind = barrier_kind

    def get_payoff(self):
        """Compute the barrier option payoff, zeroing paths based on barrier breach.

        First checks whether each path breached the barrier, then computes
        the vanilla payoff, and finally applies the knock-in/knock-out logic.

        Returns:
            Array of payoffs across all Monte Carlo paths, shape (n_paths,).

        Raises:
            ValueError: If barrier_direction or barrier_kind is not recognised.
        """
        if self.barrier_direction == "up":
            breached = np.max(self.all_returns, axis=0) >= self.barrier_price
        elif self.barrier_direction == "down":
            breached = np.min(self.all_returns, axis=0) <= self.barrier_price
        else:
            raise ValueError()

        if self.option_type == "c":
            payoff = np.maximum(0, self.all_returns[-1] - self.K)
        else:
            payoff = np.maximum(0, self.K - self.all_returns[-1])
        
        if self.barrier_kind == "knock-out":
            payoff = np.where(breached, 0, payoff)
        elif self.barrier_kind == "knock-in":
            payoff = np.where(breached, payoff, 0)
        else:
            raise ValueError()

        return payoff

class BasketOption(MultiAssetInstrument):
    """Basket option whose payoff depends on a weighted sum of multiple assets.
    
    The basket price is computed as the weighted average of terminal
    asset prices. Supports absolute and relative strike conventions.
    """
        
    def __init__(self, S_arr, vol_arr, corr_matrix, T, K, option_type, weights=None, r=None, curve=None):
        """Initialise a basket option on multiple correlated assets.

        Args:
            S_arr: Array of spot prices for each underlying.
            vol_arr: Array of annualised volatilities for each asset.
            corr_matrix: Correlation matrix (k x k) between the assets.
            T: Time to maturity in years.
            K: Strike price (absolute) or strike multiplier (relative).
            option_type: 'c' for call, 'p' for put.
            weights: Portfolio weights for each asset. Defaults to equal weights.
            r: Constant risk-free rate (used if curve is None).
            curve: YieldCurve for discounting. Takes priority over r.
        """
        super().__init__(S_arr, vol_arr, corr_matrix, T, K, option_type, weights, r=r, curve=curve)

    def get_payoff(self, strike_type="absolute"):
        """Compute the basket option payoff from simulated multi-asset paths.

        The basket level is the weighted sum of terminal prices across assets.

        Args:
            strike_type: 'absolute' uses K directly; 'relative' uses K as a
                multiplier of the initial basket price.

        Returns:
            Array of payoffs across all Monte Carlo paths, shape (n_paths,).
        """
        basket_price = np.sum(self.weights.reshape(-1, 1) * self.all_returns[:, -1, :], axis=0)

        if strike_type == "absolute":
            strike_price = self.K
        else:
            strike_price = self.K * basket_price[0]

        if self.option_type == "c":
            payoff = np.maximum(0, basket_price - strike_price)
        else:
            payoff = np.maximum(0, strike_price - basket_price)

        return payoff

class RainbowOption(MultiAssetInstrument):
    """Rainbow option whose payoff depends on the best or worst performing
    asset from a basket of correlated underlyings.
    """

    def __init__(self, S_arr, vol_arr, corr_matrix, T, K, option_type, weights=None, r=None, curve=None):
        """Initialise a rainbow option on multiple correlated assets.

        Args:
            S_arr: Array of spot prices for each underlying.
            vol_arr: Array of annualised volatilities for each asset.
            corr_matrix: Correlation matrix (k x k) between the assets.
            T: Time to maturity in years.
            K: Strike price.
            option_type: 'c' for call, 'p' for put.
            weights: Portfolio weights (unused in payoff, kept for consistency).
            r: Constant risk-free rate (used if curve is None).
            curve: YieldCurve for discounting. Takes priority over r.
        """
        super().__init__(S_arr, vol_arr, corr_matrix, T, K, option_type, weights, r=r, curve=curve)

    def get_payoff(self, rainbow_type="best"):
        """Compute the rainbow option payoff by selecting the best or worst asset.

        Args:
            rainbow_type: 'best' selects the maximum terminal price across
                assets; 'worst' selects the minimum.

        Returns:
            Array of payoffs across all Monte Carlo paths, shape (n_paths,).
        """
        terminal = self.all_returns[:, -1, :]

        if rainbow_type == "best":
            selected = np.max(terminal, axis=0)
        else:
            selected = np.min(terminal, axis=0)

        if self.option_type == "c":
            payoff = np.maximum(0, selected - self.K)
        else:
            payoff = np.maximum(0, self.K - selected)

        return payoff

class SpreadOption(MultiAssetInstrument):
    """Spread option whose payoff depends on the difference between two assets.
    
    Payoff: max(S1_T - S2_T - K, 0) for calls. Requires exactly 2 assets.
    """

    def __init__(self, S_arr, vol_arr, corr_matrix, T, K, option_type, weights=None, r=None, curve=None):
        """Initialise a spread option on exactly two correlated assets.

        Args:
            S_arr: Array of two spot prices [S1, S2].
            vol_arr: Array of two annualised volatilities [vol1, vol2].
            corr_matrix: 2x2 correlation matrix between the two assets.
            T: Time to maturity in years.
            K: Spread strike (applied to the difference S1 - S2).
            option_type: 'c' for call, 'p' for put.
            weights: Portfolio weights (unused, kept for consistency).
            r: Constant risk-free rate (used if curve is None).
            curve: YieldCurve for discounting. Takes priority over r.
        """
        super().__init__(S_arr, vol_arr, corr_matrix, T, K, option_type, weights, r=r, curve=curve)

    def get_payoff(self):
        """Compute the spread option payoff from the difference of two terminal prices.

        Returns:
            Array of payoffs across all Monte Carlo paths, shape (n_paths,).

        Raises:
            ValueError: If the instrument does not have exactly 2 assets.
        """
        if len(self.S_arr) != 2:
            raise ValueError("Spread option requires exactly 2 assets")
        terminal_one = self.all_returns[0, -1, :]
        terminal_two = self.all_returns[1, -1, :]

        if self.option_type == "c":
            payoff = np.maximum(0, (terminal_one - terminal_two) - self.K)
        else:
            payoff = np.maximum(0, self.K - (terminal_one - terminal_two))

        return payoff

class AmericanOption(Instrument):
    """American option with early-exercise right, priced via binomial tree.
    
    The get_payoff() method provides the European terminal payoff for MC
    compatibility. For true American pricing, use BinomialTree.price()
    which leverages the inherited intrinsic_value() method at each node.
    """

    def __init__(self, S, K, T, vol, curve=None, r=None, option_type=None, all_returns=None):
        """Initialise an American option.

        Args:
            S: Current spot price of the underlying.
            K: Strike price.
            T: Time to maturity in years.
            vol: Annualised volatility.
            curve: YieldCurve for discounting. Takes priority over r.
            r: Constant risk-free rate (used if curve is None).
            option_type: 'c' for call, 'p' for put.
            all_returns: Simulated paths (for MC lower-bound pricing). Set by pricer.
        """
        super().__init__(S, K, T, vol, curve, r, option_type, all_returns)

    def get_payoff(self):
        """Compute the European terminal payoff (MC lower-bound for American).

        This returns the same payoff as a vanilla option. For proper
        American pricing with early exercise, use BinomialTree.price()
        with american=True.

        Returns:
            Array of payoffs across all Monte Carlo paths, shape (n_paths,).
        """
        if self.option_type == "c":
            payoff = np.maximum(0, self.all_returns[-1] - self.K)
        else:
            payoff = np.maximum(0, self.K - self.all_returns[-1])

        return payoff