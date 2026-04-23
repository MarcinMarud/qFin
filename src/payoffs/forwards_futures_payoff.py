"""
Forward and futures contract payoff definitions. Both have linear payoffs
equal to S_T - K at maturity, priced under the risk-neutral measure via
Monte Carlo simulation.
"""

from payoffs.instruments import Instrument
import numpy as np

class Forwards(Instrument):
    """Forward contract with linear payoff S_T - K at maturity.
    
    The holder is obligated to buy the underlying at price K at time T.
    Payoff can be negative (unlike options).
    """

    def __init__(self, S, K, T, vol, curve=None, r=None, all_returns=None):
        """Initialise a forward contract.

        Args:
            S: Current spot price of the underlying.
            K: Delivery price (forward price agreed at inception).
            T: Time to delivery in years.
            vol: Annualised volatility of the underlying.
            curve: YieldCurve for discounting. Takes priority over r.
            r: Constant risk-free rate (used if curve is None).
            all_returns: Simulated paths, shape (n_steps+1, n_paths). Set by pricer.
        """
        super().__init__(S, K, T, vol, curve=curve, r=r, all_returns=all_returns)

    def get_payoff(self):
        """Compute the forward contract payoff at maturity: S_T - K.

        Returns:
            Array of payoffs across all Monte Carlo paths, shape (n_paths,).
        """
        payoff = self.all_returns[-1] - self.K

        return payoff

class Futures(Instrument):
    """Futures contract with linear payoff S_T - K at maturity.
    
    Modelled identically to a forward in this library. In practice,
    futures differ by daily margining (mark-to-market), which is not
    captured here.
    """

    def __init__(self, S, K, T, vol, curve=None, r=None, all_returns=None):
        """Initialise a futures contract.

        Args:
            S: Current spot price of the underlying.
            K: Futures price agreed at inception.
            T: Time to delivery in years.
            vol: Annualised volatility of the underlying.
            curve: YieldCurve for discounting. Takes priority over r.
            r: Constant risk-free rate (used if curve is None).
            all_returns: Simulated paths, shape (n_steps+1, n_paths). Set by pricer.
        """
        super().__init__(S, K, T, vol, curve=curve, r=r, all_returns=all_returns)

    def get_payoff(self):
        """Compute the futures contract payoff at maturity: S_T - K.

        Returns:
            Array of payoffs across all Monte Carlo paths, shape (n_paths,).
        """
        payoff = self.all_returns[-1] - self.K

        return payoff