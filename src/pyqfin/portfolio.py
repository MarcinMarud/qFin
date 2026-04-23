"""
Portfolio management module for aggregating positions, computing
total portfolio value, and calculating portfolio-level Greeks.
"""

from pyqfin.models.mc_pricer import MCPricer
from pyqfin.risk_management.greeks import Greeks

class Portfolio:
    """Collection of derivative positions with aggregated pricing and risk.
    
    Holds a list of (instrument, quantity) pairs and provides methods
    to compute the total portfolio value and net Greeks across all
    positions using Monte Carlo re-pricing.
    """

    def __init__(self):
        """Initialise an empty portfolio with no positions."""
        self.positions = []
    
    def add(self, instrument, quantity):
        """Add a derivative position to the portfolio.

        Args:
            instrument: An Instrument subclass representing the derivative.
            quantity: Number of contracts. Positive for long, negative for short.

        Returns:
            The updated list of (instrument, quantity) tuples.
        """
        self.positions.append((instrument, quantity))

        return self.positions

    def portfolio_value(self, engine):
        """Compute the total mark-to-market value of the portfolio.

        Prices each position using MCPricer and sums quantity * price.

        Args:
            engine: Monte Carlo engine to use for pricing each instrument.

        Returns:
            Total portfolio value as a float.
        """
        total = 0
        
        for inst, q in self.positions:
            price = MCPricer(engine, inst).price()
            total += (price * q)

        return total

    def portfolio_greeks(self, engine):
        """Compute the aggregate portfolio Greeks across all positions.

        Calculates finite-difference Greeks for each position, scales
        by quantity, and sums to produce net portfolio sensitivities.

        Args:
            engine: Monte Carlo engine to use for Greek calculations.

        Returns:
            Dict with keys 'delta', 'gamma', 'vega', 'theta', 'rho',
            each containing the net portfolio-level sensitivity.
        """
        greeks = {
            'delta': 0,
            'gamma': 0,
            'vega': 0,
            'theta': 0,
            'rho': 0
        }

        for inst, q in self.positions:
            greek = Greeks(engine, inst)
            delta, gamma, vega, theta, rho = greek.finite_difference()
            delta = delta * q
            gamma = gamma * q
            vega = vega * q
            theta = theta * q
            rho = rho * q

            greeks['delta'] += delta
            greeks['gamma'] += gamma
            greeks['vega'] += vega
            greeks['theta'] += theta
            greeks['rho'] += rho

        return greeks
