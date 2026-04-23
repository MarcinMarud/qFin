"""
Monte Carlo pricers that connect simulation engines with instrument payoffs.
MCPricer handles single-asset instruments; MultiAssetMCPricer handles
correlated multi-asset instruments via Cholesky Monte Carlo.
"""

import numpy as np
from market_data.yield_curve import YieldCurve


class MCPricer:
    """Monte Carlo pricer for single-asset derivatives.
    
    Orchestrates the pricing pipeline: runs the simulation engine,
    attaches paths to the instrument, computes payoffs, discounts,
    and returns the mean price across all paths.
    """

    def __init__(self, engine, instrument):
        """Initialise the MC pricer with an engine and instrument.

        Args:
            engine: A simulation engine (e.g. MonteCarlo or Heston) with a simulate() method.
            instrument: An Instrument subclass with get_payoff() and curve for discounting.
        """
        self.engine = engine
        self.instrument = instrument

    def price(self):
        """Simulate paths, compute payoffs, discount, and return the mean price.

        Runs the engine's simulate(), attaches the resulting paths to the
        instrument, calls get_payoff(), applies the discount factor, and
        averages across all Monte Carlo paths.

        Returns:
            Risk-neutral price of the instrument as a float.
        """
        all_returns = self.engine.simulate(S=self.instrument.S, T=self.instrument.T, vol=self.instrument.vol)
        self.instrument.all_returns = all_returns
        payoff = self.instrument.get_payoff()
        price = payoff * self.instrument.curve.discount_factor(self.instrument.T)
        price = np.mean(price)

        return price

class MultiAssetMCPricer:
    """Monte Carlo pricer for multi-asset derivatives using Cholesky simulation.
    
    Same pipeline as MCPricer but uses cholesky_monte_carlo() to generate
    correlated paths for basket, rainbow, and spread options.
    """

    def __init__(self, engine, instrument):
        """Initialise the multi-asset MC pricer with an engine and instrument.

        Args:
            engine: A MonteCarlo engine with cholesky_monte_carlo() method.
            instrument: A MultiAssetInstrument subclass with get_payoff() and curve.
        """
        self.engine = engine
        self.instrument = instrument

    def price(self):
        """Simulate correlated paths, compute payoffs, discount, and return mean price.

        Runs cholesky_monte_carlo() with the instrument's asset array,
        attaches paths, calls get_payoff(), discounts, and averages.

        Returns:
            Risk-neutral price of the multi-asset instrument as a float.
        """
        all_returns = self.engine.cholesky_monte_carlo(S_arr=self.instrument.S_arr, T=self.instrument.T, vol_arr=self.instrument.vol_arr, corr_matrix=self.instrument.corr_matrix)
        self.instrument.all_returns = all_returns
        payoff = self.instrument.get_payoff()
        price = payoff * self.instrument.curve.discount_factor(self.instrument.T)
        price = np.mean(price)

        return price
