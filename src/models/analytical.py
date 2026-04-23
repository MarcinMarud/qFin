"""
Analytical pricing models: Black-Scholes-Merton for European options
and cost-of-carry model for forwards/futures pricing.
"""

import numpy as np
from scipy import stats

class BlackScholes:
    """Black-Scholes-Merton analytical pricing model for European options.
    
    Provides closed-form solutions for option price and all first-order
    Greeks (delta, gamma, vega, theta, rho) under constant volatility
    and interest rate assumptions.
    """

    def __init__(self, S, K, T, vol, r, option_type):
        """Initialise the Black-Scholes model with market parameters.

        Args:
            S: Current spot price of the underlying.
            K: Strike price.
            T: Time to maturity in years.
            vol: Annualised volatility (e.g. 0.20 for 20%).
            r: Constant continuously compounded risk-free rate.
            option_type: 'c' for call, 'p' for put.
        """
        self.S = S
        self.K = K
        self.T = T
        self.vol = vol
        self.r = r
        self.option_type = option_type

    def _d1_d2(self):
        """Compute the d1 and d2 terms used in all BSM formulas.

        d1 = [ln(S/K) + (r + vol^2/2) * T] / (vol * sqrt(T))
        d2 = d1 - vol * sqrt(T)

        Returns:
            Tuple (d1, d2) as floats.
        """
        d1 = (np.log(self.S / self.K) + (self.r + self.vol**2 / 2) * self.T) / (self.vol * np.sqrt(self.T))
        d2 = d1 - self.vol * np.sqrt(self.T)

        return d1, d2

    def black_scholes(self):
        """Compute the Black-Scholes theoretical option price.

        Call: S*N(d1) - K*exp(-rT)*N(d2)
        Put:  K*exp(-rT)*N(-d2) - S*N(-d1)

        Returns:
            Theoretical option price as a float.
        """
        d1, d2 = self._d1_d2()

        if self.option_type == "c":
            theo = stats.norm.cdf(d1) * self.S - stats.norm.cdf(d2) * self.K * np.exp(-self.r * self.T)
        else:
            theo = self.K * np.exp(-self.r * self.T) * stats.norm.cdf(-d2) - self.S * stats.norm.cdf(-d1)
        
        return theo
    
    def black_scholes_delta(self):
        """Compute the option delta (sensitivity to spot price).

        Call delta: N(d1), range [0, 1].
        Put delta:  N(d1) - 1, range [-1, 0].

        Returns:
            Delta as a float.
        """
        d1, d2 = self._d1_d2()

        delta_call = stats.norm.cdf(d1)
        delta_put = stats.norm.cdf(d1) - 1
        delta = np.where(self.option_type == 'c', delta_call, delta_put)

        return delta
    
    def black_scholes_gamma(self):
        """Compute the option gamma (second derivative w.r.t. spot price).

        Gamma = N'(d1) / (S * vol * sqrt(T)). Same for calls and puts.

        Returns:
            Gamma as a float. Always non-negative.
        """
        d1, d2 = self._d1_d2()
        nprimed1 = stats.norm.pdf(d1)
        
        gamma = nprimed1 / (self.S * self.vol * np.sqrt(self.T))

        return gamma
    
    def black_scholes_vega(self):
        """Compute the option vega (sensitivity to volatility).

        Vega = S * N'(d1) * sqrt(T). Same for calls and puts.

        Returns:
            Vega as a float. Always non-negative.
        """
        d1, d2 = self._d1_d2()
        nprimed1 = stats.norm.pdf(d1)

        vega = self.S * nprimed1 * np.sqrt(self.T)

        return vega
    
    def black_scholes_theta(self):
        """Compute the option theta (sensitivity to passage of time).

        Measures the rate of time decay. Typically negative for long options.

        Returns:
            Theta as a float (per year, not per day).
        """
        d1, d2 = self._d1_d2()
        nprimed1 = stats.norm.pdf(d1)

        theta_call = -(self.S * nprimed1 * self.vol) / (2 * np.sqrt(self.T)) - self.r * self.K * np.exp(-self.r * self.T) * stats.norm.cdf(d2)
        theta_put = -(self.S * nprimed1 * self.vol) / (2 * np.sqrt(self.T)) + self.r * self.K * np.exp(-self.r * self.T) * stats.norm.cdf(-d2)
        theta = np.where(self.option_type == 'c', theta_call, theta_put)

        return theta
    
    def black_scholes_rho(self):
        """Compute the option rho (sensitivity to interest rate).

        Call rho: K * T * exp(-rT) * N(d2).
        Put rho: -K * T * exp(-rT) * N(-d2).

        Returns:
            Rho as a float.
        """
        d1, d2 = self._d1_d2()

        rho_call = self.K * self.T * np.exp(-self.r * self.T) * stats.norm.cdf(d2)
        rho_put = -self.K * self.T * np.exp(-self.r * self.T) * stats.norm.cdf(-d2)
        rho = np.where(self.option_type == 'c', rho_call, rho_put)


        return rho

class FixedContractsCalculations:
    """Cost-of-carry model for pricing forward and futures contracts.
    
    Computes the theoretical forward price based on spot price, risk-free
    rate, storage costs, and convenience yield using continuous compounding.
    """

    def __init__(self, S, r, storage_cost, convenience_yield, T):
        """Initialise the cost-of-carry calculator.

        Args:
            S: Current spot price of the underlying.
            r: Continuously compounded risk-free rate.
            storage_cost: Annualised storage cost as a continuous rate.
            convenience_yield: Annualised convenience yield as a continuous rate.
            T: Time to delivery in years.
        """
        self.S = S
        self.r = r
        self.storage_cost = storage_cost
        self.convenience_yield = convenience_yield
        self.T = T

    def cost_of_carry(self):
        """Compute the theoretical forward price using the cost-of-carry model.

        F = S * exp((r + storage_cost - convenience_yield) * T)

        Returns:
            Theoretical forward price as a float.
        """
        F = self.S * np.exp((self.r + self.storage_cost - self.convenience_yield) * self.T)

        return F