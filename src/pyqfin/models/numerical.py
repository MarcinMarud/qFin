"""
Numerical pricing engines: Monte Carlo simulation (GBM and Cholesky correlated),
Heston stochastic volatility model, and CRR binomial tree for American options.
"""

import numpy as np
from pyqfin.market_data.yield_curve import YieldCurve
from pyqfin.payoffs.instruments import Instrument

class MonteCarlo:
    """Monte Carlo simulation engine for single-asset and multi-asset GBM paths.
    
    Generates risk-neutral price paths using geometric Brownian motion with
    antithetic variates for variance reduction. Supports term-structure-aware
    drift via instantaneous forward rates from the yield curve.
    """

    def __init__(self, n, M, curve: YieldCurve, seed=None):
        """Initialise the Monte Carlo engine.

        Args:
            n: Number of time steps per path.
            M: Total number of Monte Carlo paths (half original + half antithetic).
            curve: YieldCurve used to compute the risk-neutral drift at each step.
            seed: Random seed for reproducibility. None for non-deterministic.
        """
        self.n = n
        self.M = M
        self.curve = curve
        self.seed = seed

    def simulate(self, **kwargs):
        """Dispatch method that extracts parameters and runs vanilla MC simulation.

        Args:
            **kwargs: Must contain 'S' (spot), 'T' (maturity), 'vol' (volatility).

        Returns:
            Simulated price paths, shape (n+1, M). Row 0 is the initial spot.
        """
        S = kwargs['S']
        T = kwargs['T']
        vol = kwargs['vol']
        paths = self.van_monte_carlo(S, T, vol)

        return paths

    def van_monte_carlo(self, S, T, vol):
        """Generate GBM price paths with antithetic variates.

        Uses log-normal dynamics with term-structure-aware drift from the
        yield curve's instantaneous forward rates. Produces M/2 original
        paths and M/2 antithetic paths for variance reduction.

        Args:
            S: Initial spot price.
            T: Time to maturity in years.
            vol: Annualised volatility.

        Returns:
            Simulated price paths, shape (n+1, M). Row 0 is the initial spot S.
        """
        dt = T / self.n

        fwd_rates = np.array([self.curve.instantaneous_forward(i * dt) for i in range(self.n)])
        drift = (fwd_rates - vol**2/2) * dt
        shock = vol * np.sqrt(dt)

        self.rng = np.random.default_rng(self.seed)
        Z = self.rng.standard_normal(size=(int(self.M / 2), self.n))

        path_1 = np.exp(drift + shock * Z).T
        path_2 = np.exp(drift + shock * (-Z)).T

        St_1 = S * np.cumprod(path_1, axis=0)
        St_2 = S * np.cumprod(path_2, axis=0)

        St_1 = np.concatenate(
            (np.full(shape=(1, int(self.M / 2)), fill_value=S), St_1))
        St_2 = np.concatenate(
            (np.full(shape=(1, int(self.M / 2)), fill_value=S), St_2))

        all_returns = np.hstack([St_1, St_2])

        return all_returns

    def cholesky_monte_carlo(self, S_arr, T, vol_arr, corr_matrix):
        """Generate correlated multi-asset GBM paths using Cholesky decomposition.

        Produces correlated Brownian motions by applying the Cholesky factor
        of the correlation matrix to independent normals. Uses antithetic
        variates across all assets simultaneously.

        Args:
            S_arr: Array of initial spot prices for each asset, shape (k,).
            T: Time to maturity in years.
            vol_arr: Array of annualised volatilities for each asset, shape (k,).
            corr_matrix: Correlation matrix between assets, shape (k, k).

        Returns:
            Simulated paths, shape (k, n+1, M). Axis 0 is the asset index,
            axis 1 is the time step, axis 2 is the path index.
        """
        k = len(S_arr)

        dt = T / self.n

        L = np.linalg.cholesky(corr_matrix)

        self.rng = np.random.default_rng(self.seed)
        Z = self.rng.standard_normal(size=(k, int(self.M / 2), self.n))
        Z_anti = -Z

        Z_total = np.concatenate((Z, Z_anti), axis=1)

        Z_corr = np.tensordot(L, Z_total, axes=(1, 0))

        vol_arr = np.array(vol_arr).reshape(k, 1, 1)
        S_arr = np.array(S_arr).reshape(k, 1, 1)

        fwd_rates = np.array([self.curve.instantaneous_forward(i * dt) for i in range(self.n)])
        drift = (fwd_rates - vol_arr**2/2) * dt
        shock = vol_arr * np.sqrt(dt)

        all_returns = np.exp(drift + shock * Z_corr)
        all_returns = np.cumprod(all_returns, axis=2) * S_arr
        all_returns_0 = np.ones((k, self.M, 1)) * S_arr
        all_returns = np.concatenate((all_returns_0, all_returns), axis=2)
        all_returns = all_returns.transpose(0, 2, 1)

        return all_returns
    
class Heston:
    """Heston stochastic volatility model with Euler discretisation.
    
    Simulates joint dynamics of the asset price and its variance using
    correlated Brownian motions. The variance process follows a CIR
    (Cox-Ingersoll-Ross) mean-reverting square-root diffusion.
    """

    def __init__(self, v0, kappa, theta, xi, rho, r, n, paths, seed=None):
        """Initialise the Heston model engine.

        Args:
            v0: Initial variance (not volatility) at t=0.
            kappa: Mean-reversion speed of the variance process.
            theta: Long-run variance level that the process reverts to.
            xi: Volatility of volatility (vol-of-vol).
            rho: Correlation between the asset and variance Brownian motions.
            r: Constant risk-free rate used for the drift.
            n: Number of time steps per path.
            paths: Number of Monte Carlo paths to simulate.
            seed: Random seed for reproducibility.
        """
        self.n = n
        self.kappa = kappa
        self.theta = theta
        self.paths = paths
        self.r = r
        self.xi = xi
        self.rho = rho
        self.v0 = v0
        self.seed = seed

    def simulate(self, **kwargs):
        """Dispatch method that extracts parameters and runs Heston simulation.

        Args:
            **kwargs: Must contain 'S' (spot) and 'T' (maturity).

        Returns:
            Simulated price paths, shape (n+1, paths).
        """
        S = kwargs['S']
        T = kwargs['T']
        paths = self.heston_model(S, T)[0]

        return paths
    
    def heston_model(self, S, T):
        """Simulate asset price and variance paths under the Heston model.

        Uses Euler-Maruyama discretisation with truncation (variance floored
        at zero) to prevent negative variance.

        Args:
            S: Initial spot price.
            T: Time to maturity in years.

        Returns:
            Tuple of (price_paths, variance_paths), each shape (n+1, paths).
            Rows are time steps, columns are paths.
        """
        dt = T / self.n

        all_returns = np.zeros((self.paths, self.n + 1))
        all_volatilities = np.zeros((self.paths, self.n + 1))

        all_returns[:, 0] = S
        all_volatilities[:, 0] = self.v0

        self.rng = np.random.default_rng(self.seed)

        for t in range (1, self.n + 1):
            Z1 = self.rng.standard_normal(self.paths)
            Z2 = self.rng.standard_normal(self.paths)

            dW_S = Z1
            dW_v = Z1 * self.rho + np.sqrt(1 - self.rho**2) *  Z2
                
            v_prev = all_volatilities[:, t - 1]
            dv = self.kappa * (self.theta - v_prev) * dt + self.xi * np.sqrt(v_prev) * np.sqrt(dt) * dW_v
            all_volatilities[:, t] = np.maximum(v_prev + dv, 0)

            s_prev = all_returns[:, t - 1]
            dS = (self.r - v_prev/2) * dt + np.sqrt(v_prev) * np.sqrt(dt) * dW_S
            all_returns[:, t] = s_prev * np.exp(dS)

        all_returns = all_returns.T
        all_volatilities = all_volatilities.T
        
        return all_returns, all_volatilities   

class BinominalTree:
    """Cox-Ross-Rubinstein (CRR) binomial tree for European and American options.
    
    Builds a recombining binomial lattice and prices via backward induction.
    For American options, compares the continuation value against the
    intrinsic value at each node to determine optimal early exercise.
    """

    def __init__(self, n_steps: int, curve: YieldCurve):
        """Initialise the binomial tree engine.

        Args:
            n_steps: Number of time steps in the tree (higher = more accurate).
            curve: YieldCurve used for discounting (stored for consistency).
        """
        self.n_steps = n_steps
        self.curve = curve

    def price(self, instrument: Instrument, american: bool = False) -> float:
        """Price an option using CRR backward induction.

        Builds a recombining tree of asset prices, computes terminal payoffs,
        and steps backward through the tree discounting at each node. For
        American options, checks early exercise at every intermediate node.

        Args:
            instrument: An Instrument (or subclass) providing S, K, T, vol, r,
                option_type, and intrinsic_value().
            american: If True, apply early-exercise logic at each node.
                If False, price as European.

        Returns:
            The option price at t=0 (scalar float).
        """
        S, K, T, vol = instrument.S, instrument.K, instrument.T, instrument.vol
        dt = T / self.n_steps
        df = np.exp(-instrument.r * dt)
        u = np.exp(vol * np.sqrt(dt))
        d = np.exp(-vol * np.sqrt(dt))
        p = (np.exp(instrument.r * dt) - d) / (u - d)

        # makes an array from 0 to number of steps + 1
        j = np.arange(0, self.n_steps + 1)
        # calculates the terminal price at final step by combination of up and down moves
        S_T = S * u**j * d**(self.n_steps - j)
        
        # calls a method that check what type of option it is and calculates the intrinsic value
        V = instrument.intrinsic_value(S_T)

        # backward loop, we are going from the final step one step behind 
        for i in range(self.n_steps - 1, -1, -1):
            # calculating V as a discounting factor * weighted value of the upper and lower node at step i
            V = df * (p * V[1: i + 2] + (1 - p) * V[0: i + 1])
            # check if it's an american option 
            if american:
                # calculates the value of all nodes either up or down at the step i
                S_nodes = S * u**np.arange(i + 1) * d**(i - np.arange(i + 1))
                # calls instruments' method that check what type of option it is and calculates the intrinsic value
                intrinsic = instrument.intrinsic_value(S_nodes)
                # check if value of holding an option is greater than exercising it right now
                V = np.maximum(V, intrinsic)

        return V[0]
