"""
Implied volatility solvers using iterative root-finding methods.
Provides Newton-Raphson, bisection, and a hybrid approach that
combines both for robust convergence.
"""

from models.analytical import BlackScholes

class ImpliedVolatility:
    """Implied volatility calculator using Black-Scholes as the pricing model.
    
    Finds the volatility that equates the BSM theoretical price to the
    observed market price. Offers three algorithms with different
    convergence and robustness trade-offs.
    """

    def __init__(self):
        """Initialise the implied volatility solver (stateless)."""
        pass

    def newton_raphson(self, S, K, T, r, option_type, market_price, tol=1e-6, sigma_init=0.2, max_iterations=1000):
        """Find implied volatility using Newton-Raphson iteration.

        Uses vega as the derivative of BSM price w.r.t. volatility.
        Converges quadratically near the solution but may diverge if
        the initial guess is far from the true value.

        Args:
            S: Current spot price of the underlying.
            K: Strike price.
            T: Time to maturity in years.
            r: Continuously compounded risk-free rate.
            option_type: 'c' for call, 'p' for put.
            market_price: Observed market price of the option.
            tol: Convergence tolerance on price difference. Defaults to 1e-6.
            sigma_init: Initial volatility guess. Defaults to 0.2 (20%).
            max_iterations: Maximum number of iterations. Defaults to 1000.

        Returns:
            Implied volatility as a float.
        """
        for _ in range(max_iterations):
            bs = BlackScholes(S, K, T, sigma_init, r, option_type)
            theo_price = bs.black_scholes()
            vega = bs.black_scholes_vega()
            diff = theo_price - market_price
            if abs(diff) < tol:
                break
            sigma_init = sigma_init - (theo_price - market_price) / vega

        return sigma_init

    def bisection(self, S, K, T, r, option_type, market_price, x_down=0, x_up=1, tol=1e-6):
        """Find implied volatility using the bisection method.

        Guaranteed to converge if the true IV lies within [x_down, x_up],
        but convergence is linear (slower than Newton-Raphson).

        Args:
            S: Current spot price of the underlying.
            K: Strike price.
            T: Time to maturity in years.
            r: Continuously compounded risk-free rate.
            option_type: 'c' for call, 'p' for put.
            market_price: Observed market price of the option.
            x_down: Lower bound for volatility search. Defaults to 0.
            x_up: Upper bound for volatility search. Defaults to 1.
            tol: Convergence tolerance on the bracket width. Defaults to 1e-6.

        Returns:
            Implied volatility as a float.
        """

        while x_up - x_down > tol:
            root = (x_up + x_down) / 2
            bs = BlackScholes(S, K, T, root, r, option_type)
            theo_price = bs.black_scholes()
            if theo_price > market_price:
                x_up = root
            else:
                x_down = root
        
        return root

    def hybrid_newton(self, S, K, T, r, option_type, market_price, init_vol=0.2, x_down=0, x_up=1, tol=1e-6, max_iteration=1000):
        """Find implied volatility using a hybrid Newton-bisection method.

        Combines Newton-Raphson speed with bisection safety. If vega is
        too small (near-zero), falls back to bisection. The Newton step
        is clamped to [x_down, x_up] to prevent divergence. The bracket
        is tightened at each iteration.

        Args:
            S: Current spot price of the underlying.
            K: Strike price.
            T: Time to maturity in years.
            r: Continuously compounded risk-free rate.
            option_type: 'c' for call, 'p' for put.
            market_price: Observed market price of the option.
            init_vol: Initial volatility guess. Defaults to 0.2.
            x_down: Lower bracket bound. Defaults to 0.
            x_up: Upper bracket bound. Defaults to 1.
            tol: Convergence tolerance on price difference. Defaults to 1e-6.
            max_iteration: Maximum number of iterations. Defaults to 1000.

        Returns:
            Implied volatility as a float.
        """
        for _ in range(max_iteration):
            
            bs = BlackScholes(S, K, T, init_vol, r, option_type)
            theo_price = bs.black_scholes()
            vega = bs.black_scholes_vega()

            if abs(vega) < tol:
                init_vol = (x_up + x_down) / 2
            else:
                init_vol = init_vol - (theo_price - market_price) / vega

            init_vol = max(x_down, min(x_up, init_vol))

            if theo_price > market_price:
                x_up = init_vol
            else:
                x_down = init_vol

            if abs(theo_price - market_price) < tol:
                break
        
        return init_vol