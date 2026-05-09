# pyqfin

![PyPI](https://img.shields.io/pypi/v/pyqfin)
![License](https://img.shields.io/github/license/MarcinMarud/qFin)
![Python Version](https://img.shields.io/pypi/pyversions/pyqfin)

A professional, vectorised quantitative finance library for pricing options, forwards, futures, and multi-asset derivatives. `pyqfin` provides a simple dictionary-based `Pricer` API that wires together analytical models (Black-Scholes), numerical engines (Monte Carlo, Heston, Binomial Trees), and various payoff types to deliver fast and reliable valuations and risk metrics.

## Table of Contents
- [Installation](#installation)
- [Architecture Overview](#architecture-overview)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Mathematical Reference](#mathematical-reference)

## Installation

```bash
pip install pyqfin
```

## Architecture Overview

| Module | Description | Key Classes |
|--------|-------------|-------------|
| `pyqfin.pricer` | Quick dictionary-based API for end-users | `Pricer`, `PricingResult`, `PortfolioResult` |
| `pyqfin.models` | Pricing engines and numerical methods | `BlackScholes`, `MonteCarlo`, `Heston`, `BinominalTree`, `MCPricer` |
| `pyqfin.payoffs` | Instrument payoff logic | `VanillaOptions`, `AsianOptions`, `BasketOption`, `Futures`, etc. |
| `pyqfin.risk_management` | Sensitivities and calibration | `Greeks`, `MultiAssetGreeks`, `ImpliedVolatility` |
| `pyqfin.market_data` | Term structure and discounting | `YieldCurve`, `FlatCurve`, `InterpolatedCurve` |
| `pyqfin.portfolio` | Portfolio-level aggregation | `Portfolio` |

## Quick Start

### Single Option Pricing
The `Pricer` handles all the internal wiring (engines, yield curves, payoff classes).

```python
from pyqfin import Pricer

# Vanilla Call (defaults to analytical BSM)
result = Pricer({
    'type': 'vanilla',
    'option_type': 'c',
    'S': 100, 'K': 105, 'T': 1.0, 
    'vol': 0.20, 'r': 0.05,
    'greeks': True
}).run()

print(f"Price: {result.price:.4f}")
print(f"Delta: {result.greeks['delta']:.4f}")
```

### American Options
Automatically utilizes the binomial tree engine:

```python
# American Put
result = Pricer({
    'type': 'american',
    'option_type': 'p',
    'S': 100, 'K': 105, 'T': 1.0, 
    'vol': 0.20, 'r': 0.05,
    'n_steps': 500  # Tree depth
}).run()
print(f"American Premium Price: {result.price:.4f}")
```

### Asian & Barrier Options (Monte Carlo)
For path-dependent options, `Pricer` automatically selects Monte Carlo:

```python
result = Pricer({
    'type': 'barrier',
    'option_type': 'c',
    'barrier_price': 120,
    'barrier_kind': 'knock-out',
    'barrier_direction': 'up',
    'S': 100, 'K': 100, 'T': 1.0,
    'vol': 0.20, 'r': 0.05,
    'n_paths': 10000
}).run()
```

### Multi-Asset Basket Option
Provide arrays for `S` and `vol`, and a correlation matrix. The library automatically uses Cholesky-based Monte Carlo.

```python
import numpy as np

corr_matrix = np.array([
    [1.0, 0.6, 0.3], 
    [0.6, 1.0, 0.5], 
    [0.3, 0.5, 1.0]
])

result = Pricer({
    'type': 'basket',
    'option_type': 'c',
    'S': [100, 110, 90],
    'K': 100, 'T': 1.0,
    'vol': [0.20, 0.25, 0.30],
    'corr': corr_matrix,
    'weights': [1/3, 1/3, 1/3],
    'r': 0.05,
    'n_paths': 50000
}).run()
```

### Portfolio Pricing
Pass a list of configurations (must include `'quantity'`) to price an entire book at once and aggregate risks.

```python
portfolio = Pricer.portfolio([
    {'type': 'vanilla', 'option_type': 'c', 'S': 100, 'K': 105, 'T': 1.0, 'vol': 0.20, 'r': 0.05, 'quantity': 10},
    {'type': 'american', 'option_type': 'p', 'S': 100, 'K': 90, 'T': 0.5, 'vol': 0.25, 'r': 0.05, 'quantity': -5},
], greeks=True)

print(f"Total Portfolio Value: {portfolio.total_value:.2f}")
print(f"Net Delta: {portfolio.total_greeks['delta']:.2f}")
```

### Heston Model (Stochastic Volatility)
If you select the `heston` engine, provide the specific variance parameters instead of standard volatility.

```python
result = Pricer({
    'type': 'vanilla',
    'option_type': 'c',
    'S': 100, 'K': 105, 'T': 1.0, 'r': 0.05,
    'engine': 'heston',
    'v0': 0.04,        # initial variance
    'kappa': 2.0,      # mean-reversion speed
    'theta': 0.04,     # long-run variance
    'xi': 0.3,         # vol-of-vol
    'rho_heston': -0.7 # correlation
}).run()
```

## API Reference

### Quick Pricer

The `Pricer` class expects a configuration dictionary.

- `Pricer(config: dict).run() -> PricingResult`: Runs the pricing workflow for a single instrument.
- `Pricer.portfolio(configs: List[dict], greeks: bool) -> PortfolioResult`: Prices multiple instruments.

**Configuration Keys:**

| Key | Type | Description | Required For |
|-----|------|-------------|--------------|
| `type` | `str` | `'vanilla'`, `'asian'`, `'barrier'`, `'american'`, `'basket'`, `'rainbow'`, `'spread'`, `'forward'`, `'future'` | All |
| `S` | `float` \| `list` | Spot price (list for multi-asset) | All |
| `K` | `float` | Strike price | All |
| `T` | `float` | Time to maturity (years) | All |
| `r` | `float` | Risk-free rate | All |
| `vol` | `float` \| `list` | Annualised volatility | Non-Heston |
| `option_type` | `str` | `'c'` (call) or `'p'` (put) | Options |
| `engine` | `str` | `'bsm'`, `'mc'`, `'binomial'`, `'heston'` (optional, auto-inferred) | None |
| `greeks` | `bool` | Whether to calculate sensitivities | None |

### Pricing Engines

**BlackScholes (`pyqfin.models.analytical.BlackScholes`)**
- `__init__(S, K, T, vol, r, option_type)`
- `black_scholes() -> float`: Option price
- `black_scholes_delta() -> float`
- `black_scholes_gamma() -> float`
- `black_scholes_vega() -> float`
- `black_scholes_theta() -> float`
- `black_scholes_rho() -> float`

**MonteCarlo (`pyqfin.models.numerical.MonteCarlo`)**
- `__init__(n, M, curve, seed)`
- `van_monte_carlo(S, T, vol) -> ndarray`: Generates 1D paths with antithetics.
- `cholesky_monte_carlo(S_arr, T, vol_arr, corr_matrix) -> ndarray`: Generates correlated multi-asset paths.

**Heston (`pyqfin.models.numerical.Heston`)**
- `__init__(v0, kappa, theta, xi, rho, r, n, paths, seed)`
- `heston_model(S, T) -> Tuple[ndarray, ndarray]`: Simulates joint asset and variance paths.

**BinominalTree (`pyqfin.models.numerical.BinominalTree`)**
- `__init__(n_steps, curve)`
- `price(instrument, american=False) -> float`: Backward-induction pricing.

### Payoffs & Instruments
Found in `pyqfin.payoffs`. All inherit from `Instrument` or `MultiAssetInstrument`.
- `VanillaOptions`: max(S_T - K, 0)
- `AsianOptions`: max(avg(S) - K, 0)
- `BarrierOptions`: Knock-in / knock-out, up / down barriers.
- `AmericanOption`: Designed for `BinominalTree` pricing with early-exercise.
- `BasketOption`: Weighted sum of terminal asset prices.
- `RainbowOption`: Best-of or worst-of multiple assets.
- `SpreadOption`: Difference between two assets (S1_T - S2_T - K).
- `Forwards` & `Futures`: Linear S_T - K payoffs.

### Risk Management

**Greeks (`pyqfin.risk_management.greeks.Greeks`)**
- `finite_difference() -> Tuple[float, float, float, float, float]`: Computes delta, gamma, vega, theta, rho via bump-and-reprice.

**MultiAssetGreeks (`pyqfin.risk_management.greeks.MultiAssetGreeks`)**
- `finite_difference() -> dict`: Returns arrays for delta, gamma, vega (one per asset), and scalars for theta, rho.

**ImpliedVolatility (`pyqfin.risk_management.implied_volatility.ImpliedVolatility`)**
- `newton_raphson(S, K, T, r, option_type, market_price) -> float`
- `bisection(S, K, T, r, option_type, market_price) -> float`
- `hybrid_newton(...) -> float`: Robust solver combining both.

### Market Data
Found in `pyqfin.market_data.yield_curve`.
- `YieldCurve`: Abstract base class with `discount_factor`, `zero_rate`, `forward_rate`.
- `FlatCurve(r)`: Constant rate implementation.
- `InterpolatedCurve(tenors, zero_rates)`: Cubic spline term structure.

## Mathematical Reference

### Black-Scholes-Merton
European option pricing in a continuous-time log-normal diffusion framework.

Call: `C(S, t) = S * N(d1) - K * exp(-r * (T-t)) * N(d2)`
Put:  `P(S, t) = K * exp(-r * (T-t)) * N(-d2) - S * N(-d1)`

Where:
`d1 = [ln(S/K) + (r + (vol^2)/2) * (T-t)] / (vol * sqrt(T-t))`
`d2 = d1 - vol * sqrt(T-t)`

### Monte Carlo Simulation
Under the risk-neutral measure, the asset price follows Geometric Brownian Motion (GBM). Discretised via Euler scheme:

`S_next = S * exp((r - (vol^2)/2) * dt + vol * sqrt(dt) * Z)`

Where `Z` is a standard normal random variable. We apply antithetic variates by simulating path pairs with `+Z` and `-Z`.

**Cholesky Multi-Asset**: To simulate `k` correlated assets with correlation matrix `P`, we perform Cholesky decomposition `P = L * L.T` and multiply independent normals `Z` by `L`:
`Z_corr = L * Z`

### Heston Model
Stochastic volatility framework where the variance follows a CIR (Cox-Ingersoll-Ross) mean-reverting process:

`dS = r * S * dt + sqrt(v) * S * dW_S`
`dv = kappa * (theta - v) * dt + xi * sqrt(v) * dW_v`

With `dW_S * dW_v = rho * dt`.

### Binomial Tree (CRR)
Cox-Ross-Rubinstein lattice parameters:
`u = exp(vol * sqrt(dt))`
`d = 1 / u`
`p = (exp(r * dt) - d) / (u - d)`

Backward induction at each node `i`:
`V_i = exp(-r * dt) * (p * V_up + (1-p) * V_down)`

For American options, early exercise implies:
`V_i = max(V_i, IntrinsicValue)`

### Finite-Difference Greeks
- **Delta**: `(V(S + dS) - V(S - dS)) / (2 * dS)`
- **Gamma**: `(V(S + dS) - 2*V(S) + V(S - dS)) / (dS^2)`
- **Vega**: `(V(vol + dVol) - V(vol - dVol)) / (2 * dVol)`
- **Theta**: `(V(T - dT) - V(T)) / (-dT)`
- **Rho**: `(V(r + dr) - V(r - dr)) / (2 * dr)`

### Implied Volatility
Newton-Raphson update step:
`vol_next = vol_current - (BSM(vol_current) - MarketPrice) / Vega(vol_current)`

### Yield Curve
Discount factor and zero rate relationship:
`D(0, t) = exp(-r(t) * t)`  <=>  `r(t) = -ln(D(0, t)) / t`

Forward rate between `t1` and `t2`:
`f(t1, t2) = -ln(D(0, t2) / D(0, t1)) / (t2 - t1)`

## Running Tests
If you cloned the repository and want to run the tests locally:
```bash
pip install pyqfin[dev]
pytest tests/ -v
```
