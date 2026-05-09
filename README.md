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

### Vanilla Option (BSM)
The `Pricer` handles all internal wiring (engines, yield curves, payoff classes).

```python
from pyqfin import Pricer

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

### American Option (Binomial Tree)
Automatically utilizes the binomial tree engine:

```python
result = Pricer({
    'type': 'american',
    'option_type': 'p',
    'S': 100, 'K': 105, 'T': 1.0, 
    'vol': 0.20, 'r': 0.05,
    'n_steps': 500
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

### Multi-Asset Options (Basket / Rainbow / Spread)
Provide arrays for `S` and `vol`, and a correlation matrix. Uses Cholesky-based Monte Carlo.

```python
import numpy as np

corr = np.array([[1.0, 0.6], [0.6, 1.0]])

result = Pricer({
    'type': 'spread',
    'option_type': 'c',
    'S': [100, 95],
    'K': 5, 'T': 1.0,
    'vol': [0.20, 0.25],
    'corr': corr,
    'r': 0.05
}).run()
```

### Heston Stochastic Volatility
Provide specific variance parameters instead of constant volatility.

```python
result = Pricer({
    'type': 'vanilla',
    'option_type': 'c',
    'S': 100, 'K': 105, 'T': 1.0, 'r': 0.05,
    'engine': 'heston',
    'v0': 0.04, 'kappa': 2.0, 'theta': 0.04,
    'xi': 0.3, 'rho_heston': -0.7
}).run()
```

### Portfolio Pricing
Pass a list of configurations with `'quantity'` to price an entire book at once.

```python
portfolio = Pricer.portfolio([
    {'type': 'vanilla', 'option_type': 'c', 'S': 100, 'K': 105, 'T': 1.0, 'vol': 0.20, 'r': 0.05, 'quantity': 10},
    {'type': 'american', 'option_type': 'p', 'S': 100, 'K': 90, 'T': 0.5, 'vol': 0.25, 'r': 0.05, 'quantity': -5},
], greeks=True)

print(f"Total Portfolio Value: {portfolio.total_value:.2f}")
print(f"Net Delta: {portfolio.total_greeks['delta']:.2f}")
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

```math
C(S, t) = S_t N(d_1) - K e^{-r(T-t)} N(d_2)
```
```math
P(S, t) = K e^{-r(T-t)} N(-d_2) - S_t N(-d_1)
```
Where:
```math
d_{1,2} = \frac{\ln(S_t/K) + (r \pm \frac{\sigma^2}{2})(T-t)}{\sigma \sqrt{T-t}}
```

### Monte Carlo Simulation
Under the risk-neutral measure, the asset price follows Geometric Brownian Motion (GBM). Discretised via Euler scheme:

```math
S_{t+\Delta t} = S_t \exp\left( \left( r - \frac{\sigma^2}{2} \right)\Delta t + \sigma \sqrt{\Delta t} Z \right)
```
Where $Z \sim \mathcal{N}(0, 1)$. We apply antithetic variates by simulating path pairs with $+Z$ and $-Z$.

**Cholesky Multi-Asset**: To simulate $k$ correlated assets with correlation matrix $P$, we perform Cholesky decomposition $P = L L^T$ and multiply independent normals $\mathbf{Z}$ by $L$:
```math
\mathbf{Z}_{corr} = L \mathbf{Z}
```

### Heston Model
Stochastic volatility framework where the variance follows a CIR (Cox-Ingersoll-Ross) mean-reverting process:

```math
dS_t = r S_t dt + \sqrt{v_t} S_t dW_t^S
```
```math
dv_t = \kappa (\theta - v_t) dt + \xi \sqrt{v_t} dW_t^v
```
With $dW^S dW^v = \rho dt$.

### Binomial Tree (CRR)
Cox-Ross-Rubinstein lattice parameters:
```math
u = \exp(\sigma \sqrt{\Delta t}), \quad d = \frac{1}{u}, \quad p = \frac{\exp(r \Delta t) - d}{u - d}
```
Backward induction at each node $i$:
```math
V_i = e^{-r \Delta t} (p V_{up} + (1-p) V_{down})
```
For American options, early exercise implies:
```math
V_i = \max(V_i, \text{Intrinsic})
```

### Finite-Difference Greeks
- **Delta ($\Delta$)**: $\frac{V(S+\delta S) - V(S-\delta S)}{2\delta S}$
- **Gamma ($\Gamma$)**: $\frac{V(S+\delta S) - 2V(S) + V(S-\delta S)}{(\delta S)^2}$
- **Vega ($\mathcal{V}$)**: $\frac{V(\sigma+\delta\sigma) - V(\sigma-\delta\sigma)}{2\delta\sigma}$
- **Theta ($\Theta$)**: $\frac{V(T-\delta T) - V(T)}{-\delta T}$
- **Rho ($\rho$)**: $\frac{V(r+\delta r) - V(r-\delta r)}{2\delta r}$

### Implied Volatility
Newton-Raphson update step:
```math
\sigma_{n+1} = \sigma_n - \frac{BSM(\sigma_n) - C_{mkt}}{\mathcal{V}(\sigma_n)}
```

### Yield Curve
Discount factor and zero rate relationship:
```math
D(0, t) = e^{-r(t) \cdot t} \iff r(t) = -\frac{\ln D(0, t)}{t}
```
Forward rate between $t_1$ and $t_2$:
```math
f(t_1, t_2) = -\frac{\ln(D(0, t_2)/D(0, t_1))}{t_2 - t_1}
```
