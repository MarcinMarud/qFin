# qFin
A professional, vectorised quantitative finance library for pricing options, forwards, futures, and multi-asset derivatives.

## Features
- **Dictionary-based Quick API (`Pricer`)**: Easily price single instruments and portfolios.
- **Multiple Engines**: Black-Scholes (analytical), Monte Carlo, Heston (stochastic vol), and Cox-Ross-Rubinstein Binomial Trees.
- **Exotics & Multi-Asset**: Support for Asian, Barrier, Basket, Rainbow, and Spread options with Cholesky decompositions for correlated assets.
- **American Options**: Early exercise evaluation using Binomial Trees.
- **Risk Management**: Finite-difference Greeks calculation and implied volatility root-finding (Newton-Raphson, Bisection).
- **Term Structure**: Interpolated and flat yield curves.

## Installation
```bash
pip install marcin-qfin
```

## Quick Start

### Single Option Pricing
The `Pricer` handles all the internal wiring (engines, yield curves, payoff classes).

```python
from qfin import Pricer

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

## Running Tests
If you cloned the repository and want to run the tests locally:
```bash
pip install marcin-qfin[dev]
pytest src/tests/ -v
```
