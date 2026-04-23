"""
Tests for the dictionary-based Pricer API.
"""
import pytest
from pricer import Pricer, PricingResult, PortfolioResult


class TestPricerAPI:

    def test_vanilla_bsm_default(self):
        config = {
            'type': 'vanilla',
            'option_type': 'c',
            'S': 100, 'K': 105, 'T': 1.0, 'vol': 0.20, 'r': 0.05,
            'greeks': True
        }
        res = Pricer(config).run()
        assert isinstance(res, PricingResult)
        assert res.engine_used == 'bsm'
        assert res.price > 0
        assert 'delta' in res.greeks

    def test_vanilla_mc_engine(self):
        config = {
            'type': 'vanilla',
            'option_type': 'c',
            'S': 100, 'K': 105, 'T': 1.0, 'vol': 0.20, 'r': 0.05,
            'engine': 'mc',
            'n_paths': 5000,
            'seed': 42
        }
        res = Pricer(config).run()
        assert res.engine_used == 'mc'
        assert res.price > 0

    def test_american_auto_binomial(self):
        config = {
            'type': 'american',
            'option_type': 'p',
            'S': 100, 'K': 105, 'T': 1.0, 'vol': 0.20, 'r': 0.05,
            'n_steps': 100
        }
        res = Pricer(config).run()
        assert res.engine_used == 'binomial'
        assert res.price > 0

    def test_heston_engine(self):
        config = {
            'type': 'vanilla',
            'option_type': 'c',
            'S': 100, 'K': 105, 'T': 1.0, 'r': 0.05,
            'engine': 'heston',
            'v0': 0.04, 'kappa': 2.0, 'theta': 0.04, 'xi': 0.3, 'rho_heston': -0.7,
            'n_paths': 2000,
            'seed': 42
        }
        res = Pricer(config).run()
        assert res.engine_used == 'heston'
        assert res.price > 0

    def test_basket_option(self):
        config = {
            'type': 'basket',
            'option_type': 'c',
            'S': [100, 110], 'K': 100, 'T': 1.0, 'vol': [0.2, 0.25],
            'corr': [[1, 0.5], [0.5, 1]], 'r': 0.05,
            'n_paths': 2000,
            'seed': 42
        }
        res = Pricer(config).run()
        assert res.engine_used == 'mc'
        assert res.price > 0

    def test_missing_type_raises(self):
        with pytest.raises(ValueError, match="Missing required key: 'type'"):
            Pricer({'S': 100}).run()

    def test_unsupported_engine_raises(self):
        config = {
            'type': 'basket',
            'S': [100, 110], 'K': 100, 'T': 1.0, 'vol': [0.2, 0.2],
            'corr': [[1, 0], [0, 1]], 'r': 0.05,
            'engine': 'bsm'
        }
        with pytest.raises(ValueError, match="not supported"):
            Pricer(config).run()

    def test_heston_missing_params_raises(self):
        config = {
            'type': 'vanilla', 'option_type': 'c',
            'S': 100, 'K': 105, 'T': 1.0, 'r': 0.05,
            'engine': 'heston'
        }
        with pytest.raises(ValueError, match="Heston engine requires"):
            Pricer(config).run()


class TestPortfolioAPI:

    def test_portfolio_pricing(self):
        configs = [
            {'type': 'vanilla', 'option_type': 'c', 'S': 100, 'K': 100, 'T': 1.0, 'vol': 0.2, 'r': 0.05, 'quantity': 10},
            {'type': 'vanilla', 'option_type': 'p', 'S': 100, 'K': 100, 'T': 1.0, 'vol': 0.2, 'r': 0.05, 'quantity': -5},
        ]
        port = Pricer.portfolio(configs, greeks=True)
        assert isinstance(port, PortfolioResult)
        assert len(port.breakdown) == 2
        assert port.total_value != 0
        assert port.total_greeks is not None
        assert 'delta' in port.total_greeks

    def test_missing_quantity_raises(self):
        configs = [{'type': 'vanilla', 'option_type': 'c', 'S': 100, 'K': 100, 'T': 1.0, 'vol': 0.2, 'r': 0.05}]
        with pytest.raises(ValueError, match="must include a 'quantity'"):
            Pricer.portfolio(configs)
