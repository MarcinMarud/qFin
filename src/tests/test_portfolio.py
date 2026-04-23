"""
Tests for portfolio — Portfolio class.
"""
import pytest
import numpy as np
from market_data.yield_curve import FlatCurve
from models.numerical import MonteCarlo
from models.analytical import BlackScholes
from payoffs.options_payoff import VanillaOptions
from portfolio import Portfolio


class TestPortfolio:

    @pytest.fixture
    def engine(self):
        curve = FlatCurve(0.05)
        return MonteCarlo(n=100, M=10_000, curve=curve, seed=42)

    def test_add_position(self):
        port = Portfolio()
        inst = VanillaOptions(S=100, K=100, T=1.0, vol=0.2, r=0.05, option_type="c")
        result = port.add(inst, 10)
        assert len(result) == 1
        assert result[0][1] == 10

    def test_add_multiple_positions(self):
        port = Portfolio()
        inst1 = VanillaOptions(S=100, K=100, T=1.0, vol=0.2, r=0.05, option_type="c")
        inst2 = VanillaOptions(S=100, K=110, T=1.0, vol=0.2, r=0.05, option_type="p")
        port.add(inst1, 10)
        port.add(inst2, -5)
        assert len(port.positions) == 2

    def test_portfolio_value_single_position(self, engine):
        port = Portfolio()
        inst = VanillaOptions(S=100, K=100, T=1.0, vol=0.2, r=0.05, option_type="c")
        port.add(inst, 1)
        value = port.portfolio_value(engine)
        bsm = BlackScholes(100, 100, 1.0, 0.2, 0.05, "c").black_scholes()
        assert value == pytest.approx(bsm, abs=0.5)

    def test_portfolio_value_scales_with_quantity(self, engine):
        port1 = Portfolio()
        port10 = Portfolio()
        inst = VanillaOptions(S=100, K=100, T=1.0, vol=0.2, r=0.05, option_type="c")
        port1.add(inst, 1)
        port10.add(inst, 10)
        v1 = port1.portfolio_value(engine)
        v10 = port10.portfolio_value(engine)
        assert v10 == pytest.approx(10 * v1, abs=0.1)

    def test_portfolio_greeks_returns_dict(self, engine):
        port = Portfolio()
        inst = VanillaOptions(S=100, K=100, T=1.0, vol=0.2, r=0.05, option_type="c")
        port.add(inst, 1)
        greeks = port.portfolio_greeks(engine)
        assert "delta" in greeks
        assert "gamma" in greeks
        assert "vega" in greeks
        assert "theta" in greeks
        assert "rho" in greeks

    def test_portfolio_greeks_delta_positive_for_call(self, engine):
        port = Portfolio()
        inst = VanillaOptions(S=100, K=100, T=1.0, vol=0.2, r=0.05, option_type="c")
        port.add(inst, 1)
        greeks = port.portfolio_greeks(engine)
        assert greeks["delta"] > 0
