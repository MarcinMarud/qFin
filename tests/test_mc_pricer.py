"""
Tests for models.mc_pricer — MCPricer and MultiAssetMCPricer.
"""
import pytest
import numpy as np
from pyqfin.market_data.yield_curve import FlatCurve
from pyqfin.models.numerical import MonteCarlo
from pyqfin.models.mc_pricer import MCPricer, MultiAssetMCPricer
from pyqfin.models.analytical import BlackScholes
from pyqfin.payoffs.options_payoff import VanillaOptions, BasketOption


class TestMCPricer:

    def test_call_price_close_to_bsm(self, mc_engine):
        inst = VanillaOptions(S=100, K=105, T=1.0, vol=0.2, r=0.05, option_type="c")
        mc_price = MCPricer(mc_engine, inst).price()
        bsm_price = BlackScholes(100, 105, 1.0, 0.2, 0.05, "c").black_scholes()
        assert mc_price == pytest.approx(bsm_price, abs=0.5)

    def test_put_price_close_to_bsm(self, mc_engine):
        inst = VanillaOptions(S=100, K=105, T=1.0, vol=0.2, r=0.05, option_type="p")
        mc_price = MCPricer(mc_engine, inst).price()
        bsm_price = BlackScholes(100, 105, 1.0, 0.2, 0.05, "p").black_scholes()
        assert mc_price == pytest.approx(bsm_price, abs=0.5)

    def test_price_positive(self, mc_engine):
        inst = VanillaOptions(S=100, K=100, T=1.0, vol=0.2, r=0.05, option_type="c")
        price = MCPricer(mc_engine, inst).price()
        assert price > 0

    def test_otm_call_less_than_atm(self, mc_engine):
        atm = VanillaOptions(S=100, K=100, T=1.0, vol=0.2, r=0.05, option_type="c")
        otm = VanillaOptions(S=100, K=120, T=1.0, vol=0.2, r=0.05, option_type="c")
        assert MCPricer(mc_engine, atm).price() > MCPricer(mc_engine, otm).price()


class TestMultiAssetMCPricer:

    def test_basket_price_positive(self, flat_curve):
        mc = MonteCarlo(n=100, M=5000, curve=flat_curve, seed=42)
        inst = BasketOption(
            S_arr=np.array([100.0, 200.0]),
            vol_arr=np.array([0.2, 0.3]),
            corr_matrix=np.array([[1.0, 0.5], [0.5, 1.0]]),
            T=1.0, K=150, option_type="c", r=0.05,
            weights=np.array([0.5, 0.5]),
        )
        price = MultiAssetMCPricer(mc, inst).price()
        assert price > 0

    def test_basket_reproducible(self, flat_curve):
        def _price():
            mc = MonteCarlo(n=50, M=2000, curve=flat_curve, seed=99)
            inst = BasketOption(
                S_arr=np.array([100.0, 200.0]),
                vol_arr=np.array([0.2, 0.3]),
                corr_matrix=np.array([[1.0, 0.5], [0.5, 1.0]]),
                T=1.0, K=150, option_type="c", r=0.05,
            )
            return MultiAssetMCPricer(mc, inst).price()

        assert _price() == pytest.approx(_price())
