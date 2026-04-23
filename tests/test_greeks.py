"""
Tests for risk_management.greeks — Greeks (single-asset).
MultiAssetGreeks is not tested here due to long runtime.
"""
import pytest
import numpy as np
from pyqfin.market_data.yield_curve import FlatCurve
from pyqfin.models.numerical import MonteCarlo
from pyqfin.models.analytical import BlackScholes
from pyqfin.risk_management.greeks import Greeks
from pyqfin.payoffs.options_payoff import VanillaOptions


class TestGreeks:

    @pytest.fixture
    def greeks_result(self):
        """Pre-compute Greeks once (expensive)."""
        curve = FlatCurve(0.05)
        engine = MonteCarlo(n=100, M=20_000, curve=curve, seed=42)
        inst = VanillaOptions(S=100, K=100, T=1.0, vol=0.2, r=0.05, option_type="c")
        g = Greeks(engine, inst)
        return g.finite_difference()

    @pytest.fixture
    def bsm_greeks(self):
        """Analytical BSM Greeks for comparison."""
        bsm = BlackScholes(100, 100, 1.0, 0.2, 0.05, "c")
        return {
            "delta": float(bsm.black_scholes_delta()),
            "gamma": float(bsm.black_scholes_gamma()),
            "vega": float(bsm.black_scholes_vega()),
            "theta": float(bsm.black_scholes_theta()),
            "rho": float(bsm.black_scholes_rho()),
        }

    def test_delta_close_to_bsm(self, greeks_result, bsm_greeks):
        delta = greeks_result[0]
        assert delta == pytest.approx(bsm_greeks["delta"], abs=0.05)

    def test_gamma_close_to_bsm(self, greeks_result, bsm_greeks):
        gamma = greeks_result[1]
        assert gamma == pytest.approx(bsm_greeks["gamma"], abs=0.005)

    def test_vega_close_to_bsm(self, greeks_result, bsm_greeks):
        vega = greeks_result[2]
        assert vega == pytest.approx(bsm_greeks["vega"], abs=2.0)

    def test_theta_finite_for_call(self, greeks_result):
        theta = greeks_result[3]
        assert np.isfinite(theta)

    def test_rho_positive_for_call(self, greeks_result):
        rho = greeks_result[4]
        assert rho > 0

    def test_returns_five_values(self, greeks_result):
        assert len(greeks_result) == 5
