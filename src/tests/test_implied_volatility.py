"""
Tests for risk_management.implied_volatility — all three IV solvers.
"""
import pytest
import numpy as np
from models.analytical import BlackScholes
from risk_management.implied_volatility import ImpliedVolatility


class TestImpliedVolatility:

    @pytest.fixture
    def iv_solver(self):
        return ImpliedVolatility()

    @pytest.fixture
    def market_call_price(self):
        """BSM price at 20% vol → IV should recover 0.20."""
        return BlackScholes(100, 100, 1.0, 0.2, 0.05, "c").black_scholes()

    @pytest.fixture
    def market_put_price(self):
        return BlackScholes(100, 100, 1.0, 0.2, 0.05, "p").black_scholes()

    # ── Newton-Raphson ────────────────────────────────────────────

    def test_newton_raphson_call(self, iv_solver, market_call_price):
        iv = iv_solver.newton_raphson(100, 100, 1.0, 0.05, "c", market_call_price)
        assert iv == pytest.approx(0.2, abs=1e-4)

    def test_newton_raphson_put(self, iv_solver, market_put_price):
        iv = iv_solver.newton_raphson(100, 100, 1.0, 0.05, "p", market_put_price)
        assert iv == pytest.approx(0.2, abs=1e-4)

    def test_newton_raphson_otm(self, iv_solver):
        price = BlackScholes(100, 120, 1.0, 0.3, 0.05, "c").black_scholes()
        iv = iv_solver.newton_raphson(100, 120, 1.0, 0.05, "c", price)
        assert iv == pytest.approx(0.3, abs=1e-4)

    # ── Bisection ─────────────────────────────────────────────────

    def test_bisection_call(self, iv_solver, market_call_price):
        iv = iv_solver.bisection(100, 100, 1.0, 0.05, "c", market_call_price)
        assert iv == pytest.approx(0.2, abs=1e-4)

    def test_bisection_put(self, iv_solver, market_put_price):
        iv = iv_solver.bisection(100, 100, 1.0, 0.05, "p", market_put_price)
        assert iv == pytest.approx(0.2, abs=1e-4)

    # ── Hybrid Newton ─────────────────────────────────────────────

    def test_hybrid_newton_call(self, iv_solver, market_call_price):
        iv = iv_solver.hybrid_newton(100, 100, 1.0, 0.05, "c", market_call_price)
        assert iv == pytest.approx(0.2, abs=1e-4)

    def test_hybrid_newton_put(self, iv_solver, market_put_price):
        iv = iv_solver.hybrid_newton(100, 100, 1.0, 0.05, "p", market_put_price)
        assert iv == pytest.approx(0.2, abs=1e-4)

    def test_hybrid_newton_high_vol(self, iv_solver):
        price = BlackScholes(100, 100, 1.0, 0.8, 0.05, "c").black_scholes()
        iv = iv_solver.hybrid_newton(100, 100, 1.0, 0.05, "c", price, x_up=2.0)
        assert iv == pytest.approx(0.8, abs=1e-3)
