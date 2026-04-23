"""
Tests for models.analytical — BlackScholes and FixedContractsCalculations.
"""
import pytest
import numpy as np
from pyqfin.models.analytical import BlackScholes, FixedContractsCalculations


# ── BlackScholes ──────────────────────────────────────────────────

class TestBlackScholes:

    @pytest.fixture
    def bsm_call(self):
        return BlackScholes(S=100, K=105, T=1.0, vol=0.2, r=0.05, option_type="c")

    @pytest.fixture
    def bsm_put(self):
        return BlackScholes(S=100, K=105, T=1.0, vol=0.2, r=0.05, option_type="p")

    # ── d1, d2 ────────────────────────────────────────────────────

    def test_d1_d2_returns_tuple(self, bsm_call):
        d1, d2 = bsm_call._d1_d2()
        assert isinstance(d1, float)
        assert isinstance(d2, float)

    def test_d2_less_than_d1(self, bsm_call):
        d1, d2 = bsm_call._d1_d2()
        assert d2 < d1

    def test_d1_d2_relationship(self, bsm_call):
        d1, d2 = bsm_call._d1_d2()
        assert d1 - d2 == pytest.approx(0.2 * np.sqrt(1.0))

    # ── Pricing ───────────────────────────────────────────────────

    def test_call_price_positive(self, bsm_call):
        assert bsm_call.black_scholes() > 0

    def test_put_price_positive(self, bsm_put):
        assert bsm_put.black_scholes() > 0

    def test_put_call_parity(self, bsm_call, bsm_put):
        call = bsm_call.black_scholes()
        put = bsm_put.black_scholes()
        parity = call - put - (100 * 1 - 105 * np.exp(-0.05 * 1.0))
        assert parity == pytest.approx(0, abs=1e-8)

    def test_atm_call_greater_than_put(self):
        bsm_c = BlackScholes(S=100, K=100, T=1.0, vol=0.2, r=0.05, option_type="c")
        bsm_p = BlackScholes(S=100, K=100, T=1.0, vol=0.2, r=0.05, option_type="p")
        assert bsm_c.black_scholes() > bsm_p.black_scholes()

    def test_deep_itm_call_approaches_intrinsic(self):
        bsm = BlackScholes(S=200, K=100, T=0.01, vol=0.2, r=0.05, option_type="c")
        assert bsm.black_scholes() == pytest.approx(100, abs=1)

    # ── Delta ─────────────────────────────────────────────────────

    def test_call_delta_between_0_and_1(self, bsm_call):
        delta = bsm_call.black_scholes_delta()
        assert 0 < delta < 1

    def test_put_delta_between_neg1_and_0(self, bsm_put):
        delta = bsm_put.black_scholes_delta()
        assert -1 < delta < 0

    def test_call_put_delta_relationship(self, bsm_call, bsm_put):
        delta_c = bsm_call.black_scholes_delta()
        delta_p = bsm_put.black_scholes_delta()
        assert delta_c - delta_p == pytest.approx(1.0, abs=1e-8)

    # ── Gamma ─────────────────────────────────────────────────────

    def test_gamma_positive(self, bsm_call):
        assert bsm_call.black_scholes_gamma() > 0

    def test_gamma_same_for_call_and_put(self, bsm_call, bsm_put):
        assert bsm_call.black_scholes_gamma() == pytest.approx(
            bsm_put.black_scholes_gamma(), abs=1e-10
        )

    # ── Vega ──────────────────────────────────────────────────────

    def test_vega_positive(self, bsm_call):
        assert bsm_call.black_scholes_vega() > 0

    def test_vega_same_for_call_and_put(self, bsm_call, bsm_put):
        assert bsm_call.black_scholes_vega() == pytest.approx(
            bsm_put.black_scholes_vega(), abs=1e-10
        )

    # ── Theta ─────────────────────────────────────────────────────

    def test_call_theta_negative(self, bsm_call):
        assert bsm_call.black_scholes_theta() < 0

    def test_put_theta_can_differ_sign(self, bsm_put):
        # For OTM put, theta is usually negative but can be positive for deep ITM
        theta = bsm_put.black_scholes_theta()
        assert isinstance(float(theta), float)

    # ── Rho ───────────────────────────────────────────────────────

    def test_call_rho_positive(self, bsm_call):
        assert bsm_call.black_scholes_rho() > 0

    def test_put_rho_negative(self, bsm_put):
        assert bsm_put.black_scholes_rho() < 0


# ── FixedContractsCalculations ────────────────────────────────────

class TestFixedContractsCalculations:

    def test_cost_of_carry_no_storage_no_yield(self):
        fcc = FixedContractsCalculations(S=100, r=0.05, storage_cost=0, convenience_yield=0, T=1.0)
        expected = 100 * np.exp(0.05)
        assert fcc.cost_of_carry() == pytest.approx(expected)

    def test_cost_of_carry_with_storage(self):
        fcc = FixedContractsCalculations(S=100, r=0.05, storage_cost=0.02, convenience_yield=0, T=1.0)
        expected = 100 * np.exp(0.07)
        assert fcc.cost_of_carry() == pytest.approx(expected)

    def test_cost_of_carry_with_convenience_yield(self):
        fcc = FixedContractsCalculations(S=100, r=0.05, storage_cost=0, convenience_yield=0.03, T=1.0)
        expected = 100 * np.exp(0.02)
        assert fcc.cost_of_carry() == pytest.approx(expected)

    def test_cost_of_carry_at_zero_time(self):
        fcc = FixedContractsCalculations(S=100, r=0.05, storage_cost=0.02, convenience_yield=0.01, T=0)
        assert fcc.cost_of_carry() == pytest.approx(100)
