"""
Tests for payoffs.options_payoff — all option payoff classes.
"""
import pytest
import numpy as np
from payoffs.options_payoff import (
    VanillaOptions, AsianOptions, BarrierOptions,
    BasketOption, RainbowOption, SpreadOption, AmericanOption,
)


# ── VanillaOptions ────────────────────────────────────────────────

class TestVanillaOptions:

    def test_call_payoff(self):
        inst = VanillaOptions(S=100, K=105, T=1.0, vol=0.2, r=0.05, option_type="c")
        inst.all_returns = np.array([[100, 100], [110, 95]])  # 2 paths
        payoff = inst.get_payoff()
        np.testing.assert_array_equal(payoff, [5, 0])

    def test_put_payoff(self):
        inst = VanillaOptions(S=100, K=105, T=1.0, vol=0.2, r=0.05, option_type="p")
        inst.all_returns = np.array([[100, 100], [110, 95]])
        payoff = inst.get_payoff()
        np.testing.assert_array_equal(payoff, [0, 10])

    def test_payoff_non_negative(self):
        inst = VanillaOptions(S=100, K=105, T=1.0, vol=0.2, r=0.05, option_type="c")
        inst.all_returns = np.array([[100], [80]])
        assert inst.get_payoff()[0] == 0


# ── AsianOptions ──────────────────────────────────────────────────

class TestAsianOptions:

    def test_call_payoff_uses_average(self):
        inst = AsianOptions(S=100, K=100, T=1.0, vol=0.2, r=0.05, option_type="c")
        # paths: [S0=100], [S1=110], [S2=120]  → avg of [110, 120] = 115
        inst.all_returns = np.array([[100], [110], [120]])
        payoff = inst.get_payoff()
        assert payoff[0] == pytest.approx(15)

    def test_put_payoff_uses_average(self):
        inst = AsianOptions(S=100, K=115, T=1.0, vol=0.2, r=0.05, option_type="p")
        inst.all_returns = np.array([[100], [110], [120]])
        payoff = inst.get_payoff()
        assert payoff[0] == pytest.approx(0)

    def test_average_excludes_initial_spot(self):
        inst = AsianOptions(S=50, K=100, T=1.0, vol=0.2, r=0.05, option_type="c")
        # S0=50, S1=110, S2=130 → avg([110,130]) = 120, NOT avg([50,110,130])
        inst.all_returns = np.array([[50], [110], [130]])
        payoff = inst.get_payoff()
        assert payoff[0] == pytest.approx(20)


# ── BarrierOptions ────────────────────────────────────────────────

class TestBarrierOptions:

    def test_up_and_out_breached(self):
        inst = BarrierOptions(S=100, K=100, T=1.0, vol=0.2,
                              barrier_price=120, barrier_kind="knock-out",
                              barrier_direction="up", r=0.05, option_type="c")
        inst.all_returns = np.array([[100, 100], [125, 110], [130, 115]])
        payoff = inst.get_payoff()
        assert payoff[0] == 0     # breached → knocked out
        assert payoff[1] == 15    # not breached → vanilla payoff

    def test_up_and_in_breached(self):
        inst = BarrierOptions(S=100, K=100, T=1.0, vol=0.2,
                              barrier_price=120, barrier_kind="knock-in",
                              barrier_direction="up", r=0.05, option_type="c")
        inst.all_returns = np.array([[100, 100], [125, 110], [130, 115]])
        payoff = inst.get_payoff()
        assert payoff[0] == 30    # breached → knocked in
        assert payoff[1] == 0     # not breached → no payoff

    def test_down_and_out_breached(self):
        inst = BarrierOptions(S=100, K=100, T=1.0, vol=0.2,
                              barrier_price=90, barrier_kind="knock-out",
                              barrier_direction="down", r=0.05, option_type="p")
        inst.all_returns = np.array([[100, 100], [85, 95], [80, 105]])
        payoff = inst.get_payoff()
        assert payoff[0] == 0     # breached → knocked out
        assert payoff[1] == 0     # not breached, OTM put

    def test_invalid_direction_raises(self):
        inst = BarrierOptions(S=100, K=100, T=1.0, vol=0.2,
                              barrier_price=120, barrier_kind="knock-out",
                              barrier_direction="sideways", r=0.05, option_type="c")
        inst.all_returns = np.array([[100], [110]])
        with pytest.raises(ValueError):
            inst.get_payoff()

    def test_invalid_kind_raises(self):
        inst = BarrierOptions(S=100, K=100, T=1.0, vol=0.2,
                              barrier_price=120, barrier_kind="knock-through",
                              barrier_direction="up", r=0.05, option_type="c")
        inst.all_returns = np.array([[100], [110]])
        with pytest.raises(ValueError):
            inst.get_payoff()


# ── BasketOption ──────────────────────────────────────────────────

class TestBasketOption:

    def _make_basket(self, option_type="c"):
        return BasketOption(
            S_arr=np.array([100.0, 200.0]),
            vol_arr=np.array([0.2, 0.3]),
            corr_matrix=np.array([[1.0, 0.5], [0.5, 1.0]]),
            T=1.0, K=150, option_type=option_type, r=0.05,
            weights=np.array([0.5, 0.5]),
        )

    def test_call_payoff_absolute_strike(self):
        inst = self._make_basket("c")
        # 2 assets, 2 time steps, 1 path → shape (2, 2, 1)
        inst.all_returns = np.array([[[100], [120]], [[200], [220]]])
        payoff = inst.get_payoff(strike_type="absolute")
        # basket = 0.5*120 + 0.5*220 = 170, payoff = 170 - 150 = 20
        assert payoff[0] == pytest.approx(20)

    def test_put_payoff_absolute_strike(self):
        inst = self._make_basket("p")
        inst.all_returns = np.array([[[100], [100]], [[200], [180]]])
        payoff = inst.get_payoff(strike_type="absolute")
        # basket = 0.5*100 + 0.5*180 = 140, payoff = 150 - 140 = 10
        assert payoff[0] == pytest.approx(10)


# ── RainbowOption ─────────────────────────────────────────────────

class TestRainbowOption:

    def _make_rainbow(self, option_type="c"):
        return RainbowOption(
            S_arr=np.array([100.0, 200.0]),
            vol_arr=np.array([0.2, 0.3]),
            corr_matrix=np.array([[1.0, 0.5], [0.5, 1.0]]),
            T=1.0, K=150, option_type=option_type, r=0.05,
        )

    def test_best_of_call(self):
        inst = self._make_rainbow("c")
        inst.all_returns = np.array([[[100], [120]], [[200], [160]]])
        payoff = inst.get_payoff(rainbow_type="best")
        # best = max(120, 160) = 160, payoff = 160 - 150 = 10
        assert payoff[0] == pytest.approx(10)

    def test_worst_of_call(self):
        inst = self._make_rainbow("c")
        inst.all_returns = np.array([[[100], [120]], [[200], [160]]])
        payoff = inst.get_payoff(rainbow_type="worst")
        # worst = min(120, 160) = 120, payoff = max(120 - 150, 0) = 0
        assert payoff[0] == pytest.approx(0)


# ── SpreadOption ──────────────────────────────────────────────────

class TestSpreadOption:

    def _make_spread(self, option_type="c"):
        return SpreadOption(
            S_arr=np.array([100.0, 90.0]),
            vol_arr=np.array([0.2, 0.25]),
            corr_matrix=np.array([[1.0, 0.6], [0.6, 1.0]]),
            T=1.0, K=5, option_type=option_type, r=0.05,
        )

    def test_call_payoff(self):
        inst = self._make_spread("c")
        inst.all_returns = np.array([[[100], [120]], [[90], [100]]])
        payoff = inst.get_payoff()
        # spread = 120 - 100 = 20, payoff = 20 - 5 = 15
        assert payoff[0] == pytest.approx(15)

    def test_put_payoff(self):
        inst = self._make_spread("p")
        inst.all_returns = np.array([[[100], [100]], [[90], [100]]])
        payoff = inst.get_payoff()
        # spread = 100 - 100 = 0, payoff = max(5 - 0, 0) = 5
        assert payoff[0] == pytest.approx(5)

    def test_requires_two_assets(self):
        inst = SpreadOption(
            S_arr=np.array([100.0, 90.0, 80.0]),
            vol_arr=np.array([0.2, 0.25, 0.3]),
            corr_matrix=np.eye(3),
            T=1.0, K=5, option_type="c", r=0.05,
        )
        inst.all_returns = np.zeros((3, 2, 1))
        with pytest.raises(ValueError, match="exactly 2 assets"):
            inst.get_payoff()


# ── AmericanOption ────────────────────────────────────────────────

class TestAmericanOption:

    def test_get_payoff_call(self):
        inst = AmericanOption(S=100, K=105, T=1.0, vol=0.2, r=0.05, option_type="c")
        inst.all_returns = np.array([[100, 100], [110, 95]])
        payoff = inst.get_payoff()
        np.testing.assert_array_equal(payoff, [5, 0])

    def test_get_payoff_put(self):
        inst = AmericanOption(S=100, K=105, T=1.0, vol=0.2, r=0.05, option_type="p")
        inst.all_returns = np.array([[100, 100], [110, 95]])
        payoff = inst.get_payoff()
        np.testing.assert_array_equal(payoff, [0, 10])

    def test_inherits_intrinsic_value(self):
        inst = AmericanOption(S=100, K=105, T=1.0, vol=0.2, r=0.05, option_type="c")
        assert inst.intrinsic_value(110) == 5
