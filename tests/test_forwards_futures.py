"""
Tests for payoffs.forwards_futures_payoff — Forwards and Futures.
"""
import pytest
import numpy as np
from pyqfin.payoffs.forwards_futures_payoff import Forwards, Futures


class TestForwards:

    def test_payoff_positive(self):
        inst = Forwards(S=100, K=100, T=1.0, vol=0.2, r=0.05)
        inst.all_returns = np.array([[100, 100], [120, 80]])
        payoff = inst.get_payoff()
        np.testing.assert_array_equal(payoff, [20, -20])

    def test_payoff_can_be_negative(self):
        inst = Forwards(S=100, K=110, T=1.0, vol=0.2, r=0.05)
        inst.all_returns = np.array([[100], [105]])
        assert inst.get_payoff()[0] == pytest.approx(-5)

    def test_payoff_at_delivery_price(self):
        inst = Forwards(S=100, K=100, T=1.0, vol=0.2, r=0.05)
        inst.all_returns = np.array([[100], [100]])
        assert inst.get_payoff()[0] == pytest.approx(0)


class TestFutures:

    def test_payoff_identical_to_forward(self):
        fwd = Forwards(S=100, K=100, T=1.0, vol=0.2, r=0.05)
        fut = Futures(S=100, K=100, T=1.0, vol=0.2, r=0.05)
        paths = np.array([[100, 100], [120, 80]])
        fwd.all_returns = paths
        fut.all_returns = paths
        np.testing.assert_array_equal(fwd.get_payoff(), fut.get_payoff())
