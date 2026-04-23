"""
Tests for models.numerical — MonteCarlo, Heston, BinominalTree.
"""
import pytest
import numpy as np
from market_data.yield_curve import FlatCurve
from models.numerical import MonteCarlo, Heston, BinominalTree
from models.analytical import BlackScholes
from payoffs.options_payoff import VanillaOptions, AmericanOption


# ── MonteCarlo ─────────────────────────────────────────────────────

class TestMonteCarlo:

    def test_simulate_returns_correct_shape(self, mc_engine):
        paths = mc_engine.simulate(S=100, T=1.0, vol=0.2)
        assert paths.shape == (253, 10_000)  # n+1 rows, M columns

    def test_simulate_first_row_is_spot(self, mc_engine):
        paths = mc_engine.simulate(S=100, T=1.0, vol=0.2)
        np.testing.assert_array_equal(paths[0, :], 100)

    def test_simulate_reproducible_with_seed(self, flat_curve):
        mc1 = MonteCarlo(n=50, M=1000, curve=flat_curve, seed=123)
        mc2 = MonteCarlo(n=50, M=1000, curve=flat_curve, seed=123)
        p1 = mc1.simulate(S=100, T=1.0, vol=0.2)
        p2 = mc2.simulate(S=100, T=1.0, vol=0.2)
        np.testing.assert_array_equal(p1, p2)

    def test_simulate_different_seeds_differ(self, flat_curve):
        mc1 = MonteCarlo(n=50, M=1000, curve=flat_curve, seed=1)
        mc2 = MonteCarlo(n=50, M=1000, curve=flat_curve, seed=2)
        p1 = mc1.simulate(S=100, T=1.0, vol=0.2)
        p2 = mc2.simulate(S=100, T=1.0, vol=0.2)
        assert not np.allclose(p1, p2)

    def test_terminal_prices_positive(self, mc_engine):
        paths = mc_engine.simulate(S=100, T=1.0, vol=0.2)
        assert np.all(paths[-1, :] > 0)

    def test_cholesky_returns_correct_shape(self, flat_curve):
        mc = MonteCarlo(n=50, M=1000, curve=flat_curve, seed=42)
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        paths = mc.cholesky_monte_carlo(
            S_arr=np.array([100, 200]), T=1.0,
            vol_arr=np.array([0.2, 0.3]), corr_matrix=corr
        )
        assert paths.shape == (2, 51, 1000)  # k, n+1, M

    def test_cholesky_initial_prices(self, flat_curve):
        mc = MonteCarlo(n=50, M=1000, curve=flat_curve, seed=42)
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        paths = mc.cholesky_monte_carlo(
            S_arr=np.array([100, 200]), T=1.0,
            vol_arr=np.array([0.2, 0.3]), corr_matrix=corr
        )
        np.testing.assert_array_almost_equal(paths[0, 0, :], 100)
        np.testing.assert_array_almost_equal(paths[1, 0, :], 200)


# ── Heston ────────────────────────────────────────────────────────

class TestHeston:

    @pytest.fixture
    def heston_engine(self):
        return Heston(v0=0.04, kappa=2.0, theta=0.04, xi=0.3,
                      rho=-0.7, r=0.05, n=252, paths=5000, seed=42)

    def test_simulate_returns_correct_shape(self, heston_engine):
        paths = heston_engine.simulate(S=100, T=1.0)
        assert paths.shape == (253, 5000)

    def test_initial_price_is_spot(self, heston_engine):
        paths = heston_engine.simulate(S=100, T=1.0)
        np.testing.assert_array_almost_equal(paths[0, :], 100)

    def test_heston_model_returns_both_arrays(self, heston_engine):
        prices, vols = heston_engine.heston_model(S=100, T=1.0)
        assert prices.shape == (253, 5000)
        assert vols.shape == (253, 5000)

    def test_variance_stays_non_negative(self, heston_engine):
        _, vols = heston_engine.heston_model(S=100, T=1.0)
        assert np.all(vols >= 0)

    def test_terminal_prices_positive(self, heston_engine):
        paths = heston_engine.simulate(S=100, T=1.0)
        assert np.all(paths[-1, :] > 0)


# ── BinominalTree ─────────────────────────────────────────────────

class TestBinominalTree:

    @pytest.fixture
    def tree(self, flat_curve):
        return BinominalTree(n_steps=500, curve=flat_curve)

    def test_european_call_converges_to_bsm(self, tree):
        inst = VanillaOptions(S=100, K=105, T=1.0, vol=0.2, r=0.05, option_type="c")
        tree_price = tree.price(inst, american=False)
        bsm_price = BlackScholes(100, 105, 1.0, 0.2, 0.05, "c").black_scholes()
        assert tree_price == pytest.approx(bsm_price, abs=0.05)

    def test_european_put_converges_to_bsm(self, tree):
        inst = VanillaOptions(S=100, K=105, T=1.0, vol=0.2, r=0.05, option_type="p")
        tree_price = tree.price(inst, american=False)
        bsm_price = BlackScholes(100, 105, 1.0, 0.2, 0.05, "p").black_scholes()
        assert tree_price == pytest.approx(bsm_price, abs=0.05)

    def test_american_put_geq_european_put(self, tree):
        inst_eu = VanillaOptions(S=100, K=105, T=1.0, vol=0.2, r=0.05, option_type="p")
        inst_am = AmericanOption(S=100, K=105, T=1.0, vol=0.2, r=0.05, option_type="p")
        eu_price = tree.price(inst_eu, american=False)
        am_price = tree.price(inst_am, american=True)
        assert am_price >= eu_price

    def test_american_call_equals_european_no_dividends(self, tree):
        inst_eu = VanillaOptions(S=100, K=105, T=1.0, vol=0.2, r=0.05, option_type="c")
        inst_am = AmericanOption(S=100, K=105, T=1.0, vol=0.2, r=0.05, option_type="c")
        eu_price = tree.price(inst_eu, american=False)
        am_price = tree.price(inst_am, american=True)
        assert am_price == pytest.approx(eu_price, abs=0.01)

    def test_price_increases_with_steps(self, flat_curve):
        inst = VanillaOptions(S=100, K=105, T=1.0, vol=0.2, r=0.05, option_type="c")
        bsm = BlackScholes(100, 105, 1.0, 0.2, 0.05, "c").black_scholes()
        tree_50 = BinominalTree(n_steps=50, curve=flat_curve)
        tree_500 = BinominalTree(n_steps=500, curve=flat_curve)
        err_50 = abs(tree_50.price(inst) - bsm)
        err_500 = abs(tree_500.price(inst) - bsm)
        assert err_500 < err_50

    def test_deep_itm_put_early_exercise(self, flat_curve):
        # Deep ITM American put should have significant early exercise premium
        tree = BinominalTree(n_steps=200, curve=flat_curve)
        inst = AmericanOption(S=80, K=120, T=1.0, vol=0.2, r=0.05, option_type="p")
        am_price = tree.price(inst, american=True)
        eu_price = tree.price(inst, american=False)
        assert am_price > eu_price + 0.5  # at least 0.5 premium
