"""
Tests for payoffs.instruments — Instrument, MultiAssetInstrument base classes.
"""
import pytest
import numpy as np
from market_data.yield_curve import FlatCurve
from payoffs.instruments import Instrument, MultiAssetInstrument


# ── Instrument ─────────────────────────────────────────────────────

class TestInstrument:

    def test_init_with_rate(self):
        inst = Instrument(S=100, K=105, T=1.0, vol=0.2, r=0.05, option_type="c")
        assert inst.S == 100
        assert inst.K == 105
        assert inst.T == 1.0
        assert inst.vol == 0.2
        assert inst.option_type == "c"

    def test_init_with_curve(self, flat_curve):
        inst = Instrument(S=100, K=105, T=1.0, vol=0.2, curve=flat_curve, option_type="p")
        assert inst.curve is flat_curve

    def test_init_raises_without_rate_or_curve(self):
        with pytest.raises(ValueError, match="Curve or rate must be provided"):
            Instrument(S=100, K=105, T=1.0, vol=0.2)

    def test_r_property_returns_zero_rate(self):
        inst = Instrument(S=100, K=105, T=1.0, vol=0.2, r=0.05, option_type="c")
        assert inst.r == pytest.approx(0.05)

    def test_copy_with_changes_attribute(self):
        inst = Instrument(S=100, K=105, T=1.0, vol=0.2, r=0.05, option_type="c")
        clone = inst.copy_with(S=110)
        assert clone.S == 110
        assert inst.S == 100  # original unchanged

    def test_copy_with_multiple_overrides(self):
        inst = Instrument(S=100, K=105, T=1.0, vol=0.2, r=0.05, option_type="c")
        clone = inst.copy_with(S=110, vol=0.25)
        assert clone.S == 110
        assert clone.vol == 0.25

    def test_intrinsic_value_call(self):
        inst = Instrument(S=100, K=105, T=1.0, vol=0.2, r=0.05, option_type="c")
        S_arr = np.array([90, 100, 110, 120])
        iv = inst.intrinsic_value(S_arr)
        np.testing.assert_array_equal(iv, np.array([0, 0, 5, 15]))

    def test_intrinsic_value_put(self):
        inst = Instrument(S=100, K=105, T=1.0, vol=0.2, r=0.05, option_type="p")
        S_arr = np.array([90, 100, 110, 120])
        iv = inst.intrinsic_value(S_arr)
        np.testing.assert_array_equal(iv, np.array([15, 5, 0, 0]))

    def test_intrinsic_value_scalar(self):
        inst = Instrument(S=100, K=105, T=1.0, vol=0.2, r=0.05, option_type="c")
        assert inst.intrinsic_value(110) == 5


# ── MultiAssetInstrument ──────────────────────────────────────────

class TestMultiAssetInstrument:

    def _valid_params(self):
        return dict(
            S_arr=np.array([100.0, 200.0]),
            vol_arr=np.array([0.2, 0.3]),
            corr_matrix=np.array([[1.0, 0.5], [0.5, 1.0]]),
            T=1.0, K=100, option_type="c", r=0.05,
        )

    def test_valid_init(self):
        inst = MultiAssetInstrument(**self._valid_params())
        assert len(inst.S_arr) == 2

    def test_default_equal_weights(self):
        inst = MultiAssetInstrument(**self._valid_params())
        np.testing.assert_array_almost_equal(inst.weights, [0.5, 0.5])

    def test_custom_weights(self):
        params = self._valid_params()
        params["weights"] = np.array([0.7, 0.3])
        inst = MultiAssetInstrument(**params)
        np.testing.assert_array_almost_equal(inst.weights, [0.7, 0.3])

    def test_negative_vol_raises(self):
        params = self._valid_params()
        params["vol_arr"] = np.array([0.2, -0.1])
        with pytest.raises(ValueError, match="Volatility must be positive"):
            MultiAssetInstrument(**params)

    def test_non_psd_corr_raises(self):
        params = self._valid_params()
        params["corr_matrix"] = np.array([[1.0, 1.5], [1.5, 1.0]])
        with pytest.raises(ValueError):
            MultiAssetInstrument(**params)

    def test_asymmetric_corr_raises(self):
        params = self._valid_params()
        params["corr_matrix"] = np.array([[1.0, 0.3], [0.5, 1.0]])
        with pytest.raises(ValueError, match="symmetric"):
            MultiAssetInstrument(**params)

    def test_wrong_diagonal_raises(self):
        params = self._valid_params()
        params["corr_matrix"] = np.array([[0.9, 0.5], [0.5, 0.9]])
        with pytest.raises(ValueError, match="diagonal"):
            MultiAssetInstrument(**params)

    def test_length_mismatch_raises(self):
        params = self._valid_params()
        params["S_arr"] = np.array([100.0, 200.0, 300.0])
        with pytest.raises(ValueError, match="same length"):
            MultiAssetInstrument(**params)

    def test_copy_with(self):
        inst = MultiAssetInstrument(**self._valid_params())
        clone = inst.copy_with(T=2.0)
        assert clone.T == 2.0
        assert inst.T == 1.0
