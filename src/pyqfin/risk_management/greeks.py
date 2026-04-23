"""
Numerical Greeks via finite-difference methods. Greeks computes
sensitivities for single-asset instruments; MultiAssetGreeks handles
per-asset sensitivities for multi-asset instruments.
"""

from pyqfin.models.mc_pricer import MCPricer, MultiAssetMCPricer
from pyqfin.market_data.yield_curve import YieldCurve
from pyqfin.models.numerical import MonteCarlo
import numpy as np

class Greeks:
    """Finite-difference Greeks calculator for single-asset instruments.
    
    Computes delta, gamma, vega, theta, and rho by re-pricing the
    instrument with bumped parameters. Uses central differences for
    delta, gamma, vega, and rho; forward difference for theta.
    """

    def __init__(self, engine, instrument, dS=None, dvol=None, dT=None, dr=None):
        """Initialise the Greeks calculator with bump sizes.

        Args:
            engine: Monte Carlo engine used for re-pricing bumped instruments.
            instrument: The base Instrument to compute sensitivities for.
            dS: Spot bump size. Defaults to 1% of spot.
            dvol: Volatility bump size in absolute terms. Defaults to 0.01.
            dT: Time bump size in years. Defaults to 1/365 (one day).
            dr: Rate bump size. Defaults to 0.0001 (1 basis point).
        """
        self.engine = engine
        self.instrument = instrument
        self.dS = dS if dS is not None else 0.01 * instrument.S
        self.dvol = dvol if dvol is not None else 0.01
        self.dT = dT if dT is not None else 1/365
        self.dr = dr if dr is not None else 0.0001

    def _get_price(self, engine=None, **overrides):
        """Price a copy of the instrument with overridden attributes.

        Creates a bumped instrument via copy_with(), then prices it
        using MCPricer with the specified (or default) engine.

        Args:
            engine: Optional alternative engine (used for rate-bumped pricing).
            **overrides: Attributes to override (e.g. S=101, vol=0.21).

        Returns:
            MC price of the bumped instrument as a float.
        """
        bumped_instrument = self.instrument.copy_with(**overrides)
        eng = engine if engine is not None else self.engine
        bumped_price = MCPricer(eng, bumped_instrument).price()

        return bumped_price

    def finite_difference(self):
        """Compute all five Greeks using finite-difference re-pricing.

        Delta:  central difference on spot (S ± dS).
        Gamma:  second-order central difference on spot.
        Vega:   central difference on volatility (vol ± dvol).
        Theta:  forward difference on time (T - dT vs T).
        Rho:    central difference on rate (curve ± dr), with matched engines.

        Returns:
            Tuple of (delta, gamma, vega, theta, rho) as floats.
        """
        base_price = self._get_price()

        curve_up = self.instrument.curve.shift(+self.dr)
        curve_down = self.instrument.curve.shift(-self.dr)

        engine_up = MonteCarlo(self.engine.n, self.engine.M, curve=curve_up, seed=self.engine.seed)
        engine_down = MonteCarlo(self.engine.n, self.engine.M, curve=curve_down, seed=self.engine.seed)

        d_price_up = self._get_price(S=self.instrument.S + self.dS)
        d_price_down = self._get_price(S=self.instrument.S - self.dS)
        v_vol_up = self._get_price(vol=self.instrument.vol + self.dvol)
        v_vol_down = self._get_price(vol=self.instrument.vol - self.dvol)
        t_time_down = self._get_price(T=self.instrument.T - self.dT)
        r_rate_up = self._get_price(engine=engine_up, curve=curve_up)
        r_rate_down = self._get_price(engine=engine_down, curve=curve_down)
        

        delta = (d_price_up - d_price_down) / (2 * self.dS)
        gamma = (d_price_up - 2 * base_price + d_price_down) / (self.dS ** 2)
        vega = (v_vol_up - v_vol_down) / (2 * self.dvol)
        theta = (t_time_down - base_price) / (-self.dT)
        rho = (r_rate_up - r_rate_down) / (2 * self.dr)

        return delta, gamma, vega, theta, rho   

class MultiAssetGreeks:
    """Finite-difference Greeks calculator for multi-asset instruments.
    
    Computes per-asset delta, gamma, and vega arrays, plus scalar
    theta and rho. Each asset is bumped independently while holding
    the others fixed.
    """

    def __init__(self, engine, instrument, dS=None, dvol=None, dT=None, dr=None) :
        """Initialise the multi-asset Greeks calculator with bump sizes.

        Args:
            engine: Monte Carlo engine used for re-pricing bumped instruments.
            instrument: The base MultiAssetInstrument to compute sensitivities for.
            dS: Per-asset spot bump array. Defaults to 1% of each spot.
            dvol: Volatility bump size (same for all assets). Defaults to 0.01.
            dT: Time bump size in years. Defaults to 1/365 (one day).
            dr: Rate bump size. Defaults to 0.0001 (1 basis point).
        """
        self.engine = engine
        self.instrument = instrument
        self.dS = dS if dS is not None else np.array(0.01 * instrument.S_arr)
        self.dvol = dvol if dvol is not None else 0.01
        self.dT = dT if dT is not None else 1/365
        self.dr = dr if dr is not None else 0.0001

    def _get_price(self, engine=None, **overrides):
        """Price a copy of the instrument with overridden attributes.

        Creates a bumped instrument via copy_with(), then prices it
        using MultiAssetMCPricer with the specified (or default) engine.

        Args:
            engine: Optional alternative engine (used for rate-bumped pricing).
            **overrides: Attributes to override (e.g. S_arr=bumped_spots).

        Returns:
            MC price of the bumped instrument as a float.
        """
        bumped_instrument = self.instrument.copy_with(**overrides)
        eng = engine if engine is not None else self.engine
        bumped_price = MultiAssetMCPricer(eng, bumped_instrument).price()

        return bumped_price

    def finite_difference(self):
        """Compute per-asset Greeks using finite-difference re-pricing.

        Delta and gamma are computed per asset by bumping each spot
        independently. Vega is computed per asset by bumping each
        volatility independently. Theta and rho are scalar.

        Returns:
            Dict with keys 'delta', 'gamma', 'vega' (arrays of length k),
            'theta' and 'rho' (scalars).
        """
        k = len(self.instrument.S_arr)
        base_price = self._get_price()

        delta_arr = np.zeros(k)
        gamma_arr = np.zeros(k)
        vega_arr = np.zeros(k)

        for i in range(k):

            S_up = self.instrument.S_arr.copy()
            S_up[i] += self.dS[i]
            S_down = self.instrument.S_arr.copy()
            S_down[i] -= self.dS[i]

            price_up = self._get_price(S_arr=S_up)
            price_down = self._get_price(S_arr=S_down)

            delta_arr[i] = (price_up - price_down) / (2 * self.dS[i])
            gamma_arr[i] = (price_up - 2 * base_price + price_down) / (self.dS[i]**2)
            
        for i in range(k):
            
            vol_up = self.instrument.vol_arr.copy()
            vol_up[i] += self.dvol
            vol_down = self.instrument.vol_arr.copy()
            vol_down[i] -= self.dvol

            price_up = self._get_price(vol_arr=vol_up)
            price_down = self._get_price(vol_arr=vol_down)

            vega_arr[i] = (price_up - price_down) / (2 * self.dvol)
        
        theta = (self._get_price(T=self.instrument.T - self.dT) - base_price) / (-self.dT)

        curve_up = self.instrument.curve.shift(+self.dr)
        curve_down = self.instrument.curve.shift(-self.dr)

        engine_up = MonteCarlo(self.engine.n, self.engine.M, curve=curve_up, seed=self.engine.seed)
        engine_down = MonteCarlo(self.engine.n, self.engine.M, curve=curve_down, seed=self.engine.seed)

        r_rate_up = self._get_price(engine=engine_up, curve=curve_up)
        r_rate_down = self._get_price(engine=engine_down, curve=curve_down)

        rho = (r_rate_up - r_rate_down) / (2 * self.dr)
    
        return {
            'delta': delta_arr, 
            'gamma': gamma_arr,  
            'vega': vega_arr,    
            'theta': theta,       
            'rho': rho            
        }