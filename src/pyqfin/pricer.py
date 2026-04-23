"""
Dictionary-based API for quick and easy pricing of instruments and portfolios.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import numpy as np

from pyqfin.market_data.yield_curve import FlatCurve
from pyqfin.models.numerical import MonteCarlo, Heston, BinominalTree
from pyqfin.models.analytical import BlackScholes
from pyqfin.models.mc_pricer import MCPricer, MultiAssetMCPricer
from pyqfin.payoffs.options_payoff import (
    VanillaOptions, AsianOptions, BarrierOptions,
    BasketOption, RainbowOption, SpreadOption, AmericanOption
)
from pyqfin.payoffs.forwards_futures_payoff import Forwards, Futures
from pyqfin.risk_management.greeks import Greeks, MultiAssetGreeks
from pyqfin.portfolio import Portfolio


@dataclass
class PricingResult:
    """Result of a single pricing operation."""
    price: float
    greeks: Optional[Dict[str, float | np.ndarray]]
    instrument_type: str
    engine_used: str


@dataclass
class PortfolioResult:
    """Result of a portfolio pricing operation."""
    total_value: float
    total_greeks: Optional[Dict[str, float]]
    breakdown: List[PricingResult]


class Pricer:
    """
    Main entry point for dictionary-based pricing.
    
    Validates the configuration, builds the required underlying objects,
    executes the pricing model, and optionally calculates Greeks.
    """

    SUPPORTED_TYPES = {
        'vanilla': VanillaOptions,
        'asian': AsianOptions,
        'barrier': BarrierOptions,
        'american': AmericanOption,
        'basket': BasketOption,
        'rainbow': RainbowOption,
        'spread': SpreadOption,
        'forward': Forwards,
        'future': Futures,
    }

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the pricer with a configuration dictionary.

        Args:
            config: Dictionary containing instrument parameters, pricing
                engine choice, and optional greek flags.
        """
        self.config = config.copy()
        self._validate_basic()
        
    def _validate_basic(self):
        """Validate required keys based on instrument type."""
        if 'type' not in self.config:
            raise ValueError("Missing required key: 'type'")
            
        inst_type = self.config['type']
        if inst_type not in self.SUPPORTED_TYPES:
            raise ValueError(f"Unknown instrument type '{inst_type}'. Supported: {', '.join(self.SUPPORTED_TYPES.keys())}")
            
        # Common required keys
        required = ['S', 'K', 'T', 'r']
        if inst_type not in ['forward', 'future'] and self.config.get('engine') != 'heston':
            required.append('vol')
        if inst_type not in ['forward', 'future', 'basket', 'rainbow', 'spread']:
            required.append('option_type')

        missing = [k for k in required if k not in self.config]
        if missing:
            raise ValueError(f"Missing required keys for '{inst_type}': {missing}")

    def _build_instrument(self, curve):
        """Build the appropriate Instrument based on config."""
        inst_type = self.config['type']
        cls = self.SUPPORTED_TYPES[inst_type]
        
        # Base params
        params = {
            'S': self.config['S'],
            'K': self.config['K'],
            'T': self.config['T'],
            'vol': self.config.get('vol'),
            'curve': curve,
            'r': self.config['r'],
        }

        # Multi-asset params
        if inst_type in ['basket', 'rainbow', 'spread']:
            params['S_arr'] = np.array(params.pop('S'))
            params['vol_arr'] = np.array(params.pop('vol'))
            
            if 'corr' not in self.config:
                raise ValueError(f"Missing 'corr' matrix for {inst_type}")
            params['corr_matrix'] = np.array(self.config['corr'])
            
            if inst_type in ['basket', 'rainbow', 'spread']:
                params['option_type'] = self.config.get('option_type')
                
            if 'weights' in self.config:
                params['weights'] = np.array(self.config['weights'])

        # Option type
        if 'option_type' in self.config and inst_type not in ['forward', 'future', 'basket', 'rainbow', 'spread']:
            params['option_type'] = self.config['option_type']

        # Barrier specific
        if inst_type == 'barrier':
            for bk in ['barrier_price', 'barrier_kind', 'barrier_direction']:
                if bk not in self.config:
                    raise ValueError(f"Missing '{bk}' for barrier option")
                params[bk] = self.config[bk]

        return cls(**params)

    def _get_engine_choice(self) -> str:
        """Determine which engine to use based on config and defaults."""
        inst_type = self.config['type']
        engine = self.config.get('engine')

        if not engine:
            if inst_type == 'vanilla':
                engine = 'bsm'
            elif inst_type == 'american':
                engine = 'binomial'
            else:
                engine = 'mc'

        # Validate compatibility
        if inst_type in ['basket', 'rainbow', 'spread', 'asian', 'barrier', 'forward', 'future'] and engine == 'bsm':
            raise ValueError(f"Engine 'bsm' not supported for '{inst_type}'. Use: mc")
        if inst_type == 'american' and engine != 'binomial':
             raise ValueError("American options currently require the 'binomial' engine in quick API.")

        return engine

    def _build_mc_engine(self, curve):
        """Build Monte Carlo engine."""
        n_steps = self.config.get('n_steps', 252)
        n_paths = self.config.get('n_paths', 10000)
        seed = self.config.get('seed')
        return MonteCarlo(n=n_steps, M=n_paths, curve=curve, seed=seed)

    def _build_heston_engine(self):
        """Build Heston engine."""
        req = ['v0', 'kappa', 'theta', 'xi', 'rho_heston']
        missing = [k for k in req if k not in self.config]
        if missing:
            raise ValueError(f"Heston engine requires: {', '.join(req)}")
            
        return Heston(
            v0=self.config['v0'],
            kappa=self.config['kappa'],
            theta=self.config['theta'],
            xi=self.config['xi'],
            rho=self.config['rho_heston'],
            r=self.config['r'],
            n=self.config.get('n_steps', 252),
            paths=self.config.get('n_paths', 10000),
            seed=self.config.get('seed')
        )

    def _build_binomial_engine(self, curve):
        """Build Binomial tree engine."""
        n_steps = self.config.get('n_steps', 500)
        return BinominalTree(n_steps=n_steps, curve=curve)

    def run(self) -> PricingResult:
        """Execute the pricing process."""
        inst_type = self.config['type']
        engine_choice = self._get_engine_choice()
        calc_greeks = self.config.get('greeks', False)

        curve = FlatCurve(self.config['r'])
        instrument = self._build_instrument(curve)

        price = 0.0
        greeks_dict = None

        if engine_choice == 'bsm':
            bsm = BlackScholes(
                S=instrument.S, K=instrument.K, T=instrument.T,
                vol=instrument.vol, r=instrument.r, option_type=instrument.option_type
            )
            price = bsm.black_scholes()
            if calc_greeks:
                greeks_dict = {
                    'delta': float(bsm.black_scholes_delta()),
                    'gamma': float(bsm.black_scholes_gamma()),
                    'vega': float(bsm.black_scholes_vega()),
                    'theta': float(bsm.black_scholes_theta()),
                    'rho': float(bsm.black_scholes_rho()),
                }

        elif engine_choice == 'binomial':
            tree = self._build_binomial_engine(curve)
            price = tree.price(instrument, american=(inst_type == 'american'))
            
            if calc_greeks:
                 raise NotImplementedError("Greeks for BinomialTree not exposed via Quick API yet.")

        elif engine_choice in ['mc', 'heston']:
            is_multi = inst_type in ['basket', 'rainbow', 'spread']
            
            if engine_choice == 'heston':
                if is_multi:
                    raise ValueError("Heston engine currently not supported for multi-asset in Quick API.")
                engine = self._build_heston_engine()
            else:
                engine = self._build_mc_engine(curve)

            if is_multi:
                pricer = MultiAssetMCPricer(engine, instrument)
            else:
                pricer = MCPricer(engine, instrument)

            price = pricer.price()

            if calc_greeks:
                greek_calc = MultiAssetGreeks(engine, instrument) if is_multi else Greeks(engine, instrument)
                delta, gamma, vega, theta, rho = greek_calc.finite_difference()
                
                if is_multi:
                     greeks_dict = delta  # delta, gamma, vega, theta, rho comes as a dict from MultiAssetGreeks
                else:
                    greeks_dict = {
                        'delta': delta,
                        'gamma': gamma,
                        'vega': vega,
                        'theta': theta,
                        'rho': rho
                    }

        else:
            raise ValueError(f"Unknown engine: {engine_choice}")

        return PricingResult(
            price=float(price),
            greeks=greeks_dict,
            instrument_type=inst_type,
            engine_used=engine_choice
        )

    @classmethod
    def portfolio(cls, configs: List[Dict[str, Any]], greeks: bool = False) -> PortfolioResult:
        """
        Price a portfolio of instruments defined by a list of configurations.
        
        Args:
            configs: List of instrument configuration dictionaries. Must include
                'quantity' for each position.
            greeks: Whether to compute portfolio-level Greeks.
            
        Returns:
            PortfolioResult containing total value, aggregated Greeks, and
            individual pricing breakdowns.
        """
        port = Portfolio()
        breakdown = []
        
        total_value = 0.0
        total_greeks = {'delta': 0.0, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0, 'rho': 0.0} if greeks else None

        for cfg in configs:
            if 'quantity' not in cfg:
                raise ValueError("Portfolio configs must include a 'quantity' key.")
            
            qty = cfg['quantity']
            
            # Force greek calculation if portfolio requests it, but don't overwrite if False
            run_cfg = cfg.copy()
            if greeks:
                run_cfg['greeks'] = True
                
            pricer = cls(run_cfg)
            res = pricer.run()
            
            breakdown.append(res)
            total_value += res.price * qty
            
            if greeks and res.greeks:
                for k in total_greeks:
                    val = res.greeks[k]
                    if isinstance(val, np.ndarray):
                        total_greeks[k] += np.sum(val) * qty
                    else:
                        total_greeks[k] += val * qty

        return PortfolioResult(
            total_value=total_value,
            total_greeks=total_greeks,
            breakdown=breakdown
        )
