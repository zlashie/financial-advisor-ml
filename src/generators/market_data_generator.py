########################################
#### Dependencies
########################################

import numpy as np
import pandas as pd
from ..interfaces.data_generator_interfaces import MarketDataGenerator
from ..models.config_models import MarketConfig
from ..config import config

########################################
#### Classes
########################################

class SP500MarketDataGenerator(MarketDataGenerator):
    def __init__(self, config_model: MarketConfig, random_seed: int = None):
        self.config_model = config_model
        
        self.sp500_config = config.get_section('data_generation', 'market_conditions')['sp500']
        self.treasury_config = config.get_section('data_generation', 'market_conditions')['treasury']
        self.vix_config = config.get_section('data_generation', 'market_conditions')['vix']
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def generate(self, n_samples: int) -> pd.DataFrame:
        """Generate S&P 500 market condition data."""
        market_data = []
        
        for _ in range(n_samples):
            sp500_pe = max(
                self.sp500_config['pe_min'], 
                np.random.normal(self.sp500_config['mean'], self.sp500_config['stddev'])
            )
            treasury_yield = np.random.uniform(
                self.treasury_config['min_pct'], 
                self.treasury_config['max_pct']
            )
            vix = np.random.lognormal(
                self.vix_config['normal_distribution']['mean'], 
                self.vix_config['normal_distribution']['stddev']
            )
            
            market_data.append({
                'sp500_pe': sp500_pe,
                'treasury_yield': treasury_yield,
                'vix': min(vix, self.vix_config['max'])
            })
        
        return pd.DataFrame(market_data)
