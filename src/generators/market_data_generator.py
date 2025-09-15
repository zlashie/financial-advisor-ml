########################################
#### Dependendencies
########################################

import numpy as np
import pandas as pd
from ..interfaces.data_generator_interfaces import MarketDataGenerator
from ..models.config_models import MarketConfig

########################################
#### Classes
########################################

class SP500MarketDataGenerator(MarketDataGenerator):
    def __init__(self, config: MarketConfig, random_seed: int = None):
        self.config = config
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def generate(self, n_samples: int) -> pd.DataFrame:
        """Generate S&P 500 market condition data."""
        market_data = []
        
        for _ in range(n_samples):
            sp500_pe = max(
                self.config.sp500_pe_min, 
                np.random.normal(self.config.sp500_mean, self.config.sp500_stddev)
            )
            treasury_yield = np.random.uniform(
                self.config.treasury_min_pct, 
                self.config.treasury_max_pct
            )
            vix = np.random.lognormal(
                self.config.vix_normal_mean, 
                self.config.vix_normal_stddev
            )
            
            market_data.append({
                'sp500_pe': sp500_pe,
                'treasury_yield': treasury_yield,
                'vix': min(vix, self.config.vix_max)
            })
        
        return pd.DataFrame(market_data)