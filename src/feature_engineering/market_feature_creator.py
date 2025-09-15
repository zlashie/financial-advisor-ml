########################################
#### Dependencies
########################################

import pandas as pd
import numpy as np
from ..interfaces.feature_engineering_interfaces import FeatureCreator
from ..models.feature_engineering_models import MarketValuationConfig

########################################
#### Classes
########################################

class MarketFeatureCreator(FeatureCreator):
    """Creates market-related features."""
    
    def __init__(self, market_config: MarketValuationConfig):
        self.market_config = market_config
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create market-related features."""
        df_result = df.copy()
        
        # Market valuation score based on P/E ratio
        pe_thresholds = self.market_config.pe_thresholds
        price_scoring = self.market_config.price_scoring
        
        pe_score = np.where(
            df_result['sp500_pe'] > pe_thresholds['fair_value_max'], 
            price_scoring['expensive'],
            np.where(df_result['sp500_pe'] < pe_thresholds['fair_value_min'], 
                    price_scoring['cheap'], 
                    price_scoring['fair'])
        )
        
        # Market psychology score based on VIX
        vix_thresholds = self.market_config.vix_thresholds
        psychology_scoring = self.market_config.psychology_scoring
        
        vix_score = np.where(
            df_result['vix'] > vix_thresholds['balanced_max'], 
            psychology_scoring['fearful'],
            np.where(df_result['vix'] < vix_thresholds['balanced_min'], 
                    psychology_scoring['greedy'], 
                    psychology_scoring['balanced'])
        )
        
        # Combined market attractiveness
        df_result['market_attractiveness'] = pe_score + vix_score
        
        return df_result
