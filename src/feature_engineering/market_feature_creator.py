########################################
#### Dependencies
########################################

import pandas as pd
import numpy as np
from ..interfaces.feature_engineering_interfaces import FeatureCreator
from ..models.feature_engineering_models import MarketValuationConfig
from ..config import config

########################################
#### Classes
########################################

class MarketFeatureCreator(FeatureCreator):
    """Creates market-related features."""
    
    def __init__(self, market_config: MarketValuationConfig):
        self.market_config = market_config
        
        #### Load market feature configuration ####
        self.feature_config = config.get_section('feature_engineering', 'market_features')
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create market-related features."""
        df_result = df.copy()
        
        #### Market valuation score based on P/E ratio ####
        pe_column = self.feature_config.get('pe_column', 'sp500_pe')
        pe_thresholds = self.market_config.pe_thresholds
        price_scoring = self.market_config.price_scoring
        
        pe_score = np.where(
            df_result[pe_column] > pe_thresholds['fair_value_max'], 
            price_scoring['expensive'],
            np.where(df_result[pe_column] < pe_thresholds['fair_value_min'], 
                    price_scoring['cheap'], 
                    price_scoring['fair'])
        )
        
        ##### Market psychology score based on VIX ####
        vix_column = self.feature_config.get('vix_column', 'vix')
        vix_thresholds = self.market_config.vix_thresholds
        psychology_scoring = self.market_config.psychology_scoring
        
        vix_score = np.where(
            df_result[vix_column] > vix_thresholds['balanced_max'], 
            psychology_scoring['fearful'],
            np.where(df_result[vix_column] < vix_thresholds['balanced_min'], 
                    psychology_scoring['greedy'], 
                    psychology_scoring['balanced'])
        )
        
        #### Combined market attractiveness ####
        attractiveness_column = self.feature_config.get('attractiveness_column', 'market_attractiveness')
        df_result[attractiveness_column] = pe_score + vix_score
        
        #### Additional market features if configured####
        if self.feature_config.get('create_risk_premium', True):
            treasury_column = self.feature_config.get('treasury_column', 'treasury_yield')
            expected_return = self.feature_config.get('base_market_return', 0.10)
            df_result['risk_premium'] = expected_return - df_result[treasury_column]
        
        ##### Create volatility categories as NUMERIC dummy variables, not strings ####
        if self.feature_config.get('create_volatility_categories', True):
            vix_categories = self.feature_config.get('vix_categories', {
                'low': 15,
                'high': 30
            })
            
            ##### Create binary dummy variables instead of categorical strings ####
            df_result['volatility_low'] = (df_result[vix_column] < vix_categories['low']).astype(int)
            df_result['volatility_high'] = (df_result[vix_column] > vix_categories['high']).astype(int)
            #### Medium is implied when both low and high are 0 ####
        
        return df_result
