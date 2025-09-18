########################################
#### Dependencies
########################################

import pandas as pd
import numpy as np
from ..interfaces.feature_engineering_interfaces import FeaturePreprocessor
from ..models.feature_engineering_models import DemographicsConfig, MarketValuationConfig
from ..config import config

########################################
#### Classes
########################################

class CategoricalFeaturePreprocessor(FeaturePreprocessor):
    """Preprocesses features by creating categorical variables and one-hot encoding."""
    
    def __init__(self, demographics_config: DemographicsConfig, 
                 market_config: MarketValuationConfig, 
                 market_condition_labels: list):
        self.demographics_config = demographics_config
        self.market_config = market_config
        self.market_condition_labels = market_condition_labels
        
        #### Load preprocessing configuration ####
        self.preprocessing_config = config.get_section('feature_engineering', 'preprocessing')
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML by creating categorical variables."""
        df_result = df.copy()
        
        #### Create age groups ####
        age_groups = self.demographics_config.age_groups
        age_bins = [0, age_groups['young_max'], age_groups['middle_max'], 
                   age_groups['mature_max'], age_groups['senior_max']]
        
        df_result['age_group'] = pd.cut(
            df_result['age'],
            bins=age_bins,
            labels=age_groups['labels']
        )
        
        ##### Create income groups using dynamic percentiles ####
        income_groups = self.demographics_config.income_groups
        income_percentiles = df_result['annual_income'].quantile([
            income_groups['percentiles']['low'],
            income_groups['percentiles']['high']
        ])
        
        income_bins = [0, income_percentiles.iloc[0], income_percentiles.iloc[1], float('inf')]
        df_result['income_group'] = pd.cut(
            df_result['annual_income'],
            bins=income_bins,
            labels=income_groups['labels']
        )
        
        ##### Create market condition categories ####
        market_conditions = self.market_config.market_conditions
        df_result['market_condition'] = np.where(
            df_result['market_attractiveness'] > market_conditions['favorable_threshold'],
            self.market_condition_labels[0],  # FAVORABLE
            np.where(df_result['market_attractiveness'] < market_conditions['unfavorable_threshold'],
                    self.market_condition_labels[2],  # UNFAVORABLE
                    self.market_condition_labels[1])  # NEUTRAL
        )
        
        #### One-hot encode categorical variables ####
        categorical_columns = self.preprocessing_config.get('categorical_columns', 
                                                          ['age_group', 'income_group', 'market_condition'])
        drop_first = self.preprocessing_config.get('drop_first_category', True)
        
        for col in categorical_columns:
            if col in df_result.columns:
                dummies = pd.get_dummies(df_result[col], prefix=col, drop_first=drop_first)
                df_result = pd.concat([df_result, dummies], axis=1)
                df_result.drop(col, axis=1, inplace=True)
        
        return df_result
