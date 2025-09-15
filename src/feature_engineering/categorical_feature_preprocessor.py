########################################
#### Dependencies
########################################

import pandas as pd
import numpy as np
from ..interfaces.feature_engineering_interfaces import FeaturePreprocessor
from ..models.feature_engineering_models import DemographicsConfig, MarketValuationConfig

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
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML by creating categorical variables."""
        df_result = df.copy()
        
        # Create age groups
        age_groups = self.demographics_config.age_groups
        df_result['age_group'] = pd.cut(
            df_result['age'],
            bins=[0, age_groups['young_max'], age_groups['middle_max'], 
                  age_groups['mature_max'], age_groups['senior_max']],
            labels=age_groups['labels']
        )
        
        # Create income groups
        income_groups = self.demographics_config.income_groups
        income_percentiles = df_result['annual_income'].quantile([
            income_groups['percentiles']['low'],
            income_groups['percentiles']['high']
        ])
        
        df_result['income_group'] = pd.cut(
            df_result['annual_income'],
            bins=[0, income_percentiles.iloc[0], income_percentiles.iloc[1], float('inf')],
            labels=income_groups['labels']
        )
        
        # Create market condition categories
        market_conditions = self.market_config.market_conditions
        df_result['market_condition'] = np.where(
            df_result['market_attractiveness'] > market_conditions['favorable_threshold'],
            self.market_condition_labels[0],  # FAVORABLE
            np.where(df_result['market_attractiveness'] < market_conditions['unfavorable_threshold'],
                    self.market_condition_labels[2],  # UNFAVORABLE
                    self.market_condition_labels[1])  # NEUTRAL
        )
        
        # One-hot encode categorical variables
        categorical_columns = ['age_group', 'income_group', 'market_condition']
        
        for col in categorical_columns:
            dummies = pd.get_dummies(df_result[col], prefix=col, drop_first=True)
            df_result = pd.concat([df_result, dummies], axis=1)
            df_result.drop(col, axis=1, inplace=True)
        
        return df_result
