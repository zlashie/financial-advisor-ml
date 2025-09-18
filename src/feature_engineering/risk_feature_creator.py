########################################
#### Dependencies
########################################

import pandas as pd
import numpy as np
from ..interfaces.feature_engineering_interfaces import FeatureCreator
from ..models.feature_engineering_models import FinancialConstants
from ..config import config

########################################
#### Classes
########################################

class RiskFeatureCreator(FeatureCreator):
    """Creates risk-related features."""
    
    def __init__(self, financial_constants: FinancialConstants):
        self.constants = financial_constants
        
        #### Load risk feature configuration ####
        self.feature_config = config.get_section('feature_engineering', 'risk_features')
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create risk-related features."""
        df_result = df.copy()
        
        #### Age-based risk capacity ####
        age_column = self.feature_config.get('age_column', 'age')
        retirement_age = self.feature_config.get('retirement_age', self.constants.retirement_age)
        working_years = self.feature_config.get('working_years', self.constants.working_years)
        
        df_result['years_to_retirement'] = np.maximum(
            retirement_age - df_result[age_column], 
            self.constants.no_negative_value
        )
        df_result['risk_capacity'] = df_result['years_to_retirement'] / working_years
        
        #### Financial flexibility score ####
        income_column = self.feature_config.get('income_column', 'monthly_income')
        discretionary_column = self.feature_config.get('discretionary_column', 'monthly_discretionary')
        
        df_result['discretionary_ratio'] = df_result[discretionary_column] / df_result[income_column]
        
        #### Create age risk categories as NUMERIC dummy variables, not strings ####
        if self.feature_config.get('create_age_risk_categories', True):
            age_thresholds = self.feature_config.get('age_risk_thresholds', {
                'young': 35,
                'middle': 50
            })
            
            #### Create binary dummy variables instead of categorical strings ####
            df_result['age_high_risk_capacity'] = (df_result[age_column] < age_thresholds['young']).astype(int)
            df_result['age_low_risk_capacity'] = (df_result[age_column] >= age_thresholds['middle']).astype(int)
            #### Medium risk capacity is implied when both are 0 ####
        
        if self.feature_config.get('create_financial_stability_score', True):
            #### Combine multiple factors for financial stability ####
            stability_weights = self.feature_config.get('stability_weights', {
                'discretionary_ratio': 0.4,
                'risk_capacity': 0.6
            })
            
            df_result['financial_stability_score'] = (
                df_result['discretionary_ratio'] * stability_weights['discretionary_ratio'] +
                df_result['risk_capacity'] * stability_weights['risk_capacity']
            )
        
        if self.feature_config.get('create_liquidity_features', True):
            #### Emergency fund estimation (months of expenses covered) ####
            emergency_fund_months = self.feature_config.get('emergency_fund_months', 6)
            monthly_expenses = df_result[income_column] - df_result[discretionary_column]
            df_result['emergency_fund_needed'] = monthly_expenses * emergency_fund_months
            
            #### Liquidity ratio (discretionary income as proxy for available cash flow) ####
            df_result['liquidity_ratio'] = df_result[discretionary_column] / monthly_expenses
        
        return df_result
