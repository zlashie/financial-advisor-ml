########################################
#### Dependencies
########################################

import pandas as pd
import numpy as np
from ..interfaces.feature_engineering_interfaces import FeatureCreator
from ..models.feature_engineering_models import DebtAnalysisConfig, FinancialConstants

########################################
#### Classes
########################################

class DebtFeatureCreator(FeatureCreator):
    """Creates debt-related features."""
    
    def __init__(self, debt_config: DebtAnalysisConfig, financial_constants: FinancialConstants):
        self.debt_config = debt_config
        self.constants = financial_constants
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create debt-related features."""
        df_result = df.copy()
        
        # Total debt calculation
        df_result['total_debt'] = df_result['cc_debt'] + df_result['mortgage_debt']
        
        # Annual income calculation
        df_result['annual_income'] = df_result['monthly_income'] * self.constants.months_per_year
        
        # Debt-to-income ratio
        df_result['debt_to_income_ratio'] = df_result['total_debt'] / df_result['annual_income']
        df_result['debt_to_income_ratio'] = df_result['debt_to_income_ratio'].fillna(self.constants.no_debt)
        
        # Weighted average debt rate
        df_result['weighted_debt_rate'] = np.where(
            df_result['total_debt'] > self.constants.no_debt,
            (df_result['cc_debt'] * df_result['cc_rate'] + 
             df_result['mortgage_debt'] * df_result['mortgage_rate']) / df_result['total_debt'],
            self.constants.no_debt
        )
        
        # Debt urgency score
        urgency_thresholds = self.debt_config.urgency_thresholds
        urgency_scores = self.debt_config.urgency_scores
        
        df_result['debt_urgency'] = np.where(
            df_result['weighted_debt_rate'] > urgency_thresholds['high_ratio'], 
            urgency_scores['high'],
            np.where(df_result['weighted_debt_rate'] > urgency_thresholds['medium_ratio'], 
                    urgency_scores['medium'], 
                    urgency_scores['low'])
        )
        
        return df_result
