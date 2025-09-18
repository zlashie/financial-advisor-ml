########################################
#### Dependencies
########################################

import pandas as pd
import numpy as np
from ..interfaces.feature_engineering_interfaces import FeatureCreator
from ..models.feature_engineering_models import DebtAnalysisConfig, FinancialConstants
from ..config import config

########################################
#### Classes
########################################

class DebtFeatureCreator(FeatureCreator):
    """Creates debt-related features."""
    
    def __init__(self, debt_config: DebtAnalysisConfig, financial_constants: FinancialConstants):
        self.debt_config = debt_config
        self.constants = financial_constants
        
        #### Load debt feature configuration ####
        self.feature_config = config.get_section('feature_engineering', 'debt_features')
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create debt-related features."""
        df_result = df.copy()
        
        #### Total debt calculation ####
        debt_columns = self.feature_config.get('debt_columns', ['cc_debt', 'mortgage_debt'])
        df_result['total_debt'] = df_result[debt_columns].sum(axis=1)
        
        #### Annual income calculation ####
        income_multiplier = self.feature_config.get('income_multiplier', self.constants.months_per_year)
        df_result['annual_income'] = df_result['monthly_income'] * income_multiplier
        
        #### Debt-to-income ratio ####
        df_result['debt_to_income_ratio'] = df_result['total_debt'] / df_result['annual_income']
        df_result['debt_to_income_ratio'] = df_result['debt_to_income_ratio'].fillna(self.constants.no_debt)
        
        #### Weighted average debt rate ####
        rate_columns = self.feature_config.get('rate_columns', ['cc_rate', 'mortgage_rate'])
        df_result['weighted_debt_rate'] = np.where(
            df_result['total_debt'] > self.constants.no_debt,
            (df_result[debt_columns[0]] * df_result[rate_columns[0]] + 
             df_result[debt_columns[1]] * df_result[rate_columns[1]]) / df_result['total_debt'],
            self.constants.no_debt
        )
        
        #### Debt urgency score ####
        urgency_thresholds = self.debt_config.urgency_thresholds
        urgency_scores = self.debt_config.urgency_scores
        
        df_result['debt_urgency'] = np.where(
            df_result['weighted_debt_rate'] > urgency_thresholds['high_ratio'], 
            urgency_scores['high'],
            np.where(df_result['weighted_debt_rate'] > urgency_thresholds['medium_ratio'], 
                    urgency_scores['medium'], 
                    urgency_scores['low'])
        )
        
        #### Additional debt metrics if configured ####
        if self.feature_config.get('create_debt_service_ratio', True):
            monthly_payment_rate = self.feature_config.get('monthly_payment_rate', 0.02)
            df_result['estimated_monthly_payment'] = df_result['total_debt'] * monthly_payment_rate
            df_result['debt_service_ratio'] = df_result['estimated_monthly_payment'] / df_result['monthly_income']
        
        return df_result
