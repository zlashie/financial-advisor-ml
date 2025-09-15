########################################
#### Dependencies
########################################

import pandas as pd
import numpy as np
from ..interfaces.feature_engineering_interfaces import FeatureCreator
from ..models.feature_engineering_models import FinancialConstants

########################################
#### Classes
########################################

class RiskFeatureCreator(FeatureCreator):
    """Creates risk-related features."""
    
    def __init__(self, financial_constants: FinancialConstants):
        self.constants = financial_constants
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create risk-related features."""
        df_result = df.copy()
        
        # Age-based risk capacity
        df_result['years_to_retirement'] = np.maximum(
            self.constants.retirement_age - df_result['age'], 
            self.constants.no_negative_value
        )
        df_result['risk_capacity'] = df_result['years_to_retirement'] / self.constants.working_years
        
        # Financial flexibility score
        df_result['discretionary_ratio'] = df_result['monthly_discretionary'] / df_result['monthly_income']
        
        return df_result
