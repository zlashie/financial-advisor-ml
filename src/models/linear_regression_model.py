########################################
#### Dependencies
########################################

import pandas as pd
import numpy as np
from typing import Dict, List
from sklearn.linear_model import LinearRegression

from .base_financial_model import BaseFinancialModel
from ..services.metrics_service import FinancialMetricsCalculator
from ..services.model_validation_service import ModelValidationService
from ..config import config

########################################
#### Classes
########################################

class LinearRegressionModel(BaseFinancialModel):
    """Linear Regression for financial decision prediction"""
    
    def __init__(self, metrics_calculator: FinancialMetricsCalculator, 
                 validation_service: ModelValidationService):
        super().__init__("Linear Regression", metrics_calculator, validation_service)
        
        #### Load linear regression configuration ####
        self.lr_config = config.get_section('models', 'linear_regression')
    
    def _create_model(self) -> LinearRegression:
        """Create Linear Regression Model"""
        fit_intercept = self.lr_config.get('fit_intercept', True)
        return LinearRegression(fit_intercept=fit_intercept)
    
    def get_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        """Get feature coefficients"""
        if not self._validation_service.is_model_trained(self):
            error_message = self.lr_config.get('feature_importance_error', 'Model not trained yet!')
            raise ValueError(error_message)
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': self.model.coef_,
            'abs_coefficient': np.abs(self.model.coef_)
        }).sort_values('abs_coefficient', ascending=False)
        
        return importance_df
    
    def interpret_prediction(self, X_sample: pd.DataFrame, feature_names: List[str]) -> Dict:
        """Explain how a prediction was made"""
        if not self._validation_service.is_model_trained(self):
            error_message = self.lr_config.get('interpretation_error', 'Model not trained yet!')
            raise ValueError(error_message)
        
        prediction = self.predict(X_sample)[0]
        feature_contributions = X_sample.iloc[0] * self.model.coef_
        intercept_contribution = self.model.intercept_
        
        contributions_df = pd.DataFrame({
            'feature': feature_names,
            'value': X_sample.iloc[0].values,
            'coefficient': self.model.coef_,
            'contribution': feature_contributions
        }).sort_values('contribution', key=abs, ascending=False)
        
        return {
            'prediction': prediction,
            'intercept': intercept_contribution,
            'feature_contributions': contributions_df,
            'total_contribution': intercept_contribution + feature_contributions.sum()
        }
