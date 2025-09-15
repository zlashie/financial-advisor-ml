########################################
#### Dependencies
########################################

import pandas as pd
from typing import List
from sklearn.ensemble import RandomForestRegressor

from .base_financial_model import BaseFinancialModel
from ..services.metrics_service import FinancialMetricsCalculator
from ..services.model_validation_service import ModelValidationService
from ..models.model_config_models import RandomForestConfig

########################################
#### Classes
########################################

class RandomForestModel(BaseFinancialModel):
    """Random Forest for financial decision prediction"""
    
    def __init__(self, config: RandomForestConfig, metrics_calculator: FinancialMetricsCalculator, 
                 validation_service: ModelValidationService):
        super().__init__("Random Forest", metrics_calculator, validation_service)
        self._config = config
    
    def _create_model(self) -> RandomForestRegressor:
        """Create Random Forest Model"""
        return RandomForestRegressor(
            n_estimators=self._config.n_estimators,
            max_depth=self._config.max_depth,
            random_state=self._config.random_state,
            min_samples_split=self._config.min_samples_split,
            min_samples_leaf=self._config.min_samples_leaf,
            n_jobs=self._config.n_jobs
        )
    
    def get_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        """Get feature importance from Random Forest"""
        if not self._validation_service.is_model_trained(self):
            raise ValueError("Model not trained yet!")
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df