########################################
#### Dependencies
########################################

import numpy as np
import pandas as pd
from typing import Dict
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from ..interfaces.model_interfaces import IMetricsCalculator
from ..models.model_config_models import ModelEvaluationConfig

########################################
#### Classes
########################################

class FinancialMetricsCalculator(IMetricsCalculator):
    """Service for calculating financial model metrics"""
    
    def __init__(self, config: ModelEvaluationConfig):
        self._config = config
    
    def calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, dataset_name: str) -> Dict[str, float]:
        """Calculate comprehensive regression metrics"""
        metrics = {
            'r2': r2_score(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'dataset': dataset_name,
            'mean_prediction': np.mean(y_pred)
        }
        
        valid_predictions = np.sum(
            (y_pred >= self._config.min_valid_range) & 
            (y_pred <= self._config.max_valid_range)
        ) / len(y_pred)
        metrics['valid_prediction_rate'] = valid_predictions
        
        return metrics