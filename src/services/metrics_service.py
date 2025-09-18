########################################
#### Dependencies
########################################

import numpy as np
import pandas as pd
from typing import Dict
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from ..interfaces.model_interfaces import IMetricsCalculator
from ..models.model_config_models import ModelEvaluationConfig
from ..config import config

########################################
#### Classes
########################################

class FinancialMetricsCalculator(IMetricsCalculator):
    """Service for calculating financial model metrics"""
    
    def __init__(self, config_model: ModelEvaluationConfig):
        self._config = config_model
        
        #### Load metrics configuration ####
        self.metrics_config = config.get_section('models', 'metrics')
    
    def calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, dataset_name: str) -> Dict[str, float]:
        """Calculate comprehensive regression metrics"""
        metrics = {}
        
        #### Core regression metrics ####
        enabled_metrics = self.metrics_config.get('enabled_metrics', 
                                                 ['r2', 'mae', 'rmse', 'mean_prediction'])
        
        if 'r2' in enabled_metrics:
            metrics['r2'] = r2_score(y_true, y_pred)
        
        if 'mae' in enabled_metrics:
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
        
        if 'rmse' in enabled_metrics:
            metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        
        if 'mean_prediction' in enabled_metrics:
            metrics['mean_prediction'] = np.mean(y_pred)
        
        #### Dataset identifier ####
        metrics['dataset'] = dataset_name
        
        #### Valid prediction rate ####
        if self.metrics_config.get('calculate_valid_prediction_rate', True):
            valid_predictions = np.sum(
                (y_pred >= self._config.min_valid_range) & 
                (y_pred <= self._config.max_valid_range)
            ) / len(y_pred)
            metrics['valid_prediction_rate'] = valid_predictions
        
        #### Additional financial-specific metrics ####
        if self.metrics_config.get('calculate_financial_metrics', True):
            #### Mean absolute percentage error ####
            if 'mape' in enabled_metrics:
                #### Avoid division by zero ####
                mask = y_true != 0
                if mask.any():
                    metrics['mape'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
                else:
                    metrics['mape'] = 0.0
             
            #### Prediction accuracy within tolerance ####
            tolerance = self.metrics_config.get('prediction_tolerance', 0.05)
            within_tolerance = np.abs(y_true - y_pred) <= tolerance
            metrics['accuracy_within_tolerance'] = np.mean(within_tolerance)
            
            #### Risk-adjusted metrics for investment ratios ####
            if self.metrics_config.get('calculate_risk_metrics', True):
                #### Conservative vs aggressive prediction bias ####
                conservative_threshold = self.metrics_config.get('conservative_threshold', 0.3)
                aggressive_threshold = self.metrics_config.get('aggressive_threshold', 0.7)
                
                conservative_mask = y_true <= conservative_threshold
                aggressive_mask = y_true >= aggressive_threshold
                
                if conservative_mask.any():
                    metrics['conservative_mae'] = mean_absolute_error(y_true[conservative_mask], 
                                                                   y_pred[conservative_mask])
                
                if aggressive_mask.any():
                    metrics['aggressive_mae'] = mean_absolute_error(y_true[aggressive_mask], 
                                                                  y_pred[aggressive_mask])
        
        #### Custom metrics if configured ####
        custom_metrics = self.metrics_config.get('custom_metrics', {})
        for metric_name, metric_config in custom_metrics.items():
            if metric_config.get('enabled', False):
                metrics[metric_name] = self._calculate_custom_metric(
                    y_true, y_pred, metric_config
                )
        
        return metrics
    
    def _calculate_custom_metric(self, y_true: pd.Series, y_pred: np.ndarray, 
                               metric_config: Dict) -> float:
        """Calculate custom metrics based on configuration"""
        metric_type = metric_config.get('type', 'mean_absolute_error')
        
        if metric_type == 'weighted_mae':
            #### Weight errors differently based on prediction range ####
            weights = np.ones_like(y_pred)
            
            #### Higher weight for extreme predictions ####
            extreme_threshold = metric_config.get('extreme_threshold', 0.8)
            extreme_weight = metric_config.get('extreme_weight', 2.0)
            
            extreme_mask = (y_pred >= extreme_threshold) | (y_pred <= (1 - extreme_threshold))
            weights[extreme_mask] = extreme_weight
            
            weighted_errors = weights * np.abs(y_true - y_pred)
            return np.mean(weighted_errors)
        
        elif metric_type == 'directional_accuracy': 
            #### Measure if prediction direction matches actual direction ####
            baseline = metric_config.get('baseline', 0.5)
            pred_direction = y_pred >= baseline
            true_direction = y_true >= baseline
            return np.mean(pred_direction == true_direction)
        
        else:
            #### Default to MAE ####
            return mean_absolute_error(y_true, y_pred)
    
    def calculate_cross_validation_metrics(self, cv_results: Dict) -> Dict[str, float]:
        """Calculate cross-validation summary metrics"""
        if not self.metrics_config.get('enable_cv_metrics', True):
            return {}
        
        cv_metrics = {}
        
        for metric_name, scores in cv_results.items():
            if isinstance(scores, (list, np.ndarray)):
                cv_metrics[f'{metric_name}_mean'] = np.mean(scores)
                cv_metrics[f'{metric_name}_std'] = np.std(scores)
                cv_metrics[f'{metric_name}_min'] = np.min(scores)
                cv_metrics[f'{metric_name}_max'] = np.max(scores)
        
        return cv_metrics
