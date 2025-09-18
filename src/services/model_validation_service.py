########################################
#### Dependencies
########################################

import numpy as np
from ..models.model_config_models import ModelEvaluationConfig
from ..config import config

########################################
#### Classes
########################################

class ModelValidationService:
    """Service for model validation and prediction clipping"""
    
    def __init__(self, config_model: ModelEvaluationConfig):
        self._config = config_model
        
        #### Load validation configuration ####
        self.validation_config = config.get_section('models', 'validation')
    
    def validate_and_clip_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Clip predictions to valid investment ratio range"""
        min_value = self.validation_config.get('min_prediction', self._config.invest_ratio_min)
        max_value = self.validation_config.get('max_prediction', self._config.invest_ratio_max)
        
        return np.clip(predictions, min_value, max_value)
    
    def is_model_trained(self, model) -> bool:
        """Check if model is trained"""
        trained_attribute = self.validation_config.get('trained_attribute', 'is_trained')
        return hasattr(model, trained_attribute) and getattr(model, trained_attribute)
    
    def validate_training_data(self, X_train, y_train) -> dict:
        """Validate training data quality"""
        validation_results = {}
        
        #### Check for minimum sample size ####
        min_samples = self.validation_config.get('min_training_samples', 100)
        validation_results['sufficient_samples'] = len(X_train) >= min_samples
        
        #### Check for missing values ####
        max_missing_ratio = self.validation_config.get('max_missing_ratio', 0.1)
        missing_ratio = X_train.isnull().sum().sum() / (X_train.shape[0] * X_train.shape[1])
        validation_results['acceptable_missing_data'] = missing_ratio <= max_missing_ratio
        
        #### Check target variable range ####
        target_in_range = (
            (y_train >= self._config.min_valid_range).all() and 
            (y_train <= self._config.max_valid_range).all()
        )
        validation_results['target_in_valid_range'] = target_in_range
        
        #### Check for feature variance ####
        min_variance = self.validation_config.get('min_feature_variance', 1e-6)
        low_variance_features = X_train.var() < min_variance
        validation_results['features_with_low_variance'] = low_variance_features.sum()
        validation_results['acceptable_feature_variance'] = low_variance_features.sum() == 0
        
        return validation_results
    
    def validate_predictions(self, predictions: np.ndarray) -> dict:
        """Validate model predictions"""
        validation_results = {}
        
        #### Check for NaN or infinite values ####
        validation_results['has_invalid_predictions'] = (
            np.isnan(predictions).any() or np.isinf(predictions).any()
        )
        
        #### Check prediction range ####
        in_range_count = np.sum(
            (predictions >= self._config.min_valid_range) & 
            (predictions <= self._config.max_valid_range)
        )
        validation_results['predictions_in_range_ratio'] = in_range_count / len(predictions)
        
        #### Check for prediction diversity ####
        unique_predictions = len(np.unique(np.round(predictions, 3)))
        min_diversity = self.validation_config.get('min_prediction_diversity', 10)
        validation_results['sufficient_prediction_diversity'] = unique_predictions >= min_diversity
        
        return validation_results
