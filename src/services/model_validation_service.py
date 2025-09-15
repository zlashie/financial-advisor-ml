########################################
#### Dependencies
########################################

import numpy as np
from ..models.model_config_models import ModelEvaluationConfig

########################################
#### Classes
########################################

class ModelValidationService:
    """Service for model validation and prediction clipping"""
    
    def __init__(self, config: ModelEvaluationConfig):
        self._config = config
    
    def validate_and_clip_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Clip predictions to valid investment ratio range"""
        return np.clip(predictions, self._config.invest_ratio_min, self._config.invest_ratio_max)
    
    def is_model_trained(self, model) -> bool:
        """Check if model is trained"""
        return hasattr(model, 'is_trained') and model.is_trained

