########################################
#### Dependencies
########################################

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any
import joblib

from ..interfaces.model_interfaces import IFinancialModel
from ..services.metrics_service import FinancialMetricsCalculator
from ..services.model_validation_service import ModelValidationService
from ..config import config

########################################
#### Classes
########################################

class BaseFinancialModel(IFinancialModel, ABC):
    """Abstract base class for all financial prediction models"""
    
    def __init__(self, model_name: str, metrics_calculator: FinancialMetricsCalculator, 
                 validation_service: ModelValidationService):
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        self.training_metrics = {}
        self._metrics_calculator = metrics_calculator
        self._validation_service = validation_service
        
        #### Load base model configuration ####
        self.base_config = config.get_section('models', 'base_model')
    
    @abstractmethod
    def _create_model(self) -> Any:
        """Create the specific ML algorithm instance"""
        pass
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, float]:
        """Train the model and return training metrics"""
        training_message = self.base_config.get('training_message', 'Training {model}...')
        print(training_message.format(model=self.model_name))
        
        self.model = self._create_model()
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        y_train_pred = self.model.predict(X_train)
        self.training_metrics = self._metrics_calculator.calculate_metrics(
            y_train, y_train_pred, "training"
        )
        
        completion_message = self.base_config.get('completion_message', '{model} training complete!')
        r2_message = self.base_config.get('r2_message', 'Training R²: {r2:.4f}')
        
        print(completion_message.format(model=self.model_name))
        print(r2_message.format(r2=self.training_metrics['r2']))
        
        return self.training_metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data"""
        if not self._validation_service.is_model_trained(self):
            error_message = self.base_config.get('not_trained_error', '{model} not trained yet!')
            raise ValueError(error_message.format(model=self.model_name))
        
        predictions = self.model.predict(X)
        return self._validation_service.validate_and_clip_predictions(predictions)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model performance on test data"""
        if not self._validation_service.is_model_trained(self):
            error_message = self.base_config.get('not_trained_error', '{model} not trained yet!')
            raise ValueError(error_message.format(model=self.model_name))
        
        y_pred = self.predict(X_test)
        test_metrics = self._metrics_calculator.calculate_metrics(y_test, y_pred, "test")
        
        #### Configurable test results display ####
        results_header = self.base_config.get('results_header', '\n{model} Test Results:')
        r2_result = self.base_config.get('r2_result', 'R² Score: {r2:.4f}')
        mae_result = self.base_config.get('mae_result', 'MAE: {mae:.4f}')
        rmse_result = self.base_config.get('rmse_result', 'RMSE: {rmse:.4f}')
        
        print(results_header.format(model=self.model_name))
        print(r2_result.format(r2=test_metrics['r2']))
        print(mae_result.format(mae=test_metrics['mae']))
        print(rmse_result.format(rmse=test_metrics['rmse']))
        
        return test_metrics
    
    def save_model(self, filepath: str) -> None:
        """Save trained model"""
        if not self._validation_service.is_model_trained(self):
            error_message = self.base_config.get('save_untrained_error', 'Cannot save untrained model')
            raise ValueError(error_message)
        
        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'training_metrics': self.training_metrics
        }
        
        joblib.dump(model_data, filepath)
        
        save_message = self.base_config.get('save_message', '{model} saved to {filepath}')
        print(save_message.format(model=self.model_name, filepath=filepath))
    
    def load_model(self, filepath: str) -> None:
        """Load trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.model_name = model_data['model_name']
        self.training_metrics = model_data.get('training_metrics', {})
        self.is_trained = True
        
        load_message = self.base_config.get('load_message', '{model} loaded from {filepath}')
        print(load_message.format(model=self.model_name, filepath=filepath))
