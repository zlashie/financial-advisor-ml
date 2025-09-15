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
    
    @abstractmethod
    def _create_model(self) -> Any:
        """Create the specific ML algorithm instance"""
        pass
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, float]:
        """Train the model and return training metrics"""
        print(f"Training {self.model_name}...")
        
        self.model = self._create_model()
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        y_train_pred = self.model.predict(X_train)
        self.training_metrics = self._metrics_calculator.calculate_metrics(
            y_train, y_train_pred, "training"
        )
        
        print(f"{self.model_name} training complete!")
        print(f"Training R²: {self.training_metrics['r2']:.4f}")
        
        return self.training_metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data"""
        if not self._validation_service.is_model_trained(self):
            raise ValueError(f"{self.model_name} not trained yet!")
        
        predictions = self.model.predict(X)
        return self._validation_service.validate_and_clip_predictions(predictions)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model performance on test data"""
        if not self._validation_service.is_model_trained(self):
            raise ValueError(f"{self.model_name} not trained yet!")
        
        y_pred = self.predict(X_test)
        test_metrics = self._metrics_calculator.calculate_metrics(y_test, y_pred, "test")
        
        print(f"\n{self.model_name} Test Results:")
        print(f"R² Score: {test_metrics['r2']:.4f}")
        print(f"MAE: {test_metrics['mae']:.4f}")
        print(f"RMSE: {test_metrics['rmse']:.4f}")
        
        return test_metrics
    
    def save_model(self, filepath: str) -> None:
        """Save trained model"""
        if not self._validation_service.is_model_trained(self):
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'training_metrics': self.training_metrics
        }
        
        joblib.dump(model_data, filepath)
        print(f"{self.model_name} saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.model_name = model_data['model_name']
        self.training_metrics = model_data.get('training_metrics', {})
        self.is_trained = True
        print(f"{self.model_name} loaded from {filepath}")

