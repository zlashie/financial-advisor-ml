########################################
#### Dependencies
########################################

from typing import Dict
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np

from .base_financial_model import BaseFinancialModel
from ..services.metrics_service import FinancialMetricsCalculator
from ..services.model_validation_service import ModelValidationService
from ..models.model_config_models import NeuralNetworkConfig

########################################
#### Classes
########################################

class NeuralNetworkModel(BaseFinancialModel):
    """Neural Network for financial decision prediction"""
    
    def __init__(self, config: NeuralNetworkConfig, metrics_calculator: FinancialMetricsCalculator, 
                 validation_service: ModelValidationService):
        super().__init__("Neural Network", metrics_calculator, validation_service)
        self._config = config
        self._scaler = StandardScaler()  # Add scaler for better performance
        self._is_fitted_scaler = False
    
    def _create_model(self) -> MLPRegressor:
        """Create Neural Network Model with improved parameters"""
        return MLPRegressor(
            hidden_layer_sizes=self._config.hidden_layer_sizes,
            activation=self._config.activation,
            solver=self._config.solver,
            learning_rate=self._config.learning_rate,
            max_iter=self._config.max_iterations,
            early_stopping=self._config.early_stopping,
            validation_fraction=self._config.validation_fraction,
            n_iter_no_change=self._config.patience_iterations,
            random_state=self._config.random_state,
            alpha=self._config.regularization,  
            learning_rate_init=0.001,  
            tol=1e-4  
        )
    
    def train(self, X_train, y_train):
        """Train with data scaling for better performance"""
        print(f"Training {self.model_name}...")
        
        # Scale the features for neural networks
        X_train_scaled = self._scaler.fit_transform(X_train)
        self._is_fitted_scaler = True
        
        self.model = self._create_model()
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Make predictions on scaled data
        y_train_pred = self.model.predict(X_train_scaled)
        self.training_metrics = self._metrics_calculator.calculate_metrics(
            y_train, y_train_pred, "training"
        )
        
        print(f"{self.model_name} training complete!")
        print(f"Training R²: {self.training_metrics['r2']:.4f}")
        
        return self.training_metrics
    
    def predict(self, X):
        """Make predictions with scaling"""
        if not self._validation_service.is_model_trained(self):
            raise ValueError(f"{self.model_name} not trained yet!")
        
        if not self._is_fitted_scaler:
            raise ValueError("Scaler not fitted!")
        
        # Scale the input data
        X_scaled = self._scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return self._validation_service.validate_and_clip_predictions(predictions)
    
    def get_training_history(self) -> Dict:
        """Get training history (loss curve)"""
        if not self._validation_service.is_model_trained(self):
            raise ValueError("Model not trained yet!")
        
        return {
            'loss_curve': self.model.loss_curve_,
            'n_iterations': self.model.n_iter_,
            'converged': self.model.n_iter_ < self._config.max_iterations
        }
    
    def save_model(self, filepath: str) -> None:
        """Save trained model including scaler"""
        if not self._validation_service.is_model_trained(self):
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'training_metrics': self.training_metrics,
            'scaler': self._scaler,  # Save scaler too
            'is_fitted_scaler': self._is_fitted_scaler
        }
        
        import joblib
        joblib.dump(model_data, filepath)
        print(f"{self.model_name} saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load trained model including scaler"""
        import joblib
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.model_name = model_data['model_name']
        self.training_metrics = model_data.get('training_metrics', {})
        self._scaler = model_data.get('scaler', StandardScaler())
        self._is_fitted_scaler = model_data.get('is_fitted_scaler', False)
        self.is_trained = True
        
        print(f"{self.model_name} loaded from {filepath}")
