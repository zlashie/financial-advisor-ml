########################################
#### Dependencies
########################################

from typing import Dict
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib

from .base_financial_model import BaseFinancialModel
from ..services.metrics_service import FinancialMetricsCalculator
from ..services.model_validation_service import ModelValidationService
from ..models.model_config_models import NeuralNetworkConfig
from ..config import config

########################################
#### Classes
########################################

class NeuralNetworkModel(BaseFinancialModel):
    """Neural Network for financial decision prediction"""
    
    def __init__(self, config_model: NeuralNetworkConfig, metrics_calculator: FinancialMetricsCalculator, 
                 validation_service: ModelValidationService):
        super().__init__("Neural Network", metrics_calculator, validation_service)
        self._config = config_model
        self._scaler = StandardScaler()
        self._is_fitted_scaler = False
        
        #### Load neural network configuration ####
        self.nn_config = config.get_section('models', 'neural_network')
    
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
            learning_rate_init=self.nn_config.get('learning_rate_init', 0.001),
            tol=self.nn_config.get('tolerance', 1e-4)
        )
    
    def train(self, X_train, y_train):
        """Train with data scaling for better performance"""
        training_message = self.base_config.get('training_message', 'Training {model}...')
        print(training_message.format(model=self.model_name))
        
        #### Scale the features for neural networks ####
        X_train_scaled = self._scaler.fit_transform(X_train)
        self._is_fitted_scaler = True
        
        self.model = self._create_model()
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        #### Make predictions on scaled data ####
        y_train_pred = self.model.predict(X_train_scaled)
        self.training_metrics = self._metrics_calculator.calculate_metrics(
            y_train, y_train_pred, "training"
        )
        
        completion_message = self.base_config.get('completion_message', '{model} training complete!')
        r2_message = self.base_config.get('r2_message', 'Training R²: {r2:.4f}')
        
        print(completion_message.format(model=self.model_name))
        print(r2_message.format(r2=self.training_metrics['r2']))
        
        return self.training_metrics
    
    def predict(self, X):
        """Make predictions with scaling"""
        if not self._validation_service.is_model_trained(self):
            error_message = self.base_config.get('not_trained_error', '{model} not trained yet!')
            raise ValueError(error_message.format(model=self.model_name))
        
        if not self._is_fitted_scaler:
            scaler_error = self.nn_config.get('scaler_error', 'Scaler not fitted!')
            raise ValueError(scaler_error)
        
        #### Scale the input data ####
        X_scaled = self._scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return self._validation_service.validate_and_clip_predictions(predictions)
    
    def get_training_history(self) -> Dict:
        """Get training history (loss curve)"""
        if not self._validation_service.is_model_trained(self):
            error_message = self.nn_config.get('history_error', 'Model not trained yet!')
            raise ValueError(error_message)
        
        return {
            'loss_curve': self.model.loss_curve_,
            'n_iterations': self.model.n_iter_,
            'converged': self.model.n_iter_ < self._config.max_iterations
        }
    
    def save_model(self, filepath: str) -> None:
        """Save trained model including scaler"""
        if not self._validation_service.is_model_trained(self):
            error_message = self.base_config.get('save_untrained_error', 'Cannot save untrained model')
            raise ValueError(error_message)
        
        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'training_metrics': self.training_metrics,
            'scaler': self._scaler,
            'is_fitted_scaler': self._is_fitted_scaler
        }
        
        joblib.dump(model_data, filepath)
        
        save_message = self.base_config.get('save_message', '{model} saved to {filepath}')
        print(save_message.format(model=self.model_name, filepath=filepath))
    
    def load_model(self, filepath: str) -> None:
        """Load trained model including scaler"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.model_name = model_data['model_name']
        self.training_metrics = model_data.get('training_metrics', {})
        self._scaler = model_data.get('scaler', StandardScaler())
        self._is_fitted_scaler = model_data.get('is_fitted_scaler', False)
        self.is_trained = True
        
        load_message = self.base_config.get('load_message', '{model} loaded from {filepath}')
        print(load_message.format(model=self.model_name, filepath=filepath))
