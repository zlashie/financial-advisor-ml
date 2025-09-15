########################################
#### Dependencies
########################################

from ..models.linear_regression_model import LinearRegressionModel
from ..models.random_forest_model import RandomForestModel
from ..models.neural_network_model import NeuralNetworkModel
from ..services.metrics_service import FinancialMetricsCalculator
from ..services.model_validation_service import ModelValidationService
from ..models.model_config_models import RandomForestConfig, NeuralNetworkConfig

########################################
#### Classes
########################################

class ModelFactory:
    """Factory for creating financial models"""
    
    @staticmethod
    def create_linear_regression(metrics_calculator: FinancialMetricsCalculator, 
                               validation_service: ModelValidationService) -> LinearRegressionModel:
        """Create Linear Regression model"""
        return LinearRegressionModel(metrics_calculator, validation_service)
    
    @staticmethod
    def create_random_forest(config: RandomForestConfig, 
                           metrics_calculator: FinancialMetricsCalculator,
                           validation_service: ModelValidationService) -> RandomForestModel:
        """Create Random Forest model"""
        return RandomForestModel(config, metrics_calculator, validation_service)
    
    @staticmethod
    def create_neural_network(config: NeuralNetworkConfig,
                            metrics_calculator: FinancialMetricsCalculator,
                            validation_service: ModelValidationService) -> NeuralNetworkModel:
        """Create Neural Network model"""
        return NeuralNetworkModel(config, metrics_calculator, validation_service)