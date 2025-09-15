########################################
#### Dependencies
########################################

from ..config import config
from ..models.model_config_models import (
    ModelEvaluationConfig, RandomForestConfig, 
    NeuralNetworkConfig, VisualizationConfig
)

########################################
#### Classes
########################################

class ModelConfigFactory:
    """Factory for creating model configurations"""
    
    @staticmethod
    def create_evaluation_config() -> ModelEvaluationConfig:
        """Create model evaluation configuration"""
        evaluation_config = config.get('models', 'evaluation')
        bounds_config = evaluation_config['investment_ratio_bounds']
        valid_range_config = evaluation_config['valid_range']
        
        return ModelEvaluationConfig(
            invest_ratio_min=bounds_config['min'],
            invest_ratio_max=bounds_config['max'],
            min_valid_range=valid_range_config['min'],
            max_valid_range=valid_range_config['max'],
            fallback_default=config.get('models', 'fallback_values', 'default')
        )
    
    @staticmethod
    def create_random_forest_config() -> RandomForestConfig:
        """Create Random Forest configuration"""
        rf_config = config.get('models', 'random_forest')
        random_state = config.get('common', 'general', 'random_state')
        
        return RandomForestConfig(
            n_estimators=rf_config['n_estimators'],
            max_depth=rf_config['max_depth'],
            min_samples_split=rf_config['min_samples_split'],
            min_samples_leaf=rf_config['min_samples_leaf'],
            n_jobs=rf_config['n_jobs'],
            random_state=random_state
        )
    
    @staticmethod
    def create_neural_network_config() -> NeuralNetworkConfig:
        """Create Neural Network configuration with improved defaults"""
        nn_config = config.get('models', 'neural_network')
        random_state = config.get('common', 'general', 'random_state')
        
        return NeuralNetworkConfig(
            hidden_layer_sizes=tuple(nn_config.get('hidden_layer_sizes', [100, 50])),  # Better architecture
            max_iterations=max(nn_config.get('max_iterations', 1000), 1000),  # Ensure minimum iterations
            regularization=nn_config.get('regularization', 0.001),  # Better regularization
            validation_fraction=nn_config.get('validation_fraction', 0.1),
            activation=nn_config.get('activation', 'relu'),
            solver=nn_config.get('solver', 'adam'),  # Better optimizer
            learning_rate=nn_config.get('learning_rate', 'adaptive'),  # Adaptive learning rate
            early_stopping=nn_config.get('early_stopping', True),
            patience_iterations=nn_config.get('patience_iterations', 50),  # More patience
            random_state=random_state
        )
    
    @staticmethod
    def create_visualization_config() -> VisualizationConfig:
        """Create visualization configuration"""
        evaluation_config = config.get('models', 'evaluation')
        viz_config = evaluation_config['visualization']
        figure_size = viz_config['figure_size']
        
        return VisualizationConfig(
            top_features=viz_config['top_features'],
            figure_width=figure_size['width'],
            figure_height=figure_size['height']
        )