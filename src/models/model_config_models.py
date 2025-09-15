########################################
#### Dependencies
########################################

from dataclasses import dataclass
from typing import Tuple

########################################
#### Classes
########################################

@dataclass
class ModelEvaluationConfig:
    """Configuration for model evaluation"""
    invest_ratio_min: float
    invest_ratio_max: float
    min_valid_range: float
    max_valid_range: float
    fallback_default: float

@dataclass
class RandomForestConfig:
    """Configuration for Random Forest model"""
    n_estimators: int
    max_depth: int
    min_samples_split: int
    min_samples_leaf: int
    n_jobs: int
    random_state: int

@dataclass
class NeuralNetworkConfig:
    """Configuration for Neural Network model"""
    hidden_layer_sizes: Tuple[int, ...]
    max_iterations: int
    regularization: float
    validation_fraction: float
    activation: str
    solver: str
    learning_rate: str
    early_stopping: bool
    patience_iterations: int
    random_state: int

@dataclass
class VisualizationConfig:
    """Configuration for visualization"""
    top_features: int
    figure_width: int
    figure_height: int