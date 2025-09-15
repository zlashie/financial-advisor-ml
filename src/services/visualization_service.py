########################################
#### Dependencies
########################################

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List

from ..interfaces.model_interfaces import IModelVisualizer, IFinancialModel
from ..models.model_config_models import VisualizationConfig

########################################
#### Classes
########################################

class ModelVisualizationService(IModelVisualizer):
    """Service for model visualization"""
    
    def __init__(self, config: VisualizationConfig):
        self._config = config
    
    def plot_feature_importance(self, model: IFinancialModel, feature_names: List[str], top_n: int = None) -> None:
        """Plot feature importance for models that support it"""
        if not hasattr(model, 'get_feature_importance'):
            raise ValueError(f"Model {model.model_name} does not support feature importance")
        
        top_n = top_n or self._config.top_features
        importance_df = model.get_feature_importance(feature_names)
        
        plt.figure(figsize=(self._config.figure_width, self._config.figure_height))
        top_features = importance_df.head(top_n)
        
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'{model.model_name} - Top {top_n} Feature Importances')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    def plot_predictions_comparison(self, models: Dict[str, IFinancialModel], X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """Plot actual vs predicted values for all models"""
        n_models = len(models)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (name, model) in enumerate(models.items()):
            y_pred = model.predict(X_test)
            
            axes[idx].scatter(y_test, y_pred, alpha=0.6)
            axes[idx].plot([0, 1], [0, 1], 'r--', lw=2)
            axes[idx].set_xlabel('Actual Investment Ratio')
            axes[idx].set_ylabel('Predicted Investment Ratio')
            axes[idx].set_title(f'{name}')
            axes[idx].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_training_history(self, model) -> None:
        """Plot training history for neural networks"""
        if not hasattr(model, 'get_training_history'):
            print(f"Model {model.model_name} does not support training history visualization")
            return
        
        try:
            history = model.get_training_history()
            
            plt.figure(figsize=(self._config.figure_width, self._config.figure_height))
            plt.plot(history['loss_curve'])
            plt.title(f'{model.model_name} Training Loss')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.grid(True)
            
            if history['converged']:
                plt.axvline(x=history['n_iterations'], color='red', linestyle='--', 
                        label=f'Converged at iteration {history["n_iterations"]}')
                plt.legend()
            else:
                plt.title(f'{model.model_name} Training Loss (Did not converge)')
            
            plt.show()
        except Exception as e:
            print(f"Error plotting training history: {e}")

    