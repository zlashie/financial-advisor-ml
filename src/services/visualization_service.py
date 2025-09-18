########################################
#### Dependencies
########################################

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List

from ..interfaces.model_interfaces import IModelVisualizer, IFinancialModel
from ..models.model_config_models import VisualizationConfig
from ..config import config

########################################
#### Classes
########################################

class ModelVisualizationService(IModelVisualizer):
    """Service for model visualization"""
    
    def __init__(self, config_model: VisualizationConfig):
        self._config = config_model
        
        #### Load visualization configuration ####
        self.viz_config = config.get_section('visualization', 'plots')
        self.style_config = config.get_section('visualization', 'styling')
    
    def plot_feature_importance(self, model: IFinancialModel, feature_names: List[str], top_n: int = None) -> None:
        """Plot feature importance for models that support it"""
        if not hasattr(model, 'get_feature_importance'):
            error_message = self.viz_config.get('feature_importance_error', 
                                              "Model {model} does not support feature importance")
            raise ValueError(error_message.format(model=model.model_name))
        
        top_n = top_n or self._config.top_features
        importance_df = model.get_feature_importance(feature_names)
        
        #### Configure plot ####
        figsize = (self._config.figure_width, self._config.figure_height)
        plt.figure(figsize=figsize)
        top_features = importance_df.head(top_n)
        
        #### Plot styling ####
        bar_color = self.style_config.get('bar_color', 'steelblue')
        plt.barh(range(len(top_features)), top_features['importance'], color=bar_color)
        plt.yticks(range(len(top_features)), top_features['feature'])
        
        #### Labels and title ####
        xlabel = self.viz_config.get('feature_importance_xlabel', 'Feature Importance')
        title_template = self.viz_config.get('feature_importance_title', 
                                           '{model} - Top {n} Feature Importances')
        plt.xlabel(xlabel)
        plt.title(title_template.format(model=model.model_name, n=top_n))
        
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if self.viz_config.get('show_plots', True):
            plt.show()
        
        if self.viz_config.get('save_plots', False):
            save_path = self.viz_config.get('save_directory', 'plots')
            filename = f"{model.model_name.lower().replace(' ', '_')}_feature_importance.png"
            plt.savefig(f"{save_path}/{filename}", dpi=self.style_config.get('dpi', 300))
    
    def plot_predictions_comparison(self, models: Dict[str, IFinancialModel], X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """Plot actual vs predicted values for all models"""
        n_models = len(models)
        subplot_width = self.viz_config.get('subplot_width', 5)
        subplot_height = self.viz_config.get('subplot_height', 5)
        
        fig, axes = plt.subplots(1, n_models, figsize=(subplot_width * n_models, subplot_height))
        
        if n_models == 1:
            axes = [axes]
        
        #### Styling configuration ####
        scatter_alpha = self.style_config.get('scatter_alpha', 0.6)
        scatter_color = self.style_config.get('scatter_color', 'blue')
        reference_line_color = self.style_config.get('reference_line_color', 'red')
        reference_line_style = self.style_config.get('reference_line_style', '--')
        reference_line_width = self.style_config.get('reference_line_width', 2)
        grid_enabled = self.style_config.get('grid_enabled', True)
        
        for idx, (name, model) in enumerate(models.items()):
            y_pred = model.predict(X_test)
            
            axes[idx].scatter(y_test, y_pred, alpha=scatter_alpha, color=scatter_color)
            
            #### Reference line (perfect prediction) - FIXED: separate color and linestyle ####
            ref_range = self.viz_config.get('reference_line_range', [0, 1])
            axes[idx].plot(ref_range, ref_range, 
                          color=reference_line_color, 
                          linestyle=reference_line_style,
                          linewidth=reference_line_width)
            
            #### Labels ####
            xlabel = self.viz_config.get('predictions_xlabel', 'Actual Investment Ratio')
            ylabel = self.viz_config.get('predictions_ylabel', 'Predicted Investment Ratio')
            axes[idx].set_xlabel(xlabel)
            axes[idx].set_ylabel(ylabel)
            axes[idx].set_title(name)
            
            if grid_enabled:
                axes[idx].grid(True)
        
        plt.tight_layout()
        
        if self.viz_config.get('show_plots', True):
            plt.show()
        
        if self.viz_config.get('save_plots', False):
            save_path = self.viz_config.get('save_directory', 'plots')
            filename = "model_predictions_comparison.png"
            plt.savefig(f"{save_path}/{filename}", dpi=self.style_config.get('dpi', 300))
    
    def plot_training_history(self, model) -> None:
        """Plot training history for neural networks"""
        if not hasattr(model, 'get_training_history'):
            no_history_message = self.viz_config.get('no_training_history_message',
                                                   "Model {model} does not support training history visualization")
            print(no_history_message.format(model=model.model_name))
            return
        
        try:
            history = model.get_training_history()
            
            figsize = (self._config.figure_width, self._config.figure_height)
            plt.figure(figsize=figsize)
            
            #### Plot styling ####
            line_color = self.style_config.get('line_color', 'blue')
            convergence_line_color = self.style_config.get('convergence_line_color', 'red')
            convergence_line_style = self.style_config.get('convergence_line_style', '--')
            
            plt.plot(history['loss_curve'], color=line_color)
            
            #### Title and labels ####
            title_template = self.viz_config.get('training_history_title', '{model} Training Loss')
            xlabel = self.viz_config.get('training_history_xlabel', 'Iteration')
            ylabel = self.viz_config.get('training_history_ylabel', 'Loss')
            
            plt.title(title_template.format(model=model.model_name))
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            
            if self.style_config.get('grid_enabled', True):
                plt.grid(True)
            
            #### Convergence indicator ####
            if history['converged']:
                convergence_label = self.viz_config.get('convergence_label', 
                                                      'Converged at iteration {iteration}')
                plt.axvline(x=history['n_iterations'], 
                          color=convergence_line_color, 
                          linestyle=convergence_line_style,
                          label=convergence_label.format(iteration=history["n_iterations"]))
                plt.legend()
            else:
                no_convergence_title = self.viz_config.get('no_convergence_title',
                                                         '{model} Training Loss (Did not converge)')
                plt.title(no_convergence_title.format(model=model.model_name))
            
            if self.viz_config.get('show_plots', True):
                plt.show()
            
            if self.viz_config.get('save_plots', False):
                save_path = self.viz_config.get('save_directory', 'plots')
                filename = f"{model.model_name.lower().replace(' ', '_')}_training_history.png"
                plt.savefig(f"{save_path}/{filename}", dpi=self.style_config.get('dpi', 300))
                
        except Exception as e:
            error_message = self.viz_config.get('plotting_error_message', 
                                              "Error plotting training history: {error}")
            print(error_message.format(error=e))
