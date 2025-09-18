########################################
#### Dependencies
########################################

import pandas as pd
from typing import Dict

from ..interfaces.model_interfaces import IModelComparator, IFinancialModel
from ..models.model_config_models import ModelEvaluationConfig
from ..config import config

########################################
#### Classes
########################################

class ModelComparisonService(IModelComparator):
    """Service for comparing multiple models"""
    
    def __init__(self, config_model: ModelEvaluationConfig):
        self._config = config_model
        self.models: Dict[str, IFinancialModel] = {}
        self.results: Dict[str, Dict] = {}
        
        #### Load comparison configuration ####
        self.comparison_config = config.get_section('models', 'comparison')
    
    def add_model(self, model: IFinancialModel) -> None:
        """Add a model to the comparison"""
        self.models[model.model_name] = model
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train all models on the same training data"""
        training_header = self.comparison_config.get('training_header', "=== TRAINING ALL MODELS ===")
        print(training_header)
        
        for name, model in self.models.items():
            training_message = self.comparison_config.get('training_message', "Training {model}...")
            print(f"\n{training_message.format(model=name)}")
            training_metrics = model.train(X_train, y_train)
            self.results[name] = {'training': training_metrics}
    
    def evaluate_all_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """Evaluate all models on the same test data"""
        evaluation_header = self.comparison_config.get('evaluation_header', "=== EVALUATING ALL MODELS ===")
        print(f"\n{evaluation_header}")
        
        for name, model in self.models.items():
            evaluation_message = self.comparison_config.get('evaluation_message', "Evaluating {model}...")
            print(f"\n{evaluation_message.format(model=name)}")
            test_metrics = model.evaluate(X_test, y_test)
            self.results[name]['test'] = test_metrics
    
    def compare_results(self) -> pd.DataFrame:
        """Create comparison table of all models"""
        comparison_data = []
        
        #### Get metric configurations ####
        metrics_config = self.comparison_config.get('metrics', {
            'r2': {'train_col': 'Train_R2', 'test_col': 'Test_R2'},
            'mae': {'train_col': 'Train_MAE', 'test_col': 'Test_MAE'},
            'rmse': {'train_col': 'Train_RMSE', 'test_col': 'Test_RMSE'}
        })
        
        for name, results in self.results.items():
            train_metrics = results.get('training', {})
            test_metrics = results.get('test', {})
            
            row_data = {'Model': name}
            
            #### Add configured metrics ####
            for metric, cols in metrics_config.items():
                row_data[cols['train_col']] = train_metrics.get(metric, self._config.fallback_default)
                row_data[cols['test_col']] = test_metrics.get(metric, self._config.fallback_default)
            
            #### Calculate overfitting ####
            overfitting_col = self.comparison_config.get('overfitting_column', 'Overfitting')
            primary_metric = self.comparison_config.get('primary_metric', 'r2')
            row_data[overfitting_col] = (
                train_metrics.get(primary_metric, self._config.fallback_default) - 
                test_metrics.get(primary_metric, self._config.fallback_default)
            )
            
            comparison_data.append(row_data)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        #### Sort by primary metric ####
        sort_column = metrics_config[primary_metric]['test_col']
        sort_ascending = self.comparison_config.get('sort_ascending', False)
        comparison_df = comparison_df.sort_values(sort_column, ascending=sort_ascending)
        
        #### Display results ####
        results_header = self.comparison_config.get('results_header', "=== MODEL COMPARISON RESULTS ===")
        print(f"\n{results_header}")
        
        decimal_places = self.comparison_config.get('decimal_places', 4)
        print(comparison_df.round(decimal_places))
        
        if not comparison_df.empty:
            best_model_message = self.comparison_config.get('best_model_message', 
                                                          "Best performing model: {model}")
            best_score_message = self.comparison_config.get('best_score_message', 
                                                          "Test {metric} Score: {score:.4f}")
            
            best_model = comparison_df.iloc[0]['Model']
            best_score = comparison_df.iloc[0][sort_column]
            
            print(f"\n{best_model_message.format(model=best_model)}")
            print(f"{best_score_message.format(metric=primary_metric.upper(), score=best_score)}")
        
        return comparison_df
