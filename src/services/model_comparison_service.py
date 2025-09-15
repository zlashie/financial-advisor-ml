########################################
#### Dependencies
########################################

import pandas as pd
from typing import Dict

from ..interfaces.model_interfaces import IModelComparator, IFinancialModel
from ..models.model_config_models import ModelEvaluationConfig

########################################
#### Classes
########################################

class ModelComparisonService(IModelComparator):
    """Service for comparing multiple models"""
    
    def __init__(self, config: ModelEvaluationConfig):
        self._config = config
        self.models: Dict[str, IFinancialModel] = {}
        self.results: Dict[str, Dict] = {}
    
    def add_model(self, model: IFinancialModel) -> None:
        """Add a model to the comparison"""
        self.models[model.model_name] = model
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train all models on the same training data"""
        print("=== TRAINING ALL MODELS ===")
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            training_metrics = model.train(X_train, y_train)
            self.results[name] = {'training': training_metrics}
    
    def evaluate_all_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """Evaluate all models on the same test data"""
        print("\n=== EVALUATING ALL MODELS ===")
        
        for name, model in self.models.items():
            print(f"\nEvaluating {name}...")
            test_metrics = model.evaluate(X_test, y_test)
            self.results[name]['test'] = test_metrics
    
    def compare_results(self) -> pd.DataFrame:
        """Create comparison table of all models"""
        comparison_data = []
        
        for name, results in self.results.items():
            train_metrics = results.get('training', {})
            test_metrics = results.get('test', {})
            
            comparison_data.append({
                'Model': name,
                'Train_R2': train_metrics.get('r2', self._config.fallback_default),
                'Test_R2': test_metrics.get('r2', self._config.fallback_default),
                'Train_MAE': train_metrics.get('mae', self._config.fallback_default),
                'Test_MAE': test_metrics.get('mae', self._config.fallback_default),
                'Train_RMSE': train_metrics.get('rmse', self._config.fallback_default),
                'Test_RMSE': test_metrics.get('rmse', self._config.fallback_default),
                'Overfitting': train_metrics.get('r2', self._config.fallback_default) - test_metrics.get('r2', self._config.fallback_default)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Test_R2', ascending=False)
        
        print("\n=== MODEL COMPARISON RESULTS ===")
        print(comparison_df.round(4))
        
        if not comparison_df.empty:
            best_model = comparison_df.iloc[0]['Model']
            print(f"\nBest performing model: {best_model}")
            print(f"Test R² Score: {comparison_df.iloc[0]['Test_R2']:.4f}")
        
        return comparison_df