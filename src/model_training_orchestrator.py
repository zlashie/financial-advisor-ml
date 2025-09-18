########################################
#### Dependencies
########################################

import pandas as pd
from pathlib import Path
from typing import List

from .factories.model_config_factory import ModelConfigFactory
from .factories.model_factory import ModelFactory
from .services.metrics_service import FinancialMetricsCalculator
from .services.model_validation_service import ModelValidationService
from .services.model_comparison_service import ModelComparisonService
from .services.visualization_service import ModelVisualizationService
from .config import config

########################################
#### Main
########################################

class ModelTrainingOrchestrator:
    """Main orchestrator for model training and evaluation"""
    
    def __init__(self):
        #### Create configurations ####
        self.evaluation_config = ModelConfigFactory.create_evaluation_config()
        self.rf_config = ModelConfigFactory.create_random_forest_config()
        self.nn_config = ModelConfigFactory.create_neural_network_config()
        self.viz_config = ModelConfigFactory.create_visualization_config()
        
        #### Create services ####
        self.metrics_calculator = FinancialMetricsCalculator(self.evaluation_config)
        self.validation_service = ModelValidationService(self.evaluation_config)
        self.comparison_service = ModelComparisonService(self.evaluation_config)
        self.visualization_service = ModelVisualizationService(self.viz_config)
    
    def create_models(self):
        """Create all models"""
        models = []
        
        #### Linear Regression ####
        lr_model = ModelFactory.create_linear_regression(
            self.metrics_calculator, self.validation_service
        )
        models.append(lr_model)
        
        #### Random Forest ####
        rf_model = ModelFactory.create_random_forest(
            self.rf_config, self.metrics_calculator, self.validation_service
        )
        models.append(rf_model)
        
        #### Neural Network ####
        nn_model = ModelFactory.create_neural_network(
            self.nn_config, self.metrics_calculator, self.validation_service
        )
        models.append(nn_model)
        
        return models
    
    def train_and_evaluate_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                                 X_test: pd.DataFrame, y_test: pd.Series,
                                 feature_names: List[str]):
        """Train and evaluate all models"""
        models = self.create_models()
        
        #### Add models to comparison service ####
        for model in models:
            self.comparison_service.add_model(model)
        
        #### Train all models ####
        self.comparison_service.train_all_models(X_train, y_train)
        
        #### Evaluate all models ####
        self.comparison_service.evaluate_all_models(X_test, y_test)
        
        #### Compare results ####
        comparison_df = self.comparison_service.compare_results()
        
        #### Visualizations ####
        self.visualization_service.plot_predictions_comparison(
            self.comparison_service.models, X_test, y_test
        )
        
        return comparison_df, models
    
    def save_best_model(self, models: List, comparison_df: pd.DataFrame, save_dir: str = None):
        """Save the best performing model"""
        if comparison_df.empty:
            print("No models to save")
            return
        
        #### Use config path if not provided ####
        if save_dir is None:
            save_dir = config.get('paths', 'models', 'save_directory')
        
        best_model_name = comparison_df.iloc[0]['Model']
        best_model = next(model for model in models if model.model_name == best_model_name)
        
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        #### Use config file extension ####
        model_extension = config.get('paths', 'models', 'file_extensions', 'model')
        model_file = save_path / f"{best_model_name.lower().replace(' ', '_')}_model{model_extension}"
        best_model.save_model(str(model_file))
        
        return best_model
    
    def load_training_data(self):
        """Load processed training data using config paths"""
        processed_dir = Path(config.get('paths', 'data', 'processed_directory'))
        training_files = config.get_section('paths', 'data')['training_files']
        
        #### Check if processed data exists ####
        required_files = ['X_train', 'X_test', 'y_train', 'y_test']
        for file_key in required_files:
            file_path = processed_dir / training_files[file_key]
            if not file_path.exists():
                raise FileNotFoundError(f"Processed training data not found: {file_path}")
        
        #### Load data ####
        X_train = pd.read_csv(processed_dir / training_files['X_train'])
        X_test = pd.read_csv(processed_dir / training_files['X_test'])
        y_train = pd.read_csv(processed_dir / training_files['y_train']).squeeze()
        y_test = pd.read_csv(processed_dir / training_files['y_test']).squeeze()
        
        feature_names = X_train.columns.tolist()
        
        print(f"Loaded training data: X_train {X_train.shape}, X_test {X_test.shape}")
        
        return X_train, X_test, y_train, y_test, feature_names

def main():
    """Main function for model training"""
    orchestrator = ModelTrainingOrchestrator()
    
    try:
        #### Load processed data ####
        X_train, X_test, y_train, y_test, feature_names = orchestrator.load_training_data()
        
        #### Train and evaluate models ####
        comparison_df, models = orchestrator.train_and_evaluate_models(
            X_train, y_train, X_test, y_test, feature_names
        )
        
        #### Save best model ####
        best_model = orchestrator.save_best_model(models, comparison_df)
        
        print("Model training completed successfully!")
        print("\nModel Comparison Results:")
        print(comparison_df)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run feature engineering first to generate processed data.")

if __name__ == "__main__":
    main()