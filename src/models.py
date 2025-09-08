########################################
#### Dependendencies
########################################
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import joblib
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import config  


########################################
#### Load Configuration Values
########################################

# General model constants
RANDOM_STATE = config.get('common', 'general', 'random_state')

# Investment ratio bounds from evaluation config
evaluation_config = config.get('models', 'evaluation')
bounds_config = evaluation_config['investment_ratio_bounds']
INVEST_RATIO_MIN = bounds_config['min']
INVEST_RATIO_MAX = bounds_config['max']

# Valid range for predictions
valid_range_config = evaluation_config['valid_range']
MIN_VALID_RANGE = valid_range_config['min']
MAX_VALID_RANGE = valid_range_config['max']

# Fallback values
FALLBACK_DEFAULT = config.get('models', 'fallback_values', 'default')

# Random Forest configuration
rf_config = config.get('models', 'random_forest')
N_ESTIMATORS = rf_config['n_estimators']
MAX_DEPTH = rf_config['max_depth']
MIN_SAMPLE_SPLIT = rf_config['min_samples_split']
MIN_SAMPLE_LEAF = rf_config['min_samples_leaf']
CPU_CORES = rf_config['n_jobs']

# Neural Network configuration
nn_config = config.get('models', 'neural_network')
NEURAL_HIDDEN_LAYER_SIZES = tuple(nn_config['hidden_layer_sizes'])
NEURAL_MAX_RUNS = nn_config['max_iterations']
NEURAL_RANDOM_SEED = RANDOM_STATE  # Use common random state
NEURAL_REGULARIZATION = nn_config['regularization']
NEURAL_VALIDATION_FRACTION = nn_config['validation_fraction']
NEURAL_ACTIVATION = nn_config['activation']
NEURAL_SOLVER = nn_config['solver']
NEURAL_LEARNING_STRATEGY = nn_config['learning_rate']
NEURAL_EARLY_STOPPING = nn_config['early_stopping']
NEURAL_PATIENCE_ITERATIONS = nn_config['patience_iterations']

# Visualization configuration
viz_config = evaluation_config['visualization']
TOP_N = viz_config['top_features']
figure_size = viz_config['figure_size']
FIG_SIZE_WIDTH = figure_size['width']
FIG_SIZE_HEIGHT = figure_size['height']

########################################
#### Base Financial Model Class
########################################

class BaseFinancialModel(ABC):
    """
    Abstract base class for all financial prediction models to ensure consistency and enforces model standards.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        self.training_metrics = {}

    @abstractmethod
    def _create_model(self) -> Any:
        """
        Create the specific ML algorithm instance.
        """
        pass

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, float]:
        """
        Train the model and return training metrics.
        """
        print(f"Training {self.model_name}...")

        self.model = self._create_model()
        self.model.fit(X_train, y_train)
        self.is_trained = True

        y_train_pred = self.model.predict(X_train)
        self.training_metrics = self._calculate_metrics(y_train, y_train_pred, "training")

        print(f"{self.model_name} training complete!")
        print(f"Training R²: {self.training_metrics['r2']:.4f}")

        return self.training_metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        """
        if not self.is_trained:
            raise ValueError(f"{self.model_name} not trained yet!")
        
        predictions = self.model.predict(X)
        predictions = np.clip(predictions, INVEST_RATIO_MIN, INVEST_RATIO_MAX)

        return predictions
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        """
        if not self.is_trained:
            raise ValueError(f"{self.model_name} not trained yet!")
        
        y_pred = self.predict(X_test)
        test_metrics = self._calculate_metrics(y_test, y_pred, "test")
        
        print(f"\n{self.model_name} Test Results:")
        print(f"R² Score: {test_metrics['r2']:.4f}")
        print(f"MAE: {test_metrics['mae']:.4f}")
        print(f"RMSE: {test_metrics['rmse']:.4f}")
        
        return test_metrics
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, dataset_name: str) -> Dict[str, float]:
        """
        Calculate comprehensive regression metrics.

        - R²: How much variance we explain (higher = better, max = 1.0)
        - MAE: Average absolute error (lower = better, same units as target)
        - RMSE: Penalizes large errors more (lower = better)
        - Custom: Financial-specific metrics
        """
        metrics = {
            'r2': r2_score(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'dataset': dataset_name
        }

        metrics['mean_prediction'] = np.mean(y_pred)

        valid_predictions = np.sum((y_pred >= MIN_VALID_RANGE) & (y_pred <= MAX_VALID_RANGE)) / len(y_pred)
        metrics['valid_prediction_rate'] = valid_predictions

        return metrics
    
    def save_model(self, filepath: str):
        """
        Save trained model.
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'training_metrics': self.training_metrics
        }
        
        joblib.dump(model_data, filepath)
        print(f"{self.model_name} saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load trained model.
        """
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.model_name = model_data['model_name']
        self.training_metrics = model_data.get('training_metrics', {})
        self.is_trained = True
        print(f"{self.model_name} loaded from {filepath}")

########################################
#### Linear Regression Model 
########################################

class LinearRegressionModel(BaseFinancialModel):
    """
    Linear Regression for financial decision prediction.
    """

    def __init__(self):
        super().__init__("Linear Regression")
    
    def _create_model(self) -> LinearRegression:
        """
        Create Linear Regression Model.
        """
        return LinearRegression(fit_intercept=True)
    
    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """
        Get feature coefficients. 
        - Show which features most influence the decision
        - Positive coefficient = higher feature value → more investment
        - Negative coefficient = higher feature value → less investment
        - Magnitude shows strength of influence
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': self.model.coef_,
            'abs_coefficient': np.abs(self.model.coef_)
        }).sort_values('abs_coefficient', ascending=False)

        return importance_df
    
    def interpret_prediction(self, X_sample: pd.DataFrame, feature_names: list) -> Dict: 
        """
        Explain how a prediction was made to reveal financial decisions behind results of model.
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        prediction = self.predict(X_sample)[0]

        feature_contributions = X_sample.iloc[0] * self.model.coef_
        intercept_contribution = self.model.intercept_

        contributions_df = pd.DataFrame({
            'feature': feature_names,
            'value': X_sample.iloc[0].values,
            'coefficient': self.model.coef_,
            'contribution': feature_contributions
        }).sort_values('contribution', key=abs, ascending=False)

        return {
            'prediction': prediction,
            'intercept': intercept_contribution,
            'feature_contributions': contributions_df,
            'total_contribution': intercept_contribution + feature_contributions.sum()
        }

########################################
#### Random Forest Model
########################################

class RandomForestModel(BaseFinancialModel):
    """
    Random Forest for financial decision prediction.
    - Builds many decision trees on random subsets of data
    - Each tree votes on the prediction
    - Averages all votes for final prediction
    """
    def __init__(self, n_estimators: int = N_ESTIMATORS, max_depth: int = MAX_DEPTH, random_state: int = RANDOM_STATE):
        super().__init__("Random Forest")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

    def _create_model(self) -> RandomForestRegressor:
        """
        Create Random Forest Model.
        """
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            min_samples_split=MIN_SAMPLE_SPLIT,
            min_samples_leaf=MIN_SAMPLE_LEAF,
            n_jobs=CPU_CORES  
        )
    
    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """
        Get feature importance from Random Forest.
        - Measures how much each feature decreases impurity when used for splits
        - Higher importance = feature is more useful for making decisions
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def plot_feature_importance(self, feature_names: list, top_n: int = TOP_N):
        """
        Visualize feature importance.
        """
        importance_df = self.get_feature_importance(feature_names)
        
        plt.figure(figsize=(FIG_SIZE_WIDTH, FIG_SIZE_HEIGHT))
        top_features = importance_df.head(top_n)
        
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'{self.model_name} - Top {top_n} Feature Importances')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        
        return importance_df    

########################################
#### Neural Network Model 
########################################

class NeuralNetworkModel(BaseFinancialModel):
    """
    Neural Network for financial decision prediction.
    """

    def __init__(self, hidden_layers: tuple = NEURAL_HIDDEN_LAYER_SIZES, max_iter: int = NEURAL_MAX_RUNS, random_state: int = NEURAL_RANDOM_SEED):
        super().__init__("Neural Network")
        self.hidden_layers = hidden_layers
        self.max_iter = max_iter
        self.random_state = random_state

    def _create_model(self) -> MLPRegressor:
        """
        Create Neural Network Model.
        """

        return MLPRegressor(
            hidden_layer_sizes=self.hidden_layers,
            activation=NEURAL_ACTIVATION,
            solver=NEURAL_SOLVER,
            learning_rate=NEURAL_LEARNING_STRATEGY,
            max_iter=self.max_iter,
            early_stopping=NEURAL_EARLY_STOPPING,
            validation_fraction=NEURAL_VALIDATION_FRACTION,
            n_iter_no_change=NEURAL_PATIENCE_ITERATIONS,
            random_state=self.random_state
        )
    
    def get_training_history(self) -> Dict:
        """
        Get training history (loss curve) to monitor neural network learning behavior.
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        return {
            'loss_curve': self.model.loss_curve_,
            'n_iterations': self.model.n_iter_,
            'converged': self.model.n_iter_ < self.max_iter
        }
    
    def plot_training_history(self):
        """Visualize training progress."""
        history = self.get_training_history()
        
        plt.figure(figsize=(FIG_SIZE_WIDTH, FIG_SIZE_HEIGHT))
        plt.plot(history['loss_curve'])
        plt.title(f'{self.model_name} Training Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True)
        
        if history['converged']:
            plt.axvline(x=history['n_iterations'], color='red', linestyle='--', 
                       label=f'Converged at iteration {history["n_iterations"]}')
            plt.legend()
        
        plt.show()

########################################
#### Model Compare
########################################

class ModelComparison:
    """
    Compare multiple models on the same dataset.
    """

    def __init__(self):
        self.models = {}
        self.results = {}
    
    def add_model(self, model: BaseFinancialModel):
        """Add a model to the comparison."""
        self.models[model.model_name] = model
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Train all models on the same training data."""
        print("=== TRAINING ALL MODELS ===")
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            training_metrics = model.train(X_train, y_train)
            self.results[name] = {'training': training_metrics}
    
    def evaluate_all_models(self, X_test: pd.DataFrame, y_test: pd.Series):
        """Evaluate all models on the same test data."""
        print("\n=== EVALUATING ALL MODELS ===")
        
        for name, model in self.models.items():
            print(f"\nEvaluating {name}...")
            test_metrics = model.evaluate(X_test, y_test)
            self.results[name]['test'] = test_metrics

    def compare_results(self) -> pd.DataFrame:
        """
        Create comparison table of all models.
        
        What to look for:
        - Highest R² score (explains most variance)
        - Lowest MAE/RMSE (smallest errors)
        - Small gap between training and test performance (not overfitting)        
        """

        comparison_data = []

        for name, results in self.results.items():
            train_metrics = results.get('training', {})
            test_metrics = results.get('test', {})

            comparison_data.append({
                'Model': name,
                'Train_R2': train_metrics.get('r2', FALLBACK_DEFAULT),
                'Test_R2': test_metrics.get('r2', FALLBACK_DEFAULT),
                'Train_MAE': train_metrics.get('mae', FALLBACK_DEFAULT),
                'Test_MAE': test_metrics.get('mae', FALLBACK_DEFAULT),
                'Train_RMSE': train_metrics.get('rmse', FALLBACK_DEFAULT),
                'Test_RMSE': test_metrics.get('rmse', FALLBACK_DEFAULT),
                'Overfitting': train_metrics.get('r2', FALLBACK_DEFAULT) - test_metrics.get('r2', FALLBACK_DEFAULT)
            })

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Test_R2', ascending=False)

        print("\n=== MODEL COMPARISON RESULTS ===")
        print(comparison_df.round(4))
        
        best_model = comparison_df.iloc[0]['Model']
        print(f"\nBest performing model: {best_model}")
        print(f"Test R² Score: {comparison_df.iloc[0]['Test_R2']:.4f}")
        
        return comparison_df
    
    def plot_predictions_comparison(self, X_test: pd.DataFrame, y_test: pd.Series):
        """
        Plot actual vs predicted values for all models.
        """
        n_models = len(self.models)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))

        if n_models == 1:
            axes = [axes]
        
        for idx, (name, model) in enumerate(self.models.items()):
            y_pred = model.predict(X_test)
            
            axes[idx].scatter(y_test, y_pred, alpha=0.6)
            axes[idx].plot([0, 1], [0, 1], 'r--', lw=2)  
            axes[idx].set_xlabel('Actual Investment Ratio')
            axes[idx].set_ylabel('Predicted Investment Ratio')
            axes[idx].set_title(f'{name}\nR² = {self.results[name]["test"]["r2"]:.3f}')
            axes[idx].grid(True)
        
        plt.tight_layout()
        plt.show()

