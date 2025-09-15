########################################
#### Dependencies
########################################

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

########################################
#### Interfaces
########################################

class IFinancialModel(ABC):
    """Interface for financial prediction models"""
    
    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, float]:
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass
    
    @abstractmethod
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        pass
    
    @abstractmethod
    def save_model(self, filepath: str) -> None:
        pass
    
    @abstractmethod
    def load_model(self, filepath: str) -> None:
        pass

class IModelComparator(ABC):
    """Interface for model comparison functionality"""
    
    @abstractmethod
    def add_model(self, model: IFinancialModel) -> None:
        pass
    
    @abstractmethod
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        pass
    
    @abstractmethod
    def evaluate_all_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        pass
    
    @abstractmethod
    def compare_results(self) -> pd.DataFrame:
        pass

class IMetricsCalculator(ABC):
    """Interface for metrics calculation"""
    
    @abstractmethod
    def calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, dataset_name: str) -> Dict[str, float]:
        pass

class IModelVisualizer(ABC):
    """Interface for model visualization"""
    
    @abstractmethod
    def plot_feature_importance(self, model: IFinancialModel, feature_names: list, top_n: int) -> None:
        pass
    
    @abstractmethod
    def plot_predictions_comparison(self, models: Dict[str, IFinancialModel], X_test: pd.DataFrame, y_test: pd.Series) -> None:
        pass
