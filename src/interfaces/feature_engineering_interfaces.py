########################################
#### Dependencies
########################################

from abc import ABC, abstractmethod
from typing import Tuple, Dict, List, Any
import pandas as pd

########################################
#### Interfaces
########################################

class FeatureCreator(ABC):
    """Interface for creating engineered features from raw data."""
    
    @abstractmethod
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features from existing data."""
        pass

class FeaturePreprocessor(ABC):
    """Interface for preprocessing features for ML."""
    
    @abstractmethod
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for machine learning."""
        pass

class FeatureScaler(ABC):
    """Interface for scaling features."""
    
    @abstractmethod
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit scaler and transform features."""
        pass
    
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted scaler."""
        pass
    
    @abstractmethod
    def is_fitted(self) -> bool:
        """Check if scaler is fitted."""
        pass

class DatasetSplitter(ABC):
    """Interface for splitting datasets."""
    
    @abstractmethod
    def split(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets."""
        pass

class FeatureEngineeringPipeline(ABC):
    """Interface for complete feature engineering pipeline."""
    
    @abstractmethod
    def create_ml_dataset(self, raw_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Complete feature engineering pipeline."""
        pass
    
    @abstractmethod
    def save_state(self, filepath: str) -> None:
        """Save preprocessing state."""
        pass
    
    @abstractmethod
    def load_state(self, filepath: str) -> None:
        """Load preprocessing state."""
        pass
