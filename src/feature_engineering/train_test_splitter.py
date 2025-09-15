########################################
#### Dependencies
########################################

import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from ..interfaces.feature_engineering_interfaces import DatasetSplitter

########################################
#### Classes
########################################

class TrainTestSplitter(DatasetSplitter):
    """Splits dataset into train and test sets."""
    
    def __init__(self, test_size: float, random_state: int):
        self.test_size = test_size
        self.random_state = random_state
    
    def split(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets."""
        return train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state, 
            stratify=None
        )
