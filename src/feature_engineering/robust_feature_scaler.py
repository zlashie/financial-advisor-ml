########################################
#### Dependencies
########################################

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from ..interfaces.feature_engineering_interfaces import FeatureScaler

########################################
#### Classes
########################################

class RobustFeatureScaler(FeatureScaler):
    """Scales features using RobustScaler."""
    
    def __init__(self):
        self.scaler = None
        self.feature_columns = None
        self._is_fitted = False
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit scaler and transform features."""
        feature_cols = [col for col in df.columns 
                       if col not in ['recommended_investment_ratio', 'expected_return']]
        numerical_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        df_scaled = df.copy()
        
        self.scaler = RobustScaler()
        self.feature_columns = numerical_cols
        df_scaled[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        self._is_fitted = True
        
        return df_scaled
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted scaler."""
        if not self._is_fitted:
            raise ValueError("Scaler not fitted yet. Call fit_transform first.")
        
        df_scaled = df.copy()
        df_scaled[self.feature_columns] = self.scaler.transform(df[self.feature_columns])
        
        return df_scaled
    
    def is_fitted(self) -> bool:
        """Check if scaler is fitted."""
        return self._is_fitted
