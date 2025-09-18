########################################
#### Dependencies
########################################

import pandas as pd
import joblib
import os
from pathlib import Path
from typing import Tuple, List
from ..interfaces.feature_engineering_interfaces import (
    FeatureCreator, FeaturePreprocessor, FeatureScaler, 
    DatasetSplitter, FeatureEngineeringPipeline
)
from ..config import config

########################################
#### Classes
########################################

class FinancialFeatureEngineeringPipeline(FeatureEngineeringPipeline):
    """Main feature engineering pipeline following SOLID principles."""
    
    def __init__(self, 
                 feature_creators: List[FeatureCreator],
                 feature_preprocessor: FeaturePreprocessor,
                 feature_scaler: FeatureScaler,
                 dataset_splitter: DatasetSplitter,
                 target_column: str = None):
        self.feature_creators = feature_creators
        self.feature_preprocessor = feature_preprocessor
        self.feature_scaler = feature_scaler
        self.dataset_splitter = dataset_splitter
        
        self.pipeline_config = config.get_section('feature_engineering', 'pipeline')
        self.target_column = target_column or self.pipeline_config['target_column']
    
    def create_ml_dataset(self, raw_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Complete feature engineering pipeline."""
        print("Starting complete feature engineering pipeline...")
        
        df_engineered = raw_data.copy()
        for creator in self.feature_creators:
            df_engineered = creator.create_features(df_engineered)
            print(f"Applied {creator.__class__.__name__}")
        
        df_ml_ready = self.feature_preprocessor.prepare_features(df_engineered)
        print(f"After preprocessing: {df_ml_ready.shape[1]} features")
        
        feature_exclusions = self.pipeline_config['feature_column_exclusions']
        feature_cols = [col for col in df_ml_ready.columns if col not in feature_exclusions]
        X = df_ml_ready[feature_cols]
        y = df_ml_ready[self.target_column]
        
        print(f"Final feature set: {X.shape[1]} features")
        print(f"Target variable: {self.target_column}")
        
        X_train, X_test, y_train, y_test = self.dataset_splitter.split(X, y)
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_test_scaled = self.feature_scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def save_state(self, filepath: str = None) -> None:
        """Save preprocessing state."""
        if not self.feature_scaler.is_fitted():
            raise ValueError("Feature engineering pipeline not fitted yet")
        
        if filepath is None:
            processed_dir = config.get('paths', 'data', 'processed_directory')
            preprocessing_file = config.get('paths', 'data', 'preprocessing_state_file')
            filepath = str(Path(processed_dir) / preprocessing_file)
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        preprocessing_state = {
            'scaler': self.feature_scaler.scaler,
            'feature_columns': self.feature_scaler.feature_columns
        }
        
        joblib.dump(preprocessing_state, filepath)
        print(f"Preprocessing state saved to {filepath}")
    
    def load_state(self, filepath: str = None) -> None:
        """Load preprocessing state."""
        if filepath is None:
            processed_dir = config.get('paths', 'data', 'processed_directory')
            preprocessing_file = config.get('paths', 'data', 'preprocessing_state_file')
            filepath = str(Path(processed_dir) / preprocessing_file)
        
        preprocessing_state = joblib.load(filepath)
        self.feature_scaler.scaler = preprocessing_state['scaler']
        self.feature_scaler.feature_columns = preprocessing_state['feature_columns']
        self.feature_scaler._is_fitted = True
        print(f"Preprocessing state loaded from {filepath}")
