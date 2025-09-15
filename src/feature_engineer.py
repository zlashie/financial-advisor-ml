########################################
#### Dependencies
########################################

from pathlib import Path
from .factories.config_factory import ConfigFactory
from .factories.feature_engineering_factory import FeatureEngineeringFactory
from .config import config
import pandas as pd

########################################
#### Main
########################################

def main():
    """Main function for feature engineering."""
    
    # Create configurations
    feature_config = ConfigFactory.create_feature_engineering_config()
    financial_constants = ConfigFactory.create_financial_constants()
    
    # Create feature engineering pipeline
    pipeline = FeatureEngineeringFactory.create_feature_engineering_pipeline(
        feature_config, financial_constants
    )
    
    # Load raw data
    data_dir = Path('data')
    training_data_name = config.get('data_generation', 'output', 'training_data_name')
    input_file = data_dir / f'{training_data_name}.csv'
    
    if not input_file.exists():
        raise FileNotFoundError(f"Training data not found: {input_file}")
    
    raw_data = pd.read_csv(input_file)
    print(f"Loaded raw data: {raw_data.shape}")
    
    # Run feature engineering pipeline
    X_train, X_test, y_train, y_test = pipeline.create_ml_dataset(raw_data)
    
    # Save processed data
    processed_dir = data_dir / 'processed'
    processed_dir.mkdir(exist_ok=True)
    
    X_train.to_csv(processed_dir / 'X_train.csv', index=False)
    X_test.to_csv(processed_dir / 'X_test.csv', index=False)
    y_train.to_csv(processed_dir / 'y_train.csv', index=False)
    y_test.to_csv(processed_dir / 'y_test.csv', index=False)
    
    # Save preprocessing state
    pipeline.save_state(str(processed_dir / 'preprocessing_state.joblib'))
    
    print("Feature engineering completed successfully!")

if __name__ == "__main__":
    main()
