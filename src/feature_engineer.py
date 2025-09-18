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
    
    #### Create configurations ####
    feature_config = ConfigFactory.create_feature_engineering_config()
    financial_constants = ConfigFactory.create_financial_constants()
    
    #### Create feature engineering pipeline ####
    pipeline = FeatureEngineeringFactory.create_feature_engineering_pipeline(
        feature_config, financial_constants
    )
    
    #### Load raw data using config paths ####
    data_dir = Path(config.get('paths', 'data', 'base_directory'))
    training_data_name = config.get('data_generation', 'output', 'training_data_name')
    csv_extension = config.get('paths', 'models', 'file_extensions', 'csv')
    input_file = data_dir / f'{training_data_name}{csv_extension}'
    
    if not input_file.exists():
        raise FileNotFoundError(f"Training data not found: {input_file}")
    
    raw_data = pd.read_csv(input_file)
    print(f"Loaded raw data: {raw_data.shape}")
    
    #### Run feature engineering pipeline ####
    X_train, X_test, y_train, y_test = pipeline.create_ml_dataset(raw_data)
    
    #### Save processed data using config paths ####
    processed_dir = Path(config.get('paths', 'data', 'processed_directory'))
    processed_dir.mkdir(exist_ok=True)
    
    #### Get training file names from config ####
    training_files = config.get_section('paths', 'data')['training_files']
    
    X_train.to_csv(processed_dir / training_files['X_train'], index=False)
    X_test.to_csv(processed_dir / training_files['X_test'], index=False)
    y_train.to_csv(processed_dir / training_files['y_train'], index=False)
    y_test.to_csv(processed_dir / training_files['y_test'], index=False)
    
    #### Save preprocessing state using config path ####
    preprocessing_file = config.get('paths', 'data', 'preprocessing_state_file')
    pipeline.save_state(str(processed_dir / preprocessing_file))
    
    print("Feature engineering completed successfully!")

if __name__ == "__main__":
    main()
