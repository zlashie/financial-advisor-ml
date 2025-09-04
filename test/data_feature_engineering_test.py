########################################
#### Dependendencies
########################################
import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_generator import FinancialDataGenerator
from src.feature_engineer import FinancialFeatureEngineer

########################################
#### Test Constants
########################################

SAMPLE_IDX = 0
MONTHS_PR_YEAR = 12
NO_INCOME = 0

########################################
#### Test Variables
########################################

N_TEST_RUNS = 1000
N_NUMERICAL_COL = 5

########################################
#### Test 
########################################

def test_feature_engineering():
    """
    Test the complete feature engineering pipeline.
    """
    print("=== TESTING FEATURE ENGINEERING PIPELINE ===")
    generator = FinancialDataGenerator()
    raw_data = generator.generate_complete_dataset(N_TEST_RUNS)

    feature_engineer = FinancialFeatureEngineer()

    X_train, X_test, y_train, y_test = feature_engineer.create_ml_dataset(raw_data)

    print("\n=== PIPELINE RESULTS ===")
    print(f"Training features shape: {X_train.shape}")
    print(f"Test features shape: {X_test.shape}")
    print(f"Training target shape: {y_train.shape}")
    print(f"Test target shape: {y_test.shape}")
    
    print(f"\nFeature columns: {list(X_train.columns)}")

    print("\n=== DATA QUALITY CHECKS ===")
    print(f"Training set missing values: {X_train.isnull().sum().sum()}")
    print(f"Test set missing values: {X_test.isnull().sum().sum()}")
    print(f"Target variable range: {y_train.min():.3f} to {y_train.max():.3f}")

    print("\n=== SCALING VERIFICATION ===")
    numerical_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
    for col in numerical_cols[:N_NUMERICAL_COL]:
        print(f"{col}: mean={X_train[col].mean():.3f}, std={X_train[col].std():.3f}")

    print("\n=== FEATURE ENGINEERING VALIDATION ===")
    original_debt = raw_data.iloc[SAMPLE_IDX]['cc_debt'] + raw_data.iloc[SAMPLE_IDX]['mortgage_debt']
    original_income = raw_data.iloc[SAMPLE_IDX]['monthly_income'] * MONTHS_PR_YEAR
    
    if original_income > NO_INCOME:
        expected_ratio = original_debt / original_income
        print(f"Sample debt-to-income calculation check: Expected {expected_ratio:.3f}")
    
    feature_engineer.save_preprocessing_state('models/preprocessing_state.pkl')
    
    return X_train, X_test, y_train, y_test, feature_engineer

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, fe = test_feature_engineering()