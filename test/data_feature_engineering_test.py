########################################
#### Dependencies
########################################
import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_generator import FinancialDataGenerator
from src.feature_engineer import FinancialFeatureEngineer
from src.config import config

########################################
#### Load Test Configuration
########################################

# Load constants from config
MONTHS_PR_YEAR = config.get('common', 'general', 'months_per_year')
NO_DEBT = config.get('common', 'financial_constants', 'no_debt')
NO_INCOME = 0  

# Test parameters
SAMPLE_IDX = 0
N_TEST_RUNS = config.get('data_generation', 'output', 'complete_runs')  
N_NUMERICAL_COL = 5  

# Load feature engineering thresholds for validation
market_config = config.get('feature_engineering', 'market_valuation')
pe_thresholds = market_config['pe_thresholds']
vix_thresholds = market_config['vix_thresholds']

debt_config = config.get('feature_engineering', 'debt_analysis')
urgency_thresholds = debt_config['urgency_thresholds']
urgency_scores = debt_config['urgency_scores']

demographics_config = config.get('feature_engineering', 'demographics')
age_groups = demographics_config['age_groups']
income_groups = demographics_config['income_groups']

# Load expected feature ranges
RETIREMENT_AGE = config.get('common', 'financial_constants', 'retirement_age')
WORKING_YEARS = config.get('common', 'financial_constants', 'working_years')

########################################
#### Test Feature Engineering
########################################

def test_feature_engineering():
    """
    Test the complete feature engineering pipeline.
    """
    print("=== TESTING FEATURE ENGINEERING PIPELINE ===")
    generator = FinancialDataGenerator()
    raw_data = generator.generate_complete_dataset(N_TEST_RUNS)

    feature_engineer = FinancialFeatureEngineer()

    # Test individual steps first
    print("\n=== TESTING INDIVIDUAL STEPS ===")
    
    # Step 1: Test engineered features creation
    print("1. Testing engineered features creation...")
    df_engineered = feature_engineer.create_engineer_features(raw_data)
    
    # Validate new features
    expected_new_features = [
        'total_debt', 'annual_income', 'debt_to_income_ratio', 
        'weighted_debt_rate', 'market_attractiveness', 
        'years_to_retirement', 'risk_capacity', 
        'discretionary_ratio', 'debt_urgency'
    ]
    
    for feature in expected_new_features:
        if feature in df_engineered.columns:
            print(f"   ✓ {feature} created successfully")
        else:
            print(f"   ✗ {feature} missing!")
    
    # Step 2: Test ML preparation
    print("\n2. Testing ML preparation...")
    df_ml_ready = feature_engineer.prepare_feature_for_ml(df_engineered)
    
    # Check for categorical encodings
    categorical_features = [col for col in df_ml_ready.columns if any(x in col for x in ['age_group_', 'income_group_', 'market_condition_'])]
    print(f"   Created {len(categorical_features)} categorical features: {categorical_features}")

    # Step 3: Test complete pipeline
    print("\n3. Testing complete pipeline...")
    X_train, X_test, y_train, y_test = feature_engineer.create_ml_dataset(raw_data)

    print("\n=== PIPELINE RESULTS ===")
    print(f"Training features shape: {X_train.shape}")
    print(f"Test features shape: {X_test.shape}")
    print(f"Training target shape: {y_train.shape}")
    print(f"Test target shape: {y_test.shape}")
    
    print(f"\nFeature columns ({len(X_train.columns)} total):")
    for i, col in enumerate(X_train.columns):
        print(f"  {i+1:2d}. {col}")

    print("\n=== DATA QUALITY CHECKS ===")
    
    # Check for missing values
    train_missing = X_train.isnull().sum().sum()
    test_missing = X_test.isnull().sum().sum()
    print(f"Training set missing values: {train_missing}")
    print(f"Test set missing values: {test_missing}")
    
    if train_missing == 0 and test_missing == 0:
        print("✓ PASS: No missing values in processed data")
    else:
        print("✗ FAIL: Missing values found in processed data")
    
    # Check target variable range
    target_min = y_train.min()
    target_max = y_train.max()
    expected_min = config.get('data_generation', 'investment_strategy', 'bounds', 'min_investment_ratio')
    expected_max = config.get('data_generation', 'investment_strategy', 'bounds', 'max_investment_ratio')
    
    print(f"Target variable range: {target_min:.3f} to {target_max:.3f}")
    print(f"Expected range: {expected_min} to {expected_max}")
    
    if target_min >= expected_min and target_max <= expected_max:
        print("✓ PASS: Target variable within expected bounds")
    else:
        print("✗ FAIL: Target variable outside expected bounds")

    print("\n=== SCALING VERIFICATION ===")
    numerical_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
    
    print("Checking if numerical features are properly scaled...")
    all_scaled_properly = True
    
    for col in numerical_cols[:N_NUMERICAL_COL]:
        mean_val = X_train[col].mean()
        std_val = X_train[col].std()
        print(f"  {col}: mean={mean_val:.3f}, std={std_val:.3f}")
        
        # RobustScaler should center around 0, but not necessarily std=1
        # Check if values are in reasonable scaled range
        if abs(mean_val) > 2 or std_val > 10:  
            all_scaled_properly = False
    
    if all_scaled_properly:
        print("✓ PASS: Features appear properly scaled")
    else:
        print("✗ FAIL: Some features may not be properly scaled")

    print("\n=== FEATURE ENGINEERING VALIDATION ===")
    
    # Test debt-to-income calculation
    sample_idx = SAMPLE_IDX
    original_cc_debt = raw_data.iloc[sample_idx]['cc_debt']
    original_mortgage_debt = raw_data.iloc[sample_idx]['mortgage_debt']
    original_monthly_income = raw_data.iloc[sample_idx]['monthly_income']
    
    original_total_debt = original_cc_debt + original_mortgage_debt
    original_annual_income = original_monthly_income * MONTHS_PR_YEAR
    
    if original_annual_income > NO_INCOME:
        expected_debt_ratio = original_total_debt / original_annual_income
        print(f"Sample debt-to-income calculation:")
        print(f"  Total debt: ${original_total_debt:,.0f}")
        print(f"  Annual income: ${original_annual_income:,.0f}")
        print(f"  Expected ratio: {expected_debt_ratio:.3f}")
    
    # Test market attractiveness scoring
    print(f"\nMarket attractiveness validation:")
    sample_pe = raw_data.iloc[sample_idx]['sp500_pe']
    sample_vix = raw_data.iloc[sample_idx]['vix']
    
    # Calculate expected PE score
    if sample_pe > pe_thresholds['fair_value_max']:
        expected_pe_score = market_config['price_scoring']['expensive']  # -1
    elif sample_pe < pe_thresholds['fair_value_min']:
        expected_pe_score = market_config['price_scoring']['cheap']  # 1
    else:
        expected_pe_score = market_config['price_scoring']['fair']  # 0
    
    # Calculate expected VIX score
    if sample_vix > vix_thresholds['balanced_max']:
        expected_vix_score = market_config['psychology_scoring']['fearful']  # -1
    elif sample_vix < vix_thresholds['balanced_min']:
        expected_vix_score = market_config['psychology_scoring']['greedy']  # 1
    else:
        expected_vix_score = market_config['psychology_scoring']['balanced']  # 0
    
    expected_market_score = expected_pe_score + expected_vix_score
    
    print(f"  S&P 500 P/E: {sample_pe:.1f} → PE Score: {expected_pe_score}")
    print(f"  VIX: {sample_vix:.1f} → VIX Score: {expected_vix_score}")
    print(f"  Expected market attractiveness: {expected_market_score}")
    
    # Test age-based risk capacity
    sample_age = raw_data.iloc[sample_idx]['age']
    expected_years_to_retirement = max(0, RETIREMENT_AGE - sample_age)
    expected_risk_capacity = expected_years_to_retirement / WORKING_YEARS
    
    print(f"\nAge-based risk capacity:")
    print(f"  Age: {sample_age}")
    print(f"  Years to retirement: {expected_years_to_retirement}")
    print(f"  Expected risk capacity: {expected_risk_capacity:.3f}")
    
    # Test debt urgency scoring
    if original_total_debt > NO_DEBT:
        weighted_rate = (original_cc_debt * raw_data.iloc[sample_idx]['cc_rate'] + 
                        original_mortgage_debt * raw_data.iloc[sample_idx]['mortgage_rate']) / original_total_debt
        
        if weighted_rate > urgency_thresholds['high_ratio']:
            expected_urgency = urgency_scores['high']
        elif weighted_rate > urgency_thresholds['medium_ratio']:
            expected_urgency = urgency_scores['medium']
        else:
            expected_urgency = urgency_scores['low']
        
        print(f"\nDebt urgency validation:")
        print(f"  Weighted debt rate: {weighted_rate:.3f}")
        print(f"  Expected urgency score: {expected_urgency}")

    print("\n=== CATEGORICAL ENCODING VALIDATION ===")
    
    # Check age group encoding
    age_bins = [0, age_groups['young_max'], age_groups['middle_max'], 
                age_groups['mature_max'], age_groups['senior_max']]
    age_labels = age_groups['labels']
    
    print(f"Age group bins: {age_bins}")
    print(f"Age group labels: {age_labels}")
    
    # Check income group encoding
    print(f"Income group percentiles: {income_groups['percentiles']}")
    print(f"Income group labels: {income_groups['labels']}")
    
    # Verify one-hot encoding worked
    categorical_cols = [col for col in X_train.columns if any(prefix in col for prefix in ['age_group_', 'income_group_', 'market_condition_'])]
    print(f"One-hot encoded columns: {categorical_cols}")
    
    if len(categorical_cols) > 0:
        print("✓ PASS: Categorical encoding completed")
    else:
        print("✗ FAIL: No categorical encodings found")

    print("\n=== PREPROCESSING STATE SAVE/LOAD TEST ===")
    
    # Test saving preprocessing state
    try:
        os.makedirs('models', exist_ok=True)
        feature_engineer.save_preprocessing_state('models/preprocessing_state.pkl')
        print("✓ PASS: Preprocessing state saved successfully")
        
        # Test loading
        new_fe = FinancialFeatureEngineer()
        new_fe.load_preprocessing_state('models/preprocessing_state.pkl')
        print("✓ PASS: Preprocessing state loaded successfully")
        
    except Exception as e:
        print(f"✗ FAIL: Error with preprocessing state: {e}")
    
    print(f"\n=== SUMMARY ===")
    print(f"Feature engineering pipeline completed successfully")
    print(f"Generated {X_train.shape[1]} features from {raw_data.shape[1]} original columns")
    print(f"Data split: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    
    return X_train, X_test, y_train, y_test, feature_engineer

def analyze_feature_correlations(X_train, y_train):
    """
    Analyze correlations between engineered features and target
    """
    print("\n=== FEATURE CORRELATION ANALYSIS ===")
    
    # Calculate correlations with target
    correlations = X_train.corrwith(y_train).sort_values(key=abs, ascending=False)
    
    print("Top 10 features by correlation with investment ratio:")
    for i, (feature, corr) in enumerate(correlations.head(10).items(), 1):
        print(f"  {i:2d}. {feature}: {corr:+.3f}")
    
    # Check for high inter-feature correlations
    feature_corr_matrix = X_train.corr()
    high_corr_pairs = []
    
    for i in range(len(feature_corr_matrix.columns)):
        for j in range(i+1, len(feature_corr_matrix.columns)):
            corr_val = feature_corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.8:  # High correlation threshold
                high_corr_pairs.append((
                    feature_corr_matrix.columns[i],
                    feature_corr_matrix.columns[j],
                    corr_val
                ))
    
    if high_corr_pairs:
        print(f"\nHigh inter-feature correlations (>0.8):")
        for feat1, feat2, corr in high_corr_pairs:
            print(f"  {feat1} ↔ {feat2}: {corr:+.3f}")
    else:
        print("\n✓ No problematic high inter-feature correlations found")

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, fe = test_feature_engineering()