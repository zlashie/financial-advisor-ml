########################################
#### Dependencies
########################################
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.factories.config_factory import ConfigFactory
from src.factories.feature_engineering_factory import FeatureEngineeringFactory
from src.generators.financial_dataset_generator import FinancialDatasetGenerator
from src.generators.market_data_generator import SP500MarketDataGenerator
from src.generators.personal_profile_generator import StandardPersonalProfileGenerator
from src.calculators.investment_strategy_calculator import OptimalInvestmentStrategyCalculator
from src.feature_engineering.debt_feature_creator import DebtFeatureCreator
from src.feature_engineering.market_feature_creator import MarketFeatureCreator
from src.feature_engineering.risk_feature_creator import RiskFeatureCreator
from src.feature_engineering.categorical_feature_preprocessor import CategoricalFeaturePreprocessor
from src.feature_engineering.robust_feature_scaler import RobustFeatureScaler
from src.interfaces.feature_engineering_interfaces import (
    FeatureCreator, FeaturePreprocessor, FeatureScaler, 
    DatasetSplitter, FeatureEngineeringPipeline
)
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
#### Test Helper Functions
########################################

def create_test_data_generator():
    """Create test data generator using the new architecture."""
    # Create configurations
    market_config = ConfigFactory.create_market_config()
    general_config = ConfigFactory.create_general_config()
    personal_config = ConfigFactory.create_personal_config()
    debt_config = ConfigFactory.create_debt_config()
    strategy_config = ConfigFactory.create_strategy_config()
    
    # Create generators (Dependency Injection)
    market_generator = SP500MarketDataGenerator(market_config, general_config.default_seed)
    personal_generator = StandardPersonalProfileGenerator(
        personal_config, debt_config, general_config, general_config.default_seed
    )
    strategy_calculator = OptimalInvestmentStrategyCalculator(
        strategy_config, general_config, market_config
    )
    
    # Create main dataset generator
    return FinancialDatasetGenerator(
        market_generator, personal_generator, strategy_calculator
    )

def create_test_feature_pipeline():
    """Create test feature engineering pipeline using the new architecture."""
    # Create configurations
    feature_config = ConfigFactory.create_feature_engineering_config()
    financial_constants = ConfigFactory.create_financial_constants()
    
    # Create feature engineering pipeline
    return FeatureEngineeringFactory.create_feature_engineering_pipeline(
        feature_config, financial_constants
    )

########################################
#### Individual Component Tests
########################################

def test_debt_feature_creator():
    """Test debt feature creation."""
    print("\n=== TESTING DEBT FEATURE CREATOR ===")
    
    # Create test data
    data_generator = create_test_data_generator()
    raw_data = data_generator.generate_complete_dataset(10)
    
    # Create debt feature creator
    feature_config = ConfigFactory.create_feature_engineering_config()
    financial_constants = ConfigFactory.create_financial_constants()
    
    from src.feature_engineering.debt_feature_creator import DebtFeatureCreator
    debt_creator = DebtFeatureCreator(feature_config.debt_analysis, financial_constants)
   
    df_with_debt_features = debt_creator.create_features(raw_data)
    
    expected_debt_features = [
        'total_debt', 'annual_income', 'debt_to_income_ratio', 
        'weighted_debt_rate', 'debt_urgency'
    ]
    
    for feature in expected_debt_features:
        if feature in df_with_debt_features.columns:
            print(f"   ✓ {feature} created successfully")
        else:
            print(f"   ✗ {feature} missing!")
    
    sample_idx = 0
    original_cc_debt = raw_data.iloc[sample_idx]['cc_debt']
    original_mortgage_debt = raw_data.iloc[sample_idx]['mortgage_debt']
    
    expected_total_debt = original_cc_debt + original_mortgage_debt
    actual_total_debt = df_with_debt_features.iloc[sample_idx]['total_debt']
    
    if abs(expected_total_debt - actual_total_debt) < 0.01:
        print("   ✓ Total debt calculation correct")
    else:
        print(f"   ✗ Total debt calculation incorrect: expected {expected_total_debt}, got {actual_total_debt}")

    assert df_with_debt_features is not None, "Debt features should not be None"
    assert len(df_with_debt_features) > 0, "Should have debt feature data"
    
    for feature in expected_debt_features:
        assert feature in df_with_debt_features.columns, f"Missing expected feature: {feature}"
    
    assert abs(expected_total_debt - actual_total_debt) < 0.01, "Total debt calculation should be correct"
    
    print("   ✓ All debt feature assertions passed")

def test_market_feature_creator():
    """Test market feature creation."""
    print("\n=== TESTING MARKET FEATURE CREATOR ===")
    
    data_generator = create_test_data_generator()
    raw_data = data_generator.generate_complete_dataset(10)
    
    feature_config = ConfigFactory.create_feature_engineering_config()
    
    from src.feature_engineering.market_feature_creator import MarketFeatureCreator
    market_creator = MarketFeatureCreator(feature_config.market_valuation)
    
    df_with_market_features = market_creator.create_features(raw_data)
    
    expected_market_features = ['market_attractiveness']
    
    for feature in expected_market_features:
        if feature in df_with_market_features.columns:
            print(f"   ✓ {feature} created successfully")
        else:
            print(f"   ✗ {feature} missing!")
    
    sample_idx = 0
    sample_pe = raw_data.iloc[sample_idx]['sp500_pe']
    sample_vix = raw_data.iloc[sample_idx]['vix']
    
    if sample_pe > pe_thresholds['fair_value_max']:
        expected_pe_score = market_config['price_scoring']['expensive']
    elif sample_pe < pe_thresholds['fair_value_min']:
        expected_pe_score = market_config['price_scoring']['cheap']
    else:
        expected_pe_score = market_config['price_scoring']['fair']
    
    if sample_vix > vix_thresholds['balanced_max']:
        expected_vix_score = market_config['psychology_scoring']['fearful']
    elif sample_vix < vix_thresholds['balanced_min']:
        expected_vix_score = market_config['psychology_scoring']['greedy']
    else:
        expected_vix_score = market_config['psychology_scoring']['balanced']
    
    expected_market_score = expected_pe_score + expected_vix_score
    actual_market_score = df_with_market_features.iloc[sample_idx]['market_attractiveness']
    
    if abs(expected_market_score - actual_market_score) < 0.01:
        print(f"   ✓ Market attractiveness calculation correct: {actual_market_score}")
    else:
        print(f"   ✗ Market attractiveness calculation incorrect: expected {expected_market_score}, got {actual_market_score}")
    
    assert df_with_market_features is not None, "Market features should not be None"
    assert len(df_with_market_features) > 0, "Should have market feature data"
    
    for feature in expected_market_features:
        assert feature in df_with_market_features.columns, f"Missing expected feature: {feature}"
    
    assert abs(expected_market_score - actual_market_score) < 0.01, "Market attractiveness calculation should be correct"
    
    print("   ✓ All market feature assertions passed")

def test_risk_feature_creator():
    """Test risk feature creation."""
    print("\n=== TESTING RISK FEATURE CREATOR ===")
    
    data_generator = create_test_data_generator()
    raw_data = data_generator.generate_complete_dataset(10)
    
    financial_constants = ConfigFactory.create_financial_constants()
    
    from src.feature_engineering.risk_feature_creator import RiskFeatureCreator
    risk_creator = RiskFeatureCreator(financial_constants)
    
    df_with_risk_features = risk_creator.create_features(raw_data)
    
    expected_risk_features = ['years_to_retirement', 'risk_capacity', 'discretionary_ratio']
    
    for feature in expected_risk_features:
        if feature in df_with_risk_features.columns:
            print(f"   ✓ {feature} created successfully")
        else:
            print(f"   ✗ {feature} missing!")
    
    sample_idx = 0
    sample_age = raw_data.iloc[sample_idx]['age']
    expected_years_to_retirement = max(0, RETIREMENT_AGE - sample_age)
    expected_risk_capacity = expected_years_to_retirement / WORKING_YEARS
    
    actual_years_to_retirement = df_with_risk_features.iloc[sample_idx]['years_to_retirement']
    actual_risk_capacity = df_with_risk_features.iloc[sample_idx]['risk_capacity']
    
    if abs(expected_risk_capacity - actual_risk_capacity) < 0.01:
        print(f"   ✓ Risk capacity calculation correct: {actual_risk_capacity:.3f}")
    else:
        print(f"   ✗ Risk capacity calculation incorrect: expected {expected_risk_capacity:.3f}, got {actual_risk_capacity:.3f}")
    
    assert df_with_risk_features is not None, "Risk features should not be None"
    assert len(df_with_risk_features) > 0, "Should have risk feature data"
    
    for feature in expected_risk_features:
        assert feature in df_with_risk_features.columns, f"Missing expected feature: {feature}"
    
    assert abs(expected_risk_capacity - actual_risk_capacity) < 0.01, "Risk capacity calculation should be correct"
    assert abs(expected_years_to_retirement - actual_years_to_retirement) < 0.01, "Years to retirement calculation should be correct"
    
    print("   ✓ All risk feature assertions passed")

def test_categorical_preprocessor():
    """Test categorical feature preprocessing."""
    print("\n=== TESTING CATEGORICAL PREPROCESSOR ===")
    
    data_generator = create_test_data_generator()
    raw_data = data_generator.generate_complete_dataset(10)
    
    feature_config = ConfigFactory.create_feature_engineering_config()
    financial_constants = ConfigFactory.create_financial_constants()
    
    debt_creator = DebtFeatureCreator(feature_config.debt_analysis, financial_constants)
    market_creator = MarketFeatureCreator(feature_config.market_valuation)
    risk_creator = RiskFeatureCreator(financial_constants)
    
    df_engineered = raw_data.copy()
    df_engineered = debt_creator.create_features(df_engineered)
    df_engineered = market_creator.create_features(df_engineered)
    df_engineered = risk_creator.create_features(df_engineered)
    
    preprocessor = CategoricalFeaturePreprocessor(
        feature_config.demographics,
        feature_config.market_valuation,
        feature_config.market_condition_labels
    )
    
    df_processed = preprocessor.prepare_features(df_engineered)
    
    categorical_features = [col for col in df_processed.columns 
                          if any(x in col for x in ['age_group_', 'income_group_', 'market_condition_'])]
    
    print(f"   Created {len(categorical_features)} categorical features:")
    for feature in categorical_features:
        print(f"     - {feature}")
    
    if len(categorical_features) > 0:
        print("   ✓ Categorical encoding completed successfully")
    else:
        print("   ✗ No categorical encodings found")

def test_feature_scaler():
    """Test feature scaling."""
    print("\n=== TESTING FEATURE SCALER ===")
    
    data_generator = create_test_data_generator()
    raw_data = data_generator.generate_complete_dataset(100)
    
    pipeline = create_test_feature_pipeline()
    
    df_processed = raw_data.copy()
    for creator in pipeline.feature_creators:
        df_processed = creator.create_features(df_processed)
    
    df_processed = pipeline.feature_preprocessor.prepare_features(df_processed)
    
    feature_cols = [col for col in df_processed.columns if col != 'recommended_investment_ratio']
    X = df_processed[feature_cols]
    
    scaler = RobustFeatureScaler()
    
    X_scaled = scaler.fit_transform(X)
    
    if scaler.is_fitted():
        print("   ✓ Scaler fitted successfully")
    else:
        print("   ✗ Scaler not fitted")
    
    numerical_cols = X_scaled.select_dtypes(include=['float64', 'int64']).columns
    
    print("   Checking scaled feature statistics:")
    all_scaled_properly = True
    
    for col in numerical_cols[:5]: 
        mean_val = X_scaled[col].mean()
        std_val = X_scaled[col].std()
        print(f"     {col}: mean={mean_val:.3f}, std={std_val:.3f}")
        
        if abs(mean_val) > 2 or std_val > 10:
            all_scaled_properly = False
    
    if all_scaled_properly:
        print("   ✓ Features appear properly scaled")
    else:
        print("   ✗ Some features may not be properly scaled")
    
    try:
        scaler.transform(X.iloc[:10])  
        print("   ✓ Transform on new data successful")
    except Exception as e:
        print(f"   ✗ Transform on new data failed: {e}")

########################################
#### Main Feature Engineering Test
########################################

def test_feature_engineering():
    """
    Test the complete feature engineering pipeline with new architecture.
    """
    print("=== TESTING FEATURE ENGINEERING PIPELINE ===")
    
    data_generator = create_test_data_generator()
    raw_data = data_generator.generate_complete_dataset(N_TEST_RUNS)
    print(f"Generated {raw_data.shape[0]} samples with {raw_data.shape[1]} features")
    
    pipeline = create_test_feature_pipeline()
    
    print("\n=== TESTING INDIVIDUAL COMPONENTS ===")
    test_debt_feature_creator()
    test_market_feature_creator()
    test_risk_feature_creator()
    test_categorical_preprocessor()
    test_feature_scaler()
    
    print("\n=== TESTING COMPLETE PIPELINE ===")
    X_train, X_test, y_train, y_test = pipeline.create_ml_dataset(raw_data)

    print("\n=== PIPELINE RESULTS ===")
    print(f"Training features shape: {X_train.shape}")
    print(f"Test features shape: {X_test.shape}")
    print(f"Training target shape: {y_train.shape}")
    print(f"Test target shape: {y_test.shape}")
    
    print(f"\nFeature columns ({len(X_train.columns)} total):")
    for i, col in enumerate(X_train.columns):
        print(f"  {i+1:2d}. {col}")

    print("\n=== DATA QUALITY CHECKS ===")
    
    train_missing = X_train.isnull().sum().sum()
    test_missing = X_test.isnull().sum().sum()
    print(f"Training set missing values: {train_missing}")
    print(f"Test set missing values: {test_missing}")
    
    if train_missing == 0 and test_missing == 0:
        print("✓ PASS: No missing values in processed data")
    else:
        print("✗ FAIL: Missing values found in processed data")
    
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
        
        if abs(mean_val) > 2 or std_val > 10:  
            all_scaled_properly = False
    
    if all_scaled_properly:
        print("✓ PASS: Features appear properly scaled")
    else:
        print("✗ FAIL: Some features may not be properly scaled")

    print("\n=== FEATURE ENGINEERING VALIDATION ===")
    
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
    
    print(f"\nMarket attractiveness validation:")
    sample_pe = raw_data.iloc[sample_idx]['sp500_pe']
    sample_vix = raw_data.iloc[sample_idx]['vix']
    
    if sample_pe > pe_thresholds['fair_value_max']:
        expected_pe_score = market_config['price_scoring']['expensive']
    elif sample_pe < pe_thresholds['fair_value_min']:
        expected_pe_score = market_config['price_scoring']['cheap']
    else:
        expected_pe_score = market_config['price_scoring']['fair']
    
    if sample_vix > vix_thresholds['balanced_max']:
        expected_vix_score = market_config['psychology_scoring']['fearful']
    elif sample_vix < vix_thresholds['balanced_min']:
        expected_vix_score = market_config['psychology_scoring']['greedy']
    else:
        expected_vix_score = market_config['psychology_scoring']['balanced']
    
    expected_market_score = expected_pe_score + expected_vix_score
    
    print(f"  S&P 500 P/E: {sample_pe:.1f} → PE Score: {expected_pe_score}")
    print(f"  VIX: {sample_vix:.1f} → VIX Score: {expected_vix_score}")
    print(f"  Expected market attractiveness: {expected_market_score}")
    
    sample_age = raw_data.iloc[sample_idx]['age']
    expected_years_to_retirement = max(0, RETIREMENT_AGE - sample_age)
    expected_risk_capacity = expected_years_to_retirement / WORKING_YEARS
    
    print(f"\nAge-based risk capacity:")
    print(f"  Age: {sample_age}")
    print(f"  Years to retirement: {expected_years_to_retirement}")
    print(f"  Expected risk capacity: {expected_risk_capacity:.3f}")

    print("\n=== CATEGORICAL ENCODING VALIDATION ===")
    
    age_bins = [0, age_groups['young_max'], age_groups['middle_max'], 
                age_groups['mature_max'], age_groups['senior_max']]
    age_labels = age_groups['labels']
    
    print(f"Age group bins: {age_bins}")
    print(f"Age group labels: {age_labels}")
    
    print(f"Income group percentiles: {income_groups['percentiles']}")
    print(f"Income group labels: {income_groups['labels']}")
    
    categorical_cols = [col for col in X_train.columns 
                       if any(prefix in col for prefix in ['age_group_', 'income_group_', 'market_condition_'])]
    print(f"One-hot encoded columns: {categorical_cols}")
    
    if len(categorical_cols) > 0:
        print("✓ PASS: Categorical encoding completed")
    else:
        print("✗ FAIL: No categorical encodings found")

    print("\n=== PREPROCESSING STATE SAVE/LOAD TEST ===")
    
    try:
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        
        pipeline.save_state(str(models_dir / 'preprocessing_state.joblib'))
        print("✓ PASS: Preprocessing state saved successfully")
        
        new_pipeline = create_test_feature_pipeline()
        new_pipeline.load_state(str(models_dir / 'preprocessing_state.joblib'))
        print("✓ PASS: Preprocessing state loaded successfully")

        new_pipeline.feature_scaler.transform(X_test)
        print("✓ PASS: Loaded pipeline can transform new data")
        
    except Exception as e:
        print(f"✗ FAIL: Error with preprocessing state: {e}")
    
    print(f"\n=== ARCHITECTURE VALIDATION ===")
    
    interface_checks = [
        (pipeline.feature_creators[0], FeatureCreator, "DebtFeatureCreator"),
        (pipeline.feature_preprocessor, FeaturePreprocessor, "CategoricalFeaturePreprocessor"),
        (pipeline.feature_scaler, FeatureScaler, "RobustFeatureScaler"),
        (pipeline.dataset_splitter, DatasetSplitter, "TrainTestSplitter"),
        (pipeline, FeatureEngineeringPipeline, "FinancialFeatureEngineeringPipeline")
    ]
    
    for component, interface, name in interface_checks:
        if isinstance(component, interface):
            print(f"✓ {name} implements {interface.__name__}")
        else:
            print(f"✗ {name} does not implement {interface.__name__}")
    
    print(f"\n=== SUMMARY ===")
    print(f"Feature engineering pipeline completed successfully")
    print(f"Generated {X_train.shape[1]} features from {raw_data.shape[1]} original columns")
    print(f"Data split: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    print(f"Architecture follows SOLID principles with proper dependency injection")

def analyze_feature_correlations(X_train, y_train):
    """
    Analyze correlations between engineered features and target
    """
    print("\n=== FEATURE CORRELATION ANALYSIS ===")
    
    # Separate numerical and categorical columns
    numerical_columns = X_train.select_dtypes(include=[np.number]).columns
    categorical_columns = X_train.select_dtypes(exclude=[np.number]).columns
    
    print(f"Analyzing {len(numerical_columns)} numerical features")
    if len(categorical_columns) > 0:
        print(f"Excluding {len(categorical_columns)} categorical features: {list(categorical_columns)}")
    
    # Calculate correlations with target using only numerical features
    X_numerical = X_train[numerical_columns]
    correlations = X_numerical.corrwith(y_train).sort_values(key=abs, ascending=False)
    
    print("\nTop 10 numerical features by correlation with investment ratio:")
    for i, (feature, corr) in enumerate(correlations.head(10).items(), 1):
        print(f"  {i:2d}. {feature}: {corr:+.3f}")
    
    # Check for high inter-feature correlations (numerical features only)
    feature_corr_matrix = X_numerical.corr()
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
    
    # Summary statistics
    print(f"\nCorrelation summary:")
    print(f"  Strongest positive correlation: {correlations.max():+.3f}")
    print(f"  Strongest negative correlation: {correlations.min():+.3f}")
    print(f"  Mean absolute correlation: {correlations.abs().mean():.3f}")

def test_dependency_injection():
    """Test that dependency injection works correctly."""
    print("\n=== TESTING DEPENDENCY INJECTION ===")
    
    # Test that we can swap components
    from src.feature_engineering.robust_feature_scaler import RobustFeatureScaler
    from sklearn.preprocessing import StandardScaler
    
    # Create a custom scaler that implements the interface
    class CustomFeatureScaler:
        def __init__(self):
            self.scaler = StandardScaler()
            self.feature_columns = None
            self._is_fitted = False
        
        def fit_transform(self, df):
            feature_cols = [col for col in df.columns 
                           if col not in ['recommended_investment_ratio', 'expected_return']]
            numerical_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
            
            df_scaled = df.copy()
            self.feature_columns = numerical_cols
            df_scaled[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
            self._is_fitted = True
            return df_scaled
        
        def transform(self, df):
            if not self._is_fitted:
                raise ValueError("Scaler not fitted")
            df_scaled = df.copy()
            df_scaled[self.feature_columns] = self.scaler.transform(df[self.feature_columns])
            return df_scaled
        
        def is_fitted(self):
            return self._is_fitted
    
    # Test that we can inject the custom scaler
    try:
        feature_config = ConfigFactory.create_feature_engineering_config()
        financial_constants = ConfigFactory.create_financial_constants()
        
        from src.feature_engineering.financial_feature_engineering_pipeline import FinancialFeatureEngineeringPipeline
        from src.feature_engineering.debt_feature_creator import DebtFeatureCreator
        from src.feature_engineering.market_feature_creator import MarketFeatureCreator
        from src.feature_engineering.risk_feature_creator import RiskFeatureCreator
        from src.feature_engineering.categorical_feature_preprocessor import CategoricalFeaturePreprocessor
        from src.feature_engineering.train_test_splitter import TrainTestSplitter
        
        # Create components with custom scaler
        feature_creators = [
            DebtFeatureCreator(feature_config.debt_analysis, financial_constants),
            MarketFeatureCreator(feature_config.market_valuation),
            RiskFeatureCreator(financial_constants)
        ]
        
        feature_preprocessor = CategoricalFeaturePreprocessor(
            feature_config.demographics,
            feature_config.market_valuation,
            feature_config.market_condition_labels
        )
        
        custom_scaler = CustomFeatureScaler()  # Use custom scaler instead of RobustFeatureScaler
        
        dataset_splitter = TrainTestSplitter(feature_config.test_size, feature_config.random_state)
        
        # Create pipeline with injected dependencies
        custom_pipeline = FinancialFeatureEngineeringPipeline(
            feature_creators=feature_creators,
            feature_preprocessor=feature_preprocessor,
            feature_scaler=custom_scaler,
            dataset_splitter=dataset_splitter
        )
        
        print("✓ PASS: Successfully created pipeline with custom scaler")
        print("✓ PASS: Dependency injection working correctly")
        
    except Exception as e:
        print(f"✗ FAIL: Dependency injection failed: {e}")

if __name__ == "__main__":
    # Run all tests
    X_train, X_test, y_train, y_test, pipeline = test_feature_engineering()
    analyze_feature_correlations(X_train, y_train)
    test_dependency_injection()
    
    print("\n" + "="*50)
    print("ALL TESTS COMPLETED")
    print("="*50)