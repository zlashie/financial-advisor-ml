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
from src.generators.market_data_generator import SP500MarketDataGenerator
from src.generators.personal_profile_generator import StandardPersonalProfileGenerator
from src.calculators.investment_strategy_calculator import OptimalInvestmentStrategyCalculator
from src.generators.financial_dataset_generator import FinancialDatasetGenerator
from src.factories.feature_engineering_factory import FeatureEngineeringFactory
from src.model_training_orchestrator import ModelTrainingOrchestrator
from src.factories.model_config_factory import ModelConfigFactory
from src.factories.model_factory import ModelFactory
from src.services.metrics_service import FinancialMetricsCalculator
from src.services.model_validation_service import ModelValidationService
from src.services.model_comparison_service import ModelComparisonService
from src.services.visualization_service import ModelVisualizationService
from src.models.linear_regression_model import LinearRegressionModel
from src.models.random_forest_model import RandomForestModel
from src.models.neural_network_model import NeuralNetworkModel
from src.config import config

import pytest

########################################
#### Pytest
########################################
@pytest.fixture
def models():
    """Fixture to provide models for testing"""
    pytest.skip("Models fixture not implemented yet")

@pytest.fixture
def feature_names():
    """Fixture to provide feature names for testing"""
    return ['feature1', 'feature2', 'feature3']  

@pytest.fixture
def raw_data():
    """Fixture to provide raw data for testing"""
    import pandas as pd
    return pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': [4, 5, 6]
    })

########################################
#### Data Generation Helper Class
########################################

class TestDataGenerator:
    """Helper class to generate data for testing using the existing architecture"""
    
    @classmethod
    def create_generator(cls):
        """Create a configured data generator"""
        #### Create configurations using existing factory ####
        market_config = ConfigFactory.create_market_config()
        general_config = ConfigFactory.create_general_config()
        personal_config = ConfigFactory.create_personal_config()
        debt_config = ConfigFactory.create_debt_config()
        strategy_config = ConfigFactory.create_strategy_config()
        
        #### Create generators (Dependency Injection) ####
        market_generator = SP500MarketDataGenerator(
            market_config, general_config.default_seed
        )
        personal_generator = StandardPersonalProfileGenerator(
            personal_config, debt_config, general_config, general_config.default_seed
        )
        strategy_calculator = OptimalInvestmentStrategyCalculator(
            strategy_config, general_config, market_config
        )
        
        #### Create main dataset generator ####
        return FinancialDatasetGenerator(
            market_generator, personal_generator, strategy_calculator
        )
    
    @classmethod
    def generate_complete_dataset(cls, num_samples: int):
        """Generate complete dataset"""
        generator = cls.create_generator()
        return generator.generate_complete_dataset(num_samples)

########################################
#### Feature Engineering Helper Class
########################################

class TestFeatureEngineer:
    """Helper class to handle feature engineering for testing"""
    
    @classmethod
    def create_pipeline(cls):
        """Create a configured feature engineering pipeline"""
        # Create configurations
        feature_config = ConfigFactory.create_feature_engineering_config()
        financial_constants = ConfigFactory.create_financial_constants()
        
        # Create feature engineering pipeline
        return FeatureEngineeringFactory.create_feature_engineering_pipeline(
            feature_config, financial_constants
        )
    
    @classmethod
    def create_ml_dataset(cls, raw_data):
        """Create ML dataset from raw data"""
        pipeline = cls.create_pipeline()
        return pipeline.create_ml_dataset(raw_data)

########################################
#### Load Test Configuration
########################################

# Test dataset size - could use production size or smaller for faster testing
DATASET_SIZE = config.get('data_generation', 'output', 'complete_runs') 

# Model parameters from config
rf_config = config.get('models', 'random_forest')
N_ESTIMATORS = rf_config['n_estimators']
MAX_DEPTH = rf_config['max_depth']

nn_config = config.get('models', 'neural_network')
MAX_ITER = nn_config['max_iterations']
HIDDEN_LAYERS = tuple(nn_config['hidden_layer_sizes'])

# Visualization parameters
viz_config = config.get('models', 'evaluation', 'visualization')
N_IMPORTANT = viz_config['top_features']

# Performance thresholds for validation
MIN_R2_THRESHOLD = 0.3  
MAX_MAE_THRESHOLD = 0.3 
MAX_OVERFITTING_THRESHOLD = 0.25  

# Random state for reproducibility
RANDOM_STATE = config.get('common', 'general', 'random_state')

########################################
#### Test Functions
########################################

def validate_model_performance(results_df):
    """
    Validate that model performance meets minimum requirements with model-specific thresholds
    """
    print("\n=== MODEL PERFORMANCE VALIDATION ===")
    
    validation_results = {}
    
    # Model-specific thresholds
    model_thresholds = {
        'Linear Regression': {'overfitting_max': 0.15},
        'Random Forest': {'overfitting_max': 0.15},
        'Neural Network': {'overfitting_max': 0.30}  
    }
    
    for _, row in results_df.iterrows():
        model_name = row['Model']
        test_r2 = row['Test_R2']
        test_mae = row['Test_MAE']
        overfitting = row['Overfitting']
        
        overfitting_threshold = model_thresholds.get(model_name, {}).get('overfitting_max', MAX_OVERFITTING_THRESHOLD)
        
        # Performance checks
        r2_pass = test_r2 >= MIN_R2_THRESHOLD
        mae_pass = test_mae <= MAX_MAE_THRESHOLD
        overfitting_pass = overfitting <= overfitting_threshold
        
        all_pass = r2_pass and mae_pass and overfitting_pass
        
        print(f"\n{model_name}:")
        print(f"  R² Score: {test_r2:.3f} {'✓' if r2_pass else '✗'} (min: {MIN_R2_THRESHOLD})")
        print(f"  MAE: {test_mae:.3f} {'✓' if mae_pass else '✗'} (max: {MAX_MAE_THRESHOLD})")
        print(f"  Overfitting: {overfitting:.3f} {'✓' if overfitting_pass else '✗'} (max: {overfitting_threshold})")
        print(f"  Overall: {'✓ PASS' if all_pass else '✗ FAIL'}")
        
        validation_results[model_name] = {
            'r2_pass': r2_pass,
            'mae_pass': mae_pass,
            'overfitting_pass': overfitting_pass,
            'overall_pass': all_pass
        }
    
    return validation_results

def analyze_prediction_distribution(models, X_test, y_test):
    """
    Analyze the distribution of predictions from each model
    """
    print("\n=== PREDICTION DISTRIBUTION ANALYSIS ===")
    
    # Load expected bounds from config
    bounds_config = config.get('models', 'evaluation', 'investment_ratio_bounds')
    min_ratio = bounds_config['min']
    max_ratio = bounds_config['max']
    
    for model in models:
        predictions = model.predict(X_test)
        
        print(f"\n{model.model_name}:")
        print(f"  Prediction range: {predictions.min():.3f} to {predictions.max():.3f}")
        print(f"  Mean prediction: {predictions.mean():.3f}")
        print(f"  Std prediction: {predictions.std():.3f}")
        
        # Check bounds
        within_bounds = np.all((predictions >= min_ratio) & (predictions <= max_ratio))
        print(f"  Within bounds [{min_ratio}-{max_ratio}]: {'✓' if within_bounds else '✗'}")
        
        # Check for reasonable distribution
        unique_predictions = len(np.unique(np.round(predictions, 3)))
        print(f"  Unique predictions: {unique_predictions}")
        
        # Check correlation with actual values
        correlation = np.corrcoef(predictions, y_test)[0, 1]
        print(f"  Correlation with actual: {correlation:.3f}")

def test_model_interpretability(models, feature_names):
    """
    Test interpretability features of models
    """
    print("\n=== MODEL INTERPRETABILITY TESTS ===")
    
    # Test Linear Regression interpretability
    lr_model = next((model for model in models if isinstance(model, LinearRegressionModel)), None)
    
    if lr_model:
        print(f"\nLinear Regression Interpretability:")
        try:
            importance_df = lr_model.get_feature_importance(feature_names)
            print(f"  ✓ Feature importance extraction successful")
            print(f"  ✓ Top feature: {importance_df.iloc[0]['feature']} (coef: {importance_df.iloc[0]['coefficient']:.3f})")
            
            # Test prediction interpretation
            sample_data = pd.DataFrame([np.zeros(len(feature_names))], columns=feature_names)
            interpretation = lr_model.interpret_prediction(sample_data, feature_names)
            print(f"  ✓ Prediction interpretation successful")
            
        except Exception as e:
            print(f"  ✗ Linear regression interpretability failed: {e}")
    
    # Test Random Forest interpretability
    rf_model = next((model for model in models if isinstance(model, RandomForestModel)), None)
    
    if rf_model:
        print(f"\nRandom Forest Interpretability:")
        try:
            importance_df = rf_model.get_feature_importance(feature_names)
            print(f"  ✓ Feature importance extraction successful")
            print(f"  ✓ Top feature: {importance_df.iloc[0]['feature']} (importance: {importance_df.iloc[0]['importance']:.3f})")
            
        except Exception as e:
            print(f"  ✗ Random forest interpretability failed: {e}")
    
    # Test Neural Network training history
    nn_model = next((model for model in models if isinstance(model, NeuralNetworkModel)), None)
    
    if nn_model:
        print(f"\nNeural Network Training History:")
        try:
            history = nn_model.get_training_history()
            print(f"  ✓ Training history extraction successful")
            print(f"  ✓ Converged: {history['converged']}")
            print(f"  ✓ Iterations: {history['n_iterations']}")
            print(f"  ✓ Final loss: {history['loss_curve'][-1]:.4f}")
            
        except Exception as e:
            print(f"  ✗ Neural network history failed: {e}")

def test_model_persistence(models):
    """
    Test model saving and loading functionality
    """
    print("\n=== MODEL PERSISTENCE TESTS ===")
    
    test_models_dir = Path('test_models')
    test_models_dir.mkdir(exist_ok=True)
    
    for model in models:
        model_filename = test_models_dir / f"test_{model.model_name.lower().replace(' ', '_')}.joblib"
        
        try:
            model.save_model(str(model_filename))
            print(f"  ✓ {model.model_name} saved successfully")
            
            evaluation_config = ModelConfigFactory.create_evaluation_config()
            metrics_calculator = FinancialMetricsCalculator(evaluation_config)
            validation_service = ModelValidationService(evaluation_config)
            
            if isinstance(model, LinearRegressionModel):
                new_model = ModelFactory.create_linear_regression(metrics_calculator, validation_service)
            elif isinstance(model, RandomForestModel):
                rf_config = ModelConfigFactory.create_random_forest_config()
                new_model = ModelFactory.create_random_forest(rf_config, metrics_calculator, validation_service)
            elif isinstance(model, NeuralNetworkModel):
                nn_config = ModelConfigFactory.create_neural_network_config()
                new_model = ModelFactory.create_neural_network(nn_config, metrics_calculator, validation_service)
            else:
                print(f"  ✗ Unknown model type: {type(model)}")
                continue
            
            new_model.load_model(str(model_filename))
            print(f"  ✓ {model.model_name} loaded successfully")
            
            if hasattr(new_model, 'model') and new_model.model is not None:
                print(f"  ✓ {model.model_name} model object restored")
            else:
                print(f"  ✗ {model.model_name} model object not properly restored")
                
            model_filename.unlink()
            
        except Exception as e:
            print(f"  ✗ {model.model_name} persistence failed: {e}")
    
    try:
        test_models_dir.rmdir()
    except OSError:
        pass  

def test_data_generation():
    """
    Test data generation functionality
    """
    print("\n=== DATA GENERATION TESTS ===")
    
    try:
        generator = TestDataGenerator()
        print("  ✓ Data generator creation successful")
        
        test_size = min(500, DATASET_SIZE) 
        raw_data = generator.generate_complete_dataset(test_size)
        print(f"  ✓ Dataset generation successful ({raw_data.shape[0]} samples)")
        
        required_columns = ['recommended_investment_ratio']  
        for col in required_columns:
            if col in raw_data.columns:
                print(f"  ✓ Required column '{col}' present")
            else:
                print(f"  ✗ Required column '{col}' missing")
        
        assert raw_data is not None, "Raw data should not be None"
        assert len(raw_data) > 0, "Should have generated data"
        assert 'recommended_investment_ratio' in raw_data.columns, "Should have target column"
        
    except Exception as e:
        print(f"  ✗ Data generation failed: {e}")
        raise 
    
def test_feature_engineering():
    """
    Test feature engineering functionality
    """
    print("\n=== FEATURE ENGINEERING TESTS ===")
    
    try:
        generator = TestDataGenerator()
        raw_data = generator.generate_complete_dataset(50) 
        print(f"  ✓ Generated test data ({raw_data.shape[0]} samples)")
    except Exception as e:
        print(f"  ✗ Failed to generate test data: {e}")
        raise
    
    try:
        feature_engineer = TestFeatureEngineer()
        print("  ✓ Feature engineer creation successful")
        
        X_train, X_test, y_train, y_test = feature_engineer.create_ml_dataset(raw_data)
        print(f"  ✓ Feature engineering successful")
        print(f"    Training set: {X_train.shape}")
        print(f"    Test set: {X_test.shape}")
        print(f"    Features: {X_train.shape[1]}")
        
        # Add proper assertions
        assert X_train is not None, "X_train should not be None"
        assert X_test is not None, "X_test should not be None"
        assert y_train is not None, "y_train should not be None"
        assert y_test is not None, "y_test should not be None"
        assert len(X_train) > 0, "Training set should have data"
        assert len(X_test) > 0, "Test set should have data"
        assert X_train.shape[1] > 0, "Should have features"
        
    except Exception as e:
        print(f"  ✗ Feature engineering failed: {e}")
        raise 

def test_service_layer():
    """
    Test individual services work correctly
    """
    print("\n=== SERVICE LAYER TESTS ===")
    
    try:
        evaluation_config = ModelConfigFactory.create_evaluation_config()
        rf_config = ModelConfigFactory.create_random_forest_config()
        nn_config = ModelConfigFactory.create_neural_network_config()
        viz_config = ModelConfigFactory.create_visualization_config()
        print("  ✓ Configuration creation successful")
    except Exception as e:
        print(f"  ✗ Configuration creation failed: {e}")
        raise
    
    try:
        metrics_calculator = FinancialMetricsCalculator(evaluation_config)
        validation_service = ModelValidationService(evaluation_config)
        ModelComparisonService(evaluation_config) 
        ModelVisualizationService(viz_config)    
        print("  ✓ Service creation successful")
    except Exception as e:
        print(f"  ✗ Service creation failed: {e}")
        raise
    
    try:
        lr_model = ModelFactory.create_linear_regression(metrics_calculator, validation_service)
        rf_model = ModelFactory.create_random_forest(rf_config, metrics_calculator, validation_service)
        nn_model = ModelFactory.create_neural_network(nn_config, metrics_calculator, validation_service)
        print("  ✓ Model factory successful")
    except Exception as e:
        print(f"  ✗ Model factory failed: {e}")
        raise
    
    # Add assertions
    assert evaluation_config is not None, "Evaluation config should not be None"
    assert metrics_calculator is not None, "Metrics calculator should not be None"
    assert lr_model is not None, "Linear regression model should not be None"
    assert rf_model is not None, "Random forest model should not be None"
    assert nn_model is not None, "Neural network model should not be None"

def test_orchestrator():
    """
    Test the main orchestrator functionality
    """
    print("\n=== ORCHESTRATOR TESTS ===")
    
    try:
        orchestrator = ModelTrainingOrchestrator()
        print("  ✓ Orchestrator creation successful")
        
        models = orchestrator.create_models()
        print(f"  ✓ Model creation successful ({len(models)} models)")
        
        return orchestrator, models
    except Exception as e:
        print(f"  ✗ Orchestrator test failed: {e}")
        return None, None
    
def test_orchestrator():
    """
    Test the main orchestrator functionality
    """
    print("\n=== ORCHESTRATOR TESTS ===")
    
    try:
        orchestrator = ModelTrainingOrchestrator()
        print("  ✓ Orchestrator creation successful")
        
        models = orchestrator.create_models()
        print(f"  ✓ Model creation successful ({len(models)} models)")
        
        assert orchestrator is not None, "Orchestrator should not be None"
        assert models is not None, "Models should not be None"
        assert len(models) > 0, "Should have created models"
        
    except Exception as e:
        print(f"  ✗ Orchestrator test failed: {e}")
        raise  

########################################
#### All models test
########################################

def test_all_models():
    """
    Complete model training and comparison pipeline using new architecture.
    """
    print("=== COMPLETE ML MODEL COMPARISON (NEW ARCHITECTURE) ===")
    print(f"Using dataset size: {DATASET_SIZE}")
    print(f"Random state: {RANDOM_STATE}")
    
    #### Step 0: Test service layer ####
    try:
        test_service_layer()
    except Exception as e:
        print(f"Service layer tests failed: {e}")
        pytest.fail("Service layer tests failed")
    
    #### Step 0.5: Test orchestrator ####
    try:
        test_orchestrator()
        orchestrator = ModelTrainingOrchestrator()
    except Exception as e:
        print(f"Orchestrator tests failed: {e}")
        pytest.fail("Orchestrator tests failed")
    
    #### Step 1: Generate and prepare data ####
    try:
        test_data_generation()
        raw_data = TestDataGenerator.generate_complete_dataset(DATASET_SIZE)
    except Exception as e:
        print(f"Data generation failed: {e}")
        pytest.fail("Data generation failed")
    
    try:
        test_feature_engineering()
        X_train, X_test, y_train, y_test = TestFeatureEngineer.create_ml_dataset(raw_data)

    except Exception as e:
        print(f"Feature engineering failed: {e}")
        pytest.fail("Feature engineering failed")
    
    #### Step 2: Use orchestrator to train and evaluate models ####
    print("\nTraining and evaluating models using orchestrator...")
    try:
        feature_names = X_train.columns.tolist()
        results_df, models = orchestrator.train_and_evaluate_models(
            X_train, y_train, X_test, y_test, feature_names, show_plots=False
        )
        print("  ✓ Orchestrator training and evaluation successful")
    except Exception as e:
        print(f"  ✗ Orchestrator training failed: {e}")
        pytest.fail("Orchestrator training failed")
    
    #### Step 3: Validate performance ####
    validation_results = validate_model_performance(results_df)
    
    #### Step 4: Analyze predictions ####
    analyze_prediction_distribution(models, X_test, y_test)
    
    #### Step 5: Test interpretability ####
    test_model_interpretability(models, feature_names)
    
    #### Step 6: Test model persistence ####
    test_model_persistence(models)
    
    #### Step 7: Analyze feature importance (for interpretable models) ####
    print(f"\n=== FEATURE IMPORTANCE ANALYSIS ===")
    
    #### Linear Regression coefficients ####
    lr_model = next((model for model in models if isinstance(model, LinearRegressionModel)), None)
    if lr_model:
        try:
            lr_importance = lr_model.get_feature_importance(feature_names)
            print(f"\nLinear Regression - Top {N_IMPORTANT} Most Important Features:")
            print(lr_importance.head(N_IMPORTANT))
        except Exception as e:
            print(f"Linear regression feature importance failed: {e}")
    
    #### Random Forest importance ####
    rf_model = next((model for model in models if isinstance(model, RandomForestModel)), None)
    if rf_model:
        try:
            rf_importance = rf_model.get_feature_importance(feature_names)
            print(f"\nRandom Forest - Top {N_IMPORTANT} Most Important Features:")
            print(rf_importance.head(N_IMPORTANT))
        except Exception as e:
            print(f"Random forest feature importance failed: {e}")
    
    #### Step 8: Save best model using orchestrator ####
    try:
        best_model = orchestrator.save_best_model(models, results_df)
        print(f"\nBest model saved: {best_model.model_name}")
    except Exception as e:
        print(f"Best model saving failed: {e}")
    
    #### Step 9: Summary ####
    print(f"\n=== TESTING SUMMARY ===")
    print(f"Dataset size: {DATASET_SIZE}")
    print(f"Features generated: {X_train.shape[1]}")
    print(f"Models tested: {len(models)}")
    
    if not results_df.empty:
        best_model_name = results_df.iloc[0]['Model']
        best_r2 = results_df.iloc[0]['Test_R2']
        print(f"Best model: {best_model_name} (R² = {best_r2:.3f})")
    
    #### Count passing models ####
    passing_models = sum(1 for result in validation_results.values() if result['overall_pass'])
    print(f"Models passing validation: {passing_models}/{len(models)}")
    
    if passing_models == len(models):
        print("✓ ALL MODELS PASSED VALIDATION")
    else:
        print("✗ SOME MODELS FAILED VALIDATION - CHECK RESULTS ABOVE")

if __name__ == "__main__":
    models, results, X_test, y_test, validation = test_all_models()