########################################
#### Dependencies
########################################
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the actual modules and functions that exist
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

########################################
#### Data Generation Helper Class
########################################

class TestDataGenerator:
    """Helper class to generate data for testing using the existing architecture"""
    
    def __init__(self):
        # Create configurations using existing factory
        self.market_config = ConfigFactory.create_market_config()
        self.general_config = ConfigFactory.create_general_config()
        self.personal_config = ConfigFactory.create_personal_config()
        self.debt_config = ConfigFactory.create_debt_config()
        self.strategy_config = ConfigFactory.create_strategy_config()
        
        # Create generators (Dependency Injection)
        self.market_generator = SP500MarketDataGenerator(
            self.market_config, self.general_config.default_seed
        )
        self.personal_generator = StandardPersonalProfileGenerator(
            self.personal_config, self.debt_config, self.general_config, self.general_config.default_seed
        )
        self.strategy_calculator = OptimalInvestmentStrategyCalculator(
            self.strategy_config, self.general_config, self.market_config
        )
        
        # Create main dataset generator
        self.dataset_generator = FinancialDatasetGenerator(
            self.market_generator, self.personal_generator, self.strategy_calculator
        )
    
    def generate_complete_dataset(self, num_samples: int):
        """Generate complete dataset"""
        return self.dataset_generator.generate_complete_dataset(num_samples)

########################################
#### Feature Engineering Helper Class
########################################

class TestFeatureEngineer:
    """Helper class to handle feature engineering for testing"""
    
    def __init__(self):
        # Create configurations
        self.feature_config = ConfigFactory.create_feature_engineering_config()
        self.financial_constants = ConfigFactory.create_financial_constants()
        
        # Create feature engineering pipeline
        self.pipeline = FeatureEngineeringFactory.create_feature_engineering_pipeline(
            self.feature_config, self.financial_constants
        )
    
    def create_ml_dataset(self, raw_data):
        """Create ML dataset from raw data"""
        return self.pipeline.create_ml_dataset(raw_data)

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
    
    # Create models directory
    test_models_dir = Path('test_models')
    test_models_dir.mkdir(exist_ok=True)
    
    for model in models:
        model_filename = test_models_dir / f"test_{model.model_name.lower().replace(' ', '_')}.joblib"
        
        try:
            # Test saving
            model.save_model(str(model_filename))
            print(f"  ✓ {model.model_name} saved successfully")
            
            # Test loading - create new instance using factory
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
            
            # Verify loaded model works
            if hasattr(new_model, 'model') and new_model.model is not None:
                print(f"  ✓ {model.model_name} model object restored")
            else:
                print(f"  ✗ {model.model_name} model object not properly restored")
                
            # Clean up test file
            model_filename.unlink()
            
        except Exception as e:
            print(f"  ✗ {model.model_name} persistence failed: {e}")
    
    # Clean up test directory if empty
    try:
        test_models_dir.rmdir()
    except OSError:
        pass  # Directory not empty, leave it

def test_data_generation():
    """
    Test data generation functionality
    """
    print("\n=== DATA GENERATION TESTS ===")
    
    try:
        generator = TestDataGenerator()
        print("  ✓ Data generator creation successful")
        
        # Test small dataset generation
        test_size = min(500, DATASET_SIZE) 
        raw_data = generator.generate_complete_dataset(test_size)
        print(f"  ✓ Dataset generation successful ({raw_data.shape[0]} samples)")
        
        # Validate dataset structure - Updated to match actual column name
        required_columns = ['recommended_investment_ratio']  
        for col in required_columns:
            if col in raw_data.columns:
                print(f"  ✓ Required column '{col}' present")
            else:
                print(f"  ✗ Required column '{col}' missing")
        
        return raw_data
        
    except Exception as e:
        print(f"  ✗ Data generation failed: {e}")
        return None
    
def test_feature_engineering(raw_data):
    """
    Test feature engineering functionality
    """
    print("\n=== FEATURE ENGINEERING TESTS ===")
    
    if raw_data is None:
        print("  ✗ No raw data available for feature engineering")
        return None, None, None, None
    
    try:
        feature_engineer = TestFeatureEngineer()
        print("  ✓ Feature engineer creation successful")
        
        X_train, X_test, y_train, y_test = feature_engineer.create_ml_dataset(raw_data)
        print(f"  ✓ Feature engineering successful")
        print(f"    Training set: {X_train.shape}")
        print(f"    Test set: {X_test.shape}")
        print(f"    Features: {X_train.shape[1]}")
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        print(f"  ✗ Feature engineering failed: {e}")
        return None, None, None, None

def test_service_layer():
    """
    Test individual services work correctly
    """
    print("\n=== SERVICE LAYER TESTS ===")
    
    # Test configuration creation
    try:
        evaluation_config = ModelConfigFactory.create_evaluation_config()
        rf_config = ModelConfigFactory.create_random_forest_config()
        nn_config = ModelConfigFactory.create_neural_network_config()
        viz_config = ModelConfigFactory.create_visualization_config()
        print("  ✓ Configuration creation successful")
    except Exception as e:
        print(f"  ✗ Configuration creation failed: {e}")
        return False
    
    # Test service creation
    try:
        metrics_calculator = FinancialMetricsCalculator(evaluation_config)
        validation_service = ModelValidationService(evaluation_config)
        comparison_service = ModelComparisonService(evaluation_config)
        visualization_service = ModelVisualizationService(viz_config)
        print("  ✓ Service creation successful")
    except Exception as e:
        print(f"  ✗ Service creation failed: {e}")
        return False
    
    # Test model factory
    try:
        lr_model = ModelFactory.create_linear_regression(metrics_calculator, validation_service)
        rf_model = ModelFactory.create_random_forest(rf_config, metrics_calculator, validation_service)
        nn_model = ModelFactory.create_neural_network(nn_config, metrics_calculator, validation_service)
        print("  ✓ Model factory successful")
    except Exception as e:
        print(f"  ✗ Model factory failed: {e}")
        return False
    
    return True

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
    
def test_configuration_validation():
    """Test that all configurations load correctly"""
    print("\n=== CONFIGURATION VALIDATION ===")
    
    try:
        # Test all config creation
        evaluation_config = ModelConfigFactory.create_evaluation_config()
        rf_config = ModelConfigFactory.create_random_forest_config()
        nn_config = ModelConfigFactory.create_neural_network_config()
        viz_config = ModelConfigFactory.create_visualization_config()
        
        # Validate key parameters
        assert evaluation_config.invest_ratio_min >= 0
        assert evaluation_config.invest_ratio_max <= 1
        assert rf_config.n_estimators > 0
        assert nn_config.max_iterations > 0
        
        print("  ✓ All configurations valid")
        return True
        
    except Exception as e:
        print(f"  ✗ Configuration validation failed: {e}")
        return False

def test_all_models():
    """
    Complete model training and comparison pipeline using new architecture.
    """
    print("=== COMPLETE ML MODEL COMPARISON (NEW ARCHITECTURE) ===")
    print(f"Using dataset size: {DATASET_SIZE}")
    print(f"Random state: {RANDOM_STATE}")
    
    # Step 0: Test service layer
    if not test_service_layer():
        print("Service layer tests failed. Aborting.")
        return None, None, None, None, None
    
    # Step 0.5: Test orchestrator
    orchestrator, created_models = test_orchestrator()
    if orchestrator is None:
        print("Orchestrator tests failed. Aborting.")
        return None, None, None, None, None
    
    # Step 1: Generate and prepare data
    raw_data = test_data_generation()
    if raw_data is None:
        print("Data generation failed. Aborting.")
        return None, None, None, None, None
    
    X_train, X_test, y_train, y_test = test_feature_engineering(raw_data)
    if X_train is None:
        print("Feature engineering failed. Aborting.")
        return None, None, None, None, None
    
    # Step 2: Use orchestrator to train and evaluate models
    print("\nTraining and evaluating models using orchestrator...")
    try:
        feature_names = X_train.columns.tolist()
        results_df, models = orchestrator.train_and_evaluate_models(
            X_train, y_train, X_test, y_test, feature_names
        )
        print("  ✓ Orchestrator training and evaluation successful")
    except Exception as e:
        print(f"  ✗ Orchestrator training failed: {e}")
        return None, None, None, None, None
    
    # Step 3: Validate performance
    validation_results = validate_model_performance(results_df)
    
    # Step 4: Analyze predictions
    analyze_prediction_distribution(models, X_test, y_test)
    
    # Step 5: Test interpretability
    test_model_interpretability(models, feature_names)
    
    # Step 6: Test model persistence
    test_model_persistence(models)
    
    # Step 7: Analyze feature importance (for interpretable models)
    print(f"\n=== FEATURE IMPORTANCE ANALYSIS ===")
    
    # Linear Regression coefficients
    lr_model = next((model for model in models if isinstance(model, LinearRegressionModel)), None)
    if lr_model:
        try:
            lr_importance = lr_model.get_feature_importance(feature_names)
            print(f"\nLinear Regression - Top {N_IMPORTANT} Most Important Features:")
            print(lr_importance.head(N_IMPORTANT))
        except Exception as e:
            print(f"Linear regression feature importance failed: {e}")
    
    # Random Forest importance
    rf_model = next((model for model in models if isinstance(model, RandomForestModel)), None)
    if rf_model:
        try:
            rf_importance = rf_model.get_feature_importance(feature_names)
            print(f"\nRandom Forest - Top {N_IMPORTANT} Most Important Features:")
            print(rf_importance.head(N_IMPORTANT))
            
            # Plot feature importance using visualization service
            try:
                print("\nGenerating feature importance plot...")
                orchestrator.visualization_service.plot_feature_importance(rf_model, feature_names)
            except Exception as e:
                print(f"Feature importance plot skipped: {e}")
                
        except Exception as e:
            print(f"Random forest feature importance failed: {e}")
    
    # Step 8: Neural Network training history
    nn_model = next((model for model in models if isinstance(model, NeuralNetworkModel)), None)
    if nn_model:
        try:
            print("\nGenerating neural network training history...")
            orchestrator.visualization_service.plot_training_history(nn_model)
        except Exception as e:
            print(f"Training history plot skipped: {e}")
    
    # Step 9: Save best model using orchestrator
    try:
        best_model = orchestrator.save_best_model(models, results_df)
        print(f"\nBest model saved: {best_model.model_name}")
    except Exception as e:
        print(f"Best model saving failed: {e}")
    
    # Step 10: Summary
    print(f"\n=== TESTING SUMMARY ===")
    print(f"Dataset size: {DATASET_SIZE}")
    print(f"Features generated: {X_train.shape[1]}")
    print(f"Models tested: {len(models)}")
    
    if not results_df.empty:
        best_model_name = results_df.iloc[0]['Model']
        best_r2 = results_df.iloc[0]['Test_R2']
        print(f"Best model: {best_model_name} (R² = {best_r2:.3f})")
    
    # Count passing models
    passing_models = sum(1 for result in validation_results.values() if result['overall_pass'])
    print(f"Models passing validation: {passing_models}/{len(models)}")
    
    if passing_models == len(models):
        print("✓ ALL MODELS PASSED VALIDATION")
    else:
        print("✗ SOME MODELS FAILED VALIDATION - CHECK RESULTS ABOVE")
    
    return models, results_df, X_test, y_test, validation_results

if __name__ == "__main__":
    models, results, X_test, y_test, validation = test_all_models()