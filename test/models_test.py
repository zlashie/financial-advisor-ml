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
from src.models import LinearRegressionModel, RandomForestModel, NeuralNetworkModel, ModelComparison
from src.config import config

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
MIN_R2_THRESHOLD = 0.5  # Minimum acceptable R² score
MAX_MAE_THRESHOLD = 0.2  # Maximum acceptable MAE
MAX_OVERFITTING_THRESHOLD = 0.1  # Maximum acceptable overfitting gap

# Random state for reproducibility
RANDOM_STATE = config.get('common', 'general', 'random_state')

########################################
#### Test Functions
########################################

def validate_model_performance(results_df):
    """
    Validate that model performance meets minimum requirements
    """
    print("\n=== MODEL PERFORMANCE VALIDATION ===")
    
    validation_results = {}
    
    for _, row in results_df.iterrows():
        model_name = row['Model']
        test_r2 = row['Test_R2']
        test_mae = row['Test_MAE']
        overfitting = row['Overfitting']
        
        # Performance checks
        r2_pass = test_r2 >= MIN_R2_THRESHOLD
        mae_pass = test_mae <= MAX_MAE_THRESHOLD
        overfitting_pass = overfitting <= MAX_OVERFITTING_THRESHOLD
        
        all_pass = r2_pass and mae_pass and overfitting_pass
        
        print(f"\n{model_name}:")
        print(f"  R² Score: {test_r2:.3f} {'✓' if r2_pass else '✗'} (min: {MIN_R2_THRESHOLD})")
        print(f"  MAE: {test_mae:.3f} {'✓' if mae_pass else '✗'} (max: {MAX_MAE_THRESHOLD})")
        print(f"  Overfitting: {overfitting:.3f} {'✓' if overfitting_pass else '✗'} (max: {MAX_OVERFITTING_THRESHOLD})")
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
    lr_model = next(model for model in models if isinstance(model, LinearRegressionModel))
    
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
    rf_model = next(model for model in models if isinstance(model, RandomForestModel))
    
    print(f"\nRandom Forest Interpretability:")
    try:
        importance_df = rf_model.get_feature_importance(feature_names)
        print(f"  ✓ Feature importance extraction successful")
        print(f"  ✓ Top feature: {importance_df.iloc[0]['feature']} (importance: {importance_df.iloc[0]['importance']:.3f})")
        
    except Exception as e:
        print(f"  ✗ Random forest interpretability failed: {e}")
    
    # Test Neural Network training history
    nn_model = next(model for model in models if isinstance(model, NeuralNetworkModel))
    
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
    os.makedirs('models', exist_ok=True)
    
    for model in models:
        model_filename = f"models/test_{model.model_name.lower().replace(' ', '_')}.pkl"
        
        try:
            # Test saving
            model.save_model(model_filename)
            print(f"  ✓ {model.model_name} saved successfully")
            
            # Test loading
            new_model = type(model)()
            new_model.load_model(model_filename)
            print(f"  ✓ {model.model_name} loaded successfully")
            
            # Verify loaded model works
            if hasattr(new_model, 'model') and new_model.model is not None:
                print(f"  ✓ {model.model_name} model object restored")
            else:
                print(f"  ✗ {model.model_name} model object not properly restored")
                
            # Clean up test file
            os.remove(model_filename)
            
        except Exception as e:
            print(f"  ✗ {model.model_name} persistence failed: {e}")

def test_all_models():
    """
    Complete model training and comparison pipeline.
    """
    print("=== COMPLETE ML MODEL COMPARISON ===")
    print(f"Using dataset size: {DATASET_SIZE}")
    print(f"Random state: {RANDOM_STATE}")
    
    # Step 1: Generate and prepare data
    print("\nGenerating data...")
    generator = FinancialDataGenerator()
    raw_data = generator.generate_complete_dataset(DATASET_SIZE)  
    
    print("Engineering features...")
    feature_engineer = FinancialFeatureEngineer()
    X_train, X_test, y_train, y_test = feature_engineer.create_ml_dataset(raw_data)
    
    print(f"Data prepared: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    print(f"Features: {X_train.shape[1]}")
    
    # Step 2: Create all models with config parameters
    print("\nCreating models...")
    models = [
        LinearRegressionModel(),
        RandomForestModel(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=RANDOM_STATE),
        NeuralNetworkModel(hidden_layers=HIDDEN_LAYERS, max_iter=MAX_ITER, random_state=RANDOM_STATE)
    ]
    
    print(f"Models created:")
    for model in models:
        print(f"  - {model.model_name}")
    
    # Step 3: Set up comparison framework
    print("\nSetting up model comparison...")
    comparison = ModelComparison()
    for model in models:
        comparison.add_model(model)
    
    # Step 4: Train all models
    print("\nTraining all models...")
    comparison.train_all_models(X_train, y_train)
    
    # Step 5: Evaluate all models
    print("\nEvaluating all models...")
    comparison.evaluate_all_models(X_test, y_test)
    
    # Step 6: Compare results
    print("\nComparing results...")
    results_df = comparison.compare_results()
    
    # Step 7: Validate performance
    validation_results = validate_model_performance(results_df)
    
    # Step 8: Analyze predictions
    analyze_prediction_distribution(models, X_test, y_test)
    
    # Step 9: Test interpretability
    test_model_interpretability(models, X_train.columns.tolist())
    
    # Step 10: Test model persistence
    test_model_persistence(models)
    
    # Step 11: Visualize predictions (optional - comment out if running headless)
    try:
        print("\nGenerating prediction comparison plots...")
        comparison.plot_predictions_comparison(X_test, y_test)
    except Exception as e:
        print(f"Visualization skipped: {e}")
    
    # Step 12: Analyze feature importance (for interpretable models)
    print(f"\n=== FEATURE IMPORTANCE ANALYSIS ===")
    
    # Linear Regression coefficients
    lr_model = models[0]
    lr_importance = lr_model.get_feature_importance(X_train.columns.tolist())
    print(f"\nLinear Regression - Top {N_IMPORTANT} Most Important Features:")
    print(lr_importance.head(N_IMPORTANT))
    
    # Random Forest importance
    rf_model = models[1]
    rf_importance = rf_model.get_feature_importance(X_train.columns.tolist())
    print(f"\nRandom Forest - Top {N_IMPORTANT} Most Important Features:")
    print(rf_importance.head(N_IMPORTANT))
    
    # Plot Random Forest feature importance (optional)
    try:
        print("\nGenerating feature importance plot...")
        rf_model.plot_feature_importance(X_train.columns.tolist())
    except Exception as e:
        print(f"Feature importance plot skipped: {e}")
    
    # Step 13: Neural Network training history (optional)
    try:
        print("\nGenerating neural network training history...")
        nn_model = models[2]
        nn_model.plot_training_history()
    except Exception as e:
        print(f"Training history plot skipped: {e}")
    
    # Step 14: Save best model
    best_model_name = results_df.iloc[0]['Model']
    best_model = next(model for model in models if model.model_name == best_model_name)
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    best_model_filename = f'models/{best_model_name.lower().replace(" ", "_")}_model.pkl'
    best_model.save_model(best_model_filename)
    print(f"\nBest model ({best_model_name}) saved to {best_model_filename}")
    
    # Step 15: Summary
    print(f"\n=== TESTING SUMMARY ===")
    print(f"Dataset size: {DATASET_SIZE}")
    print(f"Features generated: {X_train.shape[1]}")
    print(f"Models tested: {len(models)}")
    print(f"Best model: {best_model_name} (R² = {results_df.iloc[0]['Test_R2']:.3f})")
    
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
