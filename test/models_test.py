########################################
#### Dependendencies
########################################
import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_generator import FinancialDataGenerator
from src.feature_engineer import FinancialFeatureEngineer
from src.models import LinearRegressionModel, RandomForestModel, NeuralNetworkModel, ModelComparison

########################################
#### Test Variables
########################################
DATASET_SIZE = 2000
N_ESTIMATORS = 100
MAX_DEPTH = 10
MAX_ITER = 1000
HIDDEN_LAYERS = (100, 50)
N_IMPORTANT = 10

########################################
#### Test 
########################################

def test_all_models():
    """
    Complete model training and comparison pipeline.
    """
    print("=== COMPLETE ML MODEL COMPARISON ===")
    
    # Step 1: Generate and prepare data
    print("Generating data...")
    generator = FinancialDataGenerator()
    raw_data = generator.generate_complete_dataset(DATASET_SIZE)  
    
    print("Engineering features...")
    feature_engineer = FinancialFeatureEngineer()
    X_train, X_test, y_train, y_test = feature_engineer.create_ml_dataset(raw_data)
    
    # Step 2: Create all models
    models = [
        LinearRegressionModel(),
        RandomForestModel(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH),
        NeuralNetworkModel(hidden_layers=HIDDEN_LAYERS, max_iter=MAX_ITER)
    ]
    
    # Step 3: Set up comparison framework
    comparison = ModelComparison()
    for model in models:
        comparison.add_model(model)
    
    # Step 4: Train all models
    comparison.train_all_models(X_train, y_train)
    
    # Step 5: Evaluate all models
    comparison.evaluate_all_models(X_test, y_test)
    
    # Step 6: Compare results
    results_df = comparison.compare_results()
    
    # Step 7: Visualize predictions
    comparison.plot_predictions_comparison(X_test, y_test)
    
    # Step 8: Analyze feature importance (for interpretable models)
    print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
    
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
    
    # Plot Random Forest feature importance
    rf_model.plot_feature_importance(X_train.columns.tolist())
    
    # Step 9: Neural Network training history
    nn_model = models[2]
    nn_model.plot_training_history()
    
    # Step 10: Save best model
    best_model_name = results_df.iloc[0]['Model']
    best_model = next(model for model in models if model.model_name == best_model_name)
    best_model.save_model(f'models/{best_model_name.lower().replace(" ", "_")}_model.pkl')
    
    return models, results_df, X_test, y_test

if __name__ == "__main__":
    models, results, X_test, y_test = test_all_models()