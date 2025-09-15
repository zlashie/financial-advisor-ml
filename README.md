# financial-advisor-ml
ML project to estimate best resource allocation between debt payments and investments based on S&P500 P/E ratios, VIX ratios and Treasury bond yields.

## Usecase
The application's aim is to help in the decision making on whether or not to pay debt quickly or to allocate funds to index investments. This application maximises the financial potential, and its up to the user to evaluate the result based on personal life circumstances and goals. 

## Model Performance Analysis
The project implements a SOLID-principle architecture with three ML models for investment ratio prediction:

**Figure 1: Model Prediction Accuracy**
![Alt text](data/ML1.png "Model Type Accuracy Test")
Comparison of predicted vs actual investment ratios across all three models. Random Forest achieves the highest accuracy (R² = 0.949), followed by Linear Regression (R² = 0.930) and Neural Network (R² = 0.716). All models show strong correlation with actual values, with predictions properly bounded between 0-1.

**Figure 2: Feature Importance Analysis**
![Alt text](data/ML2.png "Feature Importance")
Random Forest feature importance revealing that `highest debt rate` is the dominant factor in investment decisions, followed by `credit card rate`, `age`, and `years to retirement`. This confirms the financial logic that high-interest debt should be prioritized before investing.

**Figure 3: Neural Network Training Convergence**
![Alt text](data/ML3.png "Neural Network Training Loss")
Training loss curve showing the neural network converged after 94 iterations with early stopping, achieving a final loss of 0.0010. The model demonstrates stable learning without overfitting during the training process.

## Architecture
Built using SOLID principles with modular services for data generation, feature engineering, model training, and evaluation. Supports easy extension with new models and comprehensive testing framework.

## File Structure
File structure can be listed in terminal using
Get-ChildItem -Recurse -Name | Where-Object { $_ -match '\.(py|yaml|yml)$' -and $_ -notmatch '^venv\\' } | Sort-Object 