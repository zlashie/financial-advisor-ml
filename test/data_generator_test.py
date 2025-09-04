########################################
#### Dependendencies
########################################
import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_generator import FinancialDataGenerator

########################################
#### Test Constants
########################################

NO_DEBT = 0
HIGH_INCOME_THRESHOLD = 100000

########################################
#### Test Variables
########################################

n_samples = 100

########################################
#### Test generate market conditions
########################################
print("========== MARKET CONDITIONS ==========")
generator = FinancialDataGenerator()
market_data = generator.generate_market_conditions(n_samples)
print(market_data.head())
print(market_data.describe())

########################################
#### Test generate personal profiles
########################################
print("========== PERSONAL PROFILES ==========")
personal_data = generator.generate_personal_profiles(n_samples)
print(personal_data.head())
print(f"% People with credit card debt: {(personal_data['cc_debt'] > NO_DEBT).sum()}%")
print(f"% People with mortgages: {(personal_data['mortgage_debt'] > NO_DEBT).sum()}%")

########################################
#### Test calculate optimal strategy
########################################
print("========== OPTIMAL STRATEGY CALCULATION ==========")

# Generate test data
market_conditions = generator.generate_market_conditions(n_samples)
personal_profiles = generator.generate_personal_profiles(n_samples)

# Calculate optimal strategies
optimal_strategies = generator.calculate_optimal_strategy(market_conditions, personal_profiles)

print("Optimal Strategy Results:")
print(optimal_strategies.head())
print("\nOptimal Strategy Statistics:")
print(optimal_strategies.describe())

# Additional analysis
print(f"\nAverage recommended investment ratio: {optimal_strategies['recommended_investment_ratio'].mean():.3f}")
print(f"Min investment ratio: {optimal_strategies['recommended_investment_ratio'].min():.3f}")
print(f"Max investment ratio: {optimal_strategies['recommended_investment_ratio'].max():.3f}")

# Distribution of investment recommendations
investment_categories = pd.cut(optimal_strategies['recommended_investment_ratio'], 
                             bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                             labels=['Conservative', 'Moderate', 'Balanced', 'Growth', 'Aggressive'])
print(f"\nInvestment Strategy Distribution:")
print(investment_categories.value_counts())

# Correlation analysis
print(f"\nCorrelation between age and investment ratio: {personal_profiles['age'].corr(optimal_strategies['recommended_investment_ratio']):.3f}")
print(f"Correlation between income and investment ratio: {personal_profiles['monthly_income'].corr(optimal_strategies['recommended_investment_ratio']):.3f}")

# Test edge cases
print(f"\nPeople with no debt: {((personal_profiles['cc_debt'] == 0) & (personal_profiles['mortgage_debt'] == 0)).sum()}")
print(f"High income individuals (>${HIGH_INCOME_THRESHOLD}): {(personal_profiles['monthly_income'] > HIGH_INCOME_THRESHOLD).sum()}")