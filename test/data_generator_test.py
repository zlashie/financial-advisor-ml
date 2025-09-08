########################################
#### Dependencies
########################################
import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_generator import FinancialDataGenerator
from src.config import config

########################################
#### Load Test Configuration
########################################

# Load constants from config
NO_DEBT = config.get('common', 'financial_constants', 'no_debt')
HIGH_INCOME_THRESHOLD = config.get('data_generation', 'personal_profile', 'income', 'high_threshold')

# Test parameters - could also be moved to a test config if needed
N_SAMPLES = 100

# Investment strategy thresholds for categorization
strategy_config = config.get('data_generation', 'investment_strategy', 'allocation_ratios')
CONSERVATIVE_THRESHOLD = strategy_config['conservative']
MODERATE_THRESHOLD = strategy_config['moderate'] 
GROWTH_THRESHOLD = strategy_config['growth']
AGGRESSIVE_THRESHOLD = strategy_config['aggressive']

########################################
#### Test generate market conditions
########################################
print("========== MARKET CONDITIONS ==========")
generator = FinancialDataGenerator()
market_data = generator.generate_market_conditions(N_SAMPLES)
print(market_data.head())
print(market_data.describe())

# Validate market conditions against config ranges
sp500_config = config.get('data_generation', 'market_conditions', 'sp500')
treasury_config = config.get('data_generation', 'market_conditions', 'treasury')
vix_config = config.get('data_generation', 'market_conditions', 'vix')

print(f"\nValidation against config ranges:")
print(f"S&P 500 P/E - Min: {market_data['sp500_pe'].min():.2f} (Config min: {sp500_config['pe_min']})")
print(f"Treasury Yield - Range: {market_data['treasury_yield'].min():.3f}-{market_data['treasury_yield'].max():.3f} (Config: {treasury_config['min_pct']}-{treasury_config['max_pct']})")
print(f"VIX - Max: {market_data['vix'].max():.2f} (Config max: {vix_config['max']})")

########################################
#### Test generate personal profiles
########################################
print("\n========== PERSONAL PROFILES ==========")
personal_data = generator.generate_personal_profiles(N_SAMPLES)
print(personal_data.head())

# Calculate debt statistics
cc_debt_pct = (personal_data['cc_debt'] > NO_DEBT).mean() * 100
mortgage_debt_pct = (personal_data['mortgage_debt'] > NO_DEBT).mean() * 100

print(f"% People with credit card debt: {cc_debt_pct:.1f}%")
print(f"% People with mortgages: {mortgage_debt_pct:.1f}%")

# Validate against config probabilities
debt_config = config.get('data_generation', 'debt')
expected_cc_pct = debt_config['credit_card']['probability'] * 100
expected_mortgage_pct = debt_config['mortgage']['probability'] * 100

print(f"\nExpected vs Actual debt rates:")
print(f"Credit Card - Expected: {expected_cc_pct}%, Actual: {cc_debt_pct:.1f}%")
print(f"Mortgage - Expected: {expected_mortgage_pct}%, Actual: {mortgage_debt_pct:.1f}%")

# Age and income validation
age_config = config.get('data_generation', 'personal_profile', 'age')
print(f"\nAge range - Min: {personal_data['age'].min()} (Config: {age_config['work_min']}), Max: {personal_data['age'].max()} (Config: {age_config['work_max']})")

########################################
#### Test calculate optimal strategy
########################################
print("\n========== OPTIMAL STRATEGY CALCULATION ==========")

# Generate test data
market_conditions = generator.generate_market_conditions(N_SAMPLES)
personal_profiles = generator.generate_personal_profiles(N_SAMPLES)

# Calculate optimal strategies
optimal_strategies = generator.calculate_optimal_strategy(market_conditions, personal_profiles)

print("Optimal Strategy Results:")
print(optimal_strategies.head())
print("\nOptimal Strategy Statistics:")
print(optimal_strategies.describe())

# Additional analysis
avg_investment_ratio = optimal_strategies['recommended_investment_ratio'].mean()
min_investment_ratio = optimal_strategies['recommended_investment_ratio'].min()
max_investment_ratio = optimal_strategies['recommended_investment_ratio'].max()

print(f"\nAverage recommended investment ratio: {avg_investment_ratio:.3f}")
print(f"Min investment ratio: {min_investment_ratio:.3f}")
print(f"Max investment ratio: {max_investment_ratio:.3f}")

# Validate bounds from config
bounds_config = config.get('data_generation', 'investment_strategy', 'bounds')
expected_min = bounds_config['min_investment_ratio']
expected_max = bounds_config['max_investment_ratio']

print(f"\nBounds validation:")
print(f"Expected range: {expected_min}-{expected_max}")
print(f"Actual range: {min_investment_ratio:.3f}-{max_investment_ratio:.3f}")
print(f"Within bounds: {min_investment_ratio >= expected_min and max_investment_ratio <= expected_max}")

# Distribution of investment recommendations using config thresholds
investment_categories = pd.cut(optimal_strategies['recommended_investment_ratio'], 
                             bins=[0, CONSERVATIVE_THRESHOLD, MODERATE_THRESHOLD, 
                                  GROWTH_THRESHOLD, AGGRESSIVE_THRESHOLD, 1.0], 
                             labels=['Very Conservative', 'Conservative', 'Moderate', 'Growth', 'Aggressive+'])

print(f"\nInvestment Strategy Distribution:")
print(investment_categories.value_counts())

# Correlation analysis
age_investment_corr = personal_profiles['age'].corr(optimal_strategies['recommended_investment_ratio'])
income_investment_corr = personal_profiles['monthly_income'].corr(optimal_strategies['recommended_investment_ratio'])

print(f"\nCorrelation Analysis:")
print(f"Age vs Investment ratio: {age_investment_corr:.3f}")
print(f"Income vs Investment ratio: {income_investment_corr:.3f}")

# Test edge cases
no_debt_count = ((personal_profiles['cc_debt'] == NO_DEBT) & 
                (personal_profiles['mortgage_debt'] == NO_DEBT)).sum()
high_income_count = (personal_profiles['monthly_income'] > HIGH_INCOME_THRESHOLD).sum()

print(f"\nEdge Cases:")
print(f"People with no debt: {no_debt_count}")
print(f"High income individuals (>${HIGH_INCOME_THRESHOLD:,}): {high_income_count}")

# Test specific scenarios
print(f"\nScenario Analysis:")

# High debt scenario
high_debt_mask = (personal_profiles['cc_debt'] > 0) | (personal_profiles['mortgage_debt'] > 0)
if high_debt_mask.any():
    avg_investment_with_debt = optimal_strategies[high_debt_mask]['recommended_investment_ratio'].mean()
    avg_investment_no_debt = optimal_strategies[~high_debt_mask]['recommended_investment_ratio'].mean()
    
    print(f"Average investment ratio with debt: {avg_investment_with_debt:.3f}")
    print(f"Average investment ratio without debt: {avg_investment_no_debt:.3f}")
    print(f"Debt impact: {avg_investment_no_debt - avg_investment_with_debt:.3f} lower investment ratio")

# Age impact
young_mask = personal_profiles['age'] < 35
old_mask = personal_profiles['age'] > 55

if young_mask.any() and old_mask.any():
    avg_young_investment = optimal_strategies[young_mask]['recommended_investment_ratio'].mean()
    avg_old_investment = optimal_strategies[old_mask]['recommended_investment_ratio'].mean()
    
    print(f"Average investment ratio (young <35): {avg_young_investment:.3f}")
    print(f"Average investment ratio (older >55): {avg_old_investment:.3f}")
    print(f"Age impact: {avg_young_investment - avg_old_investment:.3f} higher for young")

print("\n========== TEST COMPLETED ==========")
