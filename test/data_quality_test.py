########################################
#### Dependencies
########################################
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import config
from src.generators.financial_dataset_generator import FinancialDatasetGenerator
from src.generators.market_data_generator import SP500MarketDataGenerator
from src.generators.personal_profile_generator import StandardPersonalProfileGenerator
from src.calculators.investment_strategy_calculator import OptimalInvestmentStrategyCalculator
from src.factories.config_factory import ConfigFactory

########################################
#### Setup Generators
########################################

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
dataset_generator = FinancialDatasetGenerator(
    market_generator, personal_generator, strategy_calculator
)

########################################
#### Load Test Configuration
########################################

# Test parameters
N_TEST_RUNS = config.get('data_generation', 'output', 'complete_runs') 

# Load constants from config
NO_DEBT = config.get('common', 'financial_constants', 'no_debt')

# Load debt rate thresholds
debt_thresholds = config.get('data_generation', 'investment_strategy', 'debt_rate_thresholds')
HIGH_CC_RATE = debt_thresholds['highest'] 
MEDIUM_CC_RATE = debt_thresholds['medium']  

# Load allocation ratios for validation
allocation_ratios = config.get('data_generation', 'investment_strategy', 'allocation_ratios')
CONSERVATIVE_RATIO = allocation_ratios['conservative']  
AGGRESSIVE_RATIO = allocation_ratios['aggressive']  

# Load age thresholds
age_config = config.get('data_generation', 'personal_profile', 'age')
YOUNG_AGE = 30  
OLD_AGE = 50   

# Load investment bounds
bounds_config = config.get('data_generation', 'investment_strategy', 'bounds')
MIN_INVESTMENT_RATIO = bounds_config['min_investment_ratio']
MAX_INVESTMENT_RATIO = bounds_config['max_investment_ratio']

# Load market thresholds
market_config_json = config.get('data_generation', 'market_conditions')
SP500_OVERVALUED = market_config_json['sp500']['overvalued_threshold']
SP500_UNDERVALUED = market_config_json['sp500']['undervalued_threshold']
VIX_HIGH = market_config_json['vix']['high_threshold']

########################################
#### Test Data Quality
########################################

def analyze_generated_data():
    """
    Inspect synthesized training data from data_generator.py
    """

    data = dataset_generator.generate_complete_dataset(N_TEST_RUNS)

    print("=== DATA QUALITY ANALYSIS ===")
    print(f"Dataset shape: {data.shape}")
    print(f"Missing values: {data.isnull().sum().sum()}")
    
    print("\n=== FEATURE DISTRIBUTIONS ===")
    print(data.describe())
    
    print("\n=== BUSINESS LOGIC VALIDATION ===")

    # Test 1: People with high credit card debt should mostly pay debt first
    high_cc_debt = data[data['cc_rate'] > HIGH_CC_RATE]
    if len(high_cc_debt) > 0:
        avg_invest_ratio_high_debt = high_cc_debt['recommended_investment_ratio'].mean()
        print(f"Average invest ratio for high CC debt (>{HIGH_CC_RATE:.1%}): {avg_invest_ratio_high_debt:.2f}")
        print(f"Expected: Should be low (< {CONSERVATIVE_RATIO + 0.2:.1f})")
        
        # Validation
        if avg_invest_ratio_high_debt < (CONSERVATIVE_RATIO + 0.2):
            print("✓ PASS: High debt leads to conservative investment")
        else:
            print("✗ FAIL: High debt should lead to more conservative investment")
    else:
        print("No high CC debt cases found in sample")

    # Test 2: People with no debt should invest more
    no_debt = data[(data['cc_debt'] == NO_DEBT) & (data['mortgage_debt'] == NO_DEBT)]
    if len(no_debt) > 0:
        avg_invest_ratio_no_debt = no_debt['recommended_investment_ratio'].mean()
        print(f"\nAverage invest ratio for no debt: {avg_invest_ratio_no_debt:.2f}")
        print(f"Expected: Should be high (> {AGGRESSIVE_RATIO - 0.1:.1f})")
        
        # Validation
        if avg_invest_ratio_no_debt > (AGGRESSIVE_RATIO - 0.1):
            print("✓ PASS: No debt leads to aggressive investment")
        else:
            print("✗ FAIL: No debt should lead to more aggressive investment")
    else:
        print("No debt-free cases found in sample")

    # Test 3: Younger people should invest more (given same debt situation)
    young_no_debt = data[(data['age'] < YOUNG_AGE) & (data['cc_debt'] == NO_DEBT) & (data['mortgage_debt'] == NO_DEBT)]
    old_no_debt = data[(data['age'] > OLD_AGE) & (data['cc_debt'] == NO_DEBT) & (data['mortgage_debt'] == NO_DEBT)]

    if len(young_no_debt) > 0 and len(old_no_debt) > 0:
        young_invest = young_no_debt['recommended_investment_ratio'].mean()
        old_invest = old_no_debt['recommended_investment_ratio'].mean()
        print(f"\nYoung people (<{YOUNG_AGE}, no debt) invest ratio: {young_invest:.2f}")
        print(f"Older people (>{OLD_AGE}, no debt) invest ratio: {old_invest:.2f}")
        print("Expected: Young > Old")
        
        # Validation
        if young_invest > old_invest:
            print("✓ PASS: Younger people invest more aggressively")
        else:
            print("✗ FAIL: Age factor not working correctly")
    else:
        print(f"\nInsufficient samples for age comparison (Young: {len(young_no_debt)}, Old: {len(old_no_debt)})")

    # Test 4: Investment ratio should be between configured bounds
    invalid_ratios = data[(data['recommended_investment_ratio'] < MIN_INVESTMENT_RATIO) | 
                         (data['recommended_investment_ratio'] > MAX_INVESTMENT_RATIO)]
    print(f"\nInvestment ratio bounds validation:")
    print(f"Expected range: {MIN_INVESTMENT_RATIO} - {MAX_INVESTMENT_RATIO}")
    print(f"Actual range: {data['recommended_investment_ratio'].min():.3f} - {data['recommended_investment_ratio'].max():.3f}")
    print(f"Invalid investment ratios (outside bounds): {len(invalid_ratios)}")
    
    if len(invalid_ratios) == 0:
        print("✓ PASS: All investment ratios within valid bounds")
    else:
        print("✗ FAIL: Some investment ratios outside valid bounds")

    # Test 5: Check correlation between debt rates and investment ratios
    correlation = data['highest_debt_rate'].corr(data['recommended_investment_ratio'])
    print(f"\nCorrelation between highest debt rate and investment ratio: {correlation:.3f}")
    print("Expected: Should be negative (higher debt rate = lower investment ratio)")
    
    if correlation < -0.1:  # Reasonable negative correlation threshold
        print("✓ PASS: Negative correlation between debt rate and investment")
    else:
        print("✗ FAIL: Expected stronger negative correlation")

    # Test 6: Market condition impact validation
    print(f"\n=== MARKET CONDITION VALIDATION ===")
    
    # Overvalued market should lead to lower investment
    overvalued_market = data[data['sp500_pe'] > SP500_OVERVALUED]
    undervalued_market = data[data['sp500_pe'] < SP500_UNDERVALUED]
    
    if len(overvalued_market) > 0 and len(undervalued_market) > 0:
        overvalued_avg = overvalued_market['recommended_investment_ratio'].mean()
        undervalued_avg = undervalued_market['recommended_investment_ratio'].mean()
        
        print(f"Investment ratio in overvalued market (P/E > {SP500_OVERVALUED}): {overvalued_avg:.3f}")
        print(f"Investment ratio in undervalued market (P/E < {SP500_UNDERVALUED}): {undervalued_avg:.3f}")
        
        if undervalued_avg > overvalued_avg:
            print("✓ PASS: Higher investment in undervalued markets")
        else:
            print("✗ FAIL: Market valuation impact not working correctly")

    # High VIX should lead to lower investment
    high_vix = data[data['vix'] > VIX_HIGH]
    low_vix = data[data['vix'] <= VIX_HIGH]
    
    if len(high_vix) > 0:
        high_vix_avg = high_vix['recommended_investment_ratio'].mean()
        low_vix_avg = low_vix['recommended_investment_ratio'].mean()
        
        print(f"\nInvestment ratio in high volatility (VIX > {VIX_HIGH}): {high_vix_avg:.3f}")
        print(f"Investment ratio in normal volatility (VIX ≤ {VIX_HIGH}): {low_vix_avg:.3f}")
        
        if low_vix_avg > high_vix_avg:
            print("✓ PASS: Lower investment during high volatility")
        else:
            print("✗ FAIL: VIX impact not working correctly")

    # Test 7: Data distribution validation
    print(f"\n=== DATA DISTRIBUTION VALIDATION ===")
    
    # Check debt prevalence matches config
    debt_config_json = config.get('data_generation', 'debt')
    expected_cc_rate = debt_config_json['credit_card']['probability']
    expected_mortgage_rate = debt_config_json['mortgage']['probability']
    
    actual_cc_rate = (data['cc_debt'] > NO_DEBT).mean()
    actual_mortgage_rate = (data['mortgage_debt'] > NO_DEBT).mean()
    
    print(f"Credit card debt prevalence - Expected: {expected_cc_rate:.1%}, Actual: {actual_cc_rate:.1%}")
    print(f"Mortgage debt prevalence - Expected: {expected_mortgage_rate:.1%}, Actual: {actual_mortgage_rate:.1%}")
    
    # Allow for some statistical variation (±10%)
    cc_tolerance = 0.1
    mortgage_tolerance = 0.1
    
    if abs(actual_cc_rate - expected_cc_rate) <= cc_tolerance:
        print("✓ PASS: Credit card debt prevalence within expected range")
    else:
        print("✗ FAIL: Credit card debt prevalence outside expected range")
    
    if abs(actual_mortgage_rate - expected_mortgage_rate) <= mortgage_tolerance:
        print("✓ PASS: Mortgage debt prevalence within expected range")
    else:
        print("✗ FAIL: Mortgage debt prevalence outside expected range")

    print(f"\n=== SUMMARY ===")
    print(f"Data generation completed successfully with {len(data)} samples")
    print("Review validation results above for any failing tests")

    return data

def plot_data_distributions(data):
    """
    Create visualizations to inspect data quality
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Investment ratio distribution
    axes[0, 0].hist(data['recommended_investment_ratio'], bins=30, alpha=0.7)
    axes[0, 0].set_title('Investment Ratio Distribution')
    axes[0, 0].set_xlabel('Investment Ratio')
    
    # Age vs Investment Ratio
    axes[0, 1].scatter(data['age'], data['recommended_investment_ratio'], alpha=0.5)
    axes[0, 1].set_title('Age vs Investment Ratio')
    axes[0, 1].set_xlabel('Age')
    axes[0, 1].set_ylabel('Investment Ratio')
    
    # Debt Rate vs Investment Ratio
    axes[0, 2].scatter(data['highest_debt_rate'], data['recommended_investment_ratio'], alpha=0.5)
    axes[0, 2].set_title('Debt Rate vs Investment Ratio')
    axes[0, 2].set_xlabel('Highest Debt Rate')
    axes[0, 2].set_ylabel('Investment Ratio')
    
    # Market conditions
    axes[1, 0].hist(data['sp500_pe'], bins=30, alpha=0.7)
    axes[1, 0].set_title('S&P 500 P/E Distribution')
    axes[1, 0].set_xlabel('P/E Ratio')
    
    axes[1, 1].hist(data['vix'], bins=30, alpha=0.7)
    axes[1, 1].set_title('VIX Distribution')
    axes[1, 1].set_xlabel('VIX')
    
    # Income distribution
    axes[1, 2].hist(data['monthly_income'], bins=30, alpha=0.7)
    axes[1, 2].set_title('Monthly Income Distribution')
    axes[1, 2].set_xlabel('Monthly Income')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data = analyze_generated_data()
    
    # Uncomment to show plots
    # plot_data_distributions(data)