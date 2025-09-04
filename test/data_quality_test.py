########################################
#### Dependendencies
########################################
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_generator import FinancialDataGenerator

########################################
#### Test Constants
########################################

N_TEST_RUNS = 1000
HIGH_CC_RATE = 0.2
NO_DEBT = 0
YOUNG_AGE = 30
OLD_AGE = 50

########################################
#### Test Data Quality
########################################

def analyze_generated_data():
    """
    Inspect synthesized training data from data.generator.py
    """

    generator = FinancialDataGenerator()
    data = generator.generate_complete_dataset(N_TEST_RUNS)

    print("=== DATA QUALITY ANALYSIS ===")
    print(f"Dataset shape: {data.shape}")
    print(f"Missing values: {data.isnull().sum().sum()}")
    
    print("\n=== FEATURE DISTRIBUTIONS ===")
    print(data.describe())
    
    print("\n=== BUSINESS LOGIC VALIDATION ===")

    # Test 1: People with high credit card debt should mostly pay debt first
    high_cc_debt = data[data['cc_rate'] > HIGH_CC_RATE]
    avg_invest_ratio_high_debt = high_cc_debt['recommended_investment_ratio'].mean()

    print(f"Average invest ratio for high CC debt (>20%): {avg_invest_ratio_high_debt:.2f}")
    print("Expected: Should be low (< 0.3)")


    # Test 2: People with no debt should invest more
    no_debt = data[(data['cc_debt'] == NO_DEBT) & (data['mortgage_debt'] == NO_DEBT)]
    avg_invest_ratio_no_debt = no_debt['recommended_investment_ratio'].mean()
    print(f"Average invest ratio for no debt: {avg_invest_ratio_no_debt:.2f}")
    print("Expected: Should be high (> 0.7)")

    # Test 3: Younger people should invest more (given same debt situation)
    young_no_debt = data[(data['age'] < YOUNG_AGE) & (data['cc_debt'] == NO_DEBT)]
    old_no_debt = data[(data['age'] > OLD_AGE) & (data['cc_debt'] == NO_DEBT)]

    if len(young_no_debt) > NO_DEBT and len(old_no_debt) > NO_DEBT:
        young_invest = young_no_debt['recommended_investment_ratio'].mean()
        old_invest = old_no_debt['recommended_investment_ratio'].mean()
        print(f"Young people (no debt) invest ratio: {young_invest:.2f}")
        print(f"Older people (no debt) invest ratio: {old_invest:.2f}")
        print("Expected: Young > Old")

    # Test 4: Investment ratio should be between 0 and 1
    invalid_ratios = data[(data['recommended_investment_ratio'] < 0) | (data['recommended_investment_ratio'] > 1)]
    print(f"Invalid investment ratios (outside 0-1): {len(invalid_ratios)}")

    # Test 5: Check correlation between debt rates and investment ratios
    correlation = data['highest_debt_rate'].corr(data['recommended_investment_ratio'])
    print(f"Correlation between highest debt rate and investment ratio: {correlation:.3f}")
    print("Expected: Should be negative (higher debt rate = lower investment ratio)")

    return data

if __name__ == "__main__":
    data = analyze_generated_data()