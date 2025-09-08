########################################
#### Dependendencies
########################################
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import config  

########################################
#### Load Configuration Values
########################################

# General constants
DEFAULT_SEED = config.get('common', 'general', 'default_seed')
MONTHS_PR_YEAR = config.get('common', 'general', 'months_per_year')
NO_DEBT = config.get('common', 'financial_constants', 'no_debt')
NO_RATE = config.get('common', 'financial_constants', 'no_rate')
RETIREMENT_AGE = config.get('common', 'financial_constants', 'retirement_age')

# Data generation settings
FIN_TRAIN_DATA_NAME = config.get('data_generation', 'output', 'training_data_name')
COMPLETE_RUNS = config.get('data_generation', 'output', 'complete_runs')
SEED_VAL = DEFAULT_SEED

# Market conditions - S&P 500
SP500_MEAN = config.get('data_generation', 'market_conditions', 'sp500', 'mean')
SP500_STDDEV = config.get('data_generation', 'market_conditions', 'sp500', 'stddev')
SP500_PE_MIN = config.get('data_generation', 'market_conditions', 'sp500', 'pe_min')
SP500_OVERVALUED = config.get('data_generation', 'market_conditions', 'sp500', 'overvalued_threshold')
SP500_UNDERVALUED = config.get('data_generation', 'market_conditions', 'sp500', 'undervalued_threshold')

# Treasury rates
TREASURY_MAX_PCT = config.get('data_generation', 'market_conditions', 'treasury', 'max_pct')
TREASURY_MIN_PCT = config.get('data_generation', 'market_conditions', 'treasury', 'min_pct')

# VIX configuration
vix_config = config.get('data_generation', 'market_conditions', 'vix')
VIX_NORMALDIS_MEAN = vix_config['normal_distribution']['mean']
VIX_NORMALDIS_STDDEV = vix_config['normal_distribution']['stddev']
VIX_MAX = vix_config['max']
VIX_HIGH = vix_config['high_threshold']

# Personal profile - Age
personal_config = config.get('data_generation', 'personal_profile')
WORKAGE_MIN = personal_config['age']['work_min']
WORKAGE_MAX = personal_config['age']['work_max']
M_DEBT_AGE_MIN = personal_config['age']['mortgage_debt_min']

# Personal profile - Income
income_config = personal_config['income']
MONTHLY_INCOME_NORMALDIS_MEAN = income_config['normal_distribution']['mean']
MONTHLY_INCOME_NORMALDIS_STDDEV = income_config['normal_distribution']['stddev']
MONTHLY_INCOME_DIS_MIN = income_config['discretionary']['min']
MONTHLY_INCOME_DIS_MAX = income_config['discretionary']['max']
HIGH_INCOME_THRESHOLD = income_config['high_threshold']

# Debt configuration
debt_config = config.get('data_generation', 'debt')

# Credit card debt
cc_debt_config = debt_config['credit_card']
CC_DEBT_P = cc_debt_config['probability']
CC_DEBT_NORMALDIS_MEAN = cc_debt_config['normal_distribution']['mean']
CC_DEBT_NORMALDIS_STDDEV = cc_debt_config['normal_distribution']['stddev']
CC_RATE_MIN = cc_debt_config['rate']['min']
CC_RATE_MAX = cc_debt_config['rate']['max']

# Mortgage debt
m_debt_config = debt_config['mortgage']
M_DEBT_P = m_debt_config['probability']
M_DEBT_INCOME_MIN = m_debt_config['income_multiple']['min']
M_DEBT_INCOME_MAX = m_debt_config['income_multiple']['max']
M_DEBT_INCOME_CAP = m_debt_config['income_multiple']['cap']
M_RATE_MIN = m_debt_config['rate']['min']
M_RATE_MAX = m_debt_config['rate']['max']

# Investment strategy configuration
strategy_config = config.get('data_generation', 'investment_strategy')

# Returns configuration
returns_config = strategy_config['returns']
RETURN_BASE = returns_config['base']
RETURN_ADJUST = returns_config['adjustment']
RETURN_REALRATIO = returns_config['real_ratio']
RETURN_VIXRATIO = returns_config['vix_ratio']

# Allocation ratios
allocation_config = strategy_config['allocation_ratios']
INVEST_CONSERVATIVE = allocation_config['conservative']
INVEST_MODERATE = allocation_config['moderate']
INVEST_GROWTH = allocation_config['growth']
INVEST_AGGRESSIVE = allocation_config['aggressive']
INVEST_ALLIN = allocation_config['all_in']

# Debt rate thresholds
debt_thresholds = strategy_config['debt_rate_thresholds']
DEBT_RATE_HIGHEST = debt_thresholds['highest']
DEBT_RATE_HIGH = debt_thresholds['high']
DEBT_RATE_MEDIUM = debt_thresholds['medium']

# Age adjustments
age_adjustments = strategy_config['age_adjustments']
AGE_RISK_DIVISOR = age_adjustments['risk_divisor']
MIN_AGE_FACTOR = age_adjustments['min_age_factor']

# Income adjustments
INCOME_BOOST_MULTIPLIER = strategy_config['income_boost_multiplier']

# Investment bounds
bounds_config = strategy_config['bounds']
MIN_INVESTMENT_RATIO = bounds_config['min_investment_ratio']
MAX_INVESTMENT_RATIO = bounds_config['max_investment_ratio']

# Market adjustments - Generic approach
market_adjustments = strategy_config.get('market_adjustments', {})

# Valuation adjustments
valuation_config = market_adjustments.get('valuation_impact', {})
OVERVALUED_REDUCTION = valuation_config.get('overvalued_reduction', 0.15)
UNDERVALUED_BOOST = valuation_config.get('undervalued_boost', 0.15)

# Volatility adjustments  
volatility_config = market_adjustments.get('volatility_impact', {})
HIGH_VIX_REDUCTION = volatility_config.get('high_vix_reduction', 0.10)

# Debt-free dampening
DEBT_FREE_DAMPENING = market_adjustments.get('debt_free_dampening', 0.5)

# Calculate multipliers dynamically
OVERVALUED_MULTIPLIER = 1.0 - OVERVALUED_REDUCTION  
UNDERVALUED_MULTIPLIER = 1.0 + UNDERVALUED_BOOST   
HIGH_VIX_MULTIPLIER = 1.0 - HIGH_VIX_REDUCTION    

########################################
#### Class Definitions
########################################

class FinancialDataGenerator:
    """
    Class to generate synthetic financial scenarios for training the ML models.
    """

    def __init__(self, random_seed: int = SEED_VAL):
        """
        Set random_seed to seed_val to ensure reproducability in testing.
        """
        np.random.seed(random_seed)
        random.seed(random_seed)

    def generate_market_conditions(self, n_samples: int) -> pd.DataFrame:
        """
        Generate synthetic data replicating real world market condition data.

        Specifically we are adding:
        - S&P 500 P/E Ratio: as a measurement of how expensive the stock market is priced for greed or fear, in terms of company earnings reports
        - 10-Year Treasury Yield: as a measurement of how attractive the stock market is compared to the risk-free treasury yield. 
        - VIX: Quantitative Market volatility/Fear measure - high VIX means more uncertainty. 

        Choice of probability distributions:
        - S&P 500 P/E ratio: normal distribution as it clusters around the mean, defined as the fair value of a stock.
        - 10-Year Treasury Yield: uniform as the Fed can set rates anywhere in the range based on market atmosphere
        - VIX: log-normal because it is usually low but occasionally spikes very high during crisis (dot-com, financial, corona, etc)
        """
        market_data = []

        for _ in range(n_samples):
            sp500_pe = max(SP500_PE_MIN, np.random.normal(SP500_MEAN, SP500_STDDEV))
            treasury_yield = np.random.uniform(TREASURY_MIN_PCT, TREASURY_MAX_PCT)
            vix = np.random.lognormal(VIX_NORMALDIS_MEAN, VIX_NORMALDIS_STDDEV)

            market_data.append({
                'sp500_pe': sp500_pe,
                'treasury_yield': treasury_yield,
                'vix': min(vix, VIX_MAX)
            })

        return pd.DataFrame(market_data)
    
    def generate_personal_profiles(self, n_samples: int) -> pd.DataFrame:
        """
        Generate synthetic data replicating realistic personal financial situations.

        What was added:
        - Age: investment probability and risk profile depends heavily on investment time horizon
        - Monthly income: investment opportunity depend on available income
        - Debt: allocation of resources depends on debt profile
        -- Credit card debt
        -- Mortgage debt
        - Monthly discretionaries: available income for investments
        """

        personal_data = []

        for _ in range(n_samples):
            age = np.random.randint(WORKAGE_MIN, WORKAGE_MAX)
            monthly_income = np.random.lognormal(MONTHLY_INCOME_NORMALDIS_MEAN, MONTHLY_INCOME_NORMALDIS_STDDEV)
            
            has_cc_debt = np.random.random() < CC_DEBT_P
            if has_cc_debt:
                cc_debt = np.random.lognormal(CC_DEBT_NORMALDIS_MEAN, CC_DEBT_NORMALDIS_STDDEV)
                cc_rate = np.random.uniform(CC_RATE_MIN, CC_RATE_MAX)
            else:
                cc_debt = NO_DEBT
                cc_rate = NO_RATE

            has_mortgage = (age > M_DEBT_AGE_MIN) and (np.random.random() < M_DEBT_P)
            if has_mortgage: 
                mortgage_debt = monthly_income * MONTHS_PR_YEAR * np.random.uniform(M_DEBT_INCOME_MIN, M_DEBT_INCOME_MAX)
                max_mortgate = monthly_income * MONTHS_PR_YEAR * M_DEBT_INCOME_CAP
                mortgage_debt = min(mortgage_debt, max_mortgate)
                mortgage_rate = np.random.uniform(M_RATE_MIN, M_RATE_MAX)
            else:
                mortgage_debt = NO_DEBT
                mortgage_rate = NO_RATE
            
            monthly_discretionary = monthly_income * np.random.uniform(MONTHLY_INCOME_DIS_MIN, MONTHLY_INCOME_DIS_MAX)

            personal_data.append({
                'age': age,
                'monthly_income': monthly_income,
                'cc_debt': cc_debt,
                'cc_rate': cc_rate,
                'mortgage_debt': mortgage_debt,
                'mortgage_rate': mortgage_rate,
                'monthly_discretionary': monthly_discretionary
            })

        return pd.DataFrame(personal_data)
    
    def calculate_optimal_strategy(self, market_conditions: pd.DataFrame, personal_profiles: pd.DataFrame) -> pd.DataFrame:
        """
        CORE ML Calculation

        Business logic for defining what "optimal strategy" is financially so we can treat it as an ML problem.
        The results of the model runs is based on achieving the definitions of optimal found here.

        Finetune this to archieve the desired outcome.

        Steps:
        - Step 1: Calculate expected market return
        - Step 2: Decision logic based on debt interest rates
        - Step 3: Finetuning for randomized personal factors
        """

        results = []

        for i in range(len(market_conditions)):
            market = market_conditions.iloc[i]
            person = personal_profiles.iloc[i]

            # Step 1: Market returns
            
            if market['sp500_pe'] > SP500_OVERVALUED:
                market_return = RETURN_BASE - RETURN_ADJUST
            elif market['sp500_pe'] < SP500_UNDERVALUED:
                market_return = RETURN_BASE + RETURN_ADJUST
            else:
                market_return = RETURN_BASE

            risk_premium = market_return - market['treasury_yield']
            adjusted_return = market['treasury_yield'] + risk_premium * RETURN_REALRATIO

            if market['vix'] > VIX_HIGH:
                adjusted_return *= RETURN_VIXRATIO
            
            # Step 2: Investment allocation

            highest_debt_rate = max(person['cc_rate'], person['mortgage_rate'])
            has_no_debt = (person['cc_debt'] == NO_DEBT and person['mortgage_debt'] == NO_DEBT)

            if has_no_debt:
                invest_ratio = INVEST_ALLIN
            elif highest_debt_rate > DEBT_RATE_HIGHEST:  
                invest_ratio = INVEST_CONSERVATIVE
            elif highest_debt_rate > DEBT_RATE_HIGH:  
                if adjusted_return > highest_debt_rate + RETURN_ADJUST:
                    invest_ratio = INVEST_MODERATE 
                else:
                    invest_ratio = INVEST_CONSERVATIVE
            elif highest_debt_rate > DEBT_RATE_MEDIUM: 
                if adjusted_return > highest_debt_rate + RETURN_ADJUST:
                    invest_ratio = INVEST_GROWTH
                else:
                    invest_ratio = INVEST_MODERATE
            else:  
                invest_ratio = INVEST_AGGRESSIVE

            # Step 2.5: Market condition adjustments
            market_adjustment_factor = 1.0

            if market['sp500_pe'] > SP500_OVERVALUED:
                market_adjustment_factor *= OVERVALUED_MULTIPLIER
            elif market['sp500_pe'] < SP500_UNDERVALUED:
                market_adjustment_factor *= UNDERVALUED_MULTIPLIER

            if market['vix'] > VIX_HIGH:
                market_adjustment_factor *= HIGH_VIX_MULTIPLIER

            # Apply market adjustments with dampening for debt-free individuals
            if has_no_debt:
                # Debt-free individuals are less affected by market conditions
                dampened_adjustment = 1.0 + (market_adjustment_factor - 1.0) * DEBT_FREE_DAMPENING
                invest_ratio *= dampened_adjustment
            else:
                invest_ratio *= market_adjustment_factor

            # Step 3: Personal adjustments

            age_factor = max(MIN_AGE_FACTOR, (RETIREMENT_AGE - person['age']) / AGE_RISK_DIVISOR)
            invest_ratio *= age_factor

            if person['monthly_income'] > HIGH_INCOME_THRESHOLD:
                invest_ratio = min(MAX_INVESTMENT_RATIO, invest_ratio * INCOME_BOOST_MULTIPLIER)
            
            invest_ratio = max(MIN_INVESTMENT_RATIO, min(MAX_INVESTMENT_RATIO, invest_ratio))

            results.append({
                'market_return': market_return,
                'adjusted_return': adjusted_return,
                'highest_debt_rate': highest_debt_rate,
                'age_factor': age_factor,
                'recommended_investment_ratio': invest_ratio
            })

        return pd.DataFrame(results)
    
    def generate_complete_dataset(self, n_samples: int = COMPLETE_RUNS) -> pd.DataFrame:
        """
        Generate complete training data with features and labels.
        """

        print(f"Generating {n_samples} financial scenarios...")

        market_data = self.generate_market_conditions(n_samples)
        personal_data = self.generate_personal_profiles(n_samples)
        targets = self.calculate_optimal_strategy(market_data, personal_data)

        complete_data = pd.concat([market_data, personal_data, targets], axis=1)

        print("Dataset generation complete!")
        print(f"Shape: {complete_data.shape}")
        print(f"Features: {list(complete_data.columns)}")
        
        return complete_data
    

########################################
#### Main Run
########################################

if __name__ == "__main__":
    # Create data directory using pathlib
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    generate = FinancialDataGenerator()
    dataset = generate.generate_complete_dataset(COMPLETE_RUNS)

    # Use pathlib for file path
    output_file = data_dir / f'{FIN_TRAIN_DATA_NAME}.csv'
    dataset.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")