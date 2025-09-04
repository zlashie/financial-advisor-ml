########################################
#### Dependendencies
########################################
import os
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List
import random

########################################
#### Constants
########################################

DEFAULT_SEED = 42
NO_DEBT = 0
NO_RATE = 0

MONTHS_PR_YEAR = 12
M_DEBT_AGE_MIN = 25

RETURN_BASE = 0.1
RETURN_ADJUST = 0.02
RETURN_REALRATIO = 0.8
RETURN_VIXRATIO = 0.9

SP500_OVERVALUED = 20
SP500_UNDERVALUED = 12

VIX_HIGH = 30

########################################
#### Variables
########################################

## Generated data name

FIN_TRAIN_DATA_NAME = 'financial_training_data'

## Seed values
SEED_VAL = DEFAULT_SEED

## Number of runs
COMPLETE_RUNS = 1000

## market condition values
SP500_MEAN = 16
SP500_STDDEV = 4
SP500_PE_MIN = 8

TREASURY_MAX_PCT = 8
TREASURY_MIN_PCT = 0.5

VIX_NORMALDIS_MEAN = 2.8
VIX_NORMALDIS_STDDEV = 0.5
VIX_MAX = 80

## personal profile values
WORKAGE_MIN = 22
WORKAGE_MAX = 65
MONTHLY_INCOME_NORMALDIS_MEAN = 10.5
MONTHLY_INCOME_NORMALDIS_STDDEV = 0.6
MONTHLY_INCOME_DIS_MIN = 0.05
MONTHLY_INCOME_DIS_MAX = 0.25

## Credit card debt values
CC_DEBT_P = 0.6
CC_DEBT_NORMALDIS_MEAN = 8.5
CC_DEBT_NORMALDIS_STDDEV = 1.2
CC_RATE_MIN = 0.15
CC_RATE_MAX = 0.25

## Mortgage debt values
M_DEBT_P = 0.4
M_DEBT_INCOME_MIN = 1.5
M_DEBT_INCOME_MAX = 4.5
M_DEBT_INCOME_CAP = 4
M_RATE_MIN = 0.03
M_RATE_MAX = 0.07

## Investment ratios
INVEST_CONSERVATIVE = 0.1
INVEST_MODERATE = 0.3
INVEST_GROWTH = 0.6
INVEST_AGGRESSIVE = 0.8
INVEST_ALLIN = 1

## Debt rate ratios
DEBT_RATE_HIGHEST = 0.20
DEBT_RATE_HIGH = 0.12
DEBT_RATE_MEDIUM = 0.06

## Age-based investment adjustments
RETIREMENT_AGE = 65
AGE_RISK_DIVISOR = 40
MIN_AGE_FACTOR = 0.6

## Income-based investment adjustments
HIGH_INCOME_THRESHOLD = 100000
INCOME_BOOST_MULTIPLIER = 1.15

## Investment ratio bounds
MIN_INVESTMENT_RATIO = 0
MAX_INVESTMENT_RATIO = 1

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