########################################
#### Dependendencies
########################################
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List
import joblib

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import config  

########################################
#### Load Configuration Values
########################################

# General constants
NOT_RELEVANT = None
MONTHS_PR_YEAR = config.get('common', 'general', 'months_per_year')
NO_DEBT = config.get('common', 'financial_constants', 'no_debt')
NO_NEG_VAL = config.get('common', 'financial_constants', 'no_negative_value')
TEST_SIZE = config.get('common', 'general', 'test_size')
MAX_AGE = config.get('common', 'financial_constants', 'retirement_age')
WORKING_YEARS = config.get('common', 'financial_constants', 'working_years')

# Market valuation configuration
market_config = config.get('feature_engineering', 'market_valuation')

# Price scoring
price_scoring = market_config['price_scoring']
MARKET_PRICE_EXPENSIVE = price_scoring['expensive']
MARKET_PRICE_FAIR = price_scoring['fair']
MARKET_PRICE_CHEAP = price_scoring['cheap']

# P/E thresholds
pe_thresholds = market_config['pe_thresholds']
PE_FAIRVALUEMAX = pe_thresholds['fair_value_max']
PE_FAIRVALUEMIN = pe_thresholds['fair_value_min']

# Psychology scoring
psychology_scoring = market_config['psychology_scoring']
MARKET_PSY_FEARFUL = psychology_scoring['fearful']
MARKET_PSY_BALANCED = psychology_scoring['balanced']
MARKET_PSY_GREEDY = psychology_scoring['greedy']

# VIX thresholds
vix_thresholds = market_config['vix_thresholds']
VIX_BALANCEDMIN = vix_thresholds['balanced_min']
VIX_BALANCEDMAX = vix_thresholds['balanced_max']

# Market conditions
market_conditions = market_config['market_conditions']
MARKET_FAVORABLE_THRESHOLD = market_conditions['favorable_threshold']
MARKET_UNFAVORABLE_THRESHOLD = market_conditions['unfavorable_threshold']

# Debt analysis configuration
debt_config = config.get('feature_engineering', 'debt_analysis')

# Urgency thresholds
urgency_thresholds = debt_config['urgency_thresholds']
HIGH_URGENCY_RATIO = urgency_thresholds['high_ratio']
MED_URGENCY_RATIO = urgency_thresholds['medium_ratio']

# Urgency scores
urgency_scores = debt_config['urgency_scores']
HIGH_URGENCY = urgency_scores['high']
MED_URGENCY = urgency_scores['medium']
LOW_URGENCY = urgency_scores['low']

# Demographics configuration
demographics_config = config.get('feature_engineering', 'demographics')

# Age groups
age_groups = demographics_config['age_groups']
AGE_YOUNG_MAX = age_groups['young_max']
AGE_MIDDLE_MAX = age_groups['middle_max']
AGE_MATURE_MAX = age_groups['mature_max']
AGE_SENIOR_MAX = age_groups['senior_max']

# Age group labels
age_labels = age_groups['labels']
AGE_GROUP_YOUNG = age_labels[0]
AGE_GROUP_MIDDLE = age_labels[1]
AGE_GROUP_MATURE = age_labels[2]
AGE_GROUP_SENIOR = age_labels[3]

# Income groups
income_groups = demographics_config['income_groups']
income_percentiles_config = income_groups['percentiles']
INCOME_LOW_PERCENTILE = income_percentiles_config['low']
INCOME_HIGH_PERCENTILE = income_percentiles_config['high']

# Income group labels
income_labels = income_groups['labels']
INCOME_GROUP_LOW = income_labels[0]
INCOME_GROUP_MEDIUM = income_labels[1]
INCOME_GROUP_HIGH = income_labels[2]

# Market condition labels
market_condition_labels = config.get('feature_engineering', 'market_condition_labels')
MARKET_FAVORABLE = market_condition_labels[0]
MARKET_NEUTRAL = market_condition_labels[1]
MARKET_UNFAVORABLE = market_condition_labels[2]

########################################
#### Classes
########################################

class FinancialFeatureEngineer:
    """
    Simple Pipeline to prepare and transform raw financial data into ML-ready features.
    """

    def __init__(self):
        self.scaler = NOT_RELEVANT
        self.feature_columns = NOT_RELEVANT
        self.is_fitted = NOT_RELEVANT

    def create_engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features from existing ones. ML domain and context understanding to improve performance.
        """

        df_eng = df.copy()

        print("Creating engineered features...")

        # 1. Debt-to-Income Ratios
        df_eng['total_debt'] = df_eng['cc_debt'] + df_eng['mortgage_debt']
        df_eng['annual_income'] = df_eng['monthly_income'] * MONTHS_PR_YEAR
        df_eng['debt_to_income_ratio'] = df_eng['total_debt'] / df_eng['annual_income']

        df_eng['debt_to_income_ratio'] = df_eng['debt_to_income_ratio'].fillna(NO_DEBT)

        # 2. Weighted Average Debt Rate
        df_eng['weighted_debt_rate'] = np.where(
            df_eng['total_debt'] > NO_DEBT,
            (df_eng['cc_debt'] * df_eng['cc_rate'] + 
                df_eng['mortgage_debt'] * df_eng['mortgage_rate']) / df_eng['total_debt'],
            NO_DEBT
        )

        # 3. Market Valuation Score
        pe_score = np.where(df_eng['sp500_pe'] > PE_FAIRVALUEMAX, MARKET_PRICE_EXPENSIVE,  
                np.where(df_eng['sp500_pe'] < PE_FAIRVALUEMIN, MARKET_PRICE_CHEAP, MARKET_PRICE_FAIR)
                )  
        
        vix_score = np.where(df_eng['vix'] > VIX_BALANCEDMAX, MARKET_PSY_FEARFUL,  
                    np.where(df_eng['vix'] < VIX_BALANCEDMIN, MARKET_PSY_GREEDY, MARKET_PSY_BALANCED)
                    )  
        
        df_eng['market_attractiveness'] = pe_score + vix_score

        # 4. Age-based Risk Capacity
        df_eng['years_to_retirement'] = np.maximum(MAX_AGE - df_eng['age'], NO_NEG_VAL)
        df_eng['risk_capacity'] = df_eng['years_to_retirement'] / WORKING_YEARS
            
        # 5. Financial Flexibility Score
        df_eng['discretionary_ratio'] = df_eng['monthly_discretionary'] / df_eng['monthly_income']
        
        # 6. Debt Urgency Score
        df_eng['debt_urgency'] = np.where(
            df_eng['weighted_debt_rate'] > HIGH_URGENCY_RATIO, HIGH_URGENCY,  
            np.where(df_eng['weighted_debt_rate'] > MED_URGENCY_RATIO, MED_URGENCY, LOW_URGENCY)  
        )

        print(f"Created {len(df_eng.columns) - len(df.columns)} new features")
        return df_eng
    
    def prepare_feature_for_ml(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for ML model run.
        """

        df_ml = df.copy()

        # 1. Create categorical features from continuous ones

        df_ml['age_group'] = pd.cut(df_ml['age'], 
                             bins=[0, AGE_YOUNG_MAX, AGE_MIDDLE_MAX, AGE_MATURE_MAX, AGE_SENIOR_MAX], 
                             labels=[AGE_GROUP_YOUNG, AGE_GROUP_MIDDLE, AGE_GROUP_MATURE, AGE_GROUP_SENIOR]
                            )
        
        income_percentiles = df_ml['annual_income'].quantile([INCOME_LOW_PERCENTILE, INCOME_HIGH_PERCENTILE])
        df_ml['income_group'] = pd.cut(df_ml['annual_income'],
                                bins=[0, income_percentiles.iloc[0], 
                                income_percentiles.iloc[1], float('inf')],
                                labels=[INCOME_GROUP_LOW, INCOME_GROUP_MEDIUM, INCOME_GROUP_HIGH]
                            )
        
        df_ml['market_condition'] = np.where(
                                    df_ml['market_attractiveness'] > MARKET_FAVORABLE_THRESHOLD, MARKET_FAVORABLE,
                                    np.where(df_ml['market_attractiveness'] < MARKET_UNFAVORABLE_THRESHOLD, 
                                    MARKET_UNFAVORABLE, MARKET_NEUTRAL)
                            )

        # 2. One-hot encode categorical variables

        categorical_columns = ['age_group', 'income_group', 'market_condition']

        for col in categorical_columns:
            dummies = pd.get_dummies(df_ml[col], prefix=col, drop_first=True)
            df_ml = pd.concat([df_ml, dummies], axis=1)
            df_ml.drop(col, axis=1, inplace=True)
    
        print(f"After categorical encoding: {df_ml.shape[1]} features")
        return df_ml
    
    def scale_features(self, df: pd.DataFrame, fit_scaler: bool = True) -> pd.DataFrame:
        """
        Scale numerical features to similar ranges. 
        
        This is a normalization method to suppress outlier stretching of ML outcomes and to prevent data leakage.
        """

        feature_cols = [col for col in df.columns if col not in ['recommended_investment_ratio', 'expected_return']]
        numerical_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

        df_scaled = df.copy()

        if fit_scaler:
            self.scaler = RobustScaler()
            self.feature_columns = numerical_cols
            df_scaled[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
            self.is_fitted = True
            print(f"Fitted scaler on {len(numerical_cols)} numerical features")
        else:
            if not self.is_fitted:
                raise ValueError("Scaler not fitted yet. Call with fit_scaler=True first.")
            df_scaled[self.feature_columns] = self.scaler.transform(df[self.feature_columns])
            print("Applied existing scaler to features")

        return df_scaled
    
    def create_ml_dataset(self, raw_data: pd.DataFrame,
                            test_size: float = TEST_SIZE) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Complete feature engineering pipeline.

        Returns: X_train, X_test, y_train, y_test
        """

        print("Starting complete feature engineering pipeline...")

        # Step 1: Create engineered features
        df_engineered = self.create_engineer_features(raw_data)

        # Step 2: Prepare for ML
        df_ml_ready = self.prepare_feature_for_ml(df_engineered)

        # Step 3: Separate features and target
        target_col = 'recommended_investment_ratio'
        feature_cols = [col for col in df_ml_ready.columns if col != target_col]

        X = df_ml_ready[feature_cols]
        y = df_ml_ready[target_col]

        print(f"Final feature set: {X.shape[1]} features")
        print(f"Target variable: {target_col}")

        # Step 4: Train/test split BEFORE scaling
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=config.get('common', 'general', 'random_state'), stratify=None
        )

        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")

        # Step 5: Scale features (fit on train, apply to both)
        X_train_scaled = self.scale_features(X_train, fit_scaler=True)
        X_test_scaled = self.scale_features(X_test, fit_scaler=False)

        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def save_preprocessing_state(self, filepath: str):
        """
        Save the fitted scaler and feature columns.
        """
        if not self.is_fitted:
            raise ValueError("Feature engineer not fitted yet")
        
        # Create directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        preprocessing_state = {
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
            
        joblib.dump(preprocessing_state, filepath)
        print(f"Preprocessing state saved to {filepath}")

    def load_preprocessing_state(self, filepath: str):
        """
        Load previously saved preprocessing state.
        """
        preprocessing_state = joblib.load(filepath)
        self.scaler = preprocessing_state['scaler']  
        self.feature_columns = preprocessing_state['feature_columns']  
        self.is_fitted = True
        print(f"Preprocessing state loaded from {filepath}")