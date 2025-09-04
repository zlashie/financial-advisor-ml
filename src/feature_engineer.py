########################################
#### Dependendencies
########################################
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List
import joblib

########################################
#### Constants
########################################

NOT_RELEVANT = None
MONTHS_PR_YEAR = 12
NO_DEBT = 0
NO_NEG_VAL = 0
TEST_SIZE = 0.2

MARKET_PRICE_EXPENSIVE = -1
MARKET_PRICE_FAIR = 0
MARKET_PRICE_CHEAP = 1

PE_FAIRVALUEMAX = 20
PE_FAIRVALUEMIN = 12

MARKET_PSY_FEARFUL = -1
MARKET_PSY_BALANCED = 0
MARKET_PSY_GREEDY = 1

VIX_BALANCEDMIN = 15
VIX_BALANCEDMAX = 30

MAX_AGE = 65
WORKING_YEARS = 40

HIGH_URGENCY_RATIO = 0.15
MED_URGENCY_RATIO = 0.08

HIGH_URGENCY = 2
MED_URGENCY = 1
LOW_URGENCY = 0

AGE_YOUNG_MAX = 30
AGE_MIDDLE_MAX = 45
AGE_MATURE_MAX = 55
AGE_SENIOR_MAX = 100

AGE_GROUP_YOUNG = 'young'
AGE_GROUP_MIDDLE = 'middle'
AGE_GROUP_MATURE = 'mature'
AGE_GROUP_SENIOR = 'senior'

INCOME_LOW_PERCENTILE = 0.33
INCOME_HIGH_PERCENTILE = 0.67 

INCOME_GROUP_LOW = 'low'
INCOME_GROUP_MEDIUM = 'medium'
INCOME_GROUP_HIGH = 'high'

MARKET_FAVORABLE_THRESHOLD = 0 
MARKET_UNFAVORABLE_THRESHOLD = 0   

MARKET_FAVORABLE = 'favorable'
MARKET_NEUTRAL = 'neutral'
MARKET_UNFAVORABLE = 'unfavorable'

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
        Creatue new features from existing ones. ML domain and context understanding to improve performance.
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
        
        This is a normalization method to supress outlier stretching of ML outcomes and to prevent data leakage.
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
            X, y, test_size=test_size, random_state=42, stratify=None
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