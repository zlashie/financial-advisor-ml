########################################
#### Dependencies
########################################

from typing import List
from ..interfaces.feature_engineering_interfaces import FeatureEngineeringPipeline
from ..feature_engineering.debt_feature_creator import DebtFeatureCreator
from ..feature_engineering.market_feature_creator import MarketFeatureCreator
from ..feature_engineering.risk_feature_creator import RiskFeatureCreator
from ..feature_engineering.categorical_feature_preprocessor import CategoricalFeaturePreprocessor
from ..feature_engineering.robust_feature_scaler import RobustFeatureScaler
from ..feature_engineering.train_test_splitter import TrainTestSplitter
from ..feature_engineering.financial_feature_engineering_pipeline import FinancialFeatureEngineeringPipeline
from ..models.feature_engineering_models import FeatureEngineeringConfig, FinancialConstants

########################################
#### Classes
########################################

class FeatureEngineeringFactory:
    """Factory for creating feature engineering components."""
    
    @staticmethod
    def create_feature_engineering_pipeline(
        config: FeatureEngineeringConfig,
        financial_constants: FinancialConstants
    ) -> FeatureEngineeringPipeline:
        """Create a complete feature engineering pipeline."""
        
        # Create feature creators
        feature_creators = [
            DebtFeatureCreator(config.debt_analysis, financial_constants),
            MarketFeatureCreator(config.market_valuation),
            RiskFeatureCreator(financial_constants)
        ]
        
        # Create feature preprocessor
        feature_preprocessor = CategoricalFeaturePreprocessor(
            config.demographics,
            config.market_valuation,
            config.market_condition_labels
        )
        
        # Create feature scaler
        feature_scaler = RobustFeatureScaler()
        
        # Create dataset splitter
        dataset_splitter = TrainTestSplitter(config.test_size, config.random_state)
        
        # Create main pipeline
        return FinancialFeatureEngineeringPipeline(
            feature_creators=feature_creators,
            feature_preprocessor=feature_preprocessor,
            feature_scaler=feature_scaler,
            dataset_splitter=dataset_splitter
        )
