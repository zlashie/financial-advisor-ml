########################################
#### Dependencies
########################################

from dataclasses import dataclass
from typing import List, Dict, Any

########################################
#### Models
########################################

@dataclass
class MarketValuationConfig:
    """Configuration for market valuation features."""
    price_scoring: Dict[str, float]
    pe_thresholds: Dict[str, float]
    psychology_scoring: Dict[str, float]
    vix_thresholds: Dict[str, float]
    market_conditions: Dict[str, float]

@dataclass
class DebtAnalysisConfig:
    """Configuration for debt analysis features."""
    urgency_thresholds: Dict[str, float]
    urgency_scores: Dict[str, float]

@dataclass
class DemographicsConfig:
    """Configuration for demographic features."""
    age_groups: Dict[str, Any]
    income_groups: Dict[str, Any]

@dataclass
class FeatureEngineeringConfig:
    """Main configuration for feature engineering."""
    market_valuation: MarketValuationConfig
    debt_analysis: DebtAnalysisConfig
    demographics: DemographicsConfig
    market_condition_labels: List[str]
    test_size: float
    random_state: int

@dataclass
class FinancialConstants:
    """Financial constants used in feature engineering."""
    months_per_year: int
    no_debt: float
    no_negative_value: float
    retirement_age: int
    working_years: int
