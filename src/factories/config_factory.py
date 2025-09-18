########################################
#### Dependencies
########################################
from ..models.config_models import (
    MarketConfig, 
    PersonalConfig, 
    DebtConfig, 
    StrategyConfig, 
    GeneralConfig
)

from ..config import config

from ..models import (
    MarketValuationConfig,
    DebtAnalysisConfig,
    DemographicsConfig
)

from .feature_engineering_factory import (
    FeatureEngineeringConfig,
    FinancialConstants
)

########################################
#### Classes
########################################

class ConfigFactory:
    @staticmethod
    def create_market_config() -> MarketConfig:
        vix_config = config.get('data_generation', 'market_conditions', 'vix')
        return MarketConfig(
            sp500_mean=config.get('data_generation', 'market_conditions', 'sp500', 'mean'),
            sp500_stddev=config.get('data_generation', 'market_conditions', 'sp500', 'stddev'),
            sp500_pe_min=config.get('data_generation', 'market_conditions', 'sp500', 'pe_min'),
            sp500_overvalued=config.get('data_generation', 'market_conditions', 'sp500', 'overvalued_threshold'),
            sp500_undervalued=config.get('data_generation', 'market_conditions', 'sp500', 'undervalued_threshold'),
            treasury_max_pct=config.get('data_generation', 'market_conditions', 'treasury', 'max_pct'),
            treasury_min_pct=config.get('data_generation', 'market_conditions', 'treasury', 'min_pct'),
            vix_normal_mean=vix_config['normal_distribution']['mean'],
            vix_normal_stddev=vix_config['normal_distribution']['stddev'],
            vix_max=vix_config['max'],
            vix_high=vix_config['high_threshold']
        )
    
    @staticmethod
    def create_general_config() -> GeneralConfig:
        return GeneralConfig(
            default_seed=config.get('common', 'general', 'default_seed'),
            months_per_year=config.get('common', 'general', 'months_per_year'),
            no_debt=config.get('common', 'financial_constants', 'no_debt'),
            no_rate=config.get('common', 'financial_constants', 'no_rate'),
            retirement_age=config.get('common', 'financial_constants', 'retirement_age')
        )
    
    @staticmethod
    def create_personal_config() -> PersonalConfig:
        personal_config = config.get('data_generation', 'personal_profile')
        income_config = personal_config['income']
        
        return PersonalConfig(
            work_age_min=personal_config['age']['work_min'],
            work_age_max=personal_config['age']['work_max'],
            mortgage_debt_age_min=personal_config['age']['mortgage_debt_min'],
            monthly_income_mean=income_config['normal_distribution']['mean'],
            monthly_income_stddev=income_config['normal_distribution']['stddev'],
            monthly_income_dis_min=income_config['discretionary']['min'],
            monthly_income_dis_max=income_config['discretionary']['max'],
            high_income_threshold=income_config['high_threshold']
        )
    
    @staticmethod
    def create_debt_config() -> DebtConfig:
        debt_config = config.get('data_generation', 'debt')
        cc_debt_config = debt_config['credit_card']
        m_debt_config = debt_config['mortgage']
        
        return DebtConfig(
            cc_debt_probability=cc_debt_config['probability'],
            cc_debt_mean=cc_debt_config['normal_distribution']['mean'],
            cc_debt_stddev=cc_debt_config['normal_distribution']['stddev'],
            cc_rate_min=cc_debt_config['rate']['min'],
            cc_rate_max=cc_debt_config['rate']['max'],
            mortgage_debt_probability=m_debt_config['probability'],
            mortgage_income_min=m_debt_config['income_multiple']['min'],
            mortgage_income_max=m_debt_config['income_multiple']['max'],
            mortgage_income_cap=m_debt_config['income_multiple']['cap'],
            mortgage_rate_min=m_debt_config['rate']['min'],
            mortgage_rate_max=m_debt_config['rate']['max']
        )
    
    @staticmethod
    def create_strategy_config() -> StrategyConfig:
        strategy_config = config.get('data_generation', 'investment_strategy')
        returns_config = strategy_config['returns']
        allocation_config = strategy_config['allocation_ratios']
        debt_thresholds = strategy_config['debt_rate_thresholds']
        age_adjustments = strategy_config['age_adjustments']
        bounds_config = strategy_config['bounds']
        market_adjustments = strategy_config.get('market_adjustments', {})
        valuation_config = market_adjustments.get('valuation_impact', {})
        volatility_config = market_adjustments.get('volatility_impact', {})
        income_boost_config = strategy_config.get('income_boost', {})
        
        return StrategyConfig(
            return_base=returns_config['base'],
            return_adjust=returns_config['adjustment'],
            return_real_ratio=returns_config['real_ratio'],
            return_vix_ratio=returns_config['vix_ratio'],
            invest_conservative=allocation_config['conservative'],
            invest_moderate=allocation_config['moderate'],
            invest_growth=allocation_config['growth'],
            invest_aggressive=allocation_config['aggressive'],
            invest_all_in=allocation_config['all_in'],
            debt_rate_highest=debt_thresholds['highest'],
            debt_rate_high=debt_thresholds['high'],
            debt_rate_medium=debt_thresholds['medium'],
            age_risk_divisor=age_adjustments['risk_divisor'],
            min_age_factor=age_adjustments['min_age_factor'],
            income_boost_multiplier=income_boost_config.get('multiplier', 1.15),
            min_investment_ratio=bounds_config['min_investment_ratio'],
            max_investment_ratio=bounds_config['max_investment_ratio'],
            overvalued_reduction=valuation_config.get('overvalued_reduction', 0.15),
            undervalued_boost=valuation_config.get('undervalued_boost', 0.15),
            high_vix_reduction=volatility_config.get('high_vix_reduction', 0.10),
            debt_free_dampening=market_adjustments.get('debt_free_dampening', 0.5)
        )
    
    @staticmethod
    def create_feature_engineering_config() -> FeatureEngineeringConfig:
        """Create feature engineering configuration."""
        config_data = config.get('feature_engineering')
        
        market_valuation = MarketValuationConfig(**config_data['market_valuation'])
        debt_analysis = DebtAnalysisConfig(**config_data['debt_analysis'])
        demographics = DemographicsConfig(**config_data['demographics'])
        
        return FeatureEngineeringConfig(
            market_valuation=market_valuation,
            debt_analysis=debt_analysis,
            demographics=demographics,
            market_condition_labels=config_data['market_condition_labels'],
            test_size=config.get('common', 'general', 'test_size'),
            random_state=config.get('common', 'general', 'random_state')
        )

    @staticmethod
    def create_financial_constants() -> FinancialConstants:
        """Create financial constants."""
        return FinancialConstants(
            months_per_year=config.get('common', 'general', 'months_per_year'),
            no_debt=config.get('common', 'financial_constants', 'no_debt'),
            no_negative_value=config.get('common', 'financial_constants', 'no_negative_value'),
            retirement_age=config.get('common', 'financial_constants', 'retirement_age'),
            working_years=config.get('common', 'financial_constants', 'working_years')
        )
