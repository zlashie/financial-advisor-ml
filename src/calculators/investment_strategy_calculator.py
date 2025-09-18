########################################
#### Dependencies
########################################
import pandas as pd
from ..interfaces.data_generator_interfaces import StrategyCalculator
from ..models.config_models import StrategyConfig, GeneralConfig, MarketConfig
from ..config import config

########################################
#### Class
########################################

class OptimalInvestmentStrategyCalculator(StrategyCalculator):
    def __init__(self, strategy_config: StrategyConfig, general_config: GeneralConfig, 
                 market_config: MarketConfig):
        self.strategy_config = strategy_config
        self.general_config = general_config
        self.market_config = market_config
        
        # Load configuration values
        self.returns_config = config.get_section('data_generation', 'investment_strategy')['returns']
        self.allocation_config = config.get_section('data_generation', 'investment_strategy')['allocation_ratios']
        self.debt_thresholds = config.get_section('data_generation', 'investment_strategy')['debt_rate_thresholds']
        self.age_config = config.get_section('data_generation', 'investment_strategy')['age_adjustments']
        self.income_boost_config = config.get_section('data_generation', 'investment_strategy')['income_boost']
        self.market_adjustments = config.get_section('data_generation', 'investment_strategy')['market_adjustments']
        self.bounds_config = config.get_section('data_generation', 'investment_strategy')['bounds']
    
    def calculate(self, market_data: pd.DataFrame, personal_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate optimal investment strategy."""
        results = []
        
        for i in range(len(market_data)):
            market = market_data.iloc[i]
            person = personal_data.iloc[i]
            
            # Step 1: Calculate market returns
            market_return = self._calculate_market_return(market)
            adjusted_return = self._calculate_adjusted_return(market, market_return)
            
            # Step 2: Calculate investment allocation
            invest_ratio = self._calculate_base_investment_ratio(person, adjusted_return)
            
            # Step 3: Apply market adjustments
            invest_ratio = self._apply_market_adjustments(invest_ratio, market, person)
            
            # Step 4: Apply personal adjustments
            invest_ratio = self._apply_personal_adjustments(invest_ratio, person)
            
            results.append({
                'market_return': market_return,
                'adjusted_return': adjusted_return,
                'highest_debt_rate': max(person['cc_rate'], person['mortgage_rate']),
                'age_factor': self._calculate_age_factor(person['age']),
                'recommended_investment_ratio': invest_ratio
            })
        
        return pd.DataFrame(results)
    
    def _calculate_market_return(self, market: pd.Series) -> float:
        """Calculate base market return based on P/E ratio."""
        overvalued_threshold = config.get('data_generation', 'market_conditions', 'sp500', 'overvalued_threshold')
        undervalued_threshold = config.get('data_generation', 'market_conditions', 'sp500', 'undervalued_threshold')
        
        if market['sp500_pe'] > overvalued_threshold:
            return self.returns_config['base'] - self.returns_config['adjustment']
        elif market['sp500_pe'] < undervalued_threshold:
            return self.returns_config['base'] + self.returns_config['adjustment']
        else:
            return self.returns_config['base']
    
    def _calculate_adjusted_return(self, market: pd.Series, market_return: float) -> float:
        """Calculate risk-adjusted return."""
        risk_premium = market_return - market['treasury_yield']
        adjusted_return = market['treasury_yield'] + risk_premium * self.returns_config['real_ratio']
        
        vix_high_threshold = config.get('data_generation', 'market_conditions', 'vix', 'high_threshold')
        if market['vix'] > vix_high_threshold:
            adjusted_return *= self.returns_config['vix_ratio']
        
        return adjusted_return
    
    def _calculate_base_investment_ratio(self, person: pd.Series, adjusted_return: float) -> float:
        """Calculate base investment ratio based on debt situation."""
        highest_debt_rate = max(person['cc_rate'], person['mortgage_rate'])
        has_no_debt = (person['cc_debt'] == self.general_config.no_debt and 
                      person['mortgage_debt'] == self.general_config.no_debt)
        
        if has_no_debt:
            return self.allocation_config['all_in']
        elif highest_debt_rate > self.debt_thresholds['highest']:
            return self.allocation_config['conservative']
        elif highest_debt_rate > self.debt_thresholds['high']:
            if adjusted_return > highest_debt_rate + self.returns_config['adjustment']:
                return self.allocation_config['moderate']
            else:
                return self.allocation_config['conservative']
        elif highest_debt_rate > self.debt_thresholds['medium']:
            if adjusted_return > highest_debt_rate + self.returns_config['adjustment']:
                return self.allocation_config['growth']
            else:
                return self.allocation_config['moderate']
        else:
            return self.allocation_config['aggressive']
    
    def _apply_market_adjustments(self, invest_ratio: float, market: pd.Series, person: pd.Series) -> float:
        """Apply market condition adjustments."""
        market_adjustment_factor = 1.0
        
        overvalued_threshold = config.get('data_generation', 'market_conditions', 'sp500', 'overvalued_threshold')
        undervalued_threshold = config.get('data_generation', 'market_conditions', 'sp500', 'undervalued_threshold')
        vix_high_threshold = config.get('data_generation', 'market_conditions', 'vix', 'high_threshold')
        
        if market['sp500_pe'] > overvalued_threshold:
            market_adjustment_factor *= (1.0 - self.market_adjustments['valuation_impact']['overvalued_reduction'])
        elif market['sp500_pe'] < undervalued_threshold:
            market_adjustment_factor *= (1.0 + self.market_adjustments['valuation_impact']['undervalued_boost'])
        
        if market['vix'] > vix_high_threshold:
            market_adjustment_factor *= (1.0 - self.market_adjustments['volatility_impact']['high_vix_reduction'])
        
        # Apply dampening for debt-free individuals
        has_no_debt = (person['cc_debt'] == self.general_config.no_debt and 
                      person['mortgage_debt'] == self.general_config.no_debt)
        
        if has_no_debt:
            dampened_adjustment = 1.0 + (market_adjustment_factor - 1.0) * self.market_adjustments['debt_free_dampening']
            invest_ratio *= dampened_adjustment
        else:
            invest_ratio *= market_adjustment_factor
        
        return invest_ratio
    
    def _apply_personal_adjustments(self, invest_ratio: float, person: pd.Series) -> float:
        """Apply personal factor adjustments."""
        # Age factor
        age_factor = self._calculate_age_factor(person['age'])
        invest_ratio *= age_factor
        
        # Income boost
        if person['monthly_income'] > self.income_boost_config['threshold']:
            invest_ratio = min(
                self.bounds_config['max_investment_ratio'], 
                invest_ratio * self.income_boost_config['multiplier']
            )
        
        # Apply bounds
        invest_ratio = max(
            self.bounds_config['min_investment_ratio'], 
            min(self.bounds_config['max_investment_ratio'], invest_ratio)
        )
        
        return invest_ratio
    
    def _calculate_age_factor(self, age: int) -> float:
        """Calculate age-based risk factor."""
        return max(
            self.age_config['min_age_factor'], 
            (self.general_config.retirement_age - age) / self.age_config['risk_divisor']
        )
