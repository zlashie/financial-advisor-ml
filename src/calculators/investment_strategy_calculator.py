########################################
#### Dependendencies
########################################
import pandas as pd
from ..interfaces.data_generator_interfaces import StrategyCalculator
from ..models.config_models import StrategyConfig, GeneralConfig, MarketConfig

########################################
#### Class
########################################

class OptimalInvestmentStrategyCalculator(StrategyCalculator):
    def __init__(self, strategy_config: StrategyConfig, general_config: GeneralConfig, 
                 market_config: MarketConfig):
        self.strategy_config = strategy_config
        self.general_config = general_config
        self.market_config = market_config
    
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
        if market['sp500_pe'] > self.market_config.sp500_overvalued:
            return self.strategy_config.return_base - self.strategy_config.return_adjust
        elif market['sp500_pe'] < self.market_config.sp500_undervalued:
            return self.strategy_config.return_base + self.strategy_config.return_adjust
        else:
            return self.strategy_config.return_base
    
    def _calculate_adjusted_return(self, market: pd.Series, market_return: float) -> float:
        """Calculate risk-adjusted return."""
        risk_premium = market_return - market['treasury_yield']
        adjusted_return = market['treasury_yield'] + risk_premium * self.strategy_config.return_real_ratio
        
        if market['vix'] > self.market_config.vix_high:
            adjusted_return *= self.strategy_config.return_vix_ratio
        
        return adjusted_return
    
    def _calculate_base_investment_ratio(self, person: pd.Series, adjusted_return: float) -> float:
        """Calculate base investment ratio based on debt situation."""
        highest_debt_rate = max(person['cc_rate'], person['mortgage_rate'])
        has_no_debt = (person['cc_debt'] == self.general_config.no_debt and 
                      person['mortgage_debt'] == self.general_config.no_debt)
        
        if has_no_debt:
            return self.strategy_config.invest_all_in
        elif highest_debt_rate > self.strategy_config.debt_rate_highest:
            return self.strategy_config.invest_conservative
        elif highest_debt_rate > self.strategy_config.debt_rate_high:
            if adjusted_return > highest_debt_rate + self.strategy_config.return_adjust:
                return self.strategy_config.invest_moderate
            else:
                return self.strategy_config.invest_conservative
        elif highest_debt_rate > self.strategy_config.debt_rate_medium:
            if adjusted_return > highest_debt_rate + self.strategy_config.return_adjust:
                return self.strategy_config.invest_growth
            else:
                return self.strategy_config.invest_moderate
        else:
            return self.strategy_config.invest_aggressive
    
    def _apply_market_adjustments(self, invest_ratio: float, market: pd.Series, person: pd.Series) -> float:
        """Apply market condition adjustments."""
        market_adjustment_factor = 1.0
        
        if market['sp500_pe'] > self.market_config.sp500_overvalued:
            market_adjustment_factor *= (1.0 - self.strategy_config.overvalued_reduction)
        elif market['sp500_pe'] < self.market_config.sp500_undervalued:
            market_adjustment_factor *= (1.0 + self.strategy_config.undervalued_boost)
        
        if market['vix'] > self.market_config.vix_high:
            market_adjustment_factor *= (1.0 - self.strategy_config.high_vix_reduction)
        
        # Apply dampening for debt-free individuals
        has_no_debt = (person['cc_debt'] == self.general_config.no_debt and 
                      person['mortgage_debt'] == self.general_config.no_debt)
        
        if has_no_debt:
            dampened_adjustment = 1.0 + (market_adjustment_factor - 1.0) * self.strategy_config.debt_free_dampening
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
        if person['monthly_income'] > self.strategy_config.income_boost_multiplier:
            invest_ratio = min(
                self.strategy_config.max_investment_ratio, 
                invest_ratio * self.strategy_config.income_boost_multiplier
            )
        
        # Apply bounds
        invest_ratio = max(
            self.strategy_config.min_investment_ratio, 
            min(self.strategy_config.max_investment_ratio, invest_ratio)
        )
        
        return invest_ratio
    
    def _calculate_age_factor(self, age: int) -> float:
        """Calculate age-based risk factor."""
        return max(
            self.strategy_config.min_age_factor, 
            (self.general_config.retirement_age - age) / self.strategy_config.age_risk_divisor
        )