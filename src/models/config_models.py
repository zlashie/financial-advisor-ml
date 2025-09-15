########################################
#### Dependendencies
########################################

from dataclasses import dataclass
from typing import Dict, Any

########################################
#### Classes
########################################

@dataclass
class MarketConfig:
    sp500_mean: float
    sp500_stddev: float
    sp500_pe_min: float
    sp500_overvalued: float
    sp500_undervalued: float
    treasury_max_pct: float
    treasury_min_pct: float
    vix_normal_mean: float
    vix_normal_stddev: float
    vix_max: float
    vix_high: float

@dataclass
class PersonalConfig:
    work_age_min: int
    work_age_max: int
    mortgage_debt_age_min: int
    monthly_income_mean: float
    monthly_income_stddev: float
    monthly_income_dis_min: float
    monthly_income_dis_max: float
    high_income_threshold: float

@dataclass
class DebtConfig:
    cc_debt_probability: float
    cc_debt_mean: float
    cc_debt_stddev: float
    cc_rate_min: float
    cc_rate_max: float
    mortgage_debt_probability: float
    mortgage_income_min: float
    mortgage_income_max: float
    mortgage_income_cap: float
    mortgage_rate_min: float
    mortgage_rate_max: float

@dataclass
class StrategyConfig:
    return_base: float
    return_adjust: float
    return_real_ratio: float
    return_vix_ratio: float
    invest_conservative: float
    invest_moderate: float
    invest_growth: float
    invest_aggressive: float
    invest_all_in: float
    debt_rate_highest: float
    debt_rate_high: float
    debt_rate_medium: float
    age_risk_divisor: float
    min_age_factor: float
    income_boost_multiplier: float
    min_investment_ratio: float
    max_investment_ratio: float
    overvalued_reduction: float
    undervalued_boost: float
    high_vix_reduction: float
    debt_free_dampening: float

@dataclass
class GeneralConfig:
    default_seed: int
    months_per_year: int
    no_debt: float
    no_rate: float
    retirement_age: int