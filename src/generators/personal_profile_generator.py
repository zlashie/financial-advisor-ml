########################################
#### Dependendencies
########################################
import numpy as np
import pandas as pd
from ..interfaces.data_generator_interfaces import PersonalProfileGenerator
from ..models.config_models import PersonalConfig, DebtConfig, GeneralConfig

########################################
#### Classes
########################################

class StandardPersonalProfileGenerator(PersonalProfileGenerator):
    def __init__(self, personal_config: PersonalConfig, debt_config: DebtConfig, 
                 general_config: GeneralConfig, random_seed: int = None):
        self.personal_config = personal_config
        self.debt_config = debt_config
        self.general_config = general_config
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def generate(self, n_samples: int) -> pd.DataFrame:
        """Generate personal financial profiles."""
        personal_data = []
        
        for _ in range(n_samples):
            age = np.random.randint(
                self.personal_config.work_age_min, 
                self.personal_config.work_age_max
            )
            monthly_income = np.random.lognormal(
                self.personal_config.monthly_income_mean, 
                self.personal_config.monthly_income_stddev
            )
            
            # Credit card debt
            cc_debt, cc_rate = self._generate_credit_card_debt()
            
            # Mortgage debt
            mortgage_debt, mortgage_rate = self._generate_mortgage_debt(age, monthly_income)
            
            monthly_discretionary = monthly_income * np.random.uniform(
                self.personal_config.monthly_income_dis_min,
                self.personal_config.monthly_income_dis_max
            )
            
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
    
    def _generate_credit_card_debt(self) -> tuple[float, float]:
        """Generate credit card debt data."""
        if np.random.random() < self.debt_config.cc_debt_probability:
            debt = np.random.lognormal(
                self.debt_config.cc_debt_mean, 
                self.debt_config.cc_debt_stddev
            )
            rate = np.random.uniform(
                self.debt_config.cc_rate_min, 
                self.debt_config.cc_rate_max
            )
            return debt, rate
        return self.general_config.no_debt, self.general_config.no_rate
    
    def _generate_mortgage_debt(self, age: int, monthly_income: float) -> tuple[float, float]:
        """Generate mortgage debt data."""
        has_mortgage = (
            age > self.personal_config.mortgage_debt_age_min and 
            np.random.random() < self.debt_config.mortgage_debt_probability
        )
        
        if has_mortgage:
            mortgage_debt = monthly_income * self.general_config.months_per_year * np.random.uniform(
                self.debt_config.mortgage_income_min, 
                self.debt_config.mortgage_income_max
            )
            max_mortgage = monthly_income * self.general_config.months_per_year * self.debt_config.mortgage_income_cap
            mortgage_debt = min(mortgage_debt, max_mortgage)
            mortgage_rate = np.random.uniform(
                self.debt_config.mortgage_rate_min, 
                self.debt_config.mortgage_rate_max
            )
            return mortgage_debt, mortgage_rate
        
        return self.general_config.no_debt, self.general_config.no_rate