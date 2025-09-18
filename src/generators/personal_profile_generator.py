########################################
#### Dependencies
########################################
import numpy as np
import pandas as pd
from ..interfaces.data_generator_interfaces import PersonalProfileGenerator
from ..models.config_models import PersonalConfig, DebtConfig, GeneralConfig
from ..config import config

########################################
#### Classes
########################################

class StandardPersonalProfileGenerator(PersonalProfileGenerator):
    def __init__(self, personal_config: PersonalConfig, debt_config: DebtConfig, 
                 general_config: GeneralConfig, random_seed: int = None):
        self.personal_config = personal_config
        self.debt_config = debt_config
        self.general_config = general_config
        
        self.age_config = config.get_section('data_generation', 'personal_profile')['age']
        self.income_config = config.get_section('data_generation', 'personal_profile')['income']
        self.cc_debt_config = config.get_section('data_generation', 'debt')['credit_card']
        self.mortgage_config = config.get_section('data_generation', 'debt')['mortgage']
        self.common_config = config.get_section('common', 'general')
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def generate(self, n_samples: int) -> pd.DataFrame:
        """Generate personal financial profiles."""
        personal_data = []
        
        for _ in range(n_samples):
            age = np.random.randint(
                self.age_config['work_min'], 
                self.age_config['work_max']
            )
            monthly_income = np.random.lognormal(
                self.income_config['normal_distribution']['mean'], 
                self.income_config['normal_distribution']['stddev']
            )
            
            #### Credit card debt ####
            cc_debt, cc_rate = self._generate_credit_card_debt()
            
            #### Mortgage debt ####
            mortgage_debt, mortgage_rate = self._generate_mortgage_debt(age, monthly_income)
            
            monthly_discretionary = monthly_income * np.random.uniform(
                self.income_config['discretionary']['min'],
                self.income_config['discretionary']['max']
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
        if np.random.random() < self.cc_debt_config['probability']:
            debt = np.random.lognormal(
                self.cc_debt_config['normal_distribution']['mean'], 
                self.cc_debt_config['normal_distribution']['stddev']
            )
            rate = np.random.uniform(
                self.cc_debt_config['rate']['min'], 
                self.cc_debt_config['rate']['max']
            )
            return debt, rate
        return self.general_config.no_debt, self.general_config.no_rate
    
    def _generate_mortgage_debt(self, age: int, monthly_income: float) -> tuple[float, float]:
        """Generate mortgage debt data."""
        has_mortgage = (
            age > self.age_config['mortgage_debt_min'] and 
            np.random.random() < self.mortgage_config['probability']
        )
        
        if has_mortgage:
            mortgage_debt = monthly_income * self.common_config['months_per_year'] * np.random.uniform(
                self.mortgage_config['income_multiple']['min'], 
                self.mortgage_config['income_multiple']['max']
            )
            max_mortgage = monthly_income * self.common_config['months_per_year'] * self.mortgage_config['income_multiple']['cap']
            mortgage_debt = min(mortgage_debt, max_mortgage)
            mortgage_rate = np.random.uniform(
                self.mortgage_config['rate']['min'], 
                self.mortgage_config['rate']['max']
            )
            return mortgage_debt, mortgage_rate
        
        return self.general_config.no_debt, self.general_config.no_rate
