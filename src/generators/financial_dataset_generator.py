########################################
#### Dependendencies
########################################
import pandas as pd
from ..interfaces.data_generator_interfaces import (
    MarketDataGenerator, PersonalProfileGenerator, StrategyCalculator, DatasetGenerator
)

########################################
#### Classes
########################################

class FinancialDatasetGenerator(DatasetGenerator):
    def __init__(self, 
                 market_generator: MarketDataGenerator,
                 personal_generator: PersonalProfileGenerator,
                 strategy_calculator: StrategyCalculator):
        self.market_generator = market_generator
        self.personal_generator = personal_generator
        self.strategy_calculator = strategy_calculator
    
    def generate_complete_dataset(self, n_samples: int) -> pd.DataFrame:
        """Generate complete training dataset."""
        print(f"Generating {n_samples} financial scenarios...")
        
        market_data = self.market_generator.generate(n_samples)
        personal_data = self.personal_generator.generate(n_samples)
        targets = self.strategy_calculator.calculate(market_data, personal_data)
        
        complete_data = pd.concat([market_data, personal_data, targets], axis=1)
        
        print("Dataset generation complete!")
        print(f"Shape: {complete_data.shape}")
        print(f"Features: {list(complete_data.columns)}")
        
        return complete_data