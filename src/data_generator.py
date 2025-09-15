########################################
#### Dependendencies
########################################
from pathlib import Path
from .factories.config_factory import ConfigFactory
from .generators.market_data_generator import SP500MarketDataGenerator
from .generators.personal_profile_generator import StandardPersonalProfileGenerator
from .calculators.investment_strategy_calculator import OptimalInvestmentStrategyCalculator
from .generators.financial_dataset_generator import FinancialDatasetGenerator
from .config import config

########################################
#### Main
########################################

def main():
    # Create configurations
    market_config = ConfigFactory.create_market_config()
    general_config = ConfigFactory.create_general_config()
    personal_config = ConfigFactory.create_personal_config()
    debt_config = ConfigFactory.create_debt_config()
    strategy_config = ConfigFactory.create_strategy_config()
    
    # Create generators (Dependency Injection)
    market_generator = SP500MarketDataGenerator(market_config, general_config.default_seed)
    personal_generator = StandardPersonalProfileGenerator(
        personal_config, debt_config, general_config, general_config.default_seed
    )
    strategy_calculator = OptimalInvestmentStrategyCalculator(
        strategy_config, general_config, market_config
    )
    
    # Create main dataset generator
    dataset_generator = FinancialDatasetGenerator(
        market_generator, personal_generator, strategy_calculator
    )
    
    # Generate dataset
    complete_runs = config.get('data_generation', 'output', 'complete_runs')
    dataset = dataset_generator.generate_complete_dataset(complete_runs)
    
    # Save dataset
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    fin_train_data_name = config.get('data_generation', 'output', 'training_data_name')
    output_file = data_dir / f'{fin_train_data_name}.csv'
    dataset.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")

if __name__ == "__main__":
    main()