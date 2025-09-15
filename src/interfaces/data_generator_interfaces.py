########################################
#### Dependendencies
########################################

from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd

########################################
#### Classes
########################################

class MarketDataGenerator(ABC):
    @abstractmethod
    def generate(self, n_samples: int) -> pd.DataFrame:
        pass

class PersonalProfileGenerator(ABC):
    @abstractmethod
    def generate(self, n_samples: int) -> pd.DataFrame:
        pass

class StrategyCalculator(ABC):
    @abstractmethod
    def calculate(self, market_data: pd.DataFrame, personal_data: pd.DataFrame) -> pd.DataFrame:
        pass

class DatasetGenerator(ABC):
    @abstractmethod
    def generate_complete_dataset(self, n_samples: int) -> pd.DataFrame:
        pass