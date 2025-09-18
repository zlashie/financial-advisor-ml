########################################
#### Dependencies
########################################
import json
import os
from pathlib import Path
from typing import Dict, Any, Union

########################################
#### Class
########################################
class ConfigLoader:
    """
    Centralized configuration loader.
    """
    
    def __init__(self, config_dir: Union[str, Path] = None):
        if config_dir is None:
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent 
            self.config_dir = project_root / 'config'
        else:
            self.config_dir = Path(config_dir)
        
        self._configs = {}
        self._load_all_configs()
    
    def _load_all_configs(self):
        """Load all JSON configuration files."""
        config_files = {
            'common': 'common.json',
            'data_generation': 'data_generation.json', 
            'feature_engineering': 'feature_engineering.json',
            'models': 'models.json',
            'paths': 'paths.json'
        }
        
        for config_name, filename in config_files.items():
            config_path = self.config_dir / filename
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    self._configs[config_name] = json.load(f)
                print(f"Loaded {config_name} configuration from {filename}")
            else:
                print(f"Warning: Configuration file {filename} not found")
                self._configs[config_name] = {}
    
    def get(self, config_type: str, *keys) -> Any:
        """
        Get configuration value using dot notation.
        
        Examples:
        - config.get('common', 'general', 'default_seed')
        - config.get('data_generation', 'market_conditions', 'sp500', 'mean')
        - config.get('paths', 'data', 'base_directory')
        """
        try:
            config = self._configs[config_type]
            for key in keys:
                config = config[key]
            return config
        except KeyError as e:
            raise KeyError(f"Configuration key not found: {config_type}.{'.'.join(keys)}")
    
    def get_section(self, config_type: str, section: str = None) -> Dict[str, Any]:
        """Get entire configuration section."""
        if section is None:
            return self._configs.get(config_type, {})
        return self._configs.get(config_type, {}).get(section, {})
    
    def reload_config(self, config_type: str):
        """Reload a specific configuration file."""
        config_files = {
            'common': 'common.json',
            'data_generation': 'data_generation.json',
            'feature_engineering': 'feature_engineering.json', 
            'models': 'models.json',
            'paths': 'paths.json'
        }
        
        if config_type in config_files:
            config_path = self.config_dir / config_files[config_type]
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    self._configs[config_type] = json.load(f)
                print(f"Reloaded {config_type} configuration")

config = ConfigLoader()
