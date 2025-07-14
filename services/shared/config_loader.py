import os
import yaml
from typing import Dict, Any
from dotenv import load_dotenv

class ConfigLoader:
    """Centralized configuration loader"""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        load_dotenv()
        self.config_path = config_path
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            # Override with environment variables
            self._override_with_env(config)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
    
    def _override_with_env(self, config: Dict[str, Any]):
        """Override configuration with environment variables"""
        env_mappings = {
            'KAFKA_BROKER': ['kafka', 'broker'],
            'CLICKHOUSE_HOST': ['clickhouse', 'host'],
            'CLICKHOUSE_PORT': ['clickhouse', 'port'],
            'CLICKHOUSE_USER': ['clickhouse', 'user'],
            'CLICKHOUSE_PASSWORD': ['clickhouse', 'password'],
            'SLACK_WEBHOOK': ['alerting', 'slack_webhook']
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value:
                current = config
                for key in config_path[:-1]:
                    current = current.setdefault(key, {})
                current[config_path[-1]] = value
    
    def get(self, key_path: str, default=None):
        """Get configuration value by dot-separated path"""
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section"""
        return self._config.get(section, {})