import logging
import os
import yaml
from typing import Dict, List, Optional
from enum import Enum
from dataclasses import dataclass
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define constants
CONFIG_FILE = 'config.yaml'
DEFAULT_CONFIG = {
    'model': {
        'type': 'ceiling',
        'threshold': 0.5,
        'learning_rate': 0.01
    },
    'data': {
        'path': '/path/to/data',
        'format': 'csv'
    },
    'logging': {
        'level': 'INFO',
        'file': 'config.log'
    }
}

# Define an Enum for model types
class ModelType(Enum):
    CEILING = 'ceiling'
    FLOOR = 'floor'

# Define a dataclass for model configuration
@dataclass
class ModelConfig:
    type: ModelType
    threshold: float
    learning_rate: float

# Define a dataclass for data configuration
@dataclass
class DataConfig:
    path: str
    format: str

# Define a dataclass for logging configuration
@dataclass
class LoggingConfig:
    level: str
    file: str

# Define a dataclass for the main configuration
@dataclass
class Config:
    model: ModelConfig
    data: DataConfig
    logging: LoggingConfig

# Define a class for configuration management
class ConfigManager:
    def __init__(self, config_file: str = CONFIG_FILE):
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self) -> Config:
        try:
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
                return Config(
                    model=ModelConfig(
                        type=ModelType(config['model']['type']),
                        threshold=config['model']['threshold'],
                        learning_rate=config['model']['learning_rate']
                    ),
                    data=DataConfig(
                        path=config['data']['path'],
                        format=config['data']['format']
                    ),
                    logging=LoggingConfig(
                        level=config['logging']['level'],
                        file=config['logging']['file']
                    )
                )
        except FileNotFoundError:
            logger.warning(f'Config file not found: {self.config_file}')
            return Config(
                model=ModelConfig(
                    type=ModelType.CEILING,
                    threshold=DEFAULT_CONFIG['model']['threshold'],
                    learning_rate=DEFAULT_CONFIG['model']['learning_rate']
                ),
                data=DataConfig(
                    path=DEFAULT_CONFIG['data']['path'],
                    format=DEFAULT_CONFIG['data']['format']
                ),
                logging=LoggingConfig(
                    level=DEFAULT_CONFIG['logging']['level'],
                    file=DEFAULT_CONFIG['logging']['file']
                )
            )
        except yaml.YAMLError as e:
            logger.error(f'Error loading config: {e}')
            return Config(
                model=ModelConfig(
                    type=ModelType.CEILING,
                    threshold=DEFAULT_CONFIG['model']['threshold'],
                    learning_rate=DEFAULT_CONFIG['model']['learning_rate']
                ),
                data=DataConfig(
                    path=DEFAULT_CONFIG['data']['path'],
                    format=DEFAULT_CONFIG['data']['format']
                ),
                logging=LoggingConfig(
                    level=DEFAULT_CONFIG['logging']['level'],
                    file=DEFAULT_CONFIG['logging']['file']
                )
            )

    def save_config(self, config: Config):
        with open(self.config_file, 'w') as f:
            yaml.dump({
                'model': {
                    'type': config.model.type.value,
                    'threshold': config.model.threshold,
                    'learning_rate': config.model.learning_rate
                },
                'data': {
                    'path': config.data.path,
                    'format': config.data.format
                },
                'logging': {
                    'level': config.logging.level,
                    'file': config.logging.file
                }
            }, f, default_flow_style=False)

# Create a ConfigManager instance
config_manager = ConfigManager()

# Get the current configuration
config = config_manager.config

# Print the configuration
logger.info(f'Current configuration: {config}')

# Save the configuration to a file
config_manager.save_config(config)