import json
import os
from typing import Dict, Any, List


class Config:
    """Configuration manager for the heartbeat classification project."""
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize configuration from JSON file.
        
        Args:
            config_path (str): Path to the configuration JSON file
        """
        self.config_path = config_path
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return json.load(f)
    
    def save_config(self, config_path: str = None) -> None:
        """Save current configuration to JSON file."""
        path = config_path or self.config_path
        with open(path, 'w') as f:
            json.dump(self._config, f, indent=2)
    
    def get(self, key_path: str, default=None):
        """
        Get configuration value using dot notation.
        
        Args:
            key_path (str): Dot-separated path to the configuration value
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Args:
            key_path (str): Dot-separated path to the configuration value
            value: Value to set
        """
        keys = key_path.split('.')
        config = self._config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def update_from_args(self, args) -> None:
        """
        Update configuration with command line arguments.
        
        Args:
            args: Parsed arguments from argparse
        """
        arg_mapping = {
            'batch_size': 'training.batch_size',
            'epochs': 'training.epochs',
            'lr': 'training.learning_rate',
            'sample_rate': 'audio.sample_rate',
            'duration': 'audio.duration',
            'freeze_encoder': 'training.freeze_encoder',
            'multi_label': 'training.multi_label',
            'model_name': 'models.default_model'
        }
        
        for arg_name, config_path in arg_mapping.items():
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                self.set(config_path, getattr(args, arg_name))
    
    # Convenience properties for commonly used values
    @property
    def disease_classes(self) -> List[str]:
        return self.get('dataset.disease_classes')
    
    @property
    def normal_class(self) -> str:
        return self.get('dataset.normal_class')
    
    @property
    def train_csv(self) -> str:
        return self.get('dataset.train_csv')
    
    @property
    def metadata_csv(self) -> str:
        return self.get('dataset.metadata_csv')
    
    @property
    def recording_columns(self) -> List[str]:
        return self.get('dataset.recording_columns')
    
    @property
    def demographics_encoding(self) -> Dict[str, int]:
        return self.get('demographics.encoding')
    
    @property
    def sample_rate(self) -> int:
        return self.get('audio.sample_rate')
    
    @property
    def duration(self) -> float:
        return self.get('audio.duration')
    
    @property
    def batch_size(self) -> int:
        return self.get('training.batch_size')
    
    @property
    def epochs(self) -> int:
        return self.get('training.epochs')
    
    @property
    def learning_rate(self) -> float:
        return self.get('training.learning_rate')
    
    @property
    def test_ratio(self) -> float:
        return self.get('training.test_ratio')
    
    @property
    def random_seed(self) -> int:
        return self.get('training.random_seed')
    
    @property
    def available_models(self) -> List[str]:
        return self.get('models.available_models')
    
    @property
    def default_model(self) -> str:
        return self.get('models.default_model')
    
    @property
    def freeze_encoder(self) -> bool:
        return self.get('training.freeze_encoder')
    
    @property
    def multi_label(self) -> bool:
        return self.get('training.multi_label')
    
    @property
    def dataset_folder(self) -> str:
        return self.get('dataset.dataset_folder')
    
    @property
    def repo_url(self) -> str:
        return self.get('dataset.repo_url')


# Global configuration instance
config = Config()


def load_config(config_path: str = "config.json") -> Config:
    """Load configuration from file."""
    return Config(config_path)


def get_config() -> Config:
    """Get global configuration instance."""
    return config