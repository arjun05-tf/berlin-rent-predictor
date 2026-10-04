"""Configuration management for BerlinRentML."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
CONFIG_DIR = PROJECT_ROOT / "config"

# Ensure directories exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR, CONFIG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


class Config:
    """Configuration container."""

    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value

    def __getitem__(self, key: str) -> Any:
        """Get configuration value."""
        return self.get(key)


def load_config(config_path: Optional[Path] = None) -> Config:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = CONFIG_DIR / "config.yaml"

    if not config_path.exists():
        # Return default configuration
        return Config({})

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return Config(config_dict)


# Default configuration values
DEFAULT_CONFIG = {
    "data": {
        "raw_file": "immo_data.csv",
        "target_column": "baseRent",
        "berlin_filter_column": "regio1",
        "berlin_filter_value": "Berlin",
    },
    "features": {
        "numeric": ["livingSpace", "rooms", "floor", "yearConstructed"],
        "categorical": ["heatingType", "condition", "interiorQual"],
        "boolean": ["hasKitchen", "hasBalcony", "hasGarden", "cellar"],
    },
    "model": {
        "random_state": 42,
        "test_size": 0.2,
        "val_size": 0.1,
    },
    "evaluation": {
        "spatial_holdout_districts": [],  # To be determined after EDA
        "grouped_by": "geo_plz",  # Group by postal code
    },
}
