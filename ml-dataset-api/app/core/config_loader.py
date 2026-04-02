import yaml
from pathlib import Path

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def load_config(config_file: str = "setting_ds_dv.yml"):
    """Load configuration from a YAML file."""
    config_file_path = _CONFIG_DIR / config_file
    if not config_file_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_file_path}")
    with open(config_file_path, "r") as file:
        return yaml.safe_load(file)
