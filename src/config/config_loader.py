import yaml
from pathlib import Path

def load_config(config_file: str = "setting_ds_dv.yml"):
    """
    Load configuration from a YAML file.
    """
    fullpath = Path(__file__).resolve().parents[1] / "config" / config_file
    config_file_path = Path(fullpath).resolve()
    if not config_file_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_file_path}")
    with open(config_file_path, "r") as file:
        return yaml.safe_load(file)