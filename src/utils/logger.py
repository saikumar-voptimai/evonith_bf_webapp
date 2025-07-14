from pathlib import Path
import logging.config
import yaml
import os

def setup_logger(config_path: str = "src/config/logger_setting.yml"):
    """
    Setup logger using the provided YAML configuration file.
    
    Args:
        config_path (str): Path to the YAML configuration file.
    """
    config_file_path = Path(config_path).resolve()
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Logging configuration file not found at {config_file_path}")
    
    with open(config_file_path, "r") as file:
        config = yaml.safe_load(file)
        logging.config.dictConfig(config)

    logger = logging.getLogger("root")
    return logger