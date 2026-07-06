from pathlib import Path
import os

import yaml


def load_config(config_file: str = "setting_ds_dv.yml"):
    """
    Load configuration from a YAML file.
    """
    frontend_config_dir = Path(__file__).resolve().parents[1] / "config"
    repo_root = Path(__file__).resolve().parents[3]
    candidate_dirs = [frontend_config_dir]
    configured_dir = os.getenv("FURNACE_CONFIG_DIR", "").strip()
    if configured_dir:
        candidate_dirs.append(Path(configured_dir))
    candidate_dirs.append(repo_root / "src" / "config")

    checked_paths: list[Path] = []
    for config_dir in candidate_dirs:
        config_file_path = (config_dir / config_file).resolve()
        checked_paths.append(config_file_path)
        if config_file_path.is_file():
            with open(config_file_path, "r", encoding="utf-8") as file:
                return yaml.safe_load(file)
    checked = ", ".join(str(path) for path in checked_paths)
    raise FileNotFoundError(f"Configuration file not found in: {checked}")
