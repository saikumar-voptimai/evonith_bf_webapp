"""Configuration loader for the furnace_data package.

Searches for YAML config files in this order:
  1. ``FURNACE_CONFIG_DIR`` environment variable (if set)
  2. ``<repo_root>/src/config/`` — works when package is installed editably
     from the evonith_webapp repo root.

The API service (furnace-data-service) sets ``FURNACE_CONFIG_DIR`` in its
``.env`` to point at ``furnace-data-service/config/``.

Usage::

    from furnace_data.config import load_config
    cfg = load_config("setting_ds_dv.yml")
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

# Default: repo_root/src/config/ (works for editable install at repo root)
_DEFAULT_CONFIG_DIR: Path = Path(__file__).resolve().parent.parent / "src" / "config"


def load_config(config_file: str = "setting_ds_dv.yml") -> dict:
    """Load a YAML configuration file.

    Args:
        config_file: Filename inside the config directory.

    Returns:
        Parsed YAML as a dict.

    Raises:
        FileNotFoundError: If the config file cannot be found.
    """
    config_dir = Path(os.environ.get("FURNACE_CONFIG_DIR", str(_DEFAULT_CONFIG_DIR)))
    config_path = config_dir / config_file
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            "Set FURNACE_CONFIG_DIR to the directory containing your YAML config files."
        )
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)
