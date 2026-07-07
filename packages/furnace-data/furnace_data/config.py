"""Shared YAML configuration loader for Evonith data/domain code.

Resolution order:
1. ``FURNACE_CONFIG_DIR`` when explicitly set by the environment.
2. The packaged config assets at ``furnace_data/assets/config``.
3. ``<cwd>/config`` for external deployments that mount a config directory.

The lookup is evaluated lazily on each call so test and deployment code can set
``FURNACE_CONFIG_DIR`` after import time.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml


_PACKAGE_CONFIG_DIR = Path(__file__).resolve().parent / "assets" / "config"


def get_config_dir() -> Path:
    """Return the active shared config directory."""
    env_dir = os.environ.get("FURNACE_CONFIG_DIR")
    if env_dir:
        return Path(env_dir)
    if _PACKAGE_CONFIG_DIR.is_dir():
        return _PACKAGE_CONFIG_DIR
    return Path.cwd() / "config"


def get_config_path(config_file: str = "setting_ds_dv.yml") -> Path:
    """Return the resolved path for a config file."""
    return get_config_dir() / config_file


def load_config(config_file: str = "setting_ds_dv.yml") -> dict:
    """Load a YAML configuration file from the active shared config directory."""
    config_path = get_config_path(config_file)
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            "Set FURNACE_CONFIG_DIR to the directory containing your YAML config files."
        )
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)