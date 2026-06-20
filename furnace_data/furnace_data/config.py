"""Configuration loader for the furnace_data package.

Searches for YAML config files in this order:
  1. ``FURNACE_CONFIG_DIR`` environment variable (if set)
  2. ``<cwd>/config/`` — works when Streamlit runs from ``src/``
  3. ``<cwd>/src/config/`` — works when running from the repo root

Evaluated lazily at each call so that ``cwd`` is resolved at call time,
not at module import time (which breaks non-editable installs where the
package lives in site-packages and cwd may not yet be the app directory).

The API service (furnace-data-service) sets ``FURNACE_CONFIG_DIR`` at startup
to point at the monorepo's shared ``src/config/`` directory unless explicitly
overridden by the environment.

Usage::

    from furnace_data.config import load_config
    cfg = load_config("setting_ds_dv.yml")
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml


def _find_config_dir() -> Path:
    """Locate the YAML config directory without an env var.

    Evaluated at call time so ``Path.cwd()`` reflects the actual working
    directory of the running process (e.g. ``src/`` under Streamlit).

    Resolution order (first directory that exists wins):
    1. ``<cwd>/config``     — app runs from ``src/`` (Streamlit Cloud & local dev)
    2. ``<cwd>/src/config`` — app runs from the repo root
    """
    candidates = [
        Path.cwd() / "config",
        Path.cwd() / "src" / "config",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]  # let load_config raise a clear error


def load_config(config_file: str = "setting_ds_dv.yml") -> dict:
    """Load a YAML configuration file.

    Args:
        config_file: Filename inside the config directory.

    Returns:
        Parsed YAML as a dict.

    Raises:
        FileNotFoundError: If the config file cannot be found.
    """
    env_dir = os.environ.get("FURNACE_CONFIG_DIR")
    config_dir = Path(env_dir) if env_dir else _find_config_dir()
    config_path = config_dir / config_file
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            "Set FURNACE_CONFIG_DIR to the directory containing your YAML config files."
        )
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)
