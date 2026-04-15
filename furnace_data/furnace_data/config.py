"""Configuration loader for the furnace_data package.

Searches for YAML config files in this order:
  1. ``FURNACE_CONFIG_DIR`` environment variable (if set)
  2. ``<cwd>/config/`` — works when Streamlit runs from ``src/``
  3. ``<cwd>/src/config/`` — works when running from the repo root
  4. ``../../src/config/`` relative to this file — legacy editable-install path

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

def _find_default_config_dir() -> Path:
    """Locate the YAML config directory without an env var.

    Resolution order (first directory that exists wins):
    1. ``<cwd>/config``       — app runs from ``src/`` (Streamlit Cloud & local dev)
    2. ``<cwd>/src/config``   — app runs from the repo root
    3. ``../../src/config``   — legacy editable-install path relative to this file
    """
    candidates = [
        Path.cwd() / "config",
        Path.cwd() / "src" / "config",
        Path(__file__).resolve().parent.parent / "src" / "config",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]  # let load_config raise a clear error


_DEFAULT_CONFIG_DIR: Path = _find_default_config_dir()


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
