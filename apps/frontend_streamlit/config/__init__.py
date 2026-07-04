"""Canonical frontend config package with a Phase 12 legacy path alias."""

from __future__ import annotations

from pathlib import Path


_LEGACY_CONFIG = Path(__file__).resolve().parents[3] / "src" / "config"
if _LEGACY_CONFIG.exists():
    _legacy_config_path = str(_LEGACY_CONFIG)
    if _legacy_config_path not in __path__:
        __path__.append(_legacy_config_path)

