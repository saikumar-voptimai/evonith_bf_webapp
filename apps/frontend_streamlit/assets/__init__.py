"""Canonical frontend assets package with a Phase 12 legacy path alias."""

from __future__ import annotations

from pathlib import Path


_LEGACY_ASSETS = Path(__file__).resolve().parents[3] / "src" / "assets"
if _LEGACY_ASSETS.exists():
    _legacy_assets_path = str(_LEGACY_ASSETS)
    if _legacy_assets_path not in __path__:
        __path__.append(_legacy_assets_path)

