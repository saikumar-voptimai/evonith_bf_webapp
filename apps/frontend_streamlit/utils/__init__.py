"""Canonical frontend utility package with a Phase 12 legacy path alias."""

from __future__ import annotations

from pathlib import Path


_LEGACY_UTILS = Path(__file__).resolve().parents[3] / "src" / "utils"
if _LEGACY_UTILS.exists():
    _legacy_utils_path = str(_LEGACY_UTILS)
    if _legacy_utils_path not in __path__:
        __path__.append(_legacy_utils_path)

