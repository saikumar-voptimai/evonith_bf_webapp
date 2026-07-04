"""Canonical frontend UI package with a Phase 12 legacy path alias."""

from __future__ import annotations

from pathlib import Path


_LEGACY_UI = Path(__file__).resolve().parents[3] / "src" / "ui"
if _LEGACY_UI.exists():
    _legacy_ui_path = str(_LEGACY_UI)
    if _legacy_ui_path not in __path__:
        __path__.append(_legacy_ui_path)

