"""temporary Phase 12/cleanup compatibility shim for apps.backend_api.app."""

from __future__ import annotations

from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CANONICAL_APP_ROOT = _REPO_ROOT / "apps" / "backend_api" / "app"

if _CANONICAL_APP_ROOT.exists():
    _canonical_app_path = str(_CANONICAL_APP_ROOT)
    if _canonical_app_path not in __path__:
        __path__.append(_canonical_app_path)
