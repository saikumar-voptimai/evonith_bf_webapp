"""Canonical backend app package with Phase 12 legacy-module aliases.

Deep backend modules still live under ``furnace-data-service/app`` during this
phase.  The package path is extended here so imports such as
``apps.backend_api.app.services`` can resolve while the old ``app.*`` imports
continue to work.
"""

from __future__ import annotations

from pathlib import Path
import sys


_REPO_ROOT = Path(__file__).resolve().parents[3]
_LEGACY_SERVICE_ROOT = _REPO_ROOT / "furnace-data-service"
_LEGACY_APP_ROOT = _LEGACY_SERVICE_ROOT / "app"

_loaded_app = sys.modules.get("app")
_loaded_path = str(getattr(_loaded_app, "__file__", "")) if _loaded_app else ""
if _loaded_path.endswith("src\\app.py") or _loaded_path.endswith("src/app.py"):
    del sys.modules["app"]

for _path in (str(_LEGACY_SERVICE_ROOT), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

if _LEGACY_APP_ROOT.exists():
    _legacy_app_path = str(_LEGACY_APP_ROOT)
    if _legacy_app_path not in __path__:
        __path__.append(_legacy_app_path)

