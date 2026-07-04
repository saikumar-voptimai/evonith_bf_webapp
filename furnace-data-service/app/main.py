"""Phase 12 compatibility shim for the old backend entrypoint.

Use ``apps.backend_api.app.main:app`` for new startup commands.  This module
keeps ``uvicorn app.main:app`` working from ``furnace-data-service``.
"""

from __future__ import annotations

from pathlib import Path
import sys


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path[:0] = [str(_REPO_ROOT)]

from apps.backend_api.app.main import app, create_app

__all__ = ["app", "create_app"]
