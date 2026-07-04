"""Phase 12 OpenAPI equivalence between canonical and compatibility backend entrypoints."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVICE_ROOT = REPO_ROOT / "furnace-data-service"


def test_old_and_new_backend_openapi_paths_match(monkeypatch, tmp_path):
    monkeypatch.setenv("EVONITH_RUNTIME_DIR", str(tmp_path / "runtime"))
    if str(SERVICE_ROOT) not in sys.path:
        sys.path.insert(0, str(SERVICE_ROOT))

    canonical = importlib.import_module("apps.backend_api.app.main")
    legacy = importlib.import_module("app.main")

    assert set(canonical.app.openapi()["paths"]) == set(legacy.app.openapi()["paths"])

