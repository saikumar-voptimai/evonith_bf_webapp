"""Export the FastAPI backend OpenAPI schema without starting a server."""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SERVICE_ROOT = REPO_ROOT / "furnace-data-service"
OUTPUT_PATH = REPO_ROOT / "docs" / "api" / "openapi-v1.json"

for path in (str(REPO_ROOT), str(SERVICE_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

loaded_app = sys.modules.get("app")
loaded_path = str(getattr(loaded_app, "__file__", "")) if loaded_app else ""
if loaded_path.endswith("src\\app.py") or loaded_path.endswith("src/app.py"):
    del sys.modules["app"]

from apps.backend_api.app.main import app


def _assert_compatibility_openapi_paths(schema: dict) -> None:
    """Check the legacy backend shim exposes the same OpenAPI path set."""
    from app.main import app as legacy_app

    legacy_schema = legacy_app.openapi()
    canonical_paths = set(schema.get("paths", {}))
    legacy_paths = set(legacy_schema.get("paths", {}))
    if canonical_paths != legacy_paths:
        missing = sorted(canonical_paths - legacy_paths)
        extra = sorted(legacy_paths - canonical_paths)
        raise RuntimeError(
            "Legacy backend OpenAPI paths differ from canonical app: "
            f"missing={missing} extra={extra}"
        )


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    schema = app.openapi()
    _assert_compatibility_openapi_paths(schema)
    OUTPUT_PATH.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Exported OpenAPI schema to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
