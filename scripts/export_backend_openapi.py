"""Export the FastAPI backend OpenAPI schema without starting a server."""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "packages" / "furnace-data"
OUTPUT_PATH = REPO_ROOT / "docs" / "api" / "openapi-v1.json"

for source_path in (REPO_ROOT, PACKAGE_ROOT):
    source = str(source_path)
    if source not in sys.path:
        sys.path.insert(0, source)

from apps.backend_api.app.main import app


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    schema = app.openapi()
    OUTPUT_PATH.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Exported OpenAPI schema to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

