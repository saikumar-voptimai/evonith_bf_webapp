"""Write the twin-api OpenAPI schema for the frontend's typed client.

    python scripts/export_openapi.py          # writes frontend/src/lib/api/openapi.json
    pnpm --dir frontend api:types             # regenerates frontend/src/lib/api/schema.d.ts from it

CI runs both and fails when the committed files differ (schema drift).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "furnace_data")]

from api.main import create_app  # noqa: E402
from api.settings import ApiSettings  # noqa: E402

OUT = ROOT / "frontend" / "src" / "lib" / "api" / "openapi.json"


def schema() -> dict:
    return create_app(ApiSettings(static_dir=None)).openapi()


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(schema(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
