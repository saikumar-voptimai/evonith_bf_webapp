"""Phase 12 frontend API-adapter boundary tests."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVICE_DIRS = [
    REPO_ROOT / "apps" / "frontend_streamlit" / "services",
    REPO_ROOT / "src" / "services",
]
FORBIDDEN = (
    "from app.",
    "import app.",
    "furnace-data-service",
    "from furnace-data-service",
    "influxdb",
    "psycopg",
    "qdrant_client",
    "openai",
)


def test_frontend_api_adapters_do_not_import_backend_internals():
    for services_dir in SERVICE_DIRS:
        for path in services_dir.glob("*_api.py"):
            text = path.read_text(encoding="utf-8").lower()
            assert not any(pattern in text for pattern in FORBIDDEN), path
