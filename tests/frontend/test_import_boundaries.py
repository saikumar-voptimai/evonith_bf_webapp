"""Boundary checks for the new frontend API-client modules."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_frontend_api_modules_do_not_import_backend_internals():
    files = [
        REPO_ROOT / "src" / "services" / "api_client.py",
        REPO_ROOT / "src" / "services" / "backend_status.py",
        REPO_ROOT / "src" / "services" / "data_api.py",
        REPO_ROOT / "src" / "services" / "dataset_api.py",
        REPO_ROOT / "src" / "config" / "frontend_settings.py",
    ]
    forbidden = (
        "from app",
        "import app",
        "furnace-data-service",
        "from furnace-data-service",
        "influxdb",
        "psycopg",
    )
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert not any(pattern in text for pattern in forbidden)


def test_backend_app_does_not_import_streamlit():
    for path in (REPO_ROOT / "furnace-data-service" / "app").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "import streamlit" not in text
        assert "from streamlit" not in text
