"""Boundary checks for the new frontend API-client modules."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_frontend_api_modules_do_not_import_backend_internals():
    files = [
        REPO_ROOT / "src" / "services" / "api_client.py",
        REPO_ROOT / "src" / "services" / "backend_status.py",
        REPO_ROOT / "src" / "services" / "auth_api.py",
        REPO_ROOT / "src" / "services" / "admin_api.py",
        REPO_ROOT / "src" / "services" / "data_api.py",
        REPO_ROOT / "src" / "services" / "dataset_api.py",
        REPO_ROOT / "src" / "services" / "feedback_api.py",
        REPO_ROOT / "src" / "services" / "material_balance_api.py",
        REPO_ROOT / "src" / "services" / "recommendations_api.py",
        REPO_ROOT / "src" / "services" / "blend_optimizer_api.py",
        REPO_ROOT / "src" / "services" / "copilot_api.py",
        REPO_ROOT / "src" / "services" / "furnacemind_api.py",
        REPO_ROOT / "src" / "services" / "status_api.py",
        REPO_ROOT / "src" / "services" / "ops_api.py",
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
    copilot_text = (REPO_ROOT / "src" / "services" / "copilot_api.py").read_text(
        encoding="utf-8"
    ).lower()
    assert not any(
        pattern in copilot_text
        for pattern in ("qdrant", "furnacemind", "openai")
    )
    furnacemind_text = (
        REPO_ROOT / "src" / "services" / "furnacemind_api.py"
    ).read_text(encoding="utf-8").lower()
    assert not any(
        pattern in furnacemind_text
        for pattern in ("qdrant", "openai", "psycopg", "sqlite")
    )


def test_backend_app_does_not_import_streamlit():
    for path in (REPO_ROOT / "furnace-data-service" / "app").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "import streamlit" not in text
        assert "from streamlit" not in text
