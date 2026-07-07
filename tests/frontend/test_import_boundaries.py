"""Boundary checks for the new frontend API-client modules."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_frontend_api_modules_do_not_import_backend_internals():
    service_names = [
        "api_client.py",
        "backend_status.py",
        "auth_api.py",
        "admin_api.py",
        "data_api.py",
        "dataset_api.py",
        "feedback_api.py",
        "material_balance_api.py",
        "recommendations_api.py",
        "blend_optimizer_api.py",
        "copilot_api.py",
        "furnacemind_api.py",
        "status_api.py",
        "ops_api.py",
    ]
    files = [
        *(REPO_ROOT / "apps" / "frontend_streamlit" / "services" / name for name in service_names),
        REPO_ROOT / "apps" / "frontend_streamlit" / "config" / "frontend_settings.py",
    ]
    forbidden = (
        "from " + "app.",
        "import " + "app.",
        "furnace-data" + "-service",
        "influxdb",
        "psycopg",
    )
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert not any(pattern in text for pattern in forbidden)
    copilot_text = (
        REPO_ROOT / "apps" / "frontend_streamlit" / "services" / "copilot_api.py"
    ).read_text(encoding="utf-8").lower()
    assert not any(
        pattern in copilot_text
        for pattern in ("qdrant", "furnacemind", "openai")
    )
    furnacemind_text = (
        REPO_ROOT / "apps" / "frontend_streamlit" / "services" / "furnacemind_api.py"
    ).read_text(encoding="utf-8").lower()
    assert not any(
        pattern in furnacemind_text
        for pattern in ("qdrant", "openai", "psycopg", "sqlite")
    )


def test_backend_app_does_not_import_streamlit():
    for path in (REPO_ROOT / "apps" / "backend_api" / "app").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "import streamlit" not in text
        assert "from streamlit" not in text

