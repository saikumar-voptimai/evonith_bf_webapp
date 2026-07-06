"""Phase 12 compatibility shim tests."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_legacy_backend_main_is_thin_shim():
    text = (REPO_ROOT / "furnace-data-service" / "app" / "main.py").read_text(encoding="utf-8")

    assert "temporary Phase 12/cleanup compatibility shim" in text
    assert "from apps.backend_api.app.main import app, create_app" in text
    assert "FastAPI(" not in text
    assert "include_router" not in text
    assert "streamlit" not in text.lower()


def test_legacy_backend_package_is_path_shim():
    text = (REPO_ROOT / "furnace-data-service" / "app" / "__init__.py").read_text(encoding="utf-8")

    assert "temporary Phase 12/cleanup compatibility shim" in text
    assert "apps" in text
    assert "backend_api" in text
    assert "streamlit" not in text.lower()


def test_legacy_frontend_app_is_thin_shim():
    text = (REPO_ROOT / "src" / "app.py").read_text(encoding="utf-8")

    assert "Phase 12 compatibility shim" in text
    assert "apps" in text
    assert "frontend_streamlit" in text
    assert "st.set_page_config" not in text
