"""Phase 12 frontend entrypoint and adapter tests."""

from __future__ import annotations

import importlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_canonical_frontend_entrypoint_exists():
    app_path = REPO_ROOT / "apps" / "frontend_streamlit" / "app.py"

    assert app_path.exists()
    assert "st.set_page_config" in app_path.read_text(encoding="utf-8")


def test_frontend_services_import_from_new_and_old_paths():
    canonical = importlib.import_module("apps.frontend_streamlit.services.status_api")
    legacy = importlib.import_module("src.services.status_api")

    assert callable(canonical.get_status)
    assert callable(legacy.get_status)


def test_frontend_page_wrappers_cover_legacy_pages():
    canonical_pages = sorted((REPO_ROOT / "apps" / "frontend_streamlit" / "custom_pages").glob("*.py"))
    legacy_pages = sorted((REPO_ROOT / "src" / "custom_pages").glob("*.py"))

    assert [path.name for path in canonical_pages] == [path.name for path in legacy_pages]

