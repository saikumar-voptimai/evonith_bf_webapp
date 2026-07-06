"""Frontend entrypoint, page registry, and compatibility wrapper tests."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

from apps.frontend_streamlit._legacy import APP_ROOT
from apps.frontend_streamlit.config.frontend_settings import load_frontend_settings
from apps.frontend_streamlit.config.page_registry import get_navigation_pages


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_PAGES = [
    "1_Welcome.py",
    "2_Data_Explorer.py",
    "3_Data_Visualisation.py",
    "4_Recommendations.py",
    "5_AI_Copilot.py",
    "6_Material_Balance.py",
    "7_FurnaceMind.py",
    "8_Feedback.py",
    "9_Blend_Optimizer.py",
]


def test_canonical_frontend_entrypoint_exists():
    app_path = REPO_ROOT / "apps" / "frontend_streamlit" / "app.py"

    assert app_path.exists()
    assert "st.set_page_config" in app_path.read_text(encoding="utf-8")


def test_src_app_compatibility_shim_exists():
    shim_path = REPO_ROOT / "src" / "app.py"
    text = shim_path.read_text(encoding="utf-8")

    assert shim_path.exists()
    assert "apps" in text
    assert "frontend_streamlit" in text
    assert "runpy.run_path" in text
    assert "st.set_page_config" not in text


def test_canonical_app_and_old_app_shim_syntax_check():
    for path in [
        REPO_ROOT / "apps" / "frontend_streamlit" / "app.py",
        REPO_ROOT / "src" / "app.py",
    ]:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_canonical_app_bootstraps_repo_root_when_run_as_script(tmp_path):
    app_path = REPO_ROOT / "apps" / "frontend_streamlit" / "app.py"
    code = f"""
import os
import runpy
import sys
import types
from pathlib import Path

repo = Path({str(REPO_ROOT)!r})
app_path = Path({str(app_path)!r})
os.environ["EVONITH_RUNTIME_DIR"] = {str(tmp_path)!r}
sys.path = [str(app_path.parent)] + [
    path for path in sys.path if path not in ("", str(repo), str(repo.resolve()))
]

class _Context:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False

streamlit = types.ModuleType("streamlit")
streamlit.session_state = {{"auth_user": "tester", "role": "admin"}}
streamlit.set_page_config = lambda **kwargs: None
streamlit.cache_data = lambda *args, **kwargs: (lambda func: func)
streamlit.Page = lambda path, title=None, icon=None: {{"path": path, "title": title, "icon": icon}}
streamlit.navigation = lambda pages: types.SimpleNamespace(run=lambda: None)
streamlit.sidebar = _Context()
streamlit.expander = lambda *args, **kwargs: _Context()
streamlit.success = lambda *args, **kwargs: None
streamlit.warning = lambda *args, **kwargs: None
streamlit.info = lambda *args, **kwargs: None
streamlit.caption = lambda *args, **kwargs: None
streamlit.markdown = lambda *args, **kwargs: None
streamlit.stop = lambda: (_ for _ in ()).throw(SystemExit(0))
sys.modules["streamlit"] = streamlit

runpy.run_path(str(app_path), run_name="__main__")
print("canonical app bootstrap ok")
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "canonical app bootstrap ok" in result.stdout

def test_frontend_services_import_from_new_and_old_paths():
    canonical = importlib.import_module("apps.frontend_streamlit.services.status_api")
    legacy = importlib.import_module("src.services.status_api")

    assert callable(canonical.get_status)
    assert callable(legacy.get_status)


def test_canonical_custom_pages_directory_contains_expected_pages():
    canonical_pages = sorted(
        path.name for path in (REPO_ROOT / "apps" / "frontend_streamlit" / "custom_pages").glob("*.py")
    )

    assert canonical_pages == EXPECTED_PAGES


def test_old_src_custom_pages_wrappers_exist():
    legacy_pages = sorted(path.name for path in (REPO_ROOT / "src" / "custom_pages").glob("*.py"))

    assert legacy_pages == EXPECTED_PAGES
    for page in EXPECTED_PAGES:
        text = (REPO_ROOT / "src" / "custom_pages" / page).read_text(encoding="utf-8")
        assert "run_canonical_page" in text
        assert f"custom_pages/{page}" in text
        assert "streamlit as st" not in text


def test_page_registry_points_to_canonical_pages():
    page_paths = [descriptor.file_path for descriptor in get_navigation_pages()]

    assert page_paths == [
        "custom_pages/1_Welcome.py",
        "custom_pages/2_Data_Explorer.py",
        "custom_pages/3_Data_Visualisation.py",
        "custom_pages/4_Recommendations.py",
        "custom_pages/5_AI_Copilot.py",
        "custom_pages/6_Material_Balance.py",
        "custom_pages/7_FurnaceMind.py",
        "custom_pages/9_Blend_Optimizer.py",
        "custom_pages/8_Feedback.py",
    ]
    for relative_path in page_paths:
        resolved = APP_ROOT / relative_path
        assert resolved.exists()
        assert resolved.is_relative_to(REPO_ROOT / "apps" / "frontend_streamlit" / "custom_pages")


def test_feature_flags_still_parse(monkeypatch):
    monkeypatch.setenv("USE_BACKEND_API", "true")
    monkeypatch.setenv("USE_BACKEND_API_COPILOT", "yes")

    settings = load_frontend_settings()

    assert settings.use_backend_api is True
    assert settings.page_api_flags["copilot"] is True


def test_backend_status_badge_imports():
    module = importlib.import_module("apps.frontend_streamlit.ui.backend_status_badge")

    assert callable(module.render_backend_status_badge)


def test_direct_mode_page_modules_remain_discoverable():
    for page in EXPECTED_PAGES:
        assert (REPO_ROOT / "src" / "custom_pages" / page).exists()
        assert (REPO_ROOT / "apps" / "frontend_streamlit" / "custom_pages" / page).exists()

