"""Frontend entrypoint and page registry tests."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

from apps.frontend_streamlit.config.frontend_settings import load_frontend_settings
from apps.frontend_streamlit.config.page_registry import get_navigation_pages


REPO_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = REPO_ROOT / "apps" / "frontend_streamlit"
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
    app_path = APP_ROOT / "app.py"

    assert app_path.exists()
    assert "st.set_page_config" in app_path.read_text(encoding="utf-8")


def test_legacy_frontend_entrypoint_is_absent():
    assert not (REPO_ROOT / "src").exists()


def test_canonical_app_syntax_check():
    app_path = APP_ROOT / "app.py"
    compile(app_path.read_text(encoding="utf-8"), str(app_path), "exec")


def test_canonical_app_runs_with_repo_root_on_python_path(tmp_path):
    app_path = APP_ROOT / "app.py"
    code = f"""
import os
import runpy
import sys
import types
from pathlib import Path

repo = Path({str(REPO_ROOT)!r})
app_path = Path({str(app_path)!r})
os.environ["EVONITH_RUNTIME_DIR"] = {str(tmp_path)!r}
sys.path = [str(repo), str(app_path.parent)] + [
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
print("canonical app package context ok")
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "canonical app package context ok" in result.stdout


def test_frontend_services_import_from_canonical_paths():
    canonical = importlib.import_module("apps.frontend_streamlit.services.status_api")

    assert callable(canonical.get_status)


def test_canonical_custom_pages_directory_contains_expected_pages():
    canonical_pages = sorted(path.name for path in (APP_ROOT / "custom_pages").glob("*.py"))

    assert canonical_pages == EXPECTED_PAGES


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
        assert resolved.is_relative_to(APP_ROOT / "custom_pages")


def test_feature_flags_still_parse(monkeypatch):
    monkeypatch.setenv("USE_BACKEND_API", "true")
    monkeypatch.setenv("USE_BACKEND_API_COPILOT", "yes")

    settings = load_frontend_settings()

    assert settings.use_backend_api is True
    assert settings.page_api_flags["copilot"] is True


def test_backend_status_badge_imports():
    module = importlib.import_module("apps.frontend_streamlit.ui.backend_status_badge")

    assert callable(module.render_backend_status_badge)


def test_page_modules_remain_discoverable():
    for page in EXPECTED_PAGES:
        assert (APP_ROOT / "custom_pages" / page).exists()