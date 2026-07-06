"""Guards for the reduced legacy shim surface after canonical moves."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_BACKEND_APP = REPO_ROOT / "furnace-data-service" / "app"
LEGACY_FRONTEND_SRC = REPO_ROOT / "src"
ROOT_FURNACE_DATA = REPO_ROOT / "furnace_data"


def _files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    )


def _relative_files(root: Path) -> set[str]:
    return {path.relative_to(root).as_posix() for path in _files(root)}


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def test_old_backend_path_contains_only_allowed_shims_or_is_absent() -> None:
    if not LEGACY_BACKEND_APP.exists():
        return

    assert _relative_files(LEGACY_BACKEND_APP) == {"__init__.py", "main.py"}
    assert [
        path
        for path in LEGACY_BACKEND_APP.rglob("*")
        if path.is_dir() and "__pycache__" not in path.parts
    ] == []

    for relative in ("__init__.py", "main.py"):
        path = LEGACY_BACKEND_APP / relative
        text = _text(path)
        assert path.stat().st_size <= 1200
        assert "temporary Phase 12/cleanup compatibility shim" in text
        assert "streamlit" not in text.lower()


def test_no_duplicate_backend_business_logic_under_old_backend_path() -> None:
    if not LEGACY_BACKEND_APP.exists():
        return

    forbidden = ("FastAPI(", "APIRouter(", "include_router", "class ")
    findings = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in _files(LEGACY_BACKEND_APP)
        if any(marker in _text(path) for marker in forbidden)
    ]
    assert findings == []


def test_root_furnace_data_wrapper_contains_only_import_shims_or_is_absent() -> None:
    if not ROOT_FURNACE_DATA.exists():
        return

    assert _relative_files(ROOT_FURNACE_DATA) == {"__init__.py", "runtime_paths.py"}
    for relative in ("__init__.py", "runtime_paths.py"):
        path = ROOT_FURNACE_DATA / relative
        assert path.stat().st_size <= 1200
        assert "temporary Phase 12/cleanup compatibility shim" in _text(path)


def test_old_frontend_src_app_path_contains_only_allowed_shim_or_is_absent() -> None:
    app_path = LEGACY_FRONTEND_SRC / "app.py"
    if not app_path.exists():
        return

    text = _text(app_path)
    assert app_path.stat().st_size <= 1200
    assert "compatibility shim" in text
    assert "apps" in text and "frontend_streamlit" in text
    assert "st.set_page_config" not in text


def test_old_src_services_contains_only_reexport_wrappers_or_is_absent() -> None:
    services = LEGACY_FRONTEND_SRC / "services"
    if not services.exists():
        return

    for path in _files(services):
        relative = path.relative_to(services).as_posix()
        assert "__pycache__" not in path.parts
        assert path.suffix == ".py"
        assert path.stat().st_size <= 700
        if relative == "__init__.py":
            continue
        text = _text(path)
        assert "temporary Phase 12/cleanup compatibility shim" in text
        assert "from apps.frontend_streamlit.services." in text


def test_no_duplicate_frontend_page_logic_under_old_custom_pages_if_moved() -> None:
    custom_pages = LEGACY_FRONTEND_SRC / "custom_pages"
    if not custom_pages.exists():
        return

    for path in _files(custom_pages):
        text = _text(path)
        assert path.suffix == ".py"
        assert path.stat().st_size <= 700
        assert "temporary Phase 12/cleanup compatibility shim" in text
        assert "run_canonical_page(" in text
        assert "st." not in text
        assert "def render" not in text


def test_canonical_backend_imports() -> None:
    from apps.backend_api.app.main import app

    assert app.title == "Evonith BF Backend API"


def test_canonical_frontend_support_imports() -> None:
    from apps.frontend_streamlit.config.page_registry import get_navigation_pages
    from apps.frontend_streamlit.services.status_api import get_status

    assert callable(get_navigation_pages)
    assert callable(get_status)


def test_old_compatibility_imports_work_if_shims_retained() -> None:
    from furnace_data.runtime_paths import get_runtime_dir
    from src.config.page_registry import get_navigation_pages
    from src.services.status_api import get_status

    assert callable(get_runtime_dir)
    assert callable(get_navigation_pages)
    assert callable(get_status)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, 'furnace-data-service'); "
            "from app.main import app; print(app.title)",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Evonith BF Backend API" in result.stdout


def test_repository_structure_check_passes() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_repository_structure.py"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS repository-structure checks" in result.stdout
