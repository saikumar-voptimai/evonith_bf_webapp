"""Guards for the final reduced legacy surface after canonical moves."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
REMOVED_BACKEND_SIDECAR = REPO_ROOT / ("furnace-data" + "-service")
CANONICAL_FRONTEND = REPO_ROOT / "apps" / "frontend_streamlit"
CANONICAL_BACKEND = REPO_ROOT / "apps" / "backend_api" / "app"
CANONICAL_SHARED = REPO_ROOT / "packages" / "furnace-data" / "furnace_data"


def test_old_backend_sidecar_is_absent() -> None:
    assert not REMOVED_BACKEND_SIDECAR.exists()


def test_old_frontend_folder_is_absent() -> None:
    assert not (REPO_ROOT / "src").exists()


def test_canonical_source_roots_exist() -> None:
    assert (CANONICAL_BACKEND / "main.py").exists()
    assert (CANONICAL_FRONTEND / "app.py").exists()
    assert (CANONICAL_FRONTEND / "custom_pages").is_dir()
    assert (CANONICAL_SHARED / "__init__.py").exists()


def test_canonical_backend_imports() -> None:
    from apps.backend_api.app.main import app

    assert app.title == "Evonith BF Backend API"


def test_canonical_frontend_support_imports() -> None:
    from apps.frontend_streamlit.config.page_registry import get_navigation_pages
    from apps.frontend_streamlit.services.status_api import get_status

    assert callable(get_navigation_pages)
    assert callable(get_status)


def test_shared_package_imports_work() -> None:
    from furnace_data.runtime_paths import get_runtime_dir

    assert callable(get_runtime_dir)


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