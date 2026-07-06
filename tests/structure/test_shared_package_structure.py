"""Shared furnace_data package relocation and compatibility tests."""

from __future__ import annotations

import importlib
from pathlib import Path
import subprocess
import sys

from apps.backend_api.app.main import app


REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_PACKAGE = REPO_ROOT / "packages" / "furnace-data" / "furnace_data"
ROOT_COMPAT_PACKAGE = REPO_ROOT / "furnace_data"


def test_canonical_shared_package_exists():
    assert CANONICAL_PACKAGE.is_dir()
    assert (CANONICAL_PACKAGE / "__init__.py").exists()
    assert (REPO_ROOT / "packages" / "furnace-data" / "pyproject.toml").exists()


def test_import_furnace_data_exposes_canonical_package_path():
    import furnace_data

    package_paths = [Path(path).resolve() for path in furnace_data.__path__]

    assert CANONICAL_PACKAGE.resolve() in package_paths


def test_import_runtime_paths_uses_canonical_implementation():
    module = importlib.import_module("furnace_data.runtime_paths")

    assert Path(module.__file__).resolve() == (CANONICAL_PACKAGE / "runtime_paths.py").resolve()
    assert callable(module.get_runtime_dir)


def test_import_relational_package_works_if_present():
    if not (CANONICAL_PACKAGE / "relational").exists():
        return

    relational = importlib.import_module("furnace_data.relational")

    assert relational.__name__ == "furnace_data.relational"


def test_backend_app_imports_after_shared_package_move():
    assert app.title
    assert "/api/v1/health" in app.openapi()["paths"]


def test_openapi_export_script_works():
    result = subprocess.run(
        [sys.executable, "scripts/export_backend_openapi.py"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Exported OpenAPI schema" in result.stdout


def test_root_compatibility_package_contains_only_shims():
    allowed = {
        ROOT_COMPAT_PACKAGE / "__init__.py",
        ROOT_COMPAT_PACKAGE / "runtime_paths.py",
    }
    python_files = {
        path
        for path in ROOT_COMPAT_PACKAGE.rglob("*.py")
        if "__pycache__" not in path.parts
    }

    assert python_files == allowed
    for path in allowed:
        assert "compatibility shim" in path.read_text(encoding="utf-8")


def test_runtime_paths_has_only_one_real_implementation():
    root_shim = ROOT_COMPAT_PACKAGE / "runtime_paths.py"
    canonical = CANONICAL_PACKAGE / "runtime_paths.py"

    assert canonical.exists()
    assert root_shim.exists()
    assert "Shared runtime data paths" in canonical.read_text(encoding="utf-8")
    assert "compatibility shim" in root_shim.read_text(encoding="utf-8")
