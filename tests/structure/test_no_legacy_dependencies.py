"""Guards against canonical code depending on removed legacy source folders."""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "apps" / "backend_api" / "app"
FRONTEND_ROOT = REPO_ROOT / "apps" / "frontend_streamlit"
SHARED_ROOT = REPO_ROOT / "packages" / "furnace-data" / "furnace_data"
LEGACY_NAME = "s" + "rc"


def _python_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def test_removed_frontend_folder_is_absent() -> None:
    assert not (REPO_ROOT / LEGACY_NAME).exists()


def test_canonical_backend_has_no_top_level_app_or_legacy_dependency() -> None:
    forbidden = (
        "from " + "app.",
        "import " + "app.",
        "sys.modules[\"app\"]",
        "sys.modules.setdefault(\"app",
        "furnace-data" + "-service",
        "_ensure_" + LEGACY_NAME + "_package",
        "_ensure_" + LEGACY_NAME + "_path",
        "sys.path",
    )
    findings = []
    for path in _python_files(BACKEND_ROOT):
        text = _text(path)
        for marker in forbidden:
            if marker in text:
                findings.append(f"{path.relative_to(REPO_ROOT).as_posix()}: {marker}")
    assert findings == []


def test_canonical_frontend_has_no_legacy_import_or_bootstrap_dependency() -> None:
    forbidden = (
        "from " + LEGACY_NAME + ".",
        "import " + LEGACY_NAME + ".",
        LEGACY_NAME + ".services",
        LEGACY_NAME + ".config",
        "LEGACY_" + LEGACY_NAME.upper() + "_ROOT",
        "ensure_frontend_legacy_paths",
    )
    findings = []
    for path in _python_files(FRONTEND_ROOT):
        text = _text(path)
        for marker in forbidden:
            if marker in text:
                findings.append(f"{path.relative_to(REPO_ROOT).as_posix()}: {marker}")
        if "sys.path" in text and path.name != "app.py":
            findings.append(f"{path.relative_to(REPO_ROOT).as_posix()}: sys.path")
        if path.name == "app.py" and "sys.path" in text:
            assert "REPO_ROOT = Path(__file__).resolve().parents[2]" in text
            assert "LEGACY_" + LEGACY_NAME.upper() + "_ROOT" not in text
            assert f' / "{LEGACY_NAME}"' not in text
    assert findings == []


def test_frontend_imports_are_canonical_or_shared_package_imports() -> None:
    legacy_roots = {
        "utils",
        "data",
        "domain",
        "agents",
        "config",
        "services",
        "ui",
        "reports",
        "plotters",
        "geometries",
    }
    findings = []
    for path in _python_files(FRONTEND_ROOT):
        tree = ast.parse(_text(path), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".", 1)[0]
                    if root in legacy_roots:
                        findings.append(f"{path.relative_to(REPO_ROOT).as_posix()}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                root = str(node.module or "").split(".", 1)[0]
                if root in legacy_roots:
                    findings.append(f"{path.relative_to(REPO_ROOT).as_posix()}: from {node.module}")
    assert findings == []


def test_shared_config_resolves_packaged_assets() -> None:
    config_text = _text(SHARED_ROOT / "config.py")
    assert "assets" in config_text
    assert "config" in config_text