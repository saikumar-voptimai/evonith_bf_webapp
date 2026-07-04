#!/usr/bin/env python
"""Validate the Phase 12 repository structure and compatibility shims."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REQUIRED_PATHS = [
    "apps/__init__.py",
    "apps/backend_api/__init__.py",
    "apps/backend_api/app/__init__.py",
    "apps/backend_api/app/main.py",
    "apps/frontend_streamlit/__init__.py",
    "apps/frontend_streamlit/app.py",
    "apps/frontend_streamlit/services/status_api.py",
    "apps/frontend_streamlit/custom_pages/1_Welcome.py",
    "furnace-data-service/app/main.py",
    "src/app.py",
    "furnace_data/furnace_data/runtime_paths.py",
    "scripts/export_backend_openapi.py",
    "scripts/check_import_boundaries.py",
    "scripts/check_dependency_profiles.py",
    "scripts/check_backend_minimal_startup.py",
    "scripts/check_frontend_api_imports.py",
    "scripts/edge_start_backend.sh",
    "scripts/edge_start_frontend.sh",
    "docs/migration/phase-12-repository-restructure.md",
    "docs/migration/phase-12-test-execution-report.md",
    "docs/testing/phase-12-testing-guide.md",
    "runtime/.gitkeep",
]


SOURCE_RUNTIME_GLOBS = [
    "apps/**/*.db",
    "apps/**/*.sqlite",
    "apps/**/*.sqlite3",
    "apps/**/*.log",
    "src/**/*.db",
    "src/**/*.sqlite",
    "src/**/*.sqlite3",
    "src/**/*.log",
    "furnace-data-service/app/**/*.db",
    "furnace-data-service/app/**/*.sqlite",
    "furnace-data-service/app/**/*.sqlite3",
    "furnace-data-service/app/**/*.log",
    "furnace-data-service/data/results/**/*",
]


def _read(root: Path, relative_path: str) -> str:
    return (root / relative_path).read_text(encoding="utf-8", errors="ignore")


def _missing_paths(root: Path) -> list[str]:
    return [relative for relative in REQUIRED_PATHS if not (root / relative).exists()]


def _content_failures(root: Path) -> list[str]:
    failures: list[str] = []

    backend_shim = _read(root, "furnace-data-service/app/main.py")
    if "from apps.backend_api.app.main import app, create_app" not in backend_shim:
        failures.append("furnace-data-service/app/main.py must re-export apps.backend_api.app.main")

    frontend_shim = _read(root, "src/app.py")
    if "apps" not in frontend_shim or "frontend_streamlit" not in frontend_shim:
        failures.append("src/app.py must delegate to apps/frontend_streamlit/app.py")

    openapi_script = _read(root, "scripts/export_backend_openapi.py")
    if "apps.backend_api.app.main" not in openapi_script:
        failures.append("scripts/export_backend_openapi.py must use the canonical backend app")

    backend_edge = _read(root, "scripts/edge_start_backend.sh")
    if "apps.backend_api.app.main:app" not in backend_edge:
        failures.append("scripts/edge_start_backend.sh must start apps.backend_api.app.main:app")

    frontend_edge = _read(root, "scripts/edge_start_frontend.sh")
    if "apps/frontend_streamlit/app.py" not in frontend_edge:
        failures.append("scripts/edge_start_frontend.sh must start apps/frontend_streamlit/app.py")

    gitignore = _read(root, ".gitignore")
    for expected in ("runtime/*", "!runtime/.gitkeep", "*.db", "*.sqlite", "*.sqlite3", "*.log"):
        if expected not in gitignore:
            failures.append(f".gitignore missing {expected}")

    pyproject = _read(root, "pyproject.toml")
    if "furnace_data" not in pyproject:
        failures.append("pyproject.toml must keep furnace_data editable/shared package metadata")

    return failures


def _runtime_files_in_source(root: Path) -> list[str]:
    findings: list[str] = []
    for pattern in SOURCE_RUNTIME_GLOBS:
        for path in root.glob(pattern):
            if path.is_file() and ".gitkeep" not in path.name:
                findings.append(str(path.relative_to(root)))
    return sorted(set(findings))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    root = args.root.resolve()

    failures = [f"Missing required path: {path}" for path in _missing_paths(root)]
    failures.extend(_content_failures(root))
    runtime_findings = _runtime_files_in_source(root)

    if failures:
        print("FAIL repository-structure checks")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("PASS repository-structure checks")
    print("canonical_backend=apps/backend_api/app/main.py")
    print("canonical_frontend=apps/frontend_streamlit/app.py")
    if runtime_findings:
        print(
            "WARNING legacy runtime-like files still exist under ignored source paths: "
            + ", ".join(runtime_findings)
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
