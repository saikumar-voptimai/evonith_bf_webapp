#!/usr/bin/env python
"""Validate canonical structure and generated-artifact cleanup."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


LEGACY_FRONTEND_DIR = "s" + "rc"

REQUIRED_PATHS = [
    "apps/__init__.py",
    "apps/backend_api/__init__.py",
    "apps/backend_api/app/__init__.py",
    "apps/backend_api/app/main.py",
    "apps/backend_api/app/api",
    "apps/backend_api/app/core",
    "apps/backend_api/app/services",
    "apps/backend_api/app/repositories",
    "apps/backend_api/app/tasks",
    "apps/frontend_streamlit/__init__.py",
    "apps/frontend_streamlit/app.py",
    "apps/frontend_streamlit/assets",
    "apps/frontend_streamlit/config",
    "apps/frontend_streamlit/custom_pages",
    "apps/frontend_streamlit/services/status_api.py",
    "apps/frontend_streamlit/ui",
    "packages/furnace-data/pyproject.toml",
    "packages/furnace-data/furnace_data/__init__.py",
    "packages/furnace-data/furnace_data/runtime_paths.py",
    "packages/furnace-data/furnace_data/assets/__init__.py",
    "packages/furnace-data/furnace_data/assets/models",
    "packages/furnace-data/furnace_data/assets/copilot_analysis",
    "packages/furnace-data/furnace_data/assets/furnacemind",
    "scripts/export_backend_openapi.py",
    "scripts/check_import_boundaries.py",
    "scripts/check_dependency_profiles.py",
    "scripts/check_backend_minimal_startup.py",
    "scripts/check_frontend_api_imports.py",
    "scripts/edge_start_backend.sh",
    "scripts/edge_start_frontend.sh",
    "docs/migration/post-phase-13-structure-cleanup-plan.md",
    "docs/operations/model-assets.md",
    "tests",
    "tests/README.md",
    "tests/backend",
    "tests/dependency",
    "tests/deployment",
    "tests/fixtures",
    "tests/frontend",
    "tests/integration",
    "tests/structure",
    "runtime",
    "runtime/.gitkeep",
]

SOURCE_ROOTS = [
    "apps",
    "packages",
    "scripts",
]

GENERATED_SUFFIXES = {
    ".csv",
    ".db",
    ".log",
    ".sqlite",
    ".sqlite3",
}

OBSOLETE_PATHS = [
    "ale" + "mbic",
    "ale" + "mbic.ini",
    "bad-test.exe",
    "phase6-test.txt",
    "phase9_test_doc.txt",
    "gw_config.json",
    "main.py",
    "run_time.txt",
    "static",
    "scripts/diagnose_fetch_pipeline.py",
    "scripts/diagnose_fetch_pipeline.report.md",
    "scripts/validate_slag_balance.py",
    "scripts/slag_validation_results.csv",
]


def _read(root: Path, relative_path: str) -> str:
    path = root / relative_path
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def _missing_paths(root: Path) -> list[str]:
    required_paths = list(REQUIRED_PATHS)
    if (root / "infra").exists():
        required_paths.append("infra")
    return [relative for relative in required_paths if not (root / relative).exists()]


def _is_generated_cache_path(path: Path) -> bool:
    return "__pycache__" in path.parts or path.suffix.lower() == ".pyc"


def _python_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [path for path in root.rglob("*.py") if "__pycache__" not in path.parts]


def _canonical_backend_dependency_failures(root: Path) -> list[str]:
    backend_root = root / "apps" / "backend_api" / "app"
    forbidden_markers = {
        "from " + "app.": "imports the legacy top-level app package",
        "import " + "app.": "imports the legacy top-level app package",
        "sys.modules[\"app\"]": "registers a top-level app alias",
        "sys.modules.setdefault(\"app": "registers a top-level app alias",
        "furnace-data" + "-service": "references the removed backend folder",
        "_ensure_" + LEGACY_FRONTEND_DIR + "_package": "loads code dynamically from removed frontend folder",
        "_ensure_" + LEGACY_FRONTEND_DIR + "_path": "loads code dynamically from removed frontend folder",
        "sys.path": "mutates sys.path from canonical backend code",
    }
    failures: list[str] = []
    for path in _python_files(backend_root):
        rel = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8", errors="ignore")
        for marker, reason in forbidden_markers.items():
            if marker in text:
                failures.append(f"{rel} {reason}: {marker}")
    return failures


def _canonical_frontend_dependency_failures(root: Path) -> list[str]:
    frontend_root = root / "apps" / "frontend_streamlit"
    forbidden_markers = (
        "from " + LEGACY_FRONTEND_DIR + ".",
        "import " + LEGACY_FRONTEND_DIR + ".",
        LEGACY_FRONTEND_DIR + ".services",
        LEGACY_FRONTEND_DIR + ".config",
        "LEGACY_" + LEGACY_FRONTEND_DIR.upper() + "_ROOT",
        "ensure_frontend_legacy_paths",
    )
    failures: list[str] = []
    for path in _python_files(frontend_root):
        rel = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8", errors="ignore")
        for marker in forbidden_markers:
            if marker in text:
                failures.append(f"{rel} references removed frontend path/bootstrap: {marker}")
        if "sys.path" in text:
            if path.name != "app.py":
                failures.append(f"{rel} references frontend path/bootstrap: sys.path")
            elif "REPO_ROOT = Path(__file__).resolve().parents[2]" not in text:
                failures.append(f"{rel} must only bootstrap the repository root")
    return failures


def _content_failures(root: Path) -> list[str]:
    failures: list[str] = []
    if (root / LEGACY_FRONTEND_DIR).exists():
        failures.append(f"Removed legacy frontend folder must not exist: {LEGACY_FRONTEND_DIR}")

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
    for expected in (
        "runtime/*",
        "!runtime/.gitkeep",
        "*.db",
        "*.sqlite",
        "*.sqlite3",
        "*.log",
        "__pycache__/",
        "*.pyc",
        "scripts/*.csv",
        "scripts/*.report.md",
    ):
        if expected not in gitignore:
            failures.append(f".gitignore missing {expected}")

    pyproject = _read(root, "pyproject.toml")
    if 'furnace_data = { path = "./packages/furnace-data", editable = true }' not in pyproject:
        failures.append("pyproject.toml must point furnace_data to ./packages/furnace-data")
    if 'testpaths = ["tests"]' not in pyproject:
        failures.append("pyproject.toml must point pytest discovery at canonical tests/")

    shared_package = _read(root, "packages/furnace-data/pyproject.toml")
    if 'name = "furnace_data"' not in shared_package:
        failures.append("packages/furnace-data/pyproject.toml must define the furnace_data package")
    if '"furnace_data/assets" = "furnace_data/assets"' not in shared_package:
        failures.append("packages/furnace-data/pyproject.toml must include shared package assets")

    for relative in ("pyproject.toml", "requirements.txt", "uv.lock"):
        text = _read(root, relative).lower()
        if "ale" + "mbic" in text:
            failures.append(f"{relative} must not include legacy migration tooling")
    return failures


def _obsolete_path_failures(root: Path) -> list[str]:
    return [
        f"Obsolete generated/clutter path must be removed: {relative}"
        for relative in OBSOLETE_PATHS
        if (root / relative).exists()
    ]


def _tracked_bytecode_findings(root: Path) -> list[str]:
    if not (root / ".git").exists():
        return []
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=root,
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        return []
    findings: list[str] = []
    for line in result.stdout.splitlines():
        normalized = line.replace("\\", "/")
        if "/__pycache__/" in normalized or normalized.endswith(".pyc"):
            findings.append(normalized)
    return sorted(findings)


def _generated_artifact_findings(root: Path) -> list[str]:
    findings: list[str] = []
    for relative_root in SOURCE_ROOTS:
        source_root = root / relative_root
        if not source_root.exists():
            continue
        for path in source_root.rglob("*"):
            if path.is_dir() and path.name.lower() == "uploads":
                findings.append(str(path.relative_to(root)))
            elif path.is_file() and path.suffix.lower() in GENERATED_SUFFIXES:
                findings.append(str(path.relative_to(root)))
    findings.extend(_tracked_bytecode_findings(root))
    return sorted(set(findings))


def _runtime_ignore_failures(root: Path) -> list[str]:
    failures: list[str] = []
    gitignore_lines = {
        line.strip()
        for line in _read(root, ".gitignore").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if "runtime/*" not in gitignore_lines:
        failures.append(".gitignore missing runtime/*")
    if "!runtime/.gitkeep" not in gitignore_lines:
        failures.append(".gitignore missing !runtime/.gitkeep")

    if (root / ".git").exists():
        ignored_runtime_file = subprocess.run(
            ["git", "check-ignore", "-q", "runtime/generated-probe.db"],
            cwd=root,
            check=False,
        )
        if ignored_runtime_file.returncode != 0:
            failures.append("runtime/generated-probe.db should be ignored by git")

        ignored_gitkeep = subprocess.run(
            ["git", "check-ignore", "-q", "runtime/.gitkeep"],
            cwd=root,
            check=False,
        )
        if ignored_gitkeep.returncode == 0:
            failures.append("runtime/.gitkeep must not be ignored by git")

    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--generated-only",
        action="store_true",
        help="Only run generated-artifact checks for fixture tests.",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()

    failures: list[str] = []
    if not args.generated_only:
        failures.extend(f"Missing required path: {path}" for path in _missing_paths(root))
        failures.extend(_content_failures(root))
        failures.extend(_obsolete_path_failures(root))
        failures.extend(_runtime_ignore_failures(root))
        failures.extend(_canonical_backend_dependency_failures(root))
        failures.extend(_canonical_frontend_dependency_failures(root))

    failures.extend(
        f"Generated/runtime artifact found under source folder: {path}"
        for path in _generated_artifact_findings(root)
    )

    if failures:
        print("FAIL repository-structure checks")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("PASS repository-structure checks")
    print("canonical_backend=apps/backend_api/app")
    print("canonical_frontend=apps/frontend_streamlit")
    print("canonical_shared_package=packages/furnace-data/furnace_data")
    return 0


if __name__ == "__main__":
    sys.exit(main())