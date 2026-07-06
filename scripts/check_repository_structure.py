#!/usr/bin/env python
"""Validate canonical structure and generated-artifact cleanup."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


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
    "furnace-data-service/app/main.py",
    "src/app.py",
    "furnace_data/__init__.py",
    "furnace_data/runtime_paths.py",
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
    "src",
    "furnace-data-service",
    "scripts",
]

GENERATED_SUFFIXES = {
    ".csv",
    ".db",
    ".log",
    ".sqlite",
    ".sqlite3",
}


LEGACY_SHIM_MAX_BYTES = 1200
LEGACY_WRAPPER_MAX_BYTES = 700

LEGACY_BACKEND_ALLOWED_FILES = {
    "__init__.py",
    "main.py",
}

ROOT_FURNACE_DATA_ALLOWED_FILES = {
    "__init__.py",
    "runtime_paths.py",
}

SRC_CONFIG_WRAPPER_FILES = {
    "config_loader.py",
    "frontend_settings.py",
    "page_registry.py",
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
    "furnace-data-service/data/results",
    "furnace-data-service/data/static",
    "src/storage/feedback",
    "src/storage/shift_summaries.json",
    "src/storage/daily_summaries.json",
    "src/storage/weekly_summaries.json",
    "src/storage/biweekly_summaries.json",
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


def _legacy_file_listing(root: Path, relative_root: str) -> list[Path]:
    legacy_root = root / relative_root
    if not legacy_root.exists():
        return []
    return sorted(
        path
        for path in legacy_root.rglob("*")
        if path.is_file() and not _is_generated_cache_path(path)
    )


def _legacy_dir_listing(root: Path, relative_root: str) -> list[Path]:
    legacy_root = root / relative_root
    if not legacy_root.exists():
        return []
    return sorted(
        path
        for path in legacy_root.rglob("*")
        if path.is_dir() and "__pycache__" not in path.parts
    )


def _is_generated_cache_path(path: Path) -> bool:
    return "__pycache__" in path.parts or path.suffix.lower() == ".pyc"


def _legacy_backend_shim_failures(root: Path) -> list[str]:
    relative_root = "furnace-data-service/app"
    legacy_root = root / relative_root
    if not legacy_root.exists():
        return []

    failures: list[str] = []
    for directory in _legacy_dir_listing(root, relative_root):
        failures.append(
            f"{relative_root} must not contain duplicate backend subdirectories: "
            f"{directory.relative_to(legacy_root).as_posix()}"
        )

    for path in _legacy_file_listing(root, relative_root):
        rel = path.relative_to(legacy_root).as_posix()
        text = path.read_text(encoding="utf-8", errors="ignore")
        if _is_generated_cache_path(path):
            failures.append(f"{relative_root} must not contain generated cache file: {rel}")
            continue
        if rel not in LEGACY_BACKEND_ALLOWED_FILES:
            failures.append(f"{relative_root} contains non-shim backend file: {rel}")
            continue
        if path.stat().st_size > LEGACY_SHIM_MAX_BYTES:
            failures.append(f"{relative_root}/{rel} exceeds shim size threshold")
        if "temporary Phase 12/cleanup compatibility shim" not in text:
            failures.append(f"{relative_root}/{rel} must be marked as a temporary compatibility shim")
        if rel == "main.py" and "from apps.backend_api.app.main import app, create_app" not in text:
            failures.append(f"{relative_root}/main.py must re-export the canonical backend app")
        if any(marker in text for marker in ("FastAPI(", "APIRouter(", "include_router", "class ")):
            failures.append(f"{relative_root}/{rel} appears to contain backend business logic")
    return failures


def _root_furnace_data_shim_failures(root: Path) -> list[str]:
    relative_root = "furnace_data"
    legacy_root = root / relative_root
    if not legacy_root.exists():
        return []

    failures: list[str] = []
    for directory in _legacy_dir_listing(root, relative_root):
        failures.append(
            f"{relative_root} must not contain duplicate package subdirectories: "
            f"{directory.relative_to(legacy_root).as_posix()}"
        )

    for path in _legacy_file_listing(root, relative_root):
        rel = path.relative_to(legacy_root).as_posix()
        text = path.read_text(encoding="utf-8", errors="ignore")
        if _is_generated_cache_path(path):
            failures.append(f"{relative_root} must not contain generated cache file: {rel}")
            continue
        if rel not in ROOT_FURNACE_DATA_ALLOWED_FILES:
            failures.append(f"{relative_root} contains non-shim shared-package file: {rel}")
            continue
        if path.stat().st_size > LEGACY_SHIM_MAX_BYTES:
            failures.append(f"{relative_root}/{rel} exceeds shim size threshold")
        if "temporary Phase 12/cleanup compatibility shim" not in text:
            failures.append(f"{relative_root}/{rel} must be marked as a temporary compatibility shim")
    return failures


def _src_services_wrapper_failures(root: Path) -> list[str]:
    relative_root = "src/services"
    services_root = root / relative_root
    if not services_root.exists():
        return []

    failures: list[str] = []
    for path in _legacy_file_listing(root, relative_root):
        rel = path.relative_to(services_root).as_posix()
        if _is_generated_cache_path(path):
            failures.append(f"{relative_root} must not contain generated cache file: {rel}")
            continue
        if path.suffix != ".py":
            failures.append(f"{relative_root} contains unexpected non-Python file: {rel}")
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if rel == "__init__.py":
            continue
        if path.stat().st_size > LEGACY_WRAPPER_MAX_BYTES:
            failures.append(f"{relative_root}/{rel} exceeds wrapper size threshold")
        if "temporary Phase 12/cleanup compatibility shim" not in text:
            failures.append(f"{relative_root}/{rel} must be marked as a temporary compatibility shim")
        if "from apps.frontend_streamlit.services." not in text:
            failures.append(f"{relative_root}/{rel} must re-export the canonical frontend service")
    return failures


def _src_custom_page_wrapper_failures(root: Path) -> list[str]:
    relative_root = "src/custom_pages"
    pages_root = root / relative_root
    if not pages_root.exists():
        return []

    failures: list[str] = []
    for directory in _legacy_dir_listing(root, relative_root):
        failures.append(
            f"{relative_root} must not contain duplicate page subdirectories: "
            f"{directory.relative_to(pages_root).as_posix()}"
        )
    for path in _legacy_file_listing(root, relative_root):
        rel = path.relative_to(pages_root).as_posix()
        text = path.read_text(encoding="utf-8", errors="ignore")
        if _is_generated_cache_path(path):
            failures.append(f"{relative_root} must not contain generated cache file: {rel}")
            continue
        if path.suffix != ".py":
            failures.append(f"{relative_root} contains unexpected non-Python file: {rel}")
            continue
        if path.stat().st_size > LEGACY_WRAPPER_MAX_BYTES:
            failures.append(f"{relative_root}/{rel} exceeds wrapper size threshold")
        if "temporary Phase 12/cleanup compatibility shim" not in text:
            failures.append(f"{relative_root}/{rel} must be marked as a temporary compatibility shim")
        if "run_canonical_page(" not in text:
            failures.append(f"{relative_root}/{rel} must delegate to the canonical page")
    return failures


def _src_config_wrapper_failures(root: Path) -> list[str]:
    relative_root = "src/config"
    config_root = root / relative_root
    if not config_root.exists():
        return []

    failures: list[str] = []
    for path in _legacy_file_listing(root, relative_root):
        rel = path.relative_to(config_root).as_posix()
        if _is_generated_cache_path(path):
            failures.append(f"{relative_root} must not contain generated cache file: {rel}")
            continue
        if path.suffix != ".py" or rel == "__init__.py":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if rel not in SRC_CONFIG_WRAPPER_FILES:
            failures.append(f"{relative_root} contains unexpected Python business module: {rel}")
            continue
        if path.stat().st_size > LEGACY_WRAPPER_MAX_BYTES:
            failures.append(f"{relative_root}/{rel} exceeds wrapper size threshold")
        if "temporary Phase 12/cleanup compatibility shim" not in text:
            failures.append(f"{relative_root}/{rel} must be marked as a temporary compatibility shim")
        if "from apps.frontend_streamlit.config." not in text:
            failures.append(f"{relative_root}/{rel} must re-export the canonical frontend config helper")
    return failures


def _src_ui_wrapper_failures(root: Path) -> list[str]:
    relative_root = "src/ui"
    ui_root = root / relative_root
    if not ui_root.exists():
        return []

    failures: list[str] = []
    for path in _legacy_file_listing(root, relative_root):
        rel = path.relative_to(ui_root).as_posix()
        if _is_generated_cache_path(path):
            failures.append(f"{relative_root} must not contain generated cache file: {rel}")
            continue
        if path.suffix != ".py":
            failures.append(f"{relative_root} contains unexpected non-Python file: {rel}")
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if rel.endswith("__init__.py"):
            if path.stat().st_size > LEGACY_WRAPPER_MAX_BYTES:
                failures.append(f"{relative_root}/{rel} exceeds wrapper size threshold")
            continue
        if path.stat().st_size > LEGACY_WRAPPER_MAX_BYTES:
            failures.append(f"{relative_root}/{rel} exceeds wrapper size threshold")
        if "temporary Phase 12/cleanup compatibility shim" not in text:
            failures.append(f"{relative_root}/{rel} must be marked as a temporary compatibility shim")
        if "from apps.frontend_streamlit.ui." not in text:
            failures.append(f"{relative_root}/{rel} must re-export the canonical frontend UI helper")
    return failures


def _legacy_shim_failures(root: Path) -> list[str]:
    failures: list[str] = []
    failures.extend(_legacy_backend_shim_failures(root))
    failures.extend(_root_furnace_data_shim_failures(root))
    failures.extend(_src_services_wrapper_failures(root))
    failures.extend(_src_custom_page_wrapper_failures(root))
    failures.extend(_src_config_wrapper_failures(root))
    failures.extend(_src_ui_wrapper_failures(root))
    return failures

def _content_failures(root: Path) -> list[str]:
    failures: list[str] = []

    backend_shim = _read(root, "furnace-data-service/app/main.py")
    if "from apps.backend_api.app.main import app, create_app" not in backend_shim:
        failures.append("furnace-data-service/app/main.py must re-export apps.backend_api.app.main")
    if "temporary Phase 12/cleanup compatibility shim" not in backend_shim:
        failures.append("furnace-data-service/app/main.py must be marked as a temporary compatibility shim")

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
        "src/storage/**",
        "furnace-data-service/data/results/**",
        "furnace-data-service/data/static/**",
    ):
        if expected not in gitignore:
            failures.append(f".gitignore missing {expected}")

    pyproject = _read(root, "pyproject.toml")
    if 'furnace_data = { path = "./packages/furnace-data", editable = true }' not in pyproject:
        failures.append("pyproject.toml must point furnace_data to ./packages/furnace-data")
    if 'testpaths = ["tests"]' not in pyproject:
        failures.append("pyproject.toml must point pytest discovery at canonical tests/")

    service_pyproject = _read(root, "furnace-data-service/pyproject.toml")
    if 'testpaths = ["../tests/backend"]' not in service_pyproject:
        failures.append("furnace-data-service/pyproject.toml must point pytest discovery at ../tests/backend")

    shared_package = _read(root, "packages/furnace-data/pyproject.toml")
    if 'name = "furnace_data"' not in shared_package:
        failures.append("packages/furnace-data/pyproject.toml must define the furnace_data package")
    if '"furnace_data/assets" = "furnace_data/assets"' not in shared_package:
        failures.append("packages/furnace-data/pyproject.toml must include shared package assets")

    for relative in ("pyproject.toml", "requirements.txt", "uv.lock"):
        text = _read(root, relative).lower()
        if "ale" + "mbic" in text:
            failures.append(f"{relative} must not include legacy migration tooling")

    for old_root in ("furnace-data-service/test", "furnace-data-service/tests"):
        old_path = root / old_root
        if not old_path.exists():
            continue
        active_tests = sorted(path.name for path in old_path.glob("test_*.py"))
        if active_tests:
            failures.append(f"{old_root} must not contain active tests: {active_tests}")
        readme = old_path / "README.md"
        if not readme.exists() or "deprecated" not in readme.read_text(encoding="utf-8", errors="ignore").lower():
            failures.append(f"{old_root} must contain a README.md explaining deprecation")

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
        failures.extend(_legacy_shim_failures(root))

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
