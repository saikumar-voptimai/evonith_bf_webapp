"""Guards against generated artifacts and obsolete cleanup leftovers."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOTS = [
    REPO_ROOT / "apps",
    REPO_ROOT / "packages",
    REPO_ROOT / "src",
    REPO_ROOT / "scripts",
]
GENERATED_DB_SUFFIXES = {".db", ".sqlite", ".sqlite3"}
GENERATED_CSV_SUFFIXES = {".csv"}
DEPENDENCY_FILES = [
    REPO_ROOT / "pyproject.toml",
    REPO_ROOT / "requirements.txt",
    REPO_ROOT / "uv.lock",
    REPO_ROOT / "packages" / "furnace-data" / "pyproject.toml",
]
OBSOLETE_MANUAL_FILES = [
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


def _source_files() -> list[Path]:
    files: list[Path] = []
    for root in SOURCE_ROOTS:
        if root.exists():
            files.extend(path for path in root.rglob("*") if path.is_file())
    return files


def test_no_database_files_under_source_app_package_folders():
    findings = [
        path.relative_to(REPO_ROOT)
        for path in _source_files()
        if path.suffix.lower() in GENERATED_DB_SUFFIXES
    ]

    assert findings == []


def test_no_generated_csvs_under_source_app_package_folders():
    findings = [
        path.relative_to(REPO_ROOT)
        for path in _source_files()
        if path.suffix.lower() in GENERATED_CSV_SUFFIXES
    ]

    assert findings == []


def test_no_pycache_or_pyc_tracked():
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    tracked = [line.replace("\\", "/") for line in result.stdout.splitlines()]
    findings = [
        path
        for path in tracked
        if "/__pycache__/" in path or path.endswith(".pyc")
    ]
    assert findings == []


def test_no_legacy_migration_folder_or_ini():
    legacy_name = "ale" + "mbic"

    assert not (REPO_ROOT / legacy_name).exists()
    assert not (REPO_ROOT / f"{legacy_name}.ini").exists()


def test_no_legacy_migration_dependency_in_active_dependency_files():
    legacy_name = "ale" + "mbic"
    findings: list[str] = []
    for path in DEPENDENCY_FILES:
        if path.exists() and legacy_name in path.read_text(encoding="utf-8", errors="ignore").lower():
            findings.append(str(path.relative_to(REPO_ROOT)))

    assert findings == []


def test_no_obsolete_manual_test_files():
    findings = [relative for relative in OBSOLETE_MANUAL_FILES if (REPO_ROOT / relative).exists()]

    assert findings == []


def test_repository_structure_check_fails_on_fixture_with_generated_artifact(tmp_path):
    artifact = tmp_path / "apps" / "probe.db"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("generated", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "check_repository_structure.py"),
            "--root",
            str(tmp_path),
            "--generated-only",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "Generated/runtime artifact found under source folder" in result.stdout
    assert "apps" in result.stdout and "probe.db" in result.stdout

