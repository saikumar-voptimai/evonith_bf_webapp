"""Post-Phase 13 canonical repository skeleton tests."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_canonical_backend_directory_exists():
    assert (REPO_ROOT / "apps" / "backend_api" / "app").is_dir()


def test_canonical_frontend_directory_exists():
    assert (REPO_ROOT / "apps" / "frontend_streamlit").is_dir()


def test_canonical_shared_package_skeleton_exists():
    assert (REPO_ROOT / "packages" / "furnace-data" / "furnace_data").is_dir()


def test_runtime_gitkeep_exists():
    assert (REPO_ROOT / "runtime" / ".gitkeep").is_file()


def test_repository_structure_script_exits_zero():
    result = subprocess.run(
        [sys.executable, "scripts/check_repository_structure.py"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS repository-structure checks" in result.stdout


def test_runtime_is_ignored_except_gitkeep():
    gitignore_lines = {
        line.strip()
        for line in (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert "runtime/*" in gitignore_lines
    assert "!runtime/.gitkeep" in gitignore_lines

    ignored_runtime_file = subprocess.run(
        ["git", "check-ignore", "-q", "runtime/generated-probe.db"],
        cwd=REPO_ROOT,
        check=False,
    )
    assert ignored_runtime_file.returncode == 0

    ignored_gitkeep = subprocess.run(
        ["git", "check-ignore", "-q", "runtime/.gitkeep"],
        cwd=REPO_ROOT,
        check=False,
    )
    assert ignored_gitkeep.returncode != 0
