"""Tests for the Phase 12 repository structure checker."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_repository_structure_script_passes_current_tree():
    result = subprocess.run(
        [sys.executable, "scripts/check_repository_structure.py"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS repository-structure checks" in result.stdout

