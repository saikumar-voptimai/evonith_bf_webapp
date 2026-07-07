"""Guards for the canonical tests/ layout."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_TEST_DIRS = [
    "backend",
    "frontend",
    "integration",
    "dependency",
    "structure",
    "deployment",
    "fixtures",
]
REMOVED_TEST_ROOTS = [
    REPO_ROOT / ("furnace-data" + "-service") / "test",
    REPO_ROOT / ("furnace-data" + "-service") / "tests",
]


def _pyproject(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def test_canonical_test_directories_exist() -> None:
    for directory in CANONICAL_TEST_DIRS:
        assert (REPO_ROOT / "tests" / directory).is_dir()


def test_no_root_level_test_modules_remain() -> None:
    root_tests = sorted(path.name for path in (REPO_ROOT / "tests").glob("test_*.py"))
    assert root_tests == []


def test_removed_service_test_roots_are_absent() -> None:
    for old_root in REMOVED_TEST_ROOTS:
        assert not old_root.exists()


def test_pytest_config_points_to_canonical_tests() -> None:
    root_config = _pyproject(REPO_ROOT / "pyproject.toml")

    assert root_config["tool"]["pytest"]["ini_options"]["testpaths"] == ["tests"]


def test_full_suite_discovers_from_root() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "tests", "-q"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=180,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "tests/backend" in result.stdout.replace("\\", "/")
