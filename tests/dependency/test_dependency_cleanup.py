"""Dependency cleanup guard tests for the canonical repo layout."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tomllib
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
OPTIONAL_GROUPS = {
    "backend-data",
    "backend-ml",
    "backend-ai",
    "backend-vector",
    "backend-documents",
}


def _normalize(dep: str) -> str:
    name = dep.strip().split(";", 1)[0].strip()
    for token in ("[", "<", ">", "=", "!", "~"):
        if token in name:
            name = name.split(token, 1)[0]
    if " @ " in name:
        name = name.split(" @ ", 1)[0]
    return name.strip().lower().replace("_", "-")


def _data() -> dict[str, Any]:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def _raw_groups() -> dict[str, list[Any]]:
    return _data()["dependency-groups"]


def _resolve_group(name: str, stack: tuple[str, ...] = ()) -> list[str]:
    groups = _raw_groups()
    assert name not in stack, f"dependency group include cycle: {' -> '.join((*stack, name))}"
    deps: list[str] = []
    for item in groups[name]:
        if isinstance(item, dict) and "include-group" in item:
            deps.extend(_resolve_group(str(item["include-group"]), (*stack, name)))
        else:
            deps.append(str(item))
    return deps


def _names(group_name: str) -> set[str]:
    return {_normalize(dep) for dep in _resolve_group(group_name)}


def test_backend_base_excludes_streamlit() -> None:
    assert "streamlit" not in _names("backend-base")


def test_backend_base_excludes_qdrant_client_unless_documented() -> None:
    assert "qdrant-client" not in _names("backend-base")


def test_backend_base_excludes_provider_sdks_unless_documented() -> None:
    backend_base = _names("backend-base")
    assert "openai" not in backend_base
    assert "anthropic" not in backend_base


def test_project_default_dependencies_do_not_make_optional_stacks_mandatory() -> None:
    project_deps = {_normalize(dep) for dep in _data()["project"].get("dependencies", [])}
    assert project_deps == {"furnace-data"}


def test_frontend_includes_streamlit() -> None:
    assert "streamlit" in _names("frontend")


def test_dev_includes_pytest() -> None:
    assert "pytest" in _names("dev")


def test_alembic_dependency_absent() -> None:
    dependency_files = [
        PYPROJECT,
        REPO_ROOT / "requirements.txt",
        REPO_ROOT / "uv.lock",
        REPO_ROOT / "packages" / "furnace-data" / "pyproject.toml",
        *sorted((REPO_ROOT / "requirements").glob("*.txt")),
    ]
    findings = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in dependency_files
        if path.exists() and "alembic" in path.read_text(encoding="utf-8", errors="ignore").lower()
    ]
    assert findings == []


def test_edge_profile_exists() -> None:
    assert "edge" in _raw_groups()
    assert "fastapi" in _names("edge")


def test_optional_groups_exist() -> None:
    assert OPTIONAL_GROUPS <= set(_raw_groups())


def test_dependency_profile_script_passes() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_dependency_profiles.py"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS dependency-profile checks" in result.stdout
