#!/usr/bin/env python
"""Validate dependency profile metadata without installing packages."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib  # type: ignore


REQUIRED_GROUPS = {
    "backend-base",
    "backend-data",
    "backend-ml",
    "backend-ai",
    "backend-vector",
    "backend-documents",
    "frontend",
    "dev",
    "edge",
}

OPTIONAL_GROUPS = {
    "backend-data",
    "backend-ml",
    "backend-ai",
    "backend-vector",
    "backend-documents",
}

BACKEND_BASE_FORBIDDEN = {
    "anthropic",
    "chromadb",
    "fitz",
    "langchain",
    "onnxruntime",
    "openai",
    "pydeck",
    "pygwalker",
    "pymupdf",
    "pypdf",
    "pypdf2",
    "python-docx",
    "python-pptx",
    "qdrant-client",
    "scikit-learn",
    "scipy",
    "sentence-transformers",
    "streamlit",
    "torch",
    "voyageai",
    "xgboost",
}

PROJECT_DEFAULT_FORBIDDEN = BACKEND_BASE_FORBIDDEN | {
    "streamlit-autorefresh",
    "streamlit-cookies-manager",
}

EXPECTED_OPTIONAL_DEPS = {
    "backend-data": {"influxdb3-python", "psycopg2-binary"},
    "backend-ml": {"joblib", "scikit-learn", "xgboost"},
    "backend-ai": {"openai", "anthropic"},
    "backend-vector": {"qdrant-client", "sentence-transformers", "torch"},
    "backend-documents": {"pymupdf", "python-docx", "pypdf"},
}

REQUIREMENTS_FILES = {
    "backend-base",
    "backend-data",
    "backend-ml",
    "backend-ai",
    "backend-vector",
    "backend-documents",
    "frontend",
    "dev",
    "edge",
}


def _normalize(dep: str) -> str:
    name = re.split(r"[<>=!~\[]", dep.strip(), maxsplit=1)[0]
    if " @ " in name:
        name = name.split(" @ ", 1)[0]
    if name.startswith("-e "):
        name = Path(name[3:].strip()).name
    return name.strip().lower().replace("_", "-")


def _dep_names(deps: list[str]) -> set[str]:
    return {_normalize(dep) for dep in deps if dep.strip() and not dep.strip().startswith("#")}


def _has_dep(deps: list[str], name: str) -> bool:
    return name in _dep_names(deps)


def _has_forbidden_dep(deps: list[str], forbidden: str) -> bool:
    names = _dep_names(deps)
    return any(name == forbidden or name.startswith(f"{forbidden}-") for name in names)


def _load_pyproject(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _raw_groups(data: dict[str, Any]) -> dict[str, list[Any]]:
    groups = data.get("dependency-groups", {}) or {}
    optional = data.get("project", {}).get("optional-dependencies", {}) or {}
    result: dict[str, list[Any]] = {}
    for source in (groups, optional):
        for name, deps in source.items():
            result.setdefault(str(name), []).extend(list(deps))
    return result


def _resolve_group(groups: dict[str, list[Any]], name: str, stack: tuple[str, ...] = ()) -> list[str]:
    if name in stack:
        cycle = " -> ".join((*stack, name))
        raise ValueError(f"Dependency group include cycle: {cycle}")
    deps: list[str] = []
    for item in groups.get(name, []):
        if isinstance(item, dict) and "include-group" in item:
            deps.extend(_resolve_group(groups, str(item["include-group"]), (*stack, name)))
        else:
            deps.append(str(item))
    return deps


def _load_groups(path: Path) -> dict[str, list[str]]:
    groups = _raw_groups(_load_pyproject(path))
    return {name: _resolve_group(groups, name) for name in groups}


def _read(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def _dependency_files(root: Path, requirements_dir: Path) -> list[Path]:
    files = [
        root / "pyproject.toml",
        root / "requirements.txt",
        root / "uv.lock",
        root / "packages" / "furnace-data" / "pyproject.toml",
    ]
    if requirements_dir.exists():
        files.extend(sorted(requirements_dir.glob("*.txt")))
    for relative in (
        "apps/backend_api/pyproject.toml",
        "apps/frontend_streamlit/pyproject.toml",
    ):
        candidate = root / relative
        if candidate.exists():
            files.append(candidate)
    return files


def _requirements_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in _read(path).splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyproject", type=Path, default=Path(__file__).resolve().parents[1] / "pyproject.toml")
    parser.add_argument("--requirements-dir", type=Path, default=Path(__file__).resolve().parents[1] / "requirements")
    args = parser.parse_args(argv)

    pyproject = args.pyproject.resolve()
    root = pyproject.parent
    data = _load_pyproject(pyproject)
    groups = _load_groups(pyproject)
    project_deps = [str(dep) for dep in data.get("project", {}).get("dependencies", [])]
    failures: list[str] = []
    warnings: list[str] = []

    missing = sorted(REQUIRED_GROUPS - set(groups))
    if missing:
        failures.append(f"Missing dependency groups: {', '.join(missing)}")

    backend_base = groups.get("backend-base", [])
    frontend = groups.get("frontend", [])
    dev = groups.get("dev", [])
    edge = groups.get("edge", [])

    for forbidden in sorted(BACKEND_BASE_FORBIDDEN):
        if _has_forbidden_dep(backend_base, forbidden):
            failures.append(f"backend-base must not include {forbidden}")

    for forbidden in sorted(PROJECT_DEFAULT_FORBIDDEN):
        if _has_forbidden_dep(project_deps, forbidden):
            failures.append(f"project dependencies must not make optional dependency mandatory: {forbidden}")

    if not _has_dep(project_deps, "furnace-data"):
        failures.append("project dependencies must include furnace_data shared package")
    if not _has_dep(backend_base, "fastapi"):
        failures.append("backend-base must include fastapi")
    if not any(_normalize(dep).startswith("uvicorn") for dep in backend_base):
        failures.append("backend-base must include uvicorn")
    if not _has_dep(frontend, "streamlit"):
        failures.append("frontend must include streamlit")
    if _has_dep(frontend, "qdrant-client") or _has_dep(frontend, "openai"):
        failures.append("frontend must not include backend vector/provider SDKs")
    if not _has_dep(dev, "pytest"):
        failures.append("dev must include pytest")
    if not edge:
        failures.append("edge profile must exist")

    for group_name in sorted(OPTIONAL_GROUPS):
        if group_name not in groups:
            failures.append(f"optional group missing: {group_name}")
    for group_name, expected in sorted(EXPECTED_OPTIONAL_DEPS.items()):
        group_deps = groups.get(group_name, [])
        missing_deps = sorted(dep for dep in expected if not _has_dep(group_deps, dep))
        if missing_deps:
            failures.append(f"{group_name} missing expected optional deps: {', '.join(missing_deps)}")

    root_requirements = root / "requirements.txt"
    root_req_text = _read(root_requirements)
    if "Full local development convenience profile" not in root_req_text:
        failures.append("requirements.txt must be documented as a full-dev convenience profile")
    if "-r requirements/dev.txt" not in root_req_text:
        failures.append("requirements.txt must delegate to requirements/dev.txt")

    backend_base_requirements = _requirements_lines(args.requirements_dir / "backend-base.txt")
    if "-e ./packages/furnace-data" not in backend_base_requirements:
        failures.append("requirements/backend-base.txt must install ./packages/furnace-data editable")
    for forbidden in sorted(BACKEND_BASE_FORBIDDEN):
        if _has_forbidden_dep(backend_base_requirements, forbidden):
            failures.append(f"requirements/backend-base.txt must not include {forbidden}")

    dev_requirements = _requirements_lines(args.requirements_dir / "dev.txt")
    for profile in sorted(REQUIREMENTS_FILES - {"dev", "edge"}):
        if f"-r {profile}.txt" not in dev_requirements:
            failures.append(f"requirements/dev.txt must include {profile}.txt")

    for group_name in sorted(REQUIREMENTS_FILES):
        req_file = args.requirements_dir / f"{group_name}.txt"
        if not req_file.exists():
            warnings.append(f"requirements profile missing: {req_file}")

    for dependency_file in _dependency_files(root, args.requirements_dir):
        if "alembic" in _read(dependency_file).lower():
            failures.append(f"{dependency_file.relative_to(root)} must not include Alembic")

    for required_path in (
        root / "apps" / "backend_api" / "app" / "main.py",
        root / "apps" / "frontend_streamlit" / "app.py",
        root / "furnace-data-service" / "app" / "main.py",
        root / "src" / "app.py",
        root / "packages" / "furnace-data" / "pyproject.toml",
    ):
        if not required_path.exists():
            failures.append(f"Missing canonical/compatibility path: {required_path}")

    print("Defined dependency groups:")
    for name in sorted(groups):
        print(f"- {name}: {len(groups[name])} resolved dependencies")
    print(f"Project default dependencies: {', '.join(sorted(_dep_names(project_deps))) or '(none)'}")
    for warning in warnings:
        print(f"WARNING {warning}")
    if failures:
        print("FAIL dependency-profile checks")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("PASS dependency-profile checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
