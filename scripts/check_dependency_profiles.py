#!/usr/bin/env python
"""Validate local dependency profile metadata without installing packages."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

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

BACKEND_BASE_FORBIDDEN = {
    "anthropic",
    "chromadb",
    "langchain",
    "openai",
    "qdrant-client",
    "scikit-learn",
    "scipy",
    "sentence-transformers",
    "streamlit",
    "torch",
    "xgboost",
}


def _normalize(dep: str) -> str:
    name = re.split(r"[<>=!~\[]", dep.strip(), maxsplit=1)[0]
    if " @ " in name:
        name = name.split(" @ ", 1)[0]
    return name.strip().lower().replace("_", "-")


def _load_groups(path: Path) -> dict[str, list[str]]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    groups = data.get("dependency-groups", {}) or {}
    optional = data.get("project", {}).get("optional-dependencies", {}) or {}
    result: dict[str, list[str]] = {}
    for source in (groups, optional):
        for name, deps in source.items():
            result.setdefault(str(name), []).extend(str(dep) for dep in deps)
    return result


def _has_dep(group: list[str], name: str) -> bool:
    normalized = {_normalize(dep) for dep in group}
    return name in normalized


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyproject", type=Path, default=Path(__file__).resolve().parents[1] / "pyproject.toml")
    parser.add_argument("--requirements-dir", type=Path, default=Path(__file__).resolve().parents[1] / "requirements")
    args = parser.parse_args(argv)

    groups = _load_groups(args.pyproject)
    failures: list[str] = []
    warnings: list[str] = []

    missing = sorted(REQUIRED_GROUPS - set(groups))
    if missing:
        failures.append(f"Missing dependency groups: {', '.join(missing)}")

    backend_base = groups.get("backend-base", [])
    frontend = groups.get("frontend", [])
    dev = groups.get("dev", [])

    for forbidden in sorted(BACKEND_BASE_FORBIDDEN):
        if _has_dep(backend_base, forbidden):
            failures.append(f"backend-base must not include {forbidden}")

    if not _has_dep(backend_base, "fastapi"):
        failures.append("backend-base must include fastapi")
    if not any(_normalize(dep).startswith("uvicorn") for dep in backend_base):
        failures.append("backend-base must include uvicorn")
    if not _has_dep(frontend, "streamlit"):
        failures.append("frontend must include streamlit")
    if not _has_dep(dev, "pytest"):
        failures.append("dev must include pytest")

    root = args.pyproject.resolve().parent
    for required_path in (
        root / "apps" / "backend_api" / "app" / "main.py",
        root / "apps" / "frontend_streamlit" / "app.py",
        root / "furnace-data-service" / "app" / "main.py",
        root / "src" / "app.py",
    ):
        if not required_path.exists():
            failures.append(f"Missing Phase 12 app path: {required_path}")

    for group_name in sorted(REQUIRED_GROUPS):
        req_file = args.requirements_dir / f"{group_name}.txt"
        if not req_file.exists():
            warnings.append(f"requirements profile missing: {req_file}")

    print("Defined dependency groups:")
    for name in sorted(groups):
        print(f"- {name}: {len(groups[name])} dependencies")
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
