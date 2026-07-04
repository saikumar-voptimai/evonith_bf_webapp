#!/usr/bin/env python
"""Import frontend API adapters without backend internals or a running backend."""

from __future__ import annotations

import argparse
import ast
import importlib
from pathlib import Path
import sys


ADAPTERS = [
    "auth_api",
    "admin_api",
    "data_api",
    "dataset_api",
    "feedback_api",
    "material_balance_api",
    "recommendations_api",
    "blend_optimizer_api",
    "copilot_api",
    "furnacemind_api",
    "status_api",
    "ops_api",
]

FORBIDDEN_IMPORT_ROOTS = {
    "anthropic",
    "app",
    "furnace_data",
    "influxdb3",
    "joblib",
    "langchain",
    "openai",
    "psycopg2",
    "qdrant_client",
    "scipy",
    "sentence_transformers",
    "sklearn",
    "sqlite3",
    "torch",
    "xgboost",
}


def _root(module: str | None) -> str:
    return str(module or "").split(".", 1)[0]


def _scan_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(_root(alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.append(_root(node.module))
    return modules


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    root = args.root.resolve()
    service_dirs = [
        root / "apps" / "frontend_streamlit" / "services",
        root / "src" / "services",
    ]
    failures: list[str] = []

    for services_dir in service_dirs:
        for adapter in ADAPTERS:
            path = services_dir / f"{adapter}.py"
            if path.exists():
                for module in _scan_imports(path):
                    if module in FORBIDDEN_IMPORT_ROOTS:
                        failures.append(f"{path}: imports forbidden module {module}")
                if "furnace-data-service" in path.read_text(encoding="utf-8", errors="ignore"):
                    failures.append(f"{path}: references furnace-data-service path")

    if root == Path(__file__).resolve().parents[1]:
        sys.path.insert(0, str(root))
        sys.path.insert(0, str(root / "src"))
        for adapter in ADAPTERS:
            try:
                importlib.import_module(f"apps.frontend_streamlit.services.{adapter}")
            except Exception as exc:
                failures.append(f"apps.frontend_streamlit.services.{adapter}: import failed: {exc}")
            try:
                importlib.import_module(f"services.{adapter}")
            except Exception as exc:
                failures.append(f"services.{adapter}: import failed: {exc}")
        loaded_forbidden = sorted(module for module in FORBIDDEN_IMPORT_ROOTS if module in sys.modules)
        if loaded_forbidden:
            failures.append(f"Forbidden modules loaded by frontend API adapters: {', '.join(loaded_forbidden)}")

    if failures:
        print("FAIL frontend API import checks")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("PASS frontend API import checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
