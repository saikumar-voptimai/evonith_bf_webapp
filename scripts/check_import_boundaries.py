#!/usr/bin/env python
"""Check backend/frontend import boundaries without importing project modules."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
import sys


BACKEND_FORBIDDEN_ROOTS = {
    "streamlit",
    "custom_pages",
}

BACKEND_HEAVY_TOP_LEVEL_ROOTS = {
    "anthropic",
    "docx",
    "fitz",
    "joblib",
    "langchain",
    "langchain_openai",
    "openai",
    "pptx",
    "qdrant_client",
    "scipy",
    "sentence_transformers",
    "sklearn",
    "torch",
    "xgboost",
}

FRONTEND_API_FORBIDDEN_ROOTS = {
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


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    message: str


def _root_name(module: str | None) -> str:
    return str(module or "").split(".", 1)[0]


def _imports(path: Path) -> list[tuple[str, int, bool]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[tuple[str, int, bool]] = []
    for node in ast.walk(tree):
        is_top_level = any(node is child for child in tree.body)
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((_root_name(alias.name), node.lineno, is_top_level))
        elif isinstance(node, ast.ImportFrom):
            imports.append((_root_name(node.module), node.lineno, is_top_level))
    return imports


def _python_files(root: Path) -> list[Path]:
    return [path for path in root.rglob("*.py") if "__pycache__" not in path.parts]


def _backend_app_dirs(root: Path) -> list[Path]:
    return [root / "apps" / "backend_api" / "app"]


def check_backend(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for app_dir in _backend_app_dirs(root):
        if not app_dir.exists():
            continue
        for path in _python_files(app_dir):
            try:
                imports = _imports(path)
            except SyntaxError as exc:
                findings.append(Finding(path, exc.lineno or 0, f"Could not parse Python file: {exc.msg}"))
                continue
            for module, line, is_top_level in imports:
                if module in BACKEND_FORBIDDEN_ROOTS:
                    findings.append(Finding(path, line, f"Backend imports frontend/UI module: {module}"))
                if is_top_level and module in BACKEND_HEAVY_TOP_LEVEL_ROOTS:
                    findings.append(Finding(path, line, f"Heavy optional dependency imported at backend module top level: {module}"))
    return findings


def _frontend_service_dirs(root: Path) -> list[Path]:
    return [
        root / "apps" / "frontend_streamlit" / "services",
    ]


def check_frontend_services(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for services_dir in _frontend_service_dirs(root):
        if not services_dir.exists():
            continue
        for path in sorted(services_dir.glob("*_api.py")):
            try:
                imports = _imports(path)
            except SyntaxError as exc:
                findings.append(Finding(path, exc.lineno or 0, f"Could not parse Python file: {exc.msg}"))
                continue
            for module, line, _ in imports:
                if module in FRONTEND_API_FORBIDDEN_ROOTS:
                    findings.append(Finding(path, line, f"Frontend API adapter imports backend/heavy module: {module}"))
    return findings


def check_removed_frontend_folder(root: Path) -> list[Finding]:
    legacy = root / ("s" + "rc")
    if legacy.exists():
        return [Finding(legacy, 0, "Removed legacy frontend folder still exists.")]
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    root = args.root.resolve()

    findings = check_backend(root) + check_frontend_services(root) + check_removed_frontend_folder(root)
    if findings:
        print("FAIL import-boundary checks")
        for finding in findings:
            rel = finding.path.resolve()
            try:
                rel = rel.relative_to(root)
            except ValueError:
                pass
            print(f"{rel}:{finding.line}: {finding.message}")
        return 1

    print("PASS import-boundary checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())