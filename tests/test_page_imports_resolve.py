"""Every import on every Streamlit page must actually resolve.

Why this exists: ``py_compile`` and ``ast.parse`` both pass happily on a page
whose import target does not exist, because an unresolvable import is a RUNTIME
error. A rename that catches an import path therefore reaches the browser as a
blank page with ``ModuleNotFoundError``, and no unit test sees it - the pages
themselves cannot be imported in tests, since importing one executes its
top-level Streamlit code.

So this walks each page's import statements statically and resolves the modules
and the imported names, without executing any page body.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

PAGES_DIR = Path(__file__).resolve().parents[1] / "src" / "custom_pages"
PAGES = sorted(PAGES_DIR.glob("*.py"))


def _first_party(module: str) -> bool:
    """Only check our own packages; third-party availability is uv's problem."""

    return module.split(".")[0] in {
        "agents",
        "config",
        "data",
        "plotters",
        "ui",
        "utils",
    }


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_page_imports_resolve(page: Path) -> None:
    tree = ast.parse(page.read_text(encoding="utf-8"))
    failures: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not _first_party(alias.name):
                    continue
                try:
                    importlib.import_module(alias.name)
                except Exception as exc:  # noqa: BLE001
                    failures.append(f"import {alias.name}: {type(exc).__name__}: {exc}")
            continue

        if not isinstance(node, ast.ImportFrom) or node.level != 0 or not node.module:
            continue
        if not _first_party(node.module):
            continue
        try:
            module = importlib.import_module(node.module)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"from {node.module}: {type(exc).__name__}: {exc}")
            continue
        for alias in node.names:
            if alias.name != "*" and not hasattr(module, alias.name):
                failures.append(f"from {node.module} import {alias.name}: not found")

    assert not failures, f"{page.name} has unresolvable imports:\n  " + "\n  ".join(
        failures
    )
