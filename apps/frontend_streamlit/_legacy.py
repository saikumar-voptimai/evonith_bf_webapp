"""Phase 12 helpers for frontend legacy path compatibility."""

from __future__ import annotations

from pathlib import Path
import runpy
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = Path(__file__).resolve().parent
LEGACY_SRC_ROOT = REPO_ROOT / "src"


def ensure_frontend_legacy_paths() -> None:
    """Make existing ``src`` imports available to canonical frontend modules."""
    for path in (str(LEGACY_SRC_ROOT), str(REPO_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)


def legacy_src_path(relative_path: str) -> Path:
    """Return a path inside the legacy ``src`` tree."""
    return LEGACY_SRC_ROOT / relative_path


def run_legacy_page(relative_path: str) -> None:
    """Execute a legacy Streamlit page from the canonical page wrapper."""
    ensure_frontend_legacy_paths()
    runpy.run_path(str(legacy_src_path(relative_path)), run_name="__main__")


def run_canonical_page(relative_path: str) -> None:
    """Execute a canonical Streamlit page from a legacy page wrapper."""
    ensure_frontend_legacy_paths()
    runpy.run_path(str(APP_ROOT / relative_path), run_name="__main__")

