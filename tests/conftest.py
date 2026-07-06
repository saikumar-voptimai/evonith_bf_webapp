"""Test bootstrap helpers for local package resolution."""

from __future__ import annotations

import sys
from pathlib import Path


def _prefer_local_furnace_data_package() -> None:
    """Prepend local source paths for deterministic imports."""
    repo_root = Path(__file__).resolve().parent.parent
    for source_path in [
        repo_root,
        repo_root / "src",
        repo_root / "packages" / "furnace-data",
    ]:
        if str(source_path) not in sys.path:
            sys.path.insert(0, str(source_path))


_prefer_local_furnace_data_package()
