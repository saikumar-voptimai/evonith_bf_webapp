"""Test bootstrap helpers for local package resolution."""

from __future__ import annotations

import sys
from pathlib import Path


def _prefer_local_furnace_data_package() -> None:
    """Prepend local source paths for deterministic imports."""
    repo_root = Path(__file__).resolve().parent.parent
    ordered_paths = [
        repo_root,
        repo_root / "packages" / "furnace-data",
    ]
    for source_path in ordered_paths:
        source = str(source_path)
        while source in sys.path:
            sys.path.remove(source)
    for source_path in ordered_paths:
        sys.path.insert(0, str(source_path))


_prefer_local_furnace_data_package()
