"""Test bootstrap helpers for local package resolution."""

from __future__ import annotations

import sys
from pathlib import Path


def _prefer_local_furnace_data_package() -> None:
    """Prepend local furnace_data source path for deterministic imports."""
    repo_root = Path(__file__).resolve().parent.parent
    furnace_data_src = repo_root / "furnace_data"
    if str(furnace_data_src) not in sys.path:
        sys.path.insert(0, str(furnace_data_src))


_prefer_local_furnace_data_package()
