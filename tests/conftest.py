"""Test bootstrap helpers for local package resolution."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("COKE_ENERGY_WEATHER_LIVE", "0")      # no Open-Meteo request from the test suite


def _prefer_local_furnace_data_package() -> None:
    """Prepend local source paths for deterministic imports."""
    repo_root = Path(__file__).resolve().parent.parent
    for source_path in [repo_root / "src", repo_root / "furnace_data"]:
        if str(source_path) not in sys.path:
            sys.path.insert(0, str(source_path))


_prefer_local_furnace_data_package()
