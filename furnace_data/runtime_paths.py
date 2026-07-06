"""temporary Phase 12/cleanup compatibility shim for ``furnace_data.runtime_paths``."""

from __future__ import annotations

import importlib.util
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
_CANONICAL = _REPO_ROOT / "packages" / "furnace-data" / "furnace_data" / "runtime_paths.py"
_SPEC = importlib.util.spec_from_file_location("_evonith_furnace_data_runtime_paths", _CANONICAL)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load canonical furnace_data.runtime_paths from {_CANONICAL}")

_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

for _name in dir(_MODULE):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_MODULE, _name)

__all__ = [_name for _name in globals() if not _name.startswith("_")]
