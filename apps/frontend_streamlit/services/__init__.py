"""Canonical frontend service package.

The Phase 12 canonical adapters live here as thin wrappers while the original
``src.services`` modules remain importable for compatibility.
"""

from __future__ import annotations

from pathlib import Path
import sys


_REPO_ROOT = Path(__file__).resolve().parents[3]
_LEGACY_SRC_ROOT = _REPO_ROOT / "src"
for _path in (str(_LEGACY_SRC_ROOT), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

