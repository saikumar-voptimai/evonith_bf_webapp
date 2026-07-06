"""temporary Phase 12/cleanup compatibility shim for ``furnace_data``.

The real shared package implementation lives in
``packages/furnace-data/furnace_data``. This root package keeps repo-root
imports working until the compatibility surface can be removed in a later
cleanup phase.
"""

from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path


_REPO_ROOT = Path(__file__).resolve().parents[1]
_CANONICAL_PACKAGE = _REPO_ROOT / "packages" / "furnace-data" / "furnace_data"

__path__ = extend_path(__path__, __name__)  # type: ignore[name-defined]
if _CANONICAL_PACKAGE.exists():
    _canonical = str(_CANONICAL_PACKAGE)
    __path__ = [_canonical, *[path for path in __path__ if path != _canonical]]
