"""Canonical backend app package with a temporary ``app`` import alias.

The implementation now lives in ``apps/backend_api/app``. During cleanup,
``sys.modules['app']`` is pointed at this package so existing ``app.*`` imports
inside backend modules and compatibility tests continue to resolve without
copying business logic back into ``furnace-data-service/app``.
"""

from __future__ import annotations

import sys


_loaded_app = sys.modules.get("app")
_loaded_path = str(getattr(_loaded_app, "__file__", "")) if _loaded_app else ""
if _loaded_path.endswith("src\\app.py") or _loaded_path.endswith("src/app.py"):
    del sys.modules["app"]

sys.modules["app"] = sys.modules[__name__]
