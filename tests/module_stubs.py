"""Import one module with some of its dependencies replaced by stubs, without leaving the stubs behind.

A stub put into ``sys.modules`` while a test file is collected stays there for the rest of the session: every later
test that imports the real module receives the stub (a ``streamlit`` without ``cache_data``, a
``utils.shift_windows`` "of unknown location"), and which tests fail depends on the collection order.
``import_with_stubs`` installs the stubs only for the import of one module and then restores ``sys.modules``:

- the stubs are removed and the modules they replaced are put back;
- the module under test is removed again (the caller keeps its reference to it);
- modules of the repository's source trees (``FIRST_PARTY``) first imported during that import are removed, because
  they may hold references to the stubs; so is any other new module that refers to a stub (for example a Streamlit
  add-on).

Later imports of these modules load them afresh. Other third-party modules first imported during the import stay
loaded: a C extension such as numpy cannot be imported a second time without breaking.
"""

from __future__ import annotations

import importlib
import sys
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
# Source trees of this repository. Not the repository root: the project's virtual environment (.venv) lies under it,
# and removing a third-party module such as a numpy submodule from sys.modules breaks numpy for the rest of the session.
FIRST_PARTY = (ROOT / "src", ROOT / "furnace_data", ROOT / "tests", ROOT / "scripts")
_MISSING = object()


def stub_module(name: str, **attrs: Any) -> types.ModuleType:
    """A module object named ``name`` with the given attributes."""
    module = types.ModuleType(name)
    for attr, value in attrs.items():
        setattr(module, attr, value)
    return module


def _in_repository(module: object) -> bool:
    path = getattr(module, "__file__", None)
    if not path:
        return False
    resolved = Path(path).resolve()
    if "site-packages" in resolved.parts or "dist-packages" in resolved.parts:
        return False
    return any(resolved.is_relative_to(tree) for tree in FIRST_PARTY)


def _refers_to(module: object, stubs: list[types.ModuleType]) -> bool:
    namespace = getattr(module, "__dict__", {})
    return any(value is stub for value in list(namespace.values()) for stub in stubs)


def _set_parent_attribute(name: str, value: object) -> None:
    """Bind ``value`` as the attribute of the parent package of ``name`` (remove it for ``_MISSING``)."""
    parent_name, _, child = name.rpartition(".")
    parent = sys.modules.get(parent_name) if parent_name else None
    if parent is None:
        return
    if value is _MISSING:
        if child in vars(parent):
            delattr(parent, child)
    else:
        setattr(parent, child, value)


def _parent_attribute(name: str) -> object:
    parent_name, _, child = name.rpartition(".")
    parent = sys.modules.get(parent_name) if parent_name else None
    return _MISSING if parent is None else vars(parent).get(child, _MISSING)


def import_with_stubs(name: str, stubs: Mapping[str, Mapping[str, Any]]) -> types.ModuleType:
    """Import module ``name`` while the modules in ``stubs`` (name → attributes) are stubs; see the module notes."""
    before = dict(sys.modules)
    replaced = {key: _parent_attribute(key) for key in [*stubs, name]}
    installed = [stub_module(key, **attrs) for key, attrs in stubs.items()]
    try:
        for module in installed:
            sys.modules[module.__name__] = module
            _set_parent_attribute(module.__name__, module)  # `import a.b as c` reads the parent's attribute first
        sys.modules.pop(name, None)
        return importlib.import_module(name)
    finally:
        for key, module in list(sys.modules.items()):
            if before.get(key) is module:
                continue
            if key in stubs or key == name or _in_repository(module) or _refers_to(module, installed):
                if key in before:
                    sys.modules[key] = before[key]
                else:
                    del sys.modules[key]
                if key not in replaced:
                    _set_parent_attribute(key, before.get(key, _MISSING))
        for key, attribute in replaced.items():
            _set_parent_attribute(key, attribute)
