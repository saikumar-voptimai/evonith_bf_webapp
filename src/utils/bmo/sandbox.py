"""Run the Blend Mix Optimiser page a second time, in a state namespace of its own.

WHY THIS EXISTS.

TestBMO has to be a real, independent copy of the Blend Mix Optimiser: load any
saved state, change anything, run the LP and DE, and never disturb the live page.
Both pages live in one Streamlit session, and the optimiser keeps everything in
session state under ``bmo_*`` keys - inputs, applied editor tables, results. Run
the same page twice and the two copies would read and overwrite each other.

Rewriting a ~3,000-line page into a function would change existing work and would
drift from it at the first edit. So instead the page SOURCE is parsed, and every
KEY-SHAPED string that starts with ``bmo_`` is renamed to ``testbmo_`` in the
syntax tree before compiling. That covers session keys, widget keys, form ids and
f-string keys alike. The original file is never modified, so TestBMO picks up any
change made to the optimiser page automatically.

"KEY-SHAPED" MATTERS. ``components.py`` also holds ``"bmo_style.css"`` - a file
name, not a key. Renaming it would silently strip TestBMO of its styling. So only
strings made of ``bmo_`` plus letters, digits and underscores are renamed.

HELPER MODULES. Four widget keys live in ``ui/bmo/components.py`` rather than in
the page (the hot-metal chemistry inputs). Renaming only the page would leave them
shared: chemistry would not restore into the sandbox, and a value typed in TestBMO
could carry into the live page. So every module holding a key-shaped ``bmo_``
string is found by scanning, a renamed COPY of it is built, and the sandbox page's
imports are routed to that copy through a private ``__import__``. The real module
in ``sys.modules`` is never touched - the live page shares this process.

Line numbers are preserved: trees are compiled with their original file names, so
a traceback from TestBMO points at the right line of the real file.
"""

from __future__ import annotations

import ast
import builtins
import functools
import importlib
import re
import types
from pathlib import Path

SRC = Path(__file__).resolve().parents[2]
BMO_PAGE = SRC / "custom_pages" / "9_Blend_Optimizer.py"

LIVE_PREFIX = "bmo_"
SANDBOX_PREFIX = "testbmo_"
_KEY_SHAPED = re.compile(r"^bmo_[A-Za-z0-9_]*$")

# Keys the snapshot panel and the TestBMO loader create for their own widgets.
# They are UI chrome, not optimiser state, and are never captured or restored.
UI_SUFFIX = "ui_"

# Streamlit refuses to have these widgets' state set through the session-state
# API (StreamlitValueAssignmentNotAllowedError). Restoring such a key would crash
# the page the moment the widget renders, so they are recorded but never restored.
NON_WRITABLE_WIDGETS = frozenset({
    "button", "download_button", "link_button", "form_submit_button",
    "data_editor", "file_uploader", "camera_input", "chat_input",
})

# Our own modules: they hold prefixes as data, and must never be renamed.
_OWN_FILES = frozenset({"sandbox.py", "snapshot.py", "snapshot_store.py",
                        "snapshot_report.py", "snapshot_panel.py"})
_SCANNED_PACKAGES = ("ui", "utils", "data", "domain")


def is_key_shaped(text: str) -> bool:
    return bool(_KEY_SHAPED.match(text))


class _PrefixRenamer(ast.NodeTransformer):
    """Rename key-shaped ``bmo_...`` constants (and f-string heads) to ``testbmo_...``."""

    def __init__(self) -> None:
        self.renamed = 0

    def visit_Constant(self, node: ast.Constant) -> ast.AST:  # noqa: N802
        if isinstance(node.value, str) and is_key_shaped(node.value):
            self.renamed += 1
            return ast.copy_location(
                ast.Constant(SANDBOX_PREFIX + node.value[len(LIVE_PREFIX):]), node
            )
        return node


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _renamed_code(path: Path):
    tree = ast.parse(_read(path), filename=str(path))
    renamer = _PrefixRenamer()
    tree = ast.fix_missing_locations(renamer.visit(tree))
    return compile(tree, filename=str(path), mode="exec", dont_inherit=True), renamer.renamed


def _dotted(path: Path) -> str:
    return ".".join(path.relative_to(SRC).with_suffix("").parts)


def _holds_keys(path: Path) -> bool:
    try:
        tree = ast.parse(_read(path))
    except (OSError, SyntaxError):
        return False
    return any(
        isinstance(n, ast.Constant) and isinstance(n.value, str) and is_key_shaped(n.value)
        for n in ast.walk(tree)
    )


def helper_modules_with_keys() -> list[str]:
    """Dotted names of modules outside the page that hold key-shaped ``bmo_`` strings.

    Every one of these is swapped for a renamed copy inside the sandbox.
    """

    found: list[str] = []
    for folder in _SCANNED_PACKAGES:
        base = SRC / folder
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            if path.name in _OWN_FILES or path.name == "__init__.py":
                continue
            if _holds_keys(path):
                found.append(_dotted(path))
    return found


def _stamp() -> tuple:
    files = [BMO_PAGE, *[SRC / Path(*m.split(".")).with_suffix(".py")
                         for m in helper_modules_with_keys()]]
    return tuple((str(f), f.stat().st_mtime_ns) for f in files if f.is_file())


class _ReexportProxy(types.ModuleType):
    """A package seen through the sandbox: names re-exported from a swapped
    module resolve to its renamed copy, everything else to the real package."""

    def __init__(self, real: types.ModuleType, swapped: dict[str, types.ModuleType]):
        super().__init__(real.__name__)
        self._real = real
        self._swapped = swapped  # real child module name -> renamed copy

    def __getattr__(self, name: str):
        value = getattr(self._real, name)
        for child_name, copy in self._swapped.items():
            child = importlib.import_module(child_name)
            if getattr(child, name, _MISSING) is value:
                return getattr(copy, name)
        return value


_MISSING = object()


@functools.lru_cache(maxsize=2)
def _sandbox(stamp: tuple):
    """(page code, builtins for exec, renamed-constant count, swapped modules)."""

    swapped: dict[str, types.ModuleType] = {}
    proxies: dict[str, types.ModuleType] = {}
    real_import = builtins.__import__

    def sandbox_import(name, globals=None, locals=None, fromlist=(), level=0):
        module = real_import(name, globals, locals, fromlist, level)
        if not fromlist:
            return module
        resolved = getattr(module, "__name__", name)
        if resolved in swapped:
            return swapped[resolved]
        if resolved in proxies:
            return proxies[resolved]
        return module

    sandbox_builtins = dict(builtins.__dict__)
    sandbox_builtins["__import__"] = sandbox_import

    renamed_total = 0
    for dotted in helper_modules_with_keys():
        real = importlib.import_module(dotted)
        code, count = _renamed_code(Path(real.__file__))
        renamed_total += count
        copy = types.ModuleType(f"{dotted}__testbmo")
        copy.__file__ = real.__file__
        copy.__package__ = real.__package__
        copy.__dict__["__builtins__"] = sandbox_builtins
        exec(code, copy.__dict__)  # noqa: S102 - our own module source
        swapped[dotted] = copy

    # Packages that re-export a swapped module's names (``from ui.bmo import x``)
    # need a proxy too, or the page would get the real function from the package.
    for dotted in list(swapped):
        parent = dotted.rpartition(".")[0]
        if parent:
            children = {d: m for d, m in swapped.items() if d.rpartition(".")[0] == parent}
            proxies[parent] = _ReexportProxy(importlib.import_module(parent), children)

    page_code, page_count = _renamed_code(BMO_PAGE)
    return page_code, sandbox_builtins, renamed_total + page_count, tuple(swapped)


def compile_sandbox_page() -> tuple[object, int, tuple[str, ...]]:
    """(page code, constants renamed across page and helpers, swapped module names)."""

    code, _b, count, swapped = _sandbox(_stamp())
    return code, count, swapped


def run_sandbox_page() -> None:
    """Execute the renamed page. Streamlit stop/rerun exceptions propagate as usual."""

    code, sandbox_builtins, _count, _swapped = _sandbox(_stamp())
    namespace = {
        "__name__": "__testbmo__",
        # The page resolves config and asset paths from its own location.
        "__file__": str(BMO_PAGE),
        "__builtins__": sandbox_builtins,
    }
    exec(code, namespace)  # noqa: S102 - our own page source, see module docstring


# --- which keys may be restored ----------------------------------------------------


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _key_text(value: ast.AST) -> tuple[str, bool] | None:
    """(key text, is_exact) for a key= argument; f-strings give their literal head."""

    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return value.value, True
    if isinstance(value, ast.JoinedStr) and value.values:
        head = value.values[0]
        if isinstance(head, ast.Constant) and isinstance(head.value, str):
            return head.value, False
    return None


def _scan_non_writable(path: Path) -> tuple[set[str], set[str]]:
    exact: set[str] = set()
    patterns: set[str] = set()
    try:
        tree = ast.parse(_read(path))
    except (OSError, SyntaxError):
        return exact, patterns
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _call_name(node) not in NON_WRITABLE_WIDGETS:
            continue
        for keyword in node.keywords:
            if keyword.arg != "key":
                continue
            found = _key_text(keyword.value)
            if found and found[0].startswith(LIVE_PREFIX):
                suffix = found[0][len(LIVE_PREFIX):]
                (exact if found[1] else patterns).add(suffix)
    return exact, patterns


@functools.lru_cache(maxsize=2)
def _non_writable(stamp: tuple) -> tuple[frozenset[str], frozenset[str]]:
    exact: set[str] = set()
    patterns: set[str] = set()
    for path_str, _mtime_ns in stamp:
        e, p = _scan_non_writable(Path(path_str))
        exact |= e
        patterns |= p
    return frozenset(exact), frozenset(patterns)


def _scanned_files() -> list[Path]:
    files = [BMO_PAGE]
    ui_dir = SRC / "ui"
    if ui_dir.is_dir():
        files.extend(sorted(ui_dir.rglob("*.py")))
    return [f for f in files if f.is_file()]


def non_writable_suffixes() -> tuple[frozenset[str], frozenset[str]]:
    """Key suffixes (after ``bmo_``) belonging to widgets Streamlit won't let us set.

    Read from the source whenever a file changes, so a new button added to the
    page is excluded from restore without anyone maintaining a list.

    Returns:
         - return tuple - (exact suffixes, suffix prefixes from f-string keys).
    """

    stamp = tuple((str(f), f.stat().st_mtime_ns) for f in _scanned_files())
    return _non_writable(stamp)


def is_writable_suffix(suffix: str) -> bool:
    exact, patterns = non_writable_suffixes()
    if suffix in exact:
        return False
    return not any(suffix.startswith(p) for p in patterns)
