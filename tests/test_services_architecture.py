"""The service layer and the twin-api never depend on Streamlit or on ``ui.*`` (#271)."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
FREE_PACKAGES = ("services", "api", "worker", "data/twin")
FREE_MODULES = ("utils/memo.py", "data/furnace_status/repository.py", "data/db.py", "domain/auth_service.py")


def _python_files() -> list[Path]:
    files = [p for pkg in FREE_PACKAGES for p in (SRC / pkg).rglob("*.py")]
    return files + [SRC / m for m in FREE_MODULES]


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: str(p.relative_to(SRC)))
def test_no_streamlit_or_ui_import(path: Path) -> None:
    bad = sorted(n for n in _imports(path) if n == "streamlit" or n.startswith(("streamlit.", "ui.")) or n == "ui")
    assert not bad, f"{path.relative_to(SRC)} imports {bad}"


@pytest.mark.parametrize(
    "module",
    # data.db and domain.auth_service are imported lazily by services.auth (identity store of the twin-api).
    [
        "services.coke_energy",
        "services.forecasts",
        "services.twin_views",
        "services.store",
        "worker.settings",  # every worker kind (SV-1.5 forecasts, SV-1.6 optimiser runs)
        "data.ml.static_dataset_manager",  # the worker refreshes the hourly dataset with it (SV-1.5)
        "services.optimiser",
        "services.schedule",  # SV-1.7: schedule, catch-up and audit
        "services.jobstats",
        "services.day_calibration",
        "services.learn",
        "services.twin",
        "services.status",
        "services.auth",
        "services.job_archive",
        "data.db",
        "data.twin",
        "api.main",
    ],
)
def test_no_transitive_streamlit_import(module: str) -> None:
    """Importing the module in a fresh interpreter must not load Streamlit or a ui module through any dependency."""
    code = (
        f"import importlib, sys; importlib.import_module({module!r}); "
        "bad = [k for k in sys.modules if k == 'streamlit' or k.startswith(('streamlit.', 'ui.')) or k == 'ui']; "
        "print(','.join(sorted(bad)))"
    )
    env_path = [str(SRC), str(REPO / "furnace_data")]
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(env_path)},
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == "", f"{module} loads {out.stdout.strip()}"
