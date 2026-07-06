"""Tests that the backend boundary does not depend on Streamlit."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


SERVICE_ROOT = Path(__file__).resolve().parents[3] / "furnace-data-service"


def test_backend_app_sources_do_not_import_streamlit():
    for path in (SERVICE_ROOT / "app").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "import streamlit" not in text
        assert "from streamlit" not in text


def test_importing_backend_app_does_not_import_streamlit(monkeypatch, tmp_path):
    env = os.environ.copy()
    env["EVONITH_RUNTIME_DIR"] = str(tmp_path / "runtime")
    env["PYTHONPATH"] = str(SERVICE_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import app.main; print('streamlit' in sys.modules)",
        ],
        cwd=Path(__file__).resolve().parents[3],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().endswith("False")