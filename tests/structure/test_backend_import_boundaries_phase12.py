"""Phase 12 backend import-boundary checks."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIRS = [
    REPO_ROOT / "apps" / "backend_api" / "app",
    REPO_ROOT / "furnace-data-service" / "app",
]


def test_backend_sources_do_not_import_streamlit():
    for backend_dir in BACKEND_DIRS:
        for path in backend_dir.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            assert "import streamlit" not in text
            assert "from streamlit" not in text


def test_canonical_backend_import_does_not_load_streamlit(monkeypatch, tmp_path):
    env = dict(os.environ)
    env["EVONITH_RUNTIME_DIR"] = str(tmp_path / "runtime")

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from apps.backend_api.app.main import app; "
                "print(app.title); "
                "print(any(name == 'streamlit' or name.startswith('streamlit.') for name in sys.modules))"
            ),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().splitlines()[-1] == "False"
