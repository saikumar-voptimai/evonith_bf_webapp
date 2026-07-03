"""Tests that the backend boundary does not depend on Streamlit."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


SERVICE_ROOT = Path(__file__).resolve().parents[1]


def test_backend_app_sources_do_not_import_streamlit():
    for path in (SERVICE_ROOT / "app").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "import streamlit" not in text
        assert "from streamlit" not in text


def test_importing_backend_app_does_not_import_streamlit(monkeypatch, tmp_path):
    monkeypatch.setenv("EVONITH_RUNTIME_DIR", str(tmp_path / "runtime"))
    for name in list(sys.modules):
        if name == "streamlit" or name.startswith("streamlit."):
            del sys.modules[name]
    sys.modules.pop("app.main", None)

    importlib.import_module("app.main")

    assert "streamlit" not in sys.modules
