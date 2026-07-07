"""Canonical backend entrypoint tests."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient


REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_BACKEND_APP = REPO_ROOT / "apps" / "backend_api" / "app"


def test_canonical_backend_entrypoint_imports(monkeypatch, tmp_path):
    monkeypatch.setenv("EVONITH_RUNTIME_DIR", str(tmp_path / "runtime"))

    from apps.backend_api.app.main import app

    assert app.title == "Evonith BF Backend API"
    assert isinstance(app, FastAPI)


def test_canonical_backend_openapi_includes_v1_health(monkeypatch, tmp_path):
    monkeypatch.setenv("EVONITH_RUNTIME_DIR", str(tmp_path / "runtime"))

    from apps.backend_api.app.main import app

    assert "/api/v1/health" in app.openapi()["paths"]


def test_canonical_backend_health_endpoint(monkeypatch, tmp_path):
    monkeypatch.setenv("EVONITH_RUNTIME_DIR", str(tmp_path / "runtime"))

    from apps.backend_api.app.main import app

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 200


def test_canonical_backend_path_does_not_import_streamlit():
    for path in CANONICAL_BACKEND_APP.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        assert "import streamlit" not in text
        assert "from streamlit" not in text


def test_backend_import_does_not_require_optional_runtime_dependencies(tmp_path):
    env = dict(os.environ)
    env.update(
        {
            "EVONITH_RUNTIME_DIR": str(tmp_path / "runtime"),
            "EVONITH_AUTH_SECRET_KEY": "dev-only-secret-change-me",
            "EVONITH_ENABLE_OPTIONAL_AI": "false",
            "EVONITH_ENABLE_OPTIONAL_VECTOR": "false",
            "EVONITH_ENABLE_OPTIONAL_LOCAL_LLM": "false",
            "EVONITH_MODEL_DIR": str(tmp_path / "missing-models"),
        }
    )
    code = r"""
import importlib.abc
import sys

blocked_roots = {
    "anthropic",
    "docx",
    "easyocr",
    "fitz",
    "langchain",
    "langchain_openai",
    "openai",
    "paddleocr",
    "pptx",
    "pytesseract",
    "qdrant_client",
    "sentence_transformers",
    "streamlit",
}


class BlockOptionalImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in blocked_roots:
            raise ModuleNotFoundError(f"blocked optional dependency: {fullname}")
        return None


sys.meta_path.insert(0, BlockOptionalImports())
from apps.backend_api.app.main import app

if "/api/v1/health" not in app.openapi()["paths"]:
    raise SystemExit("missing health route")

loaded = sorted(root for root in blocked_roots if root in sys.modules)
if loaded:
    raise SystemExit(f"loaded optional modules: {loaded}")

print(app.title)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Evonith BF Backend API" in result.stdout