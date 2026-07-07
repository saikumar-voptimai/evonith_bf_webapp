"""Canonical backend OpenAPI entrypoint coverage."""

from __future__ import annotations


def test_canonical_backend_openapi_paths_are_available(monkeypatch, tmp_path):
    monkeypatch.setenv("EVONITH_RUNTIME_DIR", str(tmp_path / "runtime"))

    from apps.backend_api.app.main import app

    paths = set(app.openapi()["paths"])
    assert "/api/v1/health" in paths
    assert "/api/v1/status" in paths