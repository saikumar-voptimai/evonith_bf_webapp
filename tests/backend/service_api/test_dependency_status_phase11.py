"""Tests for Phase 11 dependency status metadata."""

from __future__ import annotations

import sys

from app.core.config import BackendSettings
from app.services.dependency_status_service import DependencyStatusService


def test_dependency_status_includes_profiles_and_optional_dependencies(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)
    settings = BackendSettings(
        runtime_profile="edge",
        edge_mode=True,
        auth_secret_key="secret-for-test",
        enable_optional_ai=False,
        enable_optional_vector=False,
        furnacemind_memory_enabled=False,
        furnacemind_llm_enabled=False,
        copilot_llm_enabled=False,
    )

    before = set(sys.modules)
    status = DependencyStatusService(settings).status(force=True)
    after = set(sys.modules)

    assert status["runtime_profile"] == "edge"
    assert status["edge_mode"] is True
    assert "backend-base" in status["dependency_groups"]
    assert status["profile"]["optional_features"]["ai"] is False
    assert any(item["feature_group"] == "backend-vector" for item in status["optional_dependencies"])
    assert any(item["name"] == "qdrant" and item["status"] == "disabled" for item in status["dependencies"])
    assert "site-packages" not in str(status)
    assert "secret-for-test" not in str(status)
    assert "qdrant_client" not in after - before
    assert "openai" not in after - before
