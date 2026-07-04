"""Tests for Phase 11 runtime profile settings."""

from __future__ import annotations

from app.core.config import BackendSettings


def test_runtime_profile_defaults_are_safe():
    settings = BackendSettings(auth_secret_key="secret-for-test")

    summary = settings.safe_runtime_profile_summary()

    assert summary["runtime_profile"] == "local"
    assert summary["edge_mode"] is False
    assert summary["backend_profile"] == "backend-base"
    assert summary["frontend_profile"] == "frontend"
    assert summary["optional_features"]["ai"] is False
    assert summary["optional_features"]["vector"] is False
    assert "secret-for-test" not in str(summary)


def test_edge_runtime_profile_env_parsing(monkeypatch):
    monkeypatch.setenv("EVONITH_RUNTIME_PROFILE", "edge")
    monkeypatch.setenv("EVONITH_EDGE_MODE", "true")
    monkeypatch.setenv("EVONITH_EDGE_DEVICE_TYPE", "jetson")
    monkeypatch.setenv("EVONITH_ENABLE_OPTIONAL_AI", "false")
    monkeypatch.setenv("EVONITH_ENABLE_OPTIONAL_ML", "true")
    monkeypatch.setenv("EVONITH_ENABLE_OPTIONAL_VECTOR", "false")
    monkeypatch.setenv("EVONITH_ENABLE_OPTIONAL_DOCUMENTS", "true")
    monkeypatch.setenv("EVONITH_ENABLE_OPTIONAL_LOCAL_LLM", "false")

    settings = BackendSettings(auth_secret_key="secret-for-test")
    summary = settings.safe_runtime_profile_summary()

    assert settings.runtime_profile == "edge"
    assert settings.edge_mode is True
    assert settings.edge_device_type == "jetson"
    assert summary["optional_features"] == {
        "ai": False,
        "ml": True,
        "vector": False,
        "documents": True,
        "local_llm": False,
    }
