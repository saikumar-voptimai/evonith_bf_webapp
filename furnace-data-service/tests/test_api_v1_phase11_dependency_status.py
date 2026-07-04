"""API tests for Phase 11 dependency/runtime profile status."""

from __future__ import annotations

from test_api_v1_ops import _client_with_auth, _login


def test_phase11_status_config_and_dependency_status(app_factory, monkeypatch):
    monkeypatch.setenv("EVONITH_RUNTIME_PROFILE", "edge")
    monkeypatch.setenv("EVONITH_EDGE_MODE", "true")
    monkeypatch.setenv("EVONITH_ENABLE_OPTIONAL_AI", "false")
    monkeypatch.setenv("EVONITH_ENABLE_OPTIONAL_VECTOR", "false")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)

    with _client_with_auth(app_factory) as client:
        unauthenticated = client.get("/api/v1/status/config")
        token = _login(client)
        headers = {"Authorization": f"Bearer {token}"}
        config = client.get("/api/v1/status/config", headers=headers)
        dependencies = client.get("/api/v1/status/dependencies", headers=headers)
        health = client.get("/api/v1/health")
        readiness = client.get("/api/v1/readiness")
        me = client.get("/api/v1/auth/me", headers=headers)
        users = client.get("/api/v1/admin/users", headers=headers)
        data_sources = client.get("/api/v1/data/sources")
        feedback_config = client.get("/api/v1/feedback/config")
        material_balance = client.get("/api/v1/material-balance/config", headers=headers)
        recommendations = client.get("/api/v1/recommendations/config", headers=headers)
        blend_context = client.get("/api/v1/blend-optimizer/context", headers=headers)
        copilot_config = client.get("/api/v1/copilot/config", headers=headers)
        furnacemind_config = client.get("/api/v1/furnacemind/config", headers=headers)
        status = client.get("/api/v1/status", headers=headers)
        metrics = client.get("/api/v1/metrics", headers=headers)
        jobs = client.get("/api/v1/jobs", headers=headers)

    assert unauthenticated.status_code == 401
    assert config.status_code == 200
    assert config.json()["data"]["runtime_profile"] == "edge"
    assert config.json()["data"]["edge_mode"] is True
    assert "test-secret" not in str(config.json())
    assert dependencies.status_code == 200
    dependency_data = dependencies.json()["data"]
    assert dependency_data["runtime_profile"] == "edge"
    assert dependency_data["profile"]["optional_features"]["ai"] is False
    assert any(item["feature_group"] == "backend-ai" for item in dependency_data["optional_dependencies"])
    assert "site-packages" not in str(dependency_data)
    assert health.status_code == 200
    assert readiness.status_code == 200
    assert me.status_code == 200
    assert users.status_code == 200
    assert data_sources.status_code == 200
    assert feedback_config.status_code == 200
    assert material_balance.status_code == 200
    assert recommendations.status_code == 200
    assert blend_context.status_code == 200
    assert copilot_config.status_code == 200
    assert furnacemind_config.status_code == 200
    assert status.status_code == 200
    assert metrics.status_code == 200
    assert jobs.status_code == 200
