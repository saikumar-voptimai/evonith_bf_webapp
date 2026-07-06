"""Tests for API v1 Copilot routes."""

from __future__ import annotations

from fastapi.testclient import TestClient


def _client(app_factory, monkeypatch, tmp_path, *, require_auth: bool = False) -> TestClient:
    monkeypatch.setenv("EVONITH_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("EVONITH_COPILOT_REQUIRE_AUTH", "true" if require_auth else "false")
    app = app_factory()
    return TestClient(app, raise_server_exceptions=False)


def test_copilot_config_recent_data_anomaly_analyze_job_artifact_openapi(app_factory, monkeypatch, tmp_path):
    with _client(app_factory, monkeypatch, tmp_path) as client:
        config = client.get("/api/v1/copilot/config")
        recent = client.post(
            "/api/v1/copilot/recent-data",
            json={"source": "input_data", "filters": {"rows": [{"x": 1}, {"x": 2}]}, "limit": 1},
        )
        anomaly = client.post(
            "/api/v1/copilot/anomaly",
            json={"input_data": [{"x": 1}, {"x": 6}]},
        )
        analyze = client.post(
            "/api/v1/copilot/analyze",
            json={
                "question": "summarise",
                "input_data": [{"x": 1}, {"x": 2}],
                "options": {"export": True},
            },
        )
        artifact_id = analyze.json()["data"]["artifacts"][0]["artifact_id"]
        download = client.get(f"/api/v1/copilot/artifacts/{artifact_id}/download")
        job = client.post(
            "/api/v1/copilot/jobs",
            json={"question": "summarise", "input_data": [{"x": 1}, {"x": 2}]},
        )
        job_status = client.get(f"/api/v1/copilot/jobs/{job.json()['data']['job_id']}")
        schema = client.get("/openapi.json").json()

    assert config.status_code == 200
    assert config.json()["data"]["llm_enabled"] is False
    assert "OPENAI_API_KEY" not in str(config.json())
    assert recent.status_code == 200
    assert recent.json()["data"]["returned_rows"] == 1
    assert recent.json()["request_id"]
    assert anomaly.status_code == 200
    assert anomaly.json()["data"]["signals"]
    assert analyze.status_code == 200
    assert "Context JSON" not in analyze.json()["data"]
    assert download.status_code == 200
    assert job.json()["data"]["status"] == "completed"
    assert job_status.json()["data"]["workflow"] == "copilot"
    assert "/api/v1/copilot/analyze" in schema["paths"]


def test_copilot_auth_required_and_invalid_artifact(app_factory, monkeypatch, tmp_path):
    with _client(app_factory, monkeypatch, tmp_path, require_auth=True) as client:
        auth_required = client.get("/api/v1/copilot/config")
    with _client(app_factory, monkeypatch, tmp_path, require_auth=False) as client:
        invalid_artifact = client.get("/api/v1/copilot/artifacts/../secret/download")
        missing_job = client.get("/api/v1/copilot/jobs/missing")

    assert auth_required.status_code == 401
    assert auth_required.json()["error"]["code"] == "AUTH_REQUIRED"
    assert invalid_artifact.status_code in {400, 404}
    assert missing_job.status_code == 404
    assert missing_job.json()["error"]["code"] == "COPILOT_JOB_NOT_FOUND"


def test_copilot_mock_llm_and_timeout(app_factory, monkeypatch, tmp_path):
    monkeypatch.setenv("EVONITH_COPILOT_LLM_ENABLED", "true")
    monkeypatch.setenv("EVONITH_COPILOT_ENABLE_PROVIDER_CALLS", "true")
    monkeypatch.setenv("EVONITH_COPILOT_PROVIDER", "mock")
    with _client(app_factory, monkeypatch, tmp_path) as client:
        ok = client.post(
            "/api/v1/copilot/analyze",
            json={"question": "summarise", "input_data": [{"x": 1}], "allow_llm": True},
        )
        timeout = client.post(
            "/api/v1/copilot/analyze",
            json={
                "question": "summarise",
                "input_data": [{"x": 1}],
                "allow_llm": True,
                "options": {"simulate_timeout": True},
            },
        )

    assert ok.status_code == 200
    assert ok.json()["data"]["llm_used"] is True
    assert timeout.status_code == 504
    assert timeout.json()["error"]["code"] == "COPILOT_LLM_TIMEOUT"
