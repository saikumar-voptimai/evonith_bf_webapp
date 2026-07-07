"""Tests for frontend Copilot API adapter."""

from __future__ import annotations

from apps.frontend_streamlit.services import copilot_api


class FakeClient:
    base_url = "http://localhost:8080/api/v1"

    def __init__(self):
        self.calls = []

    def get(self, path, params=None, headers=None):
        self.calls.append(("GET", path, params, None, headers))
        return {"request_id": "rid", "data": {"ok": True}, "meta": {}}

    def post(self, path, json=None, params=None, headers=None):
        self.calls.append(("POST", path, params, json, headers))
        return {"request_id": "rid", "data": {"ok": True}, "meta": {}}


def test_copilot_config_calls_endpoint_with_token():
    client = FakeClient()

    copilot_api.get_copilot_config(token="token", client=client)

    assert client.calls == [
        ("GET", "/copilot/config", None, None, {"Authorization": "Bearer token"})
    ]


def test_copilot_recent_data_anomaly_and_analyze_calls():
    client = FakeClient()

    copilot_api.get_recent_data({"source": "input_data"}, client=client)
    copilot_api.analyze_anomaly({"input_data": []}, client=client)
    copilot_api.analyze_copilot({"question": "q"}, token="token", client=client)

    assert client.calls[0] == ("POST", "/copilot/recent-data", None, {"source": "input_data"}, {})
    assert client.calls[1] == ("POST", "/copilot/anomaly", None, {"input_data": []}, {})
    assert client.calls[2] == (
        "POST",
        "/copilot/analyze",
        None,
        {"question": "q"},
        {"Authorization": "Bearer token"},
    )


def test_copilot_jobs_and_artifact_url():
    client = FakeClient()

    copilot_api.start_copilot_job({"question": "q"}, client=client)
    copilot_api.get_copilot_job("job-1", token="token", client=client)

    assert client.calls[0][1] == "/copilot/jobs"
    assert client.calls[1] == (
        "GET",
        "/copilot/jobs/job-1",
        None,
        None,
        {"Authorization": "Bearer token"},
    )
    assert (
        copilot_api.get_copilot_artifact_download_url("abc", client)
        == "http://localhost:8080/api/v1/copilot/artifacts/abc/download"
    )
