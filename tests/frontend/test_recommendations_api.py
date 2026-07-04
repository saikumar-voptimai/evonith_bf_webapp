"""Tests for frontend Recommendations API adapter."""

from __future__ import annotations

from services import recommendations_api


class FakeClient:
    base_url = "http://localhost:8080/api/v1"

    def __init__(self):
        self.calls = []

    def get(self, path, params=None, headers=None):
        self.calls.append(("GET", path, params, None, headers))
        return {"request_id": "id", "data": {"ok": True}, "meta": {}}

    def post(self, path, json=None, params=None, headers=None):
        self.calls.append(("POST", path, params, json, headers))
        return {"request_id": "id", "data": {"ok": True}, "meta": {}}


def test_get_recommendations_config_calls_endpoint():
    client = FakeClient()

    recommendations_api.get_recommendations_config(token="token", client=client)

    assert client.calls == [
        ("GET", "/recommendations/config", None, None, {"Authorization": "Bearer token"})
    ]


def test_run_recommendations_calls_run_endpoint():
    client = FakeClient()
    payload = {"input_data": {"signals": {"PCI": 1}}}

    recommendations_api.run_recommendations(payload, client=client)

    assert client.calls == [("POST", "/recommendations/run", None, payload, {})]


def test_recommendation_job_helpers():
    client = FakeClient()

    recommendations_api.start_recommendations_job({"input_data": {}}, client=client)
    recommendations_api.get_recommendations_job("job-1", token="token", client=client)

    assert client.calls[0][1] == "/recommendations/jobs"
    assert client.calls[1][1] == "/recommendations/jobs/job-1"
    assert client.calls[1][4] == {"Authorization": "Bearer token"}


def test_recommendations_artifact_url():
    assert (
        recommendations_api.get_recommendations_artifact_download_url("abc", FakeClient())
        == "http://localhost:8080/api/v1/recommendations/artifacts/abc/download"
    )
