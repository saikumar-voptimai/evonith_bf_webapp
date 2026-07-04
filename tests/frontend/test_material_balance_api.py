"""Tests for frontend Material Balance API adapter."""

from __future__ import annotations

from services import material_balance_api


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


def test_get_material_balance_config_calls_endpoint_with_token():
    client = FakeClient()

    material_balance_api.get_material_balance_config(token="token", client=client)

    assert client.calls == [
        ("GET", "/material-balance/config", None, None, {"Authorization": "Bearer token"})
    ]


def test_validate_material_balance_calls_validate_endpoint():
    client = FakeClient()

    material_balance_api.validate_material_balance({"source": "input_data"}, client=client)

    assert client.calls == [
        ("POST", "/material-balance/validate", None, {"source": "input_data"}, {})
    ]


def test_run_material_balance_calls_run_endpoint():
    client = FakeClient()

    material_balance_api.run_material_balance({"source": "input_data"}, token="token", client=client)

    assert client.calls == [
        (
            "POST",
            "/material-balance/run",
            None,
            {"source": "input_data"},
            {"Authorization": "Bearer token"},
        )
    ]


def test_material_balance_job_helpers():
    client = FakeClient()

    material_balance_api.start_material_balance_job({"source": "input_data"}, client=client)
    material_balance_api.get_material_balance_job("job-1", token="token", client=client)

    assert client.calls[0][1] == "/material-balance/jobs"
    assert client.calls[1] == (
        "GET",
        "/material-balance/jobs/job-1",
        None,
        None,
        {"Authorization": "Bearer token"},
    )


def test_material_balance_artifact_url():
    assert (
        material_balance_api.get_material_balance_artifact_download_url("abc", FakeClient())
        == "http://localhost:8080/api/v1/material-balance/artifacts/abc/download"
    )
