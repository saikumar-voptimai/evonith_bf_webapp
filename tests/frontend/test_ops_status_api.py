"""Tests for Phase 10 frontend status and ops adapters."""

from __future__ import annotations

from apps.frontend_streamlit.services import ops_api, status_api


class FakeClient:
    def __init__(self):
        self.calls = []
        self.base_url = "http://localhost:8080/api/v1"

    def get(self, path, params=None, headers=None):
        self.calls.append(("GET", path, params, None, headers))
        return {"request_id": "id", "data": {"ok": True, "items": []}, "meta": {}}

    def post(self, path, json=None, params=None, headers=None):
        self.calls.append(("POST", path, params, json, headers))
        return {"request_id": "id", "data": {"ok": True}, "meta": {}}


def test_status_api_calls_status_endpoints():
    client = FakeClient()

    status_api.get_status(client=client)
    status_api.get_runtime_status("token-123", client=client)
    status_api.get_dependency_status("token-123", client=client)
    status_api.get_metrics("token-123", client=client)

    assert client.calls[0] == ("GET", "/status", None, None, {})
    assert client.calls[1] == (
        "GET",
        "/status/runtime/details",
        None,
        None,
        {"Authorization": "Bearer token-123"},
    )
    assert client.calls[2][1] == "/status/dependencies"
    assert client.calls[3][1] == "/metrics"


def test_ops_api_calls_jobs_cleanup_audit_and_error_codes():
    client = FakeClient()

    ops_api.list_jobs("token-123", limit=10, offset=5, client=client)
    ops_api.get_job("job-1", "token-123", client=client)
    ops_api.dry_run_cleanup("token-123", client=client)
    ops_api.run_cleanup("token-123", {"dry_run": False, "max_delete": 2}, client=client)
    ops_api.list_audit_events("token-123", event_type="auth.login.success", client=client)
    ops_api.get_error_codes("token-123", client=client)

    assert client.calls[0] == (
        "GET",
        "/jobs",
        {"limit": 10, "offset": 5},
        None,
        {"Authorization": "Bearer token-123"},
    )
    assert client.calls[1][1] == "/jobs/job-1"
    assert client.calls[2][1] == "/ops/cleanup/dry-run"
    assert client.calls[2][3] == {"dry_run": True}
    assert client.calls[3][1] == "/ops/cleanup/run"
    assert client.calls[4][2]["event_type"] == "auth.login.success"
    assert client.calls[5][1] == "/ops/error-codes"
