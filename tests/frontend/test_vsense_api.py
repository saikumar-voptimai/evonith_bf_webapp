from __future__ import annotations

from typing import Any

from apps.frontend_streamlit.services.vsense_api import VSenseApi


class FakeClient:
    base_url = "http://localhost:8080/api/v1"
    last_response_request_id = "req-1"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any] | None, Any, dict[str, str], str | None]] = []

    def get(self, path, params=None, headers=None):
        self.calls.append(("GET", path, params, None, headers or {}, None))
        return {"request_id": "req-1", "data": {"ok": True}, "meta": {"warnings": []}}

    def post(self, path, json=None, headers=None, idempotency_key=None):
        self.calls.append(("POST", path, None, json, headers or {}, idempotency_key))
        return {"request_id": "req-1", "data": {"ok": True}, "meta": {"warnings": []}}

    def put(self, path, json=None, headers=None, idempotency_key=None):
        self.calls.append(("PUT", path, None, json, headers or {}, idempotency_key))
        return {"request_id": "req-1", "data": {"ok": True}, "meta": {"warnings": []}}


def test_vsense_api_uses_exact_paths_bearer_and_idempotency_headers():
    client = FakeClient()
    api = VSenseApi("token", client=client)  # type: ignore[arg-type]

    api.get_catalog()
    api.create_context({"optimization_type_id": "eta_co"}, idempotency_key="ctx-key")
    api.get_control_profile("eta_co")
    api.update_control_profile("eta_co", {"parameters": []}, idempotency_key="profile-key")
    api.create_run({"context_id": "ctx_1"}, idempotency_key="run-key")
    api.get_run("run-1")
    api.get_run_events("run-1", after=7)
    api.cancel_run("run-1")

    assert client.calls == [
        ("GET", "/vsense/catalog", None, None, {"Authorization": "Bearer token"}, None),
        ("POST", "/vsense/contexts", None, {"optimization_type_id": "eta_co"}, {"Authorization": "Bearer token"}, "ctx-key"),
        ("GET", "/vsense/control-profiles/eta_co", None, None, {"Authorization": "Bearer token"}, None),
        ("PUT", "/vsense/control-profiles/eta_co", None, {"parameters": []}, {"Authorization": "Bearer token"}, "profile-key"),
        ("POST", "/vsense/runs", None, {"context_id": "ctx_1"}, {"Authorization": "Bearer token"}, "run-key"),
        ("GET", "/vsense/runs/run-1", None, None, {"Authorization": "Bearer token"}, None),
        ("GET", "/vsense/runs/run-1/events", {"after": 7}, None, {"Authorization": "Bearer token"}, None),
        ("POST", "/vsense/runs/run-1/cancel", None, {}, {"Authorization": "Bearer token"}, None),
    ]
