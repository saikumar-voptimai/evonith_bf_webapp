"""Tests for frontend Blend Optimizer API adapter."""

from __future__ import annotations

from apps.frontend_streamlit.services import blend_optimizer_api


class FakeClient:
    base_url = "http://localhost:8080/api/v1"

    def __init__(self):
        self.calls = []

    def get(self, path, params=None, headers=None):
        self.calls.append(("GET", path, params, None, headers))
        return {"request_id": "id", "data": {"ok": True}, "meta": {}}

    def patch(self, path, json=None, params=None, headers=None):
        self.calls.append(("PATCH", path, params, json, headers))
        return {"request_id": "id", "data": {"ok": True}, "meta": {}}

    def download(self, path, params=None, headers=None):
        self.calls.append(("DOWNLOAD", path, params, None, headers))
        return b"data"

    def post(self, path, json=None, params=None, headers=None, idempotency_key=None):
        self.calls.append(("POST", path, params, json, headers, idempotency_key))
        return {"request_id": "id", "data": {"ok": True}, "meta": {}}


def test_get_blend_optimizer_context_calls_endpoint():
    client = FakeClient()

    blend_optimizer_api.get_blend_optimizer_context(token="token", client=client)

    assert client.calls == [
        ("GET", "/blend-optimizer/context", None, None, {"Authorization": "Bearer token"})
    ]


def test_list_blend_optimizer_models_calls_endpoint():
    client = FakeClient()

    blend_optimizer_api.list_blend_optimizer_models(client=client)

    assert client.calls == [("GET", "/blend-optimizer/models", None, None, {})]


def test_predict_and_optimize_call_post_endpoints():
    client = FakeClient()

    blend_optimizer_api.predict_blend_outputs({"model_name": "m", "features": {}}, client=client)
    blend_optimizer_api.optimize_blend({"materials": []}, token="token", client=client)

    assert client.calls[0][1] == "/blend-optimizer/predict"
    assert client.calls[1] == (
        "POST",
        "/blend-optimizer/optimize",
        None,
        {"materials": []},
        {"Authorization": "Bearer token"},
        None,
    )


def test_blend_optimizer_job_helpers_and_artifact_url():
    client = FakeClient()

    blend_optimizer_api.start_blend_optimizer_job({"materials": []}, client=client)
    blend_optimizer_api.get_blend_optimizer_job("job-1", client=client)

    assert client.calls[0][1] == "/blend-optimizer/jobs"
    assert client.calls[1][1] == "/blend-optimizer/jobs/job-1"
    assert (
        blend_optimizer_api.get_blend_optimizer_artifact_download_url("abc", FakeClient())
        == "http://localhost:8080/api/v1/blend-optimizer/artifacts/abc/download"
    )


def test_api_first_run_helpers_send_idempotency_key():
    client = FakeClient()

    blend_optimizer_api.create_blend_optimizer_context({"source_refresh": "use_cached"}, idempotency_key="ctx-1", client=client)
    blend_optimizer_api.create_blend_optimizer_run({"mode": "lp_baseline"}, idempotency_key="run-1", token="token", client=client)
    blend_optimizer_api.get_blend_optimizer_run_events("run-1", after=3, client=client)
    blend_optimizer_api.download_blend_optimizer_artifact("artifact-1", client=client)

    assert client.calls[0] == ("POST", "/blend-optimizer/contexts", None, {"source_refresh": "use_cached"}, {}, "ctx-1")
    assert client.calls[1] == ("POST", "/blend-optimizer/runs", None, {"mode": "lp_baseline"}, {"Authorization": "Bearer token"}, "run-1")
    assert client.calls[2] == ("GET", "/blend-optimizer/runs/run-1/events", {"after": 3}, None, {})
    assert client.calls[3][0:2] == ("DOWNLOAD", "/blend-optimizer/artifacts/artifact-1/download")
