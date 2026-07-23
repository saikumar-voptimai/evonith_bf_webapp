"""Tests for frontend FurnaceMind API adapter."""

from __future__ import annotations

import io

import pytest

from apps.frontend_streamlit.services.api_errors import BackendUnavailableError
from apps.frontend_streamlit.services import furnacemind_api


class FakeClient:
    base_url = "http://localhost:8080/api/v1"

    def __init__(self):
        self.calls: list[tuple] = []

    def get(self, path, params=None, headers=None):
        self.calls.append(("GET", path, params, headers or {}))
        return {"request_id": "req", "data": {"ok": True}, "meta": {}}

    def post(self, path, json=None, params=None, headers=None, idempotency_key=None):
        self.calls.append(("POST", path, json, params, headers or {}, idempotency_key))
        return {"request_id": "req", "data": {"ok": True}, "meta": {}}

    def patch(self, path, json=None, params=None, headers=None):
        self.calls.append(("PATCH", path, json, params, headers or {}))
        return {"request_id": "req", "data": {"ok": True}, "meta": {}}

    def delete(self, path, params=None, headers=None):
        self.calls.append(("DELETE", path, params, headers or {}))
        return {"request_id": "req", "data": {"deleted": True}, "meta": {}}

    def upload(self, path, *, filename, content, content_type, headers=None, idempotency_key=None):
        self.calls.append(("UPLOAD", path, filename, content, content_type, headers or {}, idempotency_key))
        return {"request_id": "req", "data": {"id": "doc"}, "meta": {}}


def test_furnacemind_adapter_calls_expected_endpoints():
    client = FakeClient()

    furnacemind_api.get_furnacemind_config(token="tok", client=client)
    furnacemind_api.create_conversation({"title": "x"}, token="tok", client=client)
    furnacemind_api.list_conversations({"limit": 1}, client=client)
    furnacemind_api.get_conversation("c1", client=client)
    furnacemind_api.update_conversation("c1", {"title": "y"}, client=client)
    furnacemind_api.archive_conversation("c1", client=client)
    furnacemind_api.list_messages("c1", client=client)
    furnacemind_api.send_message("c1", {"content": "hi"}, client=client)
    furnacemind_api.start_run("c1", {"message": "hi"}, client=client)
    furnacemind_api.get_run("r1", token="tok", client=client)
    furnacemind_api.get_run_events("r1", client=client)
    furnacemind_api.list_documents(client=client)
    furnacemind_api.get_document("d1", client=client)
    furnacemind_api.index_document("d1", client=client)
    furnacemind_api.delete_document("d1", client=client)
    furnacemind_api.list_tools(client=client)
    furnacemind_api.submit_message_feedback("m1", {"helpful": True}, client=client)

    assert client.calls[0] == ("GET", "/furnacemind/config", None, {"Authorization": "Bearer tok"})
    assert client.calls[1][0:3] == ("POST", "/furnacemind/conversations", {"title": "x"})
    assert client.calls[5][1] == "/furnacemind/conversations/c1/archive"
    assert client.calls[8][1] == "/furnacemind/conversations/c1/runs"
    assert client.calls[8][-1].startswith("fm-run-")
    assert client.calls[10][1] == "/furnacemind/runs/r1/events"
    assert client.calls[13][1] == "/furnacemind/documents/d1/index"
    assert client.calls[13][-1].startswith("fm-doc-index-")
    assert client.calls[-1][1] == "/furnacemind/messages/m1/feedback"


def test_furnacemind_upload_and_artifact_url():
    client = FakeClient()
    file = io.BytesIO(b"hello")
    file.name = "manual.txt"
    file.type = "text/plain"

    furnacemind_api.upload_document(file, token="tok", client=client)

    assert client.calls == [
        (
            "UPLOAD",
            "/furnacemind/documents",
            "manual.txt",
            b"hello",
            "text/plain",
            {"Authorization": "Bearer tok"},
            None,
        )
    ]
    assert (
        furnacemind_api.download_artifact_url("abc", client)
        == "http://localhost:8080/api/v1/furnacemind/artifacts/abc/download"
    )


def test_furnacemind_backend_unavailable_propagates_cleanly():
    class UnavailableClient(FakeClient):
        def get(self, path, params=None, headers=None):
            raise BackendUnavailableError("offline", request_id="req")

    with pytest.raises(BackendUnavailableError) as exc_info:
        furnacemind_api.get_furnacemind_config(client=UnavailableClient())

    assert exc_info.value.request_id == "req"
