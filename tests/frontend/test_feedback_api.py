"""Tests for frontend feedback API adapter."""

from __future__ import annotations

from apps.frontend_streamlit.services import feedback_api


class FakeClient:
    base_url = "http://localhost:8080/api/v1"

    def __init__(self):
        self.calls: list[tuple] = []

    @staticmethod
    def _headers(headers=None, idempotency_key=None):
        output = dict(headers or {})
        if idempotency_key:
            output["Idempotency-Key"] = idempotency_key
        return output

    def get(self, path, params=None, headers=None):
        self.calls.append(("GET", path, params, None, headers))
        return {"request_id": "id", "data": {"ok": True, "items": []}, "meta": {}}

    def post(self, path, json=None, params=None, headers=None, idempotency_key=None):
        self.calls.append(("POST", path, params, json, self._headers(headers, idempotency_key)))
        return {"request_id": "id", "data": {"id": "fb_1", "ok": True}, "meta": {}}

    def patch(self, path, json=None, params=None, headers=None):
        self.calls.append(("PATCH", path, params, json, headers))
        return {"request_id": "id", "data": {"ok": True}, "meta": {}}

    def delete(self, path, params=None, headers=None):
        self.calls.append(("DELETE", path, params, None, headers))
        return {"request_id": "id", "data": {"deleted": True}, "meta": {}}

    def upload(self, path, *, filename, content, content_type, headers=None, idempotency_key=None):
        self.calls.append(("UPLOAD", path, filename, content, content_type, self._headers(headers, idempotency_key)))
        return {"request_id": "id", "data": {"id": "fba_1"}, "meta": {}}

    def download(self, path, params=None, headers=None):
        self.calls.append(("DOWNLOAD", path, params, None, headers))
        return b"attachment"


def test_list_tickets_calls_feedback_endpoint_with_token():
    client = FakeClient()

    feedback_api.list_tickets({"status": ["open"], "empty": ""}, token="token", client=client)

    assert client.calls == [("GET", "/feedback/tickets", {"status": ["open"]}, None, {"Authorization": "Bearer token"})]


def test_summary_calls_feedback_summary():
    client = FakeClient()

    feedback_api.get_summary({"page_id": "feedback"}, token="token", client=client)

    assert client.calls == [("GET", "/feedback/summary", {"page_id": "feedback"}, None, {"Authorization": "Bearer token"})]


def test_create_ticket_posts_payload_with_idempotency_key():
    client = FakeClient()
    payload = {"page_id": "feedback", "description": "Details", "ideal_closure": "Fix soon"}

    result = feedback_api.create_ticket(payload, client=client, idempotency_key="idem-create")

    assert result["id"] == "fb_1"
    assert client.calls == [("POST", "/feedback/tickets", None, payload, {"Idempotency-Key": "idem-create"})]


def test_update_ticket_patches_payload_with_token():
    client = FakeClient()

    feedback_api.update_ticket("fb_1", {"title": "Updated", "expected_version": 1}, token="token", client=client)

    assert client.calls == [("PATCH", "/feedback/tickets/fb_1", None, {"title": "Updated", "expected_version": 1}, {"Authorization": "Bearer token"})]


def test_transition_ticket_posts_expected_version_with_idempotency():
    client = FakeClient()

    feedback_api.transition_ticket("fb_1", {"target_status_id": "closed", "expected_version": 2}, token="token", client=client, idempotency_key="idem-transition")

    assert client.calls == [
        (
            "POST",
            "/feedback/tickets/fb_1/transitions",
            None,
            {"target_status_id": "closed", "expected_version": 2},
            {"Authorization": "Bearer token", "Idempotency-Key": "idem-transition"},
        )
    ]


def test_upload_attachment_uses_client_upload_with_idempotency():
    client = FakeClient()

    feedback_api.upload_attachment("fb_1", filename="a.txt", content=b"hello", content_type="text/plain", token="token", client=client, idempotency_key="idem-upload")

    assert client.calls == [("UPLOAD", "/feedback/tickets/fb_1/attachments", "a.txt", b"hello", "text/plain", {"Authorization": "Bearer token", "Idempotency-Key": "idem-upload"})]


def test_download_and_preview_attachment_return_bytes():
    client = FakeClient()

    assert feedback_api.download_attachment("fba_1", token="token", client=client) == b"attachment"
    assert feedback_api.preview_attachment("fba_1", token="token", client=client) == b"attachment"
    assert client.calls == [
        ("DOWNLOAD", "/feedback/attachments/fba_1/download", None, None, {"Authorization": "Bearer token"}),
        ("DOWNLOAD", "/feedback/attachments/fba_1/preview", None, None, {"Authorization": "Bearer token"}),
    ]


def test_delete_attachment_and_ticket_call_delete_endpoints():
    client = FakeClient()

    attachment = feedback_api.delete_attachment("fba_1", token="token", client=client, idempotency_key="idem-delete-attachment")
    ticket = feedback_api.delete_ticket("fb_1", expected_version=3, token="token", client=client, idempotency_key="idem-delete-ticket")

    assert attachment == {"deleted": True}
    assert ticket == {"deleted": True}
    assert client.calls == [
        ("DELETE", "/feedback/attachments/fba_1", None, None, {"Authorization": "Bearer token", "Idempotency-Key": "idem-delete-attachment"}),
        ("DELETE", "/feedback/tickets/fb_1", {"expected_version": 3}, None, {"Authorization": "Bearer token", "Idempotency-Key": "idem-delete-ticket"}),
    ]
