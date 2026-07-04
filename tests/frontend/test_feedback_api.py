"""Tests for frontend feedback API adapter."""

from __future__ import annotations

from services import feedback_api


class FakeClient:
    base_url = "http://localhost:8080/api/v1"

    def __init__(self):
        self.calls: list[tuple] = []

    def get(self, path, params=None, headers=None):
        self.calls.append(("GET", path, params, None, headers))
        return {"request_id": "id", "data": {"ok": True, "items": []}, "meta": {}}

    def post(self, path, json=None, params=None, headers=None):
        self.calls.append(("POST", path, params, json, headers))
        return {"request_id": "id", "data": {"id": "fb_1", "ok": True}, "meta": {}}

    def patch(self, path, json=None, params=None, headers=None):
        self.calls.append(("PATCH", path, params, json, headers))
        return {"request_id": "id", "data": {"ok": True}, "meta": {}}

    def delete(self, path, params=None, headers=None):
        self.calls.append(("DELETE", path, params, None, headers))
        return {"request_id": "id", "data": {"deleted": True}, "meta": {}}

    def upload(self, path, *, filename, content, content_type, headers=None):
        self.calls.append(("UPLOAD", path, filename, content, content_type, headers))
        return {"request_id": "id", "data": {"id": "fba_1"}, "meta": {}}

    def download(self, path, params=None, headers=None):
        self.calls.append(("DOWNLOAD", path, params, None, headers))
        return b"attachment"


def test_list_tickets_calls_feedback_endpoint_with_token():
    client = FakeClient()

    feedback_api.list_tickets({"status": "open", "empty": ""}, token="token", client=client)

    assert client.calls == [
        (
            "GET",
            "/feedback/tickets",
            {"status": "open"},
            None,
            {"Authorization": "Bearer token"},
        )
    ]


def test_create_ticket_posts_payload():
    client = FakeClient()
    payload = {"title": "Issue", "description": "Details"}

    result = feedback_api.create_ticket(payload, client=client)

    assert result["id"] == "fb_1"
    assert client.calls == [("POST", "/feedback/tickets", None, payload, {})]


def test_update_ticket_patches_payload_with_token():
    client = FakeClient()

    feedback_api.update_ticket("fb_1", {"status": "closed"}, token="token", client=client)

    assert client.calls == [
        (
            "PATCH",
            "/feedback/tickets/fb_1",
            None,
            {"status": "closed"},
            {"Authorization": "Bearer token"},
        )
    ]


def test_upload_attachment_uses_client_upload():
    client = FakeClient()

    feedback_api.upload_attachment(
        "fb_1",
        filename="a.txt",
        content=b"hello",
        content_type="text/plain",
        token="token",
        client=client,
    )

    assert client.calls == [
        (
            "UPLOAD",
            "/feedback/tickets/fb_1/attachments",
            "a.txt",
            b"hello",
            "text/plain",
            {"Authorization": "Bearer token"},
        )
    ]


def test_download_attachment_returns_bytes():
    client = FakeClient()

    assert feedback_api.download_attachment("fba_1", token="token", client=client) == b"attachment"
    assert client.calls == [
        (
            "DOWNLOAD",
            "/feedback/attachments/fba_1/download",
            None,
            None,
            {"Authorization": "Bearer token"},
        )
    ]


def test_delete_attachment_calls_delete_endpoint():
    client = FakeClient()

    result = feedback_api.delete_attachment("fba_1", token="token", client=client)

    assert result == {"deleted": True}
    assert client.calls == [
        (
            "DELETE",
            "/feedback/attachments/fba_1",
            None,
            None,
            {"Authorization": "Bearer token"},
        )
    ]
