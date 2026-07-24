
"""Tests for backend feedback API endpoints and services."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from apps.backend_api.app.core.config import BackendSettings
from apps.backend_api.app.core.errors import ApiError
from apps.backend_api.app.services.feedback_migration_service import FeedbackMigrationService
from apps.backend_api.app.services.feedback_service import FeedbackService


def _feedback_client(app_factory, monkeypatch, tmp_path) -> TestClient:
    monkeypatch.setenv("EVONITH_FEEDBACK_REQUIRE_AUTH", "false")
    monkeypatch.setenv(
        "EVONITH_FEEDBACK_DATABASE_URL",
        f"sqlite:///{(tmp_path / 'feedback.db').as_posix()}",
    )
    app = app_factory()
    return TestClient(app, raise_server_exceptions=False)


def _ticket_payload(**overrides):
    payload = {
        "page_id": "data_explorer",
        "title": "Data Explorer feedback",
        "description": "Preview table does not load",
        "ideal_closure": "Restore the preview before shift handover.",
        "priority": "high",
        "tags": ["preview"],
    }
    payload.update(overrides)
    return payload


def test_feedback_config_returns_typed_catalog_and_upload_policy(app_factory, monkeypatch, tmp_path):
    with _feedback_client(app_factory, monkeypatch, tmp_path) as client:
        response = client.get("/api/v1/feedback/config")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["catalog_version"] == "feedback-catalog-v1"
    assert {item["id"] for item in data["pages"]} == {
        "welcome",
        "data_explorer",
        "vboard",
        "vsense",
        "copilot",
        "material_balance",
        "furnacemind",
        "blend_optimizer",
        "feedback",
    }
    assert {item["id"] for item in data["statuses"]} >= {"open", "dependency_conflict", "rejected", "closed"}
    assert ".txt" in data["attachments"]["allowed_extensions"]


def test_feedback_ticket_create_list_update_comment_and_attachment_flow(app_factory, monkeypatch, tmp_path):
    with _feedback_client(app_factory, monkeypatch, tmp_path) as client:
        create = client.post(
            "/api/v1/feedback/tickets",
            json=_ticket_payload(),
            headers={"Idempotency-Key": "create-flow-1"},
        )
        ticket = create.json()["data"]
        listed = client.get("/api/v1/feedback/tickets")
        summary = client.get("/api/v1/feedback/summary")
        comment = client.post(
            f"/api/v1/feedback/tickets/{ticket['id']}/comments",
            json={"body": "closing after validation"},
            headers={"Idempotency-Key": "comment-flow-1"},
        )
        upload = client.post(
            f"/api/v1/feedback/tickets/{ticket['id']}/attachments",
            files={"file": ("note.txt", b"hello feedback", "text/plain")},
            headers={"Idempotency-Key": "upload-flow-1"},
        )
        attachment = upload.json()["data"]
        attachment_list = client.get(f"/api/v1/feedback/tickets/{ticket['id']}/attachments")
        download = client.get(f"/api/v1/feedback/attachments/{attachment['id']}/download")
        events = client.get(f"/api/v1/feedback/tickets/{ticket['id']}/events")
        updated = client.patch(
            f"/api/v1/feedback/tickets/{ticket['id']}",
            json={"status": "closed", "expected_version": ticket["version"]},
        )

    assert create.status_code == 201
    assert ticket["ticket_number"].startswith("FB-")
    assert ticket["page"]["id"] == "data_explorer"
    assert ticket["status"]["id"] == "open"
    assert ticket["version"] == 1
    assert listed.status_code == 200
    assert listed.json()["data"]["total"] == 1
    assert summary.json()["data"]["total"] == 1
    assert updated.status_code == 200
    assert updated.json()["data"]["status"]["id"] == "closed"
    assert comment.status_code == 201
    assert upload.status_code == 201
    assert attachment["original_filename"] == "note.txt"
    assert attachment["checksum_sha256"]
    assert attachment_list.json()["data"]["items"][0]["id"] == attachment["id"]
    assert download.status_code == 200
    assert download.content == b"hello feedback"
    assert download.headers["x-content-type-options"] == "nosniff"
    assert events.json()["data"]["items"][0]["event_type"] == "ticket_created"


def test_feedback_idempotency_replays_and_conflicts(app_factory, monkeypatch, tmp_path):
    with _feedback_client(app_factory, monkeypatch, tmp_path) as client:
        first = client.post("/api/v1/feedback/tickets", json=_ticket_payload(), headers={"Idempotency-Key": "same-key"})
        replay = client.post("/api/v1/feedback/tickets", json=_ticket_payload(), headers={"Idempotency-Key": "same-key"})
        conflict = client.post(
            "/api/v1/feedback/tickets",
            json=_ticket_payload(description="Different issue"),
            headers={"Idempotency-Key": "same-key"},
        )

    assert first.status_code == 201
    assert replay.status_code == 201
    assert replay.json()["data"]["id"] == first.json()["data"]["id"]
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "FEEDBACK_IDEMPOTENCY_CONFLICT"


def test_feedback_upload_rejects_disallowed_extension(app_factory, monkeypatch, tmp_path):
    with _feedback_client(app_factory, monkeypatch, tmp_path) as client:
        create = client.post(
            "/api/v1/feedback/tickets",
            json=_ticket_payload(page_id="feedback", priority="low"),
            headers={"Idempotency-Key": "create-bad-upload"},
        )
        ticket_id = create.json()["data"]["id"]
        response = client.post(
            f"/api/v1/feedback/tickets/{ticket_id}/attachments",
            files={"file": ("bad.exe", b"nope", "application/octet-stream")},
            headers={"Idempotency-Key": "upload-bad"},
        )

    assert response.status_code == 415
    assert response.json()["error"]["code"] == "FEEDBACK_ATTACHMENT_EXTENSION_NOT_ALLOWED"


def test_feedback_routes_require_auth_by_default(app_factory, monkeypatch):
    monkeypatch.setenv("EVONITH_FEEDBACK_REQUIRE_AUTH", "true")
    with TestClient(app_factory(), raise_server_exceptions=False) as client:
        response = client.get("/api/v1/feedback/tickets")

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "AUTH_REQUIRED"


def test_feedback_service_enforces_owner_and_manager_rules(tmp_path):
    settings = BackendSettings(
        backend_env="test",
        feedback_require_auth=True,
        feedback_database_url=f"sqlite:///{(tmp_path / 'feedback.db').as_posix()}",
    )
    service = FeedbackService(settings=settings)
    service.ensure_storage()
    owner = {"id": "u1", "username": "operator", "role": "user"}
    other = {"id": "u2", "username": "other", "role": "user"}
    manager = {"id": "m1", "username": "lead", "role": "supervisor"}

    ticket = service.create_ticket(
        payload=_ticket_payload(page_id="feedback", priority="medium"),
        current_user=owner,
        request_id="req",
        idempotency_key="service-create-1",
    )

    with pytest.raises(ApiError) as forbidden:
        service.get_ticket(ticket["id"], other)
    with pytest.raises(ApiError) as status_forbidden:
        service.update_ticket(
            ticket_id=ticket["id"],
            payload={"status": "closed", "expected_version": ticket["version"]},
            current_user=owner,
        )
    updated = service.update_ticket(
        ticket_id=ticket["id"],
        payload={"status": "closed", "expected_version": ticket["version"]},
        current_user=manager,
    )

    assert forbidden.value.status_code == 403
    assert status_forbidden.value.status_code == 403
    assert updated["status"]["id"] == "closed"
    assert updated["closed_at"] is not None


def test_feedback_openapi_includes_typed_feedback_endpoints(client):
    schema = client.get("/openapi.json").json()

    assert schema["paths"]["/api/v1/feedback/config"]["get"]["operationId"] == "get_feedback_config"
    assert schema["paths"]["/api/v1/feedback/tickets"]["post"]["operationId"] == "create_feedback_ticket"
    assert "/api/v1/feedback/tickets/{ticket_id}/events" in schema["paths"]
    assert "/api/v1/feedback/tickets/{ticket_id}/transitions" in schema["paths"]


def test_feedback_migration_dry_run_does_not_copy_rows(tmp_path):
    source_db = tmp_path / "legacy.db"
    attachment_file = tmp_path / "screen.png"
    attachment_file.write_bytes(b"png")
    with sqlite3.connect(source_db) as connection:
        connection.executescript(
            """
            CREATE TABLE tickets (
                id INTEGER PRIMARY KEY,
                ticket_code TEXT,
                page_name TEXT,
                reported_by TEXT,
                criticality TEXT,
                description TEXT,
                ideal_closure_text TEXT,
                status TEXT,
                created_at TEXT,
                updated_at TEXT,
                created_by TEXT,
                updated_by TEXT
            );
            CREATE TABLE ticket_events (
                id INTEGER PRIMARY KEY,
                ticket_id INTEGER,
                event_type TEXT,
                old_status TEXT,
                new_status TEXT,
                comment TEXT,
                actor TEXT,
                created_at TEXT
            );
            CREATE TABLE ticket_images (
                id INTEGER PRIMARY KEY,
                ticket_id INTEGER,
                image_path TEXT,
                original_filename TEXT,
                uploaded_by TEXT,
                created_at TEXT
            );
            """
        )
        connection.execute(
            """
            INSERT INTO tickets (
                id, ticket_code, page_name, reported_by, criticality, description,
                ideal_closure_text, status, created_at, updated_at, created_by, updated_by
            )
            VALUES (1, 'TKT-000001', 'Feedback', 'operator', 'high', 'Issue',
                    'Fix soon', 'dependency_conflict', '2026-01-01T00:00:00+00:00',
                    '2026-01-01T00:00:00+00:00', 'operator', 'operator')
            """
        )
        connection.execute(
            """
            INSERT INTO ticket_events (
                id, ticket_id, event_type, old_status, new_status, comment, actor, created_at
            )
            VALUES (1, 1, 'status_changed', 'open', 'dependency_conflict', 'Blocked', 'operator',
                    '2026-01-01T00:00:00+00:00')
            """
        )
        connection.execute(
            """
            INSERT INTO ticket_images (
                id, ticket_id, image_path, original_filename, uploaded_by, created_at
            )
            VALUES (1, 1, ?, 'screen.png', 'operator',
                    '2026-01-01T00:00:00+00:00')
            """,
            (str(attachment_file),),
        )

    target_db = tmp_path / "feedback.db"
    settings = BackendSettings(
        backend_env="test",
        feedback_require_auth=False,
        feedback_database_url=f"sqlite:///{target_db.as_posix()}",
    )
    result = FeedbackMigrationService(settings=settings).migrate(dry_run=True, source_db=source_db)

    assert result.dry_run is True
    assert result.copied_tickets == 1
    assert result.copied_comments == 1
    assert result.copied_attachments == 1
    assert not target_db.exists()
