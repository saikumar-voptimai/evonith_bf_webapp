"""Tests for Phase 6 backend feedback API endpoints and services."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import BackendSettings
from app.core.errors import ApiError
from app.services.feedback_migration_service import FeedbackMigrationService
from app.services.feedback_service import FeedbackService


def _feedback_client(app_factory, monkeypatch, tmp_path) -> TestClient:
    monkeypatch.setenv("EVONITH_FEEDBACK_REQUIRE_AUTH", "false")
    monkeypatch.setenv(
        "EVONITH_FEEDBACK_DATABASE_URL",
        f"sqlite:///{(tmp_path / 'feedback.db').as_posix()}",
    )
    app = app_factory()
    return TestClient(app, raise_server_exceptions=False)


def test_feedback_config_returns_upload_policy(app_factory, monkeypatch, tmp_path):
    with _feedback_client(app_factory, monkeypatch, tmp_path) as client:
        response = client.get("/api/v1/feedback/config")

    assert response.status_code == 200
    data = response.json()["data"]
    assert "open" in data["statuses"]
    assert ".txt" in data["allowed_attachment_extensions"]


def test_feedback_ticket_create_list_update_comment_and_attachment_flow(
    app_factory, monkeypatch, tmp_path
):
    with _feedback_client(app_factory, monkeypatch, tmp_path) as client:
        create = client.post(
            "/api/v1/feedback/tickets",
            json={
                "title": "Data Explorer feedback",
                "description": "Preview table does not load",
                "category": "Data Explorer",
                "priority": "high",
                "page": "Data Explorer",
            },
        )
        ticket = create.json()["data"]
        listed = client.get("/api/v1/feedback/tickets")
        comment = client.post(
            f"/api/v1/feedback/tickets/{ticket['id']}/comments",
            json={"body": "closing after validation"},
        )
        upload = client.post(
            f"/api/v1/feedback/tickets/{ticket['id']}/attachments",
            files={"file": ("note.txt", b"hello feedback", "text/plain")},
        )
        attachment = upload.json()["data"]
        attachment_list = client.get(
            f"/api/v1/feedback/tickets/{ticket['id']}/attachments"
        )
        download = client.get(
            f"/api/v1/feedback/attachments/{attachment['id']}/download"
        )
        updated = client.patch(
            f"/api/v1/feedback/tickets/{ticket['id']}",
            json={"status": "closed"},
        )

    assert create.status_code == 200
    assert ticket["ticket_number"].startswith("FB-")
    assert listed.status_code == 200
    assert listed.json()["data"]["total"] == 1
    assert updated.status_code == 200
    assert updated.json()["data"]["status"] == "closed"
    assert comment.status_code == 200
    assert upload.status_code == 200
    assert attachment["original_filename"] == "note.txt"
    assert attachment_list.json()["data"][0]["id"] == attachment["id"]
    assert download.status_code == 200
    assert download.content == b"hello feedback"
    assert download.headers["x-request-id"]


def test_feedback_upload_rejects_disallowed_extension(app_factory, monkeypatch, tmp_path):
    with _feedback_client(app_factory, monkeypatch, tmp_path) as client:
        create = client.post(
            "/api/v1/feedback/tickets",
            json={"title": "Issue", "description": "Details", "priority": "low"},
        )
        ticket_id = create.json()["data"]["id"]
        response = client.post(
            f"/api/v1/feedback/tickets/{ticket_id}/attachments",
            files={"file": ("bad.exe", b"nope", "application/octet-stream")},
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
        payload={
            "title": "Issue",
            "description": "Details",
            "priority": "medium",
        },
        current_user=owner,
        request_id="req",
    )

    with pytest.raises(ApiError) as forbidden:
        service.get_ticket(ticket["id"], other)
    with pytest.raises(ApiError) as status_forbidden:
        service.update_ticket(
            ticket_id=ticket["id"],
            payload={"status": "closed"},
            current_user=owner,
        )
    updated = service.update_ticket(
        ticket_id=ticket["id"],
        payload={"status": "closed"},
        current_user=manager,
    )

    assert forbidden.value.status_code == 403
    assert status_forbidden.value.status_code == 403
    assert updated["status"] == "closed"


def test_feedback_openapi_includes_feedback_endpoints(client):
    schema = client.get("/openapi.json").json()

    assert "/api/v1/feedback/tickets" in schema["paths"]
    assert "/api/v1/feedback/tickets/{ticket_id}/attachments" in schema["paths"]


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
                    'Fix soon', 'open', '2026-01-01T00:00:00+00:00',
                    '2026-01-01T00:00:00+00:00', 'operator', 'operator')
            """
        )
        connection.execute(
            """
            INSERT INTO ticket_events (
                id, ticket_id, event_type, old_status, new_status, comment, actor, created_at
            )
            VALUES (1, 1, 'created', NULL, 'open', 'Ticket created', 'operator',
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
    result = FeedbackMigrationService(settings=settings).migrate(
        dry_run=True,
        source_db=source_db,
    )

    assert result.dry_run is True
    assert result.copied_tickets == 1
    assert result.copied_comments == 1
    assert result.copied_attachments == 1
    assert not target_db.exists()
