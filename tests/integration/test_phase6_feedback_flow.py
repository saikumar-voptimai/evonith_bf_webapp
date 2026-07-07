"""Integration smoke for Phase 6 feedback API storage and attachment flow."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (str(REPO_ROOT),):
    if path not in sys.path:
        sys.path.insert(0, path)


from apps.backend_api.app.core.config import BackendSettings
from apps.backend_api.app.main import create_app


def test_phase6_feedback_create_upload_download_flow(monkeypatch, tmp_path):
    monkeypatch.setenv("EVONITH_RUNTIME_DIR", str(tmp_path / "runtime"))
    settings = BackendSettings(
        api_prefix="/api/v1",
        backend_env="test",
        feedback_require_auth=False,
        feedback_database_url=f"sqlite:///{(tmp_path / 'feedback.db').as_posix()}",
    )
    app = create_app(settings)

    with TestClient(app, raise_server_exceptions=False) as client:
        create = client.post(
            "/api/v1/feedback/tickets",
            json={"title": "Issue", "description": "Details", "priority": "medium"},
        )
        ticket_id = create.json()["data"]["id"]
        upload = client.post(
            f"/api/v1/feedback/tickets/{ticket_id}/attachments",
            files={"file": ("evidence.txt", b"evidence", "text/plain")},
        )
        attachment_id = upload.json()["data"]["id"]
        download = client.get(f"/api/v1/feedback/attachments/{attachment_id}/download")

    assert create.status_code == 200
    assert upload.status_code == 200
    assert download.status_code == 200
    assert download.content == b"evidence"
    assert (tmp_path / "runtime" / "uploads" / "feedback").exists()
