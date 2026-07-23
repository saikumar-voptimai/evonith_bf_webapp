"""Tests for API v1 FurnaceMind routes."""

from __future__ import annotations

from fastapi.testclient import TestClient


def _client(app_factory, monkeypatch, tmp_path, **env):
    monkeypatch.setenv("EVONITH_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("EVONITH_FURNACEMIND_REQUIRE_AUTH", str(env.pop("require_auth", False)).lower())
    for key, value in env.items():
        monkeypatch.setenv(key, str(value))
    app = app_factory()
    return TestClient(app, raise_server_exceptions=False)


def test_furnacemind_config_conversation_run_events_documents_tools_feedback_and_openapi(app_factory, monkeypatch, tmp_path):
    with _client(
        app_factory,
        monkeypatch,
        tmp_path,
        EVONITH_FURNACEMIND_TOOLS_ENABLED="true",
        EVONITH_FURNACEMIND_MEMORY_ENABLED="true",
        EVONITH_FURNACEMIND_VECTOR_BACKEND="fake",
        EVONITH_FURNACEMIND_LLM_ENABLED="true",
        EVONITH_FURNACEMIND_ENABLE_PROVIDER_CALLS="true",
        EVONITH_FURNACEMIND_PROVIDER="mock",
    ) as client:
        config = client.get("/api/v1/furnacemind/config")
        created = client.post("/api/v1/furnacemind/conversations", json={"title": "Ops"})
        conversation_id = created.json()["data"]["id"]
        listed = client.get("/api/v1/furnacemind/conversations")
        detail = client.get(f"/api/v1/furnacemind/conversations/{conversation_id}")
        patched = client.patch(f"/api/v1/furnacemind/conversations/{conversation_id}", json={"title": "Updated"})
        msg = client.post(f"/api/v1/furnacemind/conversations/{conversation_id}/messages", json={"content": "hello"})
        upload = client.post("/api/v1/furnacemind/documents", files={"file": ("sop.txt", b"pressure stability", "text/plain")})
        document_id = upload.json()["data"]["id"]
        indexed = client.post(f"/api/v1/furnacemind/documents/{document_id}/index", headers={"Idempotency-Key": "doc-index-1"})
        index_status = client.get(f"/api/v1/furnacemind/documents/{document_id}/index/status")
        index_events = client.get(f"/api/v1/furnacemind/documents/{document_id}/index/events")
        docs = client.get("/api/v1/furnacemind/documents")
        doc_detail = client.get(f"/api/v1/furnacemind/documents/{document_id}")
        tools = client.get("/api/v1/furnacemind/tools")
        skill = client.post(
            "/api/v1/furnacemind/skills",
            json={"name": "Pressure Coach", "instruction": "Focus on pressure stability."},
            headers={"Idempotency-Key": "skill-1"},
        )
        skill_id = skill.json()["data"]["id"]
        skill_index = client.post(f"/api/v1/furnacemind/skills/{skill_id}/index", headers={"Idempotency-Key": "skill-index-1"})
        skill_status = client.get(f"/api/v1/furnacemind/skills/{skill_id}/index/status")
        report = client.post(
            "/api/v1/furnacemind/reports",
            json={"report_type": "Shift", "selected_date": "2026-07-23", "shift_label": "A"},
            headers={"Idempotency-Key": "report-1"},
        )
        report_id = report.json()["data"]["id"]
        report_detail = client.get(f"/api/v1/furnacemind/reports/{report_id}")
        run = client.post(
            f"/api/v1/furnacemind/conversations/{conversation_id}/runs",
            json={"message": "summarise pressure", "document_ids": [document_id], "allow_llm": True, "options": {"export": True}},
            headers={"Idempotency-Key": "run-1"},
        )
        run_id = run.json()["data"]["id"]
        status = client.get(f"/api/v1/furnacemind/runs/{run_id}")
        events = client.get(f"/api/v1/furnacemind/runs/{run_id}/events")
        artifact_id = status.json()["data"]["artifacts"][0]["artifact_id"]
        artifact = client.get(f"/api/v1/furnacemind/artifacts/{artifact_id}/download")
        assistant_id = status.json()["data"]["result_message"]["id"]
        feedback = client.post(f"/api/v1/furnacemind/messages/{assistant_id}/feedback", json={"helpful": True})
        feedback_list = client.get("/api/v1/furnacemind/feedback")
        messages = client.get(f"/api/v1/furnacemind/conversations/{conversation_id}/messages")
        archive = client.post(f"/api/v1/furnacemind/conversations/{conversation_id}/archive")
        schema = client.get("/openapi.json").json()

    assert config.status_code == 200
    assert config.json()["data"]["provider_configured"] is True
    assert "OPENAI_API_KEY" not in str(config.json())
    assert "QDRANT_API_KEY" not in str(config.json())
    assert created.status_code == 200
    assert listed.json()["data"]["total"] == 1
    assert detail.status_code == 200
    assert patched.json()["data"]["title"] == "Updated"
    assert msg.json()["data"]["role"] == "user"
    assert upload.status_code == 200
    assert "stored_filename" not in str(upload.json())
    assert indexed.status_code == 202
    assert index_status.json()["data"]["result"]["document"]["indexed"] is True
    assert index_events.json()["data"][0]["sequence"] == 1
    assert docs.json()["data"]["total"] == 1
    assert doc_detail.json()["data"]["id"] == document_id
    assert tools.json()["data"][0]["enabled"] is True
    assert skill.status_code == 200
    assert skill_index.status_code == 202
    assert skill_status.json()["data"]["status"] == "completed"
    assert report.status_code == 202
    assert report_detail.json()["data"]["document"]
    assert run.status_code == 202
    assert status.json()["data"]["status"] == "completed"
    assert events.json()["data"][0]["sequence"] == 1
    assert "Context JSON" not in str(status.json()["data"])
    assert artifact.status_code == 200
    assert feedback.json()["data"]["message_id"] == assistant_id
    assert feedback_list.json()["data"]["total"] == 1
    assert len(messages.json()["data"]) >= 3
    assert archive.json()["data"]["archived"] is True
    assert "/api/v1/furnacemind/config" in schema["paths"]
    assert "/api/v1/furnacemind/conversations/{conversation_id}/runs" in schema["paths"]
    assert "/api/v1/furnacemind/reports" in schema["paths"]


def test_furnacemind_security_errors_and_disabled_modes(app_factory, monkeypatch, tmp_path):
    with _client(app_factory, monkeypatch, tmp_path, require_auth=True) as client:
        auth_required = client.get("/api/v1/furnacemind/config")
    with _client(app_factory, monkeypatch, tmp_path) as client:
        created = client.post("/api/v1/furnacemind/conversations", json={})
        conversation_id = created.json()["data"]["id"]
        code = client.post(
            f"/api/v1/furnacemind/conversations/{conversation_id}/runs",
            json={"message": "x", "options": {"enable_code_execution": True}},
            headers={"Idempotency-Key": "unsafe-code"},
        )
        shell = client.post(
            f"/api/v1/furnacemind/conversations/{conversation_id}/runs",
            json={"message": "x", "options": {"enable_shell_execution": True}},
            headers={"Idempotency-Key": "unsafe-shell"},
        )
        missing_run = client.get("/api/v1/furnacemind/runs/missing")
        missing_conversation = client.get("/api/v1/furnacemind/conversations/missing")
        bad_artifact = client.get("/api/v1/furnacemind/artifacts/../secret/download")
        disallowed_upload = client.post("/api/v1/furnacemind/documents", files={"file": ("bad.exe", b"x", "application/octet-stream")})
        pdf = client.post("/api/v1/furnacemind/documents", files={"file": ("manual.pdf", b"%PDF", "application/pdf")})
        pdf_id = pdf.json()["data"]["id"]
        index_disabled = client.post(f"/api/v1/furnacemind/documents/{pdf_id}/index", headers={"Idempotency-Key": "pdf-index"})
        client_tool_calls = client.post(
            f"/api/v1/furnacemind/conversations/{conversation_id}/runs",
            json={"message": "x", "options": {"tool_calls": [{"name": "unknown", "input": {}}]}},
            headers={"Idempotency-Key": "client-tools"},
        )
        missing_idempotency = client.post(f"/api/v1/furnacemind/conversations/{conversation_id}/runs", json={"message": "x"})

    assert auth_required.status_code == 401
    assert auth_required.json()["error"]["code"] == "AUTH_REQUIRED"
    assert code.status_code == 403
    assert code.json()["error"]["code"] == "FURNACEMIND_CODE_EXECUTION_DISABLED"
    assert shell.status_code == 403
    assert shell.json()["error"]["code"] == "FURNACEMIND_SHELL_EXECUTION_DISABLED"
    assert missing_run.status_code == 404
    assert missing_run.json()["error"]["code"] == "FURNACEMIND_RUN_NOT_FOUND"
    assert missing_conversation.status_code == 404
    assert bad_artifact.status_code in {400, 404}
    assert disallowed_upload.status_code in {415, 422}
    assert pdf.status_code == 200
    assert pdf.json()["data"]["warnings"][0]["code"] == "FURNACEMIND_DOCUMENT_EXTRACTION_UNSUPPORTED"
    assert index_disabled.status_code == 202
    assert index_disabled.json()["data"]["status"] in {"pending", "completed"}
    assert client_tool_calls.status_code == 403
    assert client_tool_calls.json()["error"]["code"] == "FURNACEMIND_UNSAFE_INPUT"
    assert missing_idempotency.status_code == 422