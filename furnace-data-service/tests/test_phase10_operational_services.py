"""Tests for Phase 10 operational services."""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone

from app.core.config import BackendSettings
from app.repositories.audit_repository import AuditRepository
from app.services.audit_service import AuditService
from app.services.error_registry_service import ErrorRegistryService
from app.services.metrics_service import MetricsService
from app.services.redaction_service import REDACTED, redact_dict, redact_headers, redact_text
from app.services.runtime_cleanup_service import RuntimeCleanupService
from app.services.runtime_status_service import RuntimeStatusService
from furnace_data.runtime_paths import runtime_path


def test_redaction_service_redacts_nested_secrets_and_runtime_path(monkeypatch, tmp_path):
    monkeypatch.setenv("EVONITH_RUNTIME_DIR", str(tmp_path / "runtime"))
    payload = {
        "password": "secret",
        "nested": {
            "authorization": "Bearer abcdefghijklmnopqrstuvwxyz",
            "path": str(tmp_path / "runtime" / "temp" / "file.txt"),
        },
        "safe": "hello",
    }

    redacted = redact_dict(payload)

    assert redacted["password"] == REDACTED
    assert redacted["nested"]["authorization"] == REDACTED
    assert "[RUNTIME_DIR]" in redacted["nested"]["path"]
    assert redacted["safe"] == "hello"
    assert payload["password"] == "secret"
    assert redact_headers({"Authorization": "Bearer token-value"})["Authorization"] == REDACTED
    assert "OPENAI_API_KEY=[REDACTED]" in redact_text("OPENAI_API_KEY=sk-test")


def test_metrics_service_records_safe_aggregates():
    service = MetricsService()

    service.record_request(method="GET", route="/health", status_code=200, duration_ms=12.5)
    service.record_request(method="POST", route="/auth/login", status_code=401, duration_ms=7.0, error_code="INVALID_CREDENTIALS")

    snapshot = service.snapshot()
    assert snapshot["requests_total"] == 2
    assert snapshot["errors_total"] == 1
    assert snapshot["status_codes"]["401"] == 1
    assert snapshot["error_codes"]["INVALID_CREDENTIALS"] == 1
    assert "Bearer" not in str(snapshot)


def test_audit_service_persists_redacted_events(monkeypatch, tmp_path):
    monkeypatch.setenv("EVONITH_RUNTIME_DIR", str(tmp_path / "runtime"))
    repository = AuditRepository(database_url=f"sqlite:///{tmp_path / 'audit.db'}")
    service = AuditService(
        settings=BackendSettings(backend_env="test", audit_log_enabled=True),
        repository=repository,
    )

    event = service.record_event(
        {
            "request_id": "req-1",
            "actor_user_id": "user-1",
            "actor_username": "admin",
            "event_type": "auth.login.success",
            "resource_type": "auth",
            "action": "post",
            "result": "success",
            "status_code": 200,
            "metadata": {"token": "secret-token", "safe": "ok"},
        }
    )
    listing = service.list_events()

    assert event is not None
    assert listing["total"] == 1
    assert listing["items"][0]["metadata"]["token"] == REDACTED
    assert listing["items"][0]["metadata"]["safe"] == "ok"


def test_runtime_status_and_cleanup_are_runtime_scoped(monkeypatch, tmp_path):
    monkeypatch.setenv("EVONITH_RUNTIME_DIR", str(tmp_path / "runtime"))
    settings = BackendSettings(
        backend_env="test",
        cleanup_enabled=True,
        cleanup_dry_run_default=True,
        cleanup_temp_ttl_hours=1,
    )
    temp_dir = runtime_path("temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    old_file = temp_dir / "old.tmp"
    old_file.write_text("old", encoding="utf-8")
    old_time = time.time() - 3 * 3600
    os.utime(old_file, (old_time, old_time))

    status = RuntimeStatusService(settings).status()
    dry_run = RuntimeCleanupService(settings).dry_run()

    assert status["status"] in {"ok", "warning", "degraded"}
    assert status["runtime"]["label"] == "runtime"
    assert "temp" in status["directories"]
    assert dry_run["would_delete"] == 1
    assert dry_run["deleted"] == 0
    assert old_file.exists()
    run = RuntimeCleanupService(settings).run({"dry_run": False, "max_delete": 10})
    assert run["deleted"] == 1
    assert not old_file.exists()
    assert all(not str(item["path"]).startswith(str(tmp_path)) for item in dry_run["candidates"])


def test_error_registry_lists_phase10_families():
    registry = ErrorRegistryService()
    codes = registry.list_codes()["items"]

    assert any(item["code"] == "OPS_*" for item in codes)
    assert any(item["code"] == "FURNACEMIND_*" for item in codes)
    assert registry.get_code("AUTH_REQUIRED")["http_status"] == 401
