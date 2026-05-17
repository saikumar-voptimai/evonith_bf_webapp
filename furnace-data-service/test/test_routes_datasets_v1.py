from __future__ import annotations

from datetime import datetime, timezone

from furnace_data.dataset.refresh_service import (
    DatasetRefreshDecision,
    DatasetRefreshPolicy,
)


def test_static_status_returns_active_version_metadata(client, monkeypatch):
    from app.routes import datasets_v1

    decision = DatasetRefreshDecision(
        state="fresh",
        latest_version_id="version-1",
        active_table="ml_dataset.active_hourly",
        confirmed_start=datetime(2026, 5, 1, tzinfo=timezone.utc),
        confirmed_end=datetime(2026, 5, 15, tzinfo=timezone.utc),
        last_refresh_at=datetime(2026, 5, 15, 10, tzinfo=timezone.utc),
        message="Dataset is fresh.",
    )
    monkeypatch.setattr(datasets_v1.refresh_service, "get_static_status", lambda **kwargs: decision)

    resp = client.get("/api/v1/datasets/static/status")

    assert resp.status_code == 200
    body = resp.json()
    assert body["state"] == "fresh"
    assert body["latest_version_id"] == "version-1"
    assert body["active_table"] == "ml_dataset.active_hourly"


def test_static_status_auto_enqueue_starts_background_job(client, monkeypatch):
    from app.routes import datasets_v1

    calls: list[str] = []
    decision = DatasetRefreshDecision(
        state="refresh_queued",
        active_table="ml_dataset.active_hourly",
        run_id="run-1",
        message="Dataset refresh queued.",
    )
    monkeypatch.setattr(datasets_v1.refresh_service, "get_static_status", lambda **kwargs: decision)
    monkeypatch.setattr(datasets_v1.refresh_service, "run_refresh_job", lambda run_id: calls.append(run_id))

    resp = client.get("/api/v1/datasets/static/status?auto_enqueue=true")

    assert resp.status_code == 200
    assert resp.json()["state"] == "refreshing"
    assert calls == ["run-1"]


def test_static_status_maps_stale_decision(client, monkeypatch):
    from app.routes import datasets_v1

    decision = DatasetRefreshDecision(
        state="skipped_recent_refresh",
        active_table="ml_dataset.active_hourly",
        message="Dataset is stale.",
    )
    monkeypatch.setattr(datasets_v1.refresh_service, "get_static_status", lambda **kwargs: decision)

    resp = client.get("/api/v1/datasets/static/status")

    assert resp.status_code == 200
    assert resp.json()["state"] == "stale"


def test_manual_refresh_respects_role_policy(client, monkeypatch):
    from app.routes import datasets_v1

    monkeypatch.setattr(
        datasets_v1.refresh_service,
        "load_refresh_policy",
        lambda: DatasetRefreshPolicy(allow_manual_refresh_roles=("admin",)),
    )

    denied = client.post(
        "/api/v1/datasets/static/refresh",
        json={"trigger_type": "manual", "triggered_by": "operator", "force": True},
        headers={"X-User-Role": "user"},
    )
    assert denied.status_code == 403

    decision = DatasetRefreshDecision(
        state="already_refreshing",
        active_table="ml_dataset.active_hourly",
        run_id="run-2",
        message="Dataset refresh is already queued or running.",
    )
    monkeypatch.setattr(datasets_v1.refresh_service, "ensure_dataset_fresh", lambda **kwargs: decision)

    allowed = client.post(
        "/api/v1/datasets/static/refresh",
        json={"trigger_type": "manual", "triggered_by": "admin", "force": True},
        headers={"X-User-Role": "admin"},
    )

    assert allowed.status_code == 200
    assert allowed.json()["state"] == "refreshing"


def test_refresh_run_endpoint_returns_metadata(client, monkeypatch):
    from app.routes import datasets_v1

    monkeypatch.setattr(
        datasets_v1.refresh_service,
        "get_refresh_run",
        lambda run_id: {"run_id": run_id, "status": "success"},
    )

    resp = client.get("/api/v1/datasets/refresh-runs/run-3")

    assert resp.status_code == 200
    assert resp.json() == {"run_id": "run-3", "status": "success"}


def test_refresh_run_endpoint_returns_404(client, monkeypatch):
    from app.routes import datasets_v1

    monkeypatch.setattr(datasets_v1.refresh_service, "get_refresh_run", lambda run_id: None)

    resp = client.get("/api/v1/datasets/refresh-runs/missing-run")

    assert resp.status_code == 404
