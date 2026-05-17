from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest
from sqlalchemy import create_engine, select, text

from furnace_data.dataset import refresh_service
from furnace_data.relational.engine import build_relational_session_factory
from furnace_data.relational.models import Base, DatasetRefreshRun, DatasetVersion


ROOT = Path(__file__).resolve().parents[1]


def test_dataset_refresh_migration_defines_schema_and_partial_indexes() -> None:
    migration = (
        ROOT
        / "alembic"
        / "versions"
        / "20260517_0002_create_dataset_refresh_tables.py"
    ).read_text(encoding="utf-8")

    assert "CREATE SCHEMA IF NOT EXISTS ml_dataset" in migration
    assert "dataset_versions" in migration
    assert "dataset_refresh_runs" in migration
    assert "status = 'active'" in migration
    assert "status IN ('queued', 'running')" in migration


def test_custom_pages_do_not_call_process_local_dataset_refresher() -> None:
    offenders = []
    for path in (ROOT / "src" / "custom_pages").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "utils.dataset_refresher" in text or "maybe_refresh(" in text:
            offenders.append(path.name)

    assert offenders == []


def test_dataset_version_model_uses_ml_dataset_schema() -> None:
    assert DatasetVersion.__table__.schema == "ml_dataset"
    assert DatasetVersion.__tablename__ == "dataset_versions"
    assert any(index.name == "uq_dataset_versions_active" for index in DatasetVersion.__table__.indexes)


def test_refresh_service_identifier_validation() -> None:
    assert (
        refresh_service._qualified_name_sql("ml_dataset.active_hourly")
        == '"ml_dataset"."active_hourly"'
    )
    with pytest.raises(ValueError):
        refresh_service._qualified_name_sql("ml_dataset.active_hourly;drop")


def test_refresh_service_freshness_uses_offline_lag() -> None:
    policy = refresh_service.DatasetRefreshPolicy(offline_lag_days=2)
    fresh_version = DatasetVersion(
        version_id=uuid4(),
        dataset_name=refresh_service.DATASET_NAME,
        rm_choice="Full",
        storage_mode="table",
        target_table="ml_dataset.static_hourly_v_test",
        row_count=10,
        confirmed_end=datetime.now(timezone.utc) - timedelta(days=1),
        status="active",
    )
    stale_version = DatasetVersion(
        version_id=uuid4(),
        dataset_name=refresh_service.DATASET_NAME,
        rm_choice="Full",
        storage_mode="table",
        target_table="ml_dataset.static_hourly_v_old",
        row_count=10,
        confirmed_end=datetime.now(timezone.utc) - timedelta(days=5),
        status="active",
    )

    assert refresh_service._version_is_fresh(fresh_version, policy) is True
    assert refresh_service._version_is_fresh(stale_version, policy) is False


@pytest.fixture()
def refresh_db(monkeypatch):
    engine = create_engine("sqlite:///:memory:", future=True)
    monkeypatch.setattr(engine, "dispose", lambda: None)
    with engine.begin() as conn:
        conn.execute(text("ATTACH DATABASE ':memory:' AS ml_dataset"))
        Base.metadata.create_all(
            conn,
            tables=[
                Base.metadata.tables["ml_dataset.dataset_versions"],
                Base.metadata.tables["ml_dataset.dataset_refresh_runs"],
            ],
        )
        conn.execute(text("DROP INDEX IF EXISTS ml_dataset.uq_dataset_versions_active"))
        conn.execute(
            text(
                "CREATE UNIQUE INDEX ml_dataset.uq_dataset_versions_active "
                "ON dataset_versions (dataset_name, rm_choice) "
                "WHERE status = 'active'"
            )
        )
        conn.execute(text("DROP INDEX IF EXISTS ml_dataset.uq_dataset_refresh_running"))
        conn.execute(
            text(
                "CREATE UNIQUE INDEX ml_dataset.uq_dataset_refresh_running "
                "ON dataset_refresh_runs (dataset_name, rm_choice) "
                "WHERE status IN ('queued', 'running')"
            )
        )

    monkeypatch.setattr(refresh_service, "build_relational_engine", lambda: engine)
    monkeypatch.setattr(
        refresh_service,
        "load_refresh_policy",
        lambda: refresh_service.DatasetRefreshPolicy(
            min_refresh_interval_hours=6,
            offline_lag_days=2,
            stale_running_timeout_hours=12,
        ),
    )
    return engine


def test_ensure_dataset_fresh_queues_once_and_then_reports_running(refresh_db) -> None:
    first = refresh_service.ensure_dataset_fresh(
        trigger_type="page_hit",
        triggered_by="admin",
    )
    second = refresh_service.ensure_dataset_fresh(
        trigger_type="page_hit",
        triggered_by="admin",
    )

    assert first.state == "refresh_queued"
    assert first.run_id
    assert second.state == "already_refreshing"
    assert second.run_id == first.run_id


def test_ensure_dataset_fresh_returns_fresh_without_queue(refresh_db) -> None:
    SessionLocal = build_relational_session_factory(refresh_db)
    with SessionLocal.begin() as session:
        session.add(
            DatasetVersion(
                dataset_name=refresh_service.DATASET_NAME,
                rm_choice="Full",
                storage_mode="table",
                target_table="ml_dataset.static_hourly_v_current",
                row_count=10,
                confirmed_start=datetime.now(timezone.utc) - timedelta(days=5),
                confirmed_end=datetime.now(timezone.utc),
                status="active",
                activated_at=datetime.now(timezone.utc),
            )
        )

    decision = refresh_service.ensure_dataset_fresh(
        trigger_type="page_hit",
        triggered_by="admin",
    )

    assert decision.state == "fresh"
    assert decision.latest_version_id
    assert decision.run_id is None


def test_stale_recent_failed_run_skips_requeue_until_interval(refresh_db) -> None:
    SessionLocal = build_relational_session_factory(refresh_db)
    with SessionLocal.begin() as session:
        session.add(
            DatasetVersion(
                dataset_name=refresh_service.DATASET_NAME,
                rm_choice="Full",
                storage_mode="table",
                target_table="ml_dataset.static_hourly_v_old",
                row_count=10,
                confirmed_end=datetime.now(timezone.utc) - timedelta(days=5),
                status="active",
                activated_at=datetime.now(timezone.utc) - timedelta(days=5),
            )
        )
        session.add(
            DatasetRefreshRun(
                dataset_name=refresh_service.DATASET_NAME,
                rm_choice="Full",
                trigger_type="page_hit",
                status="failed",
                started_at=datetime.now(timezone.utc) - timedelta(minutes=5),
                finished_at=datetime.now(timezone.utc) - timedelta(minutes=4),
                error_message="boom",
            )
        )

    decision = refresh_service.ensure_dataset_fresh(
        trigger_type="page_hit",
        triggered_by="admin",
    )

    assert decision.state == "skipped_recent_refresh"
    assert decision.error_message == "boom"


def test_stale_running_job_is_failed_before_new_queue(refresh_db) -> None:
    old_run_id = uuid4()
    SessionLocal = build_relational_session_factory(refresh_db)
    with SessionLocal.begin() as session:
        session.add(
            DatasetRefreshRun(
                run_id=old_run_id,
                dataset_name=refresh_service.DATASET_NAME,
                rm_choice="Full",
                trigger_type="page_hit",
                status="running",
                started_at=datetime.now(timezone.utc) - timedelta(hours=13),
            )
        )

    decision = refresh_service.ensure_dataset_fresh(
        trigger_type="page_hit",
        triggered_by="admin",
    )

    with SessionLocal() as session:
        old_run = session.get(DatasetRefreshRun, old_run_id)

    assert decision.state == "refresh_queued"
    assert old_run.status == "failed"
    assert decision.run_id != str(old_run_id)


def test_get_static_status_reports_stale_without_enqueueing(refresh_db) -> None:
    SessionLocal = build_relational_session_factory(refresh_db)
    with SessionLocal.begin() as session:
        session.add(
            DatasetVersion(
                dataset_name=refresh_service.DATASET_NAME,
                rm_choice="Full",
                storage_mode="table",
                target_table="ml_dataset.static_hourly_v_old",
                row_count=10,
                confirmed_end=datetime.now(timezone.utc) - timedelta(days=5),
                status="active",
                activated_at=datetime.now(timezone.utc) - timedelta(days=5),
            )
        )

    decision = refresh_service.get_static_status(auto_enqueue=False)

    with SessionLocal() as session:
        runs = session.scalars(select(DatasetRefreshRun)).all()

    assert decision.state == "skipped_recent_refresh"
    assert decision.message == "Dataset is stale."
    assert runs == []


def test_run_refresh_job_publishes_active_version_and_view(refresh_db, monkeypatch) -> None:
    run_id = uuid4()
    SessionLocal = build_relational_session_factory(refresh_db)
    with SessionLocal.begin() as session:
        session.add(
            DatasetRefreshRun(
                run_id=run_id,
                dataset_name=refresh_service.DATASET_NAME,
                rm_choice="Full",
                trigger_type="page_hit",
                status="queued",
                started_at=datetime.now(timezone.utc),
            )
        )

    df = pd.DataFrame(
        {"fuel_rate": [500.0]},
        index=pd.DatetimeIndex(["2026-05-05 06:00:00"], name="time"),
    )
    monkeypatch.setattr(refresh_service, "_build_clean_static_dataset", lambda rm_choice: df)

    refresh_service.run_refresh_job(run_id)

    with SessionLocal() as session:
        run = session.get(DatasetRefreshRun, run_id)
        version = session.get(DatasetVersion, run.output_version_id)
    with refresh_db.connect() as conn:
        rows = conn.execute(text("SELECT fuel_rate FROM ml_dataset.active_hourly")).all()

    assert run.status == "success"
    assert version.status == "active"
    assert version.row_count == 1
    assert rows == [(500.0,)]


def test_refresh_retention_keeps_active_and_marks_dropped_tables(
    refresh_db,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        refresh_service,
        "load_refresh_policy",
        lambda: refresh_service.DatasetRefreshPolicy(
            min_refresh_interval_hours=6,
            offline_lag_days=2,
            keep_versions=2,
            stale_running_timeout_hours=12,
        ),
    )
    SessionLocal = build_relational_session_factory(refresh_db)
    counter = {"value": 0}

    def build_df(rm_choice):
        counter["value"] += 1
        return pd.DataFrame(
            {"fuel_rate": [float(counter["value"])]},
            index=pd.DatetimeIndex(
                [f"2026-05-{counter['value']:02d} 06:00:00"],
                name="time",
            ),
        )

    monkeypatch.setattr(refresh_service, "_build_clean_static_dataset", build_df)

    for _ in range(4):
        run_id = uuid4()
        with SessionLocal.begin() as session:
            session.add(
                DatasetRefreshRun(
                    run_id=run_id,
                    dataset_name=refresh_service.DATASET_NAME,
                    rm_choice="Full",
                    trigger_type="page_hit",
                    status="queued",
                    started_at=datetime.now(timezone.utc),
                )
            )
        refresh_service.run_refresh_job(run_id)

    with SessionLocal() as session:
        versions = session.scalars(
            select(DatasetVersion).order_by(DatasetVersion.created_at)
        ).all()
        active = session.scalars(
            select(DatasetVersion).where(DatasetVersion.status == "active")
        ).one()
    with refresh_db.connect() as conn:
        tables = conn.execute(
            text(
                "SELECT name FROM ml_dataset.sqlite_master "
                "WHERE type = 'table' AND name LIKE 'static_hourly_v_%'"
            )
        ).scalars().all()
        active_rows = conn.execute(
            text("SELECT fuel_rate FROM ml_dataset.active_hourly")
        ).all()

    retained = [version for version in versions if version.target_table]
    dropped = [version for version in versions if version.target_table is None]

    assert len(versions) == 4
    assert len(retained) == 2
    assert len(dropped) == 2
    assert active.target_table is not None
    assert len(tables) == 2
    assert active_rows == [(4.0,)]
    assert all(version.status == "superseded" for version in dropped)
    assert all(version.metadata_json["physical_table_dropped"] is True for version in dropped)


def test_run_refresh_job_failure_keeps_no_active_version(refresh_db, monkeypatch) -> None:
    run_id = uuid4()
    SessionLocal = build_relational_session_factory(refresh_db)
    with SessionLocal.begin() as session:
        session.add(
            DatasetRefreshRun(
                run_id=run_id,
                dataset_name=refresh_service.DATASET_NAME,
                rm_choice="Full",
                trigger_type="page_hit",
                status="queued",
                started_at=datetime.now(timezone.utc),
            )
        )

    def fail_build(rm_choice):
        raise RuntimeError("cannot build")

    monkeypatch.setattr(refresh_service, "_build_clean_static_dataset", fail_build)

    refresh_service.run_refresh_job(run_id)

    with SessionLocal() as session:
        run = session.get(DatasetRefreshRun, run_id)
        active = session.scalars(
            select(DatasetVersion).where(DatasetVersion.status == "active")
        ).all()

    assert run.status == "failed"
    assert "cannot build" in run.error_message
    assert active == []
