"""Backend-managed static ML dataset refresh service."""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Literal
from uuid import UUID, uuid4

import pandas as pd
from sqlalchemy import select, text, update
from sqlalchemy.exc import IntegrityError

from furnace_data.config import load_config
from furnace_data.dataset.cleaning import DataCleaner, build_default_config
from furnace_data.dataset.fetcher import DatasetFetcher
from furnace_data.relational.engine import (
    build_relational_engine,
    build_relational_session_factory,
)
from furnace_data.relational.models import DatasetRefreshRun, DatasetVersion

log = logging.getLogger(__name__)

DATASET_NAME = "static_ml_dataset"
DEFAULT_RM_CHOICE = "Full"
DEFAULT_ACTIVE_TABLE = "ml_dataset.active_hourly"
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

RefreshTrigger = Literal["page_hit", "manual", "schedule"]
DecisionState = Literal[
    "fresh",
    "refresh_queued",
    "already_refreshing",
    "skipped_recent_refresh",
    "failed",
]


@dataclass(frozen=True)
class DatasetRefreshPolicy:
    """Configuration for backend dataset refresh decisions."""

    enabled: bool = True
    auto_enqueue_on_page_hit: bool = True
    min_refresh_interval_hours: int = 6
    offline_lag_days: int = 2
    active_table: str = DEFAULT_ACTIVE_TABLE
    keep_versions: int = 3
    allow_manual_refresh_roles: tuple[str, ...] = ("admin", "supervisor")
    stale_running_timeout_hours: int = 12


@dataclass(frozen=True)
class DatasetRefreshDecision:
    """Serializable status/decision returned by the refresh service."""

    state: DecisionState
    latest_version_id: str | None = None
    active_table: str = DEFAULT_ACTIVE_TABLE
    confirmed_start: datetime | None = None
    confirmed_end: datetime | None = None
    last_refresh_at: datetime | None = None
    run_id: str | None = None
    message: str = ""
    error_message: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def load_refresh_policy() -> DatasetRefreshPolicy:
    """Load refresh policy from ``setting_ds_dv.yml`` with safe defaults."""

    cfg = load_config("setting_ds_dv.yml").get("dataset_refresh", {}) or {}
    allowed = cfg.get("allow_manual_refresh_roles", ("admin", "supervisor"))
    return DatasetRefreshPolicy(
        enabled=bool(cfg.get("enabled", True)),
        auto_enqueue_on_page_hit=bool(cfg.get("auto_enqueue_on_page_hit", True)),
        min_refresh_interval_hours=int(cfg.get("min_refresh_interval_hours", 6)),
        offline_lag_days=int(cfg.get("offline_lag_days", 2)),
        active_table=str(cfg.get("active_table", DEFAULT_ACTIVE_TABLE)),
        keep_versions=int(cfg.get("keep_versions", 3)),
        allow_manual_refresh_roles=tuple(str(role).lower() for role in allowed),
        stale_running_timeout_hours=int(cfg.get("stale_running_timeout_hours", 12)),
    )


def ensure_dataset_fresh(
    *,
    trigger_type: RefreshTrigger,
    triggered_by: str | None,
    force: bool = False,
    rm_choice: str = DEFAULT_RM_CHOICE,
) -> DatasetRefreshDecision:
    """Check static dataset freshness and enqueue one DB-locked refresh if needed."""

    policy = load_refresh_policy()
    if not policy.enabled:
        return _status_from_db(policy, rm_choice=rm_choice, state="failed", message="Dataset refresh is disabled.")

    engine = build_relational_engine()
    SessionLocal = build_relational_session_factory(engine)
    try:
        with SessionLocal() as session:
            _fail_stale_running_jobs(session, policy)
            session.commit()

            running = _running_run(session, rm_choice)
            latest = _latest_active_version(session, rm_choice)
            latest_run = _latest_run(session, rm_choice)

            if running is not None:
                return _decision_from_rows(
                    policy,
                    latest=latest,
                    run=running,
                    state="already_refreshing",
                    message="Dataset refresh is already queued or running.",
                )

            if latest is not None and _version_is_fresh(latest, policy):
                return _decision_from_rows(
                    policy,
                    latest=latest,
                    run=latest_run,
                    state="fresh",
                    message="Dataset is fresh.",
                )

            if (
                not force
                and latest_run is not None
                and _run_finished_recently(latest_run, policy)
            ):
                return _decision_from_rows(
                    policy,
                    latest=latest,
                    run=latest_run,
                    state="skipped_recent_refresh",
                    message="Dataset is stale, but a recent refresh attempt already ran.",
                )

            run = DatasetRefreshRun(
                dataset_name=DATASET_NAME,
                rm_choice=rm_choice,
                trigger_type=trigger_type,
                triggered_by=triggered_by,
                status="queued",
                started_at=datetime.now(timezone.utc),
                metadata_json={"force": force},
            )
            session.add(run)
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
                running = _running_run(session, rm_choice)
                return _decision_from_rows(
                    policy,
                    latest=latest,
                    run=running,
                    state="already_refreshing",
                    message="Dataset refresh is already queued or running.",
                )

            return _decision_from_rows(
                policy,
                latest=latest,
                run=run,
                state="refresh_queued",
                message="Dataset refresh queued.",
            )
    except Exception as exc:
        log.exception("Dataset freshness check failed.")
        return DatasetRefreshDecision(
            state="failed",
            active_table=policy.active_table,
            message="Dataset freshness check failed.",
            error_message=str(exc),
        )
    finally:
        engine.dispose()


def get_static_status(
    *,
    auto_enqueue: bool = False,
    triggered_by: str | None = None,
    rm_choice: str = DEFAULT_RM_CHOICE,
) -> DatasetRefreshDecision:
    """Return status, optionally asking the backend to enqueue refresh-on-stale."""

    policy = load_refresh_policy()
    if auto_enqueue and policy.auto_enqueue_on_page_hit:
        return ensure_dataset_fresh(
            trigger_type="page_hit",
            triggered_by=triggered_by,
            force=False,
            rm_choice=rm_choice,
        )

    engine = build_relational_engine()
    SessionLocal = build_relational_session_factory(engine)
    try:
        with SessionLocal() as session:
            _fail_stale_running_jobs(session, policy)
            session.commit()
            running = _running_run(session, rm_choice)
            latest = _latest_active_version(session, rm_choice)
            latest_run = _latest_run(session, rm_choice)
            if running is not None:
                return _decision_from_rows(
                    policy,
                    latest=latest,
                    run=running,
                    state="already_refreshing",
                    message="Dataset refresh is queued or running.",
                )
            if latest is not None and _version_is_fresh(latest, policy):
                return _decision_from_rows(
                    policy,
                    latest=latest,
                    run=latest_run,
                    state="fresh",
                    message="Dataset is fresh.",
                )
            if latest_run is not None and latest_run.status == "failed":
                state: DecisionState = "failed"
                message = "Latest dataset refresh failed."
            else:
                state = "skipped_recent_refresh"
                message = "Dataset is stale."
            return _decision_from_rows(
                policy,
                latest=latest,
                run=latest_run,
                state=state,
                message=message,
            )
    finally:
        engine.dispose()


def get_refresh_run(run_id: str | UUID) -> dict | None:
    """Return refresh-run metadata by id."""

    engine = build_relational_engine()
    SessionLocal = build_relational_session_factory(engine)
    try:
        with SessionLocal() as session:
            run = session.get(DatasetRefreshRun, UUID(str(run_id)))
            if run is None:
                return None
            return _run_dict(run)
    finally:
        engine.dispose()


def run_refresh_job(run_id: str | UUID) -> None:
    """Build, clean, publish, and activate a static ML dataset version."""

    policy = load_refresh_policy()
    engine = build_relational_engine()
    SessionLocal = build_relational_session_factory(engine)
    version_id = uuid4()
    table_name = f"static_hourly_v_{version_id.hex[:12]}"
    target_table = f"ml_dataset.{table_name}"

    try:
        with SessionLocal() as session:
            run = session.get(DatasetRefreshRun, UUID(str(run_id)))
            if run is None:
                raise ValueError(f"Refresh run not found: {run_id}")
            if run.status not in {"queued", "running"}:
                return
            run.status = "running"
            run.started_at = datetime.now(timezone.utc)
            session.commit()
            rm_choice = run.rm_choice

        df = _build_clean_static_dataset(rm_choice)
        _write_version_table(engine, df, schema="ml_dataset", table_name=table_name)
        confirmed_start, confirmed_end = _confirmed_window(df)

        with SessionLocal.begin() as session:
            version = DatasetVersion(
                version_id=version_id,
                dataset_name=DATASET_NAME,
                rm_choice=rm_choice,
                storage_mode="table",
                target_table=target_table,
                row_count=len(df),
                confirmed_start=confirmed_start,
                confirmed_end=confirmed_end,
                status="building",
                metadata_json={"columns": list(df.columns)},
            )
            session.add(version)
            session.flush()

            session.execute(
                update(DatasetVersion)
                .where(
                    DatasetVersion.dataset_name == DATASET_NAME,
                    DatasetVersion.rm_choice == rm_choice,
                    DatasetVersion.status == "active",
                    DatasetVersion.version_id != version_id,
                )
                .values(status="superseded")
            )
            version.status = "active"
            version.activated_at = datetime.now(timezone.utc)
            _replace_active_view(session, policy.active_table, target_table)

            run = session.get(DatasetRefreshRun, UUID(str(run_id)))
            if run is not None:
                run.status = "success"
                run.output_version_id = version_id
                run.finished_at = datetime.now(timezone.utc)
                run.error_message = None

        _drop_old_version_tables(engine, rm_choice=rm_choice, keep_versions=policy.keep_versions)
    except Exception as exc:
        log.exception("Dataset refresh failed for run %s.", run_id)
        _mark_run_failed(SessionLocal, run_id, str(exc))
        _drop_table_if_exists(engine, target_table)
    finally:
        engine.dispose()


def _build_clean_static_dataset(rm_choice: str) -> pd.DataFrame:
    rm_label = "RM DPR" if str(rm_choice).strip().lower() in {"dpr", "rm dpr"} else "RM Charge"
    today = date.today()
    df = DatasetFetcher().get_dataset(
        start_date=date(2023, 1, 1),
        end_date=today,
        rm_choice=rm_label,
        cache_override=True,
    )
    if df.empty:
        raise ValueError("Static dataset fetch returned no rows.")
    cleaned = DataCleaner(build_default_config()).clean(df)
    if cleaned.empty:
        raise ValueError("Static dataset cleaning returned no rows.")
    return cleaned.sort_index()


def _write_version_table(engine, df: pd.DataFrame, *, schema: str, table_name: str) -> None:
    df_out = df.copy()
    if isinstance(df_out.index, pd.DatetimeIndex):
        df_out.index.name = df_out.index.name or "time"
    else:
        df_out.index.name = "time"
    df_out.reset_index().to_sql(
        table_name,
        engine,
        schema=schema,
        if_exists="fail",
        index=False,
        chunksize=1000,
        method="multi",
    )


def _confirmed_window(df: pd.DataFrame) -> tuple[datetime | None, datetime | None]:
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return None, None
    start = pd.Timestamp(df.index.min()).to_pydatetime()
    end = pd.Timestamp(df.index.max()).to_pydatetime()
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    return start, end


def _fail_stale_running_jobs(session, policy: DatasetRefreshPolicy) -> None:
    cutoff = datetime.now(timezone.utc) - timedelta(
        hours=policy.stale_running_timeout_hours
    )
    stale_runs = session.scalars(
        select(DatasetRefreshRun).where(
            DatasetRefreshRun.status.in_(("queued", "running")),
            DatasetRefreshRun.started_at.is_not(None),
            DatasetRefreshRun.started_at < cutoff,
        )
    ).all()
    for run in stale_runs:
        run.status = "failed"
        run.finished_at = run.started_at
        run.error_message = "Refresh marked failed after stale running timeout."


def _latest_active_version(session, rm_choice: str) -> DatasetVersion | None:
    return session.scalars(
        select(DatasetVersion)
        .where(
            DatasetVersion.dataset_name == DATASET_NAME,
            DatasetVersion.rm_choice == rm_choice,
            DatasetVersion.status == "active",
        )
        .order_by(DatasetVersion.activated_at.desc().nulls_last())
        .limit(1)
    ).first()


def _running_run(session, rm_choice: str) -> DatasetRefreshRun | None:
    return session.scalars(
        select(DatasetRefreshRun)
        .where(
            DatasetRefreshRun.dataset_name == DATASET_NAME,
            DatasetRefreshRun.rm_choice == rm_choice,
            DatasetRefreshRun.status.in_(("queued", "running")),
        )
        .order_by(DatasetRefreshRun.started_at.desc().nulls_last())
        .limit(1)
    ).first()


def _latest_run(session, rm_choice: str) -> DatasetRefreshRun | None:
    return session.scalars(
        select(DatasetRefreshRun)
        .where(
            DatasetRefreshRun.dataset_name == DATASET_NAME,
            DatasetRefreshRun.rm_choice == rm_choice,
        )
        .order_by(DatasetRefreshRun.started_at.desc().nulls_last(), DatasetRefreshRun.finished_at.desc().nulls_last())
        .limit(1)
    ).first()


def _version_is_fresh(version: DatasetVersion, policy: DatasetRefreshPolicy) -> bool:
    if version.confirmed_end is None:
        return False
    confirmed_end = version.confirmed_end
    if confirmed_end.tzinfo is None:
        confirmed_end = confirmed_end.replace(tzinfo=timezone.utc)
    target = datetime.now(timezone.utc) - timedelta(days=policy.offline_lag_days)
    return confirmed_end >= target


def _run_finished_recently(
    run: DatasetRefreshRun,
    policy: DatasetRefreshPolicy,
) -> bool:
    finished_at = run.finished_at or run.started_at
    if finished_at is None:
        return False
    if finished_at.tzinfo is None:
        finished_at = finished_at.replace(tzinfo=timezone.utc)
    return finished_at >= datetime.now(timezone.utc) - timedelta(
        hours=policy.min_refresh_interval_hours
    )


def _decision_from_rows(
    policy: DatasetRefreshPolicy,
    *,
    latest: DatasetVersion | None,
    run: DatasetRefreshRun | None,
    state: DecisionState,
    message: str,
) -> DatasetRefreshDecision:
    return DatasetRefreshDecision(
        state=state,
        latest_version_id=str(latest.version_id) if latest else None,
        active_table=policy.active_table,
        confirmed_start=latest.confirmed_start if latest else None,
        confirmed_end=latest.confirmed_end if latest else None,
        last_refresh_at=(latest.activated_at if latest else None),
        run_id=str(run.run_id) if run else None,
        message=message,
        error_message=run.error_message if run else None,
    )


def _status_from_db(
    policy: DatasetRefreshPolicy,
    *,
    rm_choice: str,
    state: DecisionState,
    message: str,
) -> DatasetRefreshDecision:
    engine = build_relational_engine()
    SessionLocal = build_relational_session_factory(engine)
    try:
        with SessionLocal() as session:
            return _decision_from_rows(
                policy,
                latest=_latest_active_version(session, rm_choice),
                run=_latest_run(session, rm_choice),
                state=state,
                message=message,
            )
    finally:
        engine.dispose()


def _run_dict(run: DatasetRefreshRun) -> dict:
    return {
        "run_id": str(run.run_id),
        "dataset_name": run.dataset_name,
        "rm_choice": run.rm_choice,
        "trigger_type": run.trigger_type,
        "triggered_by": run.triggered_by,
        "status": run.status,
        "started_at": run.started_at,
        "finished_at": run.finished_at,
        "output_version_id": str(run.output_version_id) if run.output_version_id else None,
        "error_message": run.error_message,
        "metadata": run.metadata_json,
    }


def _mark_run_failed(SessionLocal, run_id: str | UUID, error_message: str) -> None:
    try:
        with SessionLocal.begin() as session:
            run = session.get(DatasetRefreshRun, UUID(str(run_id)))
            if run is not None:
                run.status = "failed"
                run.finished_at = datetime.now(timezone.utc)
                run.error_message = error_message[:4000]
    except Exception:
        log.exception("Could not mark dataset refresh run failed.")


def _replace_active_view(session, active_table: str, target_table: str) -> None:
    active_sql = _qualified_name_sql(active_table)
    target_sql = _qualified_name_sql(target_table)
    session.execute(text(f"DROP VIEW IF EXISTS {active_sql}"))
    session.execute(text(f"CREATE VIEW {active_sql} AS SELECT * FROM {target_sql}"))


def _drop_old_version_tables(engine, *, rm_choice: str, keep_versions: int) -> None:
    if keep_versions <= 0:
        keep_versions = 1
    SessionLocal = build_relational_session_factory(engine)
    with SessionLocal() as session:
        versions = session.scalars(
            select(DatasetVersion)
            .where(
                DatasetVersion.dataset_name == DATASET_NAME,
                DatasetVersion.rm_choice == rm_choice,
                DatasetVersion.target_table.is_not(None),
                DatasetVersion.status != "active",
            )
            .order_by(DatasetVersion.created_at.desc())
        ).all()
    old_versions = versions[max(keep_versions - 1, 0):]
    dropped: list[tuple[UUID, str]] = []
    for version in old_versions:
        if version.target_table and _drop_table_if_exists(engine, version.target_table):
            dropped.append((version.version_id, version.target_table))

    if not dropped:
        return

    with SessionLocal.begin() as session:
        for version_id, target_table in dropped:
            version = session.get(DatasetVersion, version_id)
            if version is None or version.status == "active":
                continue
            metadata = dict(version.metadata_json or {})
            metadata["physical_table_dropped"] = True
            metadata["dropped_target_table"] = target_table
            version.metadata_json = metadata
            version.target_table = None


def _drop_table_if_exists(engine, qualified_table: str) -> bool:
    try:
        with engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {_qualified_name_sql(qualified_table)}"))
        return True
    except Exception:
        log.warning("Could not drop dataset table %s", qualified_table, exc_info=True)
        return False


def _qualified_name_sql(qualified_name: str) -> str:
    parts = qualified_name.split(".")
    if len(parts) != 2 or not all(_IDENTIFIER_RE.match(part) for part in parts):
        raise ValueError(f"Invalid schema-qualified name: {qualified_name!r}")
    return ".".join(f'"{part}"' for part in parts)
