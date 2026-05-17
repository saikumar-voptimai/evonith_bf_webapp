"""Canonical v1 static dataset refresh endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Query
from pydantic import BaseModel, Field

from furnace_data.dataset import refresh_service

router = APIRouter(prefix="/api/v1/datasets", tags=["datasets-v1"])

ApiState = Literal["fresh", "stale", "refreshing", "failed"]


class StaticDatasetStatusResponse(BaseModel):
    state: ApiState
    latest_version_id: str | None = None
    active_table: str
    confirmed_start: datetime | None = None
    confirmed_end: datetime | None = None
    last_refresh_at: datetime | None = None
    run_id: str | None = None
    message: str
    error_message: str | None = None


class StaticDatasetRefreshRequest(BaseModel):
    trigger_type: Literal["manual", "schedule", "page_hit"] = "manual"
    triggered_by: str | None = None
    rm_choice: str = "Full"
    force: bool = False


@router.get("/static/status", response_model=StaticDatasetStatusResponse)
def static_dataset_status(
    background_tasks: BackgroundTasks,
    auto_enqueue: bool = Query(default=False),
    rm_choice: str = Query(default="Full"),
    triggered_by: str | None = Query(default=None),
) -> StaticDatasetStatusResponse:
    """Return static dataset freshness, optionally enqueueing one backend refresh."""

    decision = refresh_service.get_static_status(
        auto_enqueue=auto_enqueue,
        triggered_by=triggered_by,
        rm_choice=rm_choice,
    )
    _start_refresh_if_queued(background_tasks, decision.run_id, decision.state)
    return _api_response(decision)


@router.post("/static/refresh", response_model=StaticDatasetStatusResponse)
def refresh_static_dataset(
    req: StaticDatasetRefreshRequest,
    background_tasks: BackgroundTasks,
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> StaticDatasetStatusResponse:
    """Manually or programmatically enqueue a static dataset refresh."""

    if req.trigger_type == "manual":
        policy = refresh_service.load_refresh_policy()
        role = str(x_user_role or "").strip().lower()
        if role not in policy.allow_manual_refresh_roles:
            raise HTTPException(status_code=403, detail="User role cannot refresh datasets.")

    decision = refresh_service.ensure_dataset_fresh(
        trigger_type=req.trigger_type,
        triggered_by=req.triggered_by,
        force=req.force,
        rm_choice=req.rm_choice,
    )
    _start_refresh_if_queued(background_tasks, decision.run_id, decision.state)
    return _api_response(decision)


@router.get("/refresh-runs/{run_id}")
def refresh_run_status(run_id: str) -> dict:
    """Return refresh-run metadata by id."""

    run = refresh_service.get_refresh_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Refresh run {run_id} not found")
    return run


def _start_refresh_if_queued(
    background_tasks: BackgroundTasks,
    run_id: str | None,
    state: str,
) -> None:
    if state == "refresh_queued" and run_id:
        background_tasks.add_task(refresh_service.run_refresh_job, run_id)


def _api_response(decision: refresh_service.DatasetRefreshDecision) -> StaticDatasetStatusResponse:
    api_state: ApiState
    if decision.state == "fresh":
        api_state = "fresh"
    elif decision.state in {"refresh_queued", "already_refreshing"}:
        api_state = "refreshing"
    elif decision.state == "failed":
        api_state = "failed"
    else:
        api_state = "stale"

    return StaticDatasetStatusResponse(
        state=api_state,
        latest_version_id=decision.latest_version_id,
        active_table=decision.active_table,
        confirmed_start=decision.confirmed_start,
        confirmed_end=decision.confirmed_end,
        last_refresh_at=decision.last_refresh_at,
        run_id=decision.run_id,
        message=decision.message,
        error_message=decision.error_message,
    )

