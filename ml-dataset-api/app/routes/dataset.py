"""Dataset API routes."""

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.config import settings
from app.core.data_cleaning import DataCleaner, build_default_config
from app.core.dataset_fetcher import MlDatasetFetcher
from app.core.static_manager import StaticDatasetManager
from app.models.schemas import (
    FetchDatasetRequest,
    TaskCreatedResponse,
    TaskStatusResponse,
    UpdateStaticRequest,
)
from app.tasks.task_manager import TaskState, task_manager

log = logging.getLogger(__name__)

router = APIRouter(prefix="/dataset", tags=["dataset"])

_fetcher = MlDatasetFetcher()


def _make_manager() -> StaticDatasetManager:
    return StaticDatasetManager(
        static_dir=settings.static_dir,
        offline_lag_days=settings.offline_lag_days,
        max_versions=settings.static_max_versions,
        legacy_csv_path=settings.legacy_csv_path or None,
    )


# ---------- Background task functions ----------

def _run_fetch(task: TaskState, req: FetchDatasetRequest):
    task.progress = "Fetching ML dataset..."
    rm_label = "RM Charge" if req.rm_choice.value == "charge" else "RM DPR"

    df = _fetcher.get_ml_dataset(
        start_date=req.start_date,
        end_date=req.end_date,
        rm_choice=rm_label,
        cache_override=not req.use_cache,
    )

    if df.empty:
        raise ValueError("No data returned for the requested date range.")

    if req.apply_cleaning:
        task.progress = "Cleaning dataset..."
        df = DataCleaner(build_default_config()).clean(df)

    task.progress = "Saving result..."
    task_manager.save_result(task, df)


def _run_update_static(task: TaskState, req: UpdateStaticRequest):
    task.progress = "Updating static dataset..."
    rm_label = "RM Charge" if req.rm_choice.value == "charge" else "RM DPR"

    manager = _make_manager()
    df = manager.update_static(
        rm_choice=rm_label,
        reprocess_from=req.reprocess_from,
        apply_cleaning=req.apply_cleaning,
    )

    if df.empty:
        raise ValueError("No data available after update.")

    task.progress = "Saving static dataset..."
    csv_path = manager.save(df, rm_label)

    # Also expose as downloadable task result
    task_manager.save_result(task, df)


# ---------- Endpoints ----------

@router.post("/fetch", response_model=TaskCreatedResponse)
def fetch_dataset(req: FetchDatasetRequest):
    """Trigger an ad-hoc ML dataset fetch for a date range. Returns task_id to poll."""
    if req.start_date > req.end_date:
        raise HTTPException(status_code=400, detail="start_date must be <= end_date")

    task = task_manager.create_task(callback_url=req.callback_url)
    task_manager.run_in_background(task, _run_fetch, req)
    return TaskCreatedResponse(task_id=task.task_id)


@router.post("/update-static", response_model=TaskCreatedResponse)
def update_static(req: UpdateStaticRequest):
    """
    Trigger a smart incremental update of the static ML dataset CSV.

    Uses lag-aware caching:
    - Rows up to (last_raw_end - offline_lag_days) are frozen.
    - The uncertain window is re-fetched to pick up delayed offline data.
    """
    task = task_manager.create_task(callback_url=req.callback_url)
    task_manager.run_in_background(task, _run_update_static, req)
    return TaskCreatedResponse(task_id=task.task_id)


@router.get("/status/{task_id}", response_model=TaskStatusResponse)
def get_task_status(task_id: str):
    """Poll the status of a running or completed task."""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    return TaskStatusResponse(
        task_id=task.task_id,
        status=task.status,
        progress=task.progress,
        created_at=task.created_at,
        completed_at=task.completed_at,
        rows=task.rows,
        columns=task.columns,
        error=task.error,
    )


@router.get("/download/{task_id}")
def download_result(task_id: str):
    """Download the result CSV for a completed ad-hoc fetch task."""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    if task.status.value != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Task is {task.status.value}, not yet completed",
        )

    if not task.result_path or not task.result_path.exists():
        raise HTTPException(status_code=410, detail="Result file no longer available")

    return FileResponse(
        path=task.result_path,
        media_type="text/csv",
        filename=f"ml_dataset_{task_id}.csv",
    )


@router.get("/static")
def download_static():
    """Download the current static ML dataset CSV."""
    manager = _make_manager()
    csv_path = manager.current_csv_path()

    if not csv_path:
        raise HTTPException(
            status_code=404,
            detail="No static dataset found. Run /dataset/update-static first.",
        )

    return FileResponse(
        path=csv_path,
        media_type="text/csv",
        filename=csv_path.name,
    )


@router.get("/cache-info")
def cache_info() -> Dict[str, Any]:
    """
    Return metadata about the current static cache:
    confirmed_end, raw_end, offline_lag_days, rows, last_updated, csv_file.
    """
    manager = _make_manager()
    meta = manager.get_meta()

    if not meta:
        return {"status": "no_cache", "detail": "No cache_meta.json found. Run /dataset/update-static first."}

    return {
        "status": "ok",
        "data_start": meta.data_start,
        "confirmed_end": meta.confirmed_end,
        "raw_end": meta.raw_end,
        "offline_lag_days": meta.offline_lag_days,
        "last_updated": meta.last_updated,
        "rows": meta.rows,
        "columns": meta.columns,
        "csv_file": meta.csv_file,
        "rm_choice": meta.rm_choice,
    }
