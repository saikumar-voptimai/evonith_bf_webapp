"""Dataset API routes — fetch, status, download, update-static, static."""

import logging

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

# Shared fetcher instance (has its own RangeCache)
_fetcher = MlDatasetFetcher()


# ---------- Background task functions ----------

def _run_fetch(task: TaskState, req: FetchDatasetRequest):
    """Background: fetch ML dataset for a date range."""
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
        cleaner = DataCleaner(build_default_config())
        df = cleaner.clean(df)

    task.progress = "Saving result..."
    task_manager.save_result(task, df)


def _run_update_static(task: TaskState, req: UpdateStaticRequest):
    """Background: incremental update of static CSV."""
    task.progress = "Updating static dataset..."

    rm_label = "RM Charge" if req.rm_choice.value == "charge" else "RM DPR"

    manager = StaticDatasetManager(settings.static_dataset_path)
    df = manager.update_static(
        rm_choice=rm_label,
        start_date=req.reprocess_from,
        apply_cleaning=req.apply_cleaning,
    )

    if df.empty:
        raise ValueError("No data available after update.")

    task.progress = "Saving static dataset..."
    manager.save(df)

    # Also save as a task result for download via /download/{id}
    task_manager.save_result(task, df)


# ---------- Endpoints ----------

@router.post("/fetch", response_model=TaskCreatedResponse)
def fetch_dataset(req: FetchDatasetRequest):
    """Trigger an ML dataset fetch. Returns a task_id to poll for status."""
    if req.start_date > req.end_date:
        raise HTTPException(status_code=400, detail="start_date must be <= end_date")

    task = task_manager.create_task(callback_url=req.callback_url)
    task_manager.run_in_background(task, _run_fetch, req)

    return TaskCreatedResponse(task_id=task.task_id)


@router.post("/update-static", response_model=TaskCreatedResponse)
def update_static(req: UpdateStaticRequest):
    """Trigger an incremental update of the static ML dataset CSV."""
    task = task_manager.create_task(callback_url=req.callback_url)
    task_manager.run_in_background(task, _run_update_static, req)

    return TaskCreatedResponse(task_id=task.task_id)


@router.get("/status/{task_id}", response_model=TaskStatusResponse)
def get_task_status(task_id: str):
    """Poll the status of a running task."""
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
    """Download the result CSV for a completed task."""
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
    path = settings.static_dataset_path
    if not path.exists():
        raise HTTPException(status_code=404, detail="Static dataset not found. Run /update-static first.")

    return FileResponse(
        path=path,
        media_type="text/csv",
        filename="ml_dataset_static.csv",
    )
