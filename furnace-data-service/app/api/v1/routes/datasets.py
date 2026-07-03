"""API v1 dataset routes that delegate to legacy dataset handlers."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter

from app.models.schemas import FetchDatasetRequest, TaskCreatedResponse, TaskStatusResponse, UpdateStaticRequest
from app.routes import dataset as legacy_dataset

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("/fetch", response_model=TaskCreatedResponse)
def fetch_dataset(req: FetchDatasetRequest):
    return legacy_dataset.fetch_dataset(req)


@router.post("/update-static", response_model=TaskCreatedResponse)
def update_static(req: UpdateStaticRequest):
    return legacy_dataset.update_static(req)


@router.get("/status/{task_id}", response_model=TaskStatusResponse)
def get_task_status(task_id: str):
    return legacy_dataset.get_task_status(task_id)


@router.get("/download/{task_id}")
def download_result(task_id: str):
    return legacy_dataset.download_result(task_id)


@router.get("/static")
def download_static():
    return legacy_dataset.download_static()


@router.get("/cache-info")
def cache_info() -> Dict[str, Any]:
    return legacy_dataset.cache_info()
