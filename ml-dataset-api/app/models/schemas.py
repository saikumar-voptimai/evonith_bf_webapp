"""Pydantic request/response models for the dataset API."""

from datetime import date, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, HttpUrl


class RmChoice(str, Enum):
    charge = "charge"
    dpr = "dpr"


class TaskStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


# ---- Requests ----

class FetchDatasetRequest(BaseModel):
    start_date: date
    end_date: date
    rm_choice: RmChoice = RmChoice.charge
    use_cache: bool = True
    apply_cleaning: bool = False
    callback_url: Optional[str] = Field(None, description="Webhook URL for completion notification")


class UpdateStaticRequest(BaseModel):
    rm_choice: RmChoice = RmChoice.charge
    reprocess_from: Optional[date] = Field(None, description="Recompute from this date onwards")
    apply_cleaning: bool = True
    callback_url: Optional[str] = Field(None, description="Webhook URL for completion notification")


# ---- Responses ----

class TaskCreatedResponse(BaseModel):
    task_id: str
    status: TaskStatus = TaskStatus.pending
    message: str = "Task created"


class TaskStatusResponse(BaseModel):
    task_id: str
    status: TaskStatus
    progress: Optional[str] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    rows: Optional[int] = None
    columns: Optional[int] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
