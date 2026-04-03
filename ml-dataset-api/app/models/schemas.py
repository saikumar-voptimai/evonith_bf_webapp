"""Pydantic request/response models for the dataset and data endpoints."""

from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RmChoice(str, Enum):
    charge = "charge"
    dpr = "dpr"


class TaskStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class QueryType(str, Enum):
    ts = "ts"
    windowed_average = "windowed-average"
    average = "average"
    avg_min_max = "avg-min-max"


class ResponseFormat(str, Enum):
    json = "json"
    csv = "csv"


class OfflineReportType(str, Enum):
    hm_slag = "HM_SLAG"
    charge = "CHARGE"
    rm_composition = "RM_COMPOSITION"
    dpr = "DPR"


# ---- Online/Offline requests ----

class OnlineFetchRequest(BaseModel):
    measurements: List[str] = Field(
        ...,
        description="One or more of: process_params, temperature_profile, heatload_delta_t, cooling_water, delta_t, miscellaneous",
        min_length=1,
    )
    preset: Optional[str] = Field(
        None,
        description="Preset time window e.g. 'last 8 hours'. Takes priority over start_time/end_time.",
        examples=["last 8 hours", "last 1 day", "last 1 week"],
    )
    start_time: Optional[datetime] = Field(None, description="UTC start time (ignored if preset is set)")
    end_time: Optional[datetime] = Field(None, description="UTC end time (ignored if preset is set)")
    query_type: QueryType = Field(QueryType.windowed_average, description="Query aggregation type")
    window: Optional[str] = Field("1h", description="Aggregation window e.g. '15m', '1h' (for windowed-average)")
    format: ResponseFormat = ResponseFormat.json


class OfflineFetchRequest(BaseModel):
    report_type: OfflineReportType
    preset: Optional[str] = Field(
        None,
        description="Preset time window e.g. 'last 1 month'.",
        examples=["last 1 month", "last 3 months"],
    )
    start_time: Optional[datetime] = Field(None, description="UTC start time (ignored if preset is set)")
    end_time: Optional[datetime] = Field(None, description="UTC end time (ignored if preset is set)")
    format: ResponseFormat = ResponseFormat.json


# ---- Online/Offline responses ----

class DataMeta(BaseModel):
    measurements: Optional[List[str]] = None
    report_type: Optional[str] = None
    query_type: Optional[str] = None
    window: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    rows: int
    columns: int


class DataFetchResponse(BaseModel):
    meta: DataMeta
    columns: List[str]
    data: List[Dict[str, Any]]


# ---- Dataset (ML) requests ----

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
