"""Shared Pydantic request/response models for the furnace data platform.

Used by both the canonical backend API routes and any caller that wants
typed request/response objects (e.g., typed HTTP clients in the webapp).
"""

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
    raw_material_strength = "RAW_MATERIAL_STRENGTH"
    dpr = "DPR"


# ---- Online/Offline fetch requests ----

class OnlineFetchRequest(BaseModel):
    measurements: List[str] = Field(
        ...,
        description=(
            "One or more of: process_params, temperature_profile, "
            "heatload_delta_t, cooling_water, delta_t, miscellaneous"
        ),
        min_length=1,
    )
    preset: Optional[str] = Field(
        None,
        description="Preset time window e.g. 'last 8 hours'. Takes priority over start/end.",
        examples=["last 8 hours", "last 1 day", "last 1 week"],
    )
    start_time: Optional[datetime] = Field(None, description="UTC start time (ignored if preset is set)")
    end_time: Optional[datetime] = Field(None, description="UTC end time (ignored if preset is set)")
    query_type: QueryType = Field(QueryType.windowed_average, description="Query aggregation type")
    window: Optional[str] = Field("1h", description="Aggregation window e.g. '15m', '1h'")
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


class RmLiveFetchRequest(BaseModel):
    """Request body for POST /data/rm/live."""
    lookback_days: int = Field(10, description="Fetch last N days of RM composition data.", ge=1, le=365)
    cadence: Optional[str] = Field(None, description="Resampling cadence: '1h', '8h', or '1d'.")
    format: ResponseFormat = ResponseFormat.json


# ---- Response models ----

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


# ---- Dataset (ML pipeline) requests ----

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


# ---- Task management ----

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
    version: str = "0.1.1"
