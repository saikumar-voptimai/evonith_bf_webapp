"""Pydantic request/response models for the dataset and data endpoints."""

from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RmChoice(str, Enum):
    """Raw material source selection for ML dataset construction."""

    charge = "charge"
    dpr = "dpr"


class TaskStatus(str, Enum):
    """Lifecycle states for background dataset fetch tasks."""

    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class QueryType(str, Enum):
    """Aggregation mode applied to InfluxDB or Neon query results."""

    ts = "ts"
    windowed_average = "windowed-average"
    average = "average"
    avg_min_max = "avg-min-max"


class OfflineDataSource(str, Enum):
    """Backend store used to serve offline furnace data."""

    influx = "influx"
    neon_db = "neon_db"


class ResponseFormat(str, Enum):
    """HTTP response encoding for data fetch endpoints."""

    json = "json"
    csv = "csv"


class OfflineReportType(str, Enum):
    """Logical category identifiers for offline report data."""

    hm_slag = "HM_SLAG"
    charge = "CHARGE"
    rm_composition = "RM_COMPOSITION"
    dpr = "DPR"


class NeonOfflineTable(str, Enum):
    """Whitelisted Neon/PostgreSQL table names for offline data access."""

    charge_data = "charge_data"
    dpr_data = "dpr_data"
    flux_chemistry = "flux_chemistry"
    fuel_chemistry = "fuel_chemistry"
    hot_metal_chemistry = "hot_metal_chemistry"
    ore_chemistry = "ore_chemistry"
    raw_materials = "raw_materials"
    rm_hm = "rm_hm"
    sinter_chemistry = "sinter_chemistry"


# ---- Online/Offline requests ----

class OnlineFetchRequest(BaseModel):
    """Request body for POST /data/online/fetch."""

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
    """Request body for POST /data/offline/fetch."""

    report_type: OfflineReportType
    source: OfflineDataSource = Field(
        OfflineDataSource.influx,
        description="Offline backend. Defaults to existing Influx behavior; use 'neon_db' for Neon/PostgreSQL.",
    )
    table_name: Optional[NeonOfflineTable] = Field(
        None,
        description="Optional Neon table override. Used only when source='neon_db'.",
    )
    preset: Optional[str] = Field(
        None,
        description="Preset time window e.g. 'last 1 month'.",
        examples=["last 1 month", "last 3 months"],
    )
    start_time: Optional[datetime] = Field(None, description="UTC start time (ignored if preset is set)")
    end_time: Optional[datetime] = Field(None, description="UTC end time (ignored if preset is set)")
    query_type: QueryType = Field(
        QueryType.ts,
        description="For Neon: 'ts' raw rows, 'windowed-average' for hourly/custom averages, or 'average'. Ignored for Influx offline.",
    )
    window: Optional[str] = Field(
        "1 hour",
        description="For Neon windowed-average. PostgreSQL interval e.g. '1 hour', '15 minutes', '8 hours'.",
    )
    format: ResponseFormat = ResponseFormat.json


# ---- Online/Offline responses ----

class DataMeta(BaseModel):
    """Metadata block included in every DataFetchResponse."""

    measurements: Optional[List[str]] = None
    report_type: Optional[str] = None
    source: Optional[str] = None
    table_name: Optional[str] = None
    query_type: Optional[str] = None
    window: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    rows: int
    columns: int


class DataFetchResponse(BaseModel):
    """Standard JSON envelope returned by online and offline data fetch endpoints."""

    meta: DataMeta
    columns: List[str]
    data: List[Dict[str, Any]]


# ---- Dataset (ML) requests ----

class FetchDatasetRequest(BaseModel):
    """Request body for triggering an ML dataset fetch over a date range."""

    start_date: date
    end_date: date
    rm_choice: RmChoice = RmChoice.charge
    use_cache: bool = True
    apply_cleaning: bool = False
    callback_url: Optional[str] = Field(None, description="Webhook URL for completion notification")


class UpdateStaticRequest(BaseModel):
    """Request body for rebuilding the static filtered ML dataset file."""

    rm_choice: RmChoice = RmChoice.charge
    reprocess_from: Optional[date] = Field(None, description="Recompute from this date onwards")
    apply_cleaning: bool = True
    callback_url: Optional[str] = Field(None, description="Webhook URL for completion notification")


# ---- Responses ----

class TaskCreatedResponse(BaseModel):
    """Response returned when a background dataset task has been accepted."""

    task_id: str
    status: TaskStatus = TaskStatus.pending
    message: str = "Task created"


class TaskStatusResponse(BaseModel):
    """Status snapshot for a running or completed background task."""

    task_id: str
    status: TaskStatus
    progress: Optional[str] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    rows: Optional[int] = None
    columns: Optional[int] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Response body for the service healthcheck endpoint."""

    status: str = "ok"
    version: str = "0.1.0"


# ---- Live RM data ----

class RmLiveFetchRequest(BaseModel):
    """Request body for POST /data/rm/live."""

    lookback_days: int = Field(10, ge=1, le=365, description="Days of history to fetch")
    cadence: str = Field("8h", description="Aggregation cadence: '8h', '1h', or '1d'")
    format: ResponseFormat = ResponseFormat.json
