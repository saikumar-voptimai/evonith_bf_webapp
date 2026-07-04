"""Copilot API schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class CopilotConfigResponse(BaseModel):
    llm_enabled: bool
    provider_configured: bool
    provider_name: str | None = None
    model_name: str | None = None
    data_redaction_enabled: bool
    raw_data_to_llm_allowed: bool
    max_context_rows: int
    max_prompt_chars: int
    max_output_chars: int
    supported_analysis_modes: list[str]
    warnings: list[dict[str, Any]] = Field(default_factory=list)


class CopilotDataWindowRequest(BaseModel):
    source: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    columns: list[str] | None = None
    filters: dict[str, Any] = Field(default_factory=dict)
    limit: int | None = 500
    timezone: str = "Asia/Kolkata"


class CopilotRecentDataResponse(BaseModel):
    columns: list[dict[str, Any]]
    rows: list[dict[str, Any]]
    row_count: int | None
    returned_rows: int
    truncated: bool
    summary: dict[str, Any]
    warnings: list[dict[str, Any]] = Field(default_factory=list)


class CopilotAnomalyRequest(BaseModel):
    source: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    input_data: dict[str, Any] | list[dict[str, Any]] | None = None
    columns: list[str] | None = None
    options: dict[str, Any] = Field(default_factory=dict)
    timezone: str = "Asia/Kolkata"


class AnomalySignal(BaseModel):
    name: str
    score: float | None = None
    severity: str | None = None
    value: Any | None = None
    baseline: Any | None = None
    unit: str | None = None
    explanation: str | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)


class CopilotAnomalyResponse(BaseModel):
    overall_score: float | None = None
    severity: str | None = None
    signals: list[AnomalySignal]
    summary: dict[str, Any]
    tables: dict[str, Any] = Field(default_factory=dict)
    charts: dict[str, Any] = Field(default_factory=dict)
    warnings: list[dict[str, Any]] = Field(default_factory=list)
    computed_at: datetime


class CopilotAnalyzeRequest(BaseModel):
    question: str
    source: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    input_data: dict[str, Any] | list[dict[str, Any]] | None = None
    analysis_mode: str = "summary"
    include_anomaly: bool = True
    include_recent_data: bool = True
    include_recommendations: bool = False
    allow_llm: bool | None = None
    options: dict[str, Any] = Field(default_factory=dict)
    timezone: str = "Asia/Kolkata"
    async_job: bool = False


class CopilotEvidence(BaseModel):
    id: str
    title: str
    type: str
    summary: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class CopilotAnalyzeResponse(BaseModel):
    answer: str
    analysis_mode: str
    llm_used: bool
    provider_name: str | None = None
    model_name: str | None = None
    confidence: float | None = None
    anomaly: CopilotAnomalyResponse | None = None
    evidence: list[CopilotEvidence] = Field(default_factory=list)
    tables: dict[str, Any] = Field(default_factory=dict)
    charts: dict[str, Any] = Field(default_factory=dict)
    warnings: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    computed_at: datetime


class CopilotJobResponse(BaseModel):
    job_id: str
    status: str
    message: str | None = None
    created_at: datetime
    updated_at: datetime | None = None
    artifact_id: str | None = None
    download_url: str | None = None


class CopilotJobStatus(BaseModel):
    job_id: str
    status: str
    progress: float | None = None
    message: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    result: dict[str, Any] | None = None
    artifact_id: str | None = None
    download_url: str | None = None
    created_at: datetime
    updated_at: datetime | None = None
    completed_at: datetime | None = None
