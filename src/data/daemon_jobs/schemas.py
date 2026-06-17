"""Pydantic schemas used by the daemon jobs data and UI layers."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DaemonJobPayload(BaseModel):
    """Create/update payload shape for daemon job definitions."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str | None = None
    enabled: bool = False
    job_kind: str
    schedule_type: str
    cron_expression: str | None = None
    on_calendar: str | None = None
    timezone: str = "Asia/Kolkata"
    systemd_unit_name: str
    working_directory: str = "/home/pi/evonith_bf_webapp"
    python_executable: str = "/home/pi/evonith_bf_webapp/.venv/bin/python"
    module_path: str = "src.jobs.runner"
    job_args_json: str = "{}"
    env_file: str = "/home/pi/evonith_bf_webapp/.env"
    user_name: str = "pi"
    group_name: str = "pi"
    restart_policy: str = "on-failure"
    restart_sec: int = 10
    timeout_sec: int = 900
    max_runtime_sec: int = 900
    concurrency_policy: str = "forbid"
    criticality: str = "normal"
    tools_allowed_json: str = "[]"
    tools_blocked_json: str = "[]"
    memory_short_json: str = "[]"
    memory_long_json: str = "[]"
    reporting_rules_json: str = "{}"
    criticality_rules_json: str = "{}"
    persist_jobs_md_path: str = "/home/pi/evonith_bf_webapp/persist_jobs.md"
    notes: str | None = None


class ValidationResult(BaseModel):
    """Result returned by daemon job payload validation."""

    is_valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    normalized_payload: dict[str, Any] = Field(default_factory=dict)


class DaemonJobView(DaemonJobPayload):
    """Read model exposed by the daemon job service."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    created_by: str | None = None
    updated_by: str | None = None
    created_at: datetime
    updated_at: datetime
    last_previewed_at: datetime | None = None
    deleted: bool = False
    deleted_at: datetime | None = None


class DaemonJobAuditEventView(BaseModel):
    """Read model for one daemon job audit event."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    job_id: str
    event_type: str
    message: str
    actor: str | None
    created_at: datetime
    snapshot_json: str | None = None


class SystemdPreview(BaseModel):
    """Rendered systemd preview output for one daemon job."""

    job_id: str
    service_unit: str
    timer_unit: str
    install_commands: str
    uninstall_commands: str
    warnings: list[str] = Field(default_factory=list)
