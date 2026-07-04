"""Audit event API schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AuditEvent(BaseModel):
    id: str
    timestamp: str
    request_id: str | None = None
    actor_user_id: str | None = None
    actor_username: str | None = None
    event_type: str
    resource_type: str | None = None
    resource_id: str | None = None
    action: str
    result: str
    status_code: int | None = None
    error_code: str | None = None
    ip_hash: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class AuditEventList(BaseModel):
    items: list[AuditEvent]
    total: int
    limit: int
    offset: int

