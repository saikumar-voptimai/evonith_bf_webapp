"""Feedback API schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class FeedbackTicketCreateRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=240)
    description: str = Field(..., min_length=1, max_length=5000)
    category: str | None = Field(default=None, max_length=128)
    priority: str = Field(default="medium", max_length=32)
    page: str | None = Field(default=None, max_length=128)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("title", "description", "category", "priority", "page")
    @classmethod
    def strip_strings(cls, value: str | None) -> str | None:
        return value.strip() if isinstance(value, str) else value


class FeedbackTicketUpdateRequest(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=240)
    description: str | None = Field(default=None, min_length=1, max_length=5000)
    category: str | None = Field(default=None, max_length=128)
    priority: str | None = Field(default=None, max_length=32)
    status: str | None = Field(default=None, max_length=32)
    assigned_to: str | int | None = None
    resolution_notes: str | None = Field(default=None, max_length=2000)
    tags: list[str] | None = None
    metadata: dict[str, Any] | None = None

    @field_validator("title", "description", "category", "priority", "status")
    @classmethod
    def strip_optional_strings(cls, value: str | None) -> str | None:
        return value.strip() if isinstance(value, str) else value


class FeedbackTicketResponse(BaseModel):
    id: str
    ticket_number: str
    title: str
    description: str
    category: str | None = None
    priority: str
    status: str
    page: str | None = None
    tags: list[str] = Field(default_factory=list)
    created_by: str | int | None = None
    created_by_username: str | None = None
    assigned_to: str | int | None = None
    created_at: datetime
    updated_at: datetime
    closed_at: datetime | None = None
    attachment_count: int = 0
    comment_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class FeedbackTicketListResponse(BaseModel):
    items: list[FeedbackTicketResponse]
    total: int
    limit: int
    offset: int


class FeedbackCommentCreateRequest(BaseModel):
    body: str = Field(..., min_length=1, max_length=4000)

    @field_validator("body")
    @classmethod
    def strip_body(cls, value: str) -> str:
        return value.strip()


class FeedbackCommentResponse(BaseModel):
    id: str
    ticket_id: str
    body: str
    created_by: str | int | None = None
    created_by_username: str | None = None
    created_at: datetime


class FeedbackAttachmentResponse(BaseModel):
    id: str
    ticket_id: str
    filename: str
    original_filename: str
    content_type: str
    size_bytes: int
    created_by: str | int | None = None
    created_at: datetime
    download_url: str


class FeedbackConfigResponse(BaseModel):
    statuses: list[str]
    priorities: list[str]
    categories: list[str]
    max_attachment_mb: int
    allowed_attachment_types: list[str]
    allowed_attachment_extensions: list[str]
    max_attachments_per_ticket: int
