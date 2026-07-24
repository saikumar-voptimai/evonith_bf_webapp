"""Feedback API schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from apps.backend_api.app.api.v1.schemas.common import ApiResponse


class FeedbackLabel(BaseModel):
    id: str
    label: str


class FeedbackStatusLabel(FeedbackLabel):
    terminal: bool = False
    allowed_next_status_ids: list[str] = Field(default_factory=list)


class FeedbackPriorityLabel(FeedbackLabel):
    rank: int


class FeedbackIdentity(BaseModel):
    user_id: str | int | None = None
    username: str | None = None


class FeedbackLimits(BaseModel):
    title_max_chars: int = 240
    description_max_chars: int = 5000
    ideal_closure_max_chars: int = 1000
    comment_max_chars: int = 4000
    max_attachment_mb: int
    max_attachments_per_ticket: int
    max_list_page_size: int = 100


class FeedbackAttachmentPolicy(BaseModel):
    allowed_content_types: list[str]
    allowed_extensions: list[str]
    image_preview_available: bool = True


class FeedbackCapabilities(BaseModel):
    can_create: bool
    can_view_all: bool
    can_moderate: bool
    can_delete_tickets: bool
    can_delete_attachments: bool


class FeedbackConfigResponse(BaseModel):
    catalog_version: str
    workflow_version: str
    display_timezone: str
    pages: list[FeedbackLabel]
    statuses: list[FeedbackStatusLabel]
    priorities: list[FeedbackPriorityLabel]
    limits: FeedbackLimits
    attachments: FeedbackAttachmentPolicy
    capabilities: FeedbackCapabilities
    etag: str


class FeedbackStatusSummaryCount(BaseModel):
    status_id: str
    count: int


class FeedbackPrioritySummaryCount(BaseModel):
    priority_id: str
    count: int


class FeedbackFacetPage(BaseModel):
    page_id: str | None = None
    label: str | None = None
    count: int


class FeedbackFacetReporter(BaseModel):
    user_id: str | int | None = None
    username: str | None = None
    count: int


class FeedbackSummaryFacets(BaseModel):
    pages: list[FeedbackFacetPage] = Field(default_factory=list)
    reporters: list[FeedbackFacetReporter] = Field(default_factory=list)


class FeedbackSummaryResponse(BaseModel):
    scope: str
    total: int
    counts_by_status: list[FeedbackStatusSummaryCount]
    counts_by_priority: list[FeedbackPrioritySummaryCount]
    resolved_or_closed_count: int
    dependency_conflict_count: int
    rejected_count: int
    high_or_critical_count: int
    facets: FeedbackSummaryFacets
    as_of: datetime


class FeedbackTicketCreateRequest(BaseModel):
    page_id: str | None = Field(default=None, max_length=128)
    title: str | None = Field(default=None, max_length=240)
    description: str = Field(..., min_length=1, max_length=5000)
    ideal_closure: str | None = Field(default=None, max_length=1000)
    priority: str = Field(default="medium", max_length=32)
    tags: list[str] = Field(default_factory=list, max_length=20)
    client_context: dict[str, Any] = Field(default_factory=dict)
    # Deprecated compatibility fields accepted but not trusted for identity.
    category: str | None = Field(default=None, max_length=128)
    page: str | None = Field(default=None, max_length=128)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("page_id", "title", "description", "ideal_closure", "priority", "category", "page")
    @classmethod
    def strip_strings(cls, value: str | None) -> str | None:
        return value.strip() if isinstance(value, str) else value


class FeedbackTicketUpdateRequest(BaseModel):
    expected_version: int | None = Field(default=None, ge=1)
    title: str | None = Field(default=None, min_length=1, max_length=240)
    description: str | None = Field(default=None, min_length=1, max_length=5000)
    ideal_closure: str | None = Field(default=None, max_length=1000)
    page_id: str | None = Field(default=None, max_length=128)
    category: str | None = Field(default=None, max_length=128)
    priority: str | None = Field(default=None, max_length=32)
    status: str | None = Field(default=None, max_length=32)
    assigned_to: str | int | None = None
    resolution_notes: str | None = Field(default=None, max_length=2000)
    tags: list[str] | None = None
    metadata: dict[str, Any] | None = None

    @field_validator("title", "description", "ideal_closure", "page_id", "category", "priority", "status")
    @classmethod
    def strip_optional_strings(cls, value: str | None) -> str | None:
        return value.strip() if isinstance(value, str) else value


class FeedbackTransitionRequest(BaseModel):
    target_status_id: str = Field(..., max_length=32)
    expected_version: int = Field(..., ge=1)
    note: str | None = Field(default=None, max_length=2000)
    resolution_notes: str | None = Field(default=None, max_length=2000)

    @field_validator("target_status_id", "note", "resolution_notes")
    @classmethod
    def strip_transition_strings(cls, value: str | None) -> str | None:
        return value.strip() if isinstance(value, str) else value


class FeedbackTicketResponse(BaseModel):
    id: str
    ticket_number: str
    version: int
    title: str
    description: str
    ideal_closure: str | None = None
    page: FeedbackLabel | None = None
    priority: FeedbackPriorityLabel
    status: FeedbackStatusLabel
    reported_by: FeedbackIdentity
    updated_by: FeedbackIdentity | None = None
    assigned_to: str | int | None = None
    resolution_notes: str | None = None
    created_at: datetime
    updated_at: datetime
    last_activity_at: datetime
    resolved_at: datetime | None = None
    closed_at: datetime | None = None
    deleted_at: datetime | None = None
    attachment_count: int = 0
    comment_count: int = 0
    event_count: int = 0
    tags: list[str] = Field(default_factory=list)
    allowed_actions: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Compatibility IDs for existing Streamlit code during gateway cutover.
    page_id: str | None = None
    priority_id: str
    status_id: str
    category: str | None = None
    created_by: str | int | None = None
    created_by_username: str | None = None


class FeedbackTicketListResponse(BaseModel):
    items: list[FeedbackTicketResponse]
    total: int
    limit: int
    offset: int
    next_offset: int | None = None


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


class FeedbackCommentListResponse(BaseModel):
    items: list[FeedbackCommentResponse]
    total: int
    limit: int
    offset: int
    next_offset: int | None = None


class FeedbackAttachmentResponse(BaseModel):
    id: str
    ticket_id: str
    filename: str
    original_filename: str
    content_type: str
    size_bytes: int
    checksum_sha256: str | None = None
    storage_status: str = "stored"
    created_by: str | int | None = None
    created_at: datetime


class FeedbackAttachmentListResponse(BaseModel):
    items: list[FeedbackAttachmentResponse]
    total: int


class FeedbackEventResponse(BaseModel):
    id: str
    ticket_id: str
    event_type: str
    sequence: int
    actor: FeedbackIdentity | None = None
    old_status_id: str | None = None
    new_status_id: str | None = None
    note: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class FeedbackEventListResponse(BaseModel):
    items: list[FeedbackEventResponse]
    total: int
    limit: int
    offset: int
    next_offset: int | None = None


class FeedbackDeleteResponse(BaseModel):
    deleted: bool
    ticket_id: str | None = None
    attachment_id: str | None = None
    version: int | None = None


class FeedbackConfigApiResponse(ApiResponse[FeedbackConfigResponse]):
    pass


class FeedbackSummaryApiResponse(ApiResponse[FeedbackSummaryResponse]):
    pass


class FeedbackTicketApiResponse(ApiResponse[FeedbackTicketResponse]):
    pass


class FeedbackTicketListApiResponse(ApiResponse[FeedbackTicketListResponse]):
    pass


class FeedbackEventListApiResponse(ApiResponse[FeedbackEventListResponse]):
    pass


class FeedbackCommentApiResponse(ApiResponse[FeedbackCommentResponse]):
    pass


class FeedbackCommentListApiResponse(ApiResponse[FeedbackCommentListResponse]):
    pass


class FeedbackAttachmentApiResponse(ApiResponse[FeedbackAttachmentResponse]):
    pass


class FeedbackAttachmentListApiResponse(ApiResponse[FeedbackAttachmentListResponse]):
    pass


class FeedbackDeleteApiResponse(ApiResponse[FeedbackDeleteResponse]):
    pass

