"""FurnaceMind API schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class FurnaceMindConfigResponse(BaseModel):
    llm_enabled: bool
    provider_configured: bool
    provider_name: str | None = None
    model_name: str | None = None
    memory_enabled: bool
    vector_backend: str | None = None
    qdrant_configured: bool
    embeddings_enabled: bool
    documents_enabled: bool
    tools_enabled: bool
    streaming_enabled: bool
    polling_fallback_enabled: bool
    code_execution_enabled: bool
    shell_execution_enabled: bool
    max_message_chars: int
    max_prompt_chars: int
    max_history_messages: int
    max_context_docs: int
    allowed_document_types: list[str]
    allowed_document_extensions: list[str]
    allowed_tools: list[str]
    warnings: list[dict[str, Any]] = Field(default_factory=list)


class ConversationCreateRequest(BaseModel):
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationUpdateRequest(BaseModel):
    title: str | None = None
    archived: bool | None = None
    metadata: dict[str, Any] | None = None


class ConversationResponse(BaseModel):
    id: str
    title: str | None = None
    owner_id: str | int | None = None
    created_at: datetime
    updated_at: datetime
    archived: bool
    message_count: int
    last_message_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationListResponse(BaseModel):
    items: list[ConversationResponse]
    total: int
    limit: int
    offset: int


MessageRole = Literal["user", "assistant", "system", "tool"]


class MessageResponse(BaseModel):
    id: str
    conversation_id: str
    role: MessageRole
    content: str
    status: str
    created_at: datetime
    updated_at: datetime | None = None
    run_id: str | None = None
    parent_message_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[dict[str, Any]] = Field(default_factory=list)


class MessageCreateRequest(BaseModel):
    content: str
    parent_message_id: str | None = None
    document_ids: list[str] = Field(default_factory=list)
    tool_mode: str = "auto"
    allow_llm: bool | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class RunCreateRequest(BaseModel):
    message: str
    document_ids: list[str] = Field(default_factory=list)
    tool_mode: str = "auto"
    allow_llm: bool | None = None
    stream: bool = False
    options: dict[str, Any] = Field(default_factory=dict)


class RunResponse(BaseModel):
    id: str
    conversation_id: str
    status: str
    user_message_id: str | None = None
    assistant_message_id: str | None = None
    created_at: datetime
    updated_at: datetime | None = None
    completed_at: datetime | None = None
    error_code: str | None = None
    error_message: str | None = None
    warnings: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)


class RunEventResponse(BaseModel):
    id: str
    run_id: str
    conversation_id: str
    event_type: str
    sequence: int
    payload: dict[str, Any]
    created_at: datetime


class RunStatusResponse(BaseModel):
    id: str
    conversation_id: str
    status: str
    progress: float | None = None
    current_step: str | None = None
    events: list[RunEventResponse] = Field(default_factory=list)
    result_message: MessageResponse | None = None
    error_code: str | None = None
    error_message: str | None = None
    artifacts: list[dict[str, Any]] = Field(default_factory=list)


class DocumentUploadResponse(BaseModel):
    id: str
    filename: str
    original_filename: str
    content_type: str
    size_bytes: int
    status: str
    indexed: bool
    chunk_count: int | None = None
    created_at: datetime
    warnings: list[dict[str, Any]] = Field(default_factory=list)


class DocumentResponse(BaseModel):
    id: str
    filename: str
    original_filename: str
    content_type: str
    size_bytes: int
    status: str
    indexed: bool
    chunk_count: int | None = None
    owner_id: str | int | None = None
    created_at: datetime
    updated_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentListResponse(BaseModel):
    items: list[DocumentResponse]
    total: int
    limit: int
    offset: int


class ToolInfo(BaseModel):
    name: str
    description: str
    enabled: bool
    input_schema: dict[str, Any]
    safe_for_user: bool
    requires_auth: bool
    timeout_seconds: int


class ToolCallResult(BaseModel):
    tool_name: str
    status: str
    output: dict[str, Any] | None = None
    error_code: str | None = None
    error_message: str | None = None
    duration_ms: int | None = None
    warnings: list[dict[str, Any]] = Field(default_factory=list)


class MessageFeedbackRequest(BaseModel):
    rating: int | None = None
    helpful: bool | None = None
    comment: str | None = None
    tags: list[str] = Field(default_factory=list)


class MessageFeedbackResponse(BaseModel):
    id: str
    message_id: str
    conversation_id: str
    created_at: datetime
