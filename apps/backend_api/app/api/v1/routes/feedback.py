"""API v1 feedback routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Header, Query, Request, Response, status
from fastapi.responses import FileResponse

from apps.backend_api.app.api.v1.schemas.common import ApiMeta, ApiResponse
from apps.backend_api.app.api.v1.schemas.feedback import (
    FeedbackAttachmentApiResponse,
    FeedbackAttachmentListApiResponse,
    FeedbackCommentApiResponse,
    FeedbackCommentCreateRequest,
    FeedbackCommentListApiResponse,
    FeedbackConfigApiResponse,
    FeedbackDeleteApiResponse,
    FeedbackEventListApiResponse,
    FeedbackSummaryApiResponse,
    FeedbackTicketApiResponse,
    FeedbackTicketCreateRequest,
    FeedbackTicketListApiResponse,
    FeedbackTicketUpdateRequest,
    FeedbackTransitionRequest,
)
from apps.backend_api.app.core.auth_dependencies import get_optional_current_user
from apps.backend_api.app.core.responses import get_request_id
from apps.backend_api.app.services.feedback_service import FeedbackService

router = APIRouter(prefix="/feedback", tags=["feedback"])


def get_feedback_service(request: Request) -> FeedbackService:
    """Return app feedback service or a lazy default."""
    service = getattr(request.app.state, "feedback_service", None)
    if service is not None:
        return service
    settings = getattr(request.app.state, "backend_settings", None)
    service = FeedbackService(
        settings=settings,
        audit_service=getattr(request.app.state, "audit_service", None),
    )
    request.app.state.feedback_service = service
    return service


def _wrap(request: Request, data: Any, warnings: list[str] | None = None) -> ApiResponse:
    return ApiResponse(request_id=get_request_id(request), data=data, meta=ApiMeta(warnings=warnings or []))


@router.get("/config", response_model=FeedbackConfigApiResponse, operation_id="get_feedback_config")
def feedback_config(
    request: Request,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    feedback_service: FeedbackService = Depends(get_feedback_service),
):
    return _wrap(request, feedback_service.config(current_user=current_user))


@router.get("/summary", response_model=FeedbackSummaryApiResponse, operation_id="get_feedback_summary")
def feedback_summary(
    request: Request,
    created_from: str | None = None,
    created_to: str | None = None,
    page_id: str | None = None,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    feedback_service: FeedbackService = Depends(get_feedback_service),
):
    return _wrap(
        request,
        feedback_service.summary(
            filters={"created_from": created_from, "created_to": created_to, "page_id": page_id},
            current_user=current_user,
        ),
    )


@router.get("/tickets", response_model=FeedbackTicketListApiResponse, operation_id="list_feedback_tickets")
def list_tickets(
    request: Request,
    status: list[str] | None = Query(default=None),
    priority: list[str] | None = Query(default=None),
    page_id: list[str] | None = Query(default=None),
    reporter_user_id: str | None = None,
    category: str | None = None,
    created_by: str | None = None,
    assigned_to: str | None = None,
    search: str | None = None,
    created_from: str | None = None,
    created_to: str | None = None,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    feedback_service: FeedbackService = Depends(get_feedback_service),
):
    data = feedback_service.list_tickets(
        filters={
            "status": status,
            "priority": priority,
            "page_id": page_id,
            "reporter_user_id": reporter_user_id,
            "category": category,
            "created_by": created_by,
            "assigned_to": assigned_to,
            "search": search,
            "created_from": created_from,
            "created_to": created_to,
            "limit": limit,
            "offset": offset,
        },
        current_user=current_user,
    )
    return _wrap(request, data)


@router.post("/tickets", response_model=FeedbackTicketApiResponse, status_code=status.HTTP_201_CREATED, operation_id="create_feedback_ticket")
def create_ticket(
    request: Request,
    payload: FeedbackTicketCreateRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    feedback_service: FeedbackService = Depends(get_feedback_service),
):
    data = feedback_service.create_ticket(
        payload=payload.model_dump(),
        current_user=current_user,
        request_id=get_request_id(request),
        idempotency_key=idempotency_key,
    )
    return _wrap(request, data)


@router.get("/tickets/{ticket_id}", response_model=FeedbackTicketApiResponse, operation_id="get_feedback_ticket")
def get_ticket(
    request: Request,
    ticket_id: str,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    feedback_service: FeedbackService = Depends(get_feedback_service),
):
    return _wrap(request, feedback_service.get_ticket(ticket_id, current_user))


@router.patch("/tickets/{ticket_id}", response_model=FeedbackTicketApiResponse, operation_id="update_feedback_ticket")
def update_ticket(
    request: Request,
    ticket_id: str,
    payload: FeedbackTicketUpdateRequest,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    feedback_service: FeedbackService = Depends(get_feedback_service),
):
    data = feedback_service.update_ticket(
        ticket_id=ticket_id,
        payload=payload.model_dump(exclude_unset=True),
        current_user=current_user,
        request_id=get_request_id(request),
    )
    return _wrap(request, data)


@router.delete("/tickets/{ticket_id}", response_model=FeedbackDeleteApiResponse, operation_id="delete_feedback_ticket")
def delete_ticket(
    request: Request,
    ticket_id: str,
    expected_version: int = Query(..., ge=1),
    reason: str | None = None,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    feedback_service: FeedbackService = Depends(get_feedback_service),
):
    return _wrap(
        request,
        feedback_service.delete_ticket(
            ticket_id=ticket_id,
            expected_version=expected_version,
            current_user=current_user,
            request_id=get_request_id(request),
            idempotency_key=idempotency_key,
            reason=reason,
        ),
    )


@router.post("/tickets/{ticket_id}/transitions", response_model=FeedbackTicketApiResponse, operation_id="transition_feedback_ticket")
def transition_ticket(
    request: Request,
    ticket_id: str,
    payload: FeedbackTransitionRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    feedback_service: FeedbackService = Depends(get_feedback_service),
):
    return _wrap(
        request,
        feedback_service.transition_ticket(
            ticket_id=ticket_id,
            payload=payload.model_dump(),
            current_user=current_user,
            request_id=get_request_id(request),
            idempotency_key=idempotency_key,
        ),
    )


@router.post("/tickets/{ticket_id}/close", response_model=FeedbackTicketApiResponse, operation_id="close_feedback_ticket")
def close_ticket(
    request: Request,
    ticket_id: str,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    feedback_service: FeedbackService = Depends(get_feedback_service),
):
    return _wrap(request, feedback_service.close_ticket(ticket_id=ticket_id, current_user=current_user, request_id=get_request_id(request), idempotency_key=idempotency_key))


@router.post("/tickets/{ticket_id}/reopen", response_model=FeedbackTicketApiResponse, operation_id="reopen_feedback_ticket")
def reopen_ticket(
    request: Request,
    ticket_id: str,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    feedback_service: FeedbackService = Depends(get_feedback_service),
):
    return _wrap(request, feedback_service.reopen_ticket(ticket_id=ticket_id, current_user=current_user, request_id=get_request_id(request), idempotency_key=idempotency_key))


@router.get("/tickets/{ticket_id}/events", response_model=FeedbackEventListApiResponse, operation_id="list_feedback_ticket_events")
def list_events(
    request: Request,
    ticket_id: str,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    feedback_service: FeedbackService = Depends(get_feedback_service),
):
    return _wrap(request, feedback_service.list_events(ticket_id=ticket_id, current_user=current_user, limit=limit, offset=offset))


@router.get("/tickets/{ticket_id}/comments", response_model=FeedbackCommentListApiResponse, operation_id="list_feedback_ticket_comments")
def list_comments(
    request: Request,
    ticket_id: str,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    feedback_service: FeedbackService = Depends(get_feedback_service),
):
    return _wrap(request, feedback_service.list_comments(ticket_id=ticket_id, current_user=current_user, limit=limit, offset=offset))


@router.post("/tickets/{ticket_id}/comments", response_model=FeedbackCommentApiResponse, status_code=status.HTTP_201_CREATED, operation_id="create_feedback_ticket_comment")
def add_comment(
    request: Request,
    ticket_id: str,
    payload: FeedbackCommentCreateRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    feedback_service: FeedbackService = Depends(get_feedback_service),
):
    return _wrap(request, feedback_service.add_comment(ticket_id=ticket_id, body=payload.body, current_user=current_user, request_id=get_request_id(request), idempotency_key=idempotency_key))


@router.get("/tickets/{ticket_id}/attachments", response_model=FeedbackAttachmentListApiResponse, operation_id="list_feedback_ticket_attachments")
def list_attachments(
    request: Request,
    ticket_id: str,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    feedback_service: FeedbackService = Depends(get_feedback_service),
):
    return _wrap(request, feedback_service.list_attachments(ticket_id=ticket_id, current_user=current_user))


@router.post("/tickets/{ticket_id}/attachments", response_model=FeedbackAttachmentApiResponse, status_code=status.HTTP_201_CREATED, operation_id="upload_feedback_ticket_attachment")
async def upload_attachment(
    request: Request,
    ticket_id: str,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    feedback_service: FeedbackService = Depends(get_feedback_service),
):
    upload = await feedback_service.attachment_service.parse_upload_request(request)
    data = feedback_service.add_attachment(ticket_id=ticket_id, upload=upload, current_user=current_user, request_id=get_request_id(request), idempotency_key=idempotency_key)
    return _wrap(request, data)


@router.get("/attachments/{attachment_id}/download", operation_id="download_feedback_attachment")
def download_attachment(
    request: Request,
    attachment_id: str,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    feedback_service: FeedbackService = Depends(get_feedback_service),
):
    attachment, path = feedback_service.get_attachment_for_download(attachment_id, current_user)
    return FileResponse(
        path=path,
        filename=attachment.original_filename,
        media_type=attachment.content_type,
        headers={"X-Request-ID": get_request_id(request), "X-Content-Type-Options": "nosniff"},
    )


@router.get("/attachments/{attachment_id}/preview", operation_id="preview_feedback_attachment")
def preview_attachment(
    request: Request,
    attachment_id: str,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    feedback_service: FeedbackService = Depends(get_feedback_service),
):
    data, content_type, attachment = feedback_service.get_attachment_preview(attachment_id, current_user)
    return Response(
        content=data,
        media_type=content_type,
        headers={
            "X-Request-ID": get_request_id(request),
            "X-Content-Type-Options": "nosniff",
            "ETag": attachment.checksum_sha256 or attachment.id,
        },
    )


@router.delete("/attachments/{attachment_id}", response_model=FeedbackDeleteApiResponse, operation_id="delete_feedback_attachment")
def delete_attachment(
    request: Request,
    attachment_id: str,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    feedback_service: FeedbackService = Depends(get_feedback_service),
):
    return _wrap(request, feedback_service.delete_attachment(attachment_id=attachment_id, current_user=current_user, request_id=get_request_id(request), idempotency_key=idempotency_key))
