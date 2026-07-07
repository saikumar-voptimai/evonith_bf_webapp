"""API v1 feedback routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import FileResponse

from apps.backend_api.app.api.v1.schemas.common import ApiMeta, ApiResponse
from apps.backend_api.app.api.v1.schemas.feedback import (
    FeedbackCommentCreateRequest,
    FeedbackTicketCreateRequest,
    FeedbackTicketUpdateRequest,
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
    service = FeedbackService(settings=settings)
    request.app.state.feedback_service = service
    return service


def _wrap(request: Request, data: Any, warnings: list[str] | None = None) -> ApiResponse:
    return ApiResponse(
        request_id=get_request_id(request),
        data=data,
        meta=ApiMeta(warnings=warnings or []),
    )


@router.get("/config", response_model=ApiResponse)
def feedback_config(
    request: Request,
    feedback_service: FeedbackService = Depends(get_feedback_service),
):
    return _wrap(request, feedback_service.config())


@router.get("/tickets", response_model=ApiResponse)
def list_tickets(
    request: Request,
    status: str | None = None,
    priority: str | None = None,
    category: str | None = None,
    created_by: str | None = None,
    assigned_to: str | None = None,
    search: str | None = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    feedback_service: FeedbackService = Depends(get_feedback_service),
):
    data = feedback_service.list_tickets(
        filters={
            "status": status,
            "priority": priority,
            "category": category,
            "created_by": created_by,
            "assigned_to": assigned_to,
            "search": search,
            "limit": limit,
            "offset": offset,
        },
        current_user=current_user,
    )
    return _wrap(request, data)


@router.post("/tickets", response_model=ApiResponse)
def create_ticket(
    request: Request,
    payload: FeedbackTicketCreateRequest,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    feedback_service: FeedbackService = Depends(get_feedback_service),
):
    data = feedback_service.create_ticket(
        payload=payload.model_dump(),
        current_user=current_user,
        request_id=get_request_id(request),
    )
    return _wrap(request, data)


@router.get("/tickets/{ticket_id}", response_model=ApiResponse)
def get_ticket(
    request: Request,
    ticket_id: str,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    feedback_service: FeedbackService = Depends(get_feedback_service),
):
    return _wrap(request, feedback_service.get_ticket(ticket_id, current_user))


@router.patch("/tickets/{ticket_id}", response_model=ApiResponse)
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
    )
    return _wrap(request, data)


@router.post("/tickets/{ticket_id}/close", response_model=ApiResponse)
def close_ticket(
    request: Request,
    ticket_id: str,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    feedback_service: FeedbackService = Depends(get_feedback_service),
):
    return _wrap(
        request,
        feedback_service.close_ticket(ticket_id=ticket_id, current_user=current_user),
    )


@router.post("/tickets/{ticket_id}/reopen", response_model=ApiResponse)
def reopen_ticket(
    request: Request,
    ticket_id: str,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    feedback_service: FeedbackService = Depends(get_feedback_service),
):
    return _wrap(
        request,
        feedback_service.reopen_ticket(ticket_id=ticket_id, current_user=current_user),
    )


@router.get("/tickets/{ticket_id}/comments", response_model=ApiResponse)
def list_comments(
    request: Request,
    ticket_id: str,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    feedback_service: FeedbackService = Depends(get_feedback_service),
):
    return _wrap(
        request,
        feedback_service.list_comments(ticket_id=ticket_id, current_user=current_user),
    )


@router.post("/tickets/{ticket_id}/comments", response_model=ApiResponse)
def add_comment(
    request: Request,
    ticket_id: str,
    payload: FeedbackCommentCreateRequest,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    feedback_service: FeedbackService = Depends(get_feedback_service),
):
    return _wrap(
        request,
        feedback_service.add_comment(
            ticket_id=ticket_id,
            body=payload.body,
            current_user=current_user,
        ),
    )


@router.get("/tickets/{ticket_id}/attachments", response_model=ApiResponse)
def list_attachments(
    request: Request,
    ticket_id: str,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    feedback_service: FeedbackService = Depends(get_feedback_service),
):
    return _wrap(
        request,
        feedback_service.list_attachments(
            ticket_id=ticket_id,
            current_user=current_user,
        ),
    )


@router.post("/tickets/{ticket_id}/attachments", response_model=ApiResponse)
async def upload_attachment(
    request: Request,
    ticket_id: str,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    feedback_service: FeedbackService = Depends(get_feedback_service),
):
    upload = await feedback_service.attachment_service.parse_upload_request(request)
    data = feedback_service.add_attachment(
        ticket_id=ticket_id,
        upload=upload,
        current_user=current_user,
        request_id=get_request_id(request),
    )
    return _wrap(request, data)


@router.get("/attachments/{attachment_id}/download")
def download_attachment(
    request: Request,
    attachment_id: str,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    feedback_service: FeedbackService = Depends(get_feedback_service),
):
    attachment, path = feedback_service.get_attachment_for_download(
        attachment_id,
        current_user,
    )
    return FileResponse(
        path=path,
        filename=attachment.original_filename,
        media_type=attachment.content_type,
        headers={"X-Request-ID": get_request_id(request)},
    )


@router.delete("/attachments/{attachment_id}", response_model=ApiResponse)
def delete_attachment(
    request: Request,
    attachment_id: str,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    feedback_service: FeedbackService = Depends(get_feedback_service),
):
    return _wrap(
        request,
        feedback_service.delete_attachment(
            attachment_id=attachment_id,
            current_user=current_user,
        ),
    )
