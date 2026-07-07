"""API v1 FurnaceMind routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import FileResponse

from apps.backend_api.app.api.v1.schemas.common import ApiMeta, ApiResponse
from apps.backend_api.app.api.v1.schemas.furnacemind import (
    ConversationCreateRequest,
    ConversationUpdateRequest,
    MessageCreateRequest,
    MessageFeedbackRequest,
    RunCreateRequest,
)
from apps.backend_api.app.core.auth_dependencies import get_optional_current_user
from apps.backend_api.app.core.config import BackendSettings
from apps.backend_api.app.core.errors import ApiError
from apps.backend_api.app.core.responses import get_request_id
from apps.backend_api.app.services.compute_artifact_service import ComputeArtifactService
from apps.backend_api.app.services.furnacemind_service import FurnaceMindService

router = APIRouter(prefix="/furnacemind", tags=["furnacemind"])


def _settings(request: Request) -> BackendSettings:
    return request.app.state.backend_settings


def get_furnacemind_service(request: Request) -> FurnaceMindService:
    service = getattr(request.app.state, "furnacemind_service", None)
    if service is not None:
        return service
    service = FurnaceMindService(settings=_settings(request))
    request.app.state.furnacemind_service = service
    return service


def _wrap(request: Request, data: Any, warnings: list[str] | None = None) -> ApiResponse:
    return ApiResponse(
        request_id=get_request_id(request),
        data=data,
        meta=ApiMeta(warnings=warnings or []),
    )


def _require_furnacemind_user(settings: BackendSettings, user: dict[str, Any] | None) -> None:
    if settings.furnacemind_require_auth and not user:
        raise ApiError("AUTH_REQUIRED", "Authentication is required.", status_code=401)


@router.get("/config", response_model=ApiResponse)
def config(
    request: Request,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    service: FurnaceMindService = Depends(get_furnacemind_service),
):
    _require_furnacemind_user(_settings(request), current_user)
    return _wrap(request, service.get_config())


@router.get("/conversations", response_model=ApiResponse)
def list_conversations(
    request: Request,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    include_archived: bool = False,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    service: FurnaceMindService = Depends(get_furnacemind_service),
):
    return _wrap(
        request,
        service.list_conversations(
            filters={"limit": limit, "offset": offset, "include_archived": include_archived},
            current_user=current_user,
        ),
    )


@router.post("/conversations", response_model=ApiResponse)
def create_conversation(
    request: Request,
    payload: ConversationCreateRequest,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    service: FurnaceMindService = Depends(get_furnacemind_service),
):
    return _wrap(request, service.create_conversation(payload.model_dump(), current_user))


@router.get("/conversations/{conversation_id}", response_model=ApiResponse)
def get_conversation(
    request: Request,
    conversation_id: str,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    service: FurnaceMindService = Depends(get_furnacemind_service),
):
    return _wrap(request, service.get_conversation(conversation_id, current_user))


@router.patch("/conversations/{conversation_id}", response_model=ApiResponse)
def update_conversation(
    request: Request,
    conversation_id: str,
    payload: ConversationUpdateRequest,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    service: FurnaceMindService = Depends(get_furnacemind_service),
):
    return _wrap(
        request,
        service.update_conversation(conversation_id, payload.model_dump(exclude_unset=True), current_user),
    )


@router.post("/conversations/{conversation_id}/archive", response_model=ApiResponse)
def archive_conversation(
    request: Request,
    conversation_id: str,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    service: FurnaceMindService = Depends(get_furnacemind_service),
):
    return _wrap(request, service.archive_conversation(conversation_id, current_user))


@router.get("/conversations/{conversation_id}/messages", response_model=ApiResponse)
def list_messages(
    request: Request,
    conversation_id: str,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    service: FurnaceMindService = Depends(get_furnacemind_service),
):
    return _wrap(request, service.list_messages(conversation_id, current_user))


@router.post("/conversations/{conversation_id}/messages", response_model=ApiResponse)
def create_message(
    request: Request,
    conversation_id: str,
    payload: MessageCreateRequest,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    service: FurnaceMindService = Depends(get_furnacemind_service),
):
    return _wrap(
        request,
        service.create_user_message(conversation_id, payload.model_dump(), current_user),
    )


@router.post("/conversations/{conversation_id}/runs", response_model=ApiResponse)
def create_run(
    request: Request,
    conversation_id: str,
    payload: RunCreateRequest,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    service: FurnaceMindService = Depends(get_furnacemind_service),
):
    return _wrap(
        request,
        service.create_run(
            conversation_id,
            payload.model_dump(),
            current_user,
            request_id=get_request_id(request),
        ),
    )


@router.get("/runs/{run_id}", response_model=ApiResponse)
def get_run(
    request: Request,
    run_id: str,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    service: FurnaceMindService = Depends(get_furnacemind_service),
):
    return _wrap(request, service.get_run_status(run_id, current_user))


@router.get("/runs/{run_id}/events", response_model=ApiResponse)
def get_run_events(
    request: Request,
    run_id: str,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    service: FurnaceMindService = Depends(get_furnacemind_service),
):
    return _wrap(request, service.list_run_events(run_id, current_user))


@router.get("/documents", response_model=ApiResponse)
def list_documents(
    request: Request,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    service: FurnaceMindService = Depends(get_furnacemind_service),
):
    _require_furnacemind_user(_settings(request), current_user)
    return _wrap(
        request,
        service.document_service.list_documents(
            filters={"limit": limit, "offset": offset},
            current_user=current_user,
        ),
    )


@router.post("/documents", response_model=ApiResponse)
async def upload_document(
    request: Request,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    service: FurnaceMindService = Depends(get_furnacemind_service),
):
    _require_furnacemind_user(_settings(request), current_user)
    upload = await service.document_service.parse_upload_request(request)
    return _wrap(
        request,
        service.document_service.store_document(
            upload=upload,
            current_user=current_user,
            request_id=get_request_id(request),
        ),
    )


@router.get("/documents/{document_id}", response_model=ApiResponse)
def get_document(
    request: Request,
    document_id: str,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    service: FurnaceMindService = Depends(get_furnacemind_service),
):
    _require_furnacemind_user(_settings(request), current_user)
    return _wrap(request, service.document_service.get_document(document_id, current_user=current_user))


@router.delete("/documents/{document_id}", response_model=ApiResponse)
def delete_document(
    request: Request,
    document_id: str,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    service: FurnaceMindService = Depends(get_furnacemind_service),
):
    _require_furnacemind_user(_settings(request), current_user)
    service.memory.delete_document(document_id)
    return _wrap(request, service.document_service.delete_document(document_id, current_user=current_user))


@router.post("/documents/{document_id}/index", response_model=ApiResponse)
def index_document(
    request: Request,
    document_id: str,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    service: FurnaceMindService = Depends(get_furnacemind_service),
):
    _require_furnacemind_user(_settings(request), current_user)
    return _wrap(
        request,
        service.document_service.index_document(
            document_id,
            current_user=current_user,
            memory_service=service.memory,
        ),
    )


@router.get("/tools", response_model=ApiResponse)
def list_tools(
    request: Request,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    service: FurnaceMindService = Depends(get_furnacemind_service),
):
    _require_furnacemind_user(_settings(request), current_user)
    return _wrap(request, service.list_tools())


@router.get("/artifacts/{artifact_id}/download")
def download_artifact(
    request: Request,
    artifact_id: str,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
):
    _require_furnacemind_user(_settings(request), current_user)
    service = ComputeArtifactService(_settings(request))
    try:
        metadata = service.get_metadata(artifact_id)
        path = service.get_path(artifact_id)
    except ValueError as exc:
        raise ApiError("FURNACEMIND_ARTIFACT_NOT_FOUND", "Invalid artifact id.", status_code=400) from exc
    except FileNotFoundError as exc:
        raise ApiError("FURNACEMIND_ARTIFACT_NOT_FOUND", "Artifact not found.", status_code=404) from exc
    if metadata.workflow != "furnacemind":
        raise ApiError("FURNACEMIND_ARTIFACT_NOT_FOUND", "Artifact not found.", status_code=404)
    return FileResponse(
        path=path,
        media_type=metadata.content_type,
        filename=metadata.filename,
        headers={"X-Request-ID": get_request_id(request)},
    )


@router.post("/messages/{message_id}/feedback", response_model=ApiResponse)
def submit_message_feedback(
    request: Request,
    message_id: str,
    payload: MessageFeedbackRequest,
    current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    service: FurnaceMindService = Depends(get_furnacemind_service),
):
    return _wrap(
        request,
        service.submit_message_feedback(message_id, payload.model_dump(), current_user),
    )
