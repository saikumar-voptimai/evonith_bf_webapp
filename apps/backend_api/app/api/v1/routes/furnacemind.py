"""API v1 FurnaceMind routes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, Header, Query, Request, status
from fastapi.responses import FileResponse

from apps.backend_api.app.api.v1.schemas.common import ApiMeta, ApiResponse
from apps.backend_api.app.api.v1.schemas.furnacemind import (
    ConversationCreateRequest,
    ConversationUpdateRequest,
    MessageCreateRequest,
    MessageFeedbackRequest,
    ReportCreateRequest,
    RunCreateRequest,
    SkillCreateRequest,
    SkillPatchRequest,
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
    return ApiResponse(request_id=get_request_id(request), data=data, meta=ApiMeta(warnings=warnings or []))


def _permission(permission: str) -> Callable[..., dict[str, Any] | None]:
    def dependency(
        request: Request,
        current_user: dict[str, Any] | None = Depends(get_optional_current_user),
    ) -> dict[str, Any] | None:
        if not _settings(request).furnacemind_require_auth:
            if current_user is not None:
                request.state.current_user = current_user
            return current_user
        if current_user is None:
            raise ApiError("AUTH_REQUIRED", "Authentication is required.", status_code=401)
        permissions = {str(item) for item in current_user.get("permissions") or []}
        if permission not in permissions:
            raise ApiError("FORBIDDEN", "Insufficient permissions.", status_code=403)
        request.state.current_user = current_user
        return current_user

    return dependency


@router.get("/config", response_model=ApiResponse, operation_id="getFurnaceMindConfig")
def config(
    request: Request,
    _: dict[str, Any] | None = Depends(_permission("furnacemind:read")),
    service: FurnaceMindService = Depends(get_furnacemind_service),
):
    return _wrap(request, service.get_config())


@router.get("/conversations", response_model=ApiResponse, operation_id="listFurnaceMindConversations")
def list_conversations(
    request: Request,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    include_archived: bool = False,
    current_user: dict[str, Any] | None = Depends(_permission("furnacemind:read")),
    service: FurnaceMindService = Depends(get_furnacemind_service),
):
    return _wrap(request, service.list_conversations(filters={"limit": limit, "offset": offset, "include_archived": include_archived}, current_user=current_user))


@router.post("/conversations", response_model=ApiResponse, operation_id="createFurnaceMindConversation")
def create_conversation(
    request: Request,
    payload: ConversationCreateRequest,
    current_user: dict[str, Any] | None = Depends(_permission("furnacemind:run")),
    service: FurnaceMindService = Depends(get_furnacemind_service),
):
    return _wrap(request, service.create_conversation(payload.model_dump(), current_user))


@router.get("/conversations/{conversation_id}", response_model=ApiResponse, operation_id="getFurnaceMindConversation")
def get_conversation(request: Request, conversation_id: str, current_user: dict[str, Any] | None = Depends(_permission("furnacemind:read")), service: FurnaceMindService = Depends(get_furnacemind_service)):
    return _wrap(request, service.get_conversation(conversation_id, current_user))


@router.patch("/conversations/{conversation_id}", response_model=ApiResponse, operation_id="updateFurnaceMindConversation")
def update_conversation(request: Request, conversation_id: str, payload: ConversationUpdateRequest, current_user: dict[str, Any] | None = Depends(_permission("furnacemind:run")), service: FurnaceMindService = Depends(get_furnacemind_service)):
    return _wrap(request, service.update_conversation(conversation_id, payload.model_dump(exclude_unset=True), current_user))


@router.post("/conversations/{conversation_id}/archive", response_model=ApiResponse, operation_id="archiveFurnaceMindConversation")
def archive_conversation(request: Request, conversation_id: str, current_user: dict[str, Any] | None = Depends(_permission("furnacemind:run")), service: FurnaceMindService = Depends(get_furnacemind_service)):
    return _wrap(request, service.archive_conversation(conversation_id, current_user))


@router.post("/conversations/{conversation_id}/finalize", response_model=ApiResponse, operation_id="finalizeFurnaceMindConversation")
def finalize_conversation(request: Request, conversation_id: str, current_user: dict[str, Any] | None = Depends(_permission("furnacemind:run")), service: FurnaceMindService = Depends(get_furnacemind_service)):
    return _wrap(request, service.finalize_conversation(conversation_id, current_user))


@router.get("/conversations/{conversation_id}/messages", response_model=ApiResponse, operation_id="listFurnaceMindMessages")
def list_messages(request: Request, conversation_id: str, current_user: dict[str, Any] | None = Depends(_permission("furnacemind:read")), service: FurnaceMindService = Depends(get_furnacemind_service)):
    return _wrap(request, service.list_messages(conversation_id, current_user))


@router.post("/conversations/{conversation_id}/messages", response_model=ApiResponse, operation_id="createFurnaceMindMessage")
def create_message(request: Request, conversation_id: str, payload: MessageCreateRequest, current_user: dict[str, Any] | None = Depends(_permission("furnacemind:run")), service: FurnaceMindService = Depends(get_furnacemind_service)):
    return _wrap(request, service.create_user_message(conversation_id, payload.model_dump(), current_user))


@router.post("/conversations/{conversation_id}/runs", response_model=ApiResponse, status_code=status.HTTP_202_ACCEPTED, operation_id="createFurnaceMindRun")
def create_run(
    request: Request,
    conversation_id: str,
    payload: RunCreateRequest,
    background_tasks: BackgroundTasks,
    idempotency_key: str = Header(..., alias="Idempotency-Key", min_length=1, max_length=255),
    current_user: dict[str, Any] | None = Depends(_permission("furnacemind:run")),
    service: FurnaceMindService = Depends(get_furnacemind_service),
):
    data = service.create_run(conversation_id, payload.model_dump(), current_user, idempotency_key=idempotency_key, request_id=get_request_id(request))
    if data.get("status") == "pending":
        background_tasks.add_task(service.process_run, data["id"], current_user, request_id=get_request_id(request))
    return _wrap(request, data)


@router.get("/runs/{run_id}", response_model=ApiResponse, operation_id="getFurnaceMindRun")
def get_run(request: Request, run_id: str, current_user: dict[str, Any] | None = Depends(_permission("furnacemind:read")), service: FurnaceMindService = Depends(get_furnacemind_service)):
    return _wrap(request, service.get_run_status(run_id, current_user))


@router.get("/runs/{run_id}/events", response_model=ApiResponse, operation_id="getFurnaceMindRunEvents")
def get_run_events(request: Request, run_id: str, current_user: dict[str, Any] | None = Depends(_permission("furnacemind:read")), service: FurnaceMindService = Depends(get_furnacemind_service)):
    return _wrap(request, service.list_run_events(run_id, current_user))


@router.post("/runs/{run_id}/cancel", response_model=ApiResponse, operation_id="cancelFurnaceMindRun")
def cancel_run(request: Request, run_id: str, current_user: dict[str, Any] | None = Depends(_permission("furnacemind:run")), service: FurnaceMindService = Depends(get_furnacemind_service)):
    return _wrap(request, service.cancel_run(run_id, current_user))


@router.get("/documents", response_model=ApiResponse, operation_id="listFurnaceMindDocuments")
def list_documents(request: Request, limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0), current_user: dict[str, Any] | None = Depends(_permission("furnacemind:read")), service: FurnaceMindService = Depends(get_furnacemind_service)):
    return _wrap(request, service.document_service.list_documents(filters={"limit": limit, "offset": offset}, current_user=current_user))


@router.post("/documents", response_model=ApiResponse, operation_id="uploadFurnaceMindDocument")
async def upload_document(request: Request, current_user: dict[str, Any] | None = Depends(_permission("furnacemind:documents:write")), service: FurnaceMindService = Depends(get_furnacemind_service)):
    upload = await service.document_service.parse_upload_request(request)
    return _wrap(request, service.document_service.store_document(upload=upload, current_user=current_user, request_id=get_request_id(request)))


@router.get("/documents/{document_id}", response_model=ApiResponse, operation_id="getFurnaceMindDocument")
def get_document(request: Request, document_id: str, current_user: dict[str, Any] | None = Depends(_permission("furnacemind:read")), service: FurnaceMindService = Depends(get_furnacemind_service)):
    return _wrap(request, service.document_service.get_document(document_id, current_user=current_user))


@router.delete("/documents/{document_id}", response_model=ApiResponse, operation_id="deleteFurnaceMindDocument")
def delete_document(request: Request, document_id: str, current_user: dict[str, Any] | None = Depends(_permission("furnacemind:documents:write")), service: FurnaceMindService = Depends(get_furnacemind_service)):
    service.memory.delete_document(document_id)
    return _wrap(request, service.document_service.delete_document(document_id, current_user=current_user))


@router.post("/documents/{document_id}/index", response_model=ApiResponse, status_code=status.HTTP_202_ACCEPTED, operation_id="indexFurnaceMindDocument")
def index_document(request: Request, document_id: str, background_tasks: BackgroundTasks, idempotency_key: str = Header(..., alias="Idempotency-Key", min_length=1, max_length=255), current_user: dict[str, Any] | None = Depends(_permission("furnacemind:documents:write")), service: FurnaceMindService = Depends(get_furnacemind_service)):
    data = service.start_document_index(document_id, current_user, idempotency_key=idempotency_key)
    if data.get("status") == "pending" and data.get("id"):
        background_tasks.add_task(service.process_document_index, data["id"], current_user)
    return _wrap(request, data)


@router.get("/documents/{document_id}/index/status", response_model=ApiResponse, operation_id="getFurnaceMindDocumentIndexStatus")
def document_index_status(request: Request, document_id: str, current_user: dict[str, Any] | None = Depends(_permission("furnacemind:read")), service: FurnaceMindService = Depends(get_furnacemind_service)):
    return _wrap(request, service.document_index_status(document_id, current_user))


@router.get("/documents/{document_id}/index/events", response_model=ApiResponse, operation_id="getFurnaceMindDocumentIndexEvents")
def document_index_events(request: Request, document_id: str, current_user: dict[str, Any] | None = Depends(_permission("furnacemind:read")), service: FurnaceMindService = Depends(get_furnacemind_service)):
    return _wrap(request, service.document_index_events(document_id, current_user))


@router.post("/documents/{document_id}/index/cancel", response_model=ApiResponse, operation_id="cancelFurnaceMindDocumentIndex")
def cancel_document_index(request: Request, document_id: str, current_user: dict[str, Any] | None = Depends(_permission("furnacemind:documents:write")), service: FurnaceMindService = Depends(get_furnacemind_service)):
    return _wrap(request, service.cancel_document_index(document_id, current_user))


@router.get("/skills", response_model=ApiResponse, operation_id="listFurnaceMindSkills")
def list_skills(request: Request, limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0), active_only: bool = False, current_user: dict[str, Any] | None = Depends(_permission("furnacemind:read")), service: FurnaceMindService = Depends(get_furnacemind_service)):
    return _wrap(request, service.skill_service.list_skills(filters={"limit": limit, "offset": offset, "active_only": active_only}, current_user=current_user))


@router.post("/skills", response_model=ApiResponse, operation_id="createFurnaceMindSkill")
def create_skill(request: Request, payload: SkillCreateRequest, idempotency_key: str = Header(..., alias="Idempotency-Key", min_length=1, max_length=255), current_user: dict[str, Any] | None = Depends(_permission("furnacemind:skills:write")), service: FurnaceMindService = Depends(get_furnacemind_service)):
    return _wrap(request, service.skill_service.create_skill(payload.model_dump(), current_user, idempotency_key=idempotency_key))


@router.get("/skills/{skill_id}", response_model=ApiResponse, operation_id="getFurnaceMindSkill")
def get_skill(request: Request, skill_id: str, current_user: dict[str, Any] | None = Depends(_permission("furnacemind:read")), service: FurnaceMindService = Depends(get_furnacemind_service)):
    return _wrap(request, service.skill_service.get_skill(skill_id, current_user))


@router.patch("/skills/{skill_id}", response_model=ApiResponse, operation_id="patchFurnaceMindSkill")
def patch_skill(request: Request, skill_id: str, payload: SkillPatchRequest, current_user: dict[str, Any] | None = Depends(_permission("furnacemind:skills:write")), service: FurnaceMindService = Depends(get_furnacemind_service)):
    return _wrap(request, service.skill_service.patch_skill(skill_id, payload.model_dump(exclude_unset=True), current_user))


@router.post("/skills/{skill_id}/index", response_model=ApiResponse, status_code=status.HTTP_202_ACCEPTED, operation_id="indexFurnaceMindSkill")
def index_skill(request: Request, skill_id: str, background_tasks: BackgroundTasks, idempotency_key: str = Header(..., alias="Idempotency-Key", min_length=1, max_length=255), current_user: dict[str, Any] | None = Depends(_permission("furnacemind:skills:write")), service: FurnaceMindService = Depends(get_furnacemind_service)):
    data = service.skill_service.start_index(skill_id, current_user, idempotency_key=idempotency_key)
    if data.get("status") == "pending" and data.get("id"):
        background_tasks.add_task(service.skill_service.process_index, data["id"])
    return _wrap(request, data)


@router.get("/skills/{skill_id}/index/status", response_model=ApiResponse, operation_id="getFurnaceMindSkillIndexStatus")
def skill_index_status(request: Request, skill_id: str, current_user: dict[str, Any] | None = Depends(_permission("furnacemind:read")), service: FurnaceMindService = Depends(get_furnacemind_service)):
    return _wrap(request, service.skill_service.latest_index_status(skill_id, current_user))


@router.get("/skills/{skill_id}/index/events", response_model=ApiResponse, operation_id="getFurnaceMindSkillIndexEvents")
def skill_index_events(request: Request, skill_id: str, current_user: dict[str, Any] | None = Depends(_permission("furnacemind:read")), service: FurnaceMindService = Depends(get_furnacemind_service)):
    return _wrap(request, service.skill_service.index_events(skill_id, current_user))


@router.post("/skills/{skill_id}/index/cancel", response_model=ApiResponse, operation_id="cancelFurnaceMindSkillIndex")
def cancel_skill_index(request: Request, skill_id: str, current_user: dict[str, Any] | None = Depends(_permission("furnacemind:skills:write")), service: FurnaceMindService = Depends(get_furnacemind_service)):
    return _wrap(request, service.skill_service.cancel_index(skill_id, current_user))


@router.get("/reports/config", response_model=ApiResponse, operation_id="getFurnaceMindReportsConfig")
def reports_config(request: Request, _: dict[str, Any] | None = Depends(_permission("furnacemind:read")), service: FurnaceMindService = Depends(get_furnacemind_service)):
    return _wrap(request, service.report_service.config())


@router.get("/reports", response_model=ApiResponse, operation_id="listFurnaceMindReports")
def list_reports(request: Request, limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0), current_user: dict[str, Any] | None = Depends(_permission("furnacemind:read")), service: FurnaceMindService = Depends(get_furnacemind_service)):
    return _wrap(request, service.report_service.list_reports(filters={"limit": limit, "offset": offset}, current_user=current_user))


@router.post("/reports", response_model=ApiResponse, status_code=status.HTTP_202_ACCEPTED, operation_id="createFurnaceMindReport")
def create_report(request: Request, payload: ReportCreateRequest, background_tasks: BackgroundTasks, idempotency_key: str = Header(..., alias="Idempotency-Key", min_length=1, max_length=255), current_user: dict[str, Any] | None = Depends(_permission("furnacemind:reports:run")), service: FurnaceMindService = Depends(get_furnacemind_service)):
    data = service.report_service.create_report(payload.model_dump(), current_user, idempotency_key=idempotency_key)
    if data.get("status") == "pending" and data.get("task_id"):
        background_tasks.add_task(service.report_service.process_report, data["task_id"])
    return _wrap(request, data)


@router.get("/reports/{report_id}", response_model=ApiResponse, operation_id="getFurnaceMindReport")
def get_report(request: Request, report_id: str, current_user: dict[str, Any] | None = Depends(_permission("furnacemind:read")), service: FurnaceMindService = Depends(get_furnacemind_service)):
    return _wrap(request, service.report_service.get_report(report_id, current_user))


@router.get("/reports/{report_id}/events", response_model=ApiResponse, operation_id="getFurnaceMindReportEvents")
def report_events(request: Request, report_id: str, current_user: dict[str, Any] | None = Depends(_permission("furnacemind:read")), service: FurnaceMindService = Depends(get_furnacemind_service)):
    return _wrap(request, service.report_service.report_events(report_id, current_user))


@router.post("/reports/{report_id}/cancel", response_model=ApiResponse, operation_id="cancelFurnaceMindReport")
def cancel_report(request: Request, report_id: str, current_user: dict[str, Any] | None = Depends(_permission("furnacemind:reports:run")), service: FurnaceMindService = Depends(get_furnacemind_service)):
    return _wrap(request, service.report_service.cancel_report(report_id, current_user))


@router.get("/tools", response_model=ApiResponse, operation_id="listFurnaceMindTools")
def list_tools(request: Request, _: dict[str, Any] | None = Depends(_permission("furnacemind:read")), service: FurnaceMindService = Depends(get_furnacemind_service)):
    return _wrap(request, service.list_tools())


@router.get("/feedback", response_model=ApiResponse, operation_id="listFurnaceMindFeedback")
def list_feedback(request: Request, limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0), conversation_id: str | None = None, current_user: dict[str, Any] | None = Depends(_permission("furnacemind:read")), service: FurnaceMindService = Depends(get_furnacemind_service)):
    return _wrap(request, service.list_feedback(filters={"limit": limit, "offset": offset, "conversation_id": conversation_id}, current_user=current_user))


@router.post("/messages/{message_id}/feedback", response_model=ApiResponse, operation_id="submitFurnaceMindFeedback")
def submit_message_feedback(request: Request, message_id: str, payload: MessageFeedbackRequest, current_user: dict[str, Any] | None = Depends(_permission("furnacemind:feedback:write")), service: FurnaceMindService = Depends(get_furnacemind_service)):
    return _wrap(request, service.submit_message_feedback(message_id, payload.model_dump(), current_user))


@router.get("/artifacts/{artifact_id}/download", operation_id="downloadFurnaceMindArtifact")
def download_artifact(request: Request, artifact_id: str, current_user: dict[str, Any] | None = Depends(_permission("furnacemind:artifacts:read"))):
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
    permissions = {str(item) for item in ((current_user or {}).get("permissions") or [])}
    owner_id = str((current_user or {}).get("id") or "")
    if _settings(request).furnacemind_require_auth and metadata.owner_user_id and metadata.owner_user_id != owner_id and "furnacemind:artifacts:read:any" not in permissions:
        raise ApiError("FORBIDDEN", "Insufficient permissions.", status_code=403)
    return FileResponse(path=path, media_type=metadata.content_type, filename=metadata.filename, headers={"X-Request-ID": get_request_id(request)})