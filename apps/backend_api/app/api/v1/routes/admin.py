"""API v1 admin routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request

from apps.backend_api.app.api.v1.schemas.admin import (
    PasswordResetRequest,
    UserCreateRequest,
    UserUpdateRequest,
)
from apps.backend_api.app.api.v1.schemas.common import ApiMeta, ApiResponse
from apps.backend_api.app.core.auth_dependencies import get_admin_service, require_admin_user
from apps.backend_api.app.core.responses import get_request_id
from apps.backend_api.app.services.admin_service import AdminService

router = APIRouter(prefix="/admin", tags=["admin"])


def _wrap(request: Request, data) -> ApiResponse:
    return ApiResponse(
        request_id=get_request_id(request),
        data=data,
        meta=ApiMeta(),
    )


@router.get("/users", response_model=ApiResponse)
def list_users(
    request: Request,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    current_user: dict = Depends(require_admin_user),
    admin_service: AdminService = Depends(get_admin_service),
):
    return _wrap(request, admin_service.list_users(limit=limit, offset=offset))


@router.post("/users", response_model=ApiResponse)
def create_user(
    request: Request,
    payload: UserCreateRequest,
    current_user: dict = Depends(require_admin_user),
    admin_service: AdminService = Depends(get_admin_service),
):
    data = admin_service.create_user(**payload.model_dump())
    return _wrap(request, data)


@router.get("/users/{user_id}", response_model=ApiResponse)
def get_user(
    request: Request,
    user_id: str,
    current_user: dict = Depends(require_admin_user),
    admin_service: AdminService = Depends(get_admin_service),
):
    return _wrap(request, admin_service.get_user(user_id))


@router.patch("/users/{user_id}", response_model=ApiResponse)
def update_user(
    request: Request,
    user_id: str,
    payload: UserUpdateRequest,
    current_user: dict = Depends(require_admin_user),
    admin_service: AdminService = Depends(get_admin_service),
):
    data = admin_service.update_user(
        user_id,
        actor_user=current_user,
        **payload.model_dump(exclude_unset=True),
    )
    return _wrap(request, data)


@router.post("/users/{user_id}/reset-password", response_model=ApiResponse)
def reset_user_password(
    request: Request,
    user_id: str,
    payload: PasswordResetRequest,
    current_user: dict = Depends(require_admin_user),
    admin_service: AdminService = Depends(get_admin_service),
):
    return _wrap(request, admin_service.reset_password(user_id, payload.new_password))


@router.post("/users/{user_id}/deactivate", response_model=ApiResponse)
def deactivate_user(
    request: Request,
    user_id: str,
    current_user: dict = Depends(require_admin_user),
    admin_service: AdminService = Depends(get_admin_service),
):
    return _wrap(
        request,
        admin_service.set_active(user_id, False, actor_user=current_user),
    )


@router.post("/users/{user_id}/activate", response_model=ApiResponse)
def activate_user(
    request: Request,
    user_id: str,
    current_user: dict = Depends(require_admin_user),
    admin_service: AdminService = Depends(get_admin_service),
):
    return _wrap(
        request,
        admin_service.set_active(user_id, True, actor_user=current_user),
    )


@router.get("/roles", response_model=ApiResponse)
def list_roles(
    request: Request,
    current_user: dict = Depends(require_admin_user),
    admin_service: AdminService = Depends(get_admin_service),
):
    return _wrap(request, admin_service.list_roles())


@router.get("/permissions", response_model=ApiResponse)
def list_permissions(
    request: Request,
    current_user: dict = Depends(require_admin_user),
    admin_service: AdminService = Depends(get_admin_service),
):
    return _wrap(request, admin_service.list_permissions())
