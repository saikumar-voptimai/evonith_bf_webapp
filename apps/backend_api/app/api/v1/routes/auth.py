"""API v1 auth routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from apps.backend_api.app.api.v1.schemas.auth import ChangePasswordRequest, LoginRequest
from apps.backend_api.app.api.v1.schemas.common import ApiMeta, ApiResponse
from apps.backend_api.app.core.auth_dependencies import get_auth_service, require_authenticated_user
from apps.backend_api.app.core.responses import get_request_id
from apps.backend_api.app.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["auth"])


def _wrap(request: Request, data) -> ApiResponse:
    return ApiResponse(
        request_id=get_request_id(request),
        data=data,
        meta=ApiMeta(),
    )


@router.post("/login", response_model=ApiResponse)
def login(
    request: Request,
    payload: LoginRequest,
    auth_service: AuthService = Depends(get_auth_service),
):
    data = auth_service.login(username=payload.username, password=payload.password)
    return _wrap(request, data)


@router.get("/me", response_model=ApiResponse)
def me(
    request: Request,
    current_user: dict = Depends(require_authenticated_user),
):
    return _wrap(request, {"user": current_user})


@router.post("/logout", response_model=ApiResponse)
def logout(
    request: Request,
    _current_user: dict = Depends(require_authenticated_user),
):
    return _wrap(request, {"logged_out": True, "token_revoked": False})


@router.post("/change-password", response_model=ApiResponse)
def change_password(
    request: Request,
    payload: ChangePasswordRequest,
    current_user: dict = Depends(require_authenticated_user),
    auth_service: AuthService = Depends(get_auth_service),
):
    data = auth_service.change_password(
        user_id=str(current_user["id"]),
        current_password=payload.current_password,
        new_password=payload.new_password,
    )
    return _wrap(request, data)
