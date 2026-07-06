"""Admin API schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class AdminUser(BaseModel):
    id: str
    username: str
    email: str | None = None
    full_name: str | None = None
    role: str
    roles: list[str] = Field(default_factory=list)
    permissions: list[str] = Field(default_factory=list)
    is_active: bool
    created_at: datetime | None = None
    updated_at: datetime | None = None
    last_login_at: datetime | None = None


class UserListResponse(BaseModel):
    items: list[AdminUser]
    total: int
    limit: int
    offset: int


class UserCreateRequest(BaseModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)
    role: str = "user"
    email: str | None = None
    full_name: str | None = None
    is_active: bool = True


class UserUpdateRequest(BaseModel):
    username: str | None = None
    role: str | None = None
    email: str | None = None
    full_name: str | None = None
    is_active: bool | None = None


class PasswordResetRequest(BaseModel):
    new_password: str = Field(..., min_length=1)


class AdminActionResponse(BaseModel):
    ok: bool = True


class RoleInfo(BaseModel):
    role: str
    permissions: list[str] = Field(default_factory=list)


class RolesResponse(BaseModel):
    roles: list[RoleInfo]
    permissions: list[str] = Field(default_factory=list)


class PermissionsResponse(BaseModel):
    permissions: list[str] = Field(default_factory=list)
