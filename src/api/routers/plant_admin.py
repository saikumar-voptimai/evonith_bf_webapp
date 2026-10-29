"""User, hopper and burden-distribution admin of the admin space (SV-4.7, #310). Administrators only.

The Welcome page's Streamlit tools for the SPA: register an account (``identity.users`` + ``identity.user_roles``),
assign materials to hoppers (``ops_config.hopper_raw_material_history``) and change the burden distribution
(``ops_config.burden_history``), with the same database writes (``services.plant_admin``: shared domain and
repository functions). Assignments and burden changes add a wide snapshot effective from the given time and keep the
earlier rows (SCD Type-2); deleting snapshots is explicit. Every write records its audit row in the same transaction
(``user.create``, ``hopper.assign``, ``hopper.delete``, ``burden.update``, ``burden.delete``). Times are UTC in the
responses; the effective time of a write carries its zone (the SPA sends the IST entry as UTC).
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Query, Request
from pydantic import AwareDatetime, BaseModel, Field

from api.auth import request_actor, require_role
from api.errors import ApiError
from api.store import twin_store
from data.twin.repositories import TwinStore
from services import plant_admin as svc

router = APIRouter(prefix="/api/admin", tags=["admin"], dependencies=[Depends(require_role("admin"))])

RoleName = Literal["user", "supervisor", "admin"]
Number = Annotated[float, Field(allow_inf_nan=False)]
Text = Annotated[str, Field(max_length=2000)]
HistoryIds = Annotated[
    list[int], Query(min_length=1, max_length=500, description="Ids of the history rows to delete, count")
]


# ------------------------------------------------------------------------------------------------------- models


class UserView(BaseModel):
    id: str = Field(..., description="Account id (identity.users UUID)")
    username: str = Field(..., description="User name")
    role: RoleName | str = Field(..., description="Role the sign-in uses: user, supervisor or admin")
    roles: list[str] = Field(..., description="Role assignments (identity.user_roles)")
    is_active: bool = Field(..., description="The account can sign in")
    created_at: datetime = Field(..., description="Time the account was created (UTC)")
    updated_at: datetime = Field(..., description="Time the account last changed (UTC)")


class UsersView(BaseModel):
    roles: list[RoleName] = Field(..., description="Roles a new account can have, lowest first")
    users: list[UserView] = Field(..., description="Every account, active or not, by user name")


class NewUser(BaseModel):
    username: str = Field(..., min_length=1, max_length=128, description="User name (leading and trailing blanks "
                          "are removed)")  # fmt: skip
    password: str = Field(..., min_length=1, max_length=256, description="Password (leading and trailing blanks are "
                          "removed; stored as its SHA-256 digest, as the Streamlit sign-in reads it)")  # fmt: skip
    role: RoleName = Field("user", description="Role of the account")


class UserCreated(BaseModel):
    user: UserView = Field(..., description="The new account")
    audit_id: int = Field(..., description="Id of the audit row of the write, count")


class HopperSlot(BaseModel):
    code: str = Field(..., description="Hopper code, hopper_01..hopper_19")
    display_name: str = Field(..., description="Name of the hopper on the plant")


class HopperSnapshot(BaseModel):
    id: int = Field(..., description="Snapshot id (ops_config.hopper_raw_material_history)")
    date_time: datetime = Field(..., description="Time the snapshot is effective from (UTC)")
    source_type: str | None = Field(..., description="Where the snapshot came from, e.g. webapp")
    ip_address: str | None = Field(..., description="Client address of the write")
    user_id: str | None = Field(..., description="Account that wrote it (identity.users UUID)")
    user_name: str | None = Field(..., description="User name of that account")
    materials: dict[str, str] = Field(..., description="Material per active hopper (client name; UNASSIGNED: none)")


class HopperAdminView(BaseModel):
    hoppers: list[HopperSlot] = Field(..., description="Active hoppers in plant order")
    materials: list[str] = Field(..., description="Materials an assignment can name: UNASSIGNED, then client names")
    current: HopperSnapshot | None = Field(..., description="Mapping in force: the latest snapshot (null: none)")
    history: list[HopperSnapshot] = Field(..., description="Every snapshot, newest first")


class HopperAssignment(BaseModel):
    effective_from: AwareDatetime = Field(..., description="Time the mapping applies from, with its zone (ISO 8601)")
    materials: dict[str, Text] = Field(
        ..., description="Material per hopper as the form shows it (client name; UNASSIGNED: none); hoppers that "
        "equal the mapping in force are not written",
    )  # fmt: skip


class SnapshotWrite(BaseModel):
    written: bool = Field(..., description="A snapshot was added (false: no field differs from the one in force)")
    id: int | None = Field(..., description="Id of the new snapshot (null: none written)")
    changed: list[str] = Field(..., description="Hoppers or burden fields the snapshot changes")
    audit_id: int | None = Field(..., description="Id of the audit row of the write (null: none written), count")


class SnapshotsDeleted(BaseModel):
    deleted: list[int] = Field(..., description="Ids of the history rows deleted")
    audit_id: int = Field(..., description="Id of the audit row that keeps the deleted rows, count")


class BurdenField(BaseModel):
    name: str = Field(..., description="Column of ops_config.burden_history")
    label: str = Field(..., description="Label for the form")
    kind: Literal["number", "text"] = Field(..., description="number or text")
    unit: str | None = Field(
        ..., description="Unit of a number: rotations (rings: chute rotations at the position), ° (chute angle) or "
        "s (discharge time); null for text",
    )  # fmt: skip
    charge: Literal["coke", "non_coke"] | None = Field(..., description="Charge the field belongs to (null: both)")
    position: int | None = Field(..., description="Chute position P01..P11 of a ring or angle field, count")
    quantity: Literal["rings", "angles"] | None = Field(..., description="rings or angles at the position")


class BurdenSnapshot(BaseModel):
    id: int = Field(..., description="Snapshot id (ops_config.burden_history)")
    date_time: datetime = Field(..., description="Time the snapshot is effective from (UTC)")
    source_type: str | None = Field(..., description="Where the snapshot came from, e.g. webapp")
    ip_address: str | None = Field(..., description="Client address of the write")
    user_id: str | None = Field(..., description="Account that wrote it (identity.users UUID)")
    user_name: str | None = Field(..., description="User name of that account")
    values: dict[str, float | str | None] = Field(
        ..., description="Value per field: rings in rotations, angles in °, discharge times in s, text as entered "
        "(null: not set)",
    )  # fmt: skip


class BurdenAdminView(BaseModel):
    fields: list[BurdenField] = Field(..., description="Fields in table order")
    at: datetime = Field(..., description="Time the snapshot in force is read at (UTC)")
    current: BurdenSnapshot | None = Field(..., description="Snapshot in force at `at` (null: none)")
    history: list[BurdenSnapshot] = Field(..., description="Every snapshot, newest first")


class BurdenChange(BaseModel):
    effective_from: AwareDatetime = Field(..., description="Time the values apply from, with its zone (ISO 8601)")
    values: dict[str, Number | Text | None] = Field(
        ..., description="Value per field as the form shows it: rings in rotations, angles in °, discharge times "
        "in s, text for patterns and purpose (null or empty: not set); fields equal to the snapshot in force are not "
        "written",
    )  # fmt: skip


# ------------------------------------------------------------------------------------------------------- helpers


def _sentence(exc: Exception) -> str:
    message = str(exc).strip()
    message = message[:1].upper() + message[1:]
    return message if message.endswith(".") else message + "."


ERRORS = (svc.Invalid, svc.Conflict, svc.NotFound, svc.Unavailable)


def _problem(exc: Exception) -> ApiError:
    if isinstance(exc, svc.NotFound):
        return ApiError(404, f"No {exc}.")
    if isinstance(exc, svc.Conflict):
        return ApiError(409, _sentence(exc))
    if isinstance(exc, svc.Invalid):
        return ApiError(422, _sentence(exc))
    if isinstance(exc, svc.Unavailable):
        return ApiError(503, _sentence(exc))
    raise exc


def _call(fn: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return fn(*args, **kwargs)
    except ERRORS as exc:
        raise _problem(exc) from exc


# -------------------------------------------------------------------------------------------------------- routes


@router.get("/users", response_model=UsersView, summary="User accounts")
def list_users(store: Annotated[TwinStore, Depends(twin_store)]) -> UsersView:
    """Every account of ``identity.users`` with its role assignments."""
    return UsersView.model_validate(_call(svc.users_view, store))


@router.post("/users", response_model=UserCreated, status_code=201, summary="Register a user")
def register_user(body: NewUser, request: Request, store: Annotated[TwinStore, Depends(twin_store)]) -> UserCreated:
    """Add an account as the Streamlit register form does, with its audit row (``user.create``). 409 when the user
    name exists; 422 when the name or password is blank."""
    result = _call(svc.register_user, store, request_actor(request), body.username, body.password, body.role)
    return UserCreated.model_validate(result)


@router.get("/hoppers", response_model=HopperAdminView, summary="Hopper materials")
def hopper_mapping(store: Annotated[TwinStore, Depends(twin_store)]) -> HopperAdminView:
    """Active hoppers, the materials they can hold, the mapping in force and every snapshot."""
    return HopperAdminView.model_validate(_call(svc.hopper_view, store))


@router.post("/hoppers", response_model=SnapshotWrite, summary="Assign materials to hoppers")
def assign_hoppers(
    body: HopperAssignment, request: Request, store: Annotated[TwinStore, Depends(twin_store)]
) -> SnapshotWrite:
    """Add a snapshot with the hoppers that differ from the mapping in force, effective from ``effective_from``, with
    its audit row (``hopper.assign``). ``written`` is false when nothing differs. 422 for an unknown hopper or a
    material not offered."""
    result = _call(svc.assign_hoppers, store, request_actor(request), body.materials, body.effective_from)
    return SnapshotWrite.model_validate(result)


@router.delete("/hoppers/history", response_model=SnapshotsDeleted, summary="Delete hopper snapshots")
def delete_hopper_history(
    ids: HistoryIds, request: Request, store: Annotated[TwinStore, Depends(twin_store)]
) -> SnapshotsDeleted:
    """Delete snapshots, with one audit row (``hopper.delete``) that keeps them. 404 when an id does not exist
    (nothing is deleted)."""
    return SnapshotsDeleted.model_validate(_call(svc.delete_hopper_history, store, request_actor(request), ids))


@router.get("/burden", response_model=BurdenAdminView, summary="Burden distribution")
def burden_distribution(store: Annotated[TwinStore, Depends(twin_store)]) -> BurdenAdminView:
    """The fields, the snapshot in force now and every snapshot."""
    return BurdenAdminView.model_validate(_call(svc.burden_view, store))


@router.post("/burden", response_model=SnapshotWrite, summary="Change the burden distribution")
def update_burden(
    body: BurdenChange, request: Request, store: Annotated[TwinStore, Depends(twin_store)]
) -> SnapshotWrite:
    """Add a snapshot with the fields that differ from the snapshot in force now, effective from
    ``effective_from``, with its audit row (``burden.update``). ``written`` is false when nothing differs. 422 for
    an unknown field or a value of the wrong kind."""
    result = _call(svc.update_burden, store, request_actor(request), body.values, body.effective_from)
    return SnapshotWrite.model_validate(result)


@router.delete("/burden/history", response_model=SnapshotsDeleted, summary="Delete burden snapshots")
def delete_burden_history(
    ids: HistoryIds, request: Request, store: Annotated[TwinStore, Depends(twin_store)]
) -> SnapshotsDeleted:
    """Delete snapshots, with one audit row (``burden.delete``) that keeps them. 404 when an id does not exist
    (nothing is deleted)."""
    return SnapshotsDeleted.model_validate(_call(svc.delete_burden_history, store, request_actor(request), ids))
