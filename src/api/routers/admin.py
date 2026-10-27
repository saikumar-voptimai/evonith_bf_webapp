"""Admin space (SV-4.1, #304): the audit log of admin writes and plan decisions. Administrators only.

``GET /api/admin/audit`` pages through ``twin.audit_log`` newest first, filtered by user, action, object type and id
and time; each entry carries the object's fields before and after an admin write and the fields that changed
(``services.audit``). The admin views of #305-#311 add their routes to this router; their writes go through
``services.audit.admin_write``, which records one audit row in the transaction of the write. Times are UTC.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from api.auth import require_role
from api.errors import ApiError
from api.store import twin_store
from data.twin.repositories import TwinStore
from services import audit

router = APIRouter(prefix="/api/admin", tags=["admin"], dependencies=[Depends(require_role("admin"))])


class AuditChange(BaseModel):
    path: str = Field(
        ..., description="Changed field: keys joined by dots, list items by index (e.g. overlay.ASM-08.value)"
    )
    kind: Literal["added", "removed", "changed"] = Field(
        ..., description="added: the field is only in after; removed: only in before; changed: in both, different"
    )
    before: Any = Field(..., description="Value before the write (JSON; null when added); units in the field name")
    after: Any = Field(..., description="Value after the write (JSON; null when removed); units in the field name")


class AuditEntry(BaseModel):
    id: int = Field(..., description="Entry id (the log is append-only)")
    at: datetime = Field(..., description="Time of the action (UTC)")
    actor: str = Field(..., description="User name of the actor")
    role: str | None = Field(..., description="Role of the actor at the time: admin, supervisor or user")
    action: str = Field(..., description="What was done, e.g. plan.decide, facts.propose, hopper.assign")
    object_type: str = Field(..., description="Kind of object written, e.g. optimiser_run, facts_version, hopper")
    object_id: str | None = Field(..., description="Id of the object written")
    client_ip: str | None = Field(..., description="Client address of the request")
    details: dict[str, Any] = Field(..., description="Context of the action (decision, comment, note)")
    before: dict[str, Any] | None = Field(
        ..., description="The object's fields before an admin write; null for a created object and for entries "
        "that record only details",
    )  # fmt: skip
    after: dict[str, Any] | None = Field(
        ..., description="The object's fields after an admin write; null for a deleted object and for entries that "
        "record only details",
    )  # fmt: skip
    changes: list[AuditChange] = Field(
        ..., description="Fields that differ between before and after, in the order of before"
    )


class AuditFacets(BaseModel):
    actors: list[str] = Field(..., description="User names present in the log, sorted")
    actions: list[str] = Field(..., description="Actions present in the log, sorted")
    object_types: list[str] = Field(..., description="Object types present in the log, sorted")


class AuditPage(BaseModel):
    total: int = Field(..., description="Entries that match the filters, count")
    limit: int = Field(..., description="Page size, count")
    offset: int = Field(..., description="Entries before this page, count")
    entries: list[AuditEntry] = Field(..., description="The page's entries, newest first")
    facets: AuditFacets = Field(..., description="Values present in the whole log, for the filter menus")


def _aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _text(value: str | None) -> str | None:
    return (value or "").strip() or None


@router.get("/audit", response_model=AuditPage, summary="Audit log")
def audit_log(
    store: Annotated[TwinStore, Depends(twin_store)],
    actor: Annotated[str | None, Query(max_length=128, description="Only this user (user name)")] = None,
    action: Annotated[str | None, Query(max_length=64, description="Only this action, e.g. plan.decide")] = None,
    object_type: Annotated[
        str | None, Query(max_length=64, description="Only this object type, e.g. facts_version")
    ] = None,
    object_id: Annotated[str | None, Query(max_length=128, description="Only this object id")] = None,
    since: Annotated[datetime | None, Query(description="From this time, inclusive (ISO 8601; naive is UTC)")] = None,
    until: Annotated[datetime | None, Query(description="Before this time, exclusive (ISO 8601; naive is UTC)")] = None,
    limit: Annotated[int, Query(ge=1, le=200, description="Page size, count")] = 50,
    offset: Annotated[int, Query(ge=0, description="Entries to skip, count")] = 0,
) -> AuditPage:
    """Entries newest first with their values before and after and the changed fields; ``total`` counts every entry
    that matches, ``facets`` lists the users, actions and object types of the whole log."""
    filters = audit.AuditFilter(
        actor=_text(actor),
        action=_text(action),
        object_type=_text(object_type),
        object_id=_text(object_id),
        since=_aware(since),
        until=_aware(until),
    )
    if filters.since is not None and filters.until is not None and filters.since >= filters.until:
        raise ApiError(422, "since must lie before until.")
    return AuditPage.model_validate(audit.read_page(store, filters, limit=limit, offset=offset))
