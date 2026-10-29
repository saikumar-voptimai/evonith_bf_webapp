"""Fact register editor of the admin space (SV-4.2, #305). Administrators only.

The register (``src/config/facts_register.json`` v1.8, 116 facts) by group and status; a change of a fact's value,
band or status is proposed as a version of ``twin.facts_version`` (an overlay on the file) and reviewed: a change of
status needs a second administrator. The approved version with the highest number made against the file's version is
in force; the engine loads it (``utils.burden_engine.facts.load_register``), else the file. Versions are compared
field by field and an approved version is exported as the register file for a repository PR (schema-valid).
Every write records its audit row with the version's fields before and after (``facts.propose``, ``facts.review``).
Values and bands are in each fact's ``unit``; times UTC.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Path, Query, Request, Response
from pydantic import BaseModel, Field

from api.auth import auth_state, client_address, require_role
from api.errors import ApiError
from api.store import optional_twin_store, twin_store
from data.twin.repositories import Actor, NotFound, ReviewRefused, TwinStore
from services import facts as svc
from services.auth import Principal

router = APIRouter(prefix="/api/admin/facts", tags=["admin"], dependencies=[Depends(require_role("admin"))])

Number = Annotated[float, Field(allow_inf_nan=False)]
FactField = Literal["value", "band", "status"]
VersionStatus = Literal["pending", "approved", "rejected"]


class FactView(BaseModel):
    id: str = Field(..., description="Fact id, GROUP-nn (e.g. ASM-08)")
    group: str = Field(..., description="Group: COKE, GAS, FLOW, DATA, BAND, ASM, MOD or VAL")
    statement: str = Field(..., description="What the fact states")
    value: float | None = Field(..., description="Value of the fact (null: none), in the fact's unit")
    unit: str = Field(..., description="Unit of value, se and band (empty: dimensionless or several in params)")
    se: float | None = Field(..., description="Standard error of value (null: none), in the fact's unit")
    band: list[float] | None = Field(
        ...,
        description="[lo, hi]: the plausibility band (params lo / hi) or the range (null: none), in the fact's unit",
    )
    band_kind: Literal["plausibility", "range"] = Field(
        ..., description="plausibility: params lo / hi, a band the engine checks plans against; range: the range"
    )
    status: str = Field(..., description="Status, one of the register's statuses (Retired: never used)")
    grade: str = Field(..., description="Evidence grade A-D (the register's grades)")
    basis: str = Field(..., description="Basis of the value")
    literature: str = Field(..., description="Published values")
    validity_window: str = Field(..., description="Where the value holds")
    next_test: str = Field(..., description="Next test of the fact")
    version: str = Field(..., description="Register version that last changed the fact")
    date: str = Field(..., description="Date of that change (YYYY-MM-DD)")
    note: str = Field(..., description="Note")
    params: dict[str, Any] = Field(..., description="Parameters the engine reads (units in the keys or the unit)")
    retired: bool = Field(..., description="Status Retired: shown, refused to the engine (RetiredFactError)")
    changed: list[FactField] = Field(..., description="Fields the shown version changes against the repository file")
    file: dict[str, Any] | None = Field(
        ..., description="The repository file's values of the changed fields (value and band in the fact's unit)"
    )


class FactRegisterView(BaseModel):
    name: str = Field(..., description="Register name")
    file_version: str = Field(..., description="Version of the repository file, e.g. 1.8")
    file_date: str = Field(..., description="Date of the repository file (YYYY-MM-DD)")
    statuses: list[str] = Field(..., description="Statuses a fact can have")
    grades: dict[str, str] = Field(..., description="Evidence grades and their meaning")
    groups: list[str] = Field(..., description="Groups in register order")
    in_force_version: int = Field(..., description="Approved version in force (0: the repository file), count")
    version: int = Field(..., description="Version shown (0: the repository file), count")
    version_status: Literal["file", "pending", "approved", "rejected"] = Field(
        ..., description="Status of the version shown (file: the repository file)"
    )
    label: str = Field(..., description="Register shown, e.g. 1.8 or 1.8 + overlay 3")
    reason: str | None = Field(..., description="Why the repository file is shown when the twin store is missing")
    facts: list[FactView] = Field(..., description="Facts in register order")


class FactsVersionView(BaseModel):
    version: int = Field(..., description="Version number, count")
    parent_version: int = Field(..., description="Version in force it was made from (0: the repository file), count")
    base_version: str = Field(..., description="Version of the repository file it applies to")
    status: VersionStatus = Field(..., description="pending until an administrator approves or rejects it")
    second_review: bool = Field(..., description="Changes a fact's status: another administrator must review it")
    note: str | None = Field(..., description="Reason given with the proposal")
    created_by: str = Field(..., description="Administrator who proposed it")
    created_at: datetime = Field(..., description="Time of the proposal (UTC)")
    reviewed_by: str | None = Field(..., description="Administrator who approved or rejected it")
    reviewed_at: datetime | None = Field(..., description="Time of the review (UTC)")
    facts: list[str] = Field(..., description="Facts that differ from the repository file in this version")
    content_hash: str = Field(..., description="SHA-256 of the overlay (canonical JSON)")
    in_force: bool = Field(..., description="The version the engine loads")
    applies: bool = Field(..., description="Made against the current repository file's version")


class FactChange(BaseModel):
    model_config = {"extra": "forbid"}

    value: Number | None = Field(None, description="New value (null: no value), in the fact's unit")
    band: Annotated[list[Number], Field(min_length=2, max_length=2)] | None = Field(
        None, description="New [lo, hi] with lo <= hi (null removes a range), in the fact's unit"
    )
    status: str | None = Field(None, min_length=1, max_length=40, description="New status (one of the statuses)")


class FactsProposal(BaseModel):
    model_config = {"extra": "forbid"}

    parent_version: int = Field(
        ..., ge=0, description="Version in force the edit started from (GET's in_force_version; 0: the file), count"
    )
    changes: dict[str, FactChange] = Field(
        ..., min_length=1, description="Changes by fact id; a field left out keeps the value in force"
    )
    note: str = Field(..., min_length=3, max_length=1000, description="Reason for the change, saved with the version")


class FactsReview(BaseModel):
    model_config = {"extra": "forbid"}

    approve: bool = Field(..., description="true approves the version (it comes into force), false rejects it")
    note: str | None = Field(None, max_length=1000, description="Reason, recorded in the audit log")


class FactDiffRow(BaseModel):
    id: str = Field(..., description="Fact id")
    group: str = Field(..., description="Group of the fact")
    statement: str = Field(..., description="What the fact states")
    unit: str = Field(..., description="Unit of value and band")
    field: FactField = Field(..., description="Field that differs")
    before: float | list[float] | str | None = Field(
        ..., description="Value in from_version (value, band [lo, hi] or status text), in the fact's unit"
    )
    after: float | list[float] | str | None = Field(
        ..., description="Value in to_version (value, band [lo, hi] or status text), in the fact's unit"
    )


class FactsDiff(BaseModel):
    from_version: int = Field(..., description="Version compared from (0: the repository file), count")
    to_version: int = Field(..., description="Version compared to (0: the repository file), count")
    changes: list[FactDiffRow] = Field(..., description="Fields that differ, in register order")


def _actor(request: Request) -> Actor:
    principal: Principal = request.state.principal
    return Actor(
        name=principal.username,
        role=principal.role,
        client_ip=client_address(request, auth_state(request).settings.trusted_proxy_set),
    )


def _sentence(exc: Exception) -> str:
    message = str(exc).strip()
    message = message[:1].upper() + message[1:]
    return message if message.endswith(".") else message + "."


def _problem(exc: Exception) -> ApiError:
    if isinstance(exc, NotFound):
        return ApiError(404, f"No {exc}.")
    if isinstance(exc, ReviewRefused | svc.FactsConflict):
        return ApiError(409, _sentence(exc))
    if isinstance(exc, svc.InvalidOverlay):
        return ApiError(422, "The register would not be usable: " + "; ".join(exc.problems) + ".")
    if isinstance(exc, svc.Unavailable):
        return ApiError(503, str(exc))
    raise exc


ERRORS = (NotFound, ReviewRefused, svc.FactsConflict, svc.InvalidOverlay, svc.Unavailable)


@router.get("", response_model=FactRegisterView, summary="Fact register")
def get_register(
    store: Annotated[TwinStore | None, Depends(optional_twin_store)],
    version: Annotated[
        int | None,
        Query(ge=0, description="Version to show (0: the repository file; default: the one in force), count"),
    ] = None,
) -> FactRegisterView:
    """The facts of the version in force (or of ``version``) with the fields it changes against the repository
    file. Without a twin store the repository file, with the ``reason``."""
    try:
        return FactRegisterView.model_validate(svc.register_view(store, version))
    except ERRORS as exc:
        raise _problem(exc) from exc


@router.get("/versions", response_model=list[FactsVersionView], summary="Fact register versions")
def list_versions(
    store: Annotated[TwinStore, Depends(twin_store)],
    limit: Annotated[int, Query(ge=1, le=500, description="Versions to return, count")] = 100,
) -> list[FactsVersionView]:
    """Newest first, with the one in force marked."""
    return [FactsVersionView.model_validate(v) for v in svc.versions(store, limit=limit)]


@router.post("/versions", response_model=FactsVersionView, status_code=201, summary="Propose a fact register version")
def propose_version(
    body: FactsProposal, request: Request, store: Annotated[TwinStore, Depends(twin_store)]
) -> FactsVersionView:
    """Save the changes on the version in force as the next pending version, with its audit row
    (``facts.propose``). 409 when another version is in force than ``parent_version``; 422 when the register would
    fail the JSON schema or the engine's checks (e.g. retiring a fact the engine reads), or nothing changes."""
    changes = {fid: change.model_dump(include=change.model_fields_set) for fid, change in body.changes.items()}
    try:
        view = svc.propose(
            store, _actor(request), parent_version=body.parent_version, changes=changes, note=body.note.strip()
        )
    except ERRORS as exc:
        raise _problem(exc) from exc
    return FactsVersionView.model_validate(view)


@router.post("/versions/{version}/review", response_model=FactsVersionView, summary="Approve or reject a version")
def review_version(
    version: Annotated[int, Path(ge=1, description="Version to review, count")],
    body: FactsReview,
    request: Request,
    store: Annotated[TwinStore, Depends(twin_store)],
) -> FactsVersionView:
    """Approve or reject a pending version, with its audit row (``facts.review``). 409 when it is not pending, when
    it changes a status and the reviewer proposed it, or (approval) when it was not made from the version in force."""
    note = (body.note or "").strip() or None
    try:
        view = svc.review(store, _actor(request), version, approve=body.approve, note=note)
    except ERRORS as exc:
        raise _problem(exc) from exc
    return FactsVersionView.model_validate(view)


@router.get("/diff", response_model=FactsDiff, summary="Compare two fact register versions")
def diff_versions(
    store: Annotated[TwinStore | None, Depends(optional_twin_store)],
    from_version: Annotated[int, Query(ge=0, description="Version from (0: the repository file), count")] = 0,
    to_version: Annotated[int | None, Query(ge=0, description="Version to (default: the one in force), count")] = None,
) -> FactsDiff:
    """Values, bands and statuses that differ between two versions."""
    try:
        target = to_version
        if target is None:
            target = svc.register_view(store, None)["in_force_version"]
        return FactsDiff.model_validate(svc.diff(store, from_version, target))
    except ERRORS as exc:
        raise _problem(exc) from exc


@router.get(
    "/export",
    summary="Export a version as the register file",
    response_class=Response,
    responses={
        200: {
            "description": "The register file (facts_register.json) of the approved version, schema-valid",
            "content": {"application/json": {"schema": {"type": "object", "additionalProperties": True}}},
        }
    },
)
def export_version(
    store: Annotated[TwinStore, Depends(twin_store)],
    version: Annotated[
        int | None, Query(ge=1, description="Approved version to export (default: the one in force), count")
    ] = None,
) -> Response:
    """``src/config/facts_register.json`` with the version applied, for a repository PR: register version + 0.1, the
    changed facts stamped with it and the approval day (IST), a revision-log entry; in the file's format. 409 when the
    version is not approved (or none is in force)."""
    try:
        doc = svc.export(store, version)
    except ERRORS as exc:
        raise _problem(exc) from exc
    return Response(
        svc.export_text(doc),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="facts_register_v{doc["version"]}.json"'},
    )
