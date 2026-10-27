"""Operator inputs of the optimisers: the version in force or the one valid at a date, and saving a new version
(SV-3.1, #298).

The inputs (ferrous availability, prices and share bounds; flux prices, stock and bounds; slag targets and charging
limits; fuel prices and ash chemistry; held hot-metal chemistry; dust) are versioned in ``twin.operator_inputs``
(``services.operator_inputs``), the source page 9, page 11 and the worker read. Without a version (empty or unreachable
store) GET answers from ``bmo_operator_inputs.yml`` with ``source`` yaml and the ``reason``; PUT needs the store and
writes one version with its audit row (``operator_inputs.update``) in the same transaction. Responses leave out what is
not set (null), so a document reads as it is stored. Prices Rs/t, times UTC.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field

from api.auth import auth_state, client_address, require_role
from api.errors import ApiError
from api.store import optional_twin_store, twin_store
from data.twin.repositories import Actor, TwinStore, VersionConflict
from services import operator_inputs as svc
from services.auth import Principal
from services.operator_inputs import OperatorInputsDocument

router = APIRouter(prefix="/api/operator-inputs", tags=["operator_inputs"])


class OperatorInputsView(BaseModel):
    inputs: OperatorInputsDocument = Field(..., description="The operator inputs (units in the field descriptions)")
    source: Literal["neon", "yaml"] = Field(
        ..., description="neon: a version of twin.operator_inputs; yaml: bmo_operator_inputs.yml (no version in force)"
    )
    version: int | None = Field(None, description="Number of the version (1, 2, ...; null for the YAML file), count")
    valid_from: datetime | None = Field(None, description="Start of the validity (UTC)")
    valid_upto: datetime | None = Field(None, description="End of the validity (UTC); null for the current version")
    author: str | None = Field(None, description="User who saved the version (seed: the YAML recorded as version 1)")
    origin: Literal["seed", "streamlit", "api"] | None = Field(None, description="Where the version was saved")
    note: str | None = Field(None, description="Note saved with the version")
    reason: str | None = Field(None, description="Why the YAML file is in force (source yaml)")
    at: datetime | None = Field(None, description="The instant the version was looked up for (UTC; null: now)")


class OperatorInputsSaved(OperatorInputsView):
    changed: bool = Field(..., description="False when the request left the inputs as they were (no new version)")


class OperatorInputsUpdate(BaseModel):
    model_config = {"extra": "forbid"}

    inputs: OperatorInputsDocument = Field(
        ...,
        description="Sections to save: a section given replaces the saved one, a section set to null is removed "
        "(the configured defaults apply), a section left out is kept",
    )
    base_version: int | None = Field(
        None,
        ge=0,
        description="The version the edit started from (GET's version; 0 when GET answered from the YAML file); "
        "another current version answers 409, count",
    )
    note: str | None = Field(None, max_length=500, description="Reason for the change, saved with the version")


def _view(current: svc.OperatorInputs, at: datetime | None = None) -> dict[str, Any]:
    return dict(
        inputs=svc.normalise(current.inputs),
        source=current.source,
        version=current.version,
        valid_from=current.valid_from,
        valid_upto=current.valid_upto,
        author=current.author,
        origin=current.origin,
        note=current.note,
        reason=current.reason,
        at=at,
    )


def _parse_at(value: str) -> date | datetime:
    text = value.strip()
    try:
        return date.fromisoformat(text) if len(text) == 10 else datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ApiError(422, f"at must be a date (YYYY-MM-DD, IST) or an ISO 8601 time, got {value!r}.") from exc


@router.get(
    "",
    response_model=OperatorInputsView,
    response_model_exclude_none=True,
    summary="Operator inputs in force or at a date",
    dependencies=[Depends(require_role("user"))],
)
def get_operator_inputs(
    store: Annotated[TwinStore | None, Depends(optional_twin_store)],
    at: Annotated[
        str | None,
        Query(
            description="A date (YYYY-MM-DD: the version in force at the end of that IST day) or an ISO 8601 time "
            "(without offset: IST); omitted: the version in force now"
        ),
    ] = None,
) -> OperatorInputsView:
    """The version in force (the YAML file while the store holds none or cannot be reached), or the version valid at
    ``at`` (404 before the first version, 503 without the store)."""
    if at is None:
        return OperatorInputsView.model_validate(_view(svc.load_current(store)))
    instant = svc.as_of(_parse_at(at))
    try:
        found = svc.load_at(instant, store)
    except svc.Unavailable as exc:
        raise ApiError(503, str(exc)) from exc
    if found is None:
        raise ApiError(404, f"No version of the operator inputs was in force at {instant.isoformat()}.")
    return OperatorInputsView.model_validate(_view(found, instant))


@router.put(
    "",
    response_model=OperatorInputsSaved,
    response_model_exclude_none=True,
    summary="Save a new version of the operator inputs",
    dependencies=[Depends(require_role("supervisor"))],
)
def put_operator_inputs(
    body: OperatorInputsUpdate, request: Request, store: Annotated[TwinStore, Depends(twin_store)]
) -> OperatorInputsSaved:
    """Apply the sections to the version in force and save the result as the next version, with its audit row
    (``operator_inputs.update``, the changed values in the details). The first save records the YAML file as version 1.
    A request that changes nothing writes no version (``changed`` false). 409 when ``base_version`` is not the current
    version, 422 when a section has values the optimisers cannot use."""
    sections = {name: getattr(body.inputs, name) for name in body.inputs.model_fields_set}
    sections = {k: (v.model_dump(exclude_unset=True, exclude_none=True) if v is not None else None)
                for k, v in sections.items()}  # fmt: skip
    try:
        svc.check(sections)
    except svc.InvalidInputs as exc:
        raise ApiError(422, "The operator inputs cannot be used: " + "; ".join(exc.problems) + ".") from exc
    principal: Principal = request.state.principal
    actor = Actor(
        name=principal.username,
        role=principal.role,
        client_ip=client_address(request, auth_state(request).settings.trusted_proxy_set),
    )
    note = (body.note or "").strip() or None
    try:
        written = svc.write(
            svc.replace_sections(sections),
            actor=actor,
            origin="api",
            note=note,
            base_version=body.base_version,
            store=store,
        )
    except VersionConflict as exc:
        message = str(exc)
        raise ApiError(409, message[:1].upper() + message[1:] + ".") from exc
    except svc.Unavailable as exc:
        raise ApiError(503, str(exc)) from exc
    return OperatorInputsSaved.model_validate({**_view(written.current), "changed": written.changed})
