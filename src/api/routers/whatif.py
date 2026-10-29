"""What-if analysis of the admin space (SV-4.3, #306). Administrators only.

Each administrator saves FORGE cases (levers on the calibration point: coke chemistry, PCI, nut coke, ηCO, HM Si,
HMT, HBT, O₂, blast moisture, production, shaft efficiency) and STRATA cases (a forced ferrous blend or the cheapest
blend, and controls) in ``twin.whatif_case``; a save or delete writes its audit row (``whatif.save``,
``whatif.delete``). A run compares up to four of the user's cases on one basis, the calibration of a what-if day
(``calibration.nightly``'s, last 30 days) or the shipped Jul-Oct 2026 calibration: the jobs ``whatif.forge`` and
``whatif.strata`` solve them in the worker and store each result, with the fact-register version it used, with its
case. Coke kg/tHM dry, costs Rs/tHM, production t/h (FORGE) or t/d (STRATA), times UTC, days IST.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Path, Query, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, ValidationError

from api.auth import auth_state, client_address, require_role
from api.errors import ApiError
from api.live import job_store
from api.store import optional_twin_store, twin_store
from data.twin.repositories import Actor, NotFound, TwinStore
from services import whatif as svc
from services.auth import Principal
from services.jobs import WHATIF_MAX_CASES, ForgeLevers, JobRecord, JobStore
from utils.burden_engine.facts import load_register

router = APIRouter(prefix="/api/admin/whatif", tags=["admin"], dependencies=[Depends(require_role("admin"))])

Kind = Literal["forge", "strata"]
CaseId = Annotated[str, Path(description="Case id (UUID)")]


class MaterialOption(BaseModel):
    id: str = Field(..., description="Material id of the engine")
    name: str
    source: Literal["live", "reference"] = Field(
        ..., description="live: in the latest optimiser run's yard; reference: the yard of 16 Oct 2026"
    )


class WhatIfBasisView(BaseModel):
    days: list[date] = Field(..., description="What-if days with a stored day calibration (IST), newest first")
    basis: svc.WhatIfBasis | None = Field(None, description="The calibration asked for; null when it is not stored")
    base: svc.ForgeBase | None = Field(None, description="The calibration point the FORGE cases change")
    message: str | None = Field(None, description="Why there is no basis")
    facts_version: str = Field(..., description="Fact-register version in force (e.g. 1.8, 1.8 + overlay 2)")
    materials: list[MaterialOption] = Field(..., description="Ferrous materials a forced blend can name")
    max_cases: int = Field(..., description="Cases a run compares at most, count")


class WhatIfCaseSave(BaseModel):
    model_config = {"extra": "forbid"}

    kind: Kind
    name: str = Field(..., min_length=1, max_length=80, description="Name of the case")
    forge: ForgeLevers | None = Field(None, description="FORGE levers (kind forge)")
    strata: svc.StrataCaseInputs | None = Field(None, description="Forced blend and controls (kind strata)")


class WhatIfRunRequest(BaseModel):
    model_config = {"extra": "forbid"}

    kind: Kind
    case_ids: list[str] = Field(
        ..., min_length=1, max_length=WHATIF_MAX_CASES, description="The user's saved cases, in column order"
    )
    calibration: Literal["day", "reference"] = Field(
        "day", description="day = calibration.nightly's for the what-if day; reference = the shipped Jul-Oct 2026 one"
    )
    day: date | None = Field(None, description="What-if day (IST); default: the latest stored day calibration")
    omega_mode: Literal["strata", "fixed"] = Field("strata", description="STRATA cases: STRATA-ω or Fixed-ω")
    yard: Literal["live", "reference"] = Field("live", description="STRATA cases: live or reference yard")
    production_t_d: float | None = Field(
        None, ge=1500, le=3000, description="STRATA cases: hot-metal production, t/d (default: the regime median)"
    )


class WhatIfRunSubmitted(BaseModel):
    job: JobRecord
    duplicate: bool = Field(..., description="True when an identical run was already queued or running")


def _actor(request: Request) -> Actor:
    principal: Principal = request.state.principal
    return Actor(
        name=principal.username,
        role=principal.role,
        client_ip=client_address(request, auth_state(request).settings.trusted_proxy_set),
    )


def _validation(exc: ValidationError, *loc: str) -> RequestValidationError:
    errors = [{**e, "loc": (*loc, *e["loc"])} for e in exc.errors(include_url=False)]
    return RequestValidationError(jsonable_encoder(errors))


def _inputs(body: WhatIfCaseSave) -> dict:
    other = body.strata if body.kind == "forge" else body.forge
    if other is not None:
        raise ApiError(422, f"A {body.kind} case carries {body.kind} inputs only.")
    try:
        return svc.case_inputs(body.kind, forge=body.forge, strata=body.strata)
    except ValidationError as exc:
        raise _validation(exc, "body", body.kind) from exc


@router.get("/basis", response_model=WhatIfBasisView, summary="What-if days, the calibration point and materials")
def basis(
    store: Annotated[TwinStore | None, Depends(optional_twin_store)],
    calibration: Annotated[Literal["day", "reference"], Query(description="day or reference")] = "day",
    day: Annotated[date | None, Query(description="What-if day (IST); default: the latest stored")] = None,
) -> WhatIfBasisView:
    """The stored what-if days, the calibration asked for with its FORGE calibration point (two balances, a few ms),
    the register version in force and the ferrous materials of the live and the reference yard."""
    base, fields, message = None, None, None
    try:
        cal, fields = svc.resolve_calibration(calibration, day)
        base = svc.forge_compute(cal, [])["base"]
    except svc.WhatIfInputError as exc:
        message = str(exc)
    except Exception as exc:  # noqa: BLE001 - a calibration that does not solve: the page shows why
        fields, message = None, f"The calibration point could not be solved ({type(exc).__name__}: {exc})."[:400]
    return WhatIfBasisView(
        days=svc.stored_days(),
        basis=fields,
        base=base,
        message=message,
        facts_version=load_register().label,
        materials=svc.materials(store),
        max_cases=WHATIF_MAX_CASES,
    )


@router.get("/cases", response_model=list[svc.WhatIfCaseView], summary="The user's saved what-if cases")
def list_cases(
    request: Request,
    store: Annotated[TwinStore, Depends(twin_store)],
    kind: Annotated[Kind | None, Query(description="Only FORGE or only STRATA cases")] = None,
) -> list[svc.WhatIfCaseView]:
    """The signed-in administrator's cases, most recently changed first, each with its latest result."""
    principal: Principal = request.state.principal
    return [svc.WhatIfCaseView.model_validate(v) for v in svc.list_cases(store, principal.username, kind)]


@router.post("/cases", response_model=svc.WhatIfCaseView, status_code=201, summary="Save a new what-if case")
def create_case(
    body: WhatIfCaseSave, request: Request, store: Annotated[TwinStore, Depends(twin_store)]
) -> svc.WhatIfCaseView:
    try:
        view = svc.save_case(store, _actor(request), kind=body.kind, name=body.name, inputs=_inputs(body))
    except svc.WhatIfInputError as exc:
        raise ApiError(422, str(exc)) from exc
    return svc.WhatIfCaseView.model_validate(view)


@router.put("/cases/{case_id}", response_model=svc.WhatIfCaseView, summary="Change a saved what-if case")
def update_case(
    case_id: CaseId, body: WhatIfCaseSave, request: Request, store: Annotated[TwinStore, Depends(twin_store)]
) -> svc.WhatIfCaseView:
    """A change of the inputs clears the case's result; a new name keeps it."""
    try:
        view = svc.save_case(
            store, _actor(request), kind=body.kind, name=body.name, inputs=_inputs(body), case_id=case_id
        )
    except NotFound as exc:
        raise ApiError(404, "No such what-if case of yours.") from exc
    except svc.WhatIfInputError as exc:
        raise ApiError(422, str(exc)) from exc
    return svc.WhatIfCaseView.model_validate(view)


@router.delete("/cases/{case_id}", status_code=204, summary="Delete a saved what-if case")
def delete_case(case_id: CaseId, request: Request, store: Annotated[TwinStore, Depends(twin_store)]) -> Response:
    try:
        svc.delete_case(store, _actor(request), case_id)
    except NotFound as exc:
        raise ApiError(404, "No such what-if case of yours.") from exc
    return Response(status_code=204)


@router.post("/runs", response_model=WhatIfRunSubmitted, status_code=202, summary="Run saved what-if cases")
async def start_run(
    body: WhatIfRunRequest,
    request: Request,
    store: Annotated[TwinStore, Depends(twin_store)],
    jobs: Annotated[JobStore, Depends(job_store)],
) -> WhatIfRunSubmitted:
    """Queue ``whatif.forge`` or ``whatif.strata`` for up to four of the user's cases; follow it with
    ``GET /api/jobs/{id}`` or the ``job.progress`` / ``job.done`` events, then read the cases (their results)."""
    principal: Principal = request.state.principal
    basis = dict(calibration=body.calibration, day=body.day)
    if body.kind == "strata":
        basis |= dict(omega_mode=body.omega_mode, yard=body.yard, production_t_d=body.production_t_d)
    try:
        params = svc.run_params(store, principal.username, body.kind, body.case_ids, **basis)
    except NotFound as exc:
        raise ApiError(404, "A case of the run is not one of yours.") from exc
    except svc.WhatIfInputError as exc:
        raise ApiError(422, str(exc)) from exc
    except ValidationError as exc:
        raise _validation(exc, "body") from exc
    record, duplicate = await jobs.submit(svc.JOB_KINDS[body.kind], params.model_dump(mode="json"), principal.username)
    return WhatIfRunSubmitted(job=record, duplicate=duplicate)
