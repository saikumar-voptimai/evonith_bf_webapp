"""Validation and STRATA-Learn console (SV-4.5, #308). Administrators only.

``GET /api/admin/learn`` reads what the shadow run ``learn.shadow`` stored (``services.learn_console``; no filter is
run): the out-of-sample skill of the challenger and the champion, the per-family table with the VAL facts in force, the
Kalman trajectories of the plant-data scales, the champion / challenger gate (60 days, Diebold-Mariano t >= 2), the
CUSUM drift alarm and the register proposal with its decisions; next to it the out-of-sample results of the deployed
asset (2026 expanding window, time splits, strict 2023-24 test). Its numbers equal the run's
(``utils.burden_engine.learn.run.shadow_run``), unrounded.

``POST /api/admin/learn/decisions`` approves or rejects proposal items. An approval writes the items' value and status
through the fact-register editor (SV-4.2) as a pending version (``facts.propose``) with the decision's audit row
(``learn.approve``) in the same transaction; the version is reviewed in the editor. A rejection records
``learn.reject``. ``POST /api/admin/learn/run`` queues ``learn.shadow`` now (audited, ``job.start``). Days are IST,
times UTC; coke in kg/tHM dry; omega, scales and skills dimensionless.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from api.auth import request_actor, require_role
from api.errors import ApiError
from api.live import job_store
from api.routers.facts import FactsVersionView
from api.routers.strata import Validation
from api.store import optional_twin_store, twin_store
from data.twin.repositories import NotFound, TwinStore
from services import facts, job_audit
from services import learn_console as console
from services.jobs import JobRecord, JobStore

router = APIRouter(prefix="/api/admin/learn", tags=["admin"], dependencies=[Depends(require_role("admin"))])

ItemState = Literal["open", "proposed", "approved", "not_taken", "rejected"]


class LearnAsset(BaseModel):
    label: str = Field(..., description="Deployed STRATA asset, e.g. STRATA 1.5 (2026-10-25)")
    version: str = Field(..., description="The asset's full version text")
    deployed: date | None = Field(None, description="Deployment day: the gate counts the days after it")
    rho: float | None = Field(None, description="AR(1) coefficient of the day errors, dimensionless")
    sigma_e: float | None = Field(None, description="Innovation sd of the day's omega deviation, dimensionless")
    n_days: int | None = Field(None, description="Days the asset's posterior was fitted on, count")


class LearnSkill(BaseModel):
    predictor: Literal["challenger", "champion"]
    label: str = Field(..., description="The run's label of the predictor")
    omega_day: float | None = Field(None, description="Skill of the day's omega against Fixed-omega, dimensionless")
    coke_day: float | None = Field(None, description="Skill of the day's burnt coke against Fixed-omega, dimensionless")


class FamilyFact(BaseModel):
    id: str = Field(..., description="VAL fact id")
    status: str = Field(..., description="Status in force (Supported, Neutral, Disputed, Modelled only)")
    value_kg_thm: float | None = Field(None, description="Value in force: STRATA burnt coke (mean), kg/tHM dry")
    se_kg_thm: float | None = Field(None, description="Standard error in force, kg/tHM")
    n: int | None = Field(None, description="Days of the fact, count")
    skill: float | None = Field(None, description="Skill of the fact, dimensionless")
    bias_kg_thm: float | None = Field(None, description="Bias of the fact, kg/tHM")
    se_bias_kg_thm: float | None = Field(None, description="Standard error of the bias, kg/tHM")


class BlendFamily(BaseModel):
    family: str = Field(..., description="Blend family (or a cross-family group)")
    n: int = Field(..., description="Evidence days since the evaluation start, count")
    realised_kg_thm: float | None = Field(None, description="Burnt coke (oxygen balance), mean, kg/tHM dry")
    fixed_omega_kg_thm: float | None = Field(None, description="Fixed-omega prediction, mean, kg/tHM dry")
    strata_kg_thm: float | None = Field(None, description="STRATA one-step prediction (challenger), mean, kg/tHM dry")
    rmse_fixed_kg_thm: float | None = Field(None, description="RMSE of Fixed-omega, kg/tHM")
    rmse_strata_kg_thm: float | None = Field(None, description="RMSE of STRATA, kg/tHM")
    skill: float | None = Field(None, description="1 - MSE(STRATA) / MSE(Fixed-omega), dimensionless")
    bias_kg_thm: float | None = Field(None, description="Mean of realised - STRATA, kg/tHM")
    se_bias_kg_thm: float | None = Field(None, description="Standard error of the bias (AR-adjusted n), kg/tHM")
    verdict: str = Field(..., description="Verdict by the rule: Supported, Neutral, Disputed or Modelled only")
    fact: FamilyFact | None = Field(None, description="The family's VAL fact in force; null: no fact")


class ScaleTrajectory(BaseModel):
    name: str = Field(..., description="Parameter, e.g. lam_pellet")
    label: str
    unit: str = Field(..., description="Unit (empty: dimensionless scale on the model channel)")
    fixed: bool = Field(..., description="Held at 0 by the asset (no random walk)")
    champion: float | None = Field(None, description="Deployed asset's value, unit of the parameter")
    latest: float | None = Field(None, description="Kalman mean after the last evidence day, unit of the parameter")
    latest_sd: float | None = Field(None, description="Kalman sd after the last evidence day, unit of the parameter")
    mean: list[float | None] = Field(..., description="Kalman mean after each day of dates, unit of the parameter")
    sd: list[float | None] = Field(..., description="Kalman sd after each day of dates, unit of the parameter")


class LearnTrajectory(BaseModel):
    dates: list[str] = Field(..., description="Evidence days (IST, YYYY-MM-DD), oldest first")
    n: int | None = Field(None, description="Days the filter has seen, count")
    last_date: date | None = Field(None, description="Last evidence day")
    params: list[ScaleTrajectory]
    observed: list[float | None] = Field(..., description="Day's omega minus its 14-day anchor, dimensionless")
    challenger: list[float | None] = Field(..., description="One-step prediction of the challenger, dimensionless")
    challenger_sd: list[float | None] = Field(..., description="1-sigma of its parameter part, dimensionless")
    champion: list[float | None] = Field(..., description="Prediction of the champion (deployed asset), dimensionless")


class LearnGate(BaseModel):
    status: str = Field(..., description="collecting (fewer than window days after deployment) or evaluated")
    n: int = Field(..., description="Days after deployment, count")
    window: int = Field(..., description="Days the gate compares, count")
    min_gain: float = Field(..., description="MSE gain the challenger needs, fraction")
    t_min: float = Field(..., description="Diebold-Mariano t the challenger needs, dimensionless")
    lags: int = Field(..., description="Newey-West lags of the Diebold-Mariano variance, days")
    mse_champion: float | None = Field(None, description="MSE of the champion's omega errors, dimensionless")
    mse_challenger: float | None = Field(None, description="MSE of the challenger's omega errors, dimensionless")
    gain: float | None = Field(None, description="1 - MSE(challenger) / MSE(champion), fraction")
    dm_t: float | None = Field(None, description="Diebold-Mariano t (positive: challenger better), dimensionless")
    promote: bool = Field(..., description="Promotion proposed (never applied by the run)")
    reason: str
    deployed: date | None = Field(None, description="Deployment day of the champion")


class LearnCusum(BaseModel):
    alarm: bool = Field(..., description="S+ or S- above h at the last day")
    ever_alarm: bool = Field(..., description="S+ or S- exceeded h on some day")
    s_plus: float | None = Field(None, description="Upper CUSUM, standardised innovations")
    s_minus: float | None = Field(None, description="Lower CUSUM, standardised innovations")
    k: float = Field(..., description="Reference value, standardised innovations")
    h: float = Field(..., description="Alarm threshold, standardised innovations")
    first_alarm: int | None = Field(None, description="Index of the first excursion, days")
    first_alarm_date: date | None = Field(None, description="Day of the first excursion")
    basis: str = Field(..., description="Errors the CUSUM runs on")
    rho: float | None = Field(None, description="AR(1) coefficient of the errors, dimensionless")
    sigma_e: float | None = Field(None, description="Innovation sd, dimensionless")


class RegisterChange(BaseModel):
    field: Literal["value", "status"]
    in_force: float | str | None = Field(..., description="Value in force or status, unit of the fact's value")
    proposed: float | str = Field(..., description="Proposed value or status, unit of the fact's value")


class ItemDecision(BaseModel):
    decision: Literal["approved", "rejected"]
    by: str = Field(..., description="Administrator")
    at: datetime = Field(..., description="Time of the decision (UTC)")
    note: str | None = None
    facts_version: int | None = Field(None, description="Facts version the approval proposed, count")
    version_status: str | None = Field(None, description="Status of that version: pending, approved or rejected")
    audit_id: int = Field(..., description="Audit-log entry of the decision")


class ProposalItem(BaseModel):
    key: str = Field(..., description="Item key: the fact id, new:<family> for a family without a VAL fact")
    fact_id: str | None = Field(None, description="Fact of the register the item changes")
    kind: str = Field(..., description="value, status, support, new, promotion or drift")
    automatic: bool = Field(..., description="Within 2 se and no status change (the run's rule)")
    reason: str
    statement: str = Field(..., description="The fact's statement, or the family")
    unit: str = Field(..., description="Unit of the fact's value")
    change: dict[str, Any] = Field(..., description="The run's proposed change (value, se, params, status)")
    register_changes: list[RegisterChange] = Field(..., description="What an approval writes through the overlay")
    not_carried: list[str] = Field(..., description="Proposed fields an overlay does not carry (se, params)")
    approvable: bool = Field(..., description="An approval changes the register")
    why: str | None = Field(None, description="Why the item cannot be approved through the register")
    state: ItemState = Field(
        ...,
        description="open; proposed (facts version pending); approved (version approved); not_taken (the editor "
        "rejected the version: open again); rejected (here)",
    )
    decidable: bool = Field(..., description="Open or not taken: it can be approved or rejected")
    decision: ItemDecision | None = None


class LearnProposal(BaseModel):
    day: date | None = Field(None, description="IST day of the run that made it")
    hash: str = Field(..., description="Hash of the items: names the proposal in a decision")
    object_id: str = Field(..., description="Audit-log object id of its decisions, <date>/<hash>")
    n_auto: int = Field(..., description="Automatic items (the run's rule), count")
    n_approval: int = Field(..., description="Items for approval (the run's rule), count")
    items: list[ProposalItem]


class LearnRunInfo(BaseModel):
    at: datetime = Field(..., description="When the run finished (UTC)")
    day: date | None = Field(None, description="IST day of the run")
    n_days: int = Field(..., description="Evidence days with all regressors, count")
    evidence_days: int | None = Field(None, description="Rows of the evidence store, count")
    evidence_to: date | None = Field(None, description="Last evidence day")
    appended_days: int | None = Field(None, description="Days the run appended, count")


class LearnRunDay(BaseModel):
    day: date | None = Field(None, description="IST day of the run")
    at: datetime | None = Field(None, description="When it finished (UTC)")
    evidence_days: int | None = Field(None, description="Evidence days, count")
    evidence_to: date | None = Field(None, description="Last evidence day")
    appended_days: int | None = Field(None, description="Days appended, count")
    gate_status: str
    promote: bool
    drift_alarm: bool
    automatic: int | None = Field(None, description="Automatic items, count")
    approval: int | None = Field(None, description="Items for approval, count")
    coke_skill_challenger: float | None = Field(None, description="Coke skill of the challenger, dimensionless")
    coke_skill_champion: float | None = Field(None, description="Coke skill of the champion, dimensionless")


class LearnJob(BaseModel):
    kind: str
    title: str
    role: str = Field(..., description="Lowest role that may start it")
    schedule: str | None = Field(None, description="Scheduled times (IST)")
    next_slot: datetime | None = Field(None, description="Next scheduled run (UTC)")
    timeout_s: float = Field(..., description="Time limit, s")


class RegisterInForce(BaseModel):
    in_force_version: int = Field(..., description="Approved facts version in force (0: the repository file), count")
    label: str = Field(..., description="Register in force, e.g. 1.8 + overlay 3")
    reason: str | None = Field(None, description="Why the repository file is shown")


class LearnConsole(BaseModel):
    today: date = Field(..., description="Today (IST)")
    asset: LearnAsset
    validation: Validation = Field(..., description="Out-of-sample results of the deployed asset (as on /strata)")
    skill_from: date | None = Field(None, description="First day of the run's skill and family table")
    facts_register: RegisterInForce = Field(..., description="Fact register in force")
    run: LearnRunInfo | None = Field(None, description="The stored run; null: none yet")
    skill: list[LearnSkill]
    families: list[BlendFamily]
    trajectory: LearnTrajectory | None = None
    gate: LearnGate | None = None
    cusum: LearnCusum | None = None
    proposal: LearnProposal | None = None
    decisions_reason: str | None = Field(None, description="Why decisions are not shown")
    runs: list[LearnRunDay] = Field(..., description="Stored runs, newest first (at most 30)")
    job: LearnJob


class LearnDecisionRequest(BaseModel):
    model_config = {"extra": "forbid"}

    proposal: str = Field(..., min_length=1, max_length=64, description="Hash of the proposal (GET proposal.hash)")
    items: list[str] = Field(..., min_length=1, max_length=100, description="Item keys to decide")
    approve: bool = Field(..., description="true proposes the items through the register editor, false rejects them")
    note: str | None = Field(None, max_length=1000, description="Reason, recorded in the audit log")


class LearnDecisionResult(BaseModel):
    decision: Literal["approved", "rejected"]
    proposal_date: date | None = None
    proposal_hash: str
    items: list[str]
    facts_version: FactsVersionView | None = Field(None, description="The pending facts version an approval made")
    audit_id: int = Field(..., description="Audit-log entry of the decision")


class LearnRunStarted(BaseModel):
    job: JobRecord
    duplicate: bool = Field(..., description="True when a shadow run was already queued or running")
    audit_id: int = Field(..., description="Audit-log entry of the start")


def _sentence(exc: Exception) -> str:
    message = str(exc).strip()
    message = message[:1].upper() + message[1:]
    return message if message.endswith(".") else message + "."


@router.get("", response_model=LearnConsole, summary="Validation and STRATA-Learn console")
async def learn_view(store: Annotated[TwinStore | None, Depends(optional_twin_store)]) -> LearnConsole:
    """The stored shadow run with its decisions, the asset's out-of-sample results, the runs and the job. Without a
    stored run the asset's results only (``run`` null)."""
    return LearnConsole.model_validate(await asyncio.to_thread(console.view, store))


@router.post("/decisions", response_model=LearnDecisionResult, summary="Approve or reject proposal items")
def decide_proposal_items(
    body: LearnDecisionRequest, request: Request, store: Annotated[TwinStore, Depends(twin_store)]
) -> LearnDecisionResult:
    """Approve: the items' value and status become the next pending facts version (``facts.propose``) with the
    decision's audit row (``learn.approve``), for review in the fact-register editor. Reject: ``learn.reject``.
    404 without a stored run; 409 when a new run replaced the proposal, an item is decided or another facts version
    is in force; 422 for an unknown item or one an approval cannot write."""
    note = (body.note or "").strip() or None
    try:
        out = console.decide(
            store, request_actor(request), proposal=body.proposal, keys=body.items, approve=body.approve, note=note
        )
    except console.NoRun as exc:
        raise ApiError(404, "No STRATA-Learn run is stored: run the shadow run first.", title="No run") from exc
    except (console.ProposalChanged, console.AlreadyDecided, facts.FactsConflict) as exc:
        raise ApiError(409, _sentence(exc)) from exc
    except console.DecisionRefused as exc:
        raise ApiError(422, _sentence(exc)) from exc
    except facts.InvalidOverlay as exc:
        raise ApiError(422, "The register would not be usable: " + "; ".join(exc.problems) + ".") from exc
    except NotFound as exc:
        raise ApiError(404, f"No {exc}.") from exc
    version = out.pop("facts_version")
    return LearnDecisionResult(
        **out, facts_version=FactsVersionView.model_validate(version) if version is not None else None
    )


@router.post("/run", response_model=LearnRunStarted, status_code=202, summary="Run the shadow run now")
async def start_shadow_run(
    request: Request,
    store: Annotated[TwinStore, Depends(twin_store)],
    jobs: Annotated[JobStore, Depends(job_store)],
) -> LearnRunStarted:
    """Queue ``learn.shadow`` now and record the start in the audit log (``job.start``); a run already queued or
    running is returned instead (``duplicate``). The run replaces the stored outputs: a proposal with other items
    is a new proposal."""
    started = await job_audit.submit(jobs, store, request_actor(request), console.KIND, {}, origin="validation console")
    return LearnRunStarted(job=started.record, duplicate=started.duplicate, audit_id=started.audit_id)
