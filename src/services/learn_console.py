"""Validation and STRATA-Learn console of the admin space (SV-4.5, #308).

The view reads what the shadow run ``learn.shadow`` (``services.learn``, 03:00 IST) wrote under
``src/storage/burden_engine/learn``: ``strata_learn_state.json`` (the result of
``utils.burden_engine.learn.run.shadow_run`` without its history), ``strata_learn_history.csv`` (the Kalman filter's
record of each evidence day, full precision) and ``runs/<IST date>.json``. It runs no filter: its numbers are the
run's, unrounded. The out-of-sample results of the deployed asset (2026 expanding window, time splits, strict 2023-24
test) come from ``assets/burden_engine/omega_model.json -> validation``; the VAL facts from the register in force
(``services.facts.register_view``).

Register proposals: an administrator approves or rejects items of the stored proposal, identified by its date and the
hash of its items (``<date>/<hash>``, a new run with other items is a new proposal). An approval writes through the
fact-register editor (SV-4.2): ``services.facts.propose`` saves the items' value and status (the fields an overlay
carries; se and params go with a release of the register file) as the next pending version, with its ``facts.propose``
audit row and, in the same transaction, the decision's row (``learn.approve``, object ``learn_proposal`` /
``<date>/<hash>``). The version is then reviewed in the editor (a status change by a second administrator). A rejection
records ``learn.reject`` on the same object. An item whose version the editor rejected can be decided again. Days are
IST, times UTC. No Streamlit (tests/test_services_architecture.py).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from data.twin.models import AuditLog, FactsVersion
from data.twin.repositories import Actor, TwinStore, audit_row
from services import audit, facts, learn, schedule, strata
from services.jobs import CATALOG
from services.twin_views import utc
from utils.burden_engine.learn import gate as learn_gate
from utils.burden_engine.learn.evidence import PARAMS, REGRESSORS
from utils.burden_engine.learn.run import EVAL_FROM
from utils.burden_engine.omega_model import MODEL_PATH

KIND = "learn.shadow"
OBJECT_TYPE = "learn_proposal"
APPROVE = "learn.approve"
REJECT = "learn.reject"
STATE_FILE = "strata_learn_state.json"
HISTORY_FILE = "strata_learn_history.csv"
RUNS_LISTED = 30
IST = ZoneInfo("Asia/Kolkata")
#: Fields of a proposal item an overlay of the register carries (``services.facts.EDIT_FIELDS`` without band).
CARRIED = ("value", "status")
#: Kalman parameters: label and unit (empty: a dimensionless scale on the model channel).
PARAM_INFO: dict[str, tuple[str, str]] = {
    "lam_pellet": ("Pellet scale", ""),
    "lam_lump": ("Lump-ore scale", ""),
    "lam_production": ("Production scale", ""),
    "lam_other": ("Other-operation scale", ""),
    "a_moisture": ("Blast-moisture term", "omega per g/Nm3"),
    "a_nut": ("Nut-coke term", "omega per kg/tHM"),
}
#: Item states: open (no decision), proposed (its facts version is pending), approved (the version was approved),
#: not_taken (the editor rejected the version: open again), rejected (rejected here).
DECIDABLE = ("open", "not_taken")


class NoRun(LookupError):
    """No shadow run is stored."""


class ProposalChanged(RuntimeError):
    """The decision names another proposal than the stored one."""


class AlreadyDecided(RuntimeError):
    """An item has a decision that stands."""


class DecisionRefused(ValueError):
    """An unknown item, or an approval of an item that changes nothing an overlay carries."""


# ------------------------------------------------------------------------------------------------------ values


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _int(value: Any) -> int | None:
    number = _num(value)
    return int(number) if number is not None else None


def _day(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10]) if value else None
    except ValueError:
        return None


def _plain(value: Any) -> Any:
    """JSON data with NaN and infinities as None."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    return value


def _series(values: Iterable[Any]) -> list[float | None]:
    return [_num(v) for v in values]


# ------------------------------------------------------------------------------------------------------ files


def store_dir(directory: Path | str | None = None) -> Path:
    return Path(directory) if directory is not None else learn.STORE_DIR


def read_state(directory: Path | str | None = None) -> dict[str, Any] | None:
    """The stored run (``strata_learn_state.json``), or None."""
    try:
        doc = json.loads((store_dir(directory) / STATE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def read_history(directory: Path | str | None = None) -> pd.DataFrame | None:
    """The Kalman filter's day records of the stored run, or None."""
    try:
        return pd.read_csv(store_dir(directory) / HISTORY_FILE)
    except (OSError, ValueError):
        return None


def read_runs(directory: Path | str | None = None, limit: int = RUNS_LISTED) -> list[dict[str, Any]]:
    """Summaries of the shadow runs (``runs/<IST date>.json``), newest day first."""
    folder = store_dir(directory) / "runs"
    out = []
    for path in sorted(folder.glob("*.json"), reverse=True)[:limit] if folder.is_dir() else []:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(doc, dict):
            out.append(run_row(path.stem, doc))
    return out


def _skill_of(skill: Mapping[str, Any], prefix: str) -> Mapping[str, Any]:
    return next((v for k, v in skill.items() if str(k).startswith(prefix)), {}) or {}


def run_row(day: str, doc: Mapping[str, Any]) -> dict[str, Any]:
    gate, drift, prop = doc.get("gate") or {}, doc.get("drift") or {}, doc.get("proposal") or {}
    skill = doc.get("skill") or {}
    return dict(
        day=_day(day),
        at=utc(doc["at"]) if doc.get("at") else None,
        evidence_days=_int(doc.get("evidence_days")),
        evidence_to=_day(doc.get("evidence_to")),
        appended_days=_int(doc.get("appended_days")),
        gate_status=str(gate.get("status") or "unknown"),
        promote=bool(gate.get("promote")),
        drift_alarm=bool(drift.get("alarm")),
        automatic=_int(prop.get("automatic")),
        approval=_int(prop.get("approval")),
        coke_skill_challenger=_num(_skill_of(skill, "challenger").get("coke_day")),
        coke_skill_champion=_num(_skill_of(skill, "champion").get("coke_day")),
    )


def asset_spec() -> dict[str, Any]:
    return json.loads(Path(MODEL_PATH).read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------------------ parts


def asset_info(spec: Mapping[str, Any], state: Mapping[str, Any] | None) -> dict[str, Any]:
    version = str(spec.get("version", ""))
    cs = spec.get("channel_scales") or {}
    m = re.search(r"\((\d{4}-\d{2}-\d{2})\)", version)
    deployed = (state or {}).get("deployed") or (m.group(1) if m else None)
    return dict(
        label=version.split(":", 1)[0].strip(),
        version=version,
        deployed=_day(deployed),
        rho=_num(cs.get("rho")),
        sigma_e=_num(cs.get("sigma_e")),
        n_days=_int(cs.get("n_days")),
    )


def skill_rows(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Out-of-sample skill of the run since ``EVAL_FROM`` against Fixed-omega, challenger first."""
    rows = []
    for label, d in (state.get("skill") or {}).items():
        rows.append(
            dict(
                predictor="challenger" if str(label).startswith("challenger") else "champion",
                label=str(label),
                omega_day=_num(d.get("omega_day")),
                coke_day=_num(d.get("coke_day")),
            )
        )
    return rows


def val_fact_of(family: str, register_facts: Iterable[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """The VAL fact of a blend family (the rule of ``learn.autofill.propose``)."""
    return next(
        (f for f in register_facts if f.get("group") == "VAL" and f", {family}:" in str(f.get("statement", ""))),
        None,
    )


def family_rows(state: Mapping[str, Any], register_facts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The run's per-family table (one-step challenger) with the VAL fact in force."""
    rows = []
    for r in state.get("families") or []:
        fact = val_fact_of(str(r.get("family")), register_facts)
        params = (fact or {}).get("params") or {}
        rows.append(
            dict(
                family=str(r.get("family")),
                n=int(r.get("n") or 0),
                realised_kg_thm=_num(r.get("realised")),
                fixed_omega_kg_thm=_num(r.get("fixed_omega")),
                strata_kg_thm=_num(r.get("strata")),
                rmse_fixed_kg_thm=_num(r.get("rmse_fixed")),
                rmse_strata_kg_thm=_num(r.get("rmse_strata")),
                skill=_num(r.get("skill")),
                bias_kg_thm=_num(r.get("bias")),
                se_bias_kg_thm=_num(r.get("se_bias")),
                verdict=str(r.get("verdict")),
                fact=None
                if fact is None
                else dict(
                    id=fact["id"],
                    status=fact["status"],
                    value_kg_thm=_num(fact.get("value")),
                    se_kg_thm=_num(fact.get("se")),
                    n=_int(params.get("n")),
                    skill=_num(params.get("skill")),
                    bias_kg_thm=_num(params.get("bias")),
                    se_bias_kg_thm=_num(params.get("se_bias")),
                ),
            )
        )
    return rows


def fixed_params(spec: Mapping[str, Any]) -> set[str]:
    """Kalman parameters the asset holds fixed (``channel_scales.fixed``, by regressor name)."""
    fixed = (spec.get("channel_scales") or {}).get("fixed") or {}
    return {PARAMS[REGRESSORS.index(name)] for name in fixed if name in REGRESSORS}


def trajectory(history: pd.DataFrame | None, state: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, Any]:
    """Kalman state after each evidence day (mean and sd of each parameter) and the one-step predictions."""
    kal = state.get("state") or {}
    champion = state.get("champion") or {}
    held = fixed_params(spec)
    h = history if history is not None else pd.DataFrame()

    def col(name: str) -> list[float | None]:
        return _series(h[name]) if name in h else []

    params = [
        dict(
            name=p,
            label=PARAM_INFO[p][0],
            unit=PARAM_INFO[p][1],
            fixed=p in held,
            champion=_num(champion.get(p)),
            latest=_num((kal.get("mean") or {}).get(p)),
            latest_sd=_num((kal.get("sd") or {}).get(p)),
            mean=col(p),
            sd=col(f"{p}_sd"),
        )
        for p in PARAMS
    ]
    return dict(
        dates=[str(d)[:10] for d in h["date"]] if "date" in h else [],
        n=_int(kal.get("n")),
        last_date=_day(kal.get("last_date")),
        params=params,
        observed=col("y"),
        challenger=col("pred"),
        challenger_sd=col("pred_sd"),
        champion=col("pred_champion"),
    )


def gate_view(state: Mapping[str, Any], deployed: date | None) -> dict[str, Any]:
    g = state.get("gate") or {}
    return dict(
        status=str(g.get("status") or "unknown"),
        n=_int(g.get("n")) or 0,
        window=_int(g.get("window")) or learn_gate.WINDOW,
        min_gain=learn_gate.MIN_GAIN,
        t_min=learn_gate.T_MIN,
        lags=learn_gate.LAGS,
        mse_champion=_num(g.get("mse_champion")),
        mse_challenger=_num(g.get("mse_challenger")),
        gain=_num(g.get("gain")),
        dm_t=_num(g.get("dm_t")),
        promote=bool(g.get("promote")),
        reason=str(g.get("reason") or ""),
        deployed=deployed,
    )


def cusum_view(state: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, Any]:
    d = state.get("drift") or {}
    cs = spec.get("channel_scales") or {}
    return dict(
        alarm=bool(d.get("alarm")),
        ever_alarm=bool(d.get("ever_alarm")),
        s_plus=_num(d.get("s_plus")),
        s_minus=_num(d.get("s_minus")),
        k=learn_gate.CUSUM_K,
        h=learn_gate.CUSUM_H,
        first_alarm=_int(d.get("first_alarm")),
        first_alarm_date=_day(d.get("first_alarm_date")),
        basis=str(d.get("basis") or ""),
        rho=_num(cs.get("rho")),
        sigma_e=_num(cs.get("sigma_e")),
    )


# ------------------------------------------------------------------------------------------------------ proposal


def proposal_hash(proposal: Mapping[str, Any]) -> str:
    """First 16 hex digits of the SHA-256 of the proposal's items (canonical JSON)."""
    canonical = json.dumps(proposal.get("items") or [], sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def proposal_object(proposal: Mapping[str, Any]) -> str:
    """Audit-log object id of a proposal: ``<date>/<hash>``."""
    return f"{proposal.get('date') or 'undated'}/{proposal_hash(proposal)}"


def item_key(item: Mapping[str, Any]) -> str:
    """Key of an item: its fact id, ``new:<family>`` for a family without a VAL fact."""
    if item.get("id"):
        return str(item["id"])
    return f"new:{(item.get('change') or {}).get('family', '?')}"


def decision_rows(rows: Iterable[AuditLog]) -> dict[str, dict[str, Any]]:
    """The newest decision per item key, from the proposal's audit rows (newest first)."""
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.action not in (APPROVE, REJECT):
            continue
        details = row.details or {}
        for key in details.get("items") or []:
            out.setdefault(
                str(key),
                dict(
                    decision="approved" if row.action == APPROVE else "rejected",
                    by=row.actor,
                    at=utc(row.at),
                    note=details.get("note"),
                    facts_version=_int(details.get("facts_version")),
                    version_status=None,
                    audit_id=int(row.id),
                ),
            )
    return out


def item_state(decision: Mapping[str, Any] | None) -> str:
    if decision is None:
        return "open"
    if decision["decision"] == "rejected":
        return "rejected"
    return {"pending": "proposed", "approved": "approved", "rejected": "not_taken"}.get(
        decision.get("version_status") or "", "proposed"
    )


def decisions(store: TwinStore, obj: str) -> dict[str, dict[str, Any]]:
    """Decisions on the proposal ``obj`` with the status of each approval's facts version."""
    out = decision_rows(store.audit.query(object_type=OBJECT_TYPE, object_id=obj, limit=1000))
    for d in out.values():
        if d["facts_version"]:
            row = store.facts.get(d["facts_version"])
            d["version_status"] = row.status if row is not None else None
    return out


def _decisions_in(session: Session, obj: str) -> dict[str, dict[str, Any]]:
    query = (
        select(AuditLog)
        .where(AuditLog.object_type == OBJECT_TYPE, AuditLog.object_id == obj)
        .order_by(AuditLog.at.desc(), AuditLog.id.desc())
    )
    out = decision_rows(session.scalars(query))
    for d in out.values():
        if d["facts_version"]:
            d["version_status"] = session.scalar(
                select(FactsVersion.status).where(FactsVersion.version == d["facts_version"])
            )
    return out


def _require_open(found: Mapping[str, Mapping[str, Any]], keys: Sequence[str]) -> dict[str, str]:
    """States of ``keys``; ``AlreadyDecided`` when one has a decision that stands."""
    states = {k: item_state(found.get(k)) for k in keys}
    taken = [k for k, s in states.items() if s not in DECIDABLE]
    if taken:
        parts = []
        for k in taken:
            d = found[k]
            where = (
                f", facts version {d['facts_version']} {d['version_status'] or ''}".rstrip()
                if d["facts_version"]
                else ""
            )
            parts.append(f"{k} was {d['decision']} by {d['by']}{where}")
        raise AlreadyDecided("; ".join(parts))
    return states


def item_view(
    item: Mapping[str, Any], fact: Mapping[str, Any] | None, decision: Mapping[str, Any] | None
) -> dict[str, Any]:
    """A proposal item with what an approval would write through the overlay (``register_changes``), why it cannot
    (``why``) and its decision."""
    change = item.get("change") or {}
    fid = item.get("id")
    kind = str(item.get("kind") or "")
    changes: list[dict[str, Any]] = []
    not_carried: list[str] = []
    why = None
    if fid == "ASSET":
        why = "Promotion of the challenger replaces the asset's plant-data scales: a release of the asset."
    elif kind == "drift":
        why = "A drift alarm asks for a recalibration of the asset; it changes no fact."
    elif fid is None:
        why = f"No VAL fact for {change.get('family', 'this family')}: a new fact comes with a release of the register."
    elif fact is None:
        why = f"{fid} is not in the register."
    elif fact.get("retired"):
        why = f"{fid} is retired."
    else:
        value = _num(change.get("value")) if "value" in change else None
        if value is not None and value != _num(fact.get("value")):
            changes.append(dict(field="value", in_force=_num(fact.get("value")), proposed=value))
        status = change.get("status")
        if status is not None and status != fact.get("status"):
            changes.append(dict(field="status", in_force=fact.get("status"), proposed=str(status)))
        not_carried = [k for k in change if k not in CARRIED]
        if not changes:
            why = "Nothing an overlay carries (value, status) changes" + (
                f"; the change of {', '.join(not_carried)} comes with a release of the register file."
                if not_carried
                else "."
            )
    state = item_state(decision)
    return dict(
        key=item_key(item),
        fact_id=fid if fid not in (None, "ASSET") else None,
        kind=kind,
        automatic=bool(item.get("automatic")),
        reason=str(item.get("reason") or ""),
        statement=str(fact.get("statement")) if fact else str(change.get("family") or ""),
        unit=str(fact.get("unit") or "") if fact else "",
        change=_plain(dict(change)),
        register_changes=changes,
        not_carried=not_carried,
        approvable=why is None,
        why=why,
        state=state,
        decidable=state in DECIDABLE,
        decision=dict(decision) if decision is not None else None,
    )


def proposal_view(
    state: Mapping[str, Any],
    register_facts: Mapping[str, Mapping[str, Any]],
    found: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    prop = state.get("proposal") or {}
    items = [
        item_view(it, register_facts.get(str(it.get("id"))) if it.get("id") else None, found.get(item_key(it)))
        for it in prop.get("items") or []
    ]
    return dict(
        day=_day(prop.get("date")),
        hash=proposal_hash(prop),
        object_id=proposal_object(prop),
        n_auto=_int(prop.get("n_auto")) or 0,
        n_approval=_int(prop.get("n_approval")) or 0,
        items=items,
    )


# ------------------------------------------------------------------------------------------------------ view


def job_info(now: datetime | None = None) -> dict[str, Any]:
    kind = CATALOG[KIND]
    now = now or datetime.now(UTC)
    return dict(
        kind=kind.name,
        title=kind.title,
        role=kind.role,
        schedule=schedule.describe(kind.schedule),
        next_slot=schedule.next_slot(kind.schedule, now) if kind.schedule else None,
        timeout_s=float(kind.timeout_s),
    )


def view(
    store: TwinStore | None,
    *,
    directory: Path | str | None = None,
    spec: Mapping[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """The console: the stored shadow run (skill, families with the VAL facts in force, Kalman trajectories, gate,
    CUSUM, register proposal with its decisions), the asset's out-of-sample results, the runs and the job."""
    now = now or datetime.now(UTC)
    spec = spec if spec is not None else asset_spec()
    state = read_state(directory)
    runs = read_runs(directory)
    register = facts.register_view(store)
    by_id = {f["id"]: f for f in register["facts"]}
    asset = asset_info(spec, state)
    out: dict[str, Any] = dict(
        today=now.astimezone(IST).date(),
        asset=asset,
        validation=strata.validation(spec),
        skill_from=_day(EVAL_FROM),
        facts_register=dict(
            in_force_version=register["in_force_version"], label=register["label"], reason=register["reason"]
        ),
        runs=runs,
        job=job_info(now),
        run=None,
        skill=[],
        families=[],
        trajectory=None,
        gate=None,
        cusum=None,
        proposal=None,
        decisions_reason=None,
    )
    if state is None:
        return out
    found: dict[str, dict[str, Any]] = {}
    prop = state.get("proposal") or {}
    if store is None:
        out["decisions_reason"] = "No twin store is configured: decisions cannot be read or made."
    else:
        try:
            found = decisions(store, proposal_object(prop))
        except SQLAlchemyError as exc:
            out["decisions_reason"] = f"The twin store is not reachable ({type(exc).__name__}): decisions not read."
    latest = runs[0] if runs else None
    kal = state.get("state") or {}
    mtime = datetime.fromtimestamp((store_dir(directory) / STATE_FILE).stat().st_mtime, UTC)
    out.update(
        run=dict(
            at=latest["at"] if latest and latest["at"] else mtime,
            day=latest["day"] if latest else None,
            n_days=_int(state.get("n_days")) or 0,
            evidence_days=latest["evidence_days"] if latest else None,
            evidence_to=_day(kal.get("last_date")),
            appended_days=latest["appended_days"] if latest else None,
        ),
        skill=skill_rows(state),
        families=family_rows(state, register["facts"]),
        trajectory=trajectory(read_history(directory), state, spec),
        gate=gate_view(state, asset["deployed"]),
        cusum=cusum_view(state, spec),
        proposal=proposal_view(state, by_id, found),
    )
    return out


# ------------------------------------------------------------------------------------------------------ decisions


def decide(
    store: TwinStore,
    actor: Actor,
    *,
    proposal: str,
    keys: Sequence[str],
    approve: bool,
    note: str | None = None,
    directory: Path | str | None = None,
    at: datetime | None = None,
) -> dict[str, Any]:
    """Approve or reject items of the stored proposal (``proposal``: its hash). An approval proposes the items' value
    and status as the next pending facts version (``services.facts.propose``) with the decision's audit row in the
    same transaction; a rejection records its audit row. ``NoRun``, ``ProposalChanged``, ``DecisionRefused``,
    ``AlreadyDecided``; from the editor ``FactsConflict`` and ``InvalidOverlay``."""
    state = read_state(directory)
    if state is None:
        raise NoRun("STRATA-Learn run stored: the shadow run has not run yet")
    prop = state.get("proposal") or {}
    current = proposal_hash(prop)
    if proposal != current:
        raise ProposalChanged(
            f"the stored proposal ({prop.get('date')}, {current}) is not the one decided on ({proposal}): a new "
            "shadow run replaced it; reload the console"
        )
    items = {item_key(it): it for it in prop.get("items") or []}
    keys = list(dict.fromkeys(str(k) for k in keys))
    if not keys:
        raise DecisionRefused("no item named")
    unknown = [k for k in keys if k not in items]
    if unknown:
        raise DecisionRefused(f"no item {', '.join(unknown)} in the proposal of {prop.get('date')}")
    obj = proposal_object(prop)
    before = _require_open(decisions(store, obj), keys)
    details: dict[str, Any] = dict(proposal_date=prop.get("date"), proposal_hash=current, items=keys, note=note)
    if not approve:
        with audit.admin_write(store, actor, REJECT, OBJECT_TYPE, obj, details=details, at=at) as write:
            facts.lock(write.session)
            before = _require_open(_decisions_in(write.session, obj), keys)
            write.before = {"items": before}
            write.after = {"items": {k: "rejected" for k in keys}}
        assert write.entry is not None
        return dict(
            decision="rejected", proposal_date=_day(prop.get("date")), proposal_hash=current, items=keys,
            facts_version=None, audit_id=int(write.entry.id),
        )  # fmt: skip
    register = facts.register_view(store)
    by_id = {f["id"]: f for f in register["facts"]}
    views = {k: item_view(items[k], by_id.get(str(items[k].get("id"))), None) for k in keys}
    refused = [f"{k}: {views[k]['why']}" for k in keys if not views[k]["approvable"]]
    if refused:
        raise DecisionRefused("cannot be approved through the register: " + " ".join(refused))
    changes = {str(items[k]["id"]): {c["field"]: c["proposed"] for c in views[k]["register_changes"]} for k in keys}
    text = f"STRATA-Learn proposal {prop.get('date')} ({current}): " + "; ".join(
        f"{k} {items[k].get('reason', '')}".strip() for k in keys
    )
    if note:
        text += f". {note}"
    entry: list[AuditLog] = []

    def link(write: audit.AdminWrite, row: FactsVersion) -> None:
        states = _require_open(_decisions_in(write.session, obj), keys)
        entry[:] = [
            audit_row(
                write.session,
                actor,
                APPROVE,
                OBJECT_TYPE,
                obj,
                audit.json_safe({**details, "facts_version": row.version, "changes": changes}),
                write.at,
                before={"items": states},
                after={"items": {k: "proposed" for k in keys}, "facts_version": row.version},
            )
        ]

    version = facts.propose(
        store,
        actor,
        parent_version=register["in_force_version"],
        changes=changes,
        note=text,
        at=at,
        details={"learn_proposal": obj, "learn_items": keys},
        also=link,
    )
    return dict(
        decision="approved", proposal_date=_day(prop.get("date")), proposal_hash=current, items=keys,
        facts_version=version, audit_id=int(entry[0].id),
    )  # fmt: skip


__all__ = [
    "APPROVE",
    "KIND",
    "OBJECT_TYPE",
    "REJECT",
    "AlreadyDecided",
    "DecisionRefused",
    "NoRun",
    "ProposalChanged",
    "decide",
    "decisions",
    "item_view",
    "proposal_hash",
    "proposal_object",
    "read_state",
    "view",
]
