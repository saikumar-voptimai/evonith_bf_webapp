"""Fact register editor (SV-4.2, #305): versions of the fact register as overlays in ``twin.facts_version``, their
review and the register the engine loads.

The repository file ``src/config/facts_register.json`` (v1.8, 116 facts) stays the base. An administrator changes a
fact's value, band or status; the change is saved as a pending version whose ``overlay`` holds every fact that differs
from the file (cumulative, ``{"ASM-08": {"value": 185}}``), made from the version in force (``parent_version``, 0 for
the file). Each version is validated before it is saved and again before it is approved: the overlaid register must
pass the JSON schema ``src/config/schemas/facts_register.schema.json`` and the register's checks (statuses, ids).

* A version that changes a fact's status (against the version in force) needs a second administrator; a version that
  changes values or bands only may be approved by its author.
* Approval needs the version to be made from the version in force (else ``FactsConflict``: propose again) and
  against the file's version.
* In force = the approved version with the highest number made against the file's version, else the file.
  ``approved_overlay()`` is the engine's source (``utils.burden_engine.facts.load_register``), re-read every 60 s;
  after the repository takes an export (new file version), older overlays no longer apply.
* ``export`` writes the file of an approved version for a repository PR: the file's version + 0.1, the changed facts
  stamped with that version and the approval day (IST), a revision-log entry; it passes the schema.

Every write goes through ``services.audit.admin_write`` (``facts.propose``, ``facts.review``): one audit row with the
version's fields before and after in the transaction of the write. No Streamlit (tests/test_services_architecture.py).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from data.twin.models import FactsVersion
from data.twin.repositories import Actor, NotFound, ReviewRefused, TwinStore, overlay_hash
from services import audit
from services import store as store_service
from services.twin_views import utc
from utils.burden_engine import facts as register
from utils.memo import memo

log = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")
#: Fields of ``twin.facts_version`` the audit rows record.
FIELDS = (
    "version",
    "base_version",
    "parent_version",
    "status",
    "overlay",
    "content_hash",
    "second_review",
    "note",
    "created_by",
    "created_at",
    "reviewed_by",
    "reviewed_at",
)
EDIT_FIELDS = register.OVERLAY_FIELDS
FACT_KEYS = (
    "id",
    "group",
    "statement",
    "value",
    "unit",
    "se",
    "status",
    "grade",
    "basis",
    "literature",
    "validity_window",
    "next_test",
    "version",
    "date",
    "note",
)


class InvalidOverlay(ValueError):
    """Changes that do not give a usable register (unknown fact or field, schema or register check)."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems


class Unavailable(RuntimeError):
    """The twin store cannot be read or written."""


class FactsConflict(RuntimeError):
    """A version made from another version than the one in force, or against another file version."""


# ------------------------------------------------------------------------------------------------------ values


def file_data() -> dict[str, Any]:
    """The repository file as a document (read on every call)."""
    return register.load_register_data()


def field_value(fact: Mapping[str, Any], name: str) -> Any:
    """Value of an editable field of a fact document: ``value``, ``band`` ([lo, hi] or None) or ``status``."""
    if name == "band":
        return register.fact_band(fact)
    value = fact.get(name)
    return float(value) if name == "value" and value is not None else value


def _normal(name: str, value: Any) -> Any:
    """A requested value in the form ``field_value`` gives, for the comparison with the file."""
    if name == "value" and isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    if name == "band" and isinstance(value, list | tuple):
        return [float(v) if isinstance(v, int | float) and not isinstance(v, bool) else v for v in value]
    return value


def _facts(doc: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {f["id"]: f for f in doc["facts"]}


def merge(data: Mapping[str, Any], parent: Mapping[str, Any], changes: Mapping[str, Mapping[str, Any]]) -> dict:
    """The overlay of ``parent`` with ``changes`` (``{fact id: {value?, band?, status?}}``): a field set to the file's
    value leaves the overlay, a fact without fields too. ``InvalidOverlay`` on an unknown fact or field."""
    base = _facts(data)
    out = {fid: dict(patch) for fid, patch in parent.items()}
    problems: list[str] = []
    for fid, patch in changes.items():
        if fid not in base:
            problems.append(f"{fid}: no such fact in register v{data['version']}")
            continue
        unknown = sorted(set(patch) - set(EDIT_FIELDS))
        if unknown:
            problems.append(f"{fid}: {', '.join(unknown)} cannot be changed (only {', '.join(EDIT_FIELDS)})")
            continue
        entry = dict(out.get(fid, {}))
        for name, value in patch.items():
            requested = _normal(name, value)
            if requested == field_value(base[fid], name):
                entry.pop(name, None)
            else:
                entry[name] = requested
        if entry:
            out[fid] = entry
        else:
            out.pop(fid, None)
    if problems:
        raise InvalidOverlay(problems)
    return out


def overlaid(data: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """The file with ``overlay`` applied, checked: ``InvalidOverlay`` with the problems found."""
    try:
        doc = register.apply_overlay(data, overlay)
    except register.OverlayError as exc:
        raise InvalidOverlay([str(exc)]) from exc
    problems = register.register_problems(doc)
    if problems:
        raise InvalidOverlay(problems)
    return doc


def status_changes(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[str]:
    """Facts whose status differs between two register documents, in register order."""
    a, b = _facts(before), _facts(after)
    return [fid for fid in a if fid in b and a[fid]["status"] != b[fid]["status"]]


def changes_between(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The editable fields that differ between two register documents, in register order: ``{id, group, statement,
    unit, field, before, after}``."""
    a, b = _facts(before), _facts(after)
    rows = []
    for fid, fa in a.items():
        fb = b.get(fid)
        if fb is None:
            continue
        for name in EDIT_FIELDS:
            old, new = field_value(fa, name), field_value(fb, name)
            if old != new:
                rows.append(
                    {
                        "id": fid,
                        "group": fa["group"],
                        "statement": fa["statement"],
                        "unit": fa.get("unit", ""),
                        "field": name,
                        "before": old,
                        "after": new,
                    }
                )
    return rows


def fact_view(fact: Mapping[str, Any], file_fact: Mapping[str, Any], overlay: Mapping[str, Any] | None) -> dict:
    """A fact as the editor shows it: its fields, band, whether it is retired and which fields the overlay changes
    (``file``: the repository's values of those fields)."""
    out = {key: fact.get(key, "" if key not in ("value", "se") else None) for key in FACT_KEYS}
    changed = [name for name in EDIT_FIELDS if name in (overlay or {})]
    out.update(
        value=field_value(fact, "value"),
        band=register.fact_band(fact),
        band_kind=register.band_kind(fact),
        params=dict(fact.get("params") or {}),
        retired=fact["status"] == register.RETIRED,
        changed=changed,
        file={name: field_value(file_fact, name) for name in changed} if changed else None,
    )
    return out


# ------------------------------------------------------------------------------------------------------ store


_PROCESS = object()


def _store(store: Any) -> TwinStore | None:
    return store_service.twin_store() if store is _PROCESS else store


def in_force(store: TwinStore, file_version: str | None = None) -> FactsVersion | None:
    """The approved version in force: the highest approved version made against the file's version."""
    return store.facts.latest_approved(base_version=file_version or file_data()["version"])


@memo(ttl_seconds=60)
def approved_overlay() -> register.ApprovedOverlay | None:
    """The overlay the engine applies (``load_register``): the version in force in the process's twin store; None
    without a store, without an approved version, or when the store cannot be read (the file is in force)."""
    store = store_service.twin_store()
    if store is None:
        return None
    try:
        row = in_force(store)
    except SQLAlchemyError as exc:
        log.warning("fact register overlay not read: %s", str(exc).splitlines()[0][:300])
        return None
    if row is None:
        return None
    return register.ApprovedOverlay(row.version, row.base_version, dict(row.overlay), row.content_hash)


def lock(session: Session) -> None:
    """Serialise proposals and reviews, and writes linked to them (PostgreSQL; SQLite writes one at a time)."""
    if session.get_bind().dialect.name == "postgresql":
        session.execute(text("SELECT pg_advisory_xact_lock(hashtext('twin.facts_version'))"))


def _in_force_row(session: Session, file_version: str) -> FactsVersion | None:
    query = (
        select(FactsVersion)
        .where(FactsVersion.status == "approved", FactsVersion.base_version == file_version)
        .order_by(FactsVersion.version.desc())
        .limit(1)
    )
    return session.scalars(query).one_or_none()


def _name(version: int | None) -> str:
    return f"version {version}" if version else "the repository file"


def version_view(row: FactsVersion, in_force_version: int | None, file_version: str) -> dict[str, Any]:
    return {
        "version": row.version,
        "parent_version": row.parent_version or 0,
        "base_version": row.base_version,
        "status": row.status,
        "second_review": bool(row.second_review),
        "note": row.note,
        "created_by": row.created_by,
        "created_at": utc(row.created_at),
        "reviewed_by": row.reviewed_by,
        "reviewed_at": utc(row.reviewed_at) if row.reviewed_at is not None else None,
        "facts": sorted(row.overlay or {}),
        "content_hash": row.content_hash,
        "in_force": row.version == in_force_version,
        "applies": row.base_version == file_version,
    }


def versions(store: TwinStore, *, limit: int = 100) -> list[dict[str, Any]]:
    """Versions newest first, with the one in force marked."""
    version = file_data()["version"]
    current = in_force(store, version)
    number = current.version if current else None
    return [version_view(row, number, version) for row in store.facts.recent(limit=limit)]


def _document(store: TwinStore | None, data: Mapping[str, Any], version: int) -> tuple[dict, FactsVersion | None]:
    """The register of ``version`` (0: the file) and its row; ``NotFound``, or ``FactsConflict`` when the version
    was made against another file version."""
    if version == 0:
        return dict(data), None
    if store is None:
        raise Unavailable("The twin store is not configured: set BF_TWIN_DB_URL or DATABASE_URL.")
    row = store.facts.get(version)
    if row is None:
        raise NotFound(f"fact-register version {version}")
    if row.base_version != data["version"]:
        raise FactsConflict(
            f"version {version} was made against register v{row.base_version}; the repository file is "
            f"v{data['version']}"
        )
    try:
        return register.apply_overlay(data, row.overlay or {}), row
    except register.OverlayError as exc:
        raise FactsConflict(f"version {version} does not apply to the repository file: {exc}") from exc


def register_view(store: Any = _PROCESS, version: int | None = None) -> dict[str, Any]:
    """The register as the editor shows it: ``version`` None the version in force, 0 the repository file, else that
    version (any status). Without a store (or one that cannot be read) the file, with the ``reason``."""
    twin = _store(store)
    data = file_data()
    reason = None
    current: FactsVersion | None = None
    if twin is None:
        reason = "No twin store is configured: the repository file is in force."
    else:
        try:
            current = in_force(twin, data["version"])
        except SQLAlchemyError as exc:
            if version:
                raise Unavailable(f"The twin store is not reachable ({type(exc).__name__}).") from exc
            reason = f"The twin store is not reachable ({type(exc).__name__}): the repository file is shown."
    shown = (current.version if current else 0) if version is None else version
    doc, row = _document(twin, data, shown)
    base = _facts(data)
    overlay = (row.overlay or {}) if row is not None else {}
    groups = list(dict.fromkeys(f["group"] for f in data["facts"]))
    facts = [fact_view(f, base[f["id"]], overlay.get(f["id"])) for f in doc["facts"]]
    return {
        "name": data["name"],
        "file_version": data["version"],
        "file_date": data["date"],
        "statuses": list(data["statuses"]),
        "grades": dict(data.get("grades") or {}),
        "groups": groups,
        "in_force_version": current.version if current else 0,
        "version": shown,
        "version_status": row.status if row is not None else "file",
        "label": data["version"] if not shown else f"{data['version']} + overlay {shown}",
        "reason": reason,
        "facts": facts,
    }


# ------------------------------------------------------------------------------------------------------ writes


def propose(
    store: TwinStore,
    actor: Actor,
    *,
    parent_version: int,
    changes: Mapping[str, Mapping[str, Any]],
    note: str,
    at: datetime | None = None,
    details: Mapping[str, Any] | None = None,
    also: Callable[[audit.AdminWrite, FactsVersion], None] | None = None,
) -> dict[str, Any]:
    """Save ``changes`` on the version in force as the next pending version, with its audit row
    (``facts.propose``). ``parent_version``: the version the edit started from (0: the file); ``FactsConflict``
    when another is in force, ``InvalidOverlay`` when the register would not be usable or nothing changes.
    ``details`` are added to the audit row's details; ``also`` runs in the write's transaction once the version is
    added (e.g. the audit row of the decision that proposed it, SV-4.5): an exception in it rolls back both."""
    data = file_data()
    for attempt in range(3):  # two proposals at once: the second takes the next number
        try:
            with audit.admin_write(store, actor, "facts.propose", "facts_version", None, at=at) as write:
                session = write.session
                lock(session)
                current = _in_force_row(session, data["version"])
                current_no = current.version if current else 0
                if parent_version != current_no:
                    raise FactsConflict(
                        f"the edit started from {_name(parent_version)}; {_name(current_no)} is in force now: "
                        "reload the register and edit again"
                    )
                parent = dict(current.overlay or {}) if current else {}
                overlay = merge(data, parent, changes)
                if overlay == parent:
                    raise InvalidOverlay([f"the changes leave every fact as it is in {_name(current_no)}"])
                before_doc = register.apply_overlay(data, parent)
                after_doc = overlaid(data, overlay)
                statuses = status_changes(before_doc, after_doc)
                number = (session.scalar(select(func.max(FactsVersion.version))) or 0) + 1
                row = FactsVersion(
                    version=number,
                    base_version=data["version"],
                    parent_version=current_no or None,
                    status="pending",
                    overlay=overlay,
                    content_hash=overlay_hash(overlay),
                    second_review=bool(statuses),
                    note=note,
                    created_by=actor.name,
                    created_at=write.at,
                )
                session.add(row)
                session.flush()
                write.object_id = str(number)
                write.details = {
                    "note": note,
                    "parent_version": current_no,
                    "facts": sorted({c["id"] for c in changes_between(before_doc, after_doc)}),
                    "status_changes": statuses,
                    "second_review": bool(statuses),
                    **dict(details or {}),
                }
                write.before = None
                write.after = audit.row_values(row, include=FIELDS)
                if also is not None:
                    also(write, row)
            return version_view(row, current_no or None, data["version"])
        except IntegrityError:
            if attempt == 2:
                raise
    raise AssertionError("unreachable")  # pragma: no cover


def review(
    store: TwinStore,
    actor: Actor,
    version: int,
    *,
    approve: bool,
    note: str | None = None,
    at: datetime | None = None,
) -> dict[str, Any]:
    """Approve or reject a pending version, with its audit row (``facts.review``). ``ReviewRefused`` when it is not
    pending or changes a status and ``actor`` proposed it; on approval ``FactsConflict`` when it was not made from
    the version in force or against the file's version, ``InvalidOverlay`` when the register would not be usable."""
    data = file_data()
    decision = "approved" if approve else "rejected"
    details = {"decision": decision, "note": note}
    with audit.admin_write(store, actor, "facts.review", "facts_version", version, details=details, at=at) as write:
        session = write.session
        lock(session)
        row = session.scalars(
            select(FactsVersion).where(FactsVersion.version == version).with_for_update()
        ).one_or_none()
        if row is None:
            raise NotFound(f"fact-register version {version}")
        write.before = audit.row_values(row, include=FIELDS)
        if row.status != "pending":
            raise ReviewRefused(f"version {version} is {row.status}, not pending")
        if row.second_review and actor.name == row.created_by:
            raise ReviewRefused(
                f"version {version} changes the status of a fact: a second administrator must review it"
            )
        current = _in_force_row(session, data["version"])
        if approve:
            if row.base_version != data["version"]:
                raise FactsConflict(
                    f"version {version} was made against register v{row.base_version}; the repository file is "
                    f"v{data['version']}"
                )
            current_no = current.version if current else 0
            if (row.parent_version or 0) != current_no:
                raise FactsConflict(
                    f"version {version} was made from {_name(row.parent_version)}; {_name(current_no)} is in force "
                    "now: propose the change again"
                )
            overlaid(data, row.overlay or {})
        row.status = decision
        row.reviewed_by = actor.name
        row.reviewed_at = write.at
        session.flush()
        write.after = audit.row_values(row, include=FIELDS)
    approved_overlay.clear()
    number = row.version if approve else (current.version if current else None)
    return version_view(row, number, data["version"])


# ------------------------------------------------------------------------------------------------------ reading


def diff(store: Any, from_version: int, to_version: int) -> dict[str, Any]:
    """The facts' values, bands and statuses that differ between two versions (0: the repository file)."""
    twin = _store(store)
    data = file_data()
    before, _ = _document(twin, data, from_version)
    after, _ = _document(twin, data, to_version)
    return {"from_version": from_version, "to_version": to_version, "changes": changes_between(before, after)}


def next_version(version: str) -> str:
    """``1.8`` -> ``1.9``, ``1.9`` -> ``1.10``."""
    major, minor = version.split(".")
    return f"{major}.{int(minor) + 1}"


def _text(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, list):
        return "[" + ", ".join(f"{v:g}" if isinstance(v, float) else str(v) for v in value) + "]"
    return f"{value:g}" if isinstance(value, float) else str(value)


def export(store: TwinStore, version: int | None = None) -> dict[str, Any]:
    """The register file of an approved version (default: the one in force) for a repository PR: version + 0.1,
    changed facts stamped with it and the approval day (IST), a revision-log entry. Schema-valid (checked)."""
    data = file_data()
    row = in_force(store, data["version"]) if version is None else store.facts.get(version)
    if row is None:
        if version is None:
            raise ReviewRefused("no approved version is in force: the repository file is the register")
        raise NotFound(f"fact-register version {version}")
    if row.status != "approved":
        raise ReviewRefused(f"version {row.version} is {row.status}: only an approved version is exported")
    doc, _ = _document(store, data, row.version)
    rows = changes_between(data, doc)
    new = next_version(data["version"])
    day = utc(row.reviewed_at).astimezone(IST).date().isoformat()
    for fact in doc["facts"]:
        if fact["id"] in (row.overlay or {}):
            fact["version"] = new
            fact["date"] = day
    listed = "; ".join(f"{r['id']} {r['field']} {_text(r['before'])} -> {_text(r['after'])}" for r in rows)
    change = (
        f"Admin overlay version {row.version} (twin.facts_version; proposed by {row.created_by}, approved by "
        f"{row.reviewed_by} on {day}): {listed}."
    )
    if row.note:
        change += f" {row.note.strip()}"
    doc["version"] = new
    doc["date"] = day
    doc["revision_log"] = [{"version": new, "date": day, "change": change}, *doc["revision_log"]]
    problems = register.register_problems(doc)
    if problems:  # pragma: no cover - the overlay was checked on approval
        raise InvalidOverlay(problems)
    return doc


def export_text(doc: Mapping[str, Any]) -> str:
    """The document in the file's format (indent 1, UTF-8, final newline), so the PR shows only the changes."""
    return json.dumps(doc, indent=1, ensure_ascii=False) + "\n"


__all__ = [
    "FIELDS",
    "FactsConflict",
    "InvalidOverlay",
    "Unavailable",
    "approved_overlay",
    "changes_between",
    "diff",
    "export",
    "export_text",
    "file_data",
    "in_force",
    "lock",
    "merge",
    "next_version",
    "propose",
    "register_view",
    "review",
    "versions",
]
