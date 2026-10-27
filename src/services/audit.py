"""Audit log of the admin space (SV-4.1, #304): admin writes with the object's values before and after, and the pages
of the log for the viewer (``GET /api/admin/audit``).

Every admin write goes through ``admin_write`` on the twin store. The block makes the change through
``write.session`` and sets ``write.before`` and ``write.after`` (the object's fields, ``row_values``); when the block
ends, the audit row is added to the same transaction, so the change and its row are committed together or not at all:

    with audit.admin_write(store, actor, "facts.propose", "facts_version", number) as write:
        current = write.session.get(FactsVersion, key, with_for_update=True)
        write.before = audit.row_values(current, include=FIELDS)
        ...  # change the row, or add a new one
        write.session.flush()
        write.after = audit.row_values(current, include=FIELDS)

``before`` is None for a created object and ``after`` for a deleted one; a block that leaves either unset, or both
None, is refused and rolled back. Read the object ``FOR UPDATE`` when two administrators can change it at once, so
that ``before`` is the value the write replaced. The guarantee covers writes through ``write.session`` (the twin
store's database, any schema in it); a change elsewhere (Redis, a file) made inside the block is kept when the audit
row fails, so make it the block's last step.

Values are stored as JSON (``json_safe``); ``changes`` lists the fields that differ. Pure apart from the store calls;
no Streamlit (tests/test_services_architecture.py).
"""

from __future__ import annotations

import dataclasses
import json
import math
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from pathlib import PurePath
from typing import Any, Literal

from sqlalchemy import inspect as sa_inspect
from sqlalchemy.orm import Session

from data.twin.models import AuditLog
from data.twin.repositories import Actor, TwinStore, audit_row, utcnow
from services.twin_views import utc

#: Lengths of the audit log's columns (``data.twin.models.AuditLog``).
MAX_ACTION = 64
MAX_OBJECT_TYPE = 64
MAX_OBJECT_ID = 128
MAX_ACTOR = 128

ChangeKind = Literal["added", "removed", "changed"]


class AuditIncomplete(ValueError):
    """An admin write that does not state the object's values before and after; the write is rolled back."""


class _Unset:
    def __repr__(self) -> str:
        return "UNSET"


UNSET: Any = _Unset()


@dataclass
class AdminWrite:
    """One admin write in progress: the session of its transaction and the values its audit row records."""

    session: Session
    actor: Actor
    action: str
    object_type: str
    object_id: str | None
    at: datetime
    details: dict[str, Any] = field(default_factory=dict)
    #: The object's fields before the write; None when the write creates it.
    before: Mapping[str, Any] | None = UNSET
    #: The object's fields after the write; None when the write deletes it.
    after: Mapping[str, Any] | None = UNSET
    #: The audit row, once the block has ended (its id is assigned inside the transaction).
    entry: AuditLog | None = None


def _checked_text(name: str, value: str | None, limit: int, *, required: bool = True) -> str | None:
    if value is None or value == "":
        if required:
            raise ValueError(f"an admin write needs {name}")
        return None
    if len(value) > limit:
        raise ValueError(f"{name} has {len(value)} characters; the audit log keeps at most {limit}")
    return value


def _values(name: str, value: Any) -> dict[str, Any] | None:
    if value is UNSET:
        raise AuditIncomplete(f"an admin write records {name}: set write.{name} (None when the object is absent)")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(f"write.{name} must be a mapping of the object's fields, got {type(value).__name__}")
    return json_safe(value)


@contextmanager
def admin_write(
    store: TwinStore,
    actor: Actor,
    action: str,
    object_type: str,
    object_id: str | int | uuid.UUID | None = None,
    *,
    details: Mapping[str, Any] | None = None,
    at: datetime | None = None,
) -> Iterator[AdminWrite]:
    """A transaction on the twin store for one admin write and its audit row (one row, with ``before`` and
    ``after``). An exception in the block, or in writing the row, rolls back both."""
    _checked_text("an actor", actor.name, MAX_ACTOR)
    _checked_text("an action", action, MAX_ACTION)
    _checked_text("an object type", object_type, MAX_OBJECT_TYPE)
    key = _checked_text("an object id", None if object_id is None else str(object_id), MAX_OBJECT_ID, required=False)
    if at is not None and (at.tzinfo is None or at.utcoffset() is None):
        raise ValueError(f"at must be timezone-aware, got {at!r}")
    when = utc(at) if at is not None else utcnow()
    with store.sessions.begin() as session:
        write = AdminWrite(session, actor, action, object_type, key, when, dict(details or {}))
        yield write
        before, after = _values("before", write.before), _values("after", write.after)
        if before is None and after is None:
            raise AuditIncomplete("an admin write changes an object: before and after cannot both be None")
        session.flush()  # the write's own statements first: a refused write fails before its audit row is added
        write.entry = audit_row(
            session,
            actor,
            action,
            object_type,
            write.object_id,
            json_safe(write.details),
            when,
            before=before,
            after=after,
        )
        session.flush()


# ------------------------------------------------------------------------------------------------------ values


def json_safe(value: Any) -> Any:
    """``value`` as JSON data: mappings with text keys, lists, text, finite numbers, booleans and None.

    Times become ISO 8601 text (aware times in UTC), dates and UUIDs text, decimals floats, enums their value, sets
    sorted lists, dataclasses and Pydantic models their fields, NumPy values Python numbers; NaN and infinities (not
    valid in PostgreSQL's JSON) become None. Anything else raises ``TypeError``.
    """
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, Enum):
        return json_safe(value.value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float | Decimal):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, datetime):
        return (utc(value) if value.tzinfo is not None else value).isoformat()
    if isinstance(value, date | time):
        return value.isoformat()
    if isinstance(value, uuid.UUID | PurePath):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, set | frozenset):
        return sorted((json_safe(v) for v in value), key=lambda v: json.dumps(v, sort_keys=True))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return json_safe(dataclasses.asdict(value))
    if hasattr(value, "model_dump"):
        return json_safe(value.model_dump(mode="json"))
    if hasattr(value, "tolist"):  # NumPy scalars and arrays
        return json_safe(value.tolist())
    raise TypeError(f"{type(value).__name__} is not JSON data")


def row_values(row: Any, *, include: Sequence[str] | None = None, exclude: Iterable[str] = ()) -> dict[str, Any] | None:
    """The fields of a row as JSON data, for ``before`` and ``after``; None for None. A row is an ORM object (its
    mapped columns), a Core result row, a mapping, a dataclass or a Pydantic model. ``include`` picks fields in its
    order (an unknown name raises), ``exclude`` drops fields."""
    if row is None:
        return None
    if isinstance(row, Mapping):
        values = dict(row)
    elif hasattr(row, "_mapping"):  # sqlalchemy.engine.Row
        values = dict(row._mapping)
    elif dataclasses.is_dataclass(row) and not isinstance(row, type):
        values = {f.name: getattr(row, f.name) for f in dataclasses.fields(row)}
    elif hasattr(row, "model_dump"):
        values = dict(row.model_dump())
    else:
        mapper = sa_inspect(row).mapper
        values = {attr.key: getattr(row, attr.key) for attr in mapper.column_attrs}
    if include is not None:
        unknown = [name for name in include if name not in values]
        if unknown:
            raise ValueError(f"no field {', '.join(unknown)}")
        values = {name: values[name] for name in include}
    dropped = set(exclude)
    return json_safe({k: v for k, v in values.items() if k not in dropped})


def _flatten(value: Any, path: str, out: dict[str, Any]) -> None:
    if isinstance(value, Mapping) and value:
        for key, item in value.items():
            _flatten(item, f"{path}.{key}", out)
    elif isinstance(value, list) and value:
        for index, item in enumerate(value):
            _flatten(item, f"{path}[{index}]", out)
    else:
        out[path] = value


def flatten(values: Mapping[str, Any] | None) -> dict[str, Any]:
    """Leaf values by path: keys joined by dots, list items by index (``coke_angles_deg[3]``); an empty mapping or
    list is a leaf of its own."""
    out: dict[str, Any] = {}
    for key, item in (values or {}).items():
        _flatten(item, str(key), out)
    return out


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):  # True == 1 in Python; not in the log
        return type(a) is type(b) and a == b
    return bool(a == b)


def changes(before: Mapping[str, Any] | None, after: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """The fields that differ between ``before`` and ``after``, in the order of ``before`` and then of the fields only
    in ``after``: ``{"path", "kind": added | removed | changed, "before", "after"}`` (the absent side is None)."""
    old, new = flatten(before), flatten(after)
    out: list[dict[str, Any]] = []
    for path, value in old.items():
        if path not in new:
            out.append({"path": path, "kind": "removed", "before": value, "after": None})
        elif not _same(value, new[path]):
            out.append({"path": path, "kind": "changed", "before": value, "after": new[path]})
    for path, value in new.items():
        if path not in old:
            out.append({"path": path, "kind": "added", "before": None, "after": value})
    return out


# ------------------------------------------------------------------------------------------------------- viewer


@dataclass(frozen=True)
class AuditFilter:
    """Filters of the viewer: exact user name, action, object type and id; time range [since, until)."""

    actor: str | None = None
    action: str | None = None
    object_type: str | None = None
    object_id: str | None = None
    since: datetime | None = None
    until: datetime | None = None


def entry_data(row: Any) -> dict[str, Any]:
    """An audit row as plain values, with the fields its write changed."""
    before = None if row.before is None else dict(row.before)
    after = None if row.after is None else dict(row.after)
    return {
        "id": int(row.id),
        "at": utc(row.at),
        "actor": row.actor,
        "role": row.role,
        "action": row.action,
        "object_type": row.object_type,
        "object_id": row.object_id,
        "client_ip": row.client_ip,
        "details": dict(row.details or {}),
        "before": before,
        "after": after,
        "changes": changes(before, after),
    }


def read_page(
    store: TwinStore, filters: AuditFilter | None = None, *, limit: int = 50, offset: int = 0
) -> dict[str, Any]:
    """One page of the log, newest first: the entries, the number that match the filters and the values present in
    the whole log for the filter menus (``facets``)."""
    filters = filters or AuditFilter()
    if filters.since is not None and filters.until is not None and filters.since >= filters.until:
        raise ValueError("since must lie before until")
    rows, total = store.audit.page(**dataclasses.asdict(filters), limit=limit, offset=offset)
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "entries": [entry_data(row) for row in rows],
        "facets": store.audit.facets(),
    }


__all__ = [
    "UNSET",
    "AdminWrite",
    "AuditFilter",
    "AuditIncomplete",
    "admin_write",
    "changes",
    "entry_data",
    "flatten",
    "json_safe",
    "read_page",
    "row_values",
]
