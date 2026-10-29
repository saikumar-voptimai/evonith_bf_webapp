"""User, hopper and burden-distribution admin (SV-4.7, #310): the Welcome page's Streamlit tools for the SPA.

The tools (``ui.user_management``, ``ui.hopper_admin_page``, ``ui.burden_admin_page``) write Neon ``identity.users`` /
``identity.user_roles``, ``ops_config.hopper_raw_material_history`` and ``ops_config.burden_history``. This module
makes the same writes through the same functions: the form is reduced to its changes by ``domain.plant_config``
(shared with the Streamlit tools), client material names map to codes through ``data.db.HopperConfigService``, and
the rows are written by the session-level repository functions of ``furnace_data.relational``
(``UserRepository.create_user``, ``HopperHistoryRepository.add_snapshot`` / ``delete_rows``,
``BurdenHistoryRepository.add_snapshot`` / ``delete_rows``) that the Streamlit services call in their own sessions.

History is SCD Type-2 as wide snapshots: an assignment or a burden change adds a full snapshot (the latest one at or
before the effective time, with the changed fields) and leaves every earlier row as it is; the current mapping is the
latest snapshot. Deleting a snapshot is an explicit admin action, as in the Streamlit tools.

The tables are read and written on the twin store's database (``BF_TWIN_DB_URL``, else ``DATABASE_URL``: the plant's
Neon database, which holds the ``identity``, ``plant_master`` and ``ops_config`` schemas), so every write commits
together with its audit row (``services.audit.admin_write``: ``user.create``, ``hopper.assign``, ``hopper.delete``,
``burden.update``, ``burden.delete``). Audit rows record the stored values (material codes, never a password hash).

Times are timezone-aware (UTC in the responses). No Streamlit (tests/test_services_architecture.py).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.exc import IntegrityError, ProgrammingError

from data.db import HopperConfigService, password_digest
from data.twin.repositories import Actor, TwinStore
from domain import plant_config
from furnace_data.relational import (
    HOPPER_COLUMNS,
    BurdenHistory,
    BurdenHistoryRepository,
    HopperHistoryRepository,
    HopperRawMaterialHistory,
    UserRepository,
    UserRole,
)
from services import audit
from services.twin_views import utc

ROLES: tuple[str, ...] = tuple(role.value for role in (UserRole.USER, UserRole.SUPERVISOR, UserRole.ADMIN))
#: The client address stored on a snapshot when the request has none (as the Streamlit tools store it).
UNKNOWN_IP = "unknown"
_SCHEMAS = "identity, plant_master and ops_config"


class Unavailable(RuntimeError):
    """The plant configuration tables cannot be read on the twin store's database."""


class Invalid(ValueError):
    """A form the tables refuse: empty field, unknown hopper, material or burden field, a value that is no number."""


class Conflict(RuntimeError):
    """The username exists."""


class NotFound(LookupError):
    """History rows that do not exist."""


def _unavailable(exc: Exception) -> Unavailable:
    return Unavailable(
        f"The plant configuration tables ({_SCHEMAS}) are not readable on the twin store's database: "
        f"{type(exc).__name__}."
    )


def _now(now: datetime | None) -> datetime:
    return datetime.now(UTC) if now is None else utc(now)


def _aware(name: str, value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise Invalid(f"{name} must carry its time zone (e.g. +05:30 for IST)")
    return utc(value)


def _ip(actor: Actor) -> str:
    return actor.client_ip or UNKNOWN_IP


def _ids_text(ids: Sequence[int]) -> str:
    text = ",".join(str(i) for i in ids)
    return text if len(text) <= audit.MAX_OBJECT_ID else f"{ids[0]}..{ids[-1]} ({len(ids)} rows)"


def _unique_ids(ids: Iterable[int]) -> list[int]:
    out = sorted({int(i) for i in ids})
    if not out:
        raise Invalid("Select at least one history row")
    return out


# --------------------------------------------------------------------------------------------------------- users


def users_view(store: TwinStore) -> dict[str, Any]:
    """Every account (active or not) by username with its role and role assignments, and the roles a new account
    can have."""
    try:
        users = UserRepository(store.sessions).list_users()
    except ProgrammingError as exc:
        raise _unavailable(exc) from exc
    return {
        "roles": list(ROLES),
        "users": [{**u, "created_at": utc(u["created_at"]), "updated_at": utc(u["updated_at"])} for u in users],
    }


def register_user(store: TwinStore, actor: Actor, username: str, password: str, role: str) -> dict[str, Any]:
    """Add an account as the Streamlit register form does (``identity.users`` + ``identity.user_roles``, password
    stored as ``data.db.password_digest``), with its audit row ``user.create`` (no password hash in it).

    Raises:
        Invalid: an empty username or password, or an unknown role.
        Conflict: the username exists.
    """
    try:
        name, secret = plant_config.registration(username, password)
    except plant_config.EmptyField as exc:
        raise Invalid(str(exc).rstrip(".")) from exc
    if role not in ROLES:
        raise Invalid(f"Unknown role {role!r}: one of {', '.join(ROLES)}")
    if len(name) > audit.MAX_OBJECT_ID:
        raise Invalid(f"A username has at most {audit.MAX_OBJECT_ID} characters")
    try:
        with audit.admin_write(store, actor, "user.create", "user", name) as write:
            try:
                user = UserRepository.create_user(
                    write.session, username=name, password_hash=password_digest(secret), role=role
                )
            except IntegrityError as exc:
                raise Conflict(f"User {name!r} already exists") from exc
            write.details["user_id"] = str(user.id)
            write.before = None
            write.after = {"username": user.username, "role": user.role, "is_active": user.is_active, "roles": [role]}
            view = {
                "id": str(user.id),
                "username": user.username,
                "role": user.role,
                "roles": [role],
                "is_active": user.is_active,
                "created_at": utc(user.created_at),
                "updated_at": utc(user.updated_at),
            }
    except ProgrammingError as exc:
        raise _unavailable(exc) from exc
    return {"user": view, "audit_id": int(write.entry.id)}


# ------------------------------------------------------------------------------------------------------- hoppers


def _hopper_service(store: TwinStore) -> HopperConfigService:
    try:
        return HopperConfigService(session_factory=store.sessions)
    except ProgrammingError as exc:
        raise _unavailable(exc) from exc
    except ValueError as exc:  # materials_map.yml names a material that plant_master does not hold
        raise Unavailable(f"The material map does not match plant_master: {exc}") from exc


def _hopper_audit(row: HopperRawMaterialHistory | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {"date_time": row.date_time, **{column: getattr(row, column) for column in HOPPER_COLUMNS}}


def _usernames(store: TwinStore) -> dict[str, str]:
    return UserRepository(store.sessions).usernames_by_id()


def hopper_view(store: TwinStore) -> dict[str, Any]:
    """The active hoppers in plant order, the materials an assignment can name (``UNASSIGNED`` first, then the client
    names of ``materials_map.yml`` active in ``plant_master``), the current mapping (the latest snapshot, as the
    Streamlit tool shows it) and every snapshot newest first, materials as client names."""
    service = _hopper_service(store)
    try:
        history = service.get_hopper_material_history()
        names = _usernames(store)
    except ProgrammingError as exc:
        raise _unavailable(exc) from exc
    rows = [
        {
            "id": int(row["id"]),
            "date_time": utc(row["date_time"]),
            "source_type": row["source_type"],
            "ip_address": row["ip_address"],
            "user_id": row["user_modified"],
            "user_name": names.get(row["user_modified"] or ""),
            "materials": {hopper: row[hopper] for hopper in service.hoppers},
        }
        for row in history
    ]
    return {
        "hoppers": [{"code": h, "display_name": service.hopper_display_names.get(h, h)} for h in service.hoppers],
        "materials": [plant_config.UNASSIGNED, *service.materials],
        "current": rows[0] if rows else None,
        "history": rows,
    }


def assign_hoppers(
    store: TwinStore, actor: Actor, materials: Mapping[str, str], effective_from: datetime
) -> dict[str, Any]:
    """Save the hopper form as the Streamlit tool does: the hoppers whose material differs from the current mapping
    are written as one snapshot effective from ``effective_from`` (a copy of the snapshot then in force with those
    hoppers set), with its audit row ``hopper.assign``. Nothing is written when nothing differs.

    Raises:
        Invalid: a hopper that is not active, a changed hopper whose material is not offered, a time without its zone.
    """
    when = _aware("effective_from", effective_from)
    service = _hopper_service(store)
    unknown = sorted(set(materials) - set(service.hoppers))
    if unknown:
        raise Invalid(f"Unknown or inactive hopper(s): {', '.join(unknown)}")
    try:
        changed = plant_config.hopper_changes(materials, service.get_current_hopper_materials())
        # Only the changed hoppers are checked: a hopper left at a material in force that is no longer offered is kept.
        offered = {plant_config.UNASSIGNED, *service.materials}
        refused = sorted({m for m in changed.values() if m not in offered})
        if refused:
            raise Invalid(f"Material(s) not offered: {', '.join(refused)}")
        if not changed:
            return {"written": False, "id": None, "changed": [], "audit_id": None}
        try:
            codes = service.material_codes(changed)
        except ValueError as exc:  # a material inactive in plant_master
            raise Invalid(str(exc)) from exc
        with audit.admin_write(
            store, actor, "hopper.assign", "hopper_mapping", details={"effective_from": when, "changed": changed}
        ) as write:
            base, row = HopperHistoryRepository.add_snapshot(
                write.session,
                hopper_material_codes=codes,
                from_time=when,
                user_id=UserRepository.user_id_in(write.session, actor.name),
                ip_address=_ip(actor),
            )
            write.object_id = str(row.id)
            write.details["based_on"] = None if base is None else int(base.id)
            write.before, write.after = _hopper_audit(base), _hopper_audit(row)
    except ProgrammingError as exc:
        raise _unavailable(exc) from exc
    return {"written": True, "id": int(row.id), "changed": list(changed), "audit_id": int(write.entry.id)}


def delete_hopper_history(store: TwinStore, actor: Actor, ids: Iterable[int]) -> dict[str, Any]:
    """Delete hopper snapshots as the Streamlit tool does (one statement), with one audit row ``hopper.delete`` that
    keeps every deleted row.

    Raises:
        Invalid: no id. NotFound: an id that does not exist (nothing is deleted).
    """
    wanted = _unique_ids(ids)
    try:
        with audit.admin_write(store, actor, "hopper.delete", "hopper_mapping", _ids_text(wanted)) as write:
            rows = HopperHistoryRepository.delete_rows(write.session, wanted)
            missing = sorted(set(wanted) - {int(r.id) for r in rows})
            if missing:
                raise NotFound(f"hopper history row(s) {', '.join(map(str, missing))}")
            write.before = {str(r.id): _hopper_audit(r) for r in rows}
            write.after = None
    except ProgrammingError as exc:
        raise _unavailable(exc) from exc
    return {"deleted": wanted, "audit_id": int(write.entry.id)}


# -------------------------------------------------------------------------------------------------------- burden

_POSITION = re.compile(r"^(coke|noncoke)_p(\d{2})_(rings|angles)$")


def burden_fields() -> list[dict[str, Any]]:
    """The burden-distribution fields in table order with their kind, unit and place in the charge matrix."""
    out: list[dict[str, Any]] = []
    for name in plant_config.BURDEN_FIELDS:
        match = _POSITION.match(name)
        if match:
            charge, position, quantity = match.group(1), int(match.group(2)), match.group(3)
            out.append(
                {
                    "name": name,
                    "label": f"{'Coke' if charge == 'coke' else 'Non-coke'} P{position:02d} {quantity}",
                    "kind": "number",
                    "unit": "°" if quantity == "angles" else "rotations",
                    "charge": "coke" if charge == "coke" else "non_coke",
                    "position": position,
                    "quantity": quantity,
                }
            )
            continue
        text = name in plant_config.BURDEN_TEXT_FIELDS
        charge = "non_coke" if name.startswith("non_coke") else "coke" if name.startswith("coke") else None
        out.append(
            {
                "name": name,
                "label": name.replace("_", " ").capitalize(),
                "kind": "text" if text else "number",
                "unit": None if text else "s",
                "charge": charge,
                "position": None,
                "quantity": None,
            }
        )
    return out


def _burden_values(row: BurdenHistory | Mapping[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return {name: None for name in plant_config.BURDEN_FIELDS}
    get = row.get if isinstance(row, Mapping) else lambda name: getattr(row, name)
    return {name: get(name) for name in plant_config.BURDEN_FIELDS}


def _burden_audit(row: BurdenHistory | None) -> dict[str, Any] | None:
    return None if row is None else {"date_time": row.date_time, **_burden_values(row)}


def burden_view(store: TwinStore, now: datetime | None = None) -> dict[str, Any]:
    """The fields, the snapshot in force at ``now`` (as the Streamlit tool shows it) and every snapshot newest first.
    Numbers: rings in chute rotations, angles in °, discharge times in s."""
    at = _now(now)
    repository = BurdenHistoryRepository(store.sessions)
    try:
        history = repository.get_burden_history()
        names = _usernames(store)
    except ProgrammingError as exc:
        raise _unavailable(exc) from exc
    rows = [
        {
            "id": int(row["id"]),
            "date_time": utc(row["date_time"]),
            "source_type": row["source_type"],
            "ip_address": row["ip_address"],
            "user_id": row["user_modified"],
            "user_name": names.get(row["user_modified"] or ""),
            "values": _burden_values(row),
        }
        for row in history
    ]
    current = next((row for row in rows if row["date_time"] <= at), None)
    return {"fields": burden_fields(), "at": at, "current": current, "history": rows}


def update_burden(
    store: TwinStore,
    actor: Actor,
    values: Mapping[str, Any],
    effective_from: datetime,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Save the burden form as the Streamlit tool does: the fields whose value differs from the snapshot in force at
    ``now`` are written as one snapshot effective from ``effective_from`` (a copy of the snapshot then in force with
    those fields set), with its audit row ``burden.update``. Nothing is written when nothing differs.

    Raises:
        Invalid: an unknown field, a number field without a number, a time without its zone.
    """
    when = _aware("effective_from", effective_from)
    at = _now(now)
    try:
        current = BurdenHistoryRepository(store.sessions).get_all_current_burden_values(at)
        try:
            changed = plant_config.burden_changes(values, current)
        except ValueError as exc:
            raise Invalid(str(exc)) from exc
        if not changed:
            return {"written": False, "id": None, "changed": [], "audit_id": None}
        with audit.admin_write(
            store, actor, "burden.update", "burden_distribution", details={"effective_from": when}
        ) as write:
            base, row = BurdenHistoryRepository.add_snapshot(
                write.session,
                row_values=changed,
                timestamp=when,
                user_id=UserRepository.user_id_in(write.session, actor.name),
                ip=_ip(actor),
            )
            write.object_id = str(row.id)
            write.details["changed"] = list(changed)
            write.details["based_on"] = None if base is None else int(base.id)
            write.before, write.after = _burden_audit(base), _burden_audit(row)
    except ProgrammingError as exc:
        raise _unavailable(exc) from exc
    return {"written": True, "id": int(row.id), "changed": list(changed), "audit_id": int(write.entry.id)}


def delete_burden_history(store: TwinStore, actor: Actor, ids: Iterable[int]) -> dict[str, Any]:
    """Delete burden snapshots as the Streamlit tool does (one statement), with one audit row ``burden.delete`` that
    keeps every deleted row.

    Raises:
        Invalid: no id. NotFound: an id that does not exist (nothing is deleted).
    """
    wanted = _unique_ids(ids)
    try:
        with audit.admin_write(store, actor, "burden.delete", "burden_distribution", _ids_text(wanted)) as write:
            rows = BurdenHistoryRepository.delete_rows(write.session, wanted)
            missing = sorted(set(wanted) - {int(r.id) for r in rows})
            if missing:
                raise NotFound(f"burden history row(s) {', '.join(map(str, missing))}")
            write.before = {str(r.id): _burden_audit(r) for r in rows}
            write.after = None
    except ProgrammingError as exc:
        raise _unavailable(exc) from exc
    return {"deleted": wanted, "audit_id": int(write.entry.id)}


__all__ = [
    "ROLES",
    "UNKNOWN_IP",
    "Conflict",
    "Invalid",
    "NotFound",
    "Unavailable",
    "assign_hoppers",
    "burden_fields",
    "burden_view",
    "delete_burden_history",
    "delete_hopper_history",
    "hopper_view",
    "register_user",
    "update_burden",
    "users_view",
]
