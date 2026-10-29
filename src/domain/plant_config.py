"""Plant configuration edits shared by the Streamlit admin tools and the twin-api (SV-4.7, #310).

The Welcome page's tools (``ui.user_management``, ``ui.hopper_admin_page``, ``ui.burden_admin_page``) and the admin
views of the SPA (``services.plant_admin``) submit the same forms: a new account, the material of every hopper, the
burden-distribution fields. These functions turn a submitted form into the write both make, so that the two write the
same rows (``furnace_data.relational`` repositories, wide SCD Type-2 snapshots). Pure: no database, no Streamlit.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from furnace_data.relational import BURDEN_VALUE_COLUMNS, BurdenHistoryRepository

#: Hopper form value of a hopper without material (stored as NULL).
UNASSIGNED = "UNASSIGNED"
#: Burden fields entered as text; every other burden field is a number.
BURDEN_TEXT_FIELDS: frozenset[str] = BurdenHistoryRepository.TEXT_FIELDS
BURDEN_FIELDS: tuple[str, ...] = tuple(BURDEN_VALUE_COLUMNS)


class EmptyField(ValueError):
    """A required form field is empty."""


def registration(username: str, password: str) -> tuple[str, str]:
    """Username and password of a new account as the register form stores them: stripped, both required.

    Raises:
        EmptyField: either is empty after stripping.
    """
    name, secret = (username or "").strip(), (password or "").strip()
    if not name or not secret:
        raise EmptyField("Please fill in all fields.")
    return name, secret


def hopper_changes(form: Mapping[str, str], current: Mapping[str, str]) -> dict[str, str]:
    """Hoppers whose submitted material differs from the current mapping (client material names; a hopper missing
    from ``current`` counts as ``UNASSIGNED``), in form order."""
    return {hopper: material for hopper, material in form.items() if material != current.get(hopper, UNASSIGNED)}


def _stored(field: str, value: Any) -> Any:
    if field in BURDEN_TEXT_FIELDS:
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{field}: {value!r} is not text")
        return BurdenHistoryRepository.stored_value(field, value)
    if isinstance(value, bool):
        raise ValueError(f"{field}: {value!r} is not a number")
    try:
        return BurdenHistoryRepository.stored_value(field, value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field}: {value!r} is not a number") from exc


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def burden_changes(form: Mapping[str, Any], current: Mapping[str, Any]) -> dict[str, Any]:
    """Burden fields whose submitted value would be stored differently from the current snapshot, in form order.

    Values compare as the snapshot stores them (``BurdenHistoryRepository.stored_value``): an empty entry is NULL,
    text fields are text, the others floats, so ``"12"`` equals ``12.0`` and an empty text field equals NULL. The
    submitted values are returned unchanged; the repository converts them on the write.

    Raises:
        ValueError: an unknown field, a number field whose value is not a number, a text field whose value is not
            text.
    """
    unknown = sorted(set(form) - set(BURDEN_FIELDS))
    if unknown:
        raise ValueError(f"Invalid burden field(s): {unknown}")
    changed: dict[str, Any] = {}
    for field, value in form.items():
        old = current.get(field)
        if not _same(_stored(field, value), None if old is None else _stored(field, old)):
            changed[field] = value
    return changed


__all__ = [
    "BURDEN_FIELDS",
    "BURDEN_TEXT_FIELDS",
    "UNASSIGNED",
    "EmptyField",
    "burden_changes",
    "hopper_changes",
    "registration",
]
