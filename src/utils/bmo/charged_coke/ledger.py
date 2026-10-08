"""Append-only records: forecasts as issued, outcomes, and planned changes.

Issued forecasts
    One ``issue`` event per completed data row, written when the forecast is
    first produced and never rewritten. A later rebuild of the dataset can
    revise history (the publisher re-imputes it), so recomputing an old hour
    would not reproduce what operators saw; the record is the truth. Matured
    outcomes are appended as separate ``outcome`` events.

Planned major changes
    An operator declares a major operating change with an explicit validity
    period. While a declaration is valid, forecasts are paused
    (``planned_change_hold``). Creation and cancellation are both recorded,
    with who and when. Expiry needs no action, and afterwards the ordinary
    data, condition and two-update recovery checks apply as usual.

Both files are JSON lines under ``src/storage/bmo_charged_coke/`` and are only
ever appended to.
"""

from __future__ import annotations

import json
import math
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

# One path string, not a bare folder name: TestBMO renames key-shaped
# ``bmo_...`` strings, and this is a folder, not a session key.
DEFAULT_STORAGE_DIR = Path(__file__).resolve().parents[3] / "storage/bmo_charged_coke"
MAX_PLAN_VALIDITY = pd.Timedelta(hours=24)
_LOCK = threading.Lock()


def _jsonable(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):  # numpy scalar
        return _jsonable(value.item())
    return value


def _append(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(_jsonable(record), ensure_ascii=False)
    with _LOCK, open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


class ForecastLedger:
    """Forecasts as issued, keyed by the data row they were made from."""

    def __init__(self, storage_dir: str | Path = DEFAULT_STORAGE_DIR) -> None:
        self.path = Path(storage_dir) / "issued_forecasts.jsonl"

    def issues(self) -> dict[pd.Timestamp, dict[str, Any]]:
        """First ``issue`` event per data row (later duplicates are ignored)."""

        out: dict[pd.Timestamp, dict[str, Any]] = {}
        for record in _read(self.path):
            if record.get("type") != "issue":
                continue
            row = pd.Timestamp(record["data_row"])
            out.setdefault(row, record)
        return dict(sorted(out.items()))

    def get(self, data_row: pd.Timestamp) -> dict[str, Any] | None:
        return self.issues().get(pd.Timestamp(data_row))

    def record_issue(self, record: dict[str, Any]) -> dict[str, Any]:
        """Write the forecast for a data row unless one was already issued.

        Returns:
            The record that stands for that row: the existing one if the row was
            already issued (a UI refresh never re-issues), else ``record``.
        """

        row = pd.Timestamp(record["data_row"])
        existing = self.get(row)
        if existing is not None:
            return existing
        stored = json.loads(json.dumps(_jsonable({"type": "issue", **record}), ensure_ascii=False))
        _append(self.path, stored)
        # Callers always see the stored form, whether just written or re-read.
        return stored

    def record_outcome(self, data_row: pd.Timestamp, actual: float, *, recorded_at: datetime) -> None:
        """Append the realised block for an issued row; the issue is untouched."""

        _append(self.path, {"type": "outcome", "data_row": pd.Timestamp(data_row),
                            "actual_kg_thm": float(actual), "recorded_at": recorded_at})

    def outcomes(self) -> dict[pd.Timestamp, float]:
        out: dict[pd.Timestamp, float] = {}
        for record in _read(self.path):
            if record.get("type") == "outcome":
                out.setdefault(pd.Timestamp(record["data_row"]), record["actual_kg_thm"])
        return out

    def frame(self) -> pd.DataFrame:
        """Issued forecasts with any matured outcome, one row per data row."""

        issues = self.issues()
        if not issues:
            return pd.DataFrame()
        frame = pd.DataFrame.from_records(list(issues.values())).set_index("data_row")
        frame.index = pd.to_datetime(frame.index)
        outcomes = self.outcomes()
        frame["actual_kg_thm"] = [outcomes.get(t) for t in frame.index]
        return frame


@dataclass(frozen=True)
class PlannedChange:
    """One declaration of a planned major operating change."""

    id: str
    reason: str
    valid_from: pd.Timestamp
    valid_until: pd.Timestamp
    created_by: str
    created_at: pd.Timestamp
    cancelled_at: pd.Timestamp | None = None
    cancelled_by: str = ""

    def active_at(self, moment: pd.Timestamp) -> bool:
        moment = pd.Timestamp(moment)
        if self.cancelled_at is not None and moment >= self.cancelled_at:
            return False
        return self.valid_from <= moment < self.valid_until


class PlannedChangeRegister:
    """Declared major changes with explicit validity and an audit trail."""

    def __init__(self, storage_dir: str | Path = DEFAULT_STORAGE_DIR) -> None:
        self.path = Path(storage_dir) / "planned_changes.jsonl"

    def declare(
        self,
        *,
        reason: str,
        valid_from: pd.Timestamp,
        valid_until: pd.Timestamp,
        created_by: str,
        now: pd.Timestamp,
    ) -> PlannedChange:
        """Record a declaration. Its validity must be explicit and bounded."""

        valid_from, valid_until = pd.Timestamp(valid_from), pd.Timestamp(valid_until)
        if not str(reason).strip():
            raise ValueError("A planned change needs a reason.")
        if not str(created_by).strip():
            raise ValueError("A planned change needs the name of whoever declares it.")
        if valid_until <= valid_from:
            raise ValueError("The validity must end after it starts.")
        if valid_until - valid_from > MAX_PLAN_VALIDITY:
            raise ValueError(f"A declaration can be valid for at most {MAX_PLAN_VALIDITY}.")
        change = PlannedChange(
            id=uuid.uuid4().hex[:12], reason=str(reason).strip(), valid_from=valid_from,
            valid_until=valid_until, created_by=str(created_by).strip(), created_at=pd.Timestamp(now),
        )
        _append(self.path, {"event": "declare", "id": change.id, "reason": change.reason,
                            "valid_from": valid_from, "valid_until": valid_until,
                            "created_by": change.created_by, "created_at": change.created_at})
        return change

    def cancel(self, change_id: str, *, cancelled_by: str, now: pd.Timestamp, note: str = "") -> None:
        if change_id not in {c.id for c in self.all()}:
            raise KeyError(change_id)
        _append(self.path, {"event": "cancel", "id": change_id, "cancelled_by": cancelled_by,
                            "cancelled_at": pd.Timestamp(now), "note": note})

    def all(self) -> list[PlannedChange]:
        changes: dict[str, PlannedChange] = {}
        for event in _read(self.path):
            if event.get("event") == "declare":
                changes[event["id"]] = PlannedChange(
                    id=event["id"], reason=event["reason"],
                    valid_from=pd.Timestamp(event["valid_from"]), valid_until=pd.Timestamp(event["valid_until"]),
                    created_by=event["created_by"], created_at=pd.Timestamp(event["created_at"]),
                )
            elif event.get("event") == "cancel" and event["id"] in changes:
                old = changes[event["id"]]
                changes[event["id"]] = PlannedChange(**{
                    **old.__dict__, "cancelled_at": pd.Timestamp(event["cancelled_at"]),
                    "cancelled_by": event["cancelled_by"],
                })
        return list(changes.values())

    def active(self, moment: pd.Timestamp) -> list[PlannedChange]:
        return [c for c in self.all() if c.active_at(moment)]

    def flags(self, issue_times: Iterable[pd.Timestamp]) -> dict[pd.Timestamp, bool]:
        """Whether any declaration was active at each issue time."""

        changes = self.all()
        return {pd.Timestamp(t): any(c.active_at(t) for c in changes) for t in issue_times}


__all__ = [
    "DEFAULT_STORAGE_DIR",
    "ForecastLedger",
    "MAX_PLAN_VALIDITY",
    "PlannedChange",
    "PlannedChangeRegister",
]
