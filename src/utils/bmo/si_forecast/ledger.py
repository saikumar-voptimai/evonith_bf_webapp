"""Append-only production Si forecasts and their raw-cast outcomes."""

from __future__ import annotations

import json
import math
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_STORAGE_DIR = Path(__file__).resolve().parents[3] / "storage/bmo_si_forecast"
_LOCK = threading.Lock()


def _jsonable(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return _jsonable(value.item())
    return value


def _stamp(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")


def _read_unlocked(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _append_unlocked(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(_jsonable(record), ensure_ascii=False) + "\n")


class SiliconForecastLedger:
    """The first production result at an origin is immutable."""

    def __init__(self, storage_dir: str | Path = DEFAULT_STORAGE_DIR) -> None:
        self.path = Path(storage_dir) / "issued_forecasts.jsonl"

    def _issues_from(self, records: list[dict[str, Any]]) -> dict[pd.Timestamp, dict[str, Any]]:
        out: dict[pd.Timestamp, dict[str, Any]] = {}
        for record in records:
            if record.get("type") == "issue" and record.get("origin_at"):
                out.setdefault(_stamp(record["origin_at"]), record)
        return dict(sorted(out.items()))

    def issues(self) -> dict[pd.Timestamp, dict[str, Any]]:
        with _LOCK:
            return self._issues_from(_read_unlocked(self.path))

    def get(self, origin_at: Any) -> dict[str, Any] | None:
        return self.issues().get(_stamp(origin_at))

    def record_issue(self, record: dict[str, Any]) -> dict[str, Any]:
        """Append once; concurrent reruns still return the same first record."""

        origin = _stamp(record["origin_at"])
        with _LOCK:
            records = _read_unlocked(self.path)
            existing = self._issues_from(records).get(origin)
            if existing is not None:
                return existing
            stored = _jsonable({"type": "issue", **record})
            _append_unlocked(self.path, stored)
            return stored

    def record_outcome(
        self,
        *,
        origin_at: Any,
        horizon_minutes: int = 120,
        sample_id: str,
        lab_sample_id: str,
        sample_at: Any,
        available_at: Any,
        actual_si_pct: float,
        recorded_at: Any,
    ) -> bool:
        """Append one raw sample match. Multiple samples in a band stay separate."""

        key = (
            _stamp(origin_at).isoformat(),
            int(horizon_minutes),
            str(sample_id),
            _stamp(sample_at).isoformat(),
        )
        with _LOCK:
            records = _read_unlocked(self.path)
            known = {
                (
                    _stamp(row["origin_at"]).isoformat(),
                    int(row.get("horizon_minutes", 120)),
                    str(row.get("sample_id", "")),
                    _stamp(row["sample_at"]).isoformat(),
                )
                for row in records
                if row.get("type") == "outcome"
            }
            if key in known:
                return False
            _append_unlocked(
                self.path,
                {
                    "type": "outcome",
                    "origin_at": origin_at,
                    "horizon_minutes": int(horizon_minutes),
                    "sample_id": sample_id,
                    "lab_sample_id": lab_sample_id,
                    "sample_at": sample_at,
                    "available_at": available_at,
                    "actual_si_pct": float(actual_si_pct),
                    "recorded_at": recorded_at,
                },
            )
            return True

    def outcomes(self) -> list[dict[str, Any]]:
        with _LOCK:
            return [
                row for row in _read_unlocked(self.path) if row.get("type") == "outcome"
            ]

    def scored_frame(self) -> pd.DataFrame:
        """One row per horizon/raw-cast match, joined to as-issued values."""

        issues = self.issues()
        rows: list[dict[str, Any]] = []
        for outcome in self.outcomes():
            origin = _stamp(outcome["origin_at"])
            issue = issues.get(origin)
            if not issue:
                continue
            horizon_minutes = int(outcome.get("horizon_minutes", 120))
            paths = issue.get("horizons") or []
            if paths:
                horizon = next(
                    (
                        row
                        for row in paths
                        if int(row.get("horizon_minutes", -1)) == horizon_minutes
                    ),
                    None,
                )
            else:
                interval = issue.get("interval") or {}
                horizon = {
                    "si_pct": issue.get("si_pct"),
                    "lower_pct": (
                        interval.get("lower_pct") if isinstance(interval, dict) else None
                    ),
                    "upper_pct": (
                        interval.get("upper_pct") if isinstance(interval, dict) else None
                    ),
                    "target_start": issue.get("target_start"),
                    "target_end": issue.get("target_end"),
                }
            if not horizon or horizon.get("si_pct") is None:
                continue
            actual = float(outcome["actual_si_pct"])
            lower = horizon.get("lower_pct")
            upper = horizon.get("upper_pct")
            rows.append(
                {
                    "origin_at": origin,
                    "sample_at": _stamp(outcome["sample_at"]),
                    "sample_id": outcome.get("sample_id"),
                    "horizon_minutes": horizon_minutes,
                    "target_start": _stamp(horizon["target_start"]),
                    "target_end": _stamp(horizon["target_end"]),
                    "actual": actual,
                    "prediction": float(horizon["si_pct"]),
                    "persistence": issue.get("last_sample_si_pct"),
                    "lower": lower,
                    "upper": upper,
                    "inside_range": (
                        bool(float(lower) <= actual <= float(upper))
                        if lower is not None and upper is not None
                        else None
                    ),
                    "status": issue.get("status"),
                    "model_version": issue.get("model_version"),
                }
            )
        return pd.DataFrame(rows)

    def forecast_frame(self) -> pd.DataFrame:
        """Flatten every immutable issued path for trend rendering."""

        rows: list[dict[str, Any]] = []
        for origin, issue in self.issues().items():
            paths = issue.get("horizons") or []
            if not paths and issue.get("target_start"):
                paths = [
                    {
                        "horizon_minutes": 120,
                        "target_start": issue.get("target_start"),
                        "target_end": issue.get("target_end"),
                        "si_pct": issue.get("si_pct"),
                        "lower_pct": None,
                        "upper_pct": None,
                        "legacy": True,
                    }
                ]
            for horizon in paths:
                rows.append(
                    {
                        "origin_at": origin,
                        "issued_at": _stamp(issue.get("issued_at", origin)),
                        "horizon_minutes": int(horizon.get("horizon_minutes", 120)),
                        "target_start": _stamp(horizon["target_start"]),
                        "target_end": _stamp(horizon["target_end"]),
                        "prediction": horizon.get("si_pct"),
                        "lower": horizon.get("lower_pct"),
                        "upper": horizon.get("upper_pct"),
                        "status": issue.get("status"),
                        "model_version": issue.get("model_version"),
                        "last_sample_si_pct": issue.get("last_sample_si_pct"),
                    }
                )
        return pd.DataFrame(rows)

    def actual_frame(self) -> pd.DataFrame:
        """Unique irregular raw Si samples already matched by the ledger."""

        rows: dict[tuple[str, str], dict[str, Any]] = {}
        for outcome in self.outcomes():
            key = (
                str(outcome.get("sample_id", "")),
                _stamp(outcome["sample_at"]).isoformat(),
            )
            rows.setdefault(
                key,
                {
                    "sample_id": outcome.get("sample_id"),
                    "sample_at": _stamp(outcome["sample_at"]),
                    "actual": float(outcome["actual_si_pct"]),
                    "available_at": _stamp(outcome["available_at"]),
                },
            )
        return pd.DataFrame(rows.values())


__all__ = ["DEFAULT_STORAGE_DIR", "SiliconForecastLedger"]
