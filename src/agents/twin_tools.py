"""FurnaceMind tools on the live twin (SV-5.3, #314): what the twin API serves, read through the same services.

* ``get_twin_snapshot`` - the latest evaluation (GET /api/twin/now, ``services.twin_reads.now``) or one window
  (GET /api/twin/window): coke rate burnt (TRACE), modelled (FORGE, STRATA ± 1σ) and charged, the coke-rate gaps
  FORGE − TRACE and STRATA − TRACE with their RMSE over the alarm days, shaft efficiencies, freshness, open alarms and
  the data age of each source;
* ``get_twin_series`` - hourly or per-shift windows (GET /api/twin/series, ``services.twin_reads.series``), also kept
  as a dataset of the tools' state for ``execute_python_plot``;
* ``get_transit`` - the burden column (GET /api/twin/transit, ``services.transit.view``);
* ``get_optimiser_result`` - the latest plan or one run (GET /api/optimiser/latest and /runs/{id}/result,
  ``services.twin_reads.run_summary`` and ``services.optimiser_results.result_view``).

Results are JSON for the model: times ISO 8601 in IST (+05:30), units in the field names or in ``units``, and a
``cite`` line that names the window and its evaluation time (the answer quotes it). Time arguments are ISO 8601; a
time without offset is IST. The store is the process's twin store (``services.store``): the API's in the agent
service, the one on ``BF_TWIN_DB_URL`` / ``DATABASE_URL`` on page 7. Without a store, an unknown run id or an
unavailable source a tool raises; the agent loop returns the error to the model.
"""

from __future__ import annotations

import json
import math
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

PLANT_TZ = "Asia/Kolkata"
STORE_NAME = "furnacemind"  # application name of the store when this process opens it first
ROWS_INLINE = 48  # newest series rows returned to the model; the dataset keeps every row
MODES = ("step1", "step2", "max_production")
COKE_FIELDS = ("k_trace", "k_forge", "k_strata", "k_strata_sd", "k_charged")
OMEGA_FIELDS = ("omega_imp", "omega_strata", "omega_strata_sd", "omega_cal")

TWIN_UNITS = {
    "k_trace": "burnt coke, TRACE (oxygen balance + 14-day offset), kg/tHM dry",
    "k_forge": "FORGE: two-zone energy balance at the 14-day medians, kg/tHM dry",
    "k_strata": "STRATA: two-zone balance at omega_STRATA (ring model), kg/tHM dry",
    "k_strata_sd": "1 sigma of k_strata, kg/tHM dry",
    "k_charged": "coke charged in the window, kg/tHM dry",
    "omega_imp": "implied shaft efficiency of the window, dimensionless",
    "omega_strata": "STRATA 1.5 shaft efficiency, dimensionless",
    "omega_strata_sd": "1 sigma of omega_strata, dimensionless",
    "omega_cal": "14-day median shaft efficiency (anchor of FORGE and STRATA), dimensionless",
    "forge_trace_kg_thm": "coke-rate gap FORGE - TRACE of the newest window that has both (forge_at), kg/tHM dry",
    "strata_trace_kg_thm": "coke-rate gap STRATA - TRACE of the newest window that has both (strata_at), kg/tHM dry",
    "forge_trace_rmse_kg_thm": "RMSE of FORGE - TRACE over the hourly windows of the last `days`, kg/tHM dry",
    "strata_trace_rmse_kg_thm": "RMSE of STRATA - TRACE over the hourly windows of the last `days`, kg/tHM dry",
    "strata_excess_h": "newest consecutive hourly windows with |STRATA - TRACE| > ratio x RMSE, h",
    "quantities": "other quantities of the window, unit in the key (e.g. hmt_c degC, tau_raceway_h h)",
    "age_s": "time since the evaluation, s",
    "lag_h": "expected window end minus the latest window end, h",
    "times": "ISO 8601, IST (+05:30); a window [end - 8 h, end) is a shift when it ends at 06:00, 14:00 or 22:00",
}
TRANSIT_UNITS = {
    "charges_per_hour": "charging rate of the last hour, charges/h",
    "tau_cz_h / tau_raceway_h": "descent time to the cohesive zone / raceways, h (_rate_h: at the current rate)",
    "coke_in_transit_t": "coke in the column, t",
    "bands": "ledger by zone (above_cz, in_cz, below_cz): charges (count), newest_h / oldest_h (age, h), "
    "coke_t_per_charge (t), coke_kg_per_t_ferrous (kg/t), coke_kg_thm (as charged, kg/tHM), coke_volume_fraction "
    "(dimensionless), coke_inventory_t (t), blend_wet_pct (wet share of the ferrous charge, %), tonnes (t)",
    "lag": "implied transit time charging to burning, h, with the 95 % bootstrap interval, h",
    "blend_change": "latest ferrous blend change in 72 h: step_pt (percentage points), shares (%), arrival times",
    "times": "ISO 8601, IST (+05:30)",
}
OPTIMISER_UNITS = {
    "cost": "Rs/tHM (ore + flux + fuel - slag credit); saving_lakh_rs_d lakh Rs/d",
    "coke": "kg/tHM dry unless named wet; fuel rate kg/tHM; burden and flux kg/tHM",
    "production": "t/d (production_t_h t/h)",
    "process": "temperatures degC, pressures bar g (pressure drop bar), gas flows Nm3/h",
    "times": "ISO 8601, IST (+05:30)",
}
RESULT_SUMMARY = (
    "run_id",
    "mode",
    "omega_mode",
    "status",
    "error",
    "run_s",
    "production_target_t_d",
    "yard_source",
    "certainty",
    "plan",
    "saving_rs_thm",
    "saving_lakh_rs_d",
    "critical_assumptions",
    "critical_assumptions_error",
)


class TwinToolError(RuntimeError):
    """The twin store is not configured (``BF_TWIN_DB_URL`` / ``DATABASE_URL``)."""


# ------------------------------------------------------------------------------------------------------- helpers
def parse_time(text: str | None) -> datetime | None:
    """An ISO 8601 argument as an aware UTC datetime (a time without offset is IST); None when empty."""
    if text is None or not str(text).strip():
        return None
    try:
        ts = pd.Timestamp(str(text).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Not an ISO 8601 time: {text!r}.") from exc
    if pd.isna(ts):
        raise ValueError(f"Not an ISO 8601 time: {text!r}.")
    ts = ts.tz_localize(PLANT_TZ) if ts.tzinfo is None else ts
    return ts.tz_convert("UTC").to_pydatetime()


def _ist(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return (ts.tz_localize("UTC") if ts.tzinfo is None else ts).tz_convert(PLANT_TZ)


def ist(value: Any) -> str | None:
    """A time (naive is UTC, as stored) as ISO 8601 in IST to the second; None stays None."""
    return None if value is None else _ist(value).isoformat(timespec="seconds")


def label(value: Any) -> str:
    """A time as the operator reads it: ``27 Oct 2026 11:00 IST``."""
    return f"{_ist(value):%d %b %Y %H:%M} IST"


def plain(value: Any) -> Any:
    """JSON-ready: times to IST ISO strings, non-finite floats to None, ids to text, containers walked."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, datetime | pd.Timestamp):
        return ist(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [plain(v) for v in value]
    return value


def dump(payload: Mapping[str, Any]) -> str:
    return json.dumps(plain(payload), ensure_ascii=False, default=str)


def _store() -> Any:
    from services import store as store_service

    store = store_service.twin_store(STORE_NAME)
    if store is None:
        raise TwinToolError("The twin store is not configured: set BF_TWIN_DB_URL or DATABASE_URL.")
    return store


def _window_cite(snapshot: Mapping[str, Any]) -> str:
    return (
        f"Twin window {label(snapshot['window_start'])} to {label(snapshot['window_end'])}, "
        f"evaluated {label(snapshot['evaluated_at'])}"
    )


def _snapshot_view(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    from services import twin_views

    return {"shift_key": twin_views.shift_key(snapshot["window_end"]), **snapshot}


# ----------------------------------------------------------------------------------------------------- the tools
def get_twin_snapshot(*, window_end: str | None = None) -> str:
    """The latest twin evaluation (``window_end`` None) or the stored window ending at the IST full hour at or before
    ``window_end``, with its coke-rate gaps."""
    from services import twin_reads, twin_views

    store = _store()
    end = parse_time(window_end)
    warnings: list[str] = []
    if end is None:
        data = twin_reads.now(store)
        snapshot = data["snapshot"]
        payload: dict[str, Any] = {"source": "latest twin evaluation (twin store, as GET /api/twin/now)"}
        payload["freshness"] = {
            k: data[k] for k in ("now", "expected_window_end", "age_s", "lag_h", "stale", "stale_after_h")
        }
        payload["alarms_open"] = data["alarms_open"]
        payload["sources"] = [
            {k: s[k] for k in ("key", "label", "data_end", "age_h", "stale", "down", "provisional")}
            for s in data["sources"]
        ]
        if snapshot is None:
            return dump({**payload, "snapshot": None, "note": "No twin evaluation is stored yet."})
        gaps = data["gaps"]
        if data["stale"]:
            warnings.append(
                f"The twin is stale: the latest window ends {label(snapshot['window_end'])}, "
                f"{data['lag_h']:.0f} h behind the expected {label(data['expected_window_end'])}."
            )
        stale = [s["label"] for s in data["sources"] if s["stale"]]
        if stale:
            warnings.append(f"Sources stale or down: {', '.join(stale)}.")
    else:
        hour = pd.Timestamp(end).tz_convert(PLANT_TZ).floor("h").tz_convert("UTC").to_pydatetime()
        row = store.snapshots.get(hour)
        payload = {"source": "stored twin window (twin store, as GET /api/twin/window)"}
        if row is None:
            return dump(
                {**payload, "snapshot": None, "note": f"No stored evaluation for the window ending {label(hour)}."}
            )
        snapshot = twin_views.snapshot_data(row)
        gaps = twin_reads.gaps(store, hour)
    if snapshot["status"] != "complete":
        warnings.append(f"The evaluation of this window is {snapshot['status']}.")
    for name, at in (("FORGE - TRACE", gaps.get("forge_at")), ("STRATA - TRACE", gaps.get("strata_at"))):
        if at is None:
            warnings.append(f"No {name} gap in the last {gaps['days']:g} days (a model or TRACE is missing).")
        elif pd.Timestamp(at) != pd.Timestamp(snapshot["window_end"]):
            warnings.append(f"The {name} gap is of the window ending {label(at)}, not of this window.")
    return dump(
        {
            "cite": _window_cite(snapshot),
            **payload,
            "snapshot": _snapshot_view(snapshot),
            "gaps": gaps,
            "warnings": warnings,
            "units": TWIN_UNITS,
        }
    )


def _series_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """The series as a DataFrame on an IST index (``time (IST)``, window end), as the other fetch tools give."""
    records = [
        {
            "shift_key": r["shift_key"],
            "status": r["status"],
            **{k: r.get(k) for k in COKE_FIELDS + OMEGA_FIELDS},
            **dict(r.get("quantities") or {}),
        }
        for r in rows
    ]
    index = pd.DatetimeIndex([_ist(r["window_end"]) for r in rows], name="time (IST)")
    return pd.DataFrame.from_records(records, index=index)


def _stats(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    out = {}
    for key in COKE_FIELDS + OMEGA_FIELDS:
        values = pd.to_numeric(frame[key], errors="coerce").dropna()
        if len(values):
            out[key] = {"n": len(values), "mean": float(values.mean()), "min": float(values.min()),
                        "max": float(values.max())}  # fmt: skip
    return out


def get_twin_series(
    *,
    start_time: str | None = None,
    end_time: str | None = None,
    resolution: str = "hour",
    quantities: Sequence[str] | None = None,
) -> str:
    """Stored windows between two window ends (default the last 7 days), hourly or the shift windows."""
    from agents import furnace_tools as ft
    from services import twin_reads

    if resolution not in ("hour", "shift"):
        raise ValueError("resolution must be 'hour' or 'shift'.")
    store = _store()
    keys = [str(k).strip() for k in quantities or [] if str(k).strip()]
    data = twin_reads.series(
        store, start=parse_time(start_time), end=parse_time(end_time), resolution=resolution, quantities=keys
    )
    rows = data["rows"]
    what = "hourly windows" if resolution == "hour" else "shift windows"
    payload: dict[str, Any] = {
        "cite": f"Twin series, {data['count']} {what} ending {label(data['start'])} to {label(data['end'])}",
        "source": "stored twin windows (twin store, as GET /api/twin/series)",
        **{k: data[k] for k in ("resolution", "start", "end", "count")},
    }
    if not rows:
        return dump({**payload, "note": "No stored twin window in this range.", "units": TWIN_UNITS})
    frame = _series_frame(rows)
    dataset_id = ft._new_dataset_id("twin_series")
    meta = {
        "dataset_id": dataset_id,
        "source": "twin store (twin.snapshot)",
        "resolution": resolution,
        "start": ist(data["start"]),
        "end": ist(data["end"]),
        "rows": len(frame),
    }
    ft._save_dataset(dataset_id=dataset_id, df=frame, meta=meta)
    shown = rows[-ROWS_INLINE:]
    return dump(
        {
            **payload,
            "dataset_id": dataset_id,
            "columns": list(frame.columns),
            "stats": _stats(frame),
            "rows": [{k: v for k, v in r.items() if v is not None and v != {}} for r in shown],
            "rows_shown": len(shown),
            "rows_not_shown": len(rows) - len(shown),
            "note": "Rows newest last; every row is in the dataset (df of execute_python_plot)."
            if len(shown) < len(rows)
            else "Rows newest last; the dataset is the df of execute_python_plot.",
            "units": TWIN_UNITS,
        }
    )


def get_transit(*, at: str | None = None, lag: bool = False) -> str:
    """The burden column at ``at`` (default now): descent times, coke in transit, the ledger by zone, the latest
    blend change and, with ``lag``, the implied lag against the stored burnt coke."""
    from services import store as store_service
    from services import transit

    data = transit.view(parse_time(at), lag=bool(lag), store=store_service.twin_store(STORE_NAME) if lag else None)
    return dump(
        {
            "cite": f"Burden column at {label(data['at'])}",
            "source": "hourly furnace dataset (as GET /api/twin/transit)",
            **data,
            "units": TRANSIT_UNITS,
        }
    )


def get_optimiser_result(*, run_id: str | None = None, mode: str | None = None, detail: bool = False) -> str:
    """The latest optimiser run with a plan (of ``mode`` if given) or the run ``run_id``: headline, decision and the
    typed result (with ``detail`` also the comparison plans, frontier, value of limits and calibration)."""
    from services import twin_reads
    from services.optimiser_results import result_view

    if mode is not None and mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}.")
    store = _store()
    if run_id:
        try:
            key = uuid.UUID(str(run_id))
        except ValueError as exc:
            raise ValueError("run_id must be a run id (UUID).") from exc
        row = store.optimiser.get(key)
        if row is None:
            return dump({"run": None, "note": f"No optimiser run with id {run_id}."})
    else:
        row = store.optimiser.latest(mode=mode)
        if row is None:
            return dump(
                {"run": None, "note": "No optimiser run has a plan yet" + (f" in mode {mode}." if mode else ".")}
            )
    run = twin_reads.run_summary(store, row)
    result = result_view(row).model_dump(exclude_none=True)
    if not detail:
        result = {k: v for k, v in result.items() if k in RESULT_SUMMARY}
    finished = f", finished {label(run['finished_at'])}" if run.get("finished_at") is not None else ""
    return dump(
        {
            "cite": f"Optimiser run {run['id']}: {run['mode']}, {run['omega_mode']}-omega, {run['status']}{finished}",
            "source": "optimiser runs (twin store, as GET /api/optimiser/latest and /runs/{id}/result)",
            "run": run,
            "result": result,
            "units": OPTIMISER_UNITS,
        }
    )


TOOLS: dict[str, Callable[..., str]] = {
    "get_twin_snapshot": get_twin_snapshot,
    "get_twin_series": get_twin_series,
    "get_transit": get_transit,
    "get_optimiser_result": get_optimiser_result,
}

#: Progress text of each tool (``agents.furnacemind.graph``).
LABELS = {
    "get_twin_snapshot": "Reading the live twin...",
    "get_twin_series": "Reading the twin series...",
    "get_transit": "Reading the burden column...",
    "get_optimiser_result": "Reading the optimiser plan...",
}

_TIME = "ISO 8601; a time without offset is IST."
SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_twin_snapshot",
            "description": (
                "Live twin of BF-2, evaluated hourly on the 8-h window [t - 8 h, t): coke rate burnt (TRACE) and "
                "modelled (FORGE energy balance, STRATA ring model +- 1 sigma) and charged, kg/tHM dry; the coke-rate "
                "gaps FORGE - TRACE and STRATA - TRACE of the newest window and their 7-day RMSE; shaft efficiencies; "
                "freshness, open alarms and the data age of each source. Use for the current coke rate, the coke-rate "
                "gap, model vs burnt coke, omega or the twin's status. Quote the returned 'cite' (window and "
                "evaluation time, IST) in the answer and mention any warnings."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "window_end": {
                        "type": "string",
                        "description": f"Optional past window end, {_TIME} The window ending at the IST full hour "
                        "at or before it. Omit for the latest evaluation.",
                    }
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_twin_series",
            "description": (
                "Stored twin windows over a range (at most 92 days, default the last 7 days): coke rates TRACE / "
                "FORGE / STRATA / charged (kg/tHM dry) and shaft efficiencies per hourly window, or per shift "
                "(windows ending 06:00, 14:00, 22:00 IST). Returns statistics, the newest 48 rows and a dataset_id; "
                "the dataset becomes df for execute_python_plot (index 'time (IST)' = window end)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {"type": "string", "description": f"First window end, {_TIME} Optional."},
                    "end_time": {"type": "string", "description": f"Last window end, {_TIME} Default now."},
                    "resolution": {
                        "type": "string",
                        "enum": ["hour", "shift"],
                        "description": "hour (default) or shift: the shift windows only.",
                    },
                    "quantities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Extra quantity keys of the windows, unit in the key, e.g. ['hmt_c', "
                        "'tau_raceway_h'] (get_twin_snapshot lists them).",
                    },
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_transit",
            "description": (
                "Burden column of BF-2 from the hourly charging record: descent times to the cohesive zone and the "
                "raceways (h), coke in transit (t), the ledger of charges by zone with coke rate and blend, the "
                "latest ferrous blend change and when it reaches the cohesive zone and the raceways; with lag=true "
                "also the implied transit time against the burnt coke (14 days)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "at": {"type": "string", "description": f"Time of the column, {_TIME} Default now."},
                    "lag": {"type": "boolean", "description": "Also the implied lag (slower). Default false."},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_optimiser_result",
            "description": (
                "Burden & Fuel Optimiser plan from the stored runs: the latest run with a plan (optionally of one "
                "mode) or one run by id. Returns the headline (production t/d, cost Rs/tHM, saving, coke kg/tHM dry, "
                "certainty), the operator decision and the plan: blend, flux, cost and fuel split, slag, charging, "
                "process, shaft efficiency, binding limits, critical assumptions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "run_id": {"type": "string", "description": "Run id (UUID). Omit for the latest plan."},
                    "mode": {
                        "type": "string",
                        "enum": list(MODES),
                        "description": "step1: cheapest production level; step2: fixed production; max_production.",
                    },
                    "detail": {
                        "type": "boolean",
                        "description": "Also the ore + flux and today's plans, the Step 1 frontier, the value of "
                        "limits and the calibration. Default false.",
                    },
                },
                "additionalProperties": False,
            },
        },
    },
]


def dispatch(name: str, arguments: Mapping[str, Any]) -> str:
    """Run the twin tool ``name`` with the model's arguments."""
    return TOOLS[name](**dict(arguments))


__all__ = [
    "LABELS",
    "SCHEMAS",
    "TOOLS",
    "TwinToolError",
    "dispatch",
    "dump",
    "get_optimiser_result",
    "get_transit",
    "get_twin_series",
    "get_twin_snapshot",
    "ist",
    "label",
    "parse_time",
    "plain",
]
