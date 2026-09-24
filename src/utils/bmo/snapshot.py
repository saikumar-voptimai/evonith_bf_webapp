"""A complete, replayable record of one Blend Mix Optimiser run.

WHAT A SNAPSHOT IS.

Every input the page ran with, every result it produced, and the plant data the
run read - written as one JSON document with a timestamp. It is a record, not a
screenshot: the report is generated from it, and Sandbox mode replays it.

    inputs        widget values and the effective input tables    restored
    results       LP / DE results, Si, manual blend                restored for display
    frozen        plant data the run read (``utils/bmo/replay.py``)  replayed
    context       derived values that explain the report          report only

HOW IT DECIDES WHAT IS AN INPUT.

By discovery, not by a fixed schema. The page keeps its state under ``bmo_*``
session keys and the set differs between branches - UAT has a PCI override, an
energy anchor and a commentary that main does not. So every key present is
classified (``classify_suffix``) as ui / internal / result / input.

WHICH MOMENT IT RECORDS.

The moment of the last RUN, not the moment the button is pressed. Plant data
refreshes hourly and the operator may edit inputs after running; a snapshot of
"now" could pair results with inputs and data that never produced them. The
replay module pins the run's inputs and data when the results appear, and the
snapshot saves that pinned record. With no run yet, it records the page as it
stands (inputs only).

EXACTNESS. Floats are written with Python's shortest round-trip repr, NaN and
infinity are tagged rather than dropped, and DataFrames keep index, dtypes and
column order - so a replayed run sees bit-identical numbers.

Keys are stored WITHOUT their prefix, so one snapshot restores into either the
live page or the sandbox.
"""

from __future__ import annotations

import dataclasses
import enum
import importlib
import json
import math
import subprocess
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from utils.bmo.sandbox import (
    LIVE_PREFIX,
    SANDBOX_PREFIX,
    UI_SUFFIX,
    is_writable_suffix,
)

SCHEMA = "bmo-snapshot/2"
IST = ZoneInfo("Asia/Kolkata")

SOURCE_BY_PREFIX = {LIVE_PREFIX: "Blend Mix Optimiser", SANDBOX_PREFIX: "Sandbox"}

# Page variable holding the EFFECTIVE table -> the session key the page reads it
# back from. Restoring into the applied key makes the page use it as its source.
PAGE_INPUT_TABLES = {
    "edited_df": "applied_ore_editor_df",
    "edited_flux_df": "applied_flux_editor_df",
    "edited_fuel_ash_df": "applied_fuel_ash_editor_df",
    "edited_dust_df": "applied_dust_editor_df",
}

# Derived on the page, not restorable, but needed to read the report correctly.
PAGE_CONTEXT_VARS = (
    "hm_chem_values", "target_fe_mt", "target_slag_qty_mt", "observed_slag_rate",
    "feo_in_slag_pct", "max_burden_qty_mt", "model_to_plant_slag_factor",
    "hm_snapshot", "slag_settings_values", "de_seed_choice",
    "target_slag_basicity_min", "target_slag_basicity_max",
    "target_slag_t_basicity_min", "target_slag_t_basicity_max",
    "target_slag_al2o3_max_pct", "target_slag_mgo_min_pct",
    "target_slag_mgo_al2o3_ratio_min", "target_slag_rate_kg_per_thm",
    "recent_fuel_rates", "chemistry_mode", "chemistry_window_days",
)

RESULT_SUFFIXES = frozenset({
    "lp_result", "de_result", "lp_si", "de_si", "manual_si", "lp_errors",
    "de_errors", "manual_quantities_mt", "manual_blend", "energy_anchor",
    "commentary", "commentary_context",
})
# Counters, caches and bulky intermediates that say nothing about the state.
SKIPPED_SUFFIXES = frozenset({
    "source_cache_version", "diagnostics_loaded", "bundle_status",
    "pci_state_last", "calibration_bust", "de_candidates",
})

# Frames longer than this inside the frozen plant data are the process history
# (thousands of hourly rows); only the rows the models can reach are kept.
COMPACT_OVER_ROWS = 400
HISTORY_TAIL_ROWS = 96       # every column, last 4 days: rolling windows, lags
HISTORY_PER_COLUMN = 24      # plus each column's last 24 non-null values

# Only our own dataclasses are rebuilt from JSON - an uploaded file must not be
# able to name an arbitrary class to construct.
_TYPED_MODULE_PREFIXES = ("utils.bmo.", "data.bmo.", "domain.")


def classify_suffix(suffix: str) -> str:
    if suffix.startswith(UI_SUFFIX):
        return "ui"
    if suffix in SKIPPED_SUFFIXES:
        return "skip"
    if suffix in RESULT_SUFFIXES:
        return "results"
    return "inputs"


# --- JSON encoding ---------------------------------------------------------------


def _float(value: float) -> Any:
    number = float(value)
    if math.isfinite(number):
        return number
    return {"__type__": "float", "v": "nan" if math.isnan(number) else ("inf" if number > 0 else "-inf")}


def _encode_index(index: pd.Index) -> dict[str, Any]:
    if isinstance(index, pd.RangeIndex):
        return {"kind": "range", "start": index.start, "stop": index.stop, "step": index.step, "name": encode(index.name)}
    if isinstance(index, pd.DatetimeIndex):
        return {
            "kind": "datetime",
            "tz": str(index.tz) if index.tz is not None else None,
            "freq": index.freqstr,
            "values": [None if pd.isna(v) else v.isoformat() for v in index],
            "name": index.name,
        }
    return {"kind": "values", "values": [encode(v) for v in index], "name": encode(index.name)}


def _decode_index(spec: Mapping[str, Any]) -> pd.Index:
    kind = spec.get("kind")
    if kind == "range":
        return pd.RangeIndex(
            spec["start"], spec["stop"], spec["step"], name=decode(spec.get("name"))
        )
    if kind == "datetime":
        idx = pd.DatetimeIndex(pd.to_datetime(spec["values"], utc=True, format="ISO8601"))
        idx = idx.tz_convert(spec["tz"]) if spec.get("tz") else idx.tz_localize(None)
        freq = spec.get("freq")
        if freq:
            try:
                idx = pd.DatetimeIndex(idx, freq=freq)
            except ValueError:
                pass
        return idx.rename(spec.get("name"))
    return pd.Index([decode(v) for v in spec.get("values", [])], name=decode(spec.get("name")))


def _encode_column(series: pd.Series) -> list[Any]:
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return [None if pd.isna(v) else pd.Timestamp(v).isoformat() for v in series]
    if pd.api.types.is_float_dtype(series.dtype):
        return [None if pd.isna(v) else _float(v) for v in series.to_numpy()]
    if pd.api.types.is_integer_dtype(series.dtype) or pd.api.types.is_bool_dtype(series.dtype):
        return [None if pd.isna(v) else v.item() if hasattr(v, "item") else v for v in series]
    return [encode(v) for v in series]


def _decode_column(values: list[Any], dtype: str) -> pd.Series:
    if dtype.startswith("datetime64"):
        series = pd.Series(pd.to_datetime(values, utc="," in dtype, format="ISO8601"))
    elif dtype.startswith("float"):
        series = pd.Series([np.nan if v is None else decode(v) for v in values], dtype="float64")
    else:
        series = pd.Series([decode(v) for v in values], dtype="object")
    try:
        return series.astype(dtype)
    except (TypeError, ValueError):
        return series


def _encode_frame(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "__type__": "frame",
        "columns": [encode(c) for c in frame.columns],
        "dtypes": [str(t) for t in frame.dtypes],
        "index": _encode_index(frame.index),
        "data": [_encode_column(frame.iloc[:, i]) for i in range(frame.shape[1])],
    }


def _decode_frame(doc: Mapping[str, Any]) -> pd.DataFrame:
    columns = [decode(c) for c in doc["columns"]]
    index = _decode_index(doc["index"])
    data = {
        i: _decode_column(values, dtype).to_numpy()
        for i, (values, dtype) in enumerate(zip(doc["data"], doc["dtypes"]))
    }
    frame = pd.DataFrame(data, index=index)
    frame.columns = pd.Index(columns)
    for i, dtype in enumerate(doc["dtypes"]):
        try:
            if str(frame.iloc[:, i].dtype) != dtype:
                frame.isetitem(i, frame.iloc[:, i].astype(dtype))
        except (TypeError, ValueError):
            pass
    return frame


def compact_frame(
    frame: pd.DataFrame,
    *,
    tail_rows: int = HISTORY_TAIL_ROWS,
    per_column: int = HISTORY_PER_COLUMN,
) -> dict[str, Any]:
    """Keep only the rows a model can reach, and the frame's true length.

    Kept, as WHOLE rows: the last ``tail_rows`` rows, and every row holding one
    of a column's last ``per_column`` non-null values (lab columns are sparse).
    Whole rows matter: derived features such as ``PCI_CALC_THM`` divide one
    column by another row by row, so a value kept without its row partners is
    useless. The row COUNT is kept too - the fuel model's ``trend_index`` is
    ``len(history) - 1``. Every other row is rebuilt empty on replay.
    """

    n = len(frame)
    tail_start = max(0, n - tail_rows)
    rows: set[int] = set(range(tail_start, n))
    for col_pos in range(frame.shape[1]):
        notna = np.flatnonzero(frame.iloc[:, col_pos].notna().to_numpy())
        rows.update(int(p) for p in notna[-per_column:])
    positions = sorted(rows)
    return {
        "__type__": "compact_frame",
        "n_rows": n,
        "positions": positions,
        "rows": _encode_frame(frame.iloc[positions]),
    }


def _decode_compact_frame(doc: Mapping[str, Any]) -> pd.DataFrame:
    n = int(doc["n_rows"])
    positions = np.asarray(doc["positions"], dtype=int)
    kept = _decode_frame(doc["rows"])
    data = {}
    for col_pos in range(kept.shape[1]):
        column = kept.iloc[:, col_pos]
        if pd.api.types.is_numeric_dtype(column.dtype) and not pd.api.types.is_bool_dtype(column.dtype):
            values = np.full(n, np.nan)
            values[positions] = column.to_numpy(dtype=float, na_value=np.nan)
            if pd.api.types.is_float_dtype(column.dtype):
                values = values.astype(column.dtype)
        elif pd.api.types.is_datetime64_any_dtype(column.dtype):
            values = pd.Series(pd.NaT, index=range(n), dtype=column.dtype)
            values.iloc[positions] = column.to_numpy()
            values = values.to_numpy()
        else:
            values = np.full(n, None, dtype=object)
            values[positions] = column.to_numpy(dtype=object)
        data[col_pos] = values
    index = kept.index
    if isinstance(index, pd.DatetimeIndex):
        full_index = pd.Series(pd.NaT, index=range(n), dtype=index.dtype)
        full_index.iloc[positions] = index
        new_index = pd.DatetimeIndex(full_index, name=index.name)
    else:
        labels = np.full(n, None, dtype=object)
        labels[positions] = index.to_numpy(dtype=object)
        new_index = pd.Index(labels, name=index.name)
    frame = pd.DataFrame(data, index=new_index)
    frame.columns = kept.columns
    return frame


def verify_compact(full: pd.DataFrame, rebuilt: pd.DataFrame, per_column: int) -> list[str]:
    """Columns whose reachable values differ between the full and rebuilt frame."""

    bad = []
    if len(full) != len(rebuilt):
        return ["<row count>"]
    if len(full) and not (pd.isna(full.index[-1]) and pd.isna(rebuilt.index[-1])) \
            and full.index[-1] != rebuilt.index[-1]:
        bad.append("<last timestamp>")
    for col_pos in range(full.shape[1]):
        a = full.iloc[:, col_pos].dropna().tail(per_column)
        b = rebuilt.iloc[:, col_pos].dropna().tail(per_column)
        if len(a) != len(b) or not a.index.equals(b.index) or not _same_values(a, b):
            bad.append(str(full.columns[col_pos]))
    return bad


def _same_values(a: pd.Series, b: pd.Series) -> bool:
    # An int column comes back as float once its gaps are NaN; the models read
    # every column through pd.to_numeric, so compare the numbers they will see.
    na, nb = pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")
    if na.notna().all() and nb.notna().all():
        return bool((na.to_numpy(dtype=float) == nb.to_numpy(dtype=float)).all())
    return [str(x) for x in a] == [str(x) for x in b]


def encode(value: Any, _depth: int = 0, _compact: list | None = None) -> Any:
    """Anything the page holds -> plain JSON, with tagged types so it decodes exactly.

    ``_compact``, when given, turns long frames into ``compact_frame`` and
    collects a verification line per frame.
    """

    if _depth > 24:
        return {"__type__": "truncated"}
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, enum.Enum):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return _float(value)
    if isinstance(value, pd.DataFrame):
        if _compact is not None and len(value) > COMPACT_OVER_ROWS:
            doc = compact_frame(value)
            rebuilt = _decode_compact_frame(doc)
            _compact.append({
                "rows": len(value),
                "kept_rows": len(doc["positions"]),
                "columns": value.shape[1],
                "mismatched_columns": verify_compact(value, rebuilt, HISTORY_PER_COLUMN),
            })
            return doc
        return _encode_frame(value)
    if isinstance(value, pd.Series):
        return {
            "__type__": "pdseries",
            "name": encode(value.name),
            "dtype": str(value.dtype),
            "index": _encode_index(value.index),
            "data": _encode_column(value),
        }
    if isinstance(value, np.ndarray):
        return {"__type__": "ndarray", "dtype": str(value.dtype),
                "data": encode(value.tolist(), _depth + 1, _compact)}
    if value is pd.NaT:
        return {"__type__": "nat"}
    if isinstance(value, pd.Timestamp):
        return {"__type__": "timestamp", "iso": value.isoformat()}
    if isinstance(value, datetime):
        return {"__type__": "datetime", "iso": value.isoformat()}
    if isinstance(value, date):
        return {"__type__": "date", "iso": value.isoformat()}
    if isinstance(value, (timedelta, pd.Timedelta)):
        return {"__type__": "timedelta", "s": pd.Timedelta(value).total_seconds()}
    if isinstance(value, enum.Enum):
        return {"__type__": "enum", "class": f"{type(value).__module__}.{type(value).__qualname__}",
                "value": encode(value.value, _depth + 1, _compact)}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "__type__": "dataclass",
            "class": f"{type(value).__module__}.{type(value).__qualname__}",
            "data": {f.name: encode(getattr(value, f.name), _depth + 1, _compact)
                     for f in dataclasses.fields(value)},
        }
    if isinstance(value, Mapping):
        if all(isinstance(k, str) for k in value) and "__type__" not in value:
            return {k: encode(v, _depth + 1, _compact) for k, v in value.items()}
        return {"__type__": "dict", "items": [
            [encode(k, _depth + 1, _compact), encode(v, _depth + 1, _compact)]
            for k, v in value.items()]}
    if isinstance(value, tuple):
        return {"__type__": "tuple", "items": [encode(v, _depth + 1, _compact) for v in value]}
    if isinstance(value, (set, frozenset)):
        return {"__type__": "set", "items": [encode(v, _depth + 1, _compact) for v in value]}
    if isinstance(value, list):
        return [encode(v, _depth + 1, _compact) for v in value]
    if isinstance(value, Path):
        return str(value)
    return {"__type__": "repr", "repr": repr(value)[:300]}


def _resolve_class(path: str) -> type | None:
    module_name, _, qualname = str(path).rpartition(".")
    if not module_name.startswith(_TYPED_MODULE_PREFIXES) or "<" in qualname:
        return None
    try:
        obj: Any = importlib.import_module(module_name)
        for part in qualname.split("."):
            obj = getattr(obj, part)
    except (ImportError, AttributeError):
        return None
    return obj if isinstance(obj, type) else None


def _build_dataclass(cls: type, fields: dict[str, Any]) -> Any:
    known = {f.name: f for f in dataclasses.fields(cls)}
    init = {k: v for k, v in fields.items() if k in known and known[k].init}
    obj = cls(**init)
    for name, value in fields.items():
        if name in known and not known[name].init:
            object.__setattr__(obj, name, value)
    return obj


def decode(value: Any, *, typed: bool = True) -> Any:
    """Tagged JSON -> the values the page used.

    Args:
         - typed: bool - Rebuild our own dataclasses and enums. ``False`` gives
           plain dicts, which is what the report works from.
    """

    if isinstance(value, list):
        return [decode(v, typed=typed) for v in value]
    if not isinstance(value, dict):
        return value
    kind = value.get("__type__")
    if kind is None:
        return {k: decode(v, typed=typed) for k, v in value.items()}
    if kind == "float":
        return float(value["v"])
    if kind == "frame":
        return _decode_frame(value)
    if kind == "compact_frame":
        return _decode_compact_frame(value)
    if kind == "dataframe":  # schema 1
        frame = pd.DataFrame(**value["data"])
        for column, dtype in (value.get("dtypes") or {}).items():
            if column in frame.columns:
                try:
                    frame[column] = frame[column].astype(dtype)
                except (TypeError, ValueError):
                    pass
        return frame
    if kind == "series":  # schema 1
        return pd.Series(**value["data"])
    if kind == "pdseries":
        series = _decode_column(value["data"], value["dtype"])
        series.index = _decode_index(value["index"])
        series.name = decode(value.get("name"), typed=typed)
        return series
    if kind == "ndarray":
        return np.array(decode(value["data"], typed=typed), dtype=value.get("dtype") or None)
    if kind == "nat":
        return pd.NaT
    if kind == "timestamp":
        return pd.Timestamp(value["iso"])
    if kind == "datetime":
        return datetime.fromisoformat(value["iso"])
    if kind == "date":
        return date.fromisoformat(value["iso"])
    if kind == "timedelta":
        return pd.Timedelta(seconds=value["s"])
    if kind == "tuple":
        return tuple(decode(v, typed=typed) for v in value["items"])
    if kind == "set":
        return set(decode(v, typed=typed) for v in value["items"])
    if kind == "dict":
        return {_hashable(decode(k, typed=typed)): decode(v, typed=typed) for k, v in value["items"]}
    if kind == "enum":
        cls = _resolve_class(value["class"]) if typed else None
        raw = decode(value["value"], typed=typed)
        try:
            return cls(raw) if cls is not None else raw
        except ValueError:
            return raw
    if kind == "dataclass":
        fields = {k: decode(v, typed=typed) for k, v in value["data"].items()}
        cls = _resolve_class(value["class"]) if typed else None
        if cls is not None and dataclasses.is_dataclass(cls):
            try:
                return _build_dataclass(cls, fields)
            except TypeError:
                pass
        return fields
    if kind in {"repr", "truncated"}:
        return None
    return {k: decode(v, typed=typed) for k, v in value.items()}


def _hashable(value: Any) -> Any:
    return tuple(value) if isinstance(value, list) else value


# --- capture -------------------------------------------------------------------


def _git_provenance() -> dict[str, str]:
    """Branch and commit the snapshot was taken on. Best effort; empty when no git."""

    out: dict[str, str] = {}
    for key, args in (("branch", ["rev-parse", "--abbrev-ref", "HEAD"]),
                      ("commit", ["rev-parse", "--short", "HEAD"])):
        try:
            out[key] = subprocess.run(
                ["git", *args], capture_output=True, text=True, timeout=3,
                cwd=Path(__file__).resolve().parent,
            ).stdout.strip()
        except Exception:  # noqa: BLE001
            pass
    return {k: v for k, v in out.items() if v}


def _prediction_check(
    page_vars: Mapping[str, Any], results: Mapping[str, Any],
    calls: Mapping[str, Any], rebuilt: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Re-run the fuel model on the recommended blend with the full and the saved history.

    Proves on the spot that the compacted history replays to the same model
    output. Best effort: returns None when the page's pieces are unavailable.
    """

    try:
        from utils.bmo.feature_builder import build_feature_payload

        model_service = page_vars.get("model_service")
        ores = page_vars.get("selected_ores")
        target = page_vars.get("target_production_mt")
        blend = results.get("de_result") or results.get("lp_result")
        history_key = next(k for k in calls if k.startswith("get_history_frame("))
        context_key = next(k for k in calls if k.startswith("get_process_context("))
        if model_service is None or not ores or blend is None:
            return None
        full_history = calls[history_key][0]
        saved_history = rebuilt[history_key][0]
        process_context = calls[context_key][0]
        payload = build_feature_payload(
            quantities_mt=dict(blend.quantities_mt),
            ore_display_name_by_id={o.ore_id: o.display_name for o in ores},
            process_context=process_context,
            ores=ores,
            hot_metal_target_mt=target,
        )
        full = float(model_service.predict(dict(payload), full_history).value)
        saved = float(model_service.predict(dict(payload), saved_history).value)
        return {"model_output_full_history": full, "model_output_saved_history": saved,
                "identical": full == saved}
    except Exception:  # noqa: BLE001 - a check, never a reason to fail the snapshot
        return None


def capture(
    state: Mapping[str, Any],
    *,
    prefix: str = LIVE_PREFIX,
    page_vars: Mapping[str, Any] | None = None,
    label: str = "",
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build a snapshot of the last run (or, with no run yet, of the page as it stands).

    Args:
         - state: Mapping - Session state (or anything shaped like it).
         - prefix: str - ``bmo_`` on the live page, ``testbmo_`` in the sandbox.
         - page_vars: Mapping | None - The page's globals: effective input tables,
           derived context, and the pieces for the history check.
         - label: str - Operator's note, e.g. "Shift B, more pellet".
         - now: datetime | None - For tests.

    Returns:
         - return dict - The snapshot, JSON-ready.
    """

    from utils.bmo.replay import K_RECORD, build_record, results_marker

    page_vars = page_vars or {}
    marker = results_marker(state, prefix)
    record = state.get(f"{prefix}{K_RECORD}")
    pinned = (
        isinstance(record, dict)
        and record.get("marker") == marker
        and any(m is not None for m in marker)
    )
    current = build_record(state, prefix, page_vars)
    if not pinned:
        record = current

    inputs = {s: encode(v) for s, v in record["inputs"].items() if is_writable_suffix(s)}
    not_restorable = sorted(s for s in record["inputs"] if not is_writable_suffix(s))
    inputs_changed = pinned and json.dumps(inputs, sort_keys=True, default=str) != json.dumps(
        {s: encode(v) for s, v in current["inputs"].items() if is_writable_suffix(s)},
        sort_keys=True, default=str,
    )

    raw_results = {}
    for key in sorted(str(k) for k in state.keys()):
        if key.startswith(prefix) and classify_suffix(key[len(prefix):]) == "results":
            raw_results[key[len(prefix):]] = state[key]
    results = {s: encode(v) for s, v in raw_results.items()}

    checks: list[dict[str, Any]] = []
    provider_calls = {}
    for key, value in record["provider_calls"].items():
        found: list[dict[str, Any]] = []
        provider_calls[key] = encode(value, _compact=found)
        checks.extend({"call": key, **c} for c in found)
    frozen = {
        "provider_calls": provider_calls,
        "page": {k: encode(v) for k, v in record["page"].items()},
        "session_at_run": {k: encode(v) for k, v in record["session_at_run"].items()},
        "history_checks": checks,
    }
    if checks:
        rebuilt = {k: decode(v) for k, v in provider_calls.items() if k.startswith("get_history_frame(")}
        verdict = _prediction_check(page_vars, raw_results, record["provider_calls"], rebuilt)
        if verdict is not None:
            frozen["model_check"] = verdict

    context = {name: encode(page_vars[name]) for name in PAGE_CONTEXT_VARS if name in page_vars}
    taken = (now or datetime.now(IST)).astimezone(IST)
    snapshot = {
        "schema": SCHEMA,
        "id": "",  # assigned by the store, which knows what already exists
        "created_at": taken.isoformat(timespec="seconds"),
        "source": SOURCE_BY_PREFIX.get(prefix, prefix),
        "label": str(label or "").strip(),
        "provenance": _git_provenance(),
        "run": {
            "at": record.get("at") if pinned else None,
            "pinned": bool(pinned),
            "inputs_changed_after_run": bool(inputs_changed),
        },
        "inputs": inputs,
        "results": results,
        "context": context,
        "frozen": frozen,
        "not_restorable": not_restorable,
    }
    snapshot["summary"] = summarise(snapshot)
    return snapshot


# --- summary: the row in the snapshot table ------------------------------------


def _fields(encoded_result: Any) -> dict[str, Any]:
    if isinstance(encoded_result, dict) and encoded_result.get("__type__") == "dataclass":
        return encoded_result.get("data") or {}
    return encoded_result if isinstance(encoded_result, dict) else {}


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _frame_rows(encoded: Any) -> list[dict[str, Any]]:
    """Rows of an encoded table (either schema), as plain dicts."""

    if not isinstance(encoded, dict):
        return []
    if encoded.get("__type__") == "frame":
        columns = [str(c) for c in encoded.get("columns") or []]
        data = encoded.get("data") or []
        n = len(data[0]) if data else 0
        return [{c: data[j][i] for j, c in enumerate(columns)} for i in range(n)]
    if encoded.get("__type__") == "dataframe":
        data = encoded.get("data") or {}
        columns = [str(c) for c in data.get("columns") or []]
        return [dict(zip(columns, row)) for row in data.get("data") or []]
    return []


def ore_names(snapshot: Mapping[str, Any]) -> dict[str, str]:
    """ore_id -> display name, from the recorded ore table where available."""

    rows = _frame_rows((snapshot.get("inputs") or {}).get("applied_ore_editor_df"))
    return {str(r["ore_id"]): str(r.get("ore_name", r["ore_id"])) for r in rows if "ore_id" in r}


def recommended_result(snapshot: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """(label, fields) of the result the page would recommend.

    DE when it ran and genuinely improved on the LP; the LP otherwise. A DE that
    fell back to the LP blend is the LP, and is labelled so.
    """

    results = snapshot.get("results") or {}
    de, lp = _fields(results.get("de_result")), _fields(results.get("lp_result"))
    if de and not (de.get("diagnostics") or {}).get("de_fell_back_to_lp"):
        return "DE total cost", de
    if lp:
        return "LP baseline", lp
    return "", {}


def summarise(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """The metadata row: when, what blend, what fuel, what slag."""

    label, result = recommended_result(snapshot)
    diag = result.get("diagnostics") or {}
    rates = diag.get("fuel_rate_estimate") or {}
    names = ore_names(snapshot)
    shares = {
        names.get(str(k), str(k)): round(float(v), 2)
        for k, v in (result.get("shares_pct") or {}).items()
        if _number(v) and float(v) > 0.05
    }
    flux_cost = _number(diag.get("flux_cost_per_thm_rs")) or 0.0
    total = _number(diag.get("adjusted_objective_rs_per_thm"))
    if total is None:
        total = _number(result.get("objective_rs_per_thm"))

    inputs = snapshot.get("inputs") or {}
    production = _number(inputs.get("target_production_mt"))
    if production is None:
        production = _number(diag.get("hot_metal_target_mt"))

    return {
        "result": label,
        "production_mt": production,
        "blend_pct": dict(sorted(shares.items(), key=lambda kv: -kv[1])),
        "fuel_rate_kg_thm": _number(rates.get("total_fuel_rate_kg_thm")),
        "coke_rate_kg_thm": _number(rates.get("coke_rate_kg_thm")),
        "nut_coke_kg_thm": _number(rates.get("nut_coke_rate_kg_thm")),
        "pci_kg_thm": _number(rates.get("pci_rate_kg_thm")),
        "slag_rate_kg_thm": _number(result.get("slag_rate_kg_per_thm")),
        "slag_mt": _number(result.get("slag_mt")),
        "basicity_b2": _number(result.get("slag_basicity")),
        "t_basicity": _number(result.get("slag_t_basicity")),
        "fe_pct": _number(result.get("fe_t_pct")),
        "total_cost_rs_thm": (total + flux_cost) if total is not None else None,
        "feasible": bool(result.get("feasible")) if result else None,
        "violations": len(result.get("violations") or []) if result else None,
        "frozen_context": bool((snapshot.get("frozen") or {}).get("provider_calls")),
    }


# --- restore -------------------------------------------------------------------


def restorable_state(
    snapshot: Mapping[str, Any], *, prefix: str = SANDBOX_PREFIX
) -> dict[str, Any]:
    """Session-state writes that put this snapshot's INPUTS back on a page.

    Keys belonging to buttons and data editors are filtered out again here,
    even if a hand-edited JSON includes them, because writing one crashes the
    page when the widget renders.
    """

    out: dict[str, Any] = {}
    for suffix, encoded in (snapshot.get("inputs") or {}).items():
        suffix = str(suffix)
        if classify_suffix(suffix) != "inputs" or not is_writable_suffix(suffix):
            continue
        value = decode(encoded)
        if value is None:
            continue
        out[f"{prefix}{suffix}"] = value
    return out


def results_state(
    snapshot: Mapping[str, Any], *, prefix: str = SANDBOX_PREFIX
) -> dict[str, Any]:
    """Session-state writes that put the recorded RESULTS back, so the page shows them.

    Only for snapshots that carry their frozen context: results from a
    schema-1 snapshot would sit next to live data they were not computed from.
    """

    if not frozen_context_present(snapshot):
        return {}
    return {
        f"{prefix}{suffix}": decode(encoded)
        for suffix, encoded in (snapshot.get("results") or {}).items()
        if classify_suffix(str(suffix)) == "results"
    }


def frozen_context_present(snapshot: Mapping[str, Any]) -> bool:
    return bool((snapshot.get("frozen") or {}).get("provider_calls"))


def frozen_state(snapshot: Mapping[str, Any]) -> dict[str, Any] | None:
    """The frozen plant data, decoded for replay (``utils/bmo/replay.py``)."""

    frozen = snapshot.get("frozen") or {}
    if not frozen.get("provider_calls") and not frozen.get("page"):
        return None
    return {
        "provider_calls": {k: decode(v) for k, v in (frozen.get("provider_calls") or {}).items()},
        "page": {k: decode(v) for k, v in (frozen.get("page") or {}).items()},
        "session_at_run": {k: decode(v) for k, v in (frozen.get("session_at_run") or {}).items()},
    }


def validate(document: Any) -> dict[str, Any]:
    """Accept a full snapshot, or a bare ``{"inputs": {...}}`` to play with.

    Raises:
         - ValueError - With a message fit to show an operator.
    """

    if not isinstance(document, dict):
        raise ValueError("The JSON must be an object.")
    schema = str(document.get("schema", ""))
    if schema and not schema.startswith("bmo-snapshot/"):
        raise ValueError(f"Unrecognised schema {schema!r}.")
    if not isinstance(document.get("inputs"), dict):
        raise ValueError('The JSON needs an "inputs" object.')
    snapshot = dict(document)
    snapshot.setdefault("schema", SCHEMA)
    snapshot.setdefault("results", {})
    snapshot.setdefault("context", {})
    snapshot.setdefault("frozen", {})
    snapshot.setdefault("source", "uploaded")
    snapshot.setdefault("created_at", "")
    snapshot.setdefault("label", "")
    return snapshot
