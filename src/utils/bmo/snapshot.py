"""A complete, replayable record of one Blend Mix Optimiser state.

WHAT A SNAPSHOT IS.

Every input the page ran with, every result it produced, and enough context to
explain both - written as one JSON document with a timestamp. It is a record, not
a screenshot: the report is generated from it, and TestBMO restores it.

HOW IT DECIDES WHAT TO CAPTURE.

By discovery, not by a fixed schema. The page keeps its state under ``bmo_*``
session keys, and the set differs between branches - UAT has a PCI override, an
energy anchor and a commentary that main does not. A hard-coded list would break
on one branch or silently miss fields on the other. So every ``bmo_*`` key present
is classified:

    results    what the optimiser produced; recorded, used for the report, and
               NOT restored - TestBMO re-runs to get its own
    internal   cache counters and flags; skipped
    ui         the snapshot panel's own widgets; skipped
    inputs     everything else; recorded and restored

Session state alone is not enough, though. The applied-editor tables only exist
once an operator presses Apply; before that the page runs on freshly fetched
defaults, and those defaults move as plant data moves. So the page also hands
over its own effective input tables (``page_vars``), which win over session state
and make the snapshot reproducible whatever was or was not applied.

Keys are stored WITHOUT their prefix, so one snapshot restores into either the
live page or the TestBMO sandbox.
"""

from __future__ import annotations

import dataclasses
import json
import math
import subprocess
from datetime import date, datetime
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

SCHEMA = "bmo-snapshot/1"
IST = ZoneInfo("Asia/Kolkata")

SOURCE_BY_PREFIX = {LIVE_PREFIX: "Blend Mix Optimiser", SANDBOX_PREFIX: "TestBMO"}

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
    "hm_snapshot",
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


# --- JSON encoding ---------------------------------------------------------------


def encode(value: Any, _depth: int = 0) -> Any:
    """Anything the page holds -> plain JSON, with tagged types where it matters.

    DataFrames keep their columns, index and dtypes so they restore as the same
    table. Dataclasses (BlendEvaluation) are recorded field by field. Anything
    unrecognised is kept as a short repr rather than failing the whole snapshot -
    a snapshot that loses one exotic field is far more useful than none at all.
    """

    if _depth > 14:
        return {"__type__": "truncated"}
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.DataFrame):
        return {
            "__type__": "dataframe",
            "data": json.loads(value.to_json(orient="split", date_format="iso",
                                             default_handler=str)),
            "dtypes": {str(c): str(t) for c, t in value.dtypes.items()},
        }
    if isinstance(value, pd.Series):
        return {
            "__type__": "series",
            "data": json.loads(value.to_json(orient="split", date_format="iso",
                                             default_handler=str)),
        }
    if isinstance(value, np.ndarray):
        return encode(value.tolist(), _depth + 1)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "__type__": "dataclass",
            "class": f"{type(value).__module__}.{type(value).__qualname__}",
            "data": {f.name: encode(getattr(value, f.name), _depth + 1)
                     for f in dataclasses.fields(value)},
        }
    if isinstance(value, Mapping):
        return {str(k): encode(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [encode(v, _depth + 1) for v in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return {"__type__": "repr", "repr": repr(value)[:300]}


def decode(value: Any) -> Any:
    """Tagged JSON -> values the page can use. Dataclasses come back as dicts."""

    if isinstance(value, list):
        return [decode(v) for v in value]
    if not isinstance(value, dict):
        return value
    kind = value.get("__type__")
    if kind == "dataframe":
        frame = pd.DataFrame(**value["data"])
        for column, dtype in (value.get("dtypes") or {}).items():
            if column in frame.columns:
                try:
                    frame[column] = frame[column].astype(dtype)
                except (TypeError, ValueError):
                    pass
        return frame
    if kind == "series":
        return pd.Series(**value["data"])
    if kind == "dataclass":
        return {k: decode(v) for k, v in value["data"].items()}
    if kind in {"repr", "truncated"}:
        return None
    return {k: decode(v) for k, v in value.items()}


# --- capture -------------------------------------------------------------------


def _classify(suffix: str) -> str:
    if suffix.startswith(UI_SUFFIX):
        return "ui"
    if suffix in SKIPPED_SUFFIXES:
        return "skip"
    if suffix in RESULT_SUFFIXES:
        return "results"
    return "inputs"


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


def capture(
    state: Mapping[str, Any],
    *,
    prefix: str = LIVE_PREFIX,
    page_vars: Mapping[str, Any] | None = None,
    label: str = "",
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build a snapshot from session state and the page's own variables.

    Args:
         - state: Mapping - Session state (or anything shaped like it).
         - prefix: str - ``bmo_`` on the live page, ``testbmo_`` in the sandbox.
         - page_vars: Mapping | None - The page's globals. Supplies the effective
           input tables and the derived context.
         - label: str - Operator's note, e.g. "Shift B, more pellet".
         - now: datetime | None - For tests.

    Returns:
         - return dict - The snapshot, JSON-ready.
    """

    page_vars = page_vars or {}
    inputs: dict[str, Any] = {}
    results: dict[str, Any] = {}
    not_restorable: list[str] = []

    for key in sorted(k for k in state.keys() if str(k).startswith(prefix)):
        suffix = str(key)[len(prefix):]
        kind = _classify(suffix)
        if kind in {"ui", "skip"}:
            continue
        encoded = encode(state[key])
        if kind == "results":
            results[suffix] = encoded
        elif is_writable_suffix(suffix):
            inputs[suffix] = encoded
        else:
            # A button or data editor: Streamlit forbids setting it, so record
            # it for the report but never offer it back.
            not_restorable.append(suffix)

    for var, suffix in PAGE_INPUT_TABLES.items():
        table = page_vars.get(var)
        if isinstance(table, pd.DataFrame) and not table.empty:
            inputs[suffix] = encode(table)

    context = {
        name: encode(page_vars[name]) for name in PAGE_CONTEXT_VARS if name in page_vars
    }

    taken = (now or datetime.now(IST)).astimezone(IST)
    snapshot = {
        "schema": SCHEMA,
        "id": "",  # assigned by the store, which knows what already exists
        "created_at": taken.isoformat(timespec="seconds"),
        "source": SOURCE_BY_PREFIX.get(prefix, prefix),
        "label": str(label or "").strip(),
        "provenance": _git_provenance(),
        "inputs": inputs,
        "results": results,
        "context": context,
        "not_restorable": sorted(not_restorable),
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


def ore_names(snapshot: Mapping[str, Any]) -> dict[str, str]:
    """ore_id -> display name, from the recorded ore table where available."""

    table = (snapshot.get("inputs") or {}).get("applied_ore_editor_df")
    if not isinstance(table, dict) or table.get("__type__") != "dataframe":
        return {}
    data = table.get("data") or {}
    columns = data.get("columns") or []
    if "ore_id" not in columns or "ore_name" not in columns:
        return {}
    i_id, i_name = columns.index("ore_id"), columns.index("ore_name")
    return {str(row[i_id]): str(row[i_name]) for row in data.get("data") or []}


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
    }


# --- restore -------------------------------------------------------------------


def restorable_state(
    snapshot: Mapping[str, Any], *, prefix: str = SANDBOX_PREFIX
) -> dict[str, Any]:
    """Session-state writes that put this snapshot's inputs back on a page.

    Only inputs. Results are not restored: the sandbox re-runs and produces its
    own, which is the point. Keys belonging to buttons and data editors are
    filtered out again here, even if a hand-edited JSON includes them, because
    writing one crashes the page when the widget renders.
    """

    out: dict[str, Any] = {}
    for suffix, encoded in (snapshot.get("inputs") or {}).items():
        suffix = str(suffix)
        if suffix.startswith(UI_SUFFIX) or not is_writable_suffix(suffix):
            continue
        value = decode(encoded)
        if value is None:
            continue
        out[f"{prefix}{suffix}"] = value
    return out


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
    snapshot.setdefault("source", "uploaded")
    snapshot.setdefault("created_at", "")
    snapshot.setdefault("label", "")
    return snapshot
