"""Build frontend-neutral JSON results from Material Balance domain output."""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from typing import Any

from furnace_data.material_balance.constants import ELEMENTS
from furnace_data.material_balance.types import BalanceResult

_SYMBOL_LABELS = {
    "Fe": "Iron",
    "C": "Carbon",
    "Si": "Silicon",
    "Ca": "Calcium",
    "Mg": "Magnesium",
    "Al": "Aluminium",
    "Mn": "Manganese",
    "S": "Sulfur",
    "P": "Phosphorus",
    "O": "Oxygen",
    "N": "Nitrogen",
    "H": "Hydrogen",
}


def build_material_balance_result(
    result: BalanceResult,
    *,
    config: dict[str, Any],
    computed_at: datetime | None = None,
    calculation_id: str | None = None,
    artifacts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return complete JSON-native result data for Streamlit and React clients."""

    computed = computed_at or datetime.now(timezone.utc)
    thresholds = _thresholds(config)
    closure_rows = [_closure_row(row, thresholds) for row in result.closure_table.to_dict(orient="records")]
    total_in = sum(float(row.get("input_t") or 0.0) for row in closure_rows)
    total_out = sum(float(row.get("output_t") or 0.0) for row in closure_rows)
    overall = (total_out / total_in * 100.0) if total_in > 0 else None
    summary = {
        "overall_closure_pct": _safe_round(overall, 2),
        "closure_status": _status(overall, thresholds),
        "total_input_element_t": _safe_round(total_in, 2),
        "total_output_element_t": _safe_round(total_out, 2),
        "delta_t": _safe_round(total_out - total_in, 2),
        "hot_metal_mass_t": _safe(result.gas_phase.get("hm_mass_t")),
        "slag_mass_t": _safe(result.gas_phase.get("slag_mass_t")),
        "burden_mass_t": _safe(result.gas_phase.get("burden_mass_t")),
        "dust_catcher_mass_t": _safe(result.gas_phase.get("dust_catcher_t")),
    }
    return {
        "calculation_id": calculation_id or f"mbr_{uuid.uuid4().hex}",
        "computed_at": _iso_z(computed),
        "day": result.day.isoformat(),
        "algorithm_version": result.algorithm_version,
        "window_policy_version": result.window_policy_version,
        "versions": dict(result.versions),
        "resolved_windows": {key: _window(value) for key, value in result.windows.items()},
        "summary": summary,
        "closure_thresholds": thresholds,
        "closure": closure_rows,
        "material_masses": _material_masses(result),
        "input_streams": _streams(result.inputs),
        "output_streams": _streams(result.outputs),
        "diagram_flows": _diagram_flows(result),
        "gas_phase": _gas_phase(result.gas_phase),
        "data_quality": _clean(result.data_quality),
        "warnings": [{"code": "MATERIAL_BALANCE_WARNING", "message": str(item), "details": {}} for item in result.warnings],
        "assumptions": list(result.assumptions),
        "artifacts": artifacts or [],
        # Compatibility payloads for existing Streamlit plotters and older tests.
        "summary_legacy": {
            "source": "static_dataset",
            "date": result.day.isoformat(),
            "used_dpr": result.used_dpr,
            "n_rm_rows": result.n_rm_rows,
            "total_in_t": _safe_round(total_in, 2),
            "total_out_t": _safe_round(total_out, 2),
        },
        "kpis": {
            "overall_closure_pct": _safe_round(overall, 2),
            "hm_mass_t": _safe(result.gas_phase.get("hm_mass_t")),
            "slag_mass_t": _safe(result.gas_phase.get("slag_mass_t")),
        },
        "tables": {"closure": _table(closure_rows)},
        "charts": {},
    }


def _thresholds(config: dict[str, Any]) -> dict[str, dict[str, float]]:
    raw = config.get("closure_thresholds") or {}
    good = raw.get("good") or [95.0, 105.0]
    warning = raw.get("warning") or [85.0, 115.0]
    return {
        "good": {"minimum": float(good[0]), "maximum": float(good[1])},
        "warning": {"minimum": float(warning[0]), "maximum": float(warning[1])},
    }


def _closure_row(row: dict[str, Any], thresholds: dict[str, dict[str, float]]) -> dict[str, Any]:
    symbol = str(row.get("Element") or "")
    closure = _safe(row.get("Closure_pct"))
    return {
        "element_id": symbol.lower(),
        "symbol": symbol,
        "label": _SYMBOL_LABELS.get(symbol, symbol),
        "input_t": _safe(row.get("In_t")),
        "output_t": _safe(row.get("Out_t")),
        "closure_pct": closure,
        "delta_t": _safe(row.get("Delta_t")),
        "status": _status(closure, thresholds),
    }


def _status(value: Any, thresholds: dict[str, dict[str, float]]) -> str:
    number = _safe(value)
    if number is None:
        return "unavailable"
    good = thresholds["good"]
    warning = thresholds["warning"]
    if good["minimum"] <= number <= good["maximum"]:
        return "good"
    if warning["minimum"] <= number <= warning["maximum"]:
        return "warning"
    return "critical"


def _material_masses(result: BalanceResult) -> list[dict[str, Any]]:
    rows = []
    for label, mass in result.material_masses.items():
        source = result.material_sources.get(label, {})
        rows.append(
            {
                "material_id": _id(label),
                "label": label,
                "mass_t": _safe(mass),
                "source": source.get("source", "static_dataset"),
                "source_field_id": source.get("source_field_id"),
                "canonical_field_id": source.get("canonical_field_id"),
                "quality": source.get("quality", "calculated"),
            }
        )
    return rows


def _streams(data: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    totals: dict[str, dict[str, Any]] = {}
    for symbol in ELEMENTS:
        for stream, tonnes in (data.get(symbol) or {}).items():
            bucket = totals.setdefault(stream, {"elements": [], "total_t": 0.0})
            safe = float(tonnes or 0.0)
            bucket["elements"].append(
                {"element_id": symbol.lower(), "symbol": symbol, "mass_t": _safe_round(safe, 4)}
            )
            bucket["total_t"] += safe
    return [
        {
            "stream_id": _id(label),
            "label": label,
            "total_t": _safe_round(payload["total_t"], 4),
            "elements": payload["elements"],
        }
        for label, payload in sorted(totals.items())
    ]


def _diagram_flows(result: BalanceResult) -> dict[str, list[dict[str, Any]]]:
    top_gas_t = sum((result.outputs.get(symbol) or {}).get("Top Gas", 0.0) for symbol in ELEMENTS)
    inputs = [
        {"flow_id": "burden", "label": "Burden", "mass_t": _safe(result.gas_phase.get("burden_mass_t"))},
        {"flow_id": "hot_blast", "label": "Hot Blast", "mass_t": _safe(result.gas_phase.get("hot_blast_mass_t"))},
        {"flow_id": "pci_plus_steam", "label": "PCI + Steam", "mass_t": _safe(result.gas_phase.get("pci_plus_steam_mass_t"))},
    ]
    outputs = [
        {"flow_id": "top_gas", "label": "Top Gas", "mass_t": _safe_round(top_gas_t, 4)},
        {"flow_id": "hot_metal", "label": "Hot Metal", "mass_t": _safe(result.gas_phase.get("hm_mass_t"))},
        {"flow_id": "slag", "label": "Slag", "mass_t": _safe(result.gas_phase.get("slag_mass_t"))},
        {"flow_id": "dust_catcher", "label": "Dust Catcher", "mass_t": _safe(result.gas_phase.get("dust_catcher_t"))},
    ]
    return {"inputs": inputs, "outputs": outputs}


def _gas_phase(gas: dict[str, Any]) -> dict[str, Any]:
    return {
        "wind_nm3_per_hour": _safe(gas.get("wind_nm3h")),
        "oxygen_flow_nm3_per_hour": _safe(gas.get("o2_flow_nm3h")),
        "steam_kg_per_hour": _safe(gas.get("steam_kgh")),
        "top_gas_nm3_per_day": _safe(gas.get("top_gas_nm3_per_day")),
        "top_gas_method": gas.get("top_gas_method") or "n2_balance",
        "top_gas_fallback_applied": bool(gas.get("top_gas_fallback_applied")),
        "hot_blast_mass_t": _safe(gas.get("hot_blast_mass_t")),
        "pci_plus_steam_mass_t": _safe(gas.get("pci_plus_steam_mass_t")),
    }


def _window(window: Any) -> dict[str, str]:
    return {
        "local_start": window.local_start.isoformat(),
        "local_end": window.local_end.isoformat(),
        "utc_start": _iso_z(window.utc_start),
        "utc_end": _iso_z(window.utc_end),
    }


def _table(rows: list[dict[str, Any]]) -> dict[str, Any]:
    legacy_rows = [
        {
            "Element": row["symbol"],
            "In_t": row["input_t"],
            "Out_t": row["output_t"],
            "Closure_pct": row["closure_pct"],
            "Delta_t": row["delta_t"],
            "Status": row["status"],
        }
        for row in rows
    ]
    columns = [
        {"name": key, "type": "number" if key not in {"Element", "Status"} else "string"}
        for key in (legacy_rows[0].keys() if legacy_rows else [])
    ]
    return {
        "columns": columns,
        "rows": legacy_rows,
        "row_count": len(legacy_rows),
        "returned_rows": len(legacy_rows),
        "truncated": False,
    }


def _safe(value: Any) -> Any:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if not math.isfinite(number):
        return None
    return number


def _safe_round(value: Any, digits: int) -> float | None:
    number = _safe(value)
    if number is None or not isinstance(number, (int, float)):
        return None
    return round(float(number), digits)


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    return _safe(value)


def _iso_z(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _id(label: str) -> str:
    return str(label or "").strip().lower().replace(" ", "_").replace("+", "plus")