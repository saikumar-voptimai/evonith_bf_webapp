"""Write the parity fixtures of the Svelte chart kit (SV-0.6, #275) from the Python models.

    python scripts/export_chart_fixtures.py

- ``frontend/src/lib/charts/fixtures/rist_shift.json``: one shift of ``tests/fixtures/coke_energy/shift_inputs.csv``
  through ``utils.coke_energy.rist_diagram.shift_rist`` (the Streamlit Rist diagram's input), with the points the
  diagram marks (A, D, E, W, T, E' and the pivots). ``RistDiagram.svelte`` derives the same points from the line
  parameters; the Vitest suite compares them.
- ``frontend/src/lib/furnace/fixtures/schematic.json``: the furnace outline, lining and zone dividers of
  ``ui.furnace_status.schematic.build_furnace_svg`` for the configured profile. ``FurnaceSchematic.svelte`` must
  draw the same points (acceptance criterion of #275).
- ``frontend/src/lib/status/fixtures/snapshot.json``: a recorded furnace-status snapshot (the plant frames of
  ``tests/test_services_status.py`` at 27 Oct 2026 15:00:15 IST) as ``GET /api/status/snapshot`` and
  ``GET /api/status/settings`` serve it, with the V-Board's layout (``ui.furnace_status.layout``) and the texts it
  renders from the same snapshot (row labels, callouts, strip spreads, heat-load shares, badges). The builders of the
  Furnace Status page (``frontend/src/lib/status/status.ts``) must reproduce them (acceptance criterion of #294).
- ``frontend/src/lib/reports/fixtures/report.json``: shift 26 Oct 2026 B of the synthetic sources of
  ``tests/test_services_reports.py`` through the report builder and the Reports tab's renderer, as
  ``GET /api/reports/{id}`` and ``GET /api/reports/thresholds`` serve it. The Shift Reports page's tables
  (``frontend/src/lib/reports/reports.ts``) must show the numbers of the tab's document (#296).
- ``frontend/src/lib/optimiser/fixtures/result.json``: a recorded Burden & Fuel Optimiser run, the 16 Oct 2026 case
  report at 2,270 t/d (Fixed-ω, reference yard, case calibration at 15 g/Nm³) with a charged blend to compare, as
  ``GET /api/optimiser/runs/{id}/result`` serves it, a Step 1 result on three levels of the same case, and
  ``GET /api/optimiser/reference``; ``expected`` holds what page 11 renders from the engine's result (``plan_tiles``,
  ``band_note``, ``plan_sentence``, ``blend_compare_table``, ``flux_compare_table``, the frontier header, the value of
  limits). The builders of the SPA's optimiser page (``frontend/src/lib/optimiser/optimiser.ts``) must reproduce them
  (#300).

``tests/test_chart_fixtures.py`` fails when a committed fixture differs from this output.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "furnace_data")]
os.environ.setdefault("COKE_ENERGY_WEATHER_LIVE", "0")  # the shipped weather record only: reproducible output

RIST_OUT = ROOT / "frontend" / "src" / "lib" / "charts" / "fixtures" / "rist_shift.json"
SCHEMATIC_OUT = ROOT / "frontend" / "src" / "lib" / "furnace" / "fixtures" / "schematic.json"
STATUS_OUT = ROOT / "frontend" / "src" / "lib" / "status" / "fixtures" / "snapshot.json"
REPORT_OUT = ROOT / "frontend" / "src" / "lib" / "reports" / "fixtures" / "report.json"
OPTIMISER_OUT = ROOT / "frontend" / "src" / "lib" / "optimiser" / "fixtures" / "result.json"
SHIFT_INPUTS = ROOT / "tests" / "fixtures" / "coke_energy" / "shift_inputs.csv"
SHIFT_KEY = "2026-10-14_A"


def _r(value: float | None, digits: int = 9) -> float | None:
    return None if value is None else round(float(value), digits)


def _pt(point: tuple[float, float] | None) -> list[float] | None:
    return None if point is None else [_r(point[0]), _r(point[1])]


def rist_fixture() -> dict[str, Any]:
    import pandas as pd

    from utils.coke_energy.bf_balance import DEFAULTS, calibrate_loss
    from utils.coke_energy.rist_diagram import rist_dict, shift_rist
    from utils.coke_energy.run_shifts import case_from_shift, shift_balance

    rows = pd.read_csv(SHIFT_INPUTS, index_col=0)
    row = rows.loc[SHIFT_KEY]
    result = shift_balance(row)
    case = case_from_shift(row)
    case["eta_H2"] = result["eta_H2"] / 100
    K = result["coke_kgthm_dry"] - 5
    _, V, _ = calibrate_loss(case, K, dict(DEFAULTS), eta_CO=result["eta_CO"] / 100)
    rs = shift_rist(
        case,
        K,
        V,
        result["eta_CO"] / 100,
        prod_th=result["prod_th"],
        omega_fixed=0.79,
        Q_LZ_GJh_fixed=40.0,
        Q_UZ_GJh_fixed=20.0,
        r_H_fixed=1.0,
        skey=SHIFT_KEY,
    )

    def rounded(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: rounded(v) for k, v in value.items()}
        if isinstance(value, list):
            return [rounded(v) for v in value]
        return _r(value) if isinstance(value, float) else value

    # The FORGE API serves the same dictionary per shift (services.forge, SV-1.11).
    payload = rounded(rist_dict(rs))
    payload["t_r"] = _r(rs.T_R, 3)
    a = rs.actual
    return {
        "source": (
            f"utils.coke_energy.rist_diagram.shift_rist, {SHIFT_KEY} of tests/fixtures/coke_energy/shift_inputs.csv"
        ),
        **payload,
        "expected": {
            "a": _pt(a.a),
            "d": _pt(a.d),
            "e": _pt(a.e),
            "w": _pt(a.w),
            "top": _pt(a.top),
            "w_full": _pt(rs.w_full),
            "e_model": _pt(rs.model.e) if rs.model is not None else None,
        },
    }


def _points(attr: str) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in (p.split(",") for p in attr.split())]


# Zone dividers of build_furnace_svg: dashed 1 px lines.
DIVIDER = re.compile(
    r'<line x1="([\d.]+)" y1="([\d.]+)" x2="([\d.]+)" y2="([\d.]+)" stroke="[^"]+" stroke-width="1" '
    r"stroke-dasharray"
)


def schematic_fixture() -> dict[str, Any]:
    from ui.furnace_status import schematic
    from ui.furnace_status.layout import ZONES

    profile = schematic.furnace_profile()
    svg = schematic.build_furnace_svg(profile)
    polygon = re.search(r'<polygon points="([^"]+)" fill="url\(#fsBody\)"', svg)
    linings = re.findall(r'<polyline points="([^"]+)"', svg)
    dividers = DIVIDER.findall(svg)
    assert polygon is not None, "outline polygon not found in build_furnace_svg output"

    return {
        "source": "ui.furnace_status.schematic.build_furnace_svg with the profile of config/setting_ds_dv.yml",
        "zones": [{"name": name, "lower_m": lower, "upper_m": upper} for name, lower, upper in ZONES],
        "callouts": [
            {"key": c.key, "side": c.side, "y": c.y, "anchor": c.anchor, "elevation_m": c.elevation}
            for c in schematic.CALLOUTS
        ],
        "callout_anchors": [[round(v, 6) for v in schematic._callout_anchor(profile, c)] for c in schematic.CALLOUTS],
        "view_box": [schematic.SVG_W, schematic.SVG_H],
        "px_per_m": schematic.PX_PER_M,
        "profile": [[x, y] for x, y in profile],
        "outline": polygon.group(1),
        "outline_points": _points(polygon.group(1)),
        "lining": linings,
        "zone_dividers": [[float(v) for v in d] for d in dividers],
    }


def status_fixture() -> dict[str, Any]:
    """The recorded snapshot of ``tests/test_services_status.py`` (fake InfluxDB, no plant source) through the service
    and the V-Board, so the Svelte builders can be checked against both."""
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    from data import furnace_status as fs
    from data.furnace_status import repository as fs_repository
    from services import status
    from tests.test_services_status import NOW, CountingFetcher, plant_frames
    from ui.furnace_status import components, layout, schematic

    fake = CountingFetcher(plant_frames())
    original = fs_repository.TimeSeriesDataFetcher
    fs_repository.TimeSeriesDataFetcher = fake
    status.clear_cache()
    try:
        page = fs.load_status_snapshot(NOW)
        view = status.snapshot_view(page, NOW)
    finally:
        fs_repository.TimeSeriesDataFetcher = original
        status.clear_cache()

    readings = {reading.spec.key: reading for reading in page.readings}
    strips: dict[str, Any] = {}
    for section, keys in layout.STRIPS.items():
        primary = [readings[key] for key in keys[:4]]
        spread = components.strip_spread([reading.value for reading in primary])
        strips[section] = {
            "keys": list(keys),
            "spread": spread,
            "spread_text": None if spread is None else fs.format_value(spread, primary[0].spec.decimals),
        }
    shares = components.quadrant_shares([readings[key].value for key in layout.HEAT_QUADRANT_KEYS])
    callouts = {}
    for callout in schematic.CALLOUTS:
        reading = readings[callout.key]
        callouts[callout.key] = {
            "label": reading.spec.label,
            "value": fs.format_reading(reading),
            "unit": reading.spec.unit if reading.value is not None and reading.spec.unit else "",
        }
    return {
        "source": "services.status.snapshot_view of tests.test_services_status.plant_frames at NOW; ui.furnace_status",
        "now": NOW.isoformat(),
        "layout": {
            "panel_columns": [list(column) for column in layout.PANEL_COLUMNS],
            "strips": {section: list(keys) for section, keys in layout.STRIPS.items()},
            "heat_total_key": layout.HEAT_TOTAL_KEY,
            "heat_quadrant_keys": list(layout.HEAT_QUADRANT_KEYS),
            "not_available": fs.NOT_AVAILABLE,
        },
        "settings": json.loads(status.settings_view().model_dump_json()),
        "snapshot": json.loads(view.model_dump_json()),
        "expected": {
            "rows": {key: components.button_label(reading) for key, reading in readings.items()},
            "callouts": callouts,
            "strips": strips,
            "heat_load": {
                "shares": None if shares.shares is None else list(shares.shares),
                "highest": list(shares.highest),
                "summary": components._share_summary(shares),
            },
            "status": {"level": page.status.level, "label": page.status.label, "detail": page.status.detail},
            "operating_state": {
                "level": page.operating_state.level,
                "label": page.operating_state.label,
                "detail": page.operating_state.detail,
            },
            "zone_notes": {
                zone: f"{zone}: highlighting {', '.join(readings[key].spec.label for key in keys)}."
                for zone, keys in layout.ZONE_READINGS.items()
            },
        },
    }


REPORT_ID = "6f1c2a4e-2d8b-4c1e-9a57-0c3d5e7f9b21"
REPORT_NARRATIVE = (
    "Shift B needs **attention**: fuel rate 541.7 kg/tHM, above 530; ETA CO 46.2 %.\n\n"
    "Against shift A:\n- production up 1.2 t/h\n- belly spread wider\n\n"
    "Watch the belly Q4 and keep the blast at 231,000 Nm³/h."
)


def report_fixture() -> dict[str, Any]:
    """A stored report as the API serves it: the builder on the synthetic sources, the tab's document stamped at a
    fixed time, the narrative of a fake LLM."""
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    from dataclasses import replace
    from datetime import UTC, datetime

    from api.routers.reports import ReportThresholds, ShiftReportDetail
    from reports.furnace_report.builder import ShiftBuilder
    from reports.furnace_report.service import ShiftReportService
    from services import reports
    from tests.test_services_reports import SHIFT, raw_for

    timeframe = SHIFT.timeframe()
    data = ShiftBuilder().build(raw_for(timeframe))
    generated = datetime(2026, 10, 26, 16, 51, tzinfo=UTC)  # 22:21 IST, the report.shift slot
    document = replace(ShiftReportService().document(data, REPORT_NARRATIVE, timeframe), generated_at_ist=generated)
    report = reports.GeneratedReport(
        SHIFT, "closed", data, document, reports.Narrative("ok", REPORT_NARRATIVE, 1, model="fake/model-1"), generated
    )
    row = {"id": REPORT_ID, **reports.store_values(report, job_id=None, trigger="schedule", requested_by="scheduler")}
    thresholds = reports.load_thresholds()
    detail = ShiftReportDetail.model_validate(reports.detail_data(row, thresholds))
    return {
        "source": (
            "services.reports.detail_data of shift 2026-10-26_B built from tests.test_services_reports.raw_for "
            "(ShiftBuilder, ShiftReportService.document)"
        ),
        "thresholds": json.loads(
            ReportThresholds.model_validate(reports.thresholds_data(thresholds)).model_dump_json()
        ),
        "detail": json.loads(detail.model_dump_json()),
    }


OPTIMISER_RUN_ID = "3c1f7a52-9b0e-4d6a-8e21-5f4b7c9d2a10"
OPTIMISER_STEP1_ID = "8a2d4e61-0c3f-4b7a-9d15-6e2c8f1a4b37"
CASE_CAL = ROOT / "tests" / "fixtures" / "cal_2z_jul_oct_2026_case_report_15g.json"
#: The case report's controls and thermal targets (as tests/test_api_optimiser.py).
CASE_SETTINGS = {
    "hmt_c": 1470.0,
    "hm_si_pct": 0.3836,
    "pci_kg_thm": 190.0,
    "nut_coke_kg_thm": 70.0,
    "blast_moisture_g_nm3": 15.0,
    "hbt_c": 1188.4,
    "o2_per_air": 0.05209,
    "hot_blast_pressure_bar_g": 2.79,
    "top_pressure_bar_g": 1.42,
}
#: A charged blend to compare Step 2 against (wet % of the ferrous charge; the reference yard has none).
TODAY_SHARES = {"sinter_sp_02": 62.0, "lloyds_pellet": 12.0, "geomin_clo": 12.0, "lloyds_clo": 8.0, "nmdc_rom": 6.0}
TODAY_WINDOW = "from 15 Oct 06:00 to 16 Oct 06:00 IST"
STEP1_LEVELS = (2200.0, 2270.0, 2340.0)


def _rounded(value: Any, digits: int = 9) -> Any:
    if isinstance(value, dict):
        return {k: _rounded(v, digits) for k, v in value.items()}
    if isinstance(value, list):
        return [_rounded(v, digits) for v in value]
    if isinstance(value, float):
        return round(value, digits) if math.isfinite(value) else None
    return value


def _records(frame: Any) -> list[dict[str, Any]]:
    return [
        {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def _page11(
    r: dict[str, Any],
    today: dict[str, Any] | None,
    other: dict[str, Any] | None,
    names: dict[str, str],
    prices: dict[str, float],
    flux_ids: list[str],
) -> dict[str, Any]:
    """What page 11 renders for a plan (``render_plan_summary``, ``plan_sentence``, the Blend tab) from the engine's
    result."""
    from utils.burden_engine.scenarios import blend_compare_table, flux_compare_table, plan_sentence, plan_tiles
    from utils.slag_bands import PLANT_SLAG_BANDS, band_note

    tiles = plan_tiles(r, flux_ids)
    out: dict[str, Any] = {"tiles": tiles}
    notes = {}
    for key, (lo, hi) in PLANT_SLAG_BANDS.items():
        value = tiles["slag"][key]
        text, inside = band_note(value if math.isfinite(value) else None, lo, hi)
        notes[key] = {"text": text, "inside": inside}
    out["band_notes"] = notes
    others = []
    if today is not None:
        others.append(("Today", today))
        t0 = plan_tiles(today, flux_ids)
        out["today_tiles"] = t0
        out["deltas"] = {grp: {k: tiles[grp][k] - t0[grp][k] for k in tiles[grp]} for grp in ("cost", "fuel", "slag")}
        out["saving_lakh_rs_d"] = (t0["cost"]["total"] - tiles["overview"]["total"]) * tiles["overview"]["P_tday"] / 1e5
    if other is not None and abs(other["cost"]["total"] - r["cost"]["total"]) >= 0.5:
        others.append(("Ore + flux only", other))
    out["lead"] = list(plan_sentence(r, names, flux_ids, today))
    out["blend_table"] = _records(blend_compare_table(r, others, names, prices))
    out["flux_table"] = _records(flux_compare_table(r, others, names, flux_ids))
    out["donut"] = {
        "segments": [[i, float(sh), float(r["t_day"].get(i, 0.0))] for i, sh in r["share_wet"].items() if sh > 0],
        "ferrous_t_d": r["ferrous_wet"] * r["P_tday"] / 1000.0,
    }
    return out


def optimiser_fixture() -> dict[str, Any]:
    """The case report's Step 2 as the worker stores it, the result view and page 11's figures from the same run."""
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    os.environ.setdefault("COKE_ENERGY_TODAY", "2026-10-15")
    from dataclasses import replace
    from datetime import UTC, datetime

    from services import optimiser as svc
    from services.optimiser_reference import reference_view
    from services.optimiser_results import result_view
    from utils.burden_engine.engine import Limits, Settings
    from utils.burden_engine.scenarios import frontier_table, limit_labels, run_fixed

    cfg = dict(svc.read_page_config(0))
    cfg.pop("prices", None)  # the case report's own (UAT) prices
    cfg["calibration"] = dict(cfg.get("calibration") or {}, fallback_path=str(CASE_CAL))
    case = svc.default_case(
        "step2",
        omega_mode="fixed",
        production_t_d=2270.0,
        production_source="requested",
        calibration_basis="reference",
        yard_basis="reference",
        settings=CASE_SETTINGS,
        cfg=cfg,
    )
    args = list(case.args)
    args[10] = svc.as_json(TODAY_SHARES)
    case = replace(case, args=tuple(args), today_shares=dict(TODAY_SHARES), today_window=TODAY_WINDOW)
    res = svc.compute_mode(*case.args)
    started = datetime(2026, 10, 16, 0, 45, tzinfo=UTC)
    finished = datetime(2026, 10, 16, 0, 46, 12, tzinfo=UTC)
    row = {
        "id": OPTIMISER_RUN_ID,
        "trigger": "user",
        "requested_by": "sup",
        "started_at": started,
        "finished_at": finished,
        **svc.run_values(case, res),
    }
    step2 = json.loads(result_view(row).model_dump_json())

    inputs = json.loads(case.args[1])
    names = dict(case.names)
    prices = {m["id"]: float(m["price"]) for m in [*inputs["materials"], *inputs["fluxes"]]}
    flux_ids = [f["id"] for f in inputs["fluxes"]]
    r, today, r0 = (svc.jsonable(res[k]) for k in ("with_fuel", "today", "ore_only"))
    expected_step2 = _page11(r, today, r0, names, prices, flux_ids)
    vol = svc.jsonable(res["value_of_limits"])
    expected_step2["value_of_limits"] = {
        "useful": [x["change"] for x in vol if abs(x["saving_rs_thm"] or 0.0) >= 0.5],
        "rest": [x["change"] for x in vol if abs(x["saving_rs_thm"] or 0.0) < 0.5],
    }

    # Step 1 on three levels of the same case: the frontier, its cheapest and its highest point.
    eng = svc.engine(inputs, json.loads(case.args[2]))
    s = Settings(**case.args[3])
    L = Limits(**case.args[4])
    plans = [svc.jsonable(run_fixed(eng, p, s, L)["with_fuel"]) for p in STEP1_LEVELS]
    best = min(plans, key=lambda x: x["cost"]["total"])
    top = max(plans, key=lambda x: x["P_tday"])
    step1_res = {
        "mode": "cheapest",
        "best": best,
        "top": top,
        "frontier": svc.jsonable(frontier_table(plans)),
        "critical_assumptions": [],
    }
    summary = dict(case.summary(), mode="step1", production_target_t_d=None, production_source=None)
    step1_row = {
        "id": OPTIMISER_STEP1_ID,
        "trigger": "schedule",
        "requested_by": "scheduler",
        "mode": "step1",
        "omega_mode": "fixed",
        "status": "ok",
        "inputs": svc.jsonable(summary),
        "result": step1_res,
        "started_at": started,
        "finished_at": finished,
    }
    step1 = json.loads(result_view(step1_row).model_dump_json())
    dp = top["P_tday"] - best["P_tday"]
    extra = (top["cost"]["total"] * top["P_tday"] - best["cost"]["total"] * best["P_tday"]) / dp if dp > 1 else None
    expected_step1 = {
        "tiles": _page11(best, None, None, names, prices, flux_ids)["tiles"],
        "best": {
            "P_tday": best["P_tday"],
            "cost": best["cost"]["total"],
            "limited_by": limit_labels(best.get("binding", [])),
        },
        "top": {
            "P_tday": top["P_tday"],
            "cost": top["cost"]["total"],
            "limited_by": limit_labels(top.get("binding", [])),
        },
        "extra_rs_per_t": extra,
    }
    return {
        "source": (
            "services.optimiser.compute_mode on the 16 Oct 2026 case report (Fixed-ω, reference yard, "
            f"{CASE_CAL.name}) at 2,270 t/d against a charged blend, stored by run_values and served by result_view; "
            "Step 1 from run_fixed at 2,200 / 2,270 / 2,340 t/d; expected = page 11's functions on the engine result"
        ),
        "step2": _rounded(step2),
        "step1": _rounded(step1),
        "reference": json.loads(reference_view().model_dump_json()),
        "expected": _rounded({"step2": expected_step2, "step1": expected_step1}),
    }


def write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {path.relative_to(ROOT)}")


def main() -> None:
    write(RIST_OUT, rist_fixture())
    write(SCHEMATIC_OUT, schematic_fixture())
    write(STATUS_OUT, status_fixture())
    write(REPORT_OUT, report_fixture())
    write(OPTIMISER_OUT, optimiser_fixture())


if __name__ == "__main__":
    main()
