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

``tests/test_chart_fixtures.py`` fails when a committed fixture differs from this output.
"""

from __future__ import annotations

import json
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


def write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {path.relative_to(ROOT)}")


def main() -> None:
    write(RIST_OUT, rist_fixture())
    write(SCHEMATIC_OUT, schematic_fixture())
    write(STATUS_OUT, status_fixture())
    write(REPORT_OUT, report_fixture())


if __name__ == "__main__":
    main()
