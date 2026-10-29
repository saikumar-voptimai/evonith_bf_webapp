"""Write the parity fixtures of the Svelte chart kit (SV-0.6, #275) from the Python models.

    python scripts/export_chart_fixtures.py

- ``frontend/src/lib/charts/fixtures/rist_shift.json``: one shift of ``tests/fixtures/coke_energy/shift_inputs.csv``
  through ``utils.coke_energy.rist_diagram.shift_rist`` (the Streamlit Rist diagram's input), with the points the
  diagram marks (A, D, E, W, T, E' and the pivots). ``RistDiagram.svelte`` derives the same points from the line
  parameters; the Vitest suite compares them.
- ``frontend/src/lib/furnace/fixtures/schematic.json``: the furnace outline, lining and zone dividers of
  ``ui.furnace_status.schematic.build_furnace_svg`` for the configured profile. ``FurnaceSchematic.svelte`` must
  draw the same points (acceptance criterion of #275).

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


def write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {path.relative_to(ROOT)}")


def main() -> None:
    write(RIST_OUT, rist_fixture())
    write(SCHEMATIC_OUT, schematic_fixture())


if __name__ == "__main__":
    main()
