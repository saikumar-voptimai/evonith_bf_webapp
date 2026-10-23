"""Plant wall signals → model quantities.

* Stave heat-load tags `heat_load_r{row}_q{quadrant}` (GJ/h; each row tag is the sum of its four quadrant tags,
  plant confirmation 15 Oct 2026) are placed at the elevations of stave rows 1–15 from the cooling-equipment
  drawing (plant.STAVE_ROWS). The row profile anchors the cohesive-zone root at the wall.
* Stave-body thermocouples at +12,824 (row 6), +15,325 (row 8) and +18,717 (row 10), 4 per level, give a second,
  coarser root indicator and a circumferential index.
* Cooling-water circuits: total and zone-wise heat from flow × ΔT, and the split of a row-resolved loss into the
  two-zone energy model's lower and upper zones at the thermal-reserve-zone height.

Quadrants: the tag names do not say which azimuths q1…q4 cover. QUADRANT_CENTRES_DEG is an ASSUMPTION
(q1 centred on 45°, then counter-clockwise, i.e. q1 = tuyeres 1–5, q2 = 6–9, q3 = 10–14, q4 = 15–18) until the
DCS configuration is checked.
"""
from __future__ import annotations

import re
import numpy as np

from .plant import ROW_BY_NO, STAVE_ROWS, COOLING_CIRCUITS, circuit_heat_gj_h, thermocouples, nearest_tuyere
from .geometry import z_from_elev
from .cohesive import root_from_heat_loads

QUADRANT_CENTRES_DEG = {1: 45.0, 2: 135.0, 3: 225.0, 4: 315.0}   # ASSUMED, see module note


def row_elevations() -> dict[str, dict]:
    """{'r7': {'bottom_mm':…, 'top_mm':…, 'centre_mm':…, 'z_centre':…}} for every stave row."""
    return {f"r{r.row}": dict(bottom_mm=r.bottom_mm, top_mm=r.top_mm, centre_mm=r.centre_mm,
                              z_bottom=r.z_bottom, z_top=r.z_top, z_centre=r.z_centre, n_staves=r.n_staves)
            for r in STAVE_ROWS}


def parse_heat_load_tags(values: dict[str, float]) -> tuple[dict[int, float], dict[int, dict[int, float]]]:
    """Split {'heat_load_r7': 1.2, 'heat_load_r7_q2': 0.4, …} into row totals and row × quadrant values.
    A row total missing from the input is the sum of its quadrants."""
    rows: dict[int, float] = {}
    quads: dict[int, dict[int, float]] = {}
    for tag, v in values.items():
        m = re.match(r".*?r(\d+)(?:_q(\d+))?$", tag)
        if not m or v is None:
            continue
        r = int(m.group(1))
        if m.group(2):
            quads.setdefault(r, {})[int(m.group(2))] = float(v)
        else:
            rows[r] = float(v)
    for r, q in quads.items():
        rows.setdefault(r, sum(q.values()))
    return rows, quads


def cz_root_from_rows(row_loads: dict[int, float], rise_frac: float = 0.6) -> float | None:
    """Cohesive-zone root at the wall (m above the tuyeres) from the row heat loads, using the row centre heights.

    Rows 1–5 (hearth and tuyere belt) are excluded: their load is set by the hearth and the tuyere coolers, not by
    the zone. Loads are per m² of stave face (row height × local circumference), so that tall or wide rows are not
    favoured."""
    use = {r: v for r, v in row_loads.items() if r in ROW_BY_NO and 6 <= r <= 15 and v is not None}
    if len(use) < 3:
        return None
    from .geometry import bf02_profile
    prof = bf02_profile()
    flux, elev = {}, {}
    for r, v in use.items():
        row = ROW_BY_NO[r]
        area = 2 * np.pi * float(prof.radius(row.z_centre)) * (row.top_mm - row.bottom_mm) / 1000.0
        flux[f"r{r}"] = v / area
        elev[f"r{r}"] = row.z_centre
    return root_from_heat_loads(flux, elev, rise_frac=rise_frac)


def stave_tc_levels(values: dict[str, float] | None = None) -> list[dict]:
    """Mean and spread of the stave-body TCs at +12,824 / +15,325 / +18,717. `values` maps tags (TI-40509A …)
    to °C; default: the DCS snapshot in the register."""
    out = []
    tcs = [t for t in thermocouples() if t.tag[:8] in ("TI-40509", "TI-40510", "TI-40511")]
    for lvl in (12824, 15325, 18717):
        pts = [t for t in tcs if t.level_mm == lvl]
        v = [(values or {}).get(t.tag, t.snapshot_c) for t in pts]
        v = [x for x in v if x is not None]
        out.append(dict(level_mm=lvl, z=float(z_from_elev(lvl)), row=pts[0].row, mean_c=float(np.mean(v)),
                        spread_c=float(np.max(v) - np.min(v)),
                        hottest=pts[int(np.argmax([(values or {}).get(t.tag, t.snapshot_c) or -1 for t in pts]))].tag,
                        hottest_tuyere=nearest_tuyere(pts[int(np.argmax([(values or {}).get(t.tag, t.snapshot_c) or -1 for t in pts]))].angle_deg)))
    return out


def quadrant_index(quads: dict[int, dict[int, float]], rows=(6, 7, 8)) -> dict[int, float]:
    """Share of the heat load of `rows` carried by each quadrant (0.25 = even)."""
    tot = {q: 0.0 for q in (1, 2, 3, 4)}
    for r in rows:
        for q, v in quads.get(r, {}).items():
            tot[q] = tot.get(q, 0.0) + v
    s = sum(tot.values())
    return {q: (v / s if s > 0 else float("nan")) for q, v in tot.items()}


# ---------------------------------------------------------------- cooling water and the two-zone loss split
def cooling_water_summary(dT_by_circuit: dict[str, float] | None = None, present: bool = True) -> dict:
    """Flows (m³/h) per circuit, GJ/h per kelvin of rise, and the heat if ΔT per circuit is given."""
    out = {}
    tot_flow = 0.0
    for c, (design, now) in COOLING_CIRCUITS.items():
        f = now if present else design
        tot_flow += f
        d = dict(flow_m3h=f, gj_h_per_K=circuit_heat_gj_h(f, 1.0))
        if dT_by_circuit and c in dT_by_circuit:
            d["heat_gj_h"] = circuit_heat_gj_h(f, dT_by_circuit[c])
        out[c] = d
    out["total"] = dict(flow_m3h=tot_flow, gj_h_per_K=circuit_heat_gj_h(tot_flow, 1.0))
    return out


def zone_split(row_loads: dict[int, float], z_boundary: float, extra_lower_gj_h: float = 0.0,
               extra_upper_gj_h: float = 0.0) -> dict:
    """Split row heat loads (GJ/h) at the lower edge of the thermal reserve zone (m above the tuyeres).
    A row straddling the boundary is split by height. `extra_*` carry circuits without row tags (bottom
    cooling, tuyere coolers → lower; throat → upper)."""
    lower, upper = extra_lower_gj_h, extra_upper_gj_h
    for r, v in row_loads.items():
        row = ROW_BY_NO.get(r)
        if row is None or v is None:
            continue
        if row.z_top <= z_boundary:
            lower += v
        elif row.z_bottom >= z_boundary:
            upper += v
        else:
            f = (z_boundary - row.z_bottom) / (row.z_top - row.z_bottom)
            lower += f * v
            upper += (1 - f) * v
    tot = lower + upper
    return dict(lower_gj_h=lower, upper_gj_h=upper, lower_share=lower / tot if tot else float("nan"))


def circuit_zone_split(z_boundary: float, dT_by_circuit: dict[str, float] | None = None) -> dict:
    """Lower/upper split of the cooling-water heat by circuit, with a row-height pro-rata for the rows 6–10
    circuits. Without ΔT data every circuit is given the same rise, so the result is the flow-weighted split
    (an indication, not a measurement)."""
    dT = dT_by_circuit or {c: 1.0 for c in COOLING_CIRCUITS}
    q = {c: circuit_heat_gj_h(f[1], dT.get(c, 1.0)) for c, f in COOLING_CIRCUITS.items()}
    lower = q["bottom cooling"] + q["rows 1–5 + tuyere coolers + jackets"]
    upper = q["rows 11–15"] if ROW_BY_NO[11].z_bottom >= z_boundary else 0.0
    if upper == 0.0:
        lower += q["rows 11–15"]
    rows_6_10 = [ROW_BY_NO[r] for r in range(6, 11)]
    h_tot = sum(r.top_mm - r.bottom_mm for r in rows_6_10)
    h_low = sum(max(0.0, min(r.z_top, z_boundary) - r.z_bottom) * 1000 for r in rows_6_10)
    f_low = h_low / h_tot
    q610 = q["rows 6–10 hot surface"] + q["rows 6–10 cold surface"]
    lower += f_low * q610
    upper += (1 - f_low) * q610
    return dict(lower_gj_h=lower, upper_gj_h=upper, lower_share=lower / (lower + upper), rows_6_10_lower_fraction=f_low)
