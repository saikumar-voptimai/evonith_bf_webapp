"""BF-02 plant layout: stave rows, cooling circuits, tuyeres, taphole, thermocouples, ports and the stave-event log.

Sources (all received 21 Oct 2026):
* BF profile drawing (general layout of BF body refractories lining, Vu 630 m³), MCC Huatian Nanjing.
* BF body cooling equipment installation drawing (developed outside view; stave rows 1–15, throat armour).
* Brick laying for cooling staves (rows 6–14), cast-in brick and FC lining material drawings.
* Temperature detection point / probing hole / gas sampling hole layout plan (11 TC levels, 110 points).
* DCS "thermocouple measurement point layout" screens (tags TI-40501 … TI-40511).
* Stave_Leakage_details.xlsx (stave damage and repair log Feb 2023 – Jul 2024, tuyere changes, plant sketch).
* Stave_water_flow_tuyere_sleeve_details.xlsx (FC volumes, cooling-water flows, tuyere sleeves and blanking).

Azimuth convention (plan seen from above): degrees counter-clockwise from tuyere 1, which faces the casthouse.
Tuyere n sits at (n − 1) × 20°; the taphole at 350° (10° clockwise of tuyere 1, between tuyeres 18 and 1);
the skip-bridge side at 90°, the gravity dust-catcher side at 270°. The TC drawing, the DCS screens (tuyere 1 to
the right, tuyere 10 to the left, taphole just below tuyere 1) and the developed stave view (bridge → C.L. →
dust catcher → taphole → casthouse, left to right) agree on this.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date
import math
import re

import numpy as np

from .geometry import ELEV_TUYERE_MM, z_from_elev

N_TUYERES = 18
TAPHOLE_DEG = 350.0
BRIDGE_SIDE_DEG = 90.0
DUST_CATCHER_SIDE_DEG = 270.0
BUSTLE_ENTRY_DEG = 250.0   # plant sketch: bustle-main entry beside tuyeres 13–14 (approximate)


def tuyere_angle(n: int) -> float:
    return ((n - 1) * 20.0) % 360.0


def nearest_tuyere(angle_deg: float) -> int:
    return int(round((angle_deg % 360.0) / 20.0)) % N_TUYERES + 1


def ang_diff(a: float, b: float) -> float:
    """Signed smallest difference a − b in degrees (−180, 180]."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return 180.0 if d == -180.0 else d


# ---------------------------------------------------------------- shell (inner diameter of the steel shell)
SHELL_INNER_D_MM = [  # (elevation mm, inner diameter mm): turning points from the profile and stave drawings
    (2536, 8800), (7323, 8800), (9500, 8331), (12020, 7788), (14133, 8470), (15932, 8470), (19307, 7733.6),
    (26336, 6200), (29005, 6200),
]


def shell_inner_radius_mm(elev_mm: float) -> float:
    e = np.array([p[0] for p in SHELL_INNER_D_MM], float)
    d = np.array([p[1] for p in SHELL_INNER_D_MM], float)
    return float(np.interp(elev_mm, e, d) / 2.0)


# ---------------------------------------------------------------- stave rows
@dataclass(frozen=True)
class StaveRow:
    row: int
    bottom_mm: float
    top_mm: float
    n_staves: int
    material: str
    circuit: str
    region: str

    @property
    def centre_mm(self) -> float:
        return 0.5 * (self.bottom_mm + self.top_mm)

    @property
    def z_bottom(self) -> float:
        return float(z_from_elev(self.bottom_mm))

    @property
    def z_top(self) -> float:
        return float(z_from_elev(self.top_mm))

    @property
    def z_centre(self) -> float:
        return float(z_from_elev(self.centre_mm))

    @property
    def pitch_deg(self) -> float:
        return 360.0 / self.n_staves


# Vertical heights from the developed view (bottom to top): 1,430 · 1,430 · 1,480 · 1,240 · 1,790 · 1,398 · 1,500 ·
# 1,360 · 1,500 · 1,500 · 1,470 · 1,470 · 1,470 · 1,470 · 1,548, then throat armour 1,050 and 720, with 30 mm joints;
# anchored on the joints at +7,323 (rows 2/3, shell turning point) and +19,307 (rows 10/11).
_CI = "cast iron (HT200 / QT400-18A)"
_CS = "cast steel ZG230-450, hot- and cold-surface pipes"
_SG = "graphite (nodular) cast iron QT400-18A"
STAVE_ROWS: list[StaveRow] = [
    StaveRow(1, 4403, 5833, 36, _CI, "rows 1–5 + tuyere coolers + jackets", "hearth bottom corner"),
    StaveRow(2, 5863, 7293, 36, _CI, "rows 1–5 + tuyere coolers + jackets", "hearth (ceramic pad level)"),
    StaveRow(3, 7323, 8803, 36, _CI, "rows 1–5 + tuyere coolers + jackets", "hearth, taphole (+8,000)"),
    StaveRow(4, 8833, 10073, 34, _CI, "rows 1–5 + tuyere coolers + jackets", "upper hearth"),
    StaveRow(5, 10103, 11893, 18, _CI, "rows 1–5 + tuyere coolers + jackets", "tuyere belt (+10,900)"),
    StaveRow(6, 11923, 13321, 32, _CS, "rows 6–10 hot / cold surface", "lower bosh"),
    StaveRow(7, 13351, 14851, 32, _CS, "rows 6–10 hot / cold surface", "upper bosh – belly (bosh top +14,400)"),
    StaveRow(8, 14881, 16241, 32, _CS, "rows 6–10 hot / cold surface", "belly – shaft bottom (+15,900)"),
    StaveRow(9, 16271, 17771, 32, _CS, "rows 6–10 hot / cold surface", "lower shaft"),
    StaveRow(10, 17801, 19301, 32, _CS, "rows 6–10 hot / cold surface", "lower shaft"),
    StaveRow(11, 19331, 20801, 30, _SG, "rows 11–15", "middle shaft"),
    StaveRow(12, 20831, 22301, 28, _SG, "rows 11–15", "middle shaft"),
    StaveRow(13, 22331, 23801, 26, _SG, "rows 11–15", "upper shaft"),
    StaveRow(14, 23831, 25301, 26, _SG, "rows 11–15", "upper shaft"),
    StaveRow(15, 25331, 26879, 22, _CI, "rows 11–15", "top of shaft"),
    StaveRow(16, 26909, 27959, 18, "throat armour, cast steel ZG270-500 (lower)", "—", "throat"),
    StaveRow(17, 27989, 28709, 36, "throat armour, cast steel ZG270-500 (upper)", "—", "throat"),
]
ROW_BY_NO = {r.row: r for r in STAVE_ROWS}


def row_at_elev(elev_mm: float) -> int | None:
    for r in STAVE_ROWS:
        if r.bottom_mm - 15 <= elev_mm <= r.top_mm + 15:
            return r.row
    return None


# ---------------------------------------------------------------- cooling water (m³/h)
COOLING_CIRCUITS = {   # design, present (Stave_water_flow sheet)
    "bottom cooling": (200, 215),
    "rows 1–5 + tuyere coolers + jackets": (960, 996),
    "rows 6–10 hot surface": (700, 1100),
    "rows 6–10 cold surface": (220, 261),
    "rows 11–15": (330, 396),
}
GJ_PER_H_PER_K_PER_M3H = 1000.0 * 4.186e-6   # 1 m³/h of water warmed by 1 K = 0.004186 GJ/h


def circuit_heat_gj_h(flow_m3h: float, dT_K: float) -> float:
    return flow_m3h * dT_K * GJ_PER_H_PER_K_PER_M3H


# ---------------------------------------------------------------- thermocouples
@dataclass(frozen=True)
class TC:
    tag: str             # DCS tag, e.g. TI-40506C
    level_mm: float      # elevation rel. ±0.00
    code: str            # letter on the DCS screen
    angle_deg: float     # azimuth (convention above); nan for the furnace centre
    radius_mm: float     # from the furnace axis
    location: str        # what the tip sits in
    depth_note: str
    mapping: str         # 'drawing' | 'inferred' | 'uncertain' | 'not on screen'
    snapshot_c: float | None = None   # DCS screen photo of 21 Oct 2026 (one instant; for orientation only)

    @property
    def z(self) -> float:
        return float(z_from_elev(self.level_mm))

    @property
    def row(self) -> int | None:
        return row_at_elev(self.level_mm)

    @property
    def nearest_tuyere(self) -> int | None:
        return None if math.isnan(self.angle_deg) else nearest_tuyere(self.angle_deg)

    @property
    def from_taphole_deg(self) -> float | None:
        return None if math.isnan(self.angle_deg) else round(ang_diff(self.angle_deg, TAPHOLE_DEG), 2)


def _pairs(prefix, level, angles, r_outer, r_inner, codes, loc, depth_outer, depth_inner, snap, mapping):
    out = []
    for (c1, c2), a in zip(codes, angles):
        v1, v2 = snap.get(c1), snap.get(c2)
        out.append(TC(prefix + c1, level, c1, a, r_outer, loc, depth_outer, mapping.get(c1, "inferred"), v1))
        out.append(TC(prefix + c2, level, c2, a, r_inner, loc, depth_inner, mapping.get(c2, "inferred"), v2))
    return out


def thermocouples() -> list[TC]:
    """The 110 points of the TC layout plan, with the DCS letter codes.

    Letters: points are lettered counter-clockwise starting at the ~100° radial line (tuyere 6); on the
    hearth-wall levels the first letter of a pair is the outer (cold-side) point and the second the inner
    (hot-side) point — inferred from the readings (the first reads lower on 26 of 29 visible pairs) and the
    screen routing, to be confirmed against the DCS I/O list.
    """
    tcs: list[TC] = []
    nan = float("nan")
    # 4374: bottom of the lowest semi-graphite carbon course, 84 mm above the under-hearth cooling-pipe axis (+4,290);
    # 6 points on Φ3,380 + centre.
    s = {"A": 68.4, "B": 67.4, "C": 66.8, "D": 67.8, "E": 53.9, "F": 67.5, "G": 62.2}
    loc = "hearth bottom, under the semi-graphite carbon (2,431 mm below the hearth floor)"
    tcs.append(TC("TI-40501A", 4374, "A", 100.0, 1690, loc, "r 1,690", "uncertain", s["A"]))
    tcs.append(TC("TI-40501B", 4374, "B", nan, 0, loc, "centre", "uncertain", s["B"]))
    for c, a in zip("CDEFG", [160, 220, 280, 340, 40]):
        tcs.append(TC("TI-40501" + c, 4374, c, float(a), 1690, loc, "r 1,690", "inferred", s[c]))
    # 5411 / 5757: bottom carbon, 13 points: centre + Φ3,380 + Φ6,760 on 6 radial lines
    snaps = {
        5411: {"A": 189.6, "B": 352.7, "C": 381.7, "D": 202.0, "E": 348.2, "F": 185.1, "G": 354.6, "H": 196.8,
               "I": 354.1, "J": 176.4, "K": 356.1, "L": 200.8, "M": 350.0},
        5757: {"A": 273.0, "B": 479.9, "C": 491.2, "D": 309.7, "E": 476.8, "F": 257.1, "G": 473.5, "H": 276.7,
               "I": 196.8, "J": 287.8, "K": 470.3, "L": 230.3, "M": 483.6},
    }
    for lvl, tag, mat in [(5411, "TI-40502", "semi-graphite carbon (top of course 3)"),
                          (5757, "TI-40503", "microporous carbon (course 4)")]:
        below = 6805 - lvl
        loc = f"hearth bottom, {mat}, {below} mm below the hearth floor"
        sn = snaps[lvl]
        tcs.append(TC(tag + "A", lvl, "A", 100.0, 3380, loc, "r 3,380 (under the hearth wall)", "inferred", sn["A"]))
        tcs.append(TC(tag + "B", lvl, "B", 100.0, 1690, loc, "r 1,690", "inferred", sn["B"]))
        tcs.append(TC(tag + "C", lvl, "C", nan, 0, loc, "centre", "inferred", sn["C"]))
        tcs += _pairs(tag, lvl, [160, 220, 280, 340, 40], 3380, 1690, ["DE", "FG", "HI", "JK", "LM"], loc,
                      "r 3,380 (under the hearth wall)", "r 1,690", sn, {})
    # 6103: hearth wall-bottom corner (702 mm below the hearth floor): Φ8,100 / Φ7,800 on 6 lines + centre
    sh = shell_inner_radius_mm(6103)
    loc = "hearth bottom corner, microporous carbon, 702 mm below the hearth floor"
    sn = {"A": 224.6, "B": 648.8, "C": 115.6, "D": 78.5, "E": 280.7, "F": 118.2, "G": 164.7, "H": 125.8, "I": 193.8,
          "L": 102.4, "M": 147.7}
    tcs.append(TC("TI-40504A", 6103, "A", 105.0, 3900, loc, f"{sh - 3900:.0f} mm in from the shell", "uncertain", sn["A"]))
    tcs.append(TC("TI-40504B", 6103, "B", nan, 0, loc, "centre (only at 6103)", "uncertain", sn["B"]))
    tcs.append(TC("TI-40504C", 6103, "C", 105.0, 4050, loc, f"{sh - 4050:.0f} mm in from the shell", "uncertain", sn["C"]))
    tcs += _pairs("TI-40504", 6103, [165, 225, 285, 345, 45], 4050, 3900, ["DE", "FG", "HI", "JK", "LM"], loc,
                  f"{sh - 4050:.0f} mm in from the shell", f"{sh - 3900:.0f} mm in from the shell", sn,
                  {"J": "not on screen", "K": "not on screen"})
    # 6795: hearth wall at the ceramic-pad top (hearth floor +6,805): Φ8,100 / Φ7,800, 6 lines
    sh = shell_inner_radius_mm(6795)
    loc = "hearth wall carbon, at the hearth floor (ceramic pad top)"
    sn = {"A": 160.2, "B": 279.3, "C": 334.9, "D": 402.7, "E": 240.7, "F": 158.8, "G": 205.2, "H": 332.3,
          "K": 125.7, "L": 192.3}
    tcs += _pairs("TI-40505", 6795, [105, 165, 225, 285, 345, 45], 4050, 3900, ["AB", "CD", "EF", "GH", "IJ", "KL"],
                  loc, f"{sh - 4050:.0f} mm in from the shell", f"{sh - 3900:.0f} mm in from the shell", sn,
                  {"I": "not on screen", "J": "not on screen"})
    # 7565 / 8335: hearth wall, 7 lines (0° and 50° added), Φ8,084 / Φ7,782 (7565) and Φ7,918 / Φ7,616 (8335)
    for lvl, tag, ro, ri, sn, mp, loc in [
        (7565, "TI-40506", 4042, 3891,
         {"A": 140.2, "B": 215.9, "C": 271.7, "D": 328.3, "E": 133.5, "F": 200.3, "G": 100.8, "H": 187.4,
          "M": 115.6, "N": 170.1},
         {"I": "not on screen", "J": "not on screen", "K": "not on screen", "L": "not on screen",
          "M": "inferred (screen label 'M/L')", "N": "inferred (screen label 'M/L')"},
         "hearth wall carbon, 435 mm below the taphole"),
        (8335, "TI-40507", 3959, 3808,
         {"A": 108.9, "B": 147.7, "C": 258.0, "D": 292.5, "E": 108.1, "F": 152.1, "G": 113.3, "H": 187.7,
          "J": 42.4, "K": 177.6, "L": 71.1, "M": 103.7, "N": 136.3},
         {"I": "not on screen", "K": "inferred (reads hotter than L: swapped or taphole effect)",
          "L": "inferred (reads hotter than L: swapped or taphole effect)"},
         "hearth wall carbon, 335 mm above the taphole"),
    ]:
        sh = shell_inner_radius_mm(lvl)
        tcs += _pairs(tag, lvl, [100, 160, 220, 280, 340, 0, 50], ro, ri, ["AB", "CD", "EF", "GH", "IJ", "KL", "MN"],
                      loc, f"{sh - ro:.0f} mm in from the shell", f"{sh - ri:.0f} mm in from the shell", sn, mp)
    # 9105: upper hearth wall, 6 lines on a 34-stave pitch (10.588°), Φ7,752 / Φ7,450
    sh = shell_inner_radius_mm(9105)
    # dimensioned on the drawing: 270° + 10.5882° = 280.59°, then 63.5294° steps, and 270° − 42.3529° = 227.65°
    a9105 = [100.59, 164.12, 227.65, 280.59, 344.12, 47.65]
    sn = {"A": 102.1, "B": 139.3, "C": 251.3, "D": 280.6, "E": 96.5, "F": 133.4, "G": 114.2, "H": 155.2,
          "I": 0.0, "J": 0.0, "K": 104.2, "L": 139.2}
    tcs += _pairs("TI-40508", 9105, a9105, 3876, 3725, ["AB", "CD", "EF", "GH", "IJ", "KL"],
                  "upper hearth wall carbon, 1,795 mm below the tuyeres",
                  f"{sh - 3876:.0f} mm in from the shell", f"{sh - 3725:.0f} mm in from the shell", sn,
                  {"I": "inferred (reads 0.0: failed)", "J": "inferred (reads 0.0: failed)"})
    # 12824 / 15325 / 18717: stave-body thermocouples, 4 per level at 50.625° + k·90° (32-stave pitch)
    for lvl, tag, depth, sn, where in [
        (12824, "TI-40509", 225, {"A": 54.3, "B": 72.3, "C": 58.0, "D": 61.4}, "bosh, stave row 6"),
        (15325, "TI-40510", 190, {"A": 73.0, "B": 62.1, "C": 54.7, "D": 57.6}, "belly, stave row 8"),
        (18717, "TI-40511", 190, {"A": 54.5, "B": 40.9, "C": 33.4, "D": 39.6}, "lower shaft, stave row 10"),
    ]:
        sh = shell_inner_radius_mm(lvl)
        for c, a in zip("ABCD", [140.625, 230.625, 320.625, 50.625]):
            tcs.append(TC(tag + c, lvl, c, a, round(sh - depth), f"stave body, {where}",
                          f"{depth} mm in from the shell inner face", "drawing + screen", sn.get(c)))
    return tcs


PORTS = [  # (kind, elevation mm, count, angles deg or None)
    ("probing (clinker) hole", 18717, 4, None),
    ("probing (clinker) hole", 21517, 4, None),
    ("gas sampling hole", 25800, 4, None),
]


def tc_table():
    rows = []
    for t in thermocouples():
        d = asdict(t)
        d.update(z_m=round(t.z, 3), stave_row=t.row, nearest_tuyere=t.nearest_tuyere,
                 from_taphole_deg=t.from_taphole_deg)
        rows.append(d)
    return rows


def hearth_wall_pairs(level_mm: float, k_carbon_w_mk: float = 18.0):
    """Heat flux through the hearth wall from each outer/inner pair at one level (snapshot readings).

    q = k ΔT / Δr (W/m²); k of microporous carbon 15–20 W/m K at 100–300 °C (default 18). Pairs with the
    outer point hotter than the inner are flagged (swapped wiring, a gap, or a local effect)."""
    tcs = [t for t in thermocouples() if t.level_mm == level_mm and not math.isnan(t.angle_deg)]
    by_angle: dict[float, list[TC]] = {}
    for t in tcs:
        by_angle.setdefault(t.angle_deg, []).append(t)
    out = []
    for a, pair in sorted(by_angle.items()):
        if len(pair) != 2 or any(p.snapshot_c in (None, 0.0) for p in pair):
            out.append(dict(angle=a, ok=False))
            continue
        o, i = sorted(pair, key=lambda p: -p.radius_mm)
        dT = i.snapshot_c - o.snapshot_c
        dr = (o.radius_mm - i.radius_mm) / 1000.0
        out.append(dict(angle=a, tuyere=nearest_tuyere(a), outer=o.code, inner=i.code,
                        t_outer=o.snapshot_c, t_inner=i.snapshot_c, dT=dT, q_kw_m2=k_carbon_w_mk * dT / dr / 1000.0,
                        ok=dT > 0))
    return out


# ---------------------------------------------------------------- tuyeres: sleeves and blanking
TUYERE_EVENTS = [  # (date, tuyeres, bore mm or 0 for blanked or None for back to normal, note)
    (date(2023, 4, 21), (12, 17), 100, "100 mm sleeve installed"),
    (date(2023, 8, 4), (12, 17), None, "sleeves removed; T18 changed"),
    (date(2023, 8, 22), (), None, "T9 and T12 changed"),
    (date(2024, 2, 3), (12, 13), 0, "T12 and T13 blanked: 16-tuyere operation"),
    (date(2024, 3, 22), (12, 17), 90, "90 mm sleeves on T12 and T17; T12, T13 reopened; cigar coolers on R7 staves 9 and 11"),
    (date(2024, 3, 22), (13,), None, "T13 reopened"),
    (date(2024, 5, 4), (13, 14), 90, "90 mm sleeves on T13 and T14 (four sleeved: 12, 13, 14, 17)"),
]


def tuyere_bores(on: date, normal_mm: float = 115.0) -> dict[int, float]:
    """Bore of each tuyere on a date from the sleeve/blanking log (0 = blanked). `normal_mm` is NOT in the data
    received: it is an assumption until the tuyere drawing or stores record gives it."""
    b = {n: normal_mm for n in range(1, N_TUYERES + 1)}
    for d, tuy, bore, _ in sorted(TUYERE_EVENTS, key=lambda e: e[0]):
        if d > on:
            break
        for n in tuy:
            b[n] = normal_mm if bore is None else float(bore)
    return b


# ---------------------------------------------------------------- stave events (Feb 2023 – Jul 2024)
@dataclass(frozen=True)
class StaveEvent:
    found: date
    rows: tuple[int, ...]
    stave: int
    face: str            # 'hot' or 'cold'
    tuyere: int          # 'adjacent tuyere no' in the log
    what: str
    action: str

    @property
    def angle_deg(self) -> float:
        """Azimuth from the logged adjacent tuyere (±20–40°: the damage and repair tables differ by 1–4 tuyeres
        for staves 21, 28 and 30). The stave numbers are not used for azimuth: in the log, stave
        numbers rise counter-clockwise in rows 6 and 8 but clockwise in row 7 (e.g. R7/8 ↔ T14, R7/21 ↔ T7,
        R7/30 ↔ T2), so the numbering origin differs by row or by recorder."""
        return tuyere_angle(self.tuyere)

    @property
    def elev_mm(self) -> float:
        rs = [ROW_BY_NO[r] for r in self.rows]
        return 0.5 * (min(r.bottom_mm for r in rs) + max(r.top_mm for r in rs))


STAVE_EVENTS: list[StaveEvent] = [
    StaveEvent(date(2023, 2, 4), (6,), 7, "cold", 4, "flame test: leak at the row 6 cold-surface inlet pipe", "N2; S/D 6 Feb split the stave from row 7 circuit; repaired, water back 1 Sep 23"),
    StaveEvent(date(2023, 3, 17), (8, 9, 10), 27, "cold", 15, "flame captured (rows 8–10 cold surface)", "N2; staves split 25 Apr; pipe repaired, water back 1 Sep 23"),
    StaveEvent(date(2023, 4, 25), (7,), 8, "hot", 14, "pressure drop in pressure test", "N2 then N2 + water mist; grouted 1 Sep 23"),
    StaveEvent(date(2023, 6, 18), (7,), 21, "hot", 7, "0.3 bar/min pressure drop (12 Jun); flame 18 Jun", "mist cooling; grouted 16 Oct 23"),
    StaveEvent(date(2023, 8, 22), (7,), 9, "hot", 13, "pressure not holding (4 and 22 Aug)", "mist cooling; grouted 19 Oct 23"),
    StaveEvent(date(2023, 8, 24), (7,), 30, "hot", 2, "pressure drop 4.5 → 4 bar then holds", "mist cooling; grouted 1 Sep 23 (the repair table gives tuyere 6 for this stave; the plant sketch puts it by T2–T3)"),
    StaveEvent(date(2023, 10, 26), (7,), 11, "hot", 12, "pressure drop 4.5 → 1 bar", "water 50 %; grouted 24 Dec 23"),
    StaveEvent(date(2023, 10, 26), (8,), 28, "hot", 16, "pressure drop 4.5 → 2 bar; flame", "mist; water 50 %; grouted 8 Jan 24"),
    StaveEvent(date(2024, 1, 8), (7,), 10, "hot", 12, "flame captured", "N2 at 3 points; grouted 8 Jan 24"),
    StaveEvent(date(2024, 1, 26), (7,), 11, "cold", 12, "flame captured (cold surface)", "outlet pipe jammed: dummy plate welded in shell 3 Feb 24"),
    StaveEvent(date(2024, 2, 2), (7,), 9, "cold", 13, "pressure drop (cold surface)", "grouted 3 Feb 24"),
    StaveEvent(date(2024, 4, 5), (6,), 13, "cold", 7, "water droplets and jet at the outlet compensator", "mist; compensator changed 3 May 24, water on"),
    StaveEvent(date(2024, 5, 8), (7,), 7, "hot", 14, "pressure drop to 2.5 bar (3 May), 0.5 bar (15 Jun); minor flame", "water kept on"),
    StaveEvent(date(2024, 6, 12), (6,), 23, "hot", 13, "flame captured", "water kept at 70 %"),
    StaveEvent(date(2024, 6, 14), (8,), 23, "hot", 13, "pressure drop to 2.5 bar, then 0.5 bar", "water kept on"),
    StaveEvent(date(2024, 6, 14), (7,), 8, "cold", 13, "pressure drop to 2.5 bar, then 0.5 bar", "—"),
    StaveEvent(date(2024, 6, 27), (6,), 22, "hot", 13, "flame captured", "water stopped, mist cooling"),
    StaveEvent(date(2024, 6, 28), (7,), 6, "hot", 14, "flame captured", "water stopped, mist cooling"),
    StaveEvent(date(2024, 7, 5), (7,), 10, "cold", 13, "flame captured", "water stopped, mist cooling"),
    StaveEvent(date(2024, 7, 5), (7,), 5, "hot", 17, "flame captured", "water stopped, mist cooling"),
    StaveEvent(date(2024, 7, 6), (8,), 26, "hot", 15, "flame captured", "water stopped, mist cooling"),
]

# Plant notes on the sketch (Diagram (2)): low stave-refractory temperature and low water ΔT, i.e. an inactive wall /
# accretion, at the belly (row 8) on the tuyere 2–8 side and at the lower stack (row 10) on the tuyere 16–2 side.
COLD_WALL_NOTES = [
    dict(rows=(8,), tuyeres=(2, 3, 4, 5, 6, 7, 8), note="belly: low refractory temperature and water ΔT"),
    dict(rows=(10,), tuyeres=(16, 17, 18, 1, 2), note="lower stack: low refractory temperature and water ΔT"),
]


def event_summary():
    ev = STAVE_EVENTS
    by_row: dict[int, dict[str, int]] = {}
    for e in ev:
        r = e.rows[0]
        by_row.setdefault(r, {"hot": 0, "cold": 0})[e.face] += 1
    in_sector = [e for e in ev if 12 <= e.tuyere <= 17]
    return dict(
        n=len(ev), hot=sum(e.face == "hot" for e in ev), cold=sum(e.face == "cold" for e in ev),
        by_row=by_row, in_T12_T17=len(in_sector),
        first=min(e.found for e in ev), last=max(e.found for e in ev),
    )


def read_stave_log(path: str) -> list[dict]:
    """Re-read the damage table of Stave_Leakage_details.xlsx (sheet 'stave damage details') into plain rows, so an
    updated log can be compared with STAVE_EVENTS. Returns dicts with date, stave, row text, tuyere, remarks."""
    import openpyxl
    ws = openpyxl.load_workbook(path, data_only=True)["stave damage details"]
    out = []
    for row in ws.iter_rows(min_row=4, values_only=True):
        d, stave, rowtxt, tuy, rem = row[3:8]
        if d is None or stave is None:
            continue
        m = re.findall(r"\d+", str(rowtxt))
        out.append(dict(date=d, stave=stave, row_text=rowtxt, rows=[int(x) for x in m],
                        face="cold" if "cold" in str(rowtxt).lower() else "hot", tuyere=tuy, remarks=rem))
    return out
