"""Plain statistics of the FurnaceMind plant skills (``plant_skills``): the usual value and spread of a baseline, how
far a reading sits from it, shift grouping, a straight-line trend and Markdown tables. Pure functions on pandas
objects, no data access and no model.

"Usual spread" = 1.4826 x the median absolute deviation (the standard deviation of a normal sample, robust to the
odd bad reading); when every baseline value is the same it falls back to the standard deviation, then to none.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

MAD_TO_SD = 1.4826
MINUS = "−"
NBSP = " "
SHIFT_HOURS = {6: "A", 14: "B", 22: "C"}


def numbers(values: Any) -> pd.Series:
    """``values`` as a float series without NaN and infinities."""
    s = pd.to_numeric(
        pd.Series(values, dtype="object") if not isinstance(values, pd.Series) else values, errors="coerce"
    )
    s = s.astype(float)
    return s[np.isfinite(s)]


def center_spread(values: Any) -> tuple[float | None, float | None]:
    """Median and usual spread of ``values`` (None when there are too few values)."""
    s = numbers(values)
    if s.empty:
        return None, None
    center = float(s.median())
    if len(s) < 3:
        return center, None
    spread = float((s - center).abs().median()) * MAD_TO_SD
    if spread <= 0:
        sd = float(s.std(ddof=1))
        spread = sd if sd > 0 else 0.0
    return center, (spread if spread > 0 else None)


def spreads_from(value: float | None, center: float | None, spread: float | None) -> float | None:
    """How many usual spreads ``value`` sits from ``center`` (signed)."""
    if value is None or center is None or not spread or not math.isfinite(value):
        return None
    return (value - center) / spread


def quantity(key: str, quantities: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Label, unit, decimals and better-direction of ``key`` (the key itself when it is not configured)."""
    meta = dict(quantities.get(key) or {})
    return {
        "label": meta.get("label", key),
        "unit": meta.get("unit", ""),
        "digits": int(meta.get("digits", 2)),
        "better": meta.get("better", "none"),
    }


def clean(df: pd.DataFrame, quantities: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    """``df`` with every reading outside its quantity's ``valid: [low, high]`` set to NaN (a stuck tag, a stoppage's
    zeros), before any averaging."""
    if df is None or df.empty:
        return df
    out = df.copy()
    for key, meta in quantities.items():
        valid = (meta or {}).get("valid")
        if key in out.columns and valid:
            col = pd.to_numeric(out[key], errors="coerce")
            out[key] = col.where((col >= float(valid[0])) & (col <= float(valid[1])))
    return out


@dataclass(frozen=True)
class Reading:
    """A quantity now against its usual value: ``usual`` the baseline median, ``low``-``high`` its 10th-90th
    percentile, ``spreads`` the distance from usual in usual spreads, ``n`` the baseline count."""

    key: str
    label: str
    unit: str
    digits: int
    better: str
    now: float | None
    usual: float | None
    low: float | None
    high: float | None
    spreads: float | None
    n: int
    #: A change smaller than this is usual however small the usual spread (a very steady flow), same unit.
    floor: float | None = None

    @property
    def change(self) -> float | None:
        if self.now is None or self.usual is None:
            return None
        return self.now - self.usual

    def level(self, unusual: float, far: float) -> str:
        """``usual``, ``unusual`` or ``far`` by the size of ``spreads`` (usual below the materiality ``floor``)."""
        if self.spreads is None:
            return "usual"
        if self.floor is not None and self.change is not None and abs(self.change) < self.floor:
            return "usual"
        size = abs(self.spreads)
        return "far" if size >= far else "unusual" if size >= unusual else "usual"

    def direction(self) -> str:
        """``better`` / ``worse`` when the quantity has a better direction and moved, else ``neutral``."""
        if self.change is None or self.better not in ("lower", "higher") or self.change == 0:
            return "neutral"
        up = self.change > 0
        return "better" if (up and self.better == "higher") or (not up and self.better == "lower") else "worse"

    def state(self, unusual: float, far: float) -> str:
        """Plain words: ``usual``, ``unusual, higher (worse)``, ``far outside, lower`` ..."""
        level = self.level(unusual, far)
        if level == "usual":
            return "usual"
        side = "higher" if (self.change or 0) > 0 else "lower"
        word = "far outside" if level == "far" else "unusual"
        direction = self.direction()
        return f"{word}, {side}" + (f" ({direction})" if direction != "neutral" else "")


def compare(key: str, now: Any, baseline: Any, quantities: Mapping[str, Mapping[str, Any]]) -> Reading:
    """``now`` (a value, or values to average) against the ``baseline`` values of ``key``."""
    if isinstance(now, (int, float, np.floating)) or now is None:
        current = float(now) if now is not None and math.isfinite(float(now)) else None
    else:
        s = numbers(now)
        current = float(s.mean()) if not s.empty else None
    base = numbers(baseline)
    center, spread = center_spread(base)
    meta = quantity(key, quantities)
    floor_pct = (quantities.get(key) or {}).get("floor_pct")
    floor = abs(center) * float(floor_pct) / 100.0 if floor_pct is not None and center is not None else None
    return Reading(
        key=key,
        now=current,
        usual=center,
        low=float(base.quantile(0.1)) if len(base) >= 3 else None,
        high=float(base.quantile(0.9)) if len(base) >= 3 else None,
        spreads=spreads_from(current, center, spread),
        n=int(len(base)),
        floor=floor,
        **meta,
    )


def fmt(value: float | None, digits: int = 1, unit: str = "") -> str:
    """``1,234.5 kg/tHM`` (no-break space before the unit, U+2212 minus); an em dash when missing."""
    if value is None or not math.isfinite(float(value)):
        return "—"
    text = f"{float(value):,.{max(0, digits)}f}".replace("-", MINUS)
    return f"{text}{NBSP}{unit}" if unit else text


def fmt_signed(value: float | None, digits: int = 1, unit: str = "") -> str:
    """``+1.2 %`` / ``−3.4 %``; an em dash when missing."""
    if value is None or not math.isfinite(float(value)):
        return "—"
    sign = "+" if value > 0 else MINUS if value < 0 else "±"
    body = fmt(abs(float(value)), digits, unit)
    return f"{sign}{body}"


def fmt_spreads(value: float | None) -> str:
    """``+2.4×`` usual spread; an em dash when unknown."""
    if value is None or not math.isfinite(value):
        return "—"
    return fmt_signed(value, 1) + "×"


def md_table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """A Markdown table (cells as given, pipes escaped)."""

    def cell(v: Any) -> str:
        return str(v).replace("|", "/").replace("\n", " ")

    lines = ["| " + " | ".join(cell(h) for h in header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


READING_HEADER = ("Quantity", "Now", "Usual", "Usual range (p10–p90)", "Change", "× usual spread", "State")


def reading_row(r: Reading, unusual: float, far: float) -> list[str]:
    rng = f"{fmt(r.low, r.digits)}–{fmt(r.high, r.digits)}" if r.low is not None and r.high is not None else "—"
    return [
        r.label,
        fmt(r.now, r.digits, r.unit),
        fmt(r.usual, r.digits, r.unit),
        rng,
        fmt_signed(r.change, r.digits, r.unit),
        fmt_spreads(r.spreads),
        r.state(unusual, far),
    ]


def readings_table(readings: Iterable[Reading], unusual: float, far: float) -> str:
    return md_table(READING_HEADER, [reading_row(r, unusual, far) for r in readings])


def ranked(readings: Iterable[Reading]) -> list[Reading]:
    """Readings by distance from usual, largest first (unknown last)."""
    return sorted(readings, key=lambda r: -abs(r.spreads) if r.spreads is not None else math.inf)


# ---------------------------------------------------------------------------------------------- time and shifts
def ist_naive(index: pd.Index) -> pd.DatetimeIndex:
    """``index`` as tz-naive IST timestamps."""
    idx = pd.DatetimeIndex(pd.to_datetime(index))
    if idx.tz is not None:
        idx = idx.tz_convert("Asia/Kolkata").tz_localize(None)
    return idx


def with_ist_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = ist_naive(out.index)
    return out.sort_index()


def shift_start(index: pd.Index) -> pd.DatetimeIndex:
    """The start of the shift (A 06:00, B 14:00, C 22:00 IST) of every timestamp (tz-naive IST)."""
    idx = ist_naive(index)
    return (idx - pd.Timedelta(hours=6)).floor("8h") + pd.Timedelta(hours=6)


def shift_name(start: pd.Timestamp) -> str:
    """``13 Nov B`` for the shift starting at ``start``."""
    return f"{start.day} {start.strftime('%b')} {SHIFT_HOURS.get(start.hour, '?')}"


def shift_means(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Mean of ``cols`` per shift (index = shift start, tz-naive IST); columns missing in ``df`` are left out."""
    present = [c for c in cols if c in df.columns]
    if df.empty or not present:
        return pd.DataFrame(columns=present)
    data = df[present].apply(pd.to_numeric, errors="coerce")
    return data.groupby(shift_start(df.index)).mean()


def slope_per_hour(values: pd.Series) -> float | None:
    """Straight-line trend of ``values`` (indexed by time) in units per hour; None for fewer than 3 points."""
    s = numbers(values)
    if len(s) < 3:
        return None
    t = (ist_naive(s.index) - ist_naive(s.index)[0]).total_seconds() / 3600.0
    if float(np.ptp(t)) <= 0:
        return None
    return float(np.polyfit(np.asarray(t, dtype=float), s.to_numpy(dtype=float), 1)[0])


def day_means(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Mean of ``cols`` per IST day (index = day start); missing columns left out."""
    present = [c for c in cols if c in df.columns]
    if df.empty or not present:
        return pd.DataFrame(columns=present)
    data = df[present].apply(pd.to_numeric, errors="coerce")
    return data.groupby(ist_naive(df.index).normalize()).mean()
