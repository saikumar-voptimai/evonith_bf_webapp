"""FurnaceMind's plant skills (user decision 13 Nov 2026: they replace Unit Cost, Shift to Best and Heatloads).

Seven quick skills, each a fixed analysis of the plant's own records whose numbers are computed here; the language
model only writes the operator note from them. Nothing here reads the live twin, the optimisers or a model state, and
nothing here explains how a V-OptimAIse model works (``skill_guard``):

==================  =====================================================================================
shift_pulse         the last completed shift against the 21 shifts before it
thermal_watch       hot-metal temperature and Si over the last 24 h against the plant limits, what moved before
gas_flow_lens       wall temperatures A-D, uptakes, top gas and pressures, last 8 h against the week before
wall_heat_map       stave heat load by row and quadrant, the staves that rose most, cooling water, 8 h against 14 d
slag_burden_check   slag chemistry, the charged blend and the sinter, last 24 h against the week before
fuel_drivers        what separated the highest-fuel days from the lowest of the last 30 days, with the plant facts
trial_board         the plant facts not yet settled on BF-02, grouped by the trial that would settle them, and why
==================  =====================================================================================

Data: ``PlantData`` - the FurnaceMind fetch tools for the process tags (so the datasets join the conversation),
``data.coke_energy.sources.fetch_hm_slag`` for the hot-metal and slag samples and ``furnace_data.offline`` for the
raw charge record and the sinter laboratory. Settings: ``config/furnacemind_skills.yml``; plant limits from
``config/shift_report.yml`` (``status_thresholds``) and ``utils.slag_bands``. Each skill returns a ``SkillOutput``;
``run`` stores its chart in the tools' state for the ``show_skill_chart`` tool and returns the prompt.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml
from plotly.subplots import make_subplots

from agents.furnacemind import plant_facts, tool_state
from agents.furnacemind import skill_stats as st
from agents.furnacemind.plant_facts import config

log = logging.getLogger(__name__)

IST = "Asia/Kolkata"
#: Tools' state key of the skills' charts: ``{chart_id: plotly JSON}`` (read by ``show_skill_chart``).
CHARTS = "fm_skill_charts"
_DATASET_ID = re.compile(r"dataset_id=([A-Za-z0-9_\-]+)")
_SHIFT_REPORT = Path(__file__).resolve().parents[2] / "config" / "shift_report.yml"

# Chart colours of the skills (Plotly figures in the chat; the SPA's token rules apply to its own components).
WORSE, BETTER, NEUTRAL, BAND = "#B45309", "#15803D", "#64748B", "#94A3B8"
SECTOR_COLOURS = ("#2563EB", "#D97706", "#0D9488", "#9333EA")


@dataclass
class SkillOutput:
    """The prompt of one skill run and its chart (``chart_id`` names it for ``show_skill_chart``)."""

    prompt: str
    chart: go.Figure | None = None
    chart_id: str = ""


@dataclass
class Notes:
    """What a run could not read, and the datasets it left in the conversation."""

    missing: list[str] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)


# ------------------------------------------------------------------------------------------------------------ data
def _utc_iso(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).tz_localize(IST).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def _naive_ist(index: pd.Index) -> pd.DatetimeIndex:
    return st.ist_naive(index)


class PlantData:
    """The skills' reads. Every method returns an empty frame and a reason instead of raising."""

    def now(self) -> pd.Timestamp:
        return pd.Timestamp.now(tz=IST).tz_localize(None).floor("min")

    def online(
        self, start: pd.Timestamp, end: pd.Timestamp, groups: Sequence[str], window: str
    ) -> tuple[pd.DataFrame, str | None, str | None]:
        """Process tags between ``start`` and ``end`` (IST) averaged over ``window``, through ``fetch_online_data``
        (the dataset joins the conversation): frame (IST index), dataset id, error."""
        from agents import furnace_tools as ft

        state = tool_state.current()
        state.pop(tool_state.DATAFRAME, None)
        try:
            message = ft.fetch_online_data(
                start_time_utc=_utc_iso(start),
                end_time_utc=_utc_iso(end),
                window=window,
                measurement_groups=list(groups),
            )
        except Exception as exc:  # noqa: BLE001 - reported in the prompt
            return pd.DataFrame(), None, f"{type(exc).__name__}: {exc}"
        match = _DATASET_ID.search(str(message))
        df = state.get(tool_state.DATAFRAME)
        if not match or not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame(), None, str(message)[:200]
        return st.with_ist_index(df), match.group(1), None

    def hot_metal(self, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, str | None]:
        """Hot-metal and slag samples (index = sample time, IST)."""
        try:
            from data.coke_energy.sources import fetch_hm_slag

            hm = fetch_hm_slag(start, end)
        except Exception as exc:  # noqa: BLE001
            return pd.DataFrame(), f"{type(exc).__name__}: {exc}"
        if hm is None or hm.empty:
            return pd.DataFrame(), "no hot-metal sample in the window"
        out = hm.copy()
        if "t" in out.columns:
            out.index = pd.to_datetime(out["t"])
        out.index = _naive_ist(out.index)
        return out.sort_index(), None

    def hmt_taps(self, hm: pd.DataFrame) -> pd.Series:
        """Hot-metal temperature per tap (°C), the app's tap filter."""
        if hm.empty:
            return pd.Series(dtype=float)
        try:
            from utils.coke_energy.hmt_forecast import taps

            frame = hm.reset_index(drop=True)
            if "t" not in frame.columns:
                frame["t"] = hm.index
            temps = taps(frame)
        except Exception as exc:  # noqa: BLE001
            log.warning("tap filter unavailable: %s", exc)
            return pd.Series(dtype=float)
        temps.index = _naive_ist(temps.index)
        return temps.sort_index()

    def table(self, name: str, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, str | None]:
        """Raw rows of an offline table (``offline_feed.charge_data``, ``offline_feed.sinter_chemistry``)."""
        try:
            from furnace_data.offline import fetch_offline_data

            utc = (
                pd.Timestamp(start).tz_localize(IST).tz_convert("UTC").to_pydatetime(),
                pd.Timestamp(end).tz_localize(IST).tz_convert("UTC").to_pydatetime(),
            )
            df = fetch_offline_data(name, utc, "ts")
        except Exception as exc:  # noqa: BLE001
            return pd.DataFrame(), f"{type(exc).__name__}: {exc}"
        if df is None or df.empty:
            return pd.DataFrame(), "no row in the window"
        out = df.copy()
        out.index = _naive_ist(out.index)
        return out.sort_index(), None


# --------------------------------------------------------------------------------------------------------- helpers
@lru_cache(maxsize=1)
def plant_limits() -> dict[str, Any]:
    """The plant's KPI limits of the shift reports (``shift_report.yml -> status_thresholds``)."""
    try:
        with _SHIFT_REPORT.open(encoding="utf-8") as fh:
            return dict((yaml.safe_load(fh) or {}).get("status_thresholds") or {})
    except OSError:
        return {}


def slag_bands() -> dict[str, tuple[float, float]]:
    from utils.slag_bands import PLANT_SLAG_BANDS

    return {"slag_basicity": PLANT_SLAG_BANDS["b2"], "slag_t_basicity": PLANT_SLAG_BANDS["t_basicity"]}


def _spread_cfg(cfg: Mapping[str, Any]) -> tuple[float, float]:
    s = cfg.get("spread", {})
    return float(s.get("unusual", 2.0)), float(s.get("far", 3.0))


def _online(
    data: PlantData,
    quantities: Mapping[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
    groups: Sequence[str],
    window: str,
) -> tuple[pd.DataFrame, str | None, str | None]:
    """``data.online`` with the readings outside their ``valid`` range dropped (``skill_stats.clean``)."""
    df, ds, err = data.online(start, end, groups, window)
    return st.clean(df, quantities), ds, err


def _col(df: pd.DataFrame, key: str) -> pd.Series:
    return pd.to_numeric(df[key], errors="coerce") if key in df.columns else pd.Series(dtype=float)


def _window(df: pd.DataFrame | pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> Any:
    if df is None or len(df) == 0:
        return df
    return df[(df.index >= start) & (df.index < end)]


def _when(ts: pd.Timestamp) -> str:
    return f"{ts.day} {ts.strftime('%b')} {ts.strftime('%H:%M')}"


def _span(start: pd.Timestamp, end: pd.Timestamp) -> str:
    return f"{_when(start)} – {_when(end)} IST"


def limit_lines(values: Mapping[str, float | None]) -> list[str]:
    """Plant-limit breaches of the shift-report KPIs (fuel rate, ETA CO, RAFT, permeability, HM Si, HMT)."""
    lim = plant_limits()
    out: list[str] = []
    f = values.get("fuel_rate")
    fr = lim.get("fuel_rate") or {}
    if f is not None and fr:
        if f >= fr.get("crit", math.inf):
            out.append(f"Fuel rate {st.fmt(f, 1, 'kg/tHM')} is above the critical level {fr['crit']:g}.")
        elif f >= fr.get("warn", math.inf):
            out.append(f"Fuel rate {st.fmt(f, 1, 'kg/tHM')} is above the warning level {fr['warn']:g}.")
    e = values.get("body_etaco")
    er = lim.get("etaco") or {}
    if e is not None and er:
        if e <= er.get("crit", -math.inf):
            out.append(f"ETA CO {st.fmt(e, 2, '%')} is below the critical level {er['crit']:g} %.")
        elif e <= er.get("warn", -math.inf):
            out.append(f"ETA CO {st.fmt(e, 2, '%')} is below the warning level {er['warn']:g} %.")
    for key, lo, hi, label, unit, digits in (
        ("body_raft", "raft_lo", "raft_hi", "RAFT", "°C", 0),
        ("body_perm", "perm_lo", "perm_hi", "Permeability index", "", 0),
        ("hmt", "hm_temp_lo", "hm_temp_hi", "Hot-metal temperature", "°C", 0),
    ):
        v = values.get(key)
        if v is None or lo not in lim or hi not in lim:
            continue
        if v < lim[lo]:
            out.append(f"{label} {st.fmt(v, digits, unit)} is below the plant range {lim[lo]:g}–{lim[hi]:g}.")
        elif v > lim[hi]:
            out.append(f"{label} {st.fmt(v, digits, unit)} is above the plant range {lim[lo]:g}–{lim[hi]:g}.")
    si = values.get("chem_pct_si")
    sr = lim.get("hm_si") or {}
    if si is not None and sr:
        if si < sr.get("crit_lo", -math.inf) or si > sr.get("crit_hi", math.inf):
            out.append(
                f"Hot-metal Si {st.fmt(si, 2, '%')} is outside the critical range "
                f"{sr['crit_lo']:g}–{sr['crit_hi']:g} %."
            )
        elif si < sr.get("lo", -math.inf) or si > sr.get("hi", math.inf):
            out.append(f"Hot-metal Si {st.fmt(si, 2, '%')} is outside the normal range {sr['lo']:g}–{sr['hi']:g} %.")
    return out


def _fact_lines(ids: Sequence[str]) -> list[str]:
    """Plant facts by id, one line each (only the facts FurnaceMind may show)."""
    wanted = list(ids)
    facts = {f["id"]: f for f in plant_facts.plant_facts() if f["id"] in wanted}
    lines = []
    for fid in wanted:
        f = facts.get(fid)
        if not f:
            continue
        value = "" if f.get("value") is None else f" = {f['value']:g} {f.get('unit', '')}".rstrip()
        lines.append(f"- {fid}: {f['statement']}{value} (status {f.get('status', '—')}, grade {f.get('grade', '—')})")
    return lines


def _facts_by_id() -> dict[str, dict[str, Any]]:
    return {f["id"]: f for f in plant_facts.plant_facts()}


def _prompt(
    *,
    title: str,
    scope: str,
    chart_id: str | None,
    chart_title: str,
    form: str,
    sections: Sequence[tuple[str, str]],
    notes: Notes,
) -> str:
    lines = [
        f"SKILL: {title} — {scope}",
        "The numbers below are computed from the plant's own records. Use them as given and do not recompute them; "
        "no other tool call is needed.",
    ]
    if chart_id:
        lines += [
            f'STEP 1 — Before writing anything, call show_skill_chart(chart_id="{chart_id}"): {chart_title}. The '
            "operator sees the chart only through this call.",
            "STEP 2 — Write the note in this form (headings in bold, at most 14 lines):",
        ]
    else:
        lines.append("Write the note in this form (headings in bold, at most 14 lines):")
    lines.append(form.strip())
    for heading, body in sections:
        if body and body.strip():
            lines += ["", f"{heading}:", body.strip()]
    if notes.missing:
        lines += ["", "DATA NOT AVAILABLE (say so in the note; never fill these in):"]
        lines += [f"- {m}" for m in notes.missing]
    if notes.datasets:
        lines += ["", "Datasets in this conversation for follow-up questions: " + ", ".join(notes.datasets) + "."]
    return "\n".join(lines)


def _bar_chart(readings: Sequence[st.Reading], title: str, unusual: float) -> go.Figure | None:
    """Horizontal bars: how many usual spreads each quantity sits from its usual value."""
    rows = [r for r in readings if r.spreads is not None]
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: abs(r.spreads or 0))
    colours = [
        WORSE
        if r.direction() == "worse" and abs(r.spreads or 0) >= unusual
        else BETTER
        if r.direction() == "better" and abs(r.spreads or 0) >= unusual
        else NEUTRAL
        for r in rows
    ]
    fig = go.Figure(
        go.Bar(
            x=[r.spreads for r in rows],
            y=[r.label for r in rows],
            orientation="h",
            marker_color=colours,
            text=[f"{st.fmt(r.now, r.digits, r.unit)} (usual {st.fmt(r.usual, r.digits, r.unit)})" for r in rows],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:.1f}× usual spread<br>%{text}<extra></extra>",
        )
    )
    for x in (-unusual, unusual):
        fig.add_vline(x=x, line_dash="dot", line_color=BAND)
    fig.update_layout(
        title=title,
        xaxis_title="Distance from usual (× usual spread)",
        height=max(360, 30 * len(rows) + 140),
        margin={"l": 10, "r": 160, "t": 60, "b": 50},
        showlegend=False,
    )
    return fig


# ------------------------------------------------------------------------------------------------------ shift pulse
def shift_pulse(data: PlantData, cfg: Mapping[str, Any], shift_date: str, label: str) -> SkillOutput:
    """The last completed shift against the shifts of the week before (shift means)."""
    from utils.shift_windows import shift_window_naive

    sc, q = cfg["skills"]["shift_pulse"], cfg["quantities"]
    unusual, far = _spread_cfg(cfg)
    start, end = (pd.Timestamp(t) for t in shift_window_naive(date.fromisoformat(str(shift_date)), label))
    # The page's shift is the one running (``last_completed_shift`` returns the current shift): its hours so far.
    end = min(end, data.now())
    running = end < start + timedelta(hours=8)
    base_start = start - timedelta(days=int(sc["baseline_days"]))
    notes = Notes()
    online, ds, err = _online(data, q, base_start, end, ["process_params"], "1 hour")
    if err:
        notes.missing.append(f"Process tags: {err}")
    if ds:
        notes.datasets.append(f"{ds} (process tags, hourly, {_span(base_start, end)})")
    shift_rows = _window(online, start, end) if not online.empty else online
    base_shifts = (
        st.shift_means(_window(online, base_start, start), sc["online"]) if not online.empty else pd.DataFrame()
    )
    readings = [
        st.compare(
            k, _col(shift_rows, k) if not online.empty else None, base_shifts[k] if k in base_shifts.columns else [], q
        )
        for k in sc["online"]
    ]
    hm, hm_err = data.hot_metal(base_start, end)
    if hm_err:
        notes.missing.append(f"Hot-metal samples: {hm_err}")
    taps = data.hmt_taps(hm)
    for key in sc["hot_metal"]:
        series = taps if key == "hmt" else _col(hm, key) if not hm.empty else pd.Series(dtype=float)
        now = _window(series, start, end)
        base = _window(series, base_start, start)
        base_shift = base.groupby(st.shift_start(base.index)).mean() if len(base) else []
        readings.append(st.compare(key, now if len(now) else None, base_shift, q))
    known = [r for r in readings if r.now is not None]
    movers = [r for r in st.ranked(known) if r.spreads is not None and abs(r.spreads) >= unusual][: int(sc["movers"])]
    limits = limit_lines({r.key: r.now for r in known})
    n_base = max((r.n for r in readings), default=0)
    name = st.shift_name(start)
    hours = (end - start).total_seconds() / 3600.0
    status = f"running, first {hours:.1f} h of 8" if running else "completed"
    chart = _bar_chart(known, f"Shift {name} against the {n_base} shifts before it", unusual)
    form = f"""
**Shift {name}** ({status}) — one line: how the shift ran against the week before (fuel rate, ETA CO, production).
**What moved** — up to 3 bullets from MOVERS: quantity, now against usual with units, × usual spread, better or worse.
**Plant limits** — each line under PLANT LIMITS, else "within the plant limits".
**Next shift** — up to 2 actions as ACTION / REASON / MAGNITUDE, only where a controllable moved (blast volume,
O₂, PCI, steam, hot blast temperature, charging); otherwise name what to watch.
"""
    sections = [
        (
            "SHIFT",
            f"{name}, {status} ({_span(start, end)}); usual = median of the {n_base} shift means of the "
            f"{int(sc['baseline_days'])} days before",
        ),
        ("READINGS (shift mean against the usual shift)", st.readings_table(readings, unusual, far)),
        (
            "MOVERS (2× usual spread or more, largest first)",
            "\n".join(
                f"- {r.label}: {st.fmt(r.now, r.digits, r.unit)} against {st.fmt(r.usual, r.digits, r.unit)} "
                f"({st.fmt_spreads(r.spreads)}; {r.state(unusual, far)})"
                for r in movers
            )
            or "- none",
        ),
        ("PLANT LIMITS", "\n".join(f"- {line}" for line in limits) or "- within the plant limits"),
    ]
    return SkillOutput(
        _prompt(
            title="Shift Pulse",
            scope=f"{name} against the week before",
            chart_id="shift_pulse" if chart else None,
            chart_title="how far each quantity sat from its usual shift",
            form=form,
            sections=sections,
            notes=notes,
        ),
        chart,
        "shift_pulse",
    )


# ---------------------------------------------------------------------------------------------------- thermal watch
def thermal_watch(data: PlantData, cfg: Mapping[str, Any]) -> SkillOutput:
    """Hot-metal temperature per tap and Si over the last 24 h against the plant limits, and what moved before."""
    sc, q = cfg["skills"]["thermal_watch"], cfg["quantities"]
    unusual, far = _spread_cfg(cfg)
    lim = plant_limits()
    now = data.now()
    w_start = now - timedelta(hours=float(sc["window_hours"]))
    h_start = now - timedelta(hours=float(sc["history_hours"]))
    notes = Notes()
    hm, err = data.hot_metal(h_start, now)
    if err:
        notes.missing.append(f"Hot-metal samples: {err}")
    taps = data.hmt_taps(hm)
    si = _col(hm, "chem_pct_si") if not hm.empty else pd.Series(dtype=float)
    si = si[(si > 0) & np.isfinite(si)] if len(si) else si
    taps_w, si_w = _window(taps, w_start, now), _window(si, w_start, now)
    lo, hi = float(lim.get("hm_temp_lo", 1440)), float(lim.get("hm_temp_hi", 1520))
    si_lim = lim.get("hm_si") or {"lo": 0.25, "hi": 0.60}
    slope = st.slope_per_hour(taps_w)
    last = float(taps_w.iloc[-1]) if len(taps_w) else None
    if last is None:
        verdict = "no tap in the window"
    elif slope is not None and slope <= float(sc["cold_drift_c_per_h"]) and last <= lo + float(sc["near_low_c"]):
        verdict = "cold drift: falling and close to the lower limit"
    elif slope is not None and slope >= -float(sc["cold_drift_c_per_h"]) and last >= hi - float(sc["near_low_c"]):
        verdict = "hot drift: rising and close to the upper limit"
    elif len(taps_w) and ((taps_w < lo).any() or (taps_w > hi).any()):
        verdict = "outside the plant range at times"
    else:
        verdict = "within the plant range"
    tap_rows = [
        [_when(t), st.fmt(v, 0, "°C"), "below" if v < lo else "above" if v > hi else "in range"]
        for t, v in taps_w.tail(8).items()
    ]
    si_out = int(((si_w < si_lim["lo"]) | (si_w > si_lim["hi"])).sum()) if len(si_w) else 0
    hm_lines = [
        f"- Taps in the window: {len(taps_w)}; mean {st.fmt(taps_w.mean() if len(taps_w) else None, 0, '°C')}, "
        f"lowest {st.fmt(taps_w.min() if len(taps_w) else None, 0, '°C')}, "
        f"highest {st.fmt(taps_w.max() if len(taps_w) else None, 0, '°C')}; plant range {lo:g}–{hi:g} °C",
        f"- Trend over the window: {st.fmt_signed(slope, 1, '°C/h')}; state: {verdict}",
        f"- Si samples: {len(si_w)}; mean {st.fmt(si_w.mean() if len(si_w) else None, 2, '%')}, "
        f"last {st.fmt(float(si_w.iloc[-1]) if len(si_w) else None, 2, '%')}, "
        f"trend {st.fmt_signed(st.slope_per_hour(si_w), 3, '%/h')}; normal range {si_lim['lo']:g}–{si_lim['hi']:g} %",
        f"- Si outside the normal range: {si_out} of {len(si_w)} samples",
    ]
    d_start = now - timedelta(hours=float(sc["drivers_hours"]))
    b_start = d_start - timedelta(days=float(sc["drivers_baseline_days"]))
    online, ds, oerr = _online(data, q, b_start, now, ["process_params"], "1 hour")
    if oerr:
        notes.missing.append(f"Process tags: {oerr}")
    if ds:
        notes.datasets.append(f"{ds} (process tags, hourly, {_span(b_start, now)})")
    drivers = [
        st.compare(
            k,
            _col(_window(online, d_start, now), k) if not online.empty else None,
            _col(_window(online, b_start, d_start), k) if not online.empty else [],
            q,
        )
        for k in sc["drivers"]
    ]
    chart = None
    if len(taps) or len(si):
        chart = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=("Hot-metal temperature per tap (°C)", "Hot-metal Si (%)"),
        )
        chart.add_trace(
            go.Scatter(
                x=taps.index, y=taps.values, mode="lines+markers", name="HMT per tap", line={"color": SECTOR_COLOURS[1]}
            ),
            row=1,
            col=1,
        )
        for y in (lo, hi):
            chart.add_hline(y=y, line_dash="dash", line_color=BAND, row=1, col=1)
        chart.add_trace(
            go.Scatter(x=si.index, y=si.values, mode="markers", name="Si", marker={"color": SECTOR_COLOURS[0]}),
            row=2,
            col=1,
        )
        for y in (si_lim["lo"], si_lim["hi"]):
            chart.add_hline(y=y, line_dash="dash", line_color=BAND, row=2, col=1)
        chart.add_vrect(x0=w_start, x1=now, fillcolor="#FEF3C7", opacity=0.35, line_width=0, row="all", col=1)
        chart.update_layout(
            title=f"Hot metal, last {int(sc['history_hours'])} h (shaded: last {int(sc['window_hours'])} h)",
            height=560,
            showlegend=False,
        )
    form = """
**Thermal state** — one line: the state, the last tap with its time, the trend in °C/h.
**Silicon** — one line: level and direction against the normal range.
**What moved before** — up to 3 quantities from DRIVERS that moved 2× usual spread or more, with direction.
**Action** — 1-2 actions as ACTION / REASON / MAGNITUDE; size them with PLANT FACTS where one applies; when the state
is within range and nothing moved, say "hold and watch the next two taps".
"""
    sections = [
        (
            "WINDOW",
            f"{_span(w_start, now)}; drivers {_span(d_start, now)} against the {int(sc['drivers_baseline_days'])} "
            "days before",
        ),
        ("HOT METAL", "\n".join(hm_lines)),
        ("LAST TAPS", st.md_table(("Tap time (IST)", "HMT", "Against the plant range"), tap_rows) if tap_rows else ""),
        ("DRIVERS (last hours against the days before)", st.readings_table(drivers, unusual, far)),
        ("PLANT FACTS (plant fact register)", "\n".join(_fact_lines(sc["facts"]))),
    ]
    return SkillOutput(
        _prompt(
            title="Thermal Watch",
            scope="hot metal over the last 24 h",
            chart_id="thermal_watch" if chart else None,
            chart_title="HMT per tap and Si with the plant ranges",
            form=form,
            sections=sections,
            notes=notes,
        ),
        chart,
        "thermal_watch",
    )


# ---------------------------------------------------------------------------------------------------- gas flow lens
def _spread_series(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    present = [c for c in cols if c in df.columns]
    if len(present) < 2 or df.empty:
        return pd.Series(dtype=float)
    block = df[present].apply(pd.to_numeric, errors="coerce")
    return (block.max(axis=1) - block.min(axis=1)).dropna()


def _hottest(df: pd.DataFrame, cols: Sequence[str]) -> tuple[str | None, float]:
    """The sector most often hottest in ``df`` and its share of the readings."""
    present = [c for c in cols if c in df.columns]
    if not present or df.empty:
        return None, 0.0
    block = df[present].apply(pd.to_numeric, errors="coerce").dropna(how="all")
    if block.empty:
        return None, 0.0
    top = block.idxmax(axis=1).value_counts(normalize=True)
    return str(top.index[0]), float(top.iloc[0])


def gas_flow_lens(data: PlantData, cfg: Mapping[str, Any]) -> SkillOutput:
    """Wall temperatures A-D, uptakes, top gas and pressures over the last 8 h against the week before."""
    sc, q = cfg["skills"]["gas_flow_lens"], dict(cfg["quantities"])
    unusual, far = _spread_cfg(cfg)
    now = data.now()
    start = now - timedelta(hours=float(sc["window_hours"]))
    base_start = start - timedelta(days=float(sc["baseline_days"]))
    groups = ["process_params", "temperature_profile", "miscellaneous"]
    notes = Notes()
    recent, ds1, e1 = _online(data, q, start, now, groups, "15 minutes")
    hist, ds2, e2 = _online(data, q, base_start, start, groups, "1 hour")
    for err, what in ((e1, "last 8 h"), (e2, "week before")):
        if err:
            notes.missing.append(f"Process and wall temperatures, {what}: {err}")
    for ds, what in ((ds1, f"15-min, {_span(start, now)}"), (ds2, f"hourly, {_span(base_start, start)}")):
        if ds:
            notes.datasets.append(f"{ds} (process, wall temperatures and stock rods, {what})")
    recent_h = recent.resample("1h").mean() if not recent.empty else recent
    level_rows, spreads = [], []
    for level, zone in sc["levels"].items():
        cols = [f"temp_{level}_{s}" for s in "abcd"]
        key = f"spread_{level}"
        q[key] = {
            "label": f"{zone} ({int(level) / 1000:.3f} m) A–D spread",
            "unit": "°C",
            "digits": 0,
            "better": "lower",
        }
        r = st.compare(key, _spread_series(recent, cols), _spread_series(hist, cols), q)
        spreads.append(r)
        hot_now, share_now = _hottest(recent, cols)
        hot_base, share_base = _hottest(hist, cols)
        means = [st.fmt(_col(recent, c).mean() if c in recent.columns else None, 0, "°C") for c in cols]
        level_rows.append(
            [
                f"{zone} {int(level) / 1000:.3f} m",
                *means,
                f"{hot_now[-1].upper() if hot_now else '—'} ({share_now:.0%} of readings)",
                f"{hot_base[-1].upper() if hot_base else '—'} ({share_base:.0%})",
            ]
        )
    uptake_key = "spread_uptake"
    q[uptake_key] = {"label": "Uptake T1–T4 spread", "unit": "°C", "digits": 0, "better": "lower"}
    spreads.append(st.compare(uptake_key, _spread_series(recent, sc["uptake"]), _spread_series(hist, sc["uptake"]), q))
    hot_up, share_up = _hottest(recent, sc["uptake"])
    level_rows.append(
        [
            "Uptakes T1–T4",
            *[st.fmt(_col(recent, c).mean() if c in recent.columns else None, 0, "°C") for c in sc["uptake"]],
            f"{('T' + hot_up[-1]) if hot_up else '—'} ({share_up:.0%} of readings)",
            "—",
        ]
    )
    levels = [st.compare(k, _col(recent, k), _col(hist, k), q) for k in [*sc["gas"], *sc["pressure"]]]
    swings = []
    for k in sc["swings"]:
        sk = f"swing_{k}"
        meta = st.quantity(k, q)
        q[sk] = {**meta, "label": f"{meta['label']} swing (SD within 8 h)", "better": "lower"}
        now_sd = (
            float(_col(recent_h, k).std(ddof=1)) if k in recent_h.columns and _col(recent_h, k).count() > 2 else None
        )
        base_sd = (
            _col(hist, k).groupby(st.shift_start(hist.index)).std(ddof=1)
            if k in hist.columns
            else pd.Series(dtype=float)
        )
        swings.append(st.compare(sk, now_sd, base_sd, q))
    rods = []
    if {"stock_rod1_pos", "stock_rod2_pos"} <= set(recent.columns) | set(hist.columns):
        q["rod_gap"] = {"label": "Stock rod 1 − rod 2", "unit": "", "digits": 2, "better": "none"}
        gap_now = _col(recent, "stock_rod1_pos") - _col(recent, "stock_rod2_pos")
        gap_base = _col(hist, "stock_rod1_pos") - _col(hist, "stock_rod2_pos")
        rods.append(st.compare("rod_gap", gap_now, gap_base, q))
    if "stock_rod_radar_level" in recent.columns:
        rods.append(
            st.compare(
                "stock_rod_radar_level",
                _col(recent, "stock_rod_radar_level"),
                _col(hist, "stock_rod_radar_level"),
                {**q, "stock_rod_radar_level": {"label": "Stock level (radar)", "digits": 2}},
            )
        )
    chart = None
    cols_top = [f"temp_18660_{s}" for s in "abcd" if f"temp_18660_{s}" in recent.columns]
    if cols_top or any(c in recent.columns for c in sc["uptake"]):
        chart = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.12,
            subplot_titles=("Upper stack wall 18.660 m, sectors A–D (°C)", "Uptakes T1–T4 (°C)"),
        )
        for i, c in enumerate(cols_top):
            chart.add_trace(
                go.Scatter(
                    x=recent.index,
                    y=_col(recent, c),
                    mode="lines",
                    name=c[-1].upper(),
                    line={"color": SECTOR_COLOURS[i % 4]},
                ),
                row=1,
                col=1,
            )
        for i, c in enumerate(c for c in sc["uptake"] if c in recent.columns):
            chart.add_trace(
                go.Scatter(
                    x=recent.index,
                    y=_col(recent, c),
                    mode="lines",
                    name=f"T{c[-1]}",
                    line={"color": SECTOR_COLOURS[i % 4], "dash": "dot"},
                ),
                row=2,
                col=1,
            )
        chart.update_layout(title=f"Gas distribution, {_span(start, now)}", height=560)
    form = """
**Gas flow** — one line: even, one-sided (name the level and sector) or unsettled (pressure swings).
**Wall and uptakes** — the level whose A–D spread moved most, the hottest sector and whether it is the usual one.
**Pressure and permeability** — ΔP, permeability and the swings against usual, one line.
**Action** — up to 2 as ACTION / REASON / MAGNITUDE (for example review the burden distribution toward the hot
sector, check tuyeres on that side, steady the blast); when everything is usual, say what to watch.
"""
    sections = [
        ("WINDOW", f"{_span(start, now)} (15-min readings) against {_span(base_start, start)} (hourly)"),
        (
            "SECTOR TEMPERATURES (mean of the window; hottest sector now and in the week before)",
            st.md_table(("Level", "A / T1", "B / T2", "C / T3", "D / T4", "Hottest now", "Hottest before"), level_rows),
        ),
        ("SPREADS ACROSS THE SECTORS", st.readings_table(spreads, unusual, far)),
        ("TOP GAS AND PRESSURES", st.readings_table(levels, unusual, far)),
        ("SWINGS", st.readings_table(swings, unusual, far)),
        ("BURDEN DESCENT", st.readings_table(rods, unusual, far) if rods else ""),
    ]
    return SkillOutput(
        _prompt(
            title="Gas Flow Lens",
            scope="the last 8 h against the week before",
            chart_id="gas_flow_lens" if chart else None,
            chart_title="upper-stack sector temperatures and uptakes over the window",
            form=form,
            sections=sections,
            notes=notes,
        ),
        chart,
        "gas_flow_lens",
    )


# ---------------------------------------------------------------------------------------------------- wall heat map
_STAVE = re.compile(r"^delta_t_r(\d+)_stave_(\d+(?:_\d+)?)$")


def wall_heat_map(data: PlantData, cfg: Mapping[str, Any]) -> SkillOutput:
    """Stave heat load by row and quadrant, the staves whose ΔT rose most and the cooling water, 8 h against 14 d."""
    sc = cfg["skills"]["wall_heat_map"]
    unusual, far = _spread_cfg(cfg)
    now = data.now()
    start = now - timedelta(hours=float(sc["window_hours"]))
    base_start = start - timedelta(days=float(sc["baseline_days"]))
    groups = ["heatload_delta_t", "delta_t", "cooling_water"]
    notes = Notes()
    recent, ds1, e1 = _online(data, cfg["quantities"], start, now, groups, "1 hour")
    hist, ds2, e2 = _online(data, cfg["quantities"], base_start, start, groups, "1 hour")
    for err, what in ((e1, "last 8 h"), (e2, "14 days before")):
        if err:
            notes.missing.append(f"Heat load, stave ΔT and cooling water, {what}: {err}")
    for ds, span in ((ds1, _span(start, now)), (ds2, _span(base_start, start))):
        if ds:
            notes.datasets.append(f"{ds} (heat load, stave ΔT, cooling water, hourly, {span})")
    q: dict[str, Any] = {}
    matrix = np.full((5, 4), np.nan)
    cells = []
    for i, r in enumerate(range(6, 11)):
        for j, k in enumerate(range(1, 5)):
            key = f"heat_load_r{r}_q{k}"
            q[key] = {"label": f"R{r} Q{k}", "digits": 2, "better": "lower"}
            reading = st.compare(key, _col(recent, key), _col(hist, key), q)
            if reading.now is not None and reading.usual:
                matrix[i, j] = 100.0 * (reading.now - reading.usual) / abs(reading.usual)
            cells.append(reading)
    rows = []
    for key, label in [
        *((f"heat_load_row_{r}", f"Row R{r}") for r in range(6, 11)),
        *((f"heat_load_row6_10_q{k}", f"Quadrant Q{k} (R6–R10)") for k in range(1, 5)),
    ]:
        q[key] = {"label": label, "digits": 2, "better": "lower"}
        rows.append(st.compare(key, _col(recent, key), _col(hist, key), q))
    rise = float(sc["rise_pct"])
    flagged = [
        c
        for c in cells
        if c.now is not None
        and c.usual
        and 100.0 * (c.now - c.usual) / abs(c.usual) >= rise
        and (c.spreads or 0) >= unusual
    ]
    staves = []
    for col in recent.columns:
        m = _STAVE.match(str(col))
        if not m:
            continue
        q[col] = {
            "label": f"R{m.group(1)} stave {m.group(2).replace('_', '/')}",
            "unit": "°C",
            "digits": 2,
            "better": "lower",
        }
        staves.append(st.compare(col, _col(recent, col), _col(hist, col), q))
    top = [s for s in sorted(staves, key=lambda s: -(s.spreads or -math.inf)) if (s.spreads or 0) > 0][
        : int(sc["top_staves"])
    ]
    water = []
    for col in [c for c in recent.columns if str(c).startswith("cw_")]:
        unit = "m³/h" if col.endswith("_m3h") else "bar" if col.endswith("_bar") else "°C"
        label = (
            col.removeprefix("cw_")
            .replace("_flow_m3h", " flow")
            .replace("_press_bar", " pressure")
            .replace("_temp_c", " temperature")
            .replace("_", " ")
        )
        q[col] = {
            "label": label[:1].upper() + label[1:],
            "unit": unit,
            "digits": 1 if unit != "bar" else 2,
            "better": "none",
            "floor_pct": float(sc.get("water_floor_pct", 3.0)),
        }
        water.append(st.compare(col, _col(recent, col), _col(hist, col), q))
    drop = float(sc["flow_drop_pct"])
    water_flags = [
        w
        for w in water
        if w.now is not None
        and w.usual
        and w.unit in ("m³/h", "bar")
        and 100.0 * (w.now - w.usual) / abs(w.usual) <= -drop
    ]
    chart = None
    if np.isfinite(matrix).any():
        text = [[("—" if not np.isfinite(v) else f"{v:+.0f} %") for v in row] for row in matrix]
        chart = go.Figure(
            go.Heatmap(
                z=matrix,
                x=[f"Q{k}" for k in range(1, 5)],
                y=[f"R{r}" for r in range(6, 11)],
                text=text,
                texttemplate="%{text}",
                colorscale=[[0, "#1D4ED8"], [0.5, "#F8FAFC"], [1, "#B91C1C"]],
                zmid=0,
                colorbar={"title": "% against usual"},
                hovertemplate="%{y} %{x}: %{text}<extra></extra>",
            )
        )
        chart.update_layout(
            title=f"Stave heat load, last {int(sc['window_hours'])} h against the "
            f"{int(sc['baseline_days'])} days before (% change)",
            yaxis={"autorange": "reversed"},
            height=420,
        )
    form = """
**Wall** — one line: where the heat load stands against usual (rows and quadrants that rose or fell).
**Hot spots** — the cells under RISES and up to 3 staves under STAVES, with their change.
**Cooling water** — circuits under COOLING WATER DROPS, else "flows and pressures usual".
**Action** — up to 2 as ACTION / REASON / MAGNITUDE (for example check the circuit or the stave, review peripheral
gas flow on that quadrant); when nothing rose, say what to watch.
"""
    sections = [
        ("WINDOW", f"{_span(start, now)} against {_span(base_start, start)}, hourly; heat load as logged by the plant"),
        ("ROWS AND QUADRANTS", st.readings_table(rows, unusual, far)),
        (
            f"RISES (cells at least {rise:g} % and 2× usual spread above usual)",
            "\n".join(
                f"- {c.label}: {st.fmt(c.now, 2)} against {st.fmt(c.usual, 2)} "
                f"({st.fmt_signed(100 * (c.now - c.usual) / abs(c.usual), 0, '%')}, {st.fmt_spreads(c.spreads)})"
                for c in flagged
            )
            or "- none",
        ),
        ("STAVES (largest rise of ΔT against usual)", st.readings_table(top, unusual, far) if top else "- none rose"),
        (
            f"COOLING WATER DROPS (flow or pressure {drop:g} % or more below usual)",
            "\n".join(
                f"- {w.label}: {st.fmt(w.now, w.digits, w.unit)} against {st.fmt(w.usual, w.digits, w.unit)}"
                for w in water_flags
            )
            or "- none",
        ),
        ("COOLING WATER", st.readings_table(water, unusual, far) if water else ""),
    ]
    return SkillOutput(
        _prompt(
            title="Wall Heat Map",
            scope="the last 8 h against 14 days",
            chart_id="wall_heat_map" if chart else None,
            chart_title="heat load by row and quadrant, % against usual",
            form=form,
            sections=sections,
            notes=notes,
        ),
        chart,
        "wall_heat_map",
    )


# ----------------------------------------------------------------------------------------------- slag & burden check
@lru_cache(maxsize=1)
def charge_groups() -> dict[str, list[str]]:
    """Charge-record columns per material group (``shift_report.yml -> charge_columns``)."""
    try:
        with _SHIFT_REPORT.open(encoding="utf-8") as fh:
            return {k: list(v) for k, v in ((yaml.safe_load(fh) or {}).get("charge_columns") or {}).items()}
    except OSError:
        return {}


def charge_masses(charge: pd.DataFrame) -> pd.DataFrame:
    """Per charge: t of sinter, ore, pellet, coke, nut coke, flux (summed over their columns) and the ferrous total."""
    out = pd.DataFrame(index=charge.index)
    for group, cols in charge_groups().items():
        present = [c for c in cols if c in charge.columns]
        out[group] = (
            charge[present].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1) if present else np.nan
        )
    ferrous = out[[c for c in ("sinter", "ore", "pellet") if c in out.columns]].sum(axis=1, min_count=1)
    out["ferrous"] = ferrous
    return out


def blend_shares(masses: pd.DataFrame) -> dict[str, float | None]:
    """Shares of sinter, pellet and lump ore in the ferrous charged (%), and coke per tonne of ferrous."""
    total = float(masses["ferrous"].sum()) if "ferrous" in masses else 0.0
    if total <= 0:
        return {"sinter_share": None, "pellet_share": None, "ore_share": None, "coke_per_ferrous": None}
    share = {f"{k}_share": 100.0 * float(masses[k].sum()) / total for k in ("sinter", "pellet", "ore") if k in masses}
    coke = float(masses["coke"].sum()) if "coke" in masses else None
    return {**share, "coke_per_ferrous": (coke / total) if coke is not None else None}


def daily_shares(masses: pd.DataFrame) -> pd.DataFrame:
    if masses.empty:
        return pd.DataFrame()
    rows = {day: blend_shares(group) for day, group in masses.groupby(masses.index.normalize())}
    return pd.DataFrame.from_dict(rows, orient="index")


def slag_burden_check(data: PlantData, cfg: Mapping[str, Any]) -> SkillOutput:
    """Slag chemistry, the charged blend and the sinter, last 24 h against the week before."""
    sc, q = cfg["skills"]["slag_burden_check"], dict(cfg["quantities"])
    unusual, far = _spread_cfg(cfg)
    now = data.now()
    start = now - timedelta(hours=float(sc["window_hours"]))
    base_start = start - timedelta(days=float(sc["baseline_days"]))
    notes = Notes()
    hm, err = data.hot_metal(base_start, now)
    if err:
        notes.missing.append(f"Slag and hot-metal samples: {err}")
    if not hm.empty and {"slag_pct_mgo", "slag_pct_al2o3"} <= set(hm.columns):
        hm = hm.copy()
        hm["mgo_al2o3"] = _col(hm, "slag_pct_mgo") / _col(hm, "slag_pct_al2o3").replace(0, np.nan)
    recent, base = _window(hm, start, now), _window(hm, base_start, start)
    slag = [
        st.compare(k, _col(recent, k) if not hm.empty else None, _col(base, k) if not hm.empty else [], q)
        for k in sc["slag"]
    ]
    bands = slag_bands()
    band_lines = []
    for key, (lo, hi) in bands.items():
        s = _col(recent, key) if not hm.empty else pd.Series(dtype=float)
        s = s.dropna()
        if len(s):
            out = int(((s < lo) | (s > hi)).sum())
            band_lines.append(
                f"- {st.quantity(key, q)['label']}: plant band {lo:g}–{hi:g}; last {s.iloc[-1]:.3f}, "
                f"{out} of {len(s)} samples of the last {int(sc['window_hours'])} h outside"
            )
    for key, rule in (sc.get("practice") or {}).items():
        s = _col(recent, key).dropna() if not hm.empty else pd.Series(dtype=float)
        if len(s):
            meta = st.quantity(key, q)
            band_lines.append(
                f"- {meta['label']}: mean of the last {int(sc['window_hours'])} h "
                f"{st.fmt(float(s.mean()), meta['digits'], meta['unit'])} against {rule['note']}"
            )
    charge, cerr = data.table("offline_feed.charge_data", base_start, now)
    if cerr:
        notes.missing.append(f"Charge record: {cerr}")
    masses = charge_masses(charge) if not charge.empty else pd.DataFrame()
    blend_rows = []
    if not masses.empty:
        now_shares = blend_shares(_window(masses, start, now))
        days = daily_shares(_window(masses, base_start, start))
        q["coke_per_ferrous"] = {"label": "Coke per t of ferrous", "unit": "t/t", "digits": 3, "better": "none"}
        blend_rows = [
            st.compare(k, now_shares.get(k), days[k] if k in days.columns else [], q)
            for k in ("sinter_share", "pellet_share", "ore_share", "coke_per_ferrous")
        ]
        # Charges of the window against each full window-length block before it (calendar days would cut the first
        # and the last day short).
        width = now - start
        before = _window(masses, base_start, start)
        blocks = ((before.index - base_start) // width).astype(int) if len(before) else []
        counts = before.groupby(blocks).size() if len(before) else pd.Series(dtype=float)
        counts = counts[counts.index < int((start - base_start) // width)]
        blend_rows.append(
            st.compare(
                "charges_window",
                float(len(_window(masses, start, now))),
                counts.astype(float),
                {
                    "charges_window": {
                        "label": f"Charges in {int(sc['window_hours'])} h",
                        "digits": 0,
                        "better": "higher",
                    }
                },
            )
        )
    sinter, serr = data.table("offline_feed.sinter_chemistry", base_start, now)
    if serr:
        notes.missing.append(f"Sinter laboratory: {serr}")
    sinter_rows = []
    if not sinter.empty:
        names = {
            "fe_t": "Fe(T)",
            "feo": "FeO",
            "sio2": "SiO₂",
            "al2o3": "Al₂O₃",
            "cao": "CaO",
            "mgo": "MgO",
            "basicity": "Basicity",
        }
        for k in sc["sinter"]:
            if k not in sinter.columns:
                continue
            s = _col(sinter, k).dropna()
            if s.empty:
                continue
            key = f"sinter_{k}"
            q[key] = {
                "label": f"Sinter {names.get(k, k)} (latest sample)",
                "unit": "" if k == "basicity" else "%",
                "digits": 3 if k == "basicity" else 2,
                "better": "none",
            }
            sinter_rows.append(st.compare(key, float(s.iloc[-1]), s.iloc[:-1], q))
    chart = None
    if not hm.empty and ("slag_basicity" in hm.columns or "slag_pct_al2o3" in hm.columns):
        chart = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.12,
            subplot_titles=("Slag B2 (CaO/SiO₂)", "Slag Al₂O₃ (%)"),
        )
        if "slag_basicity" in hm.columns:
            chart.add_trace(
                go.Scatter(
                    x=hm.index,
                    y=_col(hm, "slag_basicity"),
                    mode="markers",
                    name="B2",
                    marker={"color": SECTOR_COLOURS[0]},
                ),
                row=1,
                col=1,
            )
            for y in bands["slag_basicity"]:
                chart.add_hline(y=y, line_dash="dash", line_color=BAND, row=1, col=1)
        if "slag_pct_al2o3" in hm.columns:
            chart.add_trace(
                go.Scatter(
                    x=hm.index,
                    y=_col(hm, "slag_pct_al2o3"),
                    mode="markers",
                    name="Al₂O₃",
                    marker={"color": SECTOR_COLOURS[1]},
                ),
                row=2,
                col=1,
            )
            practice = (sc.get("practice") or {}).get("slag_pct_al2o3", {})
            if practice.get("high"):
                chart.add_hline(y=practice["high"], line_dash="dot", line_color=BAND, row=2, col=1)
        chart.add_vrect(x0=start, x1=now, fillcolor="#FEF3C7", opacity=0.35, line_width=0, row="all", col=1)
        chart.update_layout(
            title=f"Slag, {_span(base_start, now)} (shaded: last {int(sc['window_hours'])} h)",
            height=540,
            showlegend=False,
        )
    form = """
**Slag** — one line: basicity and Al₂O₃ against the plant bands and practice, and the direction.
**Burden** — one line: the blend and coke per tonne of ferrous against the week, and any sinter change.
**Link** — one line: whether the slag change follows the burden or sinter change (same direction, same time), or not.
**Action** — up to 2 as ACTION / REASON / MAGNITUDE (flux, blend, or a sample to repeat); else what to watch.
"""
    sections = [
        ("WINDOW", f"{_span(start, now)} against {_span(base_start, start)}"),
        (
            "SLAG AND HOT METAL (samples of the last 24 h against the samples before)",
            st.readings_table(slag, unusual, far),
        ),
        ("PLANT BANDS AND PRACTICE", "\n".join(band_lines)),
        (
            "CHARGED BLEND (last 24 h against each day before)",
            st.readings_table(blend_rows, unusual, far) if blend_rows else "",
        ),
        (
            "SINTER (latest sample against the samples before)",
            st.readings_table(sinter_rows, unusual, far) if sinter_rows else "",
        ),
    ]
    return SkillOutput(
        _prompt(
            title="Slag & Burden Check",
            scope="the last 24 h against the week before",
            chart_id="slag_burden_check" if chart else None,
            chart_title="slag B2 and Al₂O₃ with the plant band",
            form=form,
            sections=sections,
            notes=notes,
        ),
        chart,
        "slag_burden_check",
    )


# ----------------------------------------------------------------------------------------------------- daily frames
def daily_frame(
    data: PlantData, days: int, notes: Notes, quantities: Mapping[str, Any], now: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Day means over the last ``days`` (IST days with at least 18 hourly rows): process tags, HMT per tap, Si, and
    the pellet share of the charged ferrous."""
    now = now or data.now()
    start = (now - timedelta(days=days)).normalize()
    online, ds, err = _online(data, quantities, start, now, ["process_params"], "1 hour")
    if err:
        notes.missing.append(f"Process tags: {err}")
    if ds:
        notes.datasets.append(f"{ds} (process tags, hourly, {_span(start, now)})")
    if online.empty:
        return pd.DataFrame()
    counts = online.groupby(online.index.normalize()).size()
    daily = online.apply(pd.to_numeric, errors="coerce").groupby(online.index.normalize()).mean()
    daily = daily[counts.reindex(daily.index).fillna(0) >= 18]
    hm, herr = data.hot_metal(start, now)
    if herr:
        notes.missing.append(f"Hot-metal samples: {herr}")
    taps = data.hmt_taps(hm)
    if len(taps):
        daily["hmt"] = taps.groupby(taps.index.normalize()).mean().reindex(daily.index)
    if not hm.empty and "chem_pct_si" in hm.columns:
        si = _col(hm, "chem_pct_si")
        daily["chem_pct_si"] = si.groupby(si.index.normalize()).mean().reindex(daily.index)
    charge, cerr = data.table("offline_feed.charge_data", start, now)
    if cerr:
        notes.missing.append(f"Charge record: {cerr}")
    if not charge.empty:
        shares = daily_shares(charge_masses(charge))
        if "pellet_share" in shares.columns:
            daily["pellet_share"] = shares["pellet_share"].reindex(daily.index)
    return daily


# ---------------------------------------------------------------------------------------------------- fuel drivers
def fuel_drivers(data: PlantData, cfg: Mapping[str, Any]) -> SkillOutput:
    """What separated the highest-fuel days of the last 30 from the lowest, sized with the plant facts."""
    sc, q = cfg["skills"]["fuel_drivers"], cfg["quantities"]
    notes = Notes()
    now = data.now()
    daily = daily_frame(data, int(sc["days"]), notes, q, now)
    if daily.empty or "fuel_rate" not in daily.columns:
        notes.missing.append("Daily fuel rate: no day with enough hourly readings")
        return SkillOutput(
            _prompt(
                title="Fuel Drivers",
                scope="the last 30 days",
                chart_id=None,
                chart_title="",
                form="**Fuel Drivers** — say which data is missing and stop.",
                sections=[],
                notes=notes,
            )
        )
    prod = _col(daily, "production_per_hour")
    if prod.notna().any():
        daily = daily[prod >= float(sc["min_production_share"]) * float(prod.median())]
    fuel = _col(daily, "fuel_rate").dropna()
    n = min(int(sc["extremes"]), len(fuel) // 2)
    if n < 2:
        notes.missing.append(f"Daily fuel rate: only {len(fuel)} operating days with enough hourly readings")
        return SkillOutput(
            _prompt(
                title="Fuel Drivers",
                scope="the last 30 days",
                chart_id=None,
                chart_title="",
                form="**Fuel Drivers** — say which data is missing and stop.",
                sections=[],
                notes=notes,
            )
        )
    high_days = fuel.nlargest(n).index
    low_days = fuel.nsmallest(n).index
    high, low = daily.loc[high_days], daily.loc[low_days]
    facts = _facts_by_id()
    gap_rows, driver_rows, explained = [], [], []
    for key in ("fuel_rate", "coke_rate", "nut_coke_rate", "coal_rate_actual_value"):
        meta = st.quantity(key, q)
        hv, lv = _col(high, key).mean(), _col(low, key).mean()
        gap_rows.append(
            [
                meta["label"],
                st.fmt(hv, meta["digits"], meta["unit"]),
                st.fmt(lv, meta["digits"], meta["unit"]),
                st.fmt_signed(hv - lv, meta["digits"], meta["unit"]),
            ]
        )
    coke_gap = _col(high, "coke_rate").mean() - _col(low, "coke_rate").mean()
    for d in sc["drivers"]:
        key = d["field"]
        if key not in daily.columns:
            continue
        meta = st.quantity(key, q)
        hv, lv = _col(high, key).mean(), _col(low, key).mean()
        if not (math.isfinite(hv) and math.isfinite(lv)):
            continue
        _, spread = st.center_spread(_col(daily, key))
        diff = hv - lv
        effect_text, effect = "—", None
        fact = facts.get(d.get("fact", ""))
        if fact and fact.get("value") is not None and d.get("step"):
            step = float(d["step"])
            moved = (100.0 * diff / lv) if d.get("relative") and lv else diff
            effect = float(fact["value"]) * moved / step
            effect_text = f"{st.fmt_signed(effect, 1, 'kg/tHM')} coke ({fact['id']})"
            explained.append(effect)
        driver_rows.append(
            (
                abs(diff / spread) if spread else 0.0,
                [
                    meta["label"],
                    st.fmt(hv, meta["digits"], meta["unit"]),
                    st.fmt(lv, meta["digits"], meta["unit"]),
                    st.fmt_signed(diff, meta["digits"], meta["unit"]),
                    st.fmt_spreads(diff / spread if spread else None),
                    effect_text,
                ],
            )
        )
    driver_rows.sort(key=lambda r: -r[0])
    total = sum(explained) if explained else None
    chart = go.Figure(
        go.Bar(
            x=fuel.index,
            y=fuel.values,
            marker_color=[WORSE if d in high_days else BETTER if d in low_days else NEUTRAL for d in fuel.index],
            hovertemplate="%{x|%d %b}: %{y:.1f} kg/tHM<extra></extra>",
        )
    )
    chart.update_layout(
        title=f"Daily fuel rate, last {int(sc['days'])} days (highest and lowest {n} days marked)",
        yaxis_title="kg/tHM",
        height=420,
        showlegend=False,
    )
    form = """
**Fuel gap** — one line: the highest-fuel days against the lowest, split into coke, nut coke and PCI.
**What differed** — up to 4 drivers from DRIVERS with the largest × usual spread; give the coke effect the plant
fact states where there is one.
**Explained** — one line: how much of the coke gap the plant facts account for (EXPLAINED); the rest points to what
these records do not hold (coke and sinter quality, burden distribution, blast moisture) — name it as unmeasured.
**Levers** — up to 2 actions as ACTION / REASON / MAGNITUDE.
"""
    explained_line = (
        f"- The plant facts account for {st.fmt_signed(total, 1, 'kg/tHM')} of the "
        f"{st.fmt_signed(coke_gap, 1, 'kg/tHM')} "
        "coke gap (sum of the effects in DRIVERS)"
        if total is not None and math.isfinite(coke_gap)
        else "- no plant fact applies"
    )
    sections = [
        (
            "DAYS",
            f"{len(fuel)} operating days, {fuel.index.min():%d %b} to {fuel.index.max():%d %b} (the current day so "
            f"far included); highest {n}: "
            + ", ".join(d.strftime("%d %b") for d in sorted(high_days))
            + f"; lowest {n}: "
            + ", ".join(d.strftime("%d %b") for d in sorted(low_days)),
        ),
        ("FUEL GAP", st.md_table(("Quantity", "Highest days", "Lowest days", "Difference"), gap_rows)),
        (
            "DRIVERS (largest difference first; × usual day-to-day spread)",
            st.md_table(
                (
                    "Driver",
                    "Highest days",
                    "Lowest days",
                    "Difference",
                    "× usual spread",
                    "Coke effect stated by the plant facts",
                ),
                [r for _, r in driver_rows],
            ),
        ),
        ("EXPLAINED", explained_line),
    ]
    return SkillOutput(
        _prompt(
            title="Fuel Drivers",
            scope=f"the last {int(sc['days'])} days",
            chart_id="fuel_drivers",
            chart_title="daily fuel rate with the highest and lowest days",
            form=form,
            sections=sections,
            notes=notes,
        ),
        chart,
        "fuel_drivers",
    )


# ----------------------------------------------------------------------------------------------------- trial board
def _lever_lines(data: PlantData, sc: Mapping[str, Any], q: Mapping[str, Any], notes: Notes) -> list[str]:
    """The day-to-day range the plant already moves each trial's lever through (p10-p90 of the daily means)."""
    levers = dict(sc.get("levers") or {})
    daily = daily_frame(data, int(sc["days"]), notes, q) if levers else pd.DataFrame()
    lines = []
    for fid, key in levers.items():
        if daily.empty or key not in daily.columns:
            continue
        s = _col(daily, key).dropna()
        if len(s) < 5:
            continue
        meta = st.quantity(key, q)
        lines.append(
            f"- {fid} ({meta['label']}): day-to-day {st.fmt(float(s.quantile(0.1)), meta['digits'])}–"
            f"{st.fmt(float(s.quantile(0.9)), meta['digits'], meta['unit'])} (p10–p90 of {len(s)} days), "
            f"usual day-to-day change {st.fmt(float(s.diff().abs().median()), meta['digits'], meta['unit'])}"
        )
    return lines


def _value(fact: Mapping[str, Any]) -> str:
    return "—" if fact.get("value") is None else f"{fact['value']:g} {fact.get('unit', '')}".strip()


def trial_board(data: PlantData, cfg: Mapping[str, Any]) -> SkillOutput:
    """The register's trial list (rank order): what each trial changes, the facts it would settle and why they are
    open, its benefit and design; the other tests the open facts name; the trial rules; and the range the plant
    already moves each lever through."""
    sc, q = cfg["skills"]["trial_board"], cfg["quantities"]
    notes = Notes()
    board = plant_facts.trials()
    listed = [t for t in board.values() if t.get("listed")]
    other = [t for t in board.values() if not t.get("listed")]
    titles = sc.get("trial_titles") or {}
    lever_lines = _lever_lines(data, sc, q, notes)

    trial_rows = [
        [
            t.get("rank", "—"),
            f"{t['id']} {t['title']}",
            t["change"],
            ", ".join(f["id"] for f in t["facts"]) or "—",
            t.get("benefit", "—"),
            t.get("design", "—"),
            t.get("status", "Proposed"),
        ]
        for t in listed
    ]
    seen: dict[str, dict[str, Any]] = {}
    for t in listed:
        for f in t["facts"]:
            seen.setdefault(f["id"], {**f, "trials": []})["trials"].append(t["id"])
    fact_rows = [
        [
            f["id"],
            f["statement"],
            _value(f),
            f.get("status", "—"),
            f.get("grade", "—"),
            ", ".join(f["trials"]),
            f.get("why") or "not marked open in the register; the trial adds a BF-02 measurement",
        ]
        for f in seen.values()
    ]
    note_lines = [f"- {t['id']}: {t['notes']}" for t in listed if t.get("notes")]
    other_rows = [
        [
            t["id"],
            titles.get(t["id"], "not on the trial list; no design in the register"),
            ", ".join(f["id"] for f in t["facts"]),
        ]
        for t in other
    ]
    rules = plant_facts.trial_rules()
    rule_lines = [f"- {rules[k]}" for k in ("ranking", "detection", "rules", "grade") if rules.get(k)]
    open_ids = {f["id"] for t in board.values() for f in t["facts"] if f.get("why")}

    chart = None
    money = [t for t in listed if t.get("benefit_rs_thm")]
    if money:
        chart = go.Figure()
        for t in reversed(money):
            lo, hi = sorted(abs(float(v)) for v in t["benefit_rs_thm"])
            saves = max(t["benefit_rs_thm"]) <= 0
            chart.add_trace(
                go.Bar(
                    y=[f"{t['id']} {t['title']}"],
                    x=[hi],
                    orientation="h",
                    marker_color=BETTER if saves else WORSE,
                    error_x={"type": "data", "symmetric": False, "array": [0], "arrayminus": [hi - lo]},
                    name="saves" if saves else "costs",
                    showlegend=False,
                    hovertemplate=(
                        f"{t['id']}: {'saves' if saves else 'costs'} Rs {lo:,.0f}–{hi:,.0f}/tHM<extra></extra>"
                    ),
                )
            )
        chart.update_layout(
            title="Money at stake per trial, Rs/tHM at plant prices (green saves, amber costs; bar to the high end)",
            xaxis_title="Rs/tHM",
            height=max(320, 44 * len(money) + 120),
            margin={"l": 10, "r": 30, "t": 60, "b": 50},
        )
    header = plant_facts.register_header()
    form = """
**Trial board** — one line: how many trials are on the register's list, how many open facts they would settle, and
how many further tests the facts name without a design.
For each of the first 4 trials on the list: **code — title**: the change, the facts it would settle and why they are
open, the benefit, the design, and whether the day-to-day range under LEVER RANGES already covers the step (then the
routine variation can be read before a planned step).
**Start with** — the trial to run first and why: what it settles, its benefit, and the disruption its design implies.
Use only the trials, facts and benefits listed here; never invent a trial, a title or a number.
"""
    sections = [
        (
            "REGISTER",
            f"{header.get('name', 'Plant fact register')} {header.get('label') or header.get('version', '')}; "
            f"{len(listed)} trials on the list; {len(open_ids)} open facts; {len(other)} further tests named by facts",
        ),
        (
            "TRIAL LIST (rank order: money at stake × uncertainty)",
            st.md_table(("Rank", "Trial", "Change", "Facts it settles", "Benefit", "Design", "Status"), trial_rows),
        ),
        (
            "FACTS THE TRIALS WOULD SETTLE",
            st.md_table(("Fact", "Statement", "Value", "Status", "Grade", "Trials", "Why it is open"), fact_rows),
        ),
        ("TRIAL NOTES", "\n".join(note_lines)),
        (
            "OTHER TESTS THE FACTS NAME",
            st.md_table(("Code", "Title", "Facts that name it"), other_rows) if other_rows else "",
        ),
        ("TRIAL RULES", "\n".join(rule_lines)),
        (
            f"LEVER RANGES (what the plant already moves day to day, last {int(sc['days'])} days)",
            "\n".join(lever_lines),
        ),
    ]
    return SkillOutput(
        _prompt(
            title="Trial Board",
            scope="the register's trial list and the open facts",
            chart_id="trial_board" if chart else None,
            chart_title="money at stake per trial",
            form=form,
            sections=sections,
            notes=notes,
        ),
        chart,
        "trial_board",
    )


# ----------------------------------------------------------------------------------------------------------- runner
SKILLS: dict[str, Callable[..., SkillOutput]] = {
    "shift_pulse": shift_pulse,
    "thermal_watch": thermal_watch,
    "gas_flow_lens": gas_flow_lens,
    "wall_heat_map": wall_heat_map,
    "slag_burden_check": slag_burden_check,
    "fuel_drivers": fuel_drivers,
    "trial_board": trial_board,
}

TITLES = {
    "shift_pulse": "Shift Pulse",
    "thermal_watch": "Thermal Watch",
    "gas_flow_lens": "Gas Flow Lens",
    "wall_heat_map": "Wall Heat Map",
    "slag_burden_check": "Slag & Burden Check",
    "fuel_drivers": "Fuel Drivers",
    "trial_board": "Trial Board",
}


def run(name: str, *args: Any, data: PlantData | None = None, cfg: Mapping[str, Any] | None = None) -> str:
    """Run the skill ``name`` on the tools' state: its chart goes to ``CHARTS`` for ``show_skill_chart``; returns the
    prompt. A skill that fails returns a prompt that tells the model so (it must not guess numbers)."""
    try:
        out = SKILLS[name](data or PlantData(), cfg or config(), *args)
    except Exception as exc:  # noqa: BLE001 - the chat must answer, not break
        log.exception("FurnaceMind skill %s failed", name)
        return (
            f"SKILL: {TITLES.get(name, name)} — the skill could not compute its numbers ({type(exc).__name__}: {exc}). "
            "Tell the operator in one line which data could not be read and suggest trying again later. Do not "
            "estimate any number."
        )
    if out.chart is not None and out.chart_id:
        charts = dict(tool_state.current().get(CHARTS) or {})
        charts[out.chart_id] = out.chart.to_json()
        tool_state.current()[CHARTS] = charts
    return out.prompt
