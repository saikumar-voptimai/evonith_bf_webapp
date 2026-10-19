"""Top-gas dust for the Blend Mix Optimiser's slag balance, from the DPR and the plant's dust analyses.

Two streams leave through the top and are deducted from the slag balance (the dust was charged but never reached
the slag):

* dust catcher - DPR ``flue_dust_mt``, analyses ``DUST_1_*`` in the hourly furnace dataset;
* gas cleaning plant (GCP) - DPR ``gcp_dust_mt``, analyses ``DUST_2_*``.

Cast-house dust (leaves at the cast house) and stock-house dust (a handling loss before charging) are not deducted.

Quantities are kg per charge over a recent window: the DPR's daily dust over the charges counted in
``charge_data`` on the same days, so the deduction follows the charging rate of the blend being evaluated.
Composition is configured (``setting_bmo.yml`` dust_inputs) unless ``data_sources.dust_chemistry_source`` is
"dataset": then the hourly dataset's analyses averaged over a recent window, converted to the page's fields (Fe, Mn,
P, S, Ti, Zn as elements). Those analyses are sparse (the dust catcher's: 1-3 samples since Jun 2026). Everything
stays editable on the page. Only :func:`load_dataset_dust_analysis` does I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

IST = "Asia/Kolkata"

# oxide -> element mass fractions
FE_IN_FE2O3 = 2 * 55.845 / 159.688
MN_IN_MNO = 54.938 / 70.937
P_IN_P2O5 = 2 * 30.974 / 141.943
S_IN_SO3 = 32.06 / 80.06
TI_IN_TIO2 = 47.867 / 79.866
ZN_IN_ZNO = 65.38 / 81.38


@dataclass(frozen=True)
class DustStream:
    dust_id: str
    name: str
    dpr_column: str
    dataset_prefix: str


DUST_STREAMS: tuple[DustStream, ...] = (
    DustStream("dust_catcher", "Dust catcher", "flue_dust_mt", "DUST_1"),
    DustStream("gcp_dust", "GCP dust", "gcp_dust_mt", "DUST_2"),
)
DPR_HM_COLUMN = "total_hot_metal_mt"

# page field <- (dataset analysis suffix, factor to the page's basis)
_ANALYSIS_FIELDS: dict[str, tuple[str, float]] = {
    "sio2_pct": ("CHEMICAL_ANALYSIS_SIO2_PCT", 1.0),
    "al2o3_pct": ("CHEMICAL_ANALYSIS_AL2O3_PCT", 1.0),
    "cao_pct": ("CHEMICAL_ANALYSIS_CAO_PCT", 1.0),
    "mgo_pct": ("CHEMICAL_ANALYSIS_MGO_PCT", 1.0),
    "na2o_pct": ("CHEMICAL_ANALYSIS_NA2O_PCT", 1.0),
    "k2o_pct": ("CHEMICAL_ANALYSIS_K2O_PCT", 1.0),
    "mn_pct": ("CHEMICAL_ANALYSIS_MNO_PCT", MN_IN_MNO),
    "p_pct": ("CHEMICAL_ANALYSIS_P2O5_PCT", P_IN_P2O5),
    "s_pct": ("CHEMICAL_ANALYSIS_SO3_PCT", S_IN_SO3),
    "ti_pct": ("CHEMICAL_ANALYSIS_TIO2_PCT", TI_IN_TIO2),
    "zn_pct": ("CHEMICAL_ANALYSIS_ZNO_PCT", ZN_IN_ZNO),
}
_FE_TOTAL = "BASIC_ANALYSIS_FE_TOTAL_PCT"
_FE2O3 = "CHEMICAL_ANALYSIS_FE2O3_PCT"


def dataset_dust_columns(columns: Iterable[str]) -> list[str]:
    """The dust-analysis columns present in a furnace dataset header."""
    wanted = {f"{s.dataset_prefix}_{suffix}" for s in DUST_STREAMS
              for suffix in [v[0] for v in _ANALYSIS_FIELDS.values()] + [_FE_TOTAL, _FE2O3]}
    return [c for c in columns if c in wanted]


def _ist_dates(index: pd.Index) -> pd.Index:
    ts = pd.to_datetime(index, utc=True)
    return pd.Index(ts.tz_convert(IST).date)


def dpr_dust_rates(dpr: pd.DataFrame, charge_times: pd.Index | pd.Series, *, min_charges: int = 100) -> dict[str, dict]:
    """Dust per stream over the window: kg per charge, kg/tHM and t/day.

    ``dpr``: DPR rows indexed by time (UTC); a row stamped 18:30 UTC is the IST day that starts then.
    ``charge_times``: one timestamp (UTC) per charge. Days with fewer than ``min_charges`` charges (stops, part days)
    or no DPR dust figure are left out; the rest are pooled (sum of dust over sum of charges)."""
    out: dict[str, dict] = {}
    if dpr is None or dpr.empty:
        return out
    d = dpr.copy()
    d["_day"] = _ist_dates(d.index)
    charges = pd.Series(1, index=_ist_dates(pd.Index(charge_times))).groupby(level=0).sum()
    for s in DUST_STREAMS:
        if s.dpr_column not in d.columns:
            continue
        cols = [s.dpr_column] + ([DPR_HM_COLUMN] if DPR_HM_COLUMN in d.columns else [])
        daily = d[cols].apply(pd.to_numeric, errors="coerce").groupby(d["_day"]).sum(min_count=1)
        daily["charges"] = charges.reindex(daily.index).fillna(0)
        ok = daily[daily[s.dpr_column].notna() & (daily["charges"] >= min_charges)]
        if ok.empty:
            continue
        dust_t = float(ok[s.dpr_column].sum())
        hm_t = float(ok[DPR_HM_COLUMN].sum()) if DPR_HM_COLUMN in ok else 0.0
        out[s.dust_id] = dict(
            kg_per_charge=1000.0 * dust_t / float(ok["charges"].sum()),
            kg_per_thm=1000.0 * dust_t / hm_t if hm_t > 0 else None,
            t_per_day=dust_t / len(ok), days=int(len(ok)),
            first_day=str(min(ok.index)), last_day=str(max(ok.index)),
        )
    return out


def dataset_dust_analysis(df: pd.DataFrame, *, window_days: int = 60) -> dict[str, dict]:
    """Mean dust analysis per stream over the last ``window_days`` of the dataset, in the page's fields.

    ``df``: the hourly furnace dataset (``time`` column or a DatetimeIndex) with the ``DUST_n_*`` columns. Zeros are
    treated as missing. Fe is the basic analysis' total Fe, else Fe2O3 x 0.699. ``sample_time`` is the last time the
    stream's SiO2 analysis changed (the dataset carries an analysis forward until the next one)."""
    out: dict[str, dict] = {}
    if df is None or df.empty:
        return out
    t = pd.to_datetime(df["time"] if "time" in df.columns else df.index, errors="coerce")
    frame = df.assign(_t=t.values).dropna(subset=["_t"]).set_index("_t").sort_index()
    recent = frame[frame.index > frame.index.max() - pd.Timedelta(days=window_days)]
    for s in DUST_STREAMS:
        col = lambda suffix: f"{s.dataset_prefix}_{suffix}"  # noqa: E731
        if col(_ANALYSIS_FIELDS["sio2_pct"][0]) not in recent.columns:
            continue
        mean = lambda c: (pd.to_numeric(recent[c], errors="coerce").where(lambda v: v > 0).mean()  # noqa: E731
                          if c in recent.columns else float("nan"))
        row: dict[str, Any] = {}
        for field, (suffix, factor) in _ANALYSIS_FIELDS.items():
            v = mean(col(suffix))
            if pd.notna(v):
                row[field] = float(v) * factor
        fe = mean(col(_FE_TOTAL))
        if pd.isna(fe):
            fe = mean(col(_FE2O3)) * FE_IN_FE2O3
        if pd.notna(fe):
            row["fe_pct"] = float(fe)
        if "sio2_pct" not in row:
            continue
        si = pd.to_numeric(frame[col(_ANALYSIS_FIELDS["sio2_pct"][0])], errors="coerce").dropna()
        changes = si[si.ne(si.shift())]
        row["sample_time"] = str(changes.index.max().date()) if len(changes) else ""
        out[s.dust_id] = row
    return out


def load_dataset_dust_analysis(path: str | Path, *, window_days: int = 60) -> dict[str, dict]:
    """:func:`dataset_dust_analysis` on the hourly furnace dataset CSV (reads only the time and dust columns)."""
    p = Path(path)
    if not p.exists():
        return {}
    header = pd.read_csv(p, nrows=0).columns
    cols = dataset_dust_columns(header)
    if not cols:
        return {}
    time_col = "time" if "time" in header else header[0]
    df = pd.read_csv(p, usecols=[time_col, *cols]).rename(columns={time_col: "time"})
    return dataset_dust_analysis(df, window_days=window_days)


def dust_cfg_with_live(dust_cfg: Iterable[Mapping[str, Any]], rates: Mapping[str, Mapping[str, Any]] | None,
                       analysis: Mapping[str, Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    """Configured dust rows with the DPR quantity (kg per charge) and the dataset's analysis laid over them.

    A row keeps its configured values for whatever a source does not supply; ``source`` records what was used."""
    rows = []
    for item in dust_cfg or []:
        row = dict(item)
        did = str(row.get("dust_id", ""))
        notes = []
        r = (rates or {}).get(did)
        if r:
            row["rate_basis"] = "kg_per_charge"
            row["quantity_kg_per_charge"] = round(float(r["kg_per_charge"]), 1)
            notes.append(f"DPR {r['first_day']} to {r['last_day']} ({r['days']} d)")
        a = (analysis or {}).get(did)
        if a:
            for k, v in a.items():
                if k.endswith("_pct"):
                    row[k] = round(float(v), 3)
            notes.append(f"analysis {a.get('sample_time') or 'dataset'}")
        if notes:
            row["source"] = "; ".join(notes)
        rows.append(row)
    return rows


def dust_caption(rates: Mapping[str, Mapping[str, Any]] | None, analysis: Mapping[str, Mapping[str, Any]] | None,
                 saved_ids: Iterable[str] = ()) -> str:
    """One line per stream for the page: what the DPR and the analyses say, and whether saved values override."""
    saved = set(saved_ids)
    parts = []
    for s in DUST_STREAMS:
        r = (rates or {}).get(s.dust_id)
        a = (analysis or {}).get(s.dust_id)
        q = (f"{r['kg_per_charge']:.0f} kg/charge ({r['kg_per_thm']:.1f} kg/tHM, {r['days']} days to {r['last_day']})"
             if r and r.get("kg_per_thm") else
             f"{r['kg_per_charge']:.0f} kg/charge ({r['days']} days to {r['last_day']})" if r else
             "DPR unavailable, configured quantity")
        c = (f"CaO {a.get('cao_pct', 0):.2f} %, MgO {a.get('mgo_pct', 0):.2f} %, analysis {a.get('sample_time', '')}"
             if a else "chemistry as configured")
        parts.append(f"**{s.name}**: {q}; {c}" + (" · *saved values in use*" if s.dust_id in saved else ""))
    return "  \n".join(parts)
