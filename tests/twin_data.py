"""Synthetic raw sources for the twin tests (SV-1.1, #281), made from the three real shift rows of
``tests/fixtures/coke_energy/shift_inputs.csv``.

Each 8-hour shift copies one template row (cycled) into the four frames the pipeline reads: eight rows of the hourly
dataset, 48 rows of 10-min process data, three hot-metal / slag taps and the DPR row of its date. Masses per hour are
the template's per hour; production and coke change slightly from shift to shift (deterministic), so the 14-day medians
and the windows that straddle two shifts are not trivial. Aggregated by ``build_shift_inputs``, a synthetic shift gives
its template's composition, process means and analyses.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "coke_energy" / "shift_inputs.csv"

MASS = [
    "COKE_CALC_MT",
    "NUTCOKE_CALC_MT",
    "PCI_CALC_MT",
    "SINTER_CALC_MT",
    "ORE_CALC_MT",
    "TOTAL_PELLET_CALC_MT",
    "FLUX_CALC_MT",
    "PRODUCTIONTONNESPERHR",
]
M10_PROC = [
    "co_pct",
    "co2_pct",
    "h2_pct",
    "hot_blast_temp",
    "hot_blast_vol_nm3h",
    "hot_blast_press",
    "top_press_avg",
    "top_temp_avg",
    "body_dp_total",
    "body_perm",
    "body_raft",
    "tuyere_velocity",
    "oxygen_flow",
]
DERIVED = {
    "ORE_DRY",
    "PEL_DRY",
    "SIN_DRY",
    "COKE_DRY",
    "NUT_DRY",
    "PCI_DRY",
    "n_hours",
    "sdate",
    "shift",
    "n_ok10",
    "stave_hl_sum",
    "eta_sd",
    "perm_sd",
    "dp_sd",
    "ttop_sd",
    "hl_sd",
    "n_taps",
    "date_str",
}
DPR = {
    "dpr_hm_t": "hm_t",
    "dpr_coke_rate_dpr": "coke_rate_dpr",
    "dpr_nut_rate_dpr": "nut_rate_dpr",
    "dpr_pci_rate_dpr": "pci_rate_dpr",
    "dpr_slag_rate_dpr": "slag_rate_dpr",
    "dpr_etaco_dpr": "etaco_dpr",
    "dpr_delay_min": "delay_min",
}
SHIFT_START = {"A": 6, "B": 14, "C": 22}


def templates() -> pd.DataFrame:
    return pd.read_csv(FIXTURE, index_col=0)


def _columns(T: pd.DataFrame) -> dict[str, list[str]]:
    hm = [c for c in T.columns if c.startswith(("chem_pct", "slag_")) or c == "hmt_gt_1480c"]
    cw = [c for c in T.columns if c.startswith(("cw_", "delta_t_avg_row", "heat_load_row_"))]
    proc = [c for c in T.columns if c.startswith("h_")]
    other = (
        set(DERIVED)
        | set(hm)
        | set(cw)
        | set(proc)
        | set(M10_PROC)
        | set(MASS)
        | {c for c in T if c.startswith("dpr_")}
    )
    comp = [c for c in T.columns if c not in other]
    return {"hm": hm, "cw": cw, "proc": proc, "comp": comp}


def synthetic_frames(
    start: str = "2026-05-01", days: int = 18
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(fd, m10, hm, dpr) of ``days`` days from ``start`` (IST), in the formats of ``data.coke_energy.sources``."""

    T = templates()
    cols = _columns(T)
    fd_rows: list[dict] = []
    m10_rows: list[dict] = []
    hm_rows: list[dict] = []
    dpr_rows: list[dict] = []
    i = 0
    for d in range(days):
        day = pd.Timestamp(start) + pd.Timedelta(days=d)
        dust: dict[str, float] = {}
        for letter in ("A", "B", "C"):
            tpl = T.iloc[i % len(T)]
            prod_f = 1 + 0.015 * np.sin(0.7 * i)  # production, +-1.5 %
            coke_f = 1 + 0.008 * np.cos(1.3 * i)  # coke per tonne, +-0.8 %
            gas_f = 1 + 0.003 * np.sin(0.9 * i + 1.0)  # CO2 / CO, +-0.3 %
            t0 = day + pd.Timedelta(hours=SHIFT_START[letter])
            n = float(tpl.n_hours)
            for h in range(8):
                r = {"time": t0 + pd.Timedelta(hours=h)}
                for c in MASS:
                    r[c] = tpl[c] / n * prod_f * (coke_f if c == "COKE_CALC_MT" else 1.0)
                r.update({c: tpl[c] for c in cols["comp"]})
                r.update({c[2:]: tpl[c] for c in cols["proc"]})
                fd_rows.append(r)
            for k in range(48):
                r = {"ts": t0 + pd.Timedelta(minutes=10 * k)}
                r.update({c: tpl[c] for c in M10_PROC + cols["cw"]})
                r["co2_pct"] = tpl["co2_pct"] * gas_f
                r["co_pct"] = tpl["co_pct"] / gas_f
                m10_rows.append(r)
            for k in range(3):
                r = {"t": t0 + pd.Timedelta(hours=1 + 2.5 * k)}
                r.update({c: tpl[c] for c in cols["hm"]})
                hm_rows.append(r)
            if not dust:
                dust = {
                    "dc": tpl.dpr_dc_dust_kgthm,
                    "gcp": tpl.dpr_gcp_dust_kgthm,
                    **{v: tpl[k] for k, v in DPR.items()},
                }
            i += 1
        hm_t = dust["hm_t"]
        dpr_rows.append(
            {
                "date": day,
                **{v: dust[v] for v in DPR.values()},
                "dc_dust_t": dust["dc"] * hm_t / 1000,
                "gcp_dust_t": dust["gcp"] * hm_t / 1000,
            }
        )
    fd = pd.DataFrame(fd_rows)
    m10 = pd.DataFrame(m10_rows)
    hm = pd.DataFrame(hm_rows)
    dpr = pd.DataFrame(dpr_rows).set_index("date")
    return fd, m10, hm, dpr
