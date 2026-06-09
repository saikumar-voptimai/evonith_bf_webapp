"""Show observed slag rate and HM chemistry trends across recent days.

Pulls daily HM/Slag chemistry and DPR mass totals from Neon, computes the
per-day observed slag rate, and shows the HM Si/Mn/Ti % distribution that
feeds the slag-balance settings. Useful for sanity-checking the model
assumptions before tuning recovery factors and the slag correction factor.

Run:
    python scripts/validate_slag_balance.py [--days 30]
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

from furnace_data.neon_db.offline import fetch_offline_report  # noqa: E402


def _resample_to_day(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("Asia/Kolkata")
    df["_date"] = df.index.date
    return df


def fetch_validation_frame(days: int) -> pd.DataFrame:
    """Build a per-day comparison frame of HM chemistry and DPR slag rate."""
    end_utc = datetime.now(timezone.utc)
    start_utc = end_utc - timedelta(days=days)

    print(f"Fetching DPR for last {days} days ...")
    dpr = fetch_offline_report("DPR", (start_utc, end_utc))
    print(f"  -> {len(dpr)} DPR rows")
    print(f"Fetching HM_SLAG for last {days} days ...")
    hm = fetch_offline_report("HM_SLAG", (start_utc, end_utc))
    print(f"  -> {len(hm)} HM_SLAG rows")

    dpr = _resample_to_day(dpr)
    hm = _resample_to_day(hm)

    rows = []
    if dpr.empty:
        print("No DPR data; nothing to validate.")
        return pd.DataFrame()

    for day, dpr_day in dpr.groupby("_date"):
        dpr_row = dpr_day.mean(numeric_only=True)
        hot_metal_mt = float(dpr_row.get("total_hot_metal_mt", 0.0) or 0.0)
        observed_slag_mt = float(dpr_row.get("slag_generation_mt", 0.0) or 0.0)
        if hot_metal_mt <= 0 or observed_slag_mt <= 0:
            continue

        hm_day = (
            hm[hm["_date"] == day].mean(numeric_only=True)
            if not hm.empty
            else pd.Series(dtype=float)
        )

        observed_rate = observed_slag_mt / hot_metal_mt * 1000.0

        chem_c = float(hm_day.get("chem_pct_c", float("nan")))
        chem_si = float(hm_day.get("chem_pct_si", float("nan")))
        chem_s = float(hm_day.get("chem_pct_s", float("nan")))
        chem_mn = float(hm_day.get("chem_pct_mn", float("nan")))
        chem_ti = float(hm_day.get("chem_pct_ti", float("nan")))
        chem_p = float(hm_day.get("chem_pct_p", float("nan")))
        slag_sio2 = float(hm_day.get("slag_pct_sio2", float("nan")))
        slag_cao = float(hm_day.get("slag_pct_cao", float("nan")))
        slag_mgo = float(hm_day.get("slag_pct_mgo", float("nan")))
        slag_al2o3 = float(hm_day.get("slag_pct_al2o3", float("nan")))
        slag_mno = float(hm_day.get("slag_pct_mno", float("nan")))
        slag_tio2 = float(hm_day.get("slag_pct_tio2", float("nan")))
        slag_basicity = float(hm_day.get("slag_basicity", float("nan")))

        # Implied SiO2 reduction at observed HM Si%
        sio2_to_hm_mt = (
            (hot_metal_mt * chem_si / 100.0) * 2.14 if not pd.isna(chem_si) else float("nan")
        )
        # Implied TiO2 to HM at observed HM Ti%
        tio2_to_hm_mt = (
            (hot_metal_mt * chem_ti / 100.0) * (79.866 / 47.867)
            if not pd.isna(chem_ti)
            else float("nan")
        )
        # Implied MnO to HM at observed HM Mn%
        mno_to_hm_mt = (
            (hot_metal_mt * chem_mn / 100.0) * 1.291
            if not pd.isna(chem_mn)
            else float("nan")
        )

        rows.append(
            {
                "date": day,
                "hm_mt": round(hot_metal_mt, 1),
                "slag_mt": round(observed_slag_mt, 1),
                "slag_rate_kg_thm": round(observed_rate, 1),
                "hm_c_pct": round(chem_c, 3),
                "hm_si_pct": round(chem_si, 3),
                "hm_s_pct": round(chem_s, 4),
                "hm_mn_pct": round(chem_mn, 4),
                "hm_ti_pct": round(chem_ti, 4),
                "hm_p_pct": round(chem_p, 4),
                "slag_sio2_pct": round(slag_sio2, 2),
                "slag_cao_pct": round(slag_cao, 2),
                "slag_mgo_pct": round(slag_mgo, 2),
                "slag_al2o3_pct": round(slag_al2o3, 2),
                "slag_mno_pct": round(slag_mno, 3),
                "slag_tio2_pct": round(slag_tio2, 3),
                "slag_basicity": round(slag_basicity, 3),
                "sio2_to_hm_mt": round(sio2_to_hm_mt, 2)
                if not pd.isna(sio2_to_hm_mt)
                else None,
                "mno_to_hm_mt": round(mno_to_hm_mt, 2)
                if not pd.isna(mno_to_hm_mt)
                else None,
                "tio2_to_hm_mt": round(tio2_to_hm_mt, 2)
                if not pd.isna(tio2_to_hm_mt)
                else None,
            }
        )

    return pd.DataFrame(rows).sort_values("date")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--out", default="scripts/slag_validation_results.csv")
    args = parser.parse_args()

    if not os.getenv("DATABASE_URL"):
        print("ERROR: DATABASE_URL not set; load .env or export it.")
        sys.exit(1)

    df = fetch_validation_frame(days=args.days)
    if df.empty:
        print("No valid days to report.")
        return

    print()
    print("=== Daily observed slag rate and HM/Slag chemistry ===")
    main_cols = [
        "date",
        "hm_mt",
        "slag_mt",
        "slag_rate_kg_thm",
        "hm_si_pct",
        "hm_mn_pct",
        "hm_ti_pct",
        "slag_sio2_pct",
        "slag_mno_pct",
        "slag_tio2_pct",
        "slag_basicity",
    ]
    print(df[main_cols].to_string(index=False))

    print()
    print("=== Implied HM-side removals at observed HM percentages ===")
    impl_cols = [
        "date",
        "hm_mt",
        "sio2_to_hm_mt",
        "mno_to_hm_mt",
        "tio2_to_hm_mt",
    ]
    print(df[impl_cols].to_string(index=False))

    print()
    print("=== Summary statistics ===")
    summary_cols = [
        "slag_rate_kg_thm",
        "hm_si_pct",
        "hm_mn_pct",
        "hm_ti_pct",
        "slag_sio2_pct",
        "slag_mno_pct",
        "slag_tio2_pct",
        "slag_basicity",
        "sio2_to_hm_mt",
        "mno_to_hm_mt",
        "tio2_to_hm_mt",
    ]
    print(df[summary_cols].describe().round(3).to_string())

    out_path = REPO_ROOT / args.out
    df.to_csv(out_path, index=False)
    print(f"\nSaved table to: {out_path}")


if __name__ == "__main__":
    main()
