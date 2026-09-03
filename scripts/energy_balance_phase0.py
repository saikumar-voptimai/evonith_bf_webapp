"""Phase 0: can a blast-furnace energy balance be closed on this plant's data?

Run:  python scripts/energy_balance_phase0.py

This is a GATE, not a deliverable. It decides whether the physics-induced
recommendation design is worth building, by asking one question:

    with shell losses MEASURED, is the thermal efficiency eta a stable
    constant across quarters?

Why that question. Every data-driven attempt this week failed because the
controller had already cancelled the signal we were trying to identify. An
energy balance is different: it is an accounting identity, and a controller
cannot violate conservation of energy. So if the structure is right, eta should
hold across quarters where a fitted response surface swings from R2 = -1.4 to
+0.68. If eta drifts as badly as the ML did, the balance is mis-specified and
the design needs rethinking before any module gets written.

Sources, each chosen for a reason established earlier:
  * coke + nut coke     charge_data          (the only trustworthy fuel mass)
  * PCI, slag, hot metal DPR                 (slag is usable; PCI ratio 1.044)
  * blast, chemistry     static hourly CSV   (daily means)
  * shell heat loss      Influx cooling_water + heatload_delta_t

Units, resolved by measurement: the stave heat-load tags are in MW. Multiplying
by 3.6 gives GJ/hr (median 16.8, matching the operator's expected 15-20). The
'GW.hr -> GJ' conversion of x3600 is 1000x too large.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from furnace_data.influx.online import fetch_online_df  # noqa: E402
from furnace_data.offline import fetch_offline_data  # noqa: E402

IST = "Asia/Kolkata"
STATIC = REPO / "src" / "assets" / "data" / "furnace_dataset.csv"

# --- physical constants -------------------------------------------------------
H_SLAG_MJ_PER_KG = 1.80          # slag enthalpy at tap, findings section 2
# Hot metal at the stated 1500 C tap temperature: solid Fe 25->1150 (0.75
# kJ/kg.K), fusion of pig iron (247 kJ/kg), liquid 1150->1500 (0.82 kJ/kg.K).
H_HM_MJ_PER_T = 1378.0
# Free burden moisture: latent heat plus sensible to the top-gas temperature.
H_BURDEN_MOISTURE_MJ_PER_KG = 2.70
# Fe that stops at FeO in the slag was only partly reduced: Fe2O3 -> FeO is
# 7.38 - 4.82 = 2.56 MJ per kg Fe, not the full 7.38.
H_FE_TO_FEO_MJ_PER_KG = 2.56
FE_IN_FEO_FRAC = 55.845 / 71.844
H_SI_MJ_PER_KG = 24.6            # SiO2 + 2C -> Si + 2CO
H_MN_MJ_PER_KG = 4.8
H_H2O_MJ_PER_KG = 7.3            # H2O + C -> CO + H2
H_CO2_MJ_PER_KG = 2.46           # flux calcination
H_C_TO_CO_MJ_PER_KG = 9.2        # 2C + O2 -> 2CO, at the tuyeres
H_C_FULL_MJ_PER_KG = 32.8        # C + O2 -> CO2, full combustion potential
CV_CO_MJ_PER_NM3 = 12.63         # LHV of CO
CV_H2_MJ_PER_NM3 = 10.78         # LHV of H2
TOPGAS_CP_KJ_PER_NM3_K = 1.38
N2_IN_AIR_PCT = 79.2
# Fe2O3 -> 2Fe + 3/2 O2, +824 kJ/mol Fe2O3 = 7.38 MJ per kg Fe. This is the
# right pairing when carbon is counted at its FULL combustion potential and the
# unburnt CO leaving the top is booked as an output.
H_FE_REDUCTION_MJ_PER_KG = 7.38
HM_FE_PCT_DEFAULT = 94.5
C_FRAC_COKE = 0.87
C_FRAC_PCI = 0.75
BLAST_CP_KJ_PER_NM3_K = 1.40
T_REF_C = 25.0
HEATLOAD_MW_TO_GJ_PER_HR = 3.6   # measured, see module docstring
HEATLOAD_CLEAN_MW = (2.0, 12.0)  # drop zeros/dropouts and the 236 MW spikes


def _ist_date(index) -> pd.Series:
    idx = pd.DatetimeIndex(index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return pd.Series(idx.tz_convert(IST).date, index=index)


def daily_charge() -> pd.DataFrame:
    df = fetch_offline_data("charge_data", time_range="full", query_type="raw")
    num = lambda cols: sum(  # noqa: E731
        pd.to_numeric(df[c], errors="coerce").fillna(0.0) for c in cols if c in df.columns
    )
    out = pd.DataFrame(index=df.index)
    out["coke_mt"] = num(["coke_1_mt", "coke_2_mt"])
    out["nut_coke_mt"] = num(["nut_coke_1_mt", "nut_coke_2_mt"])
    out["flux_mt"] = num([f"flux_{i}_mt" for i in range(1, 4)])
    out["sinter_mt"] = num([f"sinter_{i}_mt" for i in range(1, 5)])
    out["ore_mt"] = num([f"ore_{i}_mt" for i in range(1, 13)])
    out["pellet_mt"] = num([f"pellet_{i}_mt" for i in range(1, 3)])
    out["date"] = _ist_date(df.index)
    daily = out.groupby("date").sum(numeric_only=True)
    daily["charges"] = out.groupby("date").size()
    daily.index = pd.to_datetime(daily.index)
    return daily[daily["charges"].between(120, 190)]


def daily_dpr() -> pd.DataFrame:
    df = fetch_offline_data("dpr_data", time_range="full", query_type="raw")
    out = pd.DataFrame(index=df.index)
    for src, dst in (
        ("slag_generation_mt", "slag_mt"),
        ("total_hot_metal_mt", "hm_mt"),
        ("pci_mt", "pci_mt"),
    ):
        out[dst] = pd.to_numeric(df[src], errors="coerce")
    out["date"] = _ist_date(df.index)
    daily = out.groupby("date").mean(numeric_only=True)
    daily.index = pd.to_datetime(daily.index)
    return daily[daily["hm_mt"].between(1200, 3200)]


def daily_static() -> pd.DataFrame:
    df = pd.read_csv(STATIC)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])
    keep = {
        "HOT BLAST VOLUMENM3/HR.": "cbv_nm3h",
        "HOT BLAST TEMP.OC": "blast_temp",
        "O2 ENRICHMENT %": "o2_enrich",
        "STEAMKGS/HR.": "steam_kgh",
        "TOPPRESSUREBAR": "top_press",
        "HOT BLAST PRESSUREBAR": "blast_press",
        "CHEM_PCT_SI": "hm_si",
        "CHEM_PCT_C": "hm_c",
        "CHEM_PCT_MN": "hm_mn",
        "CHEM_PCT_FE": "hm_fe",
        "SLAG_PCT_FEO": "slag_feo_pct",
        "ORE_TM%": "ore_tm",
        "PELLET_PCT_TM": "pellet_tm",
        "FLUX_TM%": "flux_tm",
        "COKE_MOIST%": "coke_moist",
        "NUTCOKE_MOIST%": "nutcoke_moist",
        "FLUX_LOI%": "flux_loi",
        "PRODUCTIONTONNESPERHR": "prod_tph",
    }
    have = {k: v for k, v in keep.items() if k in df.columns}
    sub = df[["time", *have]].rename(columns=have)
    for c in have.values():
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    daily = sub.set_index("time").resample("1D").mean(numeric_only=True)
    daily.index = pd.to_datetime(daily.index.date)
    return daily


def daily_heat_loss(days: int = 300) -> pd.DataFrame:
    """Shell loss two ways: measured stave rows 6-10, and flow-scaled total."""

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    df = fetch_online_df(
        selected_measurements=["heatload_delta_t", "cooling_water", "process_params"],
        time_range="last 1 week", request_type="windowed-average", window_by="1 hour",
        start_time_override=start, end_time_override=end, column_naming="field",
    )
    quad = [f"heat_load_r{r}_q{q}" for r in range(6, 11) for q in range(1, 5)]
    have = [c for c in quad if c in df.columns]
    stave_mw = df[have].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    stave_mw = stave_mw.where(stave_mw.between(*HEATLOAD_CLEAN_MW))

    flows = {c: pd.to_numeric(df[c], errors="coerce")
             for c in df.columns if c.endswith("_flow_m3h")}
    total_flow = sum(flows.values()) if flows else pd.Series(index=df.index, dtype=float)
    # Rows 6-10 are bosh/belly and lower stack. Scale their measured loss by the
    # flow ratio to estimate hearth + bottom + tuyere nose + upper shaft as well.
    stave_flow = sum(
        v for k, v in flows.items()
        if "bosh_belly" in k or "lower_stack" in k
    ) if flows else pd.Series(index=df.index, dtype=float)
    ratio = (total_flow / stave_flow).replace([np.inf, -np.inf], np.nan)

    out = pd.DataFrame(index=df.index)
    out["stave_gj_per_hr"] = stave_mw * HEATLOAD_MW_TO_GJ_PER_HR
    out["flow_ratio"] = ratio
    out["total_gj_per_hr"] = out["stave_gj_per_hr"] * ratio
    for src, dst in (
        ("co_pct", "co_pct"), ("co2_pct", "co2_pct"), ("h2_pct", "h2_pct"),
        ("top_temp_avg", "top_temp"),
    ):
        if src in df.columns:
            out[dst] = pd.to_numeric(df[src], errors="coerce")
    out["date"] = _ist_date(df.index)
    daily = out.groupby("date").mean(numeric_only=True)
    daily.index = pd.to_datetime(daily.index)
    return daily


def build() -> pd.DataFrame:
    df = (
        daily_charge()
        .join(daily_dpr(), how="inner")
        .join(daily_static(), how="inner")
        .join(daily_heat_loss(), how="left")
    )
    hm = df["hm_mt"]
    df["coke_rate"] = df["coke_mt"] / hm * 1000.0
    df["nut_rate"] = df["nut_coke_mt"] / hm * 1000.0
    df["pci_rate"] = df["pci_mt"] / hm * 1000.0
    df["slag_rate"] = df["slag_mt"] / hm * 1000.0
    df["cbv_per_thm"] = df["cbv_nm3h"] * 24.0 / hm
    df["steam_per_thm"] = df["steam_kgh"] * 24.0 / hm
    df["flux_rate"] = df["flux_mt"] / hm * 1000.0

    # --- demand, MJ per tonne hot metal --------------------------------------
    df["q_slag"] = H_SLAG_MJ_PER_KG * df["slag_rate"]
    df["q_hm"] = H_HM_MJ_PER_T
    df["q_si"] = H_SI_MJ_PER_KG * df["hm_si"].fillna(0.5) / 100.0 * 1000.0
    df["q_mn"] = H_MN_MJ_PER_KG * df["hm_mn"].fillna(0.2) / 100.0 * 1000.0
    df["q_steam"] = H_H2O_MJ_PER_KG * df["steam_per_thm"]
    df["q_fe_reduction"] = (
        H_FE_REDUCTION_MJ_PER_KG
        * df.get("hm_fe", pd.Series(HM_FE_PCT_DEFAULT, index=df.index))
        .fillna(HM_FE_PCT_DEFAULT) / 100.0 * 1000.0
    )
    # Burden moisture, from the RM moisture percentages against charged tonnes.
    water_mt = (
        df["ore_mt"] * df["ore_tm"].fillna(0.0)
        + df["pellet_mt"] * df["pellet_tm"].fillna(0.0)
        + df["flux_mt"] * df["flux_tm"].fillna(0.0)
        + df["coke_mt"] * df["coke_moist"].fillna(0.0)
        + df["nut_coke_mt"] * df["nutcoke_moist"].fillna(0.0)
    ) / 100.0
    df["burden_water_kg_per_thm"] = water_mt / hm * 1000.0
    df["q_burden_moisture"] = (
        H_BURDEN_MOISTURE_MJ_PER_KG * df["burden_water_kg_per_thm"]
    )
    # Fe held in the slag as FeO: partly reduced only.
    df["q_feo_slag"] = (
        H_FE_TO_FEO_MJ_PER_KG
        * df["slag_rate"] * df["slag_feo_pct"].fillna(0.5) / 100.0 * FE_IN_FEO_FRAC
    )
    df["q_calcination"] = (
        H_CO2_MJ_PER_KG * df["flux_rate"] * df["flux_loi"].fillna(40.0) / 100.0
    )
    df["q_loss_stave"] = df["stave_gj_per_hr"] * 24.0 * 1000.0 / hm
    df["q_loss_total"] = df["total_gj_per_hr"] * 24.0 * 1000.0 / hm

    demand_core = (
        df["q_slag"] + df["q_hm"] + df["q_fe_reduction"] + df["q_si"] + df["q_mn"]
        + df["q_steam"] + df["q_calcination"] + df["q_burden_moisture"]
        + df["q_feo_slag"]
    )
    df["q_demand_stave"] = demand_core + df["q_loss_stave"]
    df["q_demand_total"] = demand_core + df["q_loss_total"]

    # --- TOP GAS, from a nitrogen balance ------------------------------------
    #
    # N2 is inert: every Nm3 that enters with the blast leaves at the top. So
    # the top gas volume follows from measured quantities alone, with no fitted
    # constant:
    #
    #     N2% in blast   = 100 - O2%        = 79.2 - enrichment
    #     N2% in top gas = 100 - CO - CO2 - H2
    #     V_top          = V_blast * N2%_blast / N2%_top
    #
    # Top gas then carries away heat two ways, and BOTH are measured:
    #   sensible  - volume x cp x (top temp - ambient)
    #   chemical  - the CO and H2 that leave unburnt, at their calorific values.
    # The chemical part is the large one and is exactly what the furnace failed
    # to extract, which is why the balance cannot close without it.
    n2_blast_pct = N2_IN_AIR_PCT - df["o2_enrich"].fillna(0.0)
    n2_top_pct = (
        100.0 - df["co_pct"].fillna(0.0) - df["co2_pct"].fillna(0.0)
        - df["h2_pct"].fillna(0.0)
    )
    df["topgas_nm3_per_thm"] = np.where(
        n2_top_pct > 30.0, df["cbv_per_thm"] * n2_blast_pct / n2_top_pct, np.nan
    )
    df["q_topgas_sensible"] = (
        df["topgas_nm3_per_thm"] * TOPGAS_CP_KJ_PER_NM3_K
        * (df["top_temp"] - T_REF_C) / 1000.0
    )
    df["q_topgas_chemical"] = df["topgas_nm3_per_thm"] * (
        df["co_pct"] * CV_CO_MJ_PER_NM3 + df["h2_pct"] * CV_H2_MJ_PER_NM3
    ) / 100.0
    df["q_topgas"] = df["q_topgas_sensible"] + df["q_topgas_chemical"]

    # --- input, MJ per tonne hot metal ---------------------------------------
    # Carbon is counted at its FULL combustion potential (C -> CO2). What the
    # furnace does not extract leaves as unburnt CO and appears on the output
    # side as q_topgas_chemical, so the two are consistent and closure should
    # land near 1.0 rather than at some arbitrary lumped efficiency.
    df["carbon_charged_kg_per_thm"] = (
        (df["coke_rate"] + df["nut_rate"]) * C_FRAC_COKE
        + df["pci_rate"] * C_FRAC_PCI
    )
    # Carbon that dissolves into the hot metal never burns. Crediting its full
    # combustion potential as input over-states the input by ~1,400 MJ/tHM at
    # 4.3% C, which was the whole of the unexplained gap.
    df["carbon_to_hm_kg_per_thm"] = df["hm_c"].fillna(4.3) / 100.0 * 1000.0
    df["carbon_kg_per_thm"] = (
        df["carbon_charged_kg_per_thm"] - df["carbon_to_hm_kg_per_thm"]
    ).clip(lower=0.0)
    df["q_carbon"] = H_C_FULL_MJ_PER_KG * df["carbon_kg_per_thm"]
    df["q_blast"] = (
        df["cbv_per_thm"] * BLAST_CP_KJ_PER_NM3_K * (df["blast_temp"] - T_REF_C) / 1000.0
    )
    df["q_input"] = df["q_carbon"] + df["q_blast"]

    for basis in ("stave", "total"):
        df[f"q_output_{basis}"] = df[f"q_demand_{basis}"] + df["q_topgas"]
        df[f"closure_{basis}"] = df[f"q_output_{basis}"] / df["q_input"]
        # Legacy lumped efficiency, kept so the before/after is visible.
        df[f"eta_{basis}"] = df[f"q_demand_{basis}"] / df["q_input"]
    return df.replace([np.inf, -np.inf], np.nan)


def banner(t: str) -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def main() -> None:
    df = build()
    banner("0. SAMPLE")
    print(f"  days: {len(df)}   {df.index.min().date()} -> {df.index.max().date()}")
    print(f"  with heat-loss data: {int(df['q_loss_stave'].notna().sum())}")

    banner("1. SHELL HEAT LOSS")
    for col, label in (("q_loss_stave", "stave rows 6-10 (measured)"),
                       ("q_loss_total", "flow-scaled all circuits")):
        s = df[col].dropna()
        if len(s):
            print(f"  {label:32s} median {s.median():7.0f}  p5 {s.quantile(.05):7.0f}"
                  f"  p95 {s.quantile(.95):7.0f} MJ/tHM   n={len(s)}")
    r = df["flow_ratio"].dropna()
    if len(r):
        print(f"  total CW flow / stave-circuit flow: median {r.median():.2f}")
    print("  physical expectation for a BF: 200 - 400 MJ/tHM")

    banner("2. ENERGY TERMS (median, MJ/tHM)")
    terms = ["q_slag", "q_hm", "q_fe_reduction", "q_si", "q_mn", "q_calcination",
             "q_burden_moisture", "q_feo_slag", "q_loss_total",
             "carbon_charged_kg_per_thm", "carbon_to_hm_kg_per_thm",
             "q_topgas_sensible", "q_topgas_chemical", "q_output_total",
             "q_carbon", "q_blast", "q_input", "topgas_nm3_per_thm"]
    print(df[terms].median().to_string(float_format=lambda v: f"{v:10.0f}"))

    banner("3. THE GATE - does the balance CLOSE, and hold across quarters?")
    print("  closure = (demand + top gas) / (carbon + blast). Target 1.00.")
    df["quarter"] = df.index.to_period("Q").astype(str)
    for basis in ("stave", "total"):
        col = f"closure_{basis}"
        sub = df[[col, "quarter"]].dropna()
        if sub.empty:
            continue
        g = sub.groupby("quarter")[col].agg(["count", "median", "std"])
        g["cv"] = g["std"] / g["median"]
        print(f"\n  basis: {basis}")
        print(g.to_string(float_format=lambda v: f"{v:9.3f}"))
        med = g["median"]
        spread = (med.max() - med.min()) / med.median() if len(med) > 1 else np.nan
        print(f"  across-quarter spread: {spread:.1%}   overall median "
              f"{sub[col].median():.3f}")

    banner("4. SHELL LOSS BACK-CALCULATED FROM CLOSURE")
    print("  Force closure to 1.0 and see what shell loss that implies. If the")
    print("  answer is physical (200-400 MJ/tHM) and stable, the residual really")
    print("  is shell loss. If it is far larger, a term is still missing and")
    print("  calling it shell loss would just be hiding the error.")
    core = (
        df["q_demand_total"] - df["q_loss_total"]
    )  # everything except the loss term
    implied = df["q_input"] - df["q_topgas"] - core
    df["implied_shell_loss"] = implied
    sub = df[["implied_shell_loss", "quarter"]].dropna()
    if not sub.empty:
        g = sub.groupby("quarter")["implied_shell_loss"].agg(["count", "median", "std"])
        print(g.to_string(float_format=lambda v: f"{v:10.0f}"))
        print(f"  overall median {sub['implied_shell_loss'].median():,.0f} MJ/tHM")
        print(f"  measured stave rows 6-10: {df['q_loss_stave'].median():,.0f} MJ/tHM")
        print(f"  flow-scaled all circuits: {df['q_loss_total'].median():,.0f} MJ/tHM")

    banner("5. IMPLIED BLAST-TEMPERATURE COEFFICIENT")
    print("  Holding demand fixed, extra blast sensible heat displaces carbon:")
    print("      d(coke)/d(T_blast) = cbv * cp / (h_C * C_frac)")
    cbv = df["cbv_per_thm"].median()
    per_100c = cbv * BLAST_CP_KJ_PER_NM3_K * 100.0 / 1000.0 / (
        H_C_TO_CO_MJ_PER_KG * C_FRAC_COKE
    )
    print(f"  cbv = {cbv:,.0f} Nm3/tHM  ->  {-per_100c:+.1f} kg coke per 100 C")
    print("  config / literature: -8 to -12 kg per 100 C")
    print("  A large gap here means the top-gas loss term is missing, which is")
    print("  expected at this stage - it is Phase 1 work, not a failure.")


if __name__ == "__main__":
    main()
