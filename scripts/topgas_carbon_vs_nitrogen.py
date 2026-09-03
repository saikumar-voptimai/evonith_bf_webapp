"""Where is the fuel-scaled output term? Two top-gas volumes that must agree.

Run:  python scripts/topgas_carbon_vs_nitrogen.py

WHY THIS EXISTS.

Ruling out fuel hydrogen (scripts/pci_hydrogen_from_closure.py) left a sharper
question behind. The back-calculated residual scales with TOTAL FUEL: coke, nut
coke and PCI each carry a positive coefficient of +17 to +29 MJ per kg, against
a carbon credit of 28.5 MJ/kg coke. Only about a third of marginal fuel energy
is reaching a modelled output term. Something fuel-proportional is missing on
the OUTPUT side.

Top gas is the obvious suspect, because it is the only output term that should
scale with fuel, and because its volume is not measured - it is inferred.

THE TEST.

Top-gas volume can be derived two entirely independent ways, and they must agree
because they describe the same gas:

    NITROGEN   N2 is inert, so all of it leaves at the top.
               V = V_blast x N2%_blast / N2%_top          <- what the model uses

    CARBON     every kg of carbon burnt leaves as CO or CO2.
               V = (C_burnt / 12.011) x 22.414 / ((CO% + CO2%) / 100)

Neither uses a fitted constant. If they disagree, the gap is real and its size
converts directly into MJ/tHM, because the same volume multiplies the CO
calorific value and the sensible heat.

HOW TO READ THE ANSWER.

  V_carbon > V_nitrogen   the model is under-counting top gas, so the missing
                          output term is top gas itself.
  V_carbon < V_nitrogen   carbon is leaving somewhere other than the top gas -
                          flue dust and sludge are the candidates - so the
                          carbon INPUT is over-credited.

The two have opposite fixes, which is exactly why this needs measuring rather
than assuming.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

from energy_balance_phase0 import build  # noqa: E402

C_MOLAR_MASS = 12.011
MOLAR_VOLUME_NM3 = 22.414
CV_CO_MJ_PER_NM3 = 12.63
TOPGAS_CP_KJ_PER_NM3_K = 1.38
T_REF_C = 25.0


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def main() -> None:
    df = build()
    df["quarter"] = df.index.to_period("Q").astype(str)

    carbon_gas_nm3 = df["carbon_kg_per_thm"] / C_MOLAR_MASS * MOLAR_VOLUME_NM3
    cox_pct = (df["co_pct"] + df["co2_pct"]) / 100.0
    df["v_carbon"] = carbon_gas_nm3 / cox_pct
    df["v_nitrogen"] = df["topgas_nm3_per_thm"]
    df["v_ratio"] = df["v_carbon"] / df["v_nitrogen"]
    df["v_gap"] = df["v_carbon"] - df["v_nitrogen"]

    core = df["q_demand_total"] - df["q_loss_total"]
    df["implied"] = df["q_input"] - df["q_topgas"] - core

    d = df[
        ["v_carbon", "v_nitrogen", "v_ratio", "v_gap", "implied", "quarter",
         "co_pct", "co2_pct", "h2_pct", "carbon_kg_per_thm", "cbv_per_thm",
         "top_temp", "q_topgas", "q_input", "q_loss_total", "coke_rate",
         "pci_rate", "hm_mt"]
    ].replace([np.inf, -np.inf], np.nan).dropna()

    banner("0. SAMPLE")
    print(f"  days {len(d)}   {d.index.min().date()} -> {d.index.max().date()}")

    banner("1. TWO INDEPENDENT TOP-GAS VOLUMES")
    for col, label in (("v_nitrogen", "from nitrogen (model)"),
                       ("v_carbon", "from carbon")):
        s = d[col]
        print(f"  {label:24s} median {s.median():7,.0f}  p5 {s.quantile(.05):7,.0f}"
              f"  p95 {s.quantile(.95):7,.0f} Nm3/tHM")
    print(f"\n  ratio carbon/nitrogen: median {d['v_ratio'].median():.3f}  "
          f"p5 {d['v_ratio'].quantile(.05):.3f}  p95 {d['v_ratio'].quantile(.95):.3f}")
    print(f"  gap: median {d['v_gap'].median():+,.0f} Nm3/tHM")
    print("  Textbook top gas is 1,500-1,700 Nm3/tHM; both should land there.")

    banner("2. WHAT THE GAP IS WORTH IN ENERGY")
    print("  The same volume multiplies the CO calorific value and the sensible")
    print("  heat, so a volume error converts straight into MJ/tHM.")
    per_nm3 = (
        d["co_pct"] / 100.0 * CV_CO_MJ_PER_NM3
        + TOPGAS_CP_KJ_PER_NM3_K * (d["top_temp"] - T_REF_C) / 1000.0
    )
    d = d.assign(gap_mj=d["v_gap"] * per_nm3)
    print(f"  energy per Nm3 of top gas: median {per_nm3.median():.2f} MJ")
    print(f"  gap in energy terms:       median {d['gap_mj'].median():+,.0f} MJ/tHM")
    print(f"  unexplained residual:      median {d['implied'].median():+,.0f} MJ/tHM")
    print(f"  measured shell loss:       median {d['q_loss_total'].median():+,.0f} MJ/tHM")
    print("\n  If the gap in energy terms lands near (residual - measured shell),")
    print("  the top-gas volume IS the missing term.")
    target = (d["implied"] - d["q_loss_total"]).median()
    print(f"  residual - measured shell = {target:+,.0f} MJ/tHM")

    banner("3. DOES THE GAP EXPLAIN THE RESIDUAL, DAY BY DAY?")
    print("  A term that is genuinely missing must track the residual, not just")
    print("  match it on average.")
    print(f"  corr(gap_mj, implied)          = {d['gap_mj'].corr(d['implied']):+.3f}")
    print(f"  corr(v_ratio, implied)         = {d['v_ratio'].corr(d['implied']):+.3f}")
    adjusted = d["implied"] - d["gap_mj"]
    print(f"\n  residual std before adjustment {d['implied'].std():7,.0f} MJ/tHM")
    print(f"  residual std after  adjustment {adjusted.std():7,.0f} MJ/tHM")
    print(f"  residual median after          {adjusted.median():+7,.0f} MJ/tHM"
          f"   (measured shell {d['q_loss_total'].median():,.0f})")

    banner("4. THE ACROSS-QUARTER DRIFT")
    print("  The drift is the real defect. Does the gap account for it?")
    g = pd.DataFrame({
        "implied": d["implied"].groupby(d["quarter"]).median(),
        "gap_mj": d["gap_mj"].groupby(d["quarter"]).median(),
        "adjusted": adjusted.groupby(d["quarter"]).median(),
        "v_ratio": d["v_ratio"].groupby(d["quarter"]).median(),
        "n": d["implied"].groupby(d["quarter"]).size(),
    })
    print(g.to_string(float_format=lambda v: f"{v:10.2f}"))
    print(f"\n  drift in implied  {g['implied'].max() - g['implied'].min():,.0f} MJ/tHM")
    print(f"  drift in adjusted {g['adjusted'].max() - g['adjusted'].min():,.0f} MJ/tHM")

    banner("5. IF IT IS DUST INSTEAD: how much carbon would have to leave?")
    print("  Reading the gap the other way - as carbon never reaching the top gas.")
    c_gap = d["v_gap"] * cox_pct.loc[d.index] / MOLAR_VOLUME_NM3 * C_MOLAR_MASS
    print(f"  implied carbon discrepancy: median {c_gap.median():+.1f} kg C/tHM")
    print(f"  as a share of carbon burnt: {(c_gap / d['carbon_kg_per_thm']).median():+.1%}")
    print("  BF flue dust is typically 10-25 kg/tHM at 25-40% C, i.e. 3-10 kg C/tHM.")
    print("  A figure far outside that cannot be dust.")

    banner("6. WHICH ASSUMPTION IS AT FAULT?")
    print("  Adjusting fully to the carbon volume OVERSHOOTS - the residual goes")
    print("  to -174 against a measured shell loss of 550 - so the truth sits")
    print("  between the two. Solve each side for the value that would reconcile")
    print("  them and see which lands on something physically meaningful.")

    n2_top = 100.0 - d["co_pct"] - d["co2_pct"] - d["h2_pct"]
    # (a) what N2% in the blast would make the nitrogen balance agree?
    needed_n2_blast = d["v_carbon"] * n2_top / d["cbv_per_thm"]
    print(f"\n  (a) N2% in blast needed to reconcile: median "
          f"{needed_n2_blast.median():.1f}%")
    print("      model uses 79.2 - enrichment, i.e. ~75.4% on a typical day.")
    print("      A value near 79.2 would mean the enrichment oxygen is NOT")
    print("      inside the measured cold-blast volume, so subtracting it is")
    print("      wrong. Above ~80% no blast composition explains it.")

    # (b) what carbon fraction would make the carbon balance agree?
    c_needed = (
        d["v_nitrogen"] * (d["co_pct"] + d["co2_pct"]) / 100.0
        / MOLAR_VOLUME_NM3 * C_MOLAR_MASS
    )
    scale = c_needed / d["carbon_kg_per_thm"]
    print(f"\n  (b) carbon burnt would have to be {scale.median():.2f}x the "
          "charged figure")
    print(f"      i.e. {c_needed.median():.0f} against {d['carbon_kg_per_thm'].median():.0f}"
          " kg C/tHM.")
    print("      Coke is 87% C (FC 87.4) and PCI 75%, both already conservative;")
    print("      PCI at this rank is nearer 79%, which would widen the gap, not")
    print("      close it. So the carbon side cannot absorb this.")

    print("\n  (c) both effects together, per quarter:")
    q = pd.DataFrame({
        "needed_N2_blast": needed_n2_blast.groupby(d["quarter"]).median(),
        "v_ratio": d["v_ratio"].groupby(d["quarter"]).median(),
        "o2_enrich_proxy": (79.2 - needed_n2_blast).groupby(d["quarter"]).median(),
    })
    print(q.to_string(float_format=lambda v: f"{v:14.2f}"))

    banner("7. THE COMMON CAUSE: the top gas analysis itself")
    print("  Both volumes are computed FROM the gas analysis, and an under-read")
    print("  of CO+CO2 biases them in OPPOSITE directions:")
    print("      N2_top = 100 - CO - CO2 - H2  ->  overstated  ->  V_nitrogen LOW")
    print("      V_carbon divides by (CO+CO2)  ->               ->  V_carbon HIGH")
    print("  That is exactly the sign observed, from a single root cause.")
    print("  So solve for the CO+CO2 that makes the two agree.")

    k_nm3 = d["carbon_kg_per_thm"] / C_MOLAR_MASS * MOLAR_VOLUME_NM3
    # Hold the model's own blast-N2 convention, so this isolates the gas
    # analysis rather than silently absorbing the enrichment question from (a).
    n2_blast_model = d["v_nitrogen"] * n2_top / d["cbv_per_thm"]
    a_nm3 = d["cbv_per_thm"] * n2_blast_model
    # V_carbon = 100K/x and V_nitrogen = A/(100 - x - h2), with x in PERCENT.
    # Setting them equal:  x = 100K(100 - h2) / (A + 100K).
    cox_reconciling = (
        100.0 * k_nm3 * (100.0 - d["h2_pct"]) / (a_nm3 + 100.0 * k_nm3)
    )
    cox_measured = d["co_pct"] + d["co2_pct"]

    print(f"\n  measured   CO+CO2: median {cox_measured.median():5.2f} %")
    print(f"  reconciling CO+CO2: median {cox_reconciling.median():5.2f} %")
    print(f"  shortfall:                 {(cox_reconciling - cox_measured).median():+5.2f} "
          "percentage points")
    print(f"  as a relative error: {((cox_reconciling / cox_measured) - 1).median():+.1%}")

    qq = pd.DataFrame({
        "measured": cox_measured.groupby(d["quarter"]).median(),
        "reconciling": cox_reconciling.groupby(d["quarter"]).median(),
        "shortfall": (cox_reconciling - cox_measured).groupby(d["quarter"]).median(),
        "eta_co_proxy": (d["co2_pct"] / (d["co_pct"] + d["co2_pct"]) * 100.0)
        .groupby(d["quarter"]).median(),
    })
    print("\n  by quarter:")
    print(qq.to_string(float_format=lambda v: f"{v:12.2f}"))
    print("\n  A shortfall that shrinks toward zero over time is an analyser")
    print("  drifting back into calibration, not a process change. Check the")
    print("  top gas analyser service record against these dates before")
    print("  accepting any of the earlier data at face value.")


if __name__ == "__main__":
    main()
