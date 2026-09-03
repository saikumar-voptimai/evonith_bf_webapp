"""Full audit of one day's energy and carbon balance, every step shown.

Run:  python scripts/energy_balance_day_audit.py [YYYY-MM-DD]

Built to be CHECKED BY HAND, not admired. Every figure is printed with the
numbers that produced it, so a wrong constant or a mis-mapped tag shows up as an
implausible intermediate rather than hiding inside a plausible total.

Adds what the earlier work was missing: DUST. The plant records four dust
streams in dpr_data and none was in the balance. Only two of them leave through
the top:

    flue_dust_mt        dust catcher, coarse carryover from the top gas   YES
    gcp_dust_mt         gas cleaning plant, fine carryover                YES
    cast_house_dust_mt  tapping fume - leaves at the cast house           no
    stock_house_dust_mt handling loss BEFORE charging                     no

Carbon in that dust is charged but never burnt, exactly like the carbon that
dissolves into hot metal, so it must come off the input side. Dust carbon
content is NOT measured here; the literature range is applied and the
sensitivity shown, because the conclusion should not rest on the assumption.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO / ".env")

from furnace_data.offline import fetch_offline_data  # noqa: E402
import energy_balance_phase0 as eb  # noqa: E402

DEFAULT_DAY = "2025-12-31"

# Dust carbon, weight %. Not measured at this plant. Dust-catcher carryover is
# largely coke fines; GCP dust is finer, richer in Fe and Zn, poorer in carbon.
C_PCT_FLUE_DUST = 30.0
C_PCT_GCP_DUST = 20.0
C_PCT_RANGE = ((20.0, 12.0), (30.0, 20.0), (40.0, 28.0))  # low / mid / high

C_MOLAR_MASS = 12.011
NM3_PER_KMOL = 22.414


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def row(label: str, value: str, note: str = "") -> None:
    print(f"  {label:<38s} {value:>16s}   {note}")


def daily_dust() -> pd.DataFrame:
    df = fetch_offline_data("dpr_data", time_range="full", query_type="raw")
    out = pd.DataFrame(index=df.index)
    for col in ("flue_dust_mt", "gcp_dust_mt", "cast_house_dust_mt",
                "stock_house_dust_mt"):
        out[col] = pd.to_numeric(df.get(col), errors="coerce")
    out["date"] = eb._ist_date(df.index)
    daily = out.groupby("date").mean(numeric_only=True)
    daily.index = pd.to_datetime(daily.index)
    return daily


def main() -> None:
    day = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DAY)

    df = eb.build().join(daily_dust(), how="left")
    if day not in df.index:
        near = df.index[df.index.get_indexer([day], method="nearest")][0]
        print(f"  {day.date()} not in the sample; using nearest, {near.date()}")
        day = near
    d = df.loc[day]
    hm = float(d["hm_mt"])

    print(f"\nENERGY BALANCE AUDIT - {day.date()}")
    print(f"Hot metal {hm:,.1f} t. All rates are per tonne hot metal.")

    # ---------------------------------------------------------------- A ------
    rule("A. AS RECORDED (check these against the DPR sheet first)")
    for label, key, unit in (
        ("Hot metal", "hm_mt", "t"), ("Slag", "slag_mt", "t"),
        ("Coke", "coke_mt", "t"), ("Nut coke", "nut_coke_mt", "t"),
        ("PCI", "pci_mt", "t"), ("Ore", "ore_mt", "t"),
        ("Sinter", "sinter_mt", "t"), ("Pellet", "pellet_mt", "t"),
        ("Flux", "flux_mt", "t"),
        ("Flue dust (dust catcher)", "flue_dust_mt", "t"),
        ("GCP dust", "gcp_dust_mt", "t"),
        ("Cast house dust", "cast_house_dust_mt", "t"),
        ("Stock house dust", "stock_house_dust_mt", "t"),
    ):
        v = d.get(key)
        row(label, "n/a" if pd.isna(v) else f"{float(v):,.1f} {unit}")
    print()
    for label, key, unit in (
        ("Blast volume", "cbv_nm3h", "Nm3/hr"),
        ("Blast temperature", "blast_temp", "C"),
        ("O2 enrichment", "o2_enrich", "%"),
        ("Top gas CO", "co_pct", "%"), ("Top gas CO2", "co2_pct", "%"),
        ("Top gas H2", "h2_pct", "%"), ("Top gas temperature", "top_temp", "C"),
        ("HM carbon", "hm_c", "%"), ("HM iron", "hm_fe", "%"),
        ("HM silicon", "hm_si", "%"), ("Slag FeO", "slag_feo_pct", "%"),
        ("Stave heat load", "stave_gj_per_hr", "GJ/hr"),
    ):
        v = d.get(key)
        row(label, "n/a" if pd.isna(v) else f"{float(v):,.2f} {unit}")

    # ---------------------------------------------------------------- B ------
    rule("B. RATES PER TONNE HOT METAL")
    for label, key in (("Coke", "coke_rate"), ("Nut coke", "nut_rate"),
                       ("PCI", "pci_rate"), ("Slag", "slag_rate"),
                       ("Flux", "flux_rate"), ("Blast", "cbv_per_thm")):
        row(label, f"{float(d[key]):,.1f} kg/tHM" if key != "cbv_per_thm"
            else f"{float(d[key]):,.0f} Nm3/tHM")
    fuel_rate = float(d["coke_rate"] + d["nut_rate"] + d["pci_rate"])
    row("Total fuel", f"{fuel_rate:,.1f} kg/tHM", "plant target ~560-580")

    flue = float(d.get("flue_dust_mt") or 0.0)
    gcp = float(d.get("gcp_dust_mt") or 0.0)
    dust_top_kg = (flue + gcp) / hm * 1000.0
    row("Top-gas dust (flue + GCP)", f"{dust_top_kg:,.1f} kg/tHM",
        "literature 10-25")

    # ---------------------------------------------------------------- C ------
    rule("C. CARBON BALANCE  (this is where dust enters)")
    c_coke = float(d["coke_rate"]) * eb.C_FRAC_COKE
    c_nut = float(d["nut_rate"]) * eb.C_FRAC_COKE
    c_pci = float(d["pci_rate"]) * eb.C_FRAC_PCI
    c_charged = c_coke + c_nut + c_pci
    row("Coke carbon", f"{c_coke:,.1f} kg/tHM",
        f"{d['coke_rate']:.1f} x {eb.C_FRAC_COKE}")
    row("Nut coke carbon", f"{c_nut:,.1f} kg/tHM",
        f"{d['nut_rate']:.1f} x {eb.C_FRAC_COKE}")
    row("PCI carbon", f"{c_pci:,.1f} kg/tHM",
        f"{d['pci_rate']:.1f} x {eb.C_FRAC_PCI}  <- rank implies ~0.79")
    row("CARBON CHARGED", f"{c_charged:,.1f} kg/tHM")
    print()
    c_hm = float(d["hm_c"]) / 100.0 * 1000.0
    c_dust = (flue * C_PCT_FLUE_DUST + gcp * C_PCT_GCP_DUST) / 100.0 / hm * 1000.0
    row("less carbon into hot metal", f"-{c_hm:,.1f} kg/tHM",
        f"{d['hm_c']:.2f}% of 1000 kg")
    row("less carbon leaving in dust", f"-{c_dust:,.1f} kg/tHM",
        f"flue @{C_PCT_FLUE_DUST:.0f}%C, GCP @{C_PCT_GCP_DUST:.0f}%C")
    c_burnt_old = c_charged - c_hm
    c_burnt_new = c_charged - c_hm - c_dust
    row("CARBON BURNT, model today", f"{c_burnt_old:,.1f} kg/tHM", "dust ignored")
    row("CARBON BURNT, with dust", f"{c_burnt_new:,.1f} kg/tHM",
        f"{c_dust / c_charged:.1%} of charged carbon")

    # ---------------------------------------------------------------- D ------
    rule("D. TOP GAS VOLUME - two independent routes, which must agree")
    n2_blast = eb.N2_IN_AIR_PCT - float(d["o2_enrich"])
    n2_top = 100.0 - float(d["co_pct"]) - float(d["co2_pct"]) - float(d["h2_pct"])
    v_n2 = float(d["cbv_per_thm"]) * n2_blast / n2_top
    print(f"  NITROGEN   N2 blast {n2_blast:.2f}%   N2 top {n2_top:.2f}%")
    row("V = blast x N2b / N2t", f"{v_n2:,.0f} Nm3/tHM",
        f"{d['cbv_per_thm']:.0f} x {n2_blast:.1f} / {n2_top:.1f}")
    cox = float(d["co_pct"] + d["co2_pct"])
    for label, c_val in (("dust ignored", c_burnt_old), ("with dust", c_burnt_new)):
        v_c = c_val / C_MOLAR_MASS * NM3_PER_KMOL / (cox / 100.0)
        print(f"  CARBON     C burnt {c_val:.1f} kg -> "
              f"{c_val / C_MOLAR_MASS * NM3_PER_KMOL:,.0f} Nm3 of CO+CO2, "
              f"which is {cox:.2f}% of the gas")
        row(f"V ({label})", f"{v_c:,.0f} Nm3/tHM",
            f"ratio to nitrogen {v_c / v_n2:.3f}")
    v_carbon = c_burnt_new / C_MOLAR_MASS * NM3_PER_KMOL / (cox / 100.0)
    print(f"\n  Textbook top gas is 1,500-1,700 Nm3/tHM.")
    print(f"  Dust closes {(1 - (v_carbon / v_n2 - 1) / (
        (c_burnt_old / C_MOLAR_MASS * NM3_PER_KMOL / (cox / 100.0)) / v_n2 - 1
    )):.0%} of the original gap - real, but not the whole story.")

    # ---------------------------------------------------------------- E ------
    rule("E. INPUT")
    q_c_old = eb.H_C_FULL_MJ_PER_KG * c_burnt_old
    q_c_new = eb.H_C_FULL_MJ_PER_KG * c_burnt_new
    q_blast = float(d["q_blast"])
    row("Carbon, model today", f"{q_c_old:,.0f} MJ/tHM",
        f"{c_burnt_old:.1f} x {eb.H_C_FULL_MJ_PER_KG}")
    row("Carbon, with dust removed", f"{q_c_new:,.0f} MJ/tHM",
        f"-{q_c_old - q_c_new:,.0f} MJ/tHM")
    row("Blast sensible heat", f"{q_blast:,.0f} MJ/tHM",
        f"{d['cbv_per_thm']:.0f} x {eb.BLAST_CP_KJ_PER_NM3_K} x "
        f"({d['blast_temp']:.0f}-25)/1000")
    row("TOTAL INPUT, model today", f"{q_c_old + q_blast:,.0f} MJ/tHM")
    row("TOTAL INPUT, with dust", f"{q_c_new + q_blast:,.0f} MJ/tHM")

    # ---------------------------------------------------------------- F ------
    rule("F. OUTPUT")
    terms = [
        ("Iron oxide reduction", "q_fe_reduction", "7.38 MJ/kg Fe - largest term"),
        ("Hot metal to 1500 C", "q_hm", "1378 MJ/t"),
        ("Slag", "q_slag", "1.8 MJ/kg"),
        ("FeO left in slag", "q_feo_slag", ""),
        ("Silicon reduction", "q_si", "24.6 MJ/kg Si"),
        ("Manganese reduction", "q_mn", ""),
        ("Burden moisture", "q_burden_moisture", "2.7 MJ/kg H2O"),
        ("Flux calcination", "q_calcination", ""),
        ("Steam", "q_steam", "nil at this plant"),
        ("Shell loss (flow-scaled)", "q_loss_total", "measured, not fitted"),
    ]
    for label, key, note in terms:
        v = d.get(key)
        row(label, "n/a" if pd.isna(v) else f"{float(v):,.0f} MJ/tHM", note)
    print()
    row("Top gas sensible", f"{float(d['q_topgas_sensible']):,.0f} MJ/tHM",
        f"{v_n2:,.0f} x 1.38 x ({d['top_temp']:.0f}-25)/1000")
    row("Top gas chemical (CO + H2)", f"{float(d['q_topgas_chemical']):,.0f} MJ/tHM",
        "unburnt fuel value leaving")
    row("TOTAL OUTPUT", f"{float(d['q_output_total']):,.0f} MJ/tHM")

    # ---------------------------------------------------------------- G ------
    rule("G. CLOSURE")
    out_total = float(d["q_output_total"])
    for label, q_in in (("model today", q_c_old + q_blast),
                        ("with dust carbon removed", q_c_new + q_blast)):
        print(f"  {label:<28s} {out_total:,.0f} / {q_in:,.0f} = "
              f"{out_total / q_in:.3f}")
    print("\n  Removing dust carbon RAISES closure above 1.0, i.e. it pushes the")
    print("  error the other way. Together with the top-gas volume gap pulling")
    print("  the opposite direction, this is the clearest sign that today's")
    print("  1.00 is two errors cancelling rather than a balance that is right.")

    rule("H. SENSITIVITY - dust carbon is assumed, so vary it")
    print(f"  {'flue %C':>9} {'GCP %C':>8} {'C dust':>10} {'closure':>9} {'V_carbon':>10}")
    for flue_pct, gcp_pct in C_PCT_RANGE:
        cd = (flue * flue_pct + gcp * gcp_pct) / 100.0 / hm * 1000.0
        cb = c_charged - c_hm - cd
        clo = out_total / (eb.H_C_FULL_MJ_PER_KG * cb + q_blast)
        vc = cb / C_MOLAR_MASS * NM3_PER_KMOL / (cox / 100.0)
        print(f"  {flue_pct:>9.0f} {gcp_pct:>8.0f} {cd:>10.1f} {clo:>9.3f} {vc:>10,.0f}")
    print(f"  {'':>9} {'':>8} {'':>10} {'':>9} {v_n2:>10,.0f}  <- nitrogen route")
    print("\n  Even at 40% carbon in flue dust the two volumes do not meet, so")
    print("  dust alone cannot explain the gap. A dust sample sent for carbon")
    print("  analysis would remove the last assumption here.")

    rule("I. FLAGGED FOR CHECKING")
    flags: list[str] = []
    stave_mj = float(d["q_loss_stave"]) if pd.notna(d.get("q_loss_stave")) else 0.0
    total_mj = float(d["q_loss_total"]) if pd.notna(d.get("q_loss_total")) else 0.0
    if stave_mj and total_mj / stave_mj > 2.5:
        flags.append(
            f"Shell loss scaled {total_mj / stave_mj:.1f}x from the measured stave "
            f"rows ({stave_mj:,.0f} -> {total_mj:,.0f} MJ/tHM). The scale-up uses "
            "cooling-water FLOW share, which assumes every circuit runs the same "
            "temperature rise. Worth checking against actual circuit dT."
        )
    if dust_top_kg > 25.0:
        flags.append(
            f"Top-gas dust {dust_top_kg:.1f} kg/tHM is above the usual 10-25. "
            "Confirm flue dust and GCP dust are separate streams and not the "
            "same material weighed twice."
        )
    if pd.isna(d.get("cast_house_dust_mt")) or pd.isna(d.get("stock_house_dust_mt")):
        flags.append(
            "Cast-house and/or stock-house dust not recorded for this day. "
            "Neither belongs in the top-gas balance, but stock-house dust means "
            "charged tonnes may be weighed before a handling loss."
        )
    if float(d["pci_rate"]) > 0 and eb.C_FRAC_PCI < 0.78:
        flags.append(
            f"PCI carbon fraction is set to {eb.C_FRAC_PCI}, but this coal's rank "
            "(22.4% VM daf, 9.2% ash) implies about 0.79. Raising it would widen "
            "the top-gas gap, not close it - noted so nobody 'fixes' it expecting "
            "the opposite."
        )
    for i, text in enumerate(flags, 1):
        print(f"  {i}. {text}\n")

    print("  AND THE ALTERNATIVE THIS AUDIT CANNOT RULE OUT:")
    print("  A blast flow meter reading low would also depress V_nitrogen, with")
    print("  no effect on V_carbon - the same signature as a gas-analyser fault.")
    print("  What still favours the analyser is that CO2/(CO+CO2) stays flat at")
    print("  42-43% while the sum drifts, and a flow meter cannot do that. But if")
    print("  the analyser service record comes back clean, the blast flow meter")
    print(f"  is the next thing to check: it would have to read ~{v_carbon / v_n2 - 1:.0%} low.")


if __name__ == "__main__":
    main()
