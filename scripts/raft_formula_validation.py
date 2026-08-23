"""Which RAFT formula should Layer 2 use? Measured against the plant's own tag.

Run:  python scripts/raft_formula_validation.py

WHY THIS EXISTS.

Layer 2 recommends blast temperature, oxygen enrichment, blast volume and PCI.
Every one of those is something the raceway heat balance responds to, so RAFT
does not need a model of its own - it can be computed straight from the
recommended settings. The question was only which formula.

Three candidates, scored against body_raft over 120 days of hourly data:

    fitted   the correlation previously shipped. No moisture term at all, and
             PCI used in kg/tHM directly rather than as a blast concentration.
    lit_A    RAFT = 1615 + 0.76 HBT - 5.75 humidity + 53.3 O2
             No PCI term - and PCI is a strong raceway coolant.
    lit_B    RAFT = 1470 + 0.85 HBT - 5.5 (humidity + steam) + 50 O2
                  - 2 (coal as g per Nm3 of blast)

RESULT. lit_B wins clearly, and only its intercept needs plant calibration.
With the intercept fitted on the FIRST half and scored on the second:

    model       test MAE   test RMSE   test R2   corr
    fitted        27.7 C      40.2 C     0.162   0.693
    lit_B         17.4 C      26.9 C     0.625   0.851

lit_A is far worse than either, which is worth knowing: leaving PCI out of a
RAFT formula on a furnace injecting 150 kg/tHM is not a small simplification.

WHY lit_B IS THE RIGHT SHAPE. Humidity, steam and injected coal are all heat
sinks carried into the raceway in the blast, so all three belong on a per-Nm3
basis and share the same kind of coefficient. Expressing PCI as kg/tHM instead
makes the answer independent of wind rate, which is wrong - the same coal rate
cools harder on a smaller blast.

Shipped in energy_balance.yml under process_recommendation.raft.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
REPO = Path(r"e:/Personal/MarketResearch/EvonithSteel/BlastFurnaceProject/PythonBlastFurnace/evonith_webapp")
sys.path.insert(0, str(REPO / "src"))
from dotenv import load_dotenv
load_dotenv(REPO / ".env")
import numpy as np, pandas as pd
from furnace_data.influx.online import fetch_online_df

end = datetime.now(timezone.utc); start = end - timedelta(days=120)
df = fetch_online_df(selected_measurements=["process_params"], time_range="last 1 week",
    request_type="windowed-average", window_by="1 hour",
    start_time_override=start, end_time_override=end, column_naming="field")
n = lambda c: pd.to_numeric(df.get(c), errors="coerce")
d = pd.DataFrame({
    "raft": n("body_raft"), "hbt": n("hot_blast_temp"), "o2": n("oxygen_enrichment_pct"),
    "wind": n("hot_blast_vol_nm3h"), "steam": n("steam_injection"),
    "pci": n("coal_rate_actual_value"), "prod": n("production_per_hour"),
}).replace([np.inf,-np.inf], np.nan).dropna(subset=["raft","hbt","o2","wind","pci","prod"])
d = d[(d.raft.between(1800,2600)) & (d.hbt.between(800,1350)) & (d["prod"]>50) & (d["wind"]>50000)]
d["steam"] = d["steam"].fillna(0.0)
d["wind_per_thm"] = d["wind"] / d["prod"]
# steam kg/hr -> g per Nm3 of blast
d["steam_g_nm3"] = d["steam"] * 1000.0 / d["wind"]
d["coal_g_nm3"] = d["pci"] * 1000.0 / d["wind_per_thm"]
MOIST = 15.0
d["fitted"]  = 1555.4 + 0.8100*d.hbt + 26.511*d.o2 - 2.2229*d.pci
d["lit_A"]   = 1615 + 0.76*d.hbt - 5.75*MOIST + 53.3*d.o2
d["lit_B"]   = 1470 + 0.85*d.hbt - 5.5*(MOIST + d.steam_g_nm3) + 50*d.o2 - 2*d.coal_g_nm3
print(f"n = {len(d)}   {d.index.min()} -> {d.index.max()}")
print(f"plant body_raft: mean {d.raft.mean():.0f}  p5 {d.raft.quantile(.05):.0f}  p95 {d.raft.quantile(.95):.0f}")
print(f"steam g/Nm3 median {d.steam_g_nm3.median():.2f}   coal g/Nm3 median {d.coal_g_nm3.median():.1f}")
print(f"\n{'model':10s} {'bias':>8s} {'MAE':>8s} {'RMSE':>8s} {'corr':>7s} {'R2':>7s}")
for name in ("fitted","lit_A","lit_B"):
    e = d[name] - d.raft
    ss = 1 - (e**2).sum()/((d.raft-d.raft.mean())**2).sum()
    print(f"{name:10s} {e.mean():+8.1f} {e.abs().mean():8.1f} {np.sqrt((e**2).mean()):8.1f} "
          f"{d[name].corr(d.raft):+7.3f} {ss:7.3f}")
print("\nbias-corrected (intercept shifted to zero mean error):")
for name in ("fitted","lit_A","lit_B"):
    e = d[name] - d.raft - (d[name]-d.raft).mean()
    ss = 1 - (e**2).sum()/((d.raft-d.raft.mean())**2).sum()
    print(f"{name:10s} MAE {e.abs().mean():7.1f}  RMSE {np.sqrt((e**2).mean()):7.1f}  R2 {ss:7.3f}")

print("\n=== FORWARD TEST: fit intercept on first half, score on second ===")
half = len(d)//2
tr, te = d.iloc[:half], d.iloc[half:]
print(f"train {tr.index.min().date()} -> {tr.index.max().date()}  n={len(tr)}")
print(f"test  {te.index.min().date()} -> {te.index.max().date()}  n={len(te)}")
for name in ("fitted","lit_B"):
    shift = (tr[name]-tr["raft"]).mean()          # calibrated on TRAIN only
    e = (te[name]-shift) - te["raft"]
    ss = 1 - (e**2).sum()/((te["raft"]-te["raft"].mean())**2).sum()
    print(f"  {name:8s} intercept shift {-shift:+7.1f}  ->  test MAE {e.abs().mean():6.1f}"
          f"  RMSE {np.sqrt((e**2).mean()):6.1f}  R2 {ss:+6.3f}  corr {te[name].corr(te['raft']):+.3f}")
print("\n=== DIRECTIONAL SENSITIVITY (what the operator actually needs) ===")
w = d["wind_per_thm"].median()
print(f"  at wind {w:,.0f} Nm3/tHM:")
print(f"  +100 C blast temperature   -> RAFT {0.85*100:+.0f} C")
print(f"  +1 % oxygen enrichment     -> RAFT {50*1:+.0f} C")
print(f"  +1 g/Nm3 blast moisture    -> RAFT {-5.5*1:+.0f} C")
print(f"  +10 kg/tHM PCI             -> RAFT {-2*10*1000/w:+.0f} C")
print(f"  +1000 kg/hr steam @ {d['wind'].median():,.0f} Nm3/hr -> RAFT {-5.5*1000*1000/d['wind'].median():+.1f} C")
